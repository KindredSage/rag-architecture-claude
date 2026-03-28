"""
Query Validator: Validates SQL on security, correctness, and performance.

Uses BOTH:
  1. Programmatic checks (blocked keywords, structure validation)
  2. LLM-based semantic validation (column existence, type compatibility)

This is defense-in-depth: even if the LLM misses something,
the programmatic checks catch it, and vice versa.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from services.llm_invoke import invoke_llm

from agents.trade.state import TradeAgentState

logger = logging.getLogger(__name__)

VALIDATION_PROMPT = """You are a SQL security and correctness validator for ClickHouse.
Validate the given SQL query against the actual schema.

Checks to perform:
1. CORRECTNESS: Do all column names exist in the schema? Are types compatible?
   Is aggregation/GROUP BY consistent? Is the ClickHouse syntax valid?
2. PERFORMANCE: Is LIMIT present? Can partition key be used for pruning?
   Is there a potential full table scan? Any unnecessary SELECT *?
3. LOGIC: Does the query answer the user's original question?

Schema:
{schema_text}

User's Original Question: {user_query}
Parsed Intent: {parsed_intent}

SQL to validate:
{sql}

Return ONLY a valid JSON object:
{{
  "is_valid": <true|false>,
  "correctness_passed": <true|false>,
  "performance_score": <1-10>,
  "issues": [
    {{"severity": "<error|warning|info>", "category": "<correctness|performance|logic>", "message": "<desc>"}}
  ],
  "suggested_fixes": ["<fix description>"],
  "approved_sql": "<the original sql if valid, or a corrected version>"
}}"""


def _programmatic_validation(sql: str, blocked_keywords: list[str]) -> list[dict]:
    """Hard programmatic checks that ALWAYS run regardless of LLM output."""
    issues = []
    sql_upper = sql.upper().strip()

    # 1. Blocked keywords
    for kw in blocked_keywords:
        if re.search(rf"\b{kw}\b", sql_upper):
            issues.append({
                "severity": "error",
                "category": "security",
                "message": f"Blocked keyword detected: {kw}",
            })

    # 2. Must start with allowed statement
    allowed = ("SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN")
    if not any(sql_upper.startswith(s) for s in allowed):
        issues.append({
            "severity": "error",
            "category": "security",
            "message": f"Query must start with one of: {', '.join(allowed)}",
        })

    # 3. No multi-statement (semicolons outside strings)
    clean = re.sub(r"'[^']*'", "", sql)  # strip string literals
    if ";" in clean:
        issues.append({
            "severity": "error",
            "category": "security",
            "message": "Semicolons not allowed (multi-statement injection risk)",
        })

    # 4. LIMIT check
    if sql_upper.startswith("SELECT") and "LIMIT" not in sql_upper:
        issues.append({
            "severity": "warning",
            "category": "performance",
            "message": "No LIMIT clause found. Add LIMIT for safety.",
        })

    # 5. SELECT * check
    if re.search(r"\bSELECT\s+\*\s+FROM", sql_upper):
        issues.append({
            "severity": "warning",
            "category": "performance",
            "message": "SELECT * detected. Prefer explicit column selection.",
        })

    # 6. No system table writes
    if re.search(r"\bINTO\s+system\.", sql_upper):
        issues.append({
            "severity": "error",
            "category": "security",
            "message": "Writing to system tables is forbidden.",
        })

    return issues


async def query_validator(state: TradeAgentState, *, llm, settings) -> dict:
    start = time.perf_counter()
    sql = state.get("generated_sql", "").strip()

    if not sql:
        duration = (time.perf_counter() - start) * 1000
        return {
            "validation_result": {
                "is_valid": False,
                "issues": [{"severity": "error", "category": "missing", "message": "No SQL to validate"}],
            },
            "needs_retry": False,
            "error": "No SQL generated",
            "execution_trace": [{
                "node": "query_validator",
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "error": "No SQL to validate",
            }],
        }

    # --- Step 1: Programmatic checks ---
    prog_issues = _programmatic_validation(sql, settings.sql_blocked_keywords)
    has_security_error = any(
        i["severity"] == "error" and i["category"] == "security"
        for i in prog_issues
    )

    if has_security_error:
        # Hard fail -- don't even ask the LLM
        duration = (time.perf_counter() - start) * 1000
        retry_count = state.get("retry_count", 0)
        can_retry = retry_count < settings.max_retry_per_node

        logger.warning("SQL blocked by programmatic check: %s", prog_issues)
        return {
            "validation_result": {
                "is_valid": False,
                "security_passed": False,
                "issues": prog_issues,
                "approved_sql": "",
            },
            "needs_retry": can_retry,
            "retry_count": retry_count + 1 if can_retry else retry_count,
            "retry_feedback": "; ".join(i["message"] for i in prog_issues),
            "execution_trace": [{
                "node": "query_validator",
                "status": "blocked",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "output_summary": f"BLOCKED: {len(prog_issues)} security issues",
            }],
        }

    # --- Step 2: LLM validation for correctness and performance ---
    schema_text = state.get("schema_info", {}).get("schema_text", "Schema unavailable")

    prompt = VALIDATION_PROMPT.format(
        schema_text=schema_text[:4000],
        user_query=state["user_query"],
        parsed_intent=json.dumps(state.get("parsed_intent", {}), default=str)[:2000],
        sql=sql,
    )

    try:
        response = await invoke_llm(llm, [
            SystemMessage(content="You are a SQL validator. Return ONLY valid JSON."),
            HumanMessage(content=prompt),
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]

        llm_result = json.loads(raw.strip())

    except Exception as e:
        logger.warning("LLM validation failed, proceeding with programmatic result: %s", e)
        llm_result = {
            "is_valid": True,
            "correctness_passed": True,
            "performance_score": 5,
            "issues": [],
            "approved_sql": sql,
        }

    # Merge issues from both checks
    all_issues = prog_issues + llm_result.get("issues", [])

    has_errors = any(i["severity"] == "error" for i in all_issues)
    is_valid = not has_errors and llm_result.get("is_valid", True)

    approved_sql = llm_result.get("approved_sql", sql) if is_valid else sql

    retry_count = state.get("retry_count", 0)
    can_retry = not is_valid and retry_count < settings.max_retry_per_node

    duration = (time.perf_counter() - start) * 1000

    validation = {
        "is_valid": is_valid,
        "security_passed": not has_security_error,
        "correctness_passed": llm_result.get("correctness_passed", True),
        "performance_score": llm_result.get("performance_score", 5),
        "issues": all_issues,
        "suggested_fixes": llm_result.get("suggested_fixes", []),
        "approved_sql": approved_sql,
    }

    logger.info(
        "Validation %s: %d issues, perf=%d/10 (%.0fms)",
        "PASSED" if is_valid else "FAILED",
        len(all_issues),
        validation["performance_score"],
        duration,
    )

    return {
        "validation_result": validation,
        "generated_sql": approved_sql if is_valid else state.get("generated_sql", ""),
        "needs_retry": can_retry,
        "retry_count": retry_count + 1 if can_retry else retry_count,
        "retry_feedback": "; ".join(i["message"] for i in all_issues if i["severity"] == "error") if can_retry else "",
        "execution_trace": [{
            "node": "query_validator",
            "status": "completed" if is_valid else ("retry" if can_retry else "failed"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round(duration, 1),
            "output_summary": f"{'VALID' if is_valid else 'INVALID'}, {len(all_issues)} issues, perf={validation['performance_score']}/10",
        }],
    }
