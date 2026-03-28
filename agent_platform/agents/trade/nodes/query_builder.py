"""
Query Builder: Generates production-quality ClickHouse SQL.

Uses schema info, query plan, and parsed intent to build optimal queries.
On retry, incorporates error feedback to fix previous SQL.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from services.llm_invoke import invoke_llm

from agents.trade.state import TradeAgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a ClickHouse SQL expert. Generate production-quality ClickHouse SQL
using the schema info, query plan, and parsed intent.

RULES:
1. Use PREWHERE for partition key columns (much faster than WHERE for filtering)
2. Add FORMAT JSONEachRow at the end
3. Always include LIMIT (max {max_rows}) as a safety net
4. Include SETTINGS max_execution_time = {query_timeout}
5. Use ClickHouse functions: toDate(), toStartOfMonth(), toStartOfWeek(), etc.
6. For ReplacingMergeTree tables, add FINAL if the plan says needs_final=true
7. Use appropriate aggregate functions: sum(), avg(), count(), uniq(), quantile()
8. Alias all computed columns for clarity
9. Never use SELECT * -- always list explicit columns
10. Use parameterized values with {{param:Type}} syntax where appropriate

If this is a RETRY attempt, fix the issues described in the retry feedback.

Return ONLY a valid JSON object:
{{
  "sql": "<the complete ClickHouse SQL query>",
  "parameters": {{"<param_name>": "<value>"}},
  "explanation": "<brief explanation of query structure and why>",
  "estimated_scan": "<approximate data volume being scanned>"
}}"""


async def query_builder(state: TradeAgentState, *, llm, settings) -> dict:
    start = time.perf_counter()
    is_retry = state.get("needs_retry", False)
    retry_count = state.get("retry_count", 0)

    schema_text = state.get("schema_info", {}).get("schema_text", "No schema available")

    context_parts = [
        f"User Query: {state['user_query']}",
        f"Parsed Intent: {json.dumps(state.get('parsed_intent', {}), default=str)}",
        f"Query Plan: {json.dumps(state.get('query_plan', {}), default=str)}",
        f"\nSchema:\n{schema_text}",
    ]

    if is_retry:
        feedback = state.get("retry_feedback", "Unknown error")
        prev_sql = state.get("generated_sql", "")
        context_parts.append(f"\n--- RETRY (attempt {retry_count + 1}) ---")
        context_parts.append(f"Previous SQL that failed:\n{prev_sql}")
        context_parts.append(f"Error/Feedback:\n{feedback}")
        context_parts.append("Fix the issues and generate a corrected query.")

    system = SYSTEM_PROMPT.format(
        max_rows=settings.ch_max_rows,
        query_timeout=settings.ch_query_timeout,
    )

    try:
        response = await invoke_llm(llm, [
            SystemMessage(content=system),
            HumanMessage(content="\n\n".join(context_parts)),
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]

        result = json.loads(raw.strip())
        sql = result.get("sql", "").strip()
        params = result.get("parameters", {})
        duration = (time.perf_counter() - start) * 1000

        logger.info(
            "SQL generated%s: %s... (%.0fms)",
            f" (retry {retry_count + 1})" if is_retry else "",
            sql[:120],
            duration,
        )

        return {
            "generated_sql": sql,
            "sql_parameters": params,
            "needs_retry": False,  # Reset; validator may set it again
            "execution_trace": [{
                "node": "query_builder",
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "output_summary": f"{'Retry ' + str(retry_count + 1) + ': ' if is_retry else ''}{sql[:100]}...",
            }],
        }

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        logger.error("Query builder failed: %s", e)
        return {
            "generated_sql": "",
            "sql_parameters": {},
            "error": f"Query builder failed: {e}",
            "execution_trace": [{
                "node": "query_builder",
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "error": str(e),
            }],
        }
