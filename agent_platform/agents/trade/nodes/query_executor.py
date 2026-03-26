"""
Query Executor: Executes validated SQL against ClickHouse.

Only proceeds if validation passed. Uses ClickHouseService for execution
with all safety guardrails (readonly session, timeout, row limit).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from agents.trade.state import TradeAgentState

logger = logging.getLogger(__name__)


async def query_executor(state: TradeAgentState, *, ch_service, cache_service=None) -> dict:
    start = time.perf_counter()

    validation = state.get("validation_result", {})
    if not validation.get("is_valid", False):
        duration = (time.perf_counter() - start) * 1000
        error_msg = "Query validation failed. Cannot execute."
        issues = validation.get("issues", [])
        if issues:
            error_msg += " Issues: " + "; ".join(
                i.get("message", "") for i in issues if i.get("severity") == "error"
            )
        return {
            "query_results": {
                "success": False,
                "data": [],
                "columns": [],
                "row_count": 0,
                "execution_time_ms": 0,
                "error": error_msg,
            },
            "error": error_msg,
            "execution_trace": [{
                "node": "query_executor",
                "status": "skipped",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "error": error_msg,
            }],
        }

    sql = validation.get("approved_sql", "") or state.get("generated_sql", "")
    params = state.get("sql_parameters", {})

    if not sql:
        duration = (time.perf_counter() - start) * 1000
        return {
            "query_results": {
                "success": False, "data": [], "columns": [],
                "row_count": 0, "error": "No SQL to execute",
            },
            "error": "No SQL to execute",
            "execution_trace": [{
                "node": "query_executor",
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "error": "No SQL to execute",
            }],
        }

    # Check cache
    if cache_service:
        cached = await cache_service.get_query_result(sql, params)
        if cached:
            duration = (time.perf_counter() - start) * 1000
            logger.info("Cache hit for query (%.0fms)", duration)
            cached["from_cache"] = True
            return {
                "query_results": cached,
                "execution_trace": [{
                    "node": "query_executor",
                    "status": "completed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round(duration, 1),
                    "output_summary": f"CACHE HIT: {cached.get('row_count', 0)} rows",
                }],
            }

    # Execute query
    # Strip FORMAT clause if present -- clickhouse-connect handles formatting
    clean_sql = sql
    for fmt in ("FORMAT JSONEachRow", "FORMAT JSON", "FORMAT TabSeparated"):
        clean_sql = clean_sql.replace(fmt, "").strip()
    # Remove trailing semicolons
    clean_sql = clean_sql.rstrip(";").strip()

    result = ch_service.execute_query(clean_sql, params)
    duration = (time.perf_counter() - start) * 1000

    if result["success"]:
        logger.info(
            "Query executed: %d rows, %.1fms CH time, %.1fms total",
            result["row_count"],
            result["execution_time_ms"],
            duration,
        )

        # Cache successful results
        if cache_service and result["row_count"] > 0:
            await cache_service.set_query_result(sql, result, params, ttl=120)

        return {
            "query_results": result,
            "execution_trace": [{
                "node": "query_executor",
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "output_summary": (
                    f"{result['row_count']} rows, "
                    f"{result['execution_time_ms']:.0f}ms CH, "
                    f"{'TRUNCATED' if result['truncated'] else 'complete'}"
                ),
            }],
        }
    else:
        # Execution failed -- check if we can retry
        retry_count = state.get("retry_count", 0)
        max_retries = 3
        can_retry = retry_count < max_retries

        error = result.get("error", "Unknown execution error")
        logger.error("Query execution failed: %s", error)

        return {
            "query_results": result,
            "needs_retry": can_retry,
            "retry_count": retry_count + 1 if can_retry else retry_count,
            "retry_feedback": f"ClickHouse execution error: {error}",
            "error": error if not can_retry else None,
            "execution_trace": [{
                "node": "query_executor",
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "error": error,
                "output_summary": f"FAILED: {error[:100]}{'...' if len(error) > 100 else ''}",
            }],
        }
