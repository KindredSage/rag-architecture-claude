"""
Node execution wrapper with consistent timing, tracing, and event emission.

Every node in every agent graph should be wrapped with `timed_node()`.
This ensures:
  1. Precise wall-clock timing (start, end, duration_ms) in every trace entry
  2. Consistent trace format across all nodes
  3. Error capture without crashing the graph
  4. Optional callback for live event emission (SSE)

Usage in graph.py:
    from agents.node_runner import timed_node

    async def _my_node(state):
        return await my_node_func(state, llm=llm)

    graph.add_node("my_node", timed_node("my_node", "trade_agent", _my_node))
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

# Type for node functions: takes state dict, returns state update dict
NodeFunc = Callable[[dict], Awaitable[dict]]

# Type for event callback: called when a node starts/completes/fails
EventCallback = Callable[[dict], Awaitable[None]] | None


def timed_node(
    node_name: str,
    agent_id: str,
    func: NodeFunc,
    *,
    event_callback: EventCallback = None,
) -> NodeFunc:
    """
    Wrap a node function with consistent timing and tracing.

    Args:
        node_name: Name of this node (e.g., "query_builder").
        agent_id: Which agent this belongs to (e.g., "trade_agent").
        func: The actual async node function.
        event_callback: Optional async callback to emit live events.

    Returns:
        Wrapped node function that adds timing to execution_trace.
    """

    async def wrapper(state: dict) -> dict:
        start_wall = time.perf_counter()
        start_ts = datetime.now(timezone.utc)

        # Emit start event
        start_event = {
            "event": "node_start",
            "node": node_name,
            "agent_id": agent_id,
            "timestamp": start_ts.isoformat(),
        }
        if event_callback:
            try:
                await event_callback(start_event)
            except Exception:
                pass  # never let callback errors break the graph

        try:
            result = await func(state)

            duration_ms = (time.perf_counter() - start_wall) * 1000
            end_ts = datetime.now(timezone.utc)

            # Extract output summary from the result's own trace if present
            own_trace = result.get("execution_trace", [])
            output_summary = ""
            if own_trace:
                output_summary = own_trace[-1].get("output_summary", "")

            # Build enriched trace entry
            trace_entry = {
                "node": node_name,
                "agent_id": agent_id,
                "status": "completed",
                "started_at": start_ts.isoformat(),
                "completed_at": end_ts.isoformat(),
                "duration_ms": round(duration_ms, 1),
                "output_summary": output_summary,
            }

            # Merge our enriched trace with any trace the node produced
            # Replace the node's own trace entries with our enriched version
            enriched_trace = []
            for entry in own_trace:
                if entry.get("node") == node_name:
                    # Replace with our enriched version (keeps output_summary, adds timing)
                    merged = {**entry, **trace_entry}
                    enriched_trace.append(merged)
                else:
                    enriched_trace.append(entry)

            # If the node didn't produce its own trace, add ours
            if not any(e.get("node") == node_name for e in enriched_trace):
                enriched_trace.append(trace_entry)

            result["execution_trace"] = enriched_trace

            # Emit completion event with details
            end_event = {
                "event": "node_end",
                "node": node_name,
                "agent_id": agent_id,
                "status": "completed",
                "duration_ms": round(duration_ms, 1),
                "timestamp": end_ts.isoformat(),
                "output_summary": output_summary,
                # Include select fields from result for rich SSE events
                **_extract_event_payload(node_name, result),
            }
            if event_callback:
                try:
                    await event_callback(end_event)
                except Exception:
                    pass

            logger.debug(
                "[%s.%s] completed in %.1fms", agent_id, node_name, duration_ms,
            )
            return result

        except Exception as e:
            duration_ms = (time.perf_counter() - start_wall) * 1000
            end_ts = datetime.now(timezone.utc)

            logger.error(
                "[%s.%s] failed after %.1fms: %s",
                agent_id, node_name, duration_ms, e,
            )

            error_trace = {
                "node": node_name,
                "agent_id": agent_id,
                "status": "error",
                "started_at": start_ts.isoformat(),
                "completed_at": end_ts.isoformat(),
                "duration_ms": round(duration_ms, 1),
                "error": str(e),
                "traceback": traceback.format_exc()[-500:],
            }

            # Emit error event
            if event_callback:
                try:
                    await event_callback({
                        "event": "node_error",
                        "node": node_name,
                        "agent_id": agent_id,
                        "duration_ms": round(duration_ms, 1),
                        "error": str(e),
                        "timestamp": end_ts.isoformat(),
                    })
                except Exception:
                    pass

            # Return error state without crashing the graph
            return {
                "error": str(e),
                "execution_trace": [error_trace],
            }

    return wrapper


def _extract_event_payload(node_name: str, result: dict) -> dict:
    """
    Extract interesting fields from a node's result for live SSE events.
    Different nodes expose different data.
    """
    payload: dict[str, Any] = {}

    if node_name == "classify_intent":
        intent = result.get("intent_analysis", {})
        payload["intent"] = {
            "domain": intent.get("primary_domain"),
            "intent": intent.get("intent"),
            "complexity": intent.get("complexity"),
            "desired_output": intent.get("desired_output"),
            "ambiguity": intent.get("ambiguity_notes", ""),
        }

    elif node_name == "select_agents":
        decision = result.get("routing_decision", {})
        payload["routing"] = {
            "selected": [
                {"agent_id": s.get("agent_id"), "reason": s.get("reason")}
                for s in decision.get("selected_agents", [])
            ],
            "strategy": decision.get("execution_strategy", "sequential"),
        }

    elif node_name == "trade_analyst":
        ctx = result.get("trade_context", {})
        payload["trade_context"] = {
            "asset_class": ctx.get("asset_class"),
            "metrics": ctx.get("relevant_metrics", [])[:5],
            "suggested_tables": ctx.get("suggested_tables", []),
        }

    elif node_name == "schema_analyzer":
        schema = result.get("schema_info", {})
        tables = schema.get("tables", {})
        payload["schema"] = {
            "tables_found": list(tables.keys())[:10],
            "total_columns": sum(
                len(t.get("columns", [])) for t in tables.values()
            ),
        }

    elif node_name == "query_builder":
        payload["sql_preview"] = (result.get("generated_sql", ""))[:300]

    elif node_name == "query_validator":
        val = result.get("validation_result", {})
        payload["validation"] = {
            "is_valid": val.get("is_valid"),
            "security_passed": val.get("security_passed"),
            "performance_score": val.get("performance_score"),
            "issue_count": len(val.get("issues", [])),
        }

    elif node_name == "sql_approval_gate":
        pending = result.get("hitl_pending", {})
        resp = result.get("hitl_response", {})
        if pending:
            payload["hitl"] = {"status": "waiting", "interrupt_id": pending.get("interrupt_id")}
        elif resp:
            payload["hitl"] = {"status": resp.get("action", "unknown")}

    elif node_name == "query_executor":
        qr = result.get("query_results", {})
        payload["execution"] = {
            "success": qr.get("success"),
            "row_count": qr.get("row_count", 0),
            "ch_time_ms": qr.get("execution_time_ms", 0),
            "bytes_read": qr.get("bytes_read", 0),
            "truncated": qr.get("truncated", False),
        }

    elif node_name == "details_analyzer":
        analysis = result.get("analysis", {})
        payload["analysis"] = {
            "findings_count": len(analysis.get("key_findings", [])),
            "charts_recommended": len(analysis.get("visualization_recommendations", [])),
            "confidence": analysis.get("confidence", 0),
        }
        artifacts = result.get("artifacts", [])
        if artifacts:
            payload["artifacts"] = [
                {"type": a.get("type"), "name": a.get("name")}
                for a in artifacts[:5]
            ]

    elif node_name == "merge_results":
        final = result.get("final_response", {})
        payload["final"] = {
            "confidence": final.get("confidence", 0),
            "has_data": bool(final.get("data")),
            "suggestions_count": len(final.get("suggestions", [])),
        }

    return payload
