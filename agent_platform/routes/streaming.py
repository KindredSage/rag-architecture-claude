"""
SSE event stream generator.

Streams LangGraph execution events to the client. Each node's own
execution_trace entry is forwarded directly - no per-node if/else
reconstruction. Nodes already produce their own timing and summaries.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any


def sse_line(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data, default=str)}\n\n"


async def stream_graph_execution(
    graph,
    input_state: dict,
    *,
    run_id: str,
    session_id: str,
    query: str,
    session_manager,
):
    """
    Async generator that yields SSE events from a LangGraph execution.

    Events emitted:
      run_started    - once at start
      step_started   - when a node begins (name + elapsed_ms)
      step_completed - when a node ends (trace entry forwarded directly)
      run_completed   - once at end with final answer
      run_error      - on failure
    """
    yield sse_line({
        "event": "run_started",
        "run_id": run_id,
        "session_id": session_id,
        "query": query,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    start = time.monotonic()
    step_starts: dict[str, float] = {}
    final_result = None
    skip_names = ("__start__", "__end__", "", "RunnableSequence")

    try:
        async for event in graph.astream_events(input_state, version="v2"):
            etype = event.get("event", "")
            name = event.get("name", "")

            if name in skip_names:
                continue

            elapsed = round((time.monotonic() - start) * 1000, 1)

            if etype == "on_chain_start":
                step_starts[name] = time.monotonic()
                yield sse_line({
                    "event": "step_started",
                    "step": name,
                    "elapsed_ms": elapsed,
                })

            elif etype == "on_chain_end":
                step_start = step_starts.pop(name, None)
                step_ms = round((time.monotonic() - step_start) * 1000, 1) if step_start else None

                output = event.get("data", {}).get("output", {})
                payload: dict[str, Any] = {
                    "event": "step_completed",
                    "step": name,
                    "elapsed_ms": elapsed,
                }

                if step_ms is not None:
                    payload["duration_ms"] = step_ms

                # Forward the node's own trace entry (it already has
                # status, duration_ms, output_summary, error, etc.)
                if isinstance(output, dict):
                    trace = output.get("execution_trace", [])
                    if trace:
                        last_entry = trace[-1]
                        payload["trace"] = {
                            k: v for k, v in last_entry.items()
                            if k in ("status", "duration_ms", "output_summary",
                                     "error", "node")
                        }

                    if name == "merge_results":
                        final_result = output

                yield sse_line(payload)

        # -- Run complete --
        timing_ms = round((time.monotonic() - start) * 1000, 1)

        if final_result:
            resp = final_result.get("final_response", {})
            is_waiting = resp.get("status") == "waiting_human"

            yield sse_line({
                "event": "run_completed",
                "status": "waiting_human" if is_waiting else "completed",
                "answer": resp.get("answer", ""),
                "confidence": resp.get("confidence", 0),
                "suggestions": resp.get("suggestions", []),
                "timing_ms": timing_ms,
                "interrupt": resp.get("interrupt") if is_waiting else None,
            })

            if session_manager:
                await session_manager.complete_run(
                    run_id=run_id,
                    status="waiting_human" if is_waiting else "completed",
                    result=resp,
                    timing_ms=timing_ms,
                )
        else:
            yield sse_line({
                "event": "run_completed",
                "status": "completed",
                "timing_ms": timing_ms,
            })

    except Exception as e:
        elapsed = round((time.monotonic() - start) * 1000, 1)
        yield sse_line({
            "event": "run_error",
            "error": str(e),
            "elapsed_ms": elapsed,
        })
        if session_manager:
            await session_manager.complete_run(
                run_id=run_id, status="failed", error=str(e),
            )
