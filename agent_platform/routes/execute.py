from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from config import get_settings
from dependencies import check_concurrency_limit
from middleware.auth import verify_api_key
from models import ExecuteRequest, ExecuteResponse, RunStatus
from routes.streaming import stream_graph_execution

router = APIRouter(tags=["execute"])


def _build_input_state(
    req: ExecuteRequest,
    *,
    run_id: str,
    session_id: str,
    history: list[dict],
) -> dict:
    """Build the master agent input state. Single source of truth."""
    return {
        "user_query": req.query,
        "user_context": {
            "user_id": req.user_id,
            "preferences": req.preferences,
            "forced_agent": req.agent_id,
            "history": history,
            "session_id": session_id,
            "run_id": run_id,
            "hitl_config": req.preferences.get("hitl", {}),
        },
        "agent_results": {},
        "execution_trace": [],
        "artifacts": [],
        "error": None,
    }


@router.post("/execute", response_model=ExecuteResponse)
async def execute_query(
    req: ExecuteRequest,
    request: Request,
    _: Any = Depends(verify_api_key),
    __: Any = Depends(check_concurrency_limit),
):
    """Execute a user query through the master agent pipeline."""
    settings = get_settings()
    sm = request.app.state.services["session_manager"]

    run_id = str(uuid.uuid4())
    session_id = await sm.get_or_create_session(req.session_id, req.user_id)
    history = await sm.get_history(session_id, limit=10)

    await sm.create_run(run_id, session_id, req.query)
    await sm.add_message(session_id, run_id, "user", {"query": req.query})

    input_state = _build_input_state(
        req, run_id=run_id, session_id=session_id, history=history,
    )
    start = time.monotonic()

    try:
        result = await asyncio.wait_for(
            request.app.state.master_graph.ainvoke(input_state),
            timeout=settings.run_timeout,
        )

        timing_ms = (time.monotonic() - start) * 1000
        final = result.get("final_response", {})
        result_status = result.get("status")
        error = result.get("error")

        if result_status == "waiting_human":
            status = RunStatus.WAITING_HUMAN
            error = None
        elif error:
            status = RunStatus.PARTIAL
        else:
            status = RunStatus.COMPLETED

        await sm.complete_run(
            run_id=run_id, status=status.value, result=final,
            trace=result.get("execution_trace", []),
            artifacts=[a for a in result.get("artifacts", []) if isinstance(a, dict)],
            timing_ms=timing_ms, error=error,
        )
        await sm.add_message(session_id, run_id, "assistant", {
            "answer": final.get("answer", ""), "status": status.value,
        })

        return ExecuteResponse(
            run_id=run_id, session_id=session_id, status=status,
            answer=final.get("answer", ""),
            data=final.get("data"),
            visualizations=final.get("visualizations", []),
            artifacts=result.get("artifacts", []),
            suggestions=final.get("suggestions", []),
            execution_trace=result.get("execution_trace", []),
            confidence=final.get("confidence", 0.0),
            timing_ms=round(timing_ms, 1),
            error=error,
        )

    except asyncio.TimeoutError:
        timing_ms = (time.monotonic() - start) * 1000
        await sm.complete_run(run_id=run_id, status="failed",
                              error="Execution timed out", timing_ms=timing_ms)
        raise HTTPException(504, "Execution timed out")
    except Exception as e:
        timing_ms = (time.monotonic() - start) * 1000
        await sm.complete_run(run_id=run_id, status="failed",
                              error=str(e), timing_ms=timing_ms)
        raise HTTPException(500, str(e))


@router.post("/execute/stream")
async def execute_query_stream(
    req: ExecuteRequest,
    request: Request,
    _: Any = Depends(verify_api_key),
    __: Any = Depends(check_concurrency_limit),
):
    """SSE stream of execution progress."""
    sm = request.app.state.services["session_manager"]

    run_id = str(uuid.uuid4())
    session_id = await sm.get_or_create_session(req.session_id, req.user_id)
    history = await sm.get_history(session_id, limit=10)
    await sm.create_run(run_id, session_id, req.query)

    input_state = _build_input_state(
        req, run_id=run_id, session_id=session_id, history=history,
    )

    return StreamingResponse(
        stream_graph_execution(
            request.app.state.master_graph,
            input_state,
            run_id=run_id,
            session_id=session_id,
            query=req.query,
            session_manager=sm,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/execute/{run_id}")
async def get_run_status(
    run_id: str,
    request: Request,
    _: Any = Depends(verify_api_key),
):
    """Get the status and result of a previous run."""
    sm = request.app.state.services["session_manager"]
    run = await sm.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found")

    result = run.get("result")
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "run_id": run["run_id"],
        "session_id": run["session_id"],
        "status": run["status"],
        "user_query": run["user_query"],
        "result": result,
        "timing_ms": run.get("timing_ms"),
        "error": run.get("error"),
        "created_at": run["created_at"].isoformat() if run.get("created_at") else None,
        "completed_at": run["completed_at"].isoformat() if run.get("completed_at") else None,
    }
