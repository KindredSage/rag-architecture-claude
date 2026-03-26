"""
FastAPI application entrypoint.

Endpoints:
  GET  /health                  - Health check
  POST /agents                  - List registered agents
  POST /execute                 - Execute user query (sync)
  POST /execute/stream          - Execute with SSE streaming
  GET  /execute/{run_id}        - Get run status / result
  GET  /sessions/{session_id}   - Get session info
  DELETE /sessions/{session_id} - Close session
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from config import Settings, get_settings
from dependencies import check_concurrency_limit, get_services, get_session_manager
from middleware.auth import verify_api_key
from middleware.rate_limiter import setup_rate_limiter
from models import (
    AgentInfoResponse,
    ExecuteRequest,
    ExecuteResponse,
    HealthResponse,
    RunStatus,
)

# ── Logging ──────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_platform")

# ── Server Identity ──────────────────────────────────────────────

SERVER_ID = os.getenv("SERVER_ID", f"server-{uuid.uuid4().hex[:8]}")


# =====================================================================
# Lifespan
# =====================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        "Starting %s v%s (server=%s, env=%s)",
        settings.app_name, settings.app_version, SERVER_ID, settings.environment,
    )

    # ── Initialize services ──────────────────────────────────────
    from services.cache_service import CacheService
    from services.clickhouse_service import ClickHouseService
    from services.llm_service import LLMService
    from services.session_manager import SessionManager
    from tools.mcp_client import MCPClientManager

    ch = ClickHouseService(settings)
    ch.initialize()

    llm = LLMService(settings)
    llm.initialize()

    cache = CacheService(settings)
    await cache.initialize()

    sm = SessionManager(settings, SERVER_ID)
    await sm.initialize()

    mcp = MCPClientManager(settings)
    await mcp.initialize()

    # ── Build tool sets ──────────────────────────────────────────
    from tools.clickhouse_tools import create_clickhouse_tools
    from tools.email_tools import create_email_tools
    from tools.export_tools import create_export_tools
    from tools.plotting_tools import create_plotting_tools

    os.makedirs(settings.artifact_dir, exist_ok=True)

    ch_tools = create_clickhouse_tools(ch)
    plotting_tools = create_plotting_tools(settings.artifact_dir)
    email_tools = create_email_tools(settings)
    export_tools = create_export_tools(settings.artifact_dir)

    # If MCP provides ClickHouse tools, prefer those
    if mcp.available:
        mcp_ch = mcp.get_tools_by_prefix("clickhouse_")
        if mcp_ch:
            ch_tools = mcp_ch
            logger.info("Using MCP ClickHouse tools (%d tools)", len(mcp_ch))

    # ── Service registry ─────────────────────────────────────────
    services: dict[str, Any] = {
        "clickhouse": ch,
        "llm": llm,
        "cache": cache,
        "session_manager": sm,
        "mcp": mcp,
        "ch_tools": ch_tools,
        "plotting_tools": plotting_tools,
        "email_tools": email_tools,
        "export_tools": export_tools,
    }
    app.state.services = services
    app.state.server_id = SERVER_ID

    # ── Register agents (importing triggers AgentRegistry.register) ─
    import agents.trade  # noqa: F401
    import agents.reporting  # noqa: F401
    import agents.analytics  # noqa: F401

    from agents.registry import AgentRegistry

    logger.info(
        "Agents registered: %s",
        [a.agent_id for a in AgentRegistry.get_all()],
    )

    # ── Build master graph ───────────────────────────────────────
    from agents.master_agent import build_master_graph

    app.state.master_graph = build_master_graph(
        settings=settings, services=services,
    )
    logger.info("Master graph compiled")

    # ── Background: session cleanup ──────────────────────────────
    cleanup_task = asyncio.create_task(_session_cleanup_loop(sm, settings))
    logger.info("Startup complete")

    yield

    # ── Shutdown ──────────────────────────────────────────────────
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    await mcp.shutdown()
    await cache.shutdown()
    await sm.shutdown()
    ch.close()
    logger.info("Shutdown complete")


async def _session_cleanup_loop(sm, settings: Settings):
    """Periodic background task: expire stale sessions."""
    while True:
        try:
            await asyncio.sleep(settings.session_cleanup_interval_minutes * 60)
            await sm.cleanup_expired_sessions()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Session cleanup error: %s", e)


# =====================================================================
# App
# =====================================================================

app = FastAPI(
    title="Agent Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_rate_limiter(app)


# =====================================================================
# Endpoints
# =====================================================================


@app.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check returning status of all dependencies."""
    services = request.app.state.services
    ch: Any = services["clickhouse"]
    cache: Any = services["cache"]
    sm: Any = services["session_manager"]

    from agents.registry import AgentRegistry

    active_runs = await sm.get_active_run_count()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        server_id=request.app.state.server_id,
        clickhouse="connected" if ch.ping() else "disconnected",
        postgres="connected",
        redis="connected" if await cache.ping() else "disconnected",
        active_runs=active_runs,
        registered_agents=len(AgentRegistry.get_all()),
    )


@app.post("/agents", response_model=list[AgentInfoResponse])
async def list_agents(_: Any = Depends(verify_api_key)):
    """Return all registered agents with capabilities and examples."""
    from agents.registry import AgentRegistry

    return [
        AgentInfoResponse(
            agent_id=d.agent_id,
            name=d.name,
            description=d.description,
            capabilities=[c.value for c in d.capabilities],
            sub_agents=d.sub_agents,
            example_queries=d.example_queries,
            version=d.version,
            enabled=d.enabled,
        )
        for d in AgentRegistry.get_all()
    ]


@app.post("/execute", response_model=ExecuteResponse)
async def execute_query(
    req: ExecuteRequest,
    request: Request,
    _: Any = Depends(verify_api_key),
    __: Any = Depends(check_concurrency_limit),
):
    """Execute a user query through the master agent pipeline (synchronous)."""
    settings = get_settings()
    services = request.app.state.services
    sm = services["session_manager"]

    run_id = str(uuid.uuid4())
    session_id = await sm.get_or_create_session(req.session_id, req.user_id)

    # Conversation history for context
    history = await sm.get_history(session_id, limit=10)

    # Record
    await sm.create_run(run_id, session_id, req.query)
    await sm.add_message(session_id, run_id, "user", {"query": req.query})

    start = time.monotonic()

    input_state = {
        "user_query": req.query,
        "user_context": {
            "user_id": req.user_id,
            "preferences": req.preferences,
            "forced_agent": req.agent_id,
            "history": history,
            "session_id": session_id,
        },
        "agent_results": {},
        "execution_trace": [],
        "artifacts": [],
        "error": None,
    }

    try:
        result = await asyncio.wait_for(
            request.app.state.master_graph.ainvoke(input_state),
            timeout=settings.run_timeout,
        )

        timing_ms = (time.monotonic() - start) * 1000
        final = result.get("final_response", {})
        error = result.get("error")
        status = RunStatus.COMPLETED if not error else RunStatus.PARTIAL

        # Persist
        await sm.complete_run(
            run_id=run_id,
            status=status.value,
            result=final,
            trace=result.get("execution_trace", []),
            artifacts=[
                a if isinstance(a, dict) else {}
                for a in result.get("artifacts", [])
            ],
            timing_ms=timing_ms,
            error=error,
        )
        await sm.add_message(session_id, run_id, "assistant", {
            "answer": final.get("answer", ""),
            "status": status.value,
        })

        return ExecuteResponse(
            run_id=run_id,
            session_id=session_id,
            status=status,
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
        await sm.complete_run(
            run_id=run_id, status="failed",
            error="Execution timed out", timing_ms=timing_ms,
        )
        raise HTTPException(status_code=504, detail="Execution timed out")

    except Exception as e:
        timing_ms = (time.monotonic() - start) * 1000
        logger.error("Execution failed: %s", e, exc_info=True)
        await sm.complete_run(
            run_id=run_id, status="failed",
            error=str(e), timing_ms=timing_ms,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/execute/stream")
async def execute_query_stream(
    req: ExecuteRequest,
    request: Request,
    _: Any = Depends(verify_api_key),
    __: Any = Depends(check_concurrency_limit),
):
    """SSE stream of execution progress, yielding events per sub-agent step."""
    services = request.app.state.services
    sm = services["session_manager"]

    run_id = str(uuid.uuid4())
    session_id = await sm.get_or_create_session(req.session_id, req.user_id)
    history = await sm.get_history(session_id, limit=10)
    await sm.create_run(run_id, session_id, req.query)

    input_state = {
        "user_query": req.query,
        "user_context": {
            "user_id": req.user_id,
            "preferences": req.preferences,
            "forced_agent": req.agent_id,
            "history": history,
            "session_id": session_id,
        },
        "agent_results": {},
        "execution_trace": [],
        "artifacts": [],
        "error": None,
    }

    async def event_generator():
        yield _sse({"run_id": run_id, "session_id": session_id, "status": "started"})

        start = time.monotonic()
        final_result = None

        try:
            async for event in request.app.state.master_graph.astream_events(
                input_state, version="v2",
            ):
                etype = event.get("event", "")
                name = event.get("name", "")

                if etype == "on_chain_start" and name not in ("__start__", ""):
                    yield _sse({"step": name, "status": "started"})

                elif etype == "on_chain_end" and name not in ("__end__", ""):
                    output = event.get("data", {}).get("output", {})
                    summary = ""
                    if isinstance(output, dict):
                        trace = output.get("execution_trace", [])
                        if trace:
                            summary = trace[-1].get("output_summary", "")
                    yield _sse({"step": name, "status": "completed", "summary": summary})

                    if name == "merge_results":
                        final_result = output

            timing_ms = (time.monotonic() - start) * 1000

            if final_result:
                resp = final_result.get("final_response", {})
                yield _sse({
                    "status": "completed",
                    "answer": resp.get("answer", ""),
                    "timing_ms": round(timing_ms, 1),
                })
                await sm.complete_run(
                    run_id=run_id, status="completed",
                    result=resp, timing_ms=timing_ms,
                )
            else:
                yield _sse({"status": "completed", "timing_ms": round(timing_ms, 1)})

        except Exception as e:
            logger.error("Stream error: %s", e)
            yield _sse({"status": "error", "error": str(e)})
            await sm.complete_run(run_id=run_id, status="failed", error=str(e))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/execute/{run_id}")
async def get_run_status(
    run_id: str,
    request: Request,
    _: Any = Depends(verify_api_key),
):
    """Get the status and result of a previous run."""
    sm = request.app.state.services["session_manager"]
    run = await sm.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

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


@app.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    request: Request,
    _: Any = Depends(verify_api_key),
):
    """Get session info and conversation history."""
    sm = request.app.state.services["session_manager"]
    session = await sm.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    history = await sm.get_history(session_id, limit=50)
    return {
        "session_id": session["session_id"],
        "user_id": session.get("user_id"),
        "server_id": session["server_id"],
        "created_at": session["created_at"].isoformat(),
        "last_active": session["last_active"].isoformat(),
        "history": history,
    }


@app.delete("/sessions/{session_id}")
async def close_session(
    session_id: str,
    request: Request,
    _: Any = Depends(verify_api_key),
):
    """Close a session (mark inactive)."""
    sm = request.app.state.services["session_manager"]
    await sm.close_session(session_id)
    return {"status": "closed", "session_id": session_id}


# ── Helpers ──────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data, default=str)}\n\n"
