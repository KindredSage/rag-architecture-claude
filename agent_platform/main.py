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
from datetime import datetime, timezone
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

    # ── HITL Service ────────────────────────────────────────────
    from services.hitl import HITLService

    hitl_svc = HITLService(sm.pool)
    await hitl_svc.ensure_tables()
    logger.info("HITL service initialized")

    # ── Service registry ─────────────────────────────────────────
    services: dict[str, Any] = {
        "clickhouse": ch,
        "llm": llm,
        "cache": cache,
        "session_manager": sm,
        "hitl": hitl_svc,
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

    # Extract HITL config from preferences
    hitl_config = req.preferences.get("hitl", {})

    input_state = {
        "user_query": req.query,
        "user_context": {
            "user_id": req.user_id,
            "preferences": req.preferences,
            "forced_agent": req.agent_id,
            "history": history,
            "session_id": session_id,
            "run_id": run_id,
            "hitl_config": hitl_config,
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

        # Determine status
        if error == "waiting_human":
            status = RunStatus.WAITING_HUMAN
            error = None  # Not a real error
        elif error:
            status = RunStatus.PARTIAL
        else:
            status = RunStatus.COMPLETED

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
        yield _sse({
            "event": "run_started",
            "run_id": run_id,
            "session_id": session_id,
            "query": req.query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        start = time.monotonic()
        final_result = None
        # Track cumulative timing for a running total
        step_timings: dict[str, float] = {}

        try:
            async for event in request.app.state.master_graph.astream_events(
                input_state, version="v2",
            ):
                etype = event.get("event", "")
                name = event.get("name", "")
                tags = event.get("tags", [])

                # Skip internal LangGraph plumbing nodes
                if name in ("__start__", "__end__", "", "RunnableSequence"):
                    continue

                elapsed_ms = round((time.monotonic() - start) * 1000, 1)

                if etype == "on_chain_start":
                    step_timings[name] = time.monotonic()
                    yield _sse({
                        "event": "step_started",
                        "step": name,
                        "elapsed_ms": elapsed_ms,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    })

                elif etype == "on_chain_end":
                    output = event.get("data", {}).get("output", {})

                    # Calculate step duration
                    step_start = step_timings.pop(name, None)
                    step_ms = round((time.monotonic() - step_start) * 1000, 1) if step_start else None

                    # Extract rich details from execution trace
                    details: dict[str, Any] = {
                        "event": "step_completed",
                        "step": name,
                        "duration_ms": step_ms,
                        "elapsed_ms": elapsed_ms,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                    if isinstance(output, dict):
                        trace = output.get("execution_trace", [])
                        if trace:
                            last = trace[-1]
                            details["summary"] = last.get("output_summary", "")
                            details["status"] = last.get("status", "completed")
                            if last.get("error"):
                                details["error"] = last["error"]

                            # Copy node-specific timing from timed_node wrapper
                            if last.get("started_at"):
                                details["started_at"] = last["started_at"]
                            if last.get("completed_at"):
                                details["completed_at"] = last["completed_at"]

                        # ── Emit node-specific rich payloads ─────────
                        if name == "classify_intent":
                            intent = output.get("intent_analysis", {})
                            details["intent"] = {
                                "domain": intent.get("primary_domain"),
                                "intent": intent.get("intent"),
                                "complexity": intent.get("complexity"),
                                "desired_output": intent.get("desired_output"),
                                "entities": intent.get("entities", [])[:5],
                                "ambiguity": intent.get("ambiguity_notes", ""),
                            }

                        elif name == "select_agents":
                            decision = output.get("routing_decision", {})
                            details["routing"] = {
                                "selected_agents": [
                                    {"agent_id": s.get("agent_id"),
                                     "reason": s.get("reason"),
                                     "confidence": s.get("confidence")}
                                    for s in decision.get("selected_agents", [])
                                ],
                                "strategy": decision.get("execution_strategy", "sequential"),
                            }

                        elif name == "trade_analyst":
                            ctx = output.get("trade_context", {})
                            details["trade_context"] = {
                                "asset_class": ctx.get("asset_class"),
                                "metrics": ctx.get("relevant_metrics", [])[:5],
                                "tables": ctx.get("suggested_tables", []),
                                "resolved_query": ctx.get("resolved_query", ""),
                            }

                        elif name == "schema_analyzer":
                            schema = output.get("schema_info", {})
                            tables = schema.get("tables", {})
                            details["schema"] = {
                                "tables_found": list(tables.keys())[:10],
                                "total_columns": sum(len(t.get("columns", [])) for t in tables.values()),
                            }

                        elif name == "query_builder":
                            sql = output.get("generated_sql", "")
                            details["sql_preview"] = sql[:500]

                        elif name == "query_validator":
                            val = output.get("validation_result", {})
                            details["validation"] = {
                                "is_valid": val.get("is_valid"),
                                "security_passed": val.get("security_passed"),
                                "performance_score": val.get("performance_score"),
                                "issue_count": len(val.get("issues", [])),
                                "issues": [i.get("message", "") for i in val.get("issues", [])[:3]],
                            }

                        elif name == "sql_approval_gate":
                            pending = output.get("hitl_pending", {})
                            resp = output.get("hitl_response", {})
                            if pending.get("status") == "pending":
                                details["hitl"] = {
                                    "status": "waiting",
                                    "interrupt_id": pending.get("interrupt_id"),
                                    "type": pending.get("type"),
                                }
                            elif resp:
                                details["hitl"] = {"status": resp.get("action", "unknown")}

                        elif name == "query_executor":
                            qr = output.get("query_results", {})
                            details["execution"] = {
                                "success": qr.get("success"),
                                "row_count": qr.get("row_count", 0),
                                "clickhouse_ms": qr.get("execution_time_ms", 0),
                                "bytes_read": qr.get("bytes_read", 0),
                                "truncated": qr.get("truncated", False),
                            }

                        elif name == "details_analyzer":
                            analysis = output.get("analysis", {})
                            details["analysis"] = {
                                "findings_count": len(analysis.get("key_findings", [])),
                                "charts_recommended": len(analysis.get("visualization_recommendations", [])),
                                "confidence": analysis.get("confidence", 0),
                                "export": analysis.get("export_recommendation", "none"),
                            }
                            artifacts = output.get("artifacts", [])
                            if artifacts:
                                details["artifacts"] = [
                                    {"type": a.get("type"), "name": a.get("name")}
                                    for a in artifacts[:5]
                                ]

                        elif name == "merge_results":
                            final_result = output
                            final = output.get("final_response", {})
                            details["final"] = {
                                "confidence": final.get("confidence", 0),
                                "has_data": bool(final.get("data")),
                                "suggestions_count": len(final.get("suggestions", [])),
                            }

                    yield _sse(details)

                elif etype == "on_llm_start":
                    yield _sse({
                        "event": "llm_call",
                        "step": name,
                        "status": "calling_llm",
                        "elapsed_ms": elapsed_ms,
                    })

            timing_ms = round((time.monotonic() - start) * 1000, 1)

            if final_result:
                resp = final_result.get("final_response", {})

                # Check for HITL waiting
                is_waiting = resp.get("status") == "waiting_human"
                run_status = "waiting_human" if is_waiting else "completed"

                yield _sse({
                    "event": "run_completed",
                    "status": run_status,
                    "answer": resp.get("answer", ""),
                    "confidence": resp.get("confidence", 0),
                    "suggestions": resp.get("suggestions", []),
                    "timing_ms": timing_ms,
                    "interrupt": resp.get("interrupt") if is_waiting else None,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

                await sm.complete_run(
                    run_id=run_id, status=run_status,
                    result=resp, timing_ms=timing_ms,
                )
            else:
                yield _sse({
                    "event": "run_completed",
                    "status": "completed",
                    "timing_ms": timing_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        except Exception as e:
            logger.error("Stream error: %s", e, exc_info=True)
            yield _sse({
                "event": "run_error",
                "status": "error",
                "error": str(e),
                "elapsed_ms": round((time.monotonic() - start) * 1000, 1),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
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


# ── Human-in-the-Loop Endpoints ──────────────────────────────────

from models import InterruptInfo, InterruptResponse, InterruptStatus  # noqa: E402


@app.get("/interrupts/run/{run_id}")
async def get_interrupts_for_run(
    run_id: str,
    request: Request,
    _: Any = Depends(verify_api_key),
):
    """List all pending interrupts for a run."""
    hitl_svc = request.app.state.services.get("hitl")
    if not hitl_svc:
        raise HTTPException(status_code=501, detail="HITL not enabled")

    pending = await hitl_svc.get_pending_for_run(run_id)
    return {"run_id": run_id, "pending_count": len(pending), "interrupts": pending}


@app.get("/interrupts/session/{session_id}")
async def get_interrupts_for_session(
    session_id: str,
    request: Request,
    _: Any = Depends(verify_api_key),
):
    """List all pending interrupts for a session."""
    hitl_svc = request.app.state.services.get("hitl")
    if not hitl_svc:
        raise HTTPException(status_code=501, detail="HITL not enabled")

    pending = await hitl_svc.get_pending_for_session(session_id)
    return {"session_id": session_id, "pending_count": len(pending), "interrupts": pending}


@app.get("/interrupts/{interrupt_id}")
async def get_interrupt_details(
    interrupt_id: str,
    request: Request,
    _: Any = Depends(verify_api_key),
):
    """Get details of a specific interrupt."""
    hitl_svc = request.app.state.services.get("hitl")
    if not hitl_svc:
        raise HTTPException(status_code=501, detail="HITL not enabled")

    info = await hitl_svc.get_interrupt(interrupt_id)
    if not info:
        raise HTTPException(status_code=404, detail="Interrupt not found")
    return info


@app.post("/interrupts/{interrupt_id}/resolve")
async def resolve_interrupt(
    interrupt_id: str,
    body: dict,
    request: Request,
    _: Any = Depends(verify_api_key),
):
    """
    Resolve a pending interrupt with the user's decision.

    Body examples:

    Approve:
      {"action": "approved"}

    Reject:
      {"action": "rejected", "comment": "Query looks wrong"}

    Modify SQL:
      {"action": "modified", "modifications": {"sql": "SELECT ... LIMIT 10"}}

    Clarify:
      {"action": "clarified", "modifications": {"answer": "Only desk Alpha"}}
    """
    hitl_svc = request.app.state.services.get("hitl")
    if not hitl_svc:
        raise HTTPException(status_code=501, detail="HITL not enabled")

    try:
        response = InterruptResponse(
            interrupt_id=interrupt_id,
            action=InterruptStatus(body.get("action", "approved")),
            modifications=body.get("modifications", {}),
            comment=body.get("comment", ""),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action. Valid: approved, rejected, modified, clarified. Error: {e}",
        )

    result = await hitl_svc.resolve_interrupt(response)
    if not result:
        raise HTTPException(status_code=404, detail="Interrupt not found")

    return {
        "status": "resolved",
        "interrupt": result,
        "next_steps": _hitl_next_steps(result),
    }


def _hitl_next_steps(info: InterruptInfo) -> str:
    """Provide guidance on what happens next after resolution."""
    if info.status == InterruptStatus.APPROVED:
        return "The query will now execute. Poll GET /execute/{run_id} for results."
    elif info.status == InterruptStatus.REJECTED:
        return "The run has been cancelled. No query was executed."
    elif info.status == InterruptStatus.MODIFIED:
        return "The modified query will execute. Poll GET /execute/{run_id} for results."
    elif info.status == InterruptStatus.CLARIFIED:
        return "The agent will proceed with your clarification."
    return "Resolution recorded."


# ── Helpers ──────────────────────────────────────────────────────

def _sse(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data, default=str)}\n\n"
