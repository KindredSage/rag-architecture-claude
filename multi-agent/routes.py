"""
api/routes/agents.py  – GET /agents
api/routes/execute.py – POST /execute
(combined in one file for clarity, split in practice)
"""

# ══════════════════════════════════════════════════════════════════════════════
# api/routes/agents.py
# ══════════════════════════════════════════════════════════════════════════════
from fastapi import APIRouter, HTTPException, status
from app.api.models.schemas import AgentInfo, ExecuteRequest, ExecuteResponse, TraceStep
from app.registry.agent_registry import registry
from app.agents.master.graph import master_graph
from app.agents.master.state import MasterState
from app.core.logging import get_logger

logger = get_logger(__name__)

agents_router = APIRouter(prefix="/agents", tags=["Agents"])
execute_router = APIRouter(prefix="/execute", tags=["Execute"])


# ─────────────────────────────────────────────────────────────────────────────
# GET /agents – list all registered agents
# ─────────────────────────────────────────────────────────────────────────────

@agents_router.get(
    "",
    response_model=list[AgentInfo],
    summary="List all registered agents",
    description="Returns metadata for every agent currently registered in the system.",
)
async def list_agents() -> list[AgentInfo]:
    """Return all registered agents with their descriptions and capabilities."""
    agents = registry.list_agents()
    logger.info("list_agents_called", count=len(agents))
    return [
        AgentInfo(
            name=a.name,
            description=a.description,
            capabilities=a.capabilities,
        )
        for a in agents
    ]


# ─────────────────────────────────────────────────────────────────────────────
# POST /execute – run the Master Agent orchestrator
# ─────────────────────────────────────────────────────────────────────────────

@execute_router.post(
    "",
    response_model=ExecuteResponse,
    summary="Execute a query via the Master Agent",
    description=(
        "Passes the query to the Master LangGraph, which:\n"
        "1. Analyses intent\n"
        "2. Selects the best agent (or answers directly)\n"
        "3. Runs the agent's full pipeline\n"
        "4. Returns a structured response with reasoning trace"
    ),
)
async def execute_query(request: ExecuteRequest) -> ExecuteResponse:
    """Main orchestration endpoint."""
    logger.info("execute_called", query=request.query[:80])

    # ── Build initial master state ─────────────────────────────────────────────
    initial_state: MasterState = {
        "query": request.query,
        "context": request.context,
        "intent": "",
        "selected_agent": None,
        "routing_reason": "",
        "plan": [],
        "intermediate_results": {},
        "agent_raw_output": {},
        "final_answer": "",
        "confidence": 0.0,
        "trace": [],
        "error": None,
    }

    # ── Run Master LangGraph ───────────────────────────────────────────────────
    try:
        final_state: MasterState = await master_graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("master_graph_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Orchestration failed: {exc}",
        )

    # ── Build response ─────────────────────────────────────────────────────────
    trace_steps = [
        TraceStep(
            node=step.get("node", "unknown"),
            action=step.get("action", ""),
            output=step.get("output"),
        )
        for step in final_state.get("trace", [])
    ]

    agent_used = final_state.get("selected_agent") or "master_direct"

    logger.info(
        "execute_complete",
        agent_used=agent_used,
        confidence=final_state.get("confidence", 0.0),
    )

    return ExecuteResponse(
        answer=final_state.get("final_answer", ""),
        agent_used=agent_used,
        steps=trace_steps,
        confidence=final_state.get("confidence", 0.0),
    )
