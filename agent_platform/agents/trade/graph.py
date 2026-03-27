"""
Trade Agent: LangGraph sub-graph with 8 specialized nodes and a retry loop.

Flow:
  START -> trade_analyst -> query_analyst -> query_planner -> schema_analyzer
        -> query_builder -> query_validator --(conditional)--> query_executor
                                  ^  (retry loop)   |
                                  +------ if needs_retry & retry_count < 3

        query_executor -> details_analyzer -> END

The retry loop between query_validator and query_builder handles:
  - SQL syntax errors detected by ClickHouse
  - Security violations caught by the programmatic checker
  - Correctness issues (wrong column names, type mismatches)
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.trade.nodes import (
    details_analyzer,
    query_analyst,
    query_builder,
    query_executor,
    query_planner,
    query_validator,
    schema_analyzer,
    trade_analyst,
)
from agents.trade.state import TradeAgentState

logger = logging.getLogger(__name__)


def _should_retry_or_execute(state: TradeAgentState) -> str:
    """Conditional edge after query_validator."""
    if state.get("needs_retry") and state.get("retry_count", 0) <= 3:
        logger.info(
            "Retrying query_builder (attempt %d)", state.get("retry_count", 0)
        )
        return "query_builder"

    validation = state.get("validation_result", {})
    if validation.get("is_valid", False):
        return "query_executor"

    # Validation failed and no retries left -- skip to analyzer with error
    logger.warning("Validation failed after max retries, proceeding to analyzer")
    return "details_analyzer"


def _should_retry_after_execution(state: TradeAgentState) -> str:
    """Conditional edge after query_executor for execution failures."""
    if state.get("needs_retry") and state.get("retry_count", 0) <= 3:
        logger.info(
            "Retrying query_builder after execution failure (attempt %d)",
            state.get("retry_count", 0),
        )
        return "query_builder"
    return "details_analyzer"


def build_trade_agent_graph(*, settings, services) -> Any:
    """
    Build and compile the Trade Agent StateGraph.

    Dependencies (settings, services) are captured in closures so each
    node receives the right injected service.

    HITL gates:
      - clarification_gate:  after classify_intent (if ambiguous)
      - sql_approval_gate:   after query_validator (before execution)
    """
    from services.llm_service import LLMService
    from tools.plotting_tools import create_plotting_tools
    from tools.export_tools import create_export_tools
    from tools.email_tools import create_email_tools
    from agents.node_runner import timed_node

    llm_service: LLMService = services["llm"]
    llm_primary = llm_service.get_model(fast=False)
    llm_fast = llm_service.get_model(fast=True)
    ch = services["clickhouse"]
    cache = services.get("cache")
    hitl_svc = services.get("hitl")  # may be None if not initialized

    # Build tool instances
    plot_tools = create_plotting_tools(settings.artifact_dir)
    exp_tools = create_export_tools(settings.artifact_dir)
    mail_tools = create_email_tools(settings)

    # ── Raw node functions ────────────────────────────────────────

    async def _trade_analyst_raw(state: TradeAgentState) -> dict:
        return await trade_analyst(state, llm=llm_fast)

    async def _query_analyst_raw(state: TradeAgentState) -> dict:
        return await query_analyst(state, llm=llm_fast)

    async def _query_planner_raw(state: TradeAgentState) -> dict:
        return await query_planner(state, llm=llm_fast)

    async def _schema_analyzer_raw(state: TradeAgentState) -> dict:
        return await schema_analyzer(state, ch_service=ch, cache_service=cache)

    async def _query_builder_raw(state: TradeAgentState) -> dict:
        return await query_builder(state, llm=llm_primary, settings=settings)

    async def _query_validator_raw(state: TradeAgentState) -> dict:
        return await query_validator(state, llm=llm_fast, settings=settings)

    async def _query_executor_raw(state: TradeAgentState) -> dict:
        return await query_executor(state, ch_service=ch, cache_service=cache)

    async def _details_analyzer_raw(state: TradeAgentState) -> dict:
        return await details_analyzer(
            state,
            llm=llm_primary,
            plotting_tools=plot_tools,
            export_tools=exp_tools,
            email_tools=mail_tools,
        )

    # ── Wrap all nodes with timed_node for consistent timing ─────
    _trade_analyst_n = timed_node("trade_analyst", "trade_agent", _trade_analyst_raw)
    _query_analyst_n = timed_node("query_analyst", "trade_agent", _query_analyst_raw)
    _query_planner_n = timed_node("query_planner", "trade_agent", _query_planner_raw)
    _schema_analyzer_n = timed_node("schema_analyzer", "trade_agent", _schema_analyzer_raw)
    _query_builder_n = timed_node("query_builder", "trade_agent", _query_builder_raw)
    _query_validator_n = timed_node("query_validator", "trade_agent", _query_validator_raw)
    _query_executor_n = timed_node("query_executor", "trade_agent", _query_executor_raw)
    _details_analyzer_n = timed_node("details_analyzer", "trade_agent", _details_analyzer_raw)

    # ── HITL Gate Nodes ──────────────────────────────────────────

    async def _clarification_gate(state: TradeAgentState) -> dict:
        """Check if intent is ambiguous and ask user to clarify."""
        from agents.hitl_gates import clarification_gate

        intent = state.get("intent_analysis", {})
        ambiguity = intent.get("ambiguity_notes", "")

        if not ambiguity or not hitl_svc:
            return {
                "hitl_skipped": True,
                "execution_trace": [{
                    "node": "clarification_gate",
                    "status": "skipped",
                    "timestamp": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                    "output_summary": "No ambiguity detected or HITL unavailable",
                }],
            }

        return await clarification_gate(
            state,
            hitl_service=hitl_svc,
            run_id=state.get("context_overrides", {}).get("run_id", ""),
            session_id=state.get("context_overrides", {}).get("session_id", ""),
            agent_id="trade_agent",
            question=ambiguity,
        )

    async def _sql_approval_gate(state: TradeAgentState) -> dict:
        """HITL gate: pause for SQL approval before execution."""
        from agents.hitl_gates import sql_approval_gate

        if not hitl_svc:
            return {
                "hitl_skipped": True,
                "execution_trace": [{
                    "node": "sql_approval_gate",
                    "status": "skipped",
                    "timestamp": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                    "output_summary": "HITL service not available",
                }],
            }

        return await sql_approval_gate(
            state,
            hitl_service=hitl_svc,
            run_id=state.get("context_overrides", {}).get("run_id", ""),
            session_id=state.get("context_overrides", {}).get("session_id", ""),
            agent_id="trade_agent",
        )

    def _should_proceed_after_hitl(state: TradeAgentState) -> str:
        """After HITL gate: proceed to executor or stop."""
        hitl_resp = state.get("hitl_response", {})
        action = hitl_resp.get("action", "")

        if action in ("rejected", "expired"):
            return "details_analyzer"  # skip execution, go to analysis

        pending = state.get("hitl_pending", {})
        if pending.get("status") == "pending":
            return "__end__"  # graph pauses, will resume later

        # Approved, modified, or HITL skipped
        return "query_executor"

    # ── Graph Assembly ───────────────────────────────────────────

    graph = StateGraph(TradeAgentState)

    graph.add_node("trade_analyst", _trade_analyst_n)
    graph.add_node("clarification_gate", _clarification_gate)
    graph.add_node("query_analyst", _query_analyst_n)
    graph.add_node("query_planner", _query_planner_n)
    graph.add_node("schema_analyzer", _schema_analyzer_n)
    graph.add_node("query_builder", _query_builder_n)
    graph.add_node("query_validator", _query_validator_n)
    graph.add_node("sql_approval_gate", _sql_approval_gate)
    graph.add_node("query_executor", _query_executor_n)
    graph.add_node("details_analyzer", _details_analyzer_n)

    # Flow with HITL gates inserted
    graph.add_edge(START, "trade_analyst")
    graph.add_edge("trade_analyst", "clarification_gate")
    graph.add_edge("clarification_gate", "query_analyst")
    graph.add_edge("query_analyst", "query_planner")
    graph.add_edge("query_planner", "schema_analyzer")
    graph.add_edge("schema_analyzer", "query_builder")
    graph.add_edge("query_builder", "query_validator")

    # Conditional: retry loop or proceed to HITL gate
    graph.add_conditional_edges(
        "query_validator",
        _should_retry_or_execute,
        {
            "query_builder": "query_builder",
            "query_executor": "sql_approval_gate",  # HITL gate before execution
            "details_analyzer": "details_analyzer",
        },
    )

    # After HITL gate: proceed, pause, or stop
    graph.add_conditional_edges(
        "sql_approval_gate",
        _should_proceed_after_hitl,
        {
            "query_executor": "query_executor",
            "details_analyzer": "details_analyzer",
            "__end__": END,  # paused for human input
        },
    )

    # After execution: retry on failure or proceed to analysis
    graph.add_conditional_edges(
        "query_executor",
        _should_retry_after_execution,
        {
            "query_builder": "query_builder",
            "details_analyzer": "details_analyzer",
        },
    )

    graph.add_edge("details_analyzer", END)

    compiled = graph.compile()
    logger.info("Trade Agent graph compiled (with HITL gates)")
    return compiled
