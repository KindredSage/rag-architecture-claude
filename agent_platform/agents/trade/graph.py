"""
Trade Agent: LangGraph sub-graph with 8 nodes, retry loop, and HITL gates.

Flow:
  START -> trade_analyst -> clarification_gate -> query_analyst -> query_planner
        -> schema_analyzer -> query_builder -> query_validator
        --(valid)--> sql_approval_gate --(approved)--> query_executor
        -> details_analyzer -> END

Retry: query_validator/query_executor can loop back to query_builder (max 3).
HITL:  clarification_gate (if ambiguous), sql_approval_gate (before execution).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
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
    if state.get("needs_retry") and state.get("retry_count", 0) <= 3:
        return "query_builder"
    if state.get("validation_result", {}).get("is_valid", False):
        return "sql_approval_gate"
    logger.warning("Validation failed after max retries")
    return "details_analyzer"


def _should_retry_after_execution(state: TradeAgentState) -> str:
    if state.get("needs_retry") and state.get("retry_count", 0) <= 3:
        return "query_builder"
    return "details_analyzer"


def _should_proceed_after_hitl(state: TradeAgentState) -> str:
    action = state.get("hitl_response", {}).get("action", "")
    if action in ("rejected", "expired"):
        return "details_analyzer"
    if state.get("hitl_pending", {}).get("status") == "pending":
        return "__end__"
    return "query_executor"


def build_trade_agent_graph(*, settings, services) -> Any:
    from services.llm_service import LLMService
    from tools.plotting_tools import create_plotting_tools
    from tools.export_tools import create_export_tools
    from tools.email_tools import create_email_tools

    llm_svc: LLMService = services["llm"]
    llm_primary = llm_svc.get_model(fast=False)
    llm_fast = llm_svc.get_model(fast=True)
    ch = services["clickhouse"]
    cache = services.get("cache")
    hitl_svc = services.get("hitl")
    plot_tools = create_plotting_tools(settings.artifact_dir)
    exp_tools = create_export_tools(settings.artifact_dir)
    mail_tools = create_email_tools(settings)

    # -- Node bindings (each node handles its own timing internally) --

    async def _trade_analyst(s):
        return await trade_analyst(s, llm=llm_fast)

    async def _query_analyst(s):
        return await query_analyst(s, llm=llm_fast)

    async def _query_planner(s):
        return await query_planner(s, llm=llm_fast)

    async def _schema_analyzer(s):
        return await schema_analyzer(s, ch_service=ch, cache_service=cache)

    async def _query_builder(s):
        return await query_builder(s, llm=llm_primary, settings=settings)

    async def _query_validator(s):
        return await query_validator(s, llm=llm_fast, settings=settings)

    async def _query_executor(s):
        return await query_executor(s, ch_service=ch, cache_service=cache)

    async def _details_analyzer(s):
        return await details_analyzer(
            s, llm=llm_primary,
            plotting_tools=plot_tools, export_tools=exp_tools, email_tools=mail_tools,
        )

    async def _clarification_gate(s):
        intent = s.get("intent_analysis", {})
        ambiguity = intent.get("ambiguity_notes", "")
        if not ambiguity or not hitl_svc:
            return {
                "hitl_skipped": True,
                "execution_trace": [{
                    "node": "clarification_gate", "status": "skipped",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "output_summary": "No ambiguity or HITL unavailable",
                }],
            }
        from agents.hitl_gates import clarification_gate
        return await clarification_gate(
            s, hitl_service=hitl_svc,
            run_id=s.get("context_overrides", {}).get("run_id", ""),
            session_id=s.get("context_overrides", {}).get("session_id", ""),
        )

    async def _sql_approval_gate(s):
        if not hitl_svc:
            return {
                "hitl_skipped": True,
                "execution_trace": [{
                    "node": "sql_approval_gate", "status": "skipped",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "output_summary": "HITL unavailable",
                }],
            }
        from agents.hitl_gates import sql_approval_gate
        return await sql_approval_gate(
            s, hitl_service=hitl_svc,
            run_id=s.get("context_overrides", {}).get("run_id", ""),
            session_id=s.get("context_overrides", {}).get("session_id", ""),
        )

    # -- Graph --

    graph = StateGraph(TradeAgentState)
    graph.add_node("trade_analyst", _trade_analyst)
    graph.add_node("clarification_gate", _clarification_gate)
    graph.add_node("query_analyst", _query_analyst)
    graph.add_node("query_planner", _query_planner)
    graph.add_node("schema_analyzer", _schema_analyzer)
    graph.add_node("query_builder", _query_builder)
    graph.add_node("query_validator", _query_validator)
    graph.add_node("sql_approval_gate", _sql_approval_gate)
    graph.add_node("query_executor", _query_executor)
    graph.add_node("details_analyzer", _details_analyzer)

    graph.add_edge(START, "trade_analyst")
    graph.add_edge("trade_analyst", "clarification_gate")
    graph.add_edge("clarification_gate", "query_analyst")
    graph.add_edge("query_analyst", "query_planner")
    graph.add_edge("query_planner", "schema_analyzer")
    graph.add_edge("schema_analyzer", "query_builder")
    graph.add_edge("query_builder", "query_validator")
    graph.add_conditional_edges("query_validator", _should_retry_or_execute,
                                {"query_builder": "query_builder",
                                 "sql_approval_gate": "sql_approval_gate",
                                 "details_analyzer": "details_analyzer"})
    graph.add_conditional_edges("sql_approval_gate", _should_proceed_after_hitl,
                                {"query_executor": "query_executor",
                                 "details_analyzer": "details_analyzer",
                                 "__end__": END})
    graph.add_conditional_edges("query_executor", _should_retry_after_execution,
                                {"query_builder": "query_builder",
                                 "details_analyzer": "details_analyzer"})
    graph.add_edge("details_analyzer", END)

    compiled = graph.compile()
    logger.info("Trade Agent graph compiled")
    return compiled
