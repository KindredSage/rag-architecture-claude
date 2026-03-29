"""
Trade Agent: LangGraph sub-graph with schema-first architecture and HITL failure recovery.

Flow:
  START -> schema_analyzer -> trade_analyst -> clarification_gate -> query_analyst
        -> query_planner -> query_builder -> query_validator
        --(valid)--> sql_approval_gate --(approved)--> query_executor
        -> details_analyzer -> END

Retry: query_validator/query_executor can loop back to query_builder (max 3).
       When retries exhaust, failure_feedback_gate asks user for guidance.
HITL:  clarification_gate (if ambiguous), sql_approval_gate (before execution),
       failure_feedback_gate (when retries exhaust).
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
    # Retries exhausted and still invalid -> ask user for feedback
    logger.warning("Validation failed after max retries, routing to failure feedback")
    return "failure_feedback_gate"


def _should_retry_after_execution(state: TradeAgentState) -> str:
    if state.get("needs_retry") and state.get("retry_count", 0) <= 3:
        return "query_builder"
    if state.get("query_results", {}).get("success", False):
        return "details_analyzer"
    # Execution failed after retries -> ask user for feedback
    logger.warning("Execution failed after retries, routing to failure feedback")
    return "failure_feedback_gate"


def _should_proceed_after_hitl(state: TradeAgentState) -> str:
    action = state.get("hitl_response", {}).get("action", "")
    if action in ("rejected", "expired"):
        return "details_analyzer"
    if state.get("hitl_pending", {}).get("status") == "pending":
        return "__end__"
    return "query_executor"


def _after_failure_feedback(state: TradeAgentState) -> str:
    """Route after failure_feedback_gate: clarified -> re-assess, else -> error exit."""
    action = state.get("hitl_response", {}).get("action", "")

    if action == "clarified":
        # Guard against infinite loop: if we already clarified once before,
        # go to details_analyzer instead of looping back
        trace = state.get("execution_trace", [])
        clarification_count = sum(
            1 for t in trace
            if t.get("node") == "failure_feedback_gate" and t.get("status") == "clarified"
        )
        if clarification_count > 1:
            logger.warning("Multiple failure clarifications detected, stopping loop")
            return "details_analyzer"
        return "query_analyst"

    # rejected, expired, or hitl_skipped -> error exit
    return "details_analyzer"


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

    # -- Node bindings --

    async def _schema_analyzer(s):
        return await schema_analyzer(s, ch_service=ch, cache_service=cache)

    async def _trade_analyst(s):
        return await trade_analyst(s, llm=llm_fast)

    async def _query_analyst(s):
        return await query_analyst(s, llm=llm_fast)

    async def _query_planner(s):
        return await query_planner(s, llm=llm_fast)

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

    async def _failure_feedback_gate(s):
        if not hitl_svc:
            return {
                "hitl_skipped": True,
                "execution_trace": [{
                    "node": "failure_feedback_gate", "status": "skipped",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "output_summary": "HITL unavailable; cannot ask user for feedback",
                }],
            }
        from agents.hitl_gates import failure_feedback_gate
        return await failure_feedback_gate(
            s, hitl_service=hitl_svc,
            run_id=s.get("context_overrides", {}).get("run_id", ""),
            session_id=s.get("context_overrides", {}).get("session_id", ""),
        )

    # -- Graph --

    graph = StateGraph(TradeAgentState)

    graph.add_node("schema_analyzer", _schema_analyzer)
    graph.add_node("trade_analyst", _trade_analyst)
    graph.add_node("clarification_gate", _clarification_gate)
    graph.add_node("query_analyst", _query_analyst)
    graph.add_node("query_planner", _query_planner)
    graph.add_node("query_builder", _query_builder)
    graph.add_node("query_validator", _query_validator)
    graph.add_node("sql_approval_gate", _sql_approval_gate)
    graph.add_node("query_executor", _query_executor)
    graph.add_node("details_analyzer", _details_analyzer)
    graph.add_node("failure_feedback_gate", _failure_feedback_gate)

    # Schema-first: fetch real schema before any LLM reasoning
    graph.add_edge(START, "schema_analyzer")
    graph.add_edge("schema_analyzer", "trade_analyst")
    graph.add_edge("trade_analyst", "clarification_gate")
    graph.add_edge("clarification_gate", "query_analyst")
    graph.add_edge("query_analyst", "query_planner")
    graph.add_edge("query_planner", "query_builder")
    graph.add_edge("query_builder", "query_validator")

    # Validator -> retry / approve / failure feedback
    graph.add_conditional_edges("query_validator", _should_retry_or_execute,
                                {"query_builder": "query_builder",
                                 "sql_approval_gate": "sql_approval_gate",
                                 "failure_feedback_gate": "failure_feedback_gate"})

    # HITL approval -> execute / reject / stream-pause
    graph.add_conditional_edges("sql_approval_gate", _should_proceed_after_hitl,
                                {"query_executor": "query_executor",
                                 "details_analyzer": "details_analyzer",
                                 "__end__": END})

    # Executor -> retry / analyze / failure feedback
    graph.add_conditional_edges("query_executor", _should_retry_after_execution,
                                {"query_builder": "query_builder",
                                 "details_analyzer": "details_analyzer",
                                 "failure_feedback_gate": "failure_feedback_gate"})

    # Failure feedback -> re-assess from query_analyst or error exit
    graph.add_conditional_edges("failure_feedback_gate", _after_failure_feedback,
                                {"query_analyst": "query_analyst",
                                 "details_analyzer": "details_analyzer"})

    graph.add_edge("details_analyzer", END)

    compiled = graph.compile()
    logger.info("Trade Agent graph compiled (schema-first)")
    return compiled
