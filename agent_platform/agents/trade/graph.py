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
    """
    from services.llm_service import LLMService
    from tools.plotting_tools import create_plotting_tools
    from tools.export_tools import create_export_tools
    from tools.email_tools import create_email_tools

    llm_service: LLMService = services["llm"]
    llm_primary = llm_service.get_model(fast=False)
    llm_fast = llm_service.get_model(fast=True)
    ch = services["clickhouse"]
    cache = services.get("cache")

    # Build tool instances
    plot_tools = create_plotting_tools(settings.artifact_dir)
    exp_tools = create_export_tools(settings.artifact_dir)
    mail_tools = create_email_tools(settings)

    # ── Bound node functions ─────────────────────────────────────

    async def _trade_analyst(state: TradeAgentState) -> dict:
        return await trade_analyst(state, llm=llm_fast)

    async def _query_analyst(state: TradeAgentState) -> dict:
        return await query_analyst(state, llm=llm_fast)

    async def _query_planner(state: TradeAgentState) -> dict:
        return await query_planner(state, llm=llm_fast)

    async def _schema_analyzer(state: TradeAgentState) -> dict:
        return await schema_analyzer(state, ch_service=ch, cache_service=cache)

    async def _query_builder(state: TradeAgentState) -> dict:
        return await query_builder(state, llm=llm_primary, settings=settings)

    async def _query_validator(state: TradeAgentState) -> dict:
        return await query_validator(state, llm=llm_fast, settings=settings)

    async def _query_executor(state: TradeAgentState) -> dict:
        return await query_executor(state, ch_service=ch, cache_service=cache)

    async def _details_analyzer(state: TradeAgentState) -> dict:
        return await details_analyzer(
            state,
            llm=llm_primary,
            plotting_tools=plot_tools,
            export_tools=exp_tools,
            email_tools=mail_tools,
        )

    # ── Graph Assembly ───────────────────────────────────────────

    graph = StateGraph(TradeAgentState)

    graph.add_node("trade_analyst", _trade_analyst)
    graph.add_node("query_analyst", _query_analyst)
    graph.add_node("query_planner", _query_planner)
    graph.add_node("schema_analyzer", _schema_analyzer)
    graph.add_node("query_builder", _query_builder)
    graph.add_node("query_validator", _query_validator)
    graph.add_node("query_executor", _query_executor)
    graph.add_node("details_analyzer", _details_analyzer)

    # Linear flow
    graph.add_edge(START, "trade_analyst")
    graph.add_edge("trade_analyst", "query_analyst")
    graph.add_edge("query_analyst", "query_planner")
    graph.add_edge("query_planner", "schema_analyzer")
    graph.add_edge("schema_analyzer", "query_builder")
    graph.add_edge("query_builder", "query_validator")

    # Conditional: retry loop or proceed
    graph.add_conditional_edges(
        "query_validator",
        _should_retry_or_execute,
        {
            "query_builder": "query_builder",
            "query_executor": "query_executor",
            "details_analyzer": "details_analyzer",
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
    logger.info("Trade Agent graph compiled successfully")
    return compiled
