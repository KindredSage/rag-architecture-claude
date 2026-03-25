"""
agents/trade/agent.py
---------------------
The TradeAgent class wraps the compiled trade_graph and implements
the BaseAgent interface expected by the AgentRegistry.

This is the only class the registry and Master Agent interact with —
they never import nodes or graphs directly.
"""

from __future__ import annotations
from typing import Any

from app.agents.base import BaseAgent
from app.agents.trade.graph import trade_graph
from app.agents.trade.state import TradeState
from app.core.logging import get_logger

logger = get_logger(__name__)


class TradeAgent(BaseAgent):
    """
    Domain expert for trade data queries.
    Runs a full 8-node pipeline: analyse → plan → schema → build →
    validate → execute → analyse → synthesise.
    """

    name = "trade_agent"
    description = (
        "Handles all trade data queries: retrieving trade details, "
        "filtering by file ID / trade ID / date, aggregating notionals, "
        "and analyzing trade portfolios from the trade data warehouse."
    )
    capabilities = [
        "retrieve trade details by file_id",
        "retrieve trade details by trade_id",
        "filter trades by date range",
        "filter trades by counterparty",
        "filter trades by instrument",
        "aggregate trade notionals",
        "query trade status",
        "join trade tables",
        "analyze trade portfolios",
    ]

    async def execute(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Invoke the trade LangGraph and return a normalised result dict.
        """
        logger.info("trade_agent_start", query=query[:80])

        # Build initial state
        initial_state: TradeState = {
            "query": query,
            "context": context,
            "query_analysis": {},
            "table_name": None,
            "file_id": None,
            "filters": {},
            "execution_plan": [],
            "schema_info": {},
            "raw_query": "",
            "validated_query": "",
            "validation_errors": [],
            "query_result": {},
            "row_count": 0,
            "result_analysis": "",
            "answer": "",
            "confidence": 0.0,
            "steps": [],
            "error": None,
        }

        # ── Run the graph ──────────────────────────────────────────────────────
        final_state: TradeState = await trade_graph.ainvoke(initial_state)

        # ── Normalise output to BaseAgent contract ─────────────────────────────
        return {
            "answer": final_state.get("answer", "No answer produced."),
            "steps": final_state.get("steps", []),
            "confidence": final_state.get("confidence", 0.5),
            "metadata": {
                "table": final_state.get("table_name"),
                "file_id": final_state.get("file_id"),
                "row_count": final_state.get("row_count", 0),
                "validation_errors": final_state.get("validation_errors", []),
                "execution_plan": final_state.get("execution_plan", []),
            },
        }
