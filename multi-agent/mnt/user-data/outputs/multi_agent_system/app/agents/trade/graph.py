"""
agents/trade/graph.py
---------------------
Defines and compiles the Trade Agent LangGraph.

Full pipeline topology:
  START
    → query_analyst          (parse query, extract entities)
    → query_planner          (plan execution steps)
    → schema_analyzer        (get table schema)
    → query_builder          (build SQL)
    → query_validator        (validate SQL)
    ↓
  [conditional: validation passed?]
    → YES → query_executor   (run query)
           → details_analyzer (analyze results)
           → result_synthesizer (produce answer)
           → END
    → NO  → result_synthesizer (produce error answer)
           → END
"""

from langgraph.graph import StateGraph, END
from app.agents.trade.state import TradeState
from app.agents.trade.nodes import (
    query_analyst,
    query_planner,
    schema_analyzer,
    query_builder,
    query_validator,
    query_executor,
    details_analyzer,
    result_synthesizer,
)


def _validation_router(state: TradeState) -> str:
    """Route based on whether the query passed validation."""
    errors = state.get("validation_errors", [])
    # Hard block if query was outright rejected
    if any("rejected" in e.lower() for e in errors):
        return "result_synthesizer"   # skip execution, go straight to answer
    return "query_executor"


def build_trade_graph() -> StateGraph:
    """Build and compile the Trade Agent sub-graph."""
    builder = StateGraph(TradeState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    builder.add_node("query_analyst", query_analyst)
    builder.add_node("query_planner", query_planner)
    builder.add_node("schema_analyzer", schema_analyzer)
    builder.add_node("query_builder", query_builder)
    builder.add_node("query_validator", query_validator)
    builder.add_node("query_executor", query_executor)
    builder.add_node("details_analyzer", details_analyzer)
    builder.add_node("result_synthesizer", result_synthesizer)

    # ── Entry ─────────────────────────────────────────────────────────────────
    builder.set_entry_point("query_analyst")

    # ── Linear edges (main pipeline) ──────────────────────────────────────────
    builder.add_edge("query_analyst", "query_planner")
    builder.add_edge("query_planner", "schema_analyzer")
    builder.add_edge("schema_analyzer", "query_builder")
    builder.add_edge("query_builder", "query_validator")

    # ── Conditional: validated? ────────────────────────────────────────────────
    builder.add_conditional_edges(
        "query_validator",
        _validation_router,
        {
            "query_executor": "query_executor",
            "result_synthesizer": "result_synthesizer",
        },
    )

    # ── Post-execution pipeline ────────────────────────────────────────────────
    builder.add_edge("query_executor", "details_analyzer")
    builder.add_edge("details_analyzer", "result_synthesizer")

    # ── Terminal ──────────────────────────────────────────────────────────────
    builder.add_edge("result_synthesizer", END)

    return builder.compile()


# Module-level compiled sub-graph
trade_graph = build_trade_graph()
