"""
agents/master/graph.py
----------------------
Defines and compiles the Master LangGraph.

Topology:
         ┌─────────────┐
         │  START      │
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │ analyze_    │
         │  intent     │
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │ select_     │
         │  agent      │
         └──────┬──────┘
                │
        ┌───────┴────────┐
        │ conditional    │  ← LLM decided: route or direct?
        └───┬────────┬───┘
            │        │
    ┌───────▼──┐  ┌──▼──────────────┐
    │  direct  │  │ execute_agent   │
    │  answer  │  └──────┬──────────┘
    └───┬──────┘         │
        │         ┌──────▼──────┐
        │         │ synthesize  │
        │         │  response   │
        │         └──────┬──────┘
        │                │
        └────────┬────────┘
                 │
          ┌──────▼──────┐
          │    END      │
          └─────────────┘
"""

from langgraph.graph import StateGraph, END
from app.agents.master.state import MasterState
from app.agents.master.nodes import (
    analyze_intent,
    select_agent,
    direct_answer,
    execute_agent,
    synthesize_response,
)


def _routing_decision(state: MasterState) -> str:
    """
    Conditional edge function:
    Inspect the state after agent selection and decide which branch to take.
    Returns the name of the NEXT node.
    """
    if state.get("selected_agent"):
        return "execute_agent"
    return "direct_answer"


def build_master_graph() -> StateGraph:
    """
    Construct and compile the Master LangGraph.
    Call once at startup; the compiled graph is thread-safe.
    """
    builder = StateGraph(MasterState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("analyze_intent", analyze_intent)
    builder.add_node("select_agent", select_agent)
    builder.add_node("direct_answer", direct_answer)
    builder.add_node("execute_agent", execute_agent)
    builder.add_node("synthesize_response", synthesize_response)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("analyze_intent")

    # ── Linear edges ─────────────────────────────────────────────────────────
    builder.add_edge("analyze_intent", "select_agent")

    # ── Conditional branch (LLM routing decision) ─────────────────────────────
    builder.add_conditional_edges(
        "select_agent",
        _routing_decision,
        {
            "direct_answer": "direct_answer",
            "execute_agent": "execute_agent",
        },
    )

    # ── Converge paths ────────────────────────────────────────────────────────
    builder.add_edge("direct_answer", END)
    builder.add_edge("execute_agent", "synthesize_response")
    builder.add_edge("synthesize_response", END)

    return builder.compile()


# Module-level compiled graph – import this in the FastAPI layer
master_graph = build_master_graph()
