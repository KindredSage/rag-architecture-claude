"""
agents/master/state.py
----------------------
Shared state object that flows through the Master LangGraph.
All nodes READ from and WRITE to this TypedDict.

LangGraph uses Annotated fields with reducer functions (e.g. operator.add)
for list fields that need to accumulate values across nodes.
"""

from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict, Annotated
import operator


class MasterState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str                          # Raw user query
    context: dict[str, Any]            # Optional key-value context from caller

    # ── Reasoning ─────────────────────────────────────────────────────────────
    intent: str                         # Inferred intent (set by IntentAnalyzer)
    selected_agent: Optional[str]       # Which agent to route to (or None = direct)
    routing_reason: str                 # LLM explanation of why this agent was chosen

    # ── Execution ─────────────────────────────────────────────────────────────
    plan: Annotated[list[str], operator.add]            # Steps planned for execution
    intermediate_results: dict[str, Any]                # Results keyed by node name
    agent_raw_output: dict[str, Any]                    # Raw dict from the sub-agent

    # ── Output ────────────────────────────────────────────────────────────────
    final_answer: str
    confidence: float
    trace: Annotated[list[dict[str, Any]], operator.add]  # Accumulated trace steps
    error: Optional[str]                # Non-fatal error message if something degraded
