"""
agents/trade/state.py
---------------------
Shared state flowing through the Trade Agent's LangGraph.
Each sub-agent node reads its relevant fields and writes back results.
"""

from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict, Annotated
import operator


class TradeState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────────
    query: str                          # User's original query
    context: dict[str, Any]            # Caller-supplied context

    # ── Analysis ──────────────────────────────────────────────────────────────
    query_analysis: dict[str, Any]      # QueryAnalyst output: entities, intent, params
    table_name: Optional[str]           # Extracted table name
    file_id: Optional[str]              # Extracted file/trade ID
    filters: dict[str, Any]             # Additional filter conditions

    # ── Planning ──────────────────────────────────────────────────────────────
    execution_plan: list[str]           # Ordered steps the planner decided on

    # ── Schema ────────────────────────────────────────────────────────────────
    schema_info: dict[str, Any]         # Columns, types, indexes for the target table

    # ── Query Building ────────────────────────────────────────────────────────
    raw_query: str                      # SQL or query string before validation
    validated_query: str                # Query after validation pass
    validation_errors: Annotated[list[str], operator.add]

    # ── Execution ─────────────────────────────────────────────────────────────
    query_result: dict[str, Any]        # Rows / result set from executor
    row_count: int

    # ── Analysis / Output ─────────────────────────────────────────────────────
    result_analysis: str                # Natural-language summary of results
    answer: str                         # Final answer string
    confidence: float

    # ── Trace ─────────────────────────────────────────────────────────────────
    steps: Annotated[list[dict[str, Any]], operator.add]
    error: Optional[str]
