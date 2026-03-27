"""
Trade Agent state schema.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict


class TradeAgentState(TypedDict):
    # ---- Input from Master ----
    user_query: str
    intent_analysis: dict
    context_overrides: dict

    # ---- Sub-agent outputs ----
    trade_context: dict         # from trade_analyst
    parsed_intent: dict         # from query_analyst
    query_plan: dict            # from query_planner
    schema_info: dict           # from schema_analyzer
    generated_sql: str          # from query_builder
    sql_parameters: dict        # from query_builder
    validation_result: dict     # from query_validator
    query_results: dict         # from query_executor
    analysis: dict              # from details_analyzer
    artifacts: list[dict]       # generated files (charts, exports)

    # ---- Control flow ----
    execution_trace: Annotated[list[dict], list.__add__]
    needs_retry: bool
    retry_count: int
    retry_feedback: str         # error message fed back to query_builder
    error: str | None

    # ---- Human-in-the-Loop ----
    hitl_config: dict           # HITLConfig as dict
    hitl_pending: dict          # current pending interrupt (if any)
    hitl_response: dict         # user's resolution
    hitl_skipped: bool          # True if HITL is disabled or auto-approved
