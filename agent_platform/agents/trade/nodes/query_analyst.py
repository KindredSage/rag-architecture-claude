"""
Query Analyst: Parses user intent into a precise structured query specification.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from agents.trade.state import TradeAgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a query intent analyst for a ClickHouse-backed trading platform.
Using the user query AND the trade domain context, produce a precise structured breakdown.

Return ONLY a valid JSON object:
{{
  "operation": "<SELECT|AGGREGATE|JOIN|WINDOW|PIVOT|DESCRIBE>",
  "target_entities": ["<table or column guesses based on domain context>"],
  "filters": [
    {{"field": "<column>", "op": "<= | >= | != | = | IN | BETWEEN | LIKE>", "value": "<value>"}}
  ],
  "aggregations": [
    {{"function": "<SUM|AVG|COUNT|MIN|MAX|quantile|uniq|argMax>", "field": "<column>", "alias": "<name>"}}
  ],
  "group_by": ["<column1>", "<column2>"],
  "order_by": [{{"field": "<column>", "direction": "<ASC|DESC>"}}],
  "limit": <int or null>,
  "time_range": {{"start": "<date or relative>", "end": "<date or relative>"}},
  "output_format": "<table|scalar|time_series|distribution|ranking>",
  "needs_join": <true|false>,
  "join_hint": "<if join needed, describe which tables and keys>"
}}

Be specific about column names where you can infer them from domain context."""


async def query_analyst(state: TradeAgentState, *, llm) -> dict:
    start = time.perf_counter()

    prompt = f"""User Query: {state['user_query']}
Trade Context: {json.dumps(state.get('trade_context', {}), default=str)}
Intent: {json.dumps(state.get('intent_analysis', {}), default=str)}"""

    try:
        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]

        parsed = json.loads(raw.strip())
        duration = (time.perf_counter() - start) * 1000

        logger.info("Query parsed: op=%s entities=%s (%.0fms)",
                     parsed.get("operation"), parsed.get("target_entities", [])[:3], duration)

        return {
            "parsed_intent": parsed,
            "execution_trace": [{
                "node": "query_analyst",
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "output_summary": f"op={parsed.get('operation')}, format={parsed.get('output_format')}",
            }],
        }

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        logger.error("Query analyst failed: %s", e)
        return {
            "parsed_intent": {"operation": "SELECT", "target_entities": [], "output_format": "table"},
            "execution_trace": [{
                "node": "query_analyst",
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "error": str(e),
            }],
        }
