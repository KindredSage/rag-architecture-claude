"""
Trade Analyst: Provides domain context for downstream agents.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from agents.trade.state import TradeAgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a trade domain expert. Given a user query about trading data,
provide domain context that will help downstream agents build the right SQL query.

Determine:
- Asset class (equities, FX, fixed income, derivatives, commodities, crypto, mixed)
- Trade lifecycle stage (pre-trade, execution, post-trade, settlement, all)
- Relevant metrics (PnL, volume, VWAP, slippage, fill_rate, notional, spread, etc.)
- Time granularity needed (tick, minute, hourly, daily, monthly, yearly)
- Any regulatory or compliance context
- Suggested table names based on domain knowledge

Return ONLY a valid JSON object:
{{
  "asset_class": "<asset class>",
  "lifecycle_stage": "<stage>",
  "relevant_metrics": ["<metric1>", "<metric2>"],
  "time_granularity": "<granularity>",
  "domain_notes": "<any relevant domain context>",
  "suggested_tables": ["<table1>", "<table2>"],
  "special_considerations": "<e.g., need FINAL for dedup, timezone handling, etc.>"
}}"""


async def trade_analyst(state: TradeAgentState, *, llm) -> dict:
    start = time.perf_counter()

    prompt = f"""User Query: {state['user_query']}
Intent Analysis: {json.dumps(state.get('intent_analysis', {}), default=str)}"""

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

        context = json.loads(raw.strip())
        duration = (time.perf_counter() - start) * 1000

        logger.info("Trade context: asset=%s stage=%s (%.0fms)",
                     context.get("asset_class"), context.get("lifecycle_stage"), duration)

        return {
            "trade_context": context,
            "execution_trace": [{
                "node": "trade_analyst",
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "output_summary": f"asset={context.get('asset_class')}, metrics={context.get('relevant_metrics', [])[:3]}",
            }],
        }

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        logger.error("Trade analyst failed: %s", e)
        return {
            "trade_context": {"asset_class": "unknown", "relevant_metrics": [], "suggested_tables": []},
            "execution_trace": [{
                "node": "trade_analyst",
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "error": str(e),
            }],
        }
