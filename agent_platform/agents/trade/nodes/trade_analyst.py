"""
Trade Analyst: Provides domain context for downstream agents.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from services.llm_invoke import invoke_llm

from agents.trade.state import TradeAgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a trade domain expert. Given a user query about trading data,
provide domain context that will help downstream agents build the right SQL query.

IMPORTANT: The user may refer to entities, systems, or concepts from previous messages
in the conversation. Use the conversation history to resolve pronouns ("it", "them"),
references ("the same", "those"), and implicit subjects. For example, if the user
previously asked about "System A" and now asks "What are the downstreams?", you should
understand they mean "downstreams of System A".

Determine:
- Asset class (equities, FX, fixed income, derivatives, commodities, crypto, mixed)
- Trade lifecycle stage (pre-trade, execution, post-trade, settlement, all)
- Relevant metrics (PnL, volume, VWAP, slippage, fill_rate, notional, spread, etc.)
- Time granularity needed (tick, minute, hourly, daily, monthly, yearly)
- Any regulatory or compliance context
- Suggested table names based on domain knowledge
- Resolved references from conversation context

Return ONLY a valid JSON object:
{{
  "asset_class": "<asset class>",
  "lifecycle_stage": "<stage>",
  "relevant_metrics": ["<metric1>", "<metric2>"],
  "time_granularity": "<granularity>",
  "domain_notes": "<any relevant domain context>",
  "suggested_tables": ["<table1>", "<table2>"],
  "special_considerations": "<e.g., need FINAL for dedup, timezone handling, etc.>",
  "resolved_query": "<the user query with all references resolved from context>"
}}"""


async def trade_analyst(state: TradeAgentState, *, llm) -> dict:
    start = time.perf_counter()

    # Build conversation context for reference resolution
    history = state.get("conversation_history", [])
    history_text = ""
    if history:
        recent = history[-6:]  # last 3 exchanges
        history_lines = []
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", {})
            if isinstance(content, dict):
                text = content.get("query", content.get("answer", str(content)))
            else:
                text = str(content)
            history_lines.append(f"  {role}: {text[:200]}")
        history_text = "\n".join(history_lines)

    prompt = f"""User Query: {state['user_query']}
Intent Analysis: {json.dumps(state.get('intent_analysis', {}), default=str)}

Conversation History (use this to resolve references like "it", "those", "the same system"):
{history_text if history_text else "(no prior conversation)"}"""

    try:
        response = await invoke_llm(llm, [
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
