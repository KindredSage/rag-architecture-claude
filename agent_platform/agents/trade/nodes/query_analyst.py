"""
Query Analyst: Parses user intent into a precise structured query specification.

Uses conversation history and resolved_query from trade_analyst to handle
follow-up questions like "What are the downstreams?" (resolving to the
entity mentioned in the previous exchange).
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

SYSTEM_PROMPT = """You are a query intent analyst for a ClickHouse-backed trading platform.
Using the user query, trade domain context, and conversation history, produce a precise
structured breakdown of what data to fetch.

IMPORTANT: The trade context may contain a "resolved_query" field where pronouns and
references from conversation history have already been resolved. Use that as your primary
intent source. For example, if the user says "What are the downstreams?" and resolved_query
says "What are the downstreams of System A?", use the resolved version.

Return ONLY a valid JSON object:
{{
  "operation": "<SELECT|AGGREGATE|JOIN|WINDOW|PIVOT|DESCRIBE>",
  "target_entities": ["<table or column guesses based on domain context>"],
  "filters": [
    {{"field": "<column>", "op": "<= | >= | != | = | IN | BETWEEN | LIKE>", "value": "<value>"}}
  ],
  "aggregations": [
    {{"function": "<SUM|AVG|COUNT|MIN|MAX|quantile|uniq|argMax>", "field": "<column>", "alias": "<n>"}}
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

    trade_ctx = state.get("trade_context", {})
    resolved = trade_ctx.get("resolved_query", state["user_query"])

    # Build conversation history for context
    history = state.get("conversation_history", [])
    history_text = "(no prior conversation)"
    if history:
        lines = []
        for msg in history[-6:]:
            role = msg.get("role", "?")
            content = msg.get("content", {})
            if isinstance(content, dict):
                text = content.get("query", content.get("answer", str(content)))
            else:
                text = str(content)
            lines.append(f"  {role}: {text[:200]}")
        history_text = "\n".join(lines)

    prompt = f"""User Query (original): {state['user_query']}
Resolved Query (references resolved): {resolved}
Trade Context: {json.dumps(trade_ctx, default=str)}
Intent: {json.dumps(state.get('intent_analysis', {}), default=str)}

Conversation History:
{history_text}"""

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
