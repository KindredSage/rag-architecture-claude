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
Using the user query, trade domain context, ACTUAL schema, and conversation history, produce
a precise structured breakdown of what data to fetch.

CRITICAL: Use ACTUAL column names from the schema below. Do NOT guess or invent column names.
Map user terms to real columns using the column_mappings from trade context.

=== ACTUAL TABLE SCHEMA ===
{schema_text}

=== SAMPLE DATA ===
{sample_rows_text}

=== COLUMN MAPPINGS (from trade analyst) ===
{column_mappings}

IMPORTANT: The trade context may contain a "resolved_query" field where pronouns and
references from conversation history have already been resolved. Use that as your primary
intent source.

Verify filter values against the sample data where possible. If a user mentions a value
(e.g., a ticker, desk name, or status) that you can confirm exists in the sample data,
mark it as verified. If you cannot confirm it, mark it as unverified.

Return ONLY a valid JSON object:
{{
  "operation": "<SELECT|AGGREGATE|JOIN|WINDOW|PIVOT|DESCRIBE>",
  "target_entities": ["<REAL table/column names from schema>"],
  "filters": [
    {{"field": "<REAL column name>", "op": "<= | >= | != | = | IN | BETWEEN | LIKE>", "value": "<value>"}}
  ],
  "aggregations": [
    {{"function": "<SUM|AVG|COUNT|MIN|MAX|quantile|uniq|argMax>", "field": "<REAL column name>", "alias": "<n>"}}
  ],
  "group_by": ["<REAL column name>"],
  "order_by": [{{"field": "<REAL column name>", "direction": "<ASC|DESC>"}}],
  "limit": <int or null>,
  "time_range": {{"start": "<date or relative>", "end": "<date or relative>"}},
  "output_format": "<table|scalar|time_series|distribution|ranking>",
  "needs_join": <true|false>,
  "join_hint": "<if join needed, describe which tables and keys>",
  "verified_values": ["<values confirmed in sample data>"],
  "unverified_values": ["<values NOT found in sample data — handle with caution>"]
}}"""


async def query_analyst(state: TradeAgentState, *, llm) -> dict:
    start = time.perf_counter()

    trade_ctx = state.get("trade_context", {})
    resolved = trade_ctx.get("resolved_query", state["user_query"])

    # Inject real schema into system prompt
    schema_info = state.get("schema_info", {})
    schema_text = schema_info.get("schema_text", "Schema not available")
    sample_rows_text = schema_info.get("sample_rows_text", "No sample data")
    column_mappings = json.dumps(trade_ctx.get("column_mappings", {}), indent=2)

    system_prompt = SYSTEM_PROMPT.format(
        schema_text=schema_text,
        sample_rows_text=sample_rows_text,
        column_mappings=column_mappings,
    )

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
            SystemMessage(content=system_prompt),
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
