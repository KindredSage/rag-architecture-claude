"""
Query Planner: Plans the optimal ClickHouse query execution strategy.
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

SYSTEM_PROMPT = """You are a ClickHouse query strategist. Given the parsed intent and domain context,
plan the optimal query execution strategy.

Consider:
- Whether a single query suffices or multiple queries are needed
- CTE (WITH clause) usage for complex logic
- ClickHouse-specific optimizations:
  * PREWHERE for partition key filters (much faster than WHERE)
  * Sampling for approximate results on huge tables
  * Materialized view usage if available
  * Partition pruning via date/time filters
- Whether FINAL keyword is needed for ReplacingMergeTree or CollapsingMergeTree
- Memory limits and timeouts
- Data volume estimation

Return ONLY a valid JSON object:
{{
  "strategy": "<single_query|multi_step|cte_chain>",
  "steps": [
    {{
      "step": 1,
      "description": "<what this step does>",
      "depends_on": [],
      "query_type": "<main|subquery|cte>"
    }}
  ],
  "optimization_hints": [
    "<use PREWHERE on trade_date for partition pruning>",
    "<consider FINAL if ReplacingMergeTree>"
  ],
  "estimated_complexity": "<low|medium|high>",
  "estimated_data_volume": "<small (<1M rows)|medium (1-100M)|large (>100M)>",
  "needs_final": <true|false>,
  "needs_sampling": <true|false>,
  "sampling_rate": <null or float like 0.1>
}}"""


async def query_planner(state: TradeAgentState, *, llm) -> dict:
    start = time.perf_counter()

    prompt = f"""User Query: {state['user_query']}
Parsed Intent: {json.dumps(state.get('parsed_intent', {}), default=str)}
Trade Context: {json.dumps(state.get('trade_context', {}), default=str)}"""

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

        plan = json.loads(raw.strip())
        duration = (time.perf_counter() - start) * 1000

        logger.info(
            "Query planned: strategy=%s steps=%d complexity=%s (%.0fms)",
            plan.get("strategy"),
            len(plan.get("steps", [])),
            plan.get("estimated_complexity"),
            duration,
        )

        return {
            "query_plan": plan,
            "execution_trace": [{
                "node": "query_planner",
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "output_summary": f"strategy={plan.get('strategy')}, complexity={plan.get('estimated_complexity')}",
            }],
        }

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        logger.error("Query planner failed: %s", e)
        return {
            "query_plan": {
                "strategy": "single_query",
                "steps": [{"step": 1, "description": "Direct query", "depends_on": []}],
                "optimization_hints": [],
                "estimated_complexity": "low",
            },
            "execution_trace": [{
                "node": "query_planner",
                "status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "error": str(e),
            }],
        }
