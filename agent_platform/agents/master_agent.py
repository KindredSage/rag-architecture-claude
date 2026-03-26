"""
Master Agent: LangGraph StateGraph that routes user queries to sub-agent graphs.

Flow:
  START -> classify_intent -> select_agents -> execute_agents -> merge_results -> END

This is the brain of the platform. It:
  1. Parses user queries for intent, domain, and desired output
  2. Selects the optimal agent(s) from the registry
  3. Invokes sub-graphs with proper state mapping
  4. Merges results and produces a unified response
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agents.registry import AgentRegistry
from models import (
    AgentSelection,
    ExecutionStrategy,
    IntentAnalysis,
    RoutingDecision,
    TraceStep,
)

logger = logging.getLogger(__name__)


# =====================================================================
# State
# =====================================================================


def _merge_dicts(a: dict, b: dict) -> dict:
    merged = {**a}
    merged.update(b)
    return merged


class MasterState(TypedDict):
    # ---- Input ----
    user_query: str
    user_context: dict  # user_id, preferences, forced_agent, session history

    # ---- Routing ----
    intent_analysis: dict
    routing_decision: dict
    routing_reasoning: str

    # ---- Execution ----
    agent_results: Annotated[dict, _merge_dicts]
    execution_trace: Annotated[list[dict], list.__add__]

    # ---- Output ----
    final_response: dict
    artifacts: list[dict]
    error: str | None


# =====================================================================
# Prompts
# =====================================================================

INTENT_CLASSIFIER_PROMPT = """You are an intent classifier for a multi-agent analytical data platform backed by ClickHouse.

Analyze the user query and return ONLY a valid JSON object (no markdown fences, no explanation) with this exact structure:
{{
  "primary_domain": "<trade|analytics|reporting|general>",
  "intent": "<query_data|generate_report|plot_chart|send_email|summarize|explore_schema|anomaly_check|export_data>",
  "entities": ["<list of key entities: table names, metrics, dates, tickers, etc.>"],
  "desired_output": "<table|chart|report|email|summary|raw_data|schema|export>",
  "complexity": "<simple|moderate|complex>",
  "requires_multi_agent": <true|false>,
  "sub_intents": ["<if multi-agent, list each sub-intent>"],
  "time_range": "<extracted time range or null>",
  "filters": {{"<any explicit filters mentioned>": "<value>"}},
  "ambiguity_notes": "<anything unclear that might need clarification>"
}}

Available agents and their capabilities:
{agent_context}

User Query: {user_query}

Conversation history (last few exchanges, if any):
{history_context}
"""

AGENT_SELECTOR_PROMPT = """You are an agent router for a multi-agent platform.
Given the analyzed intent and available agents, select the best agent(s) to handle this request.

Return ONLY a valid JSON object (no markdown fences):
{{
  "selected_agents": [
    {{"agent_id": "<id>", "reason": "<why this agent>", "confidence": <0.0-1.0>}}
  ],
  "execution_strategy": "<sequential|parallel|pipeline>",
  "context_overrides": {{"<any special context to pass to the sub-agent>": "<value>"}}
}}

Selection criteria (weighted):
  1. Capability match with desired_output (0.4)
  2. Domain keyword overlap (0.3)
  3. Description relevance (0.2)
  4. Priority score (0.1)

If the intent maps clearly to one agent, select only that one.
If the query requires multiple capabilities across agents, select multiple with strategy.
If forced_agent is specified and valid, use it.

Intent Analysis: {intent_analysis}
Forced Agent: {forced_agent}

Available Agents:
{agent_context}
"""

RESULT_MERGER_PROMPT = """You are a senior analyst producing a final response.
Combine the agent results into a clear, actionable answer.

Return ONLY a valid JSON object (no markdown fences):
{{
  "answer": "<Natural language answer to the user's question>",
  "data_summary": {{"<key metrics and their values>": "<value>"}},
  "visualizations": [
    {{"chart_type": "<type>", "title": "<title>", "x_axis": "<col>", "y_axis": "<col>", "config": {{}}}}
  ],
  "suggestions": ["<2-3 follow-up questions the user might ask>"],
  "confidence": <0.0-1.0>,
  "execution_summary": "<brief trace of what was done>"
}}

User Query: {user_query}
Agent Results: {agent_results}
"""


# =====================================================================
# Node Implementations
# =====================================================================


async def classify_intent(state: MasterState, *, llm, **kwargs) -> dict:
    """Classify user intent using LLM."""
    start = time.perf_counter()
    agent_context = AgentRegistry.get_routing_context()
    history_context = json.dumps(
        state.get("user_context", {}).get("history", [])[-6:],
        default=str,
    )

    prompt = INTENT_CLASSIFIER_PROMPT.format(
        agent_context=agent_context,
        user_query=state["user_query"],
        history_context=history_context,
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content="You are a precise JSON-only intent classifier."),
            HumanMessage(content=prompt),
        ])

        raw = response.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        intent = json.loads(raw)
        duration = (time.perf_counter() - start) * 1000

        logger.info(
            "Intent classified: domain=%s intent=%s complexity=%s (%.0fms)",
            intent.get("primary_domain"),
            intent.get("intent"),
            intent.get("complexity"),
            duration,
        )

        return {
            "intent_analysis": intent,
            "execution_trace": [
                {
                    "node": "classify_intent",
                    "status": "completed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round(duration, 1),
                    "output_summary": f"domain={intent.get('primary_domain')}, intent={intent.get('intent')}",
                }
            ],
        }

    except (json.JSONDecodeError, Exception) as e:
        logger.error("Intent classification failed: %s", e)
        duration = (time.perf_counter() - start) * 1000
        # Fallback intent
        fallback = {
            "primary_domain": "general",
            "intent": "query_data",
            "entities": [],
            "desired_output": "table",
            "complexity": "simple",
            "requires_multi_agent": False,
            "sub_intents": [],
            "time_range": None,
            "filters": {},
            "ambiguity_notes": f"Classification failed: {e}. Using fallback.",
        }
        return {
            "intent_analysis": fallback,
            "execution_trace": [
                {
                    "node": "classify_intent",
                    "status": "fallback",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round(duration, 1),
                    "error": str(e),
                    "output_summary": "Using fallback intent",
                }
            ],
        }


async def select_agents(state: MasterState, *, llm, **kwargs) -> dict:
    """Select the best agent(s) for the classified intent."""
    start = time.perf_counter()
    intent = state.get("intent_analysis", {})
    forced = state.get("user_context", {}).get("forced_agent")

    # Fast path: forced agent
    if forced and AgentRegistry.has_agent(forced):
        duration = (time.perf_counter() - start) * 1000
        decision = {
            "selected_agents": [
                {"agent_id": forced, "reason": "User-forced selection", "confidence": 1.0}
            ],
            "execution_strategy": "sequential",
            "context_overrides": {},
        }
        return {
            "routing_decision": decision,
            "routing_reasoning": f"Forced to agent: {forced}",
            "execution_trace": [
                {
                    "node": "select_agents",
                    "status": "completed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round(duration, 1),
                    "output_summary": f"Forced: {forced}",
                }
            ],
        }

    agent_context = AgentRegistry.get_routing_context()
    prompt = AGENT_SELECTOR_PROMPT.format(
        intent_analysis=json.dumps(intent, default=str),
        forced_agent=forced or "None",
        agent_context=agent_context,
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content="You are a precise JSON-only agent router."),
            HumanMessage(content=prompt),
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        decision = json.loads(raw)
        duration = (time.perf_counter() - start) * 1000

        # Validate selected agents exist
        valid_selections = []
        for sel in decision.get("selected_agents", []):
            if AgentRegistry.has_agent(sel["agent_id"]):
                valid_selections.append(sel)
            else:
                logger.warning("Selected agent %s not found in registry", sel["agent_id"])

        if not valid_selections:
            # Fallback: pick the highest-priority agent
            all_agents = AgentRegistry.get_all()
            if all_agents:
                best = max(all_agents, key=lambda a: a.priority)
                valid_selections = [
                    {"agent_id": best.agent_id, "reason": "Fallback: highest priority", "confidence": 0.5}
                ]

        decision["selected_agents"] = valid_selections
        selected_ids = [s["agent_id"] for s in valid_selections]

        logger.info("Agents selected: %s (%.0fms)", selected_ids, duration)

        return {
            "routing_decision": decision,
            "routing_reasoning": json.dumps(
                [{"agent": s["agent_id"], "reason": s["reason"]} for s in valid_selections]
            ),
            "execution_trace": [
                {
                    "node": "select_agents",
                    "status": "completed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round(duration, 1),
                    "output_summary": f"Selected: {selected_ids}",
                }
            ],
        }

    except Exception as e:
        logger.error("Agent selection failed: %s", e)
        duration = (time.perf_counter() - start) * 1000

        # Fallback
        all_agents = AgentRegistry.get_all()
        fallback_id = all_agents[0].agent_id if all_agents else "trade_agent"
        decision = {
            "selected_agents": [
                {"agent_id": fallback_id, "reason": f"Fallback due to error: {e}", "confidence": 0.3}
            ],
            "execution_strategy": "sequential",
            "context_overrides": {},
        }
        return {
            "routing_decision": decision,
            "routing_reasoning": f"Fallback due to error: {e}",
            "execution_trace": [
                {
                    "node": "select_agents",
                    "status": "fallback",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round(duration, 1),
                    "error": str(e),
                }
            ],
        }


async def execute_agents(state: MasterState, *, services, settings, **kwargs) -> dict:
    """Invoke selected sub-agent graphs and collect results."""
    start = time.perf_counter()
    decision = state.get("routing_decision", {})
    selections = decision.get("selected_agents", [])
    strategy = decision.get("execution_strategy", "sequential")
    overrides = decision.get("context_overrides", {})

    if not selections:
        return {
            "agent_results": {},
            "error": "No agents selected",
            "execution_trace": [
                {
                    "node": "execute_agents",
                    "status": "error",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": "No agents selected",
                }
            ],
        }

    async def _run_one_agent(sel: dict) -> tuple[str, dict]:
        agent_id = sel["agent_id"]
        factory = AgentRegistry.get_graph_factory(agent_id)
        if not factory:
            return agent_id, {"error": f"Agent {agent_id} graph factory not found"}

        try:
            graph = factory(settings=settings, services=services)

            agent_input = {
                "user_query": state["user_query"],
                "intent_analysis": state.get("intent_analysis", {}),
                "context_overrides": overrides,
                "execution_trace": [],
                "needs_retry": False,
                "retry_count": 0,
                "error": None,
            }

            result = await graph.ainvoke(agent_input)
            return agent_id, result

        except Exception as e:
            logger.error("Agent %s execution failed: %s", agent_id, e, exc_info=True)
            return agent_id, {"error": str(e)}

    # Execute based on strategy
    results: dict[str, Any] = {}
    sub_traces: list[dict] = []

    if strategy == "parallel" and len(selections) > 1:
        tasks = [_run_one_agent(sel) for sel in selections]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, Exception):
                logger.error("Parallel agent error: %s", outcome)
                continue
            agent_id, result = outcome
            results[agent_id] = result
            sub_traces.extend(result.get("execution_trace", []))
    else:
        # Sequential (default) or pipeline
        prev_output = None
        for sel in selections:
            agent_id, result = await _run_one_agent(sel)
            results[agent_id] = result
            sub_traces.extend(result.get("execution_trace", []))
            prev_output = result

    duration = (time.perf_counter() - start) * 1000
    logger.info("All agents executed: %s (%.0fms)", list(results.keys()), duration)

    # Collect artifacts from all agents
    all_artifacts = []
    for agent_id, result in results.items():
        all_artifacts.extend(result.get("artifacts", []))

    return {
        "agent_results": results,
        "artifacts": all_artifacts,
        "execution_trace": sub_traces + [
            {
                "node": "execute_agents",
                "status": "completed",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "output_summary": f"Executed {len(results)} agents",
            }
        ],
    }


async def merge_results(state: MasterState, *, llm, **kwargs) -> dict:
    """Merge results from all sub-agents into a unified response."""
    start = time.perf_counter()
    results = state.get("agent_results", {})

    # If only one agent and it produced a clean analysis, use it directly
    if len(results) == 1:
        agent_id, result = next(iter(results.items()))
        if result.get("analysis") and not result.get("error"):
            analysis = result["analysis"]
            duration = (time.perf_counter() - start) * 1000
            return {
                "final_response": {
                    "answer": analysis.get("narrative", ""),
                    "data_summary": analysis.get("data_summary", {}),
                    "visualizations": analysis.get("visualization_recommendations", []),
                    "suggestions": analysis.get("follow_up_questions", []),
                    "confidence": analysis.get("confidence", 0.8),
                    "execution_summary": f"Processed by {agent_id}",
                    "data": result.get("query_results", {}).get("data", []),
                    "raw_sql": result.get("generated_sql", ""),
                },
                "execution_trace": [
                    {
                        "node": "merge_results",
                        "status": "completed",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "duration_ms": round(duration, 1),
                        "output_summary": "Direct passthrough from single agent",
                    }
                ],
            }

    # Multi-agent or fallback: use LLM to merge
    # Prepare a sanitized version of results for the LLM
    sanitized = {}
    for agent_id, result in results.items():
        sanitized[agent_id] = {
            "analysis": result.get("analysis", {}),
            "query_results_summary": {
                "row_count": result.get("query_results", {}).get("row_count", 0),
                "columns": result.get("query_results", {}).get("columns", []),
                "sample_data": result.get("query_results", {}).get("data", [])[:5],
            },
            "error": result.get("error"),
        }

    prompt = RESULT_MERGER_PROMPT.format(
        user_query=state["user_query"],
        agent_results=json.dumps(sanitized, default=str, indent=2)[:8000],
    )

    try:
        response = await llm.ainvoke([
            SystemMessage(content="You are a senior analyst. Return ONLY valid JSON."),
            HumanMessage(content=prompt),
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]

        final = json.loads(raw.strip())
        duration = (time.perf_counter() - start) * 1000

        # Attach raw data from first successful agent
        for result in results.values():
            qr = result.get("query_results", {})
            if qr.get("data"):
                final["data"] = qr["data"]
                final["raw_sql"] = result.get("generated_sql", "")
                break

        return {
            "final_response": final,
            "execution_trace": [
                {
                    "node": "merge_results",
                    "status": "completed",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round(duration, 1),
                    "output_summary": f"Merged {len(results)} agent results",
                }
            ],
        }

    except Exception as e:
        logger.error("Result merge failed: %s", e)
        duration = (time.perf_counter() - start) * 1000

        # Best-effort fallback
        fallback_answer = ""
        for agent_id, result in results.items():
            narrative = result.get("analysis", {}).get("narrative", "")
            if narrative:
                fallback_answer += f"[{agent_id}]: {narrative}\n"
            elif result.get("error"):
                fallback_answer += f"[{agent_id}]: Error - {result['error']}\n"

        return {
            "final_response": {
                "answer": fallback_answer or "Unable to process the query.",
                "confidence": 0.3,
                "execution_summary": f"Merge failed: {e}. Showing raw agent outputs.",
            },
            "execution_trace": [
                {
                    "node": "merge_results",
                    "status": "fallback",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round(duration, 1),
                    "error": str(e),
                }
            ],
        }


# =====================================================================
# Graph Assembly
# =====================================================================


def build_master_graph(*, settings, services):
    """
    Build and compile the Master Agent StateGraph.

    The `settings` and `services` are captured in closures so each
    node function has access to LLM, ClickHouse, etc.
    """
    from services.llm_service import LLMService

    llm_service: LLMService = services["llm"]
    llm_fast = llm_service.get_model(fast=True)
    llm_primary = llm_service.get_model(fast=False)

    # Bind dependencies into node functions
    async def _classify(state: MasterState) -> dict:
        return await classify_intent(state, llm=llm_fast)

    async def _select(state: MasterState) -> dict:
        return await select_agents(state, llm=llm_fast)

    async def _execute(state: MasterState) -> dict:
        return await execute_agents(state, services=services, settings=settings)

    async def _merge(state: MasterState) -> dict:
        return await merge_results(state, llm=llm_primary)

    # Build graph
    graph = StateGraph(MasterState)

    graph.add_node("classify_intent", _classify)
    graph.add_node("select_agents", _select)
    graph.add_node("execute_agents", _execute)
    graph.add_node("merge_results", _merge)

    graph.add_edge(START, "classify_intent")
    graph.add_edge("classify_intent", "select_agents")
    graph.add_edge("select_agents", "execute_agents")
    graph.add_edge("execute_agents", "merge_results")
    graph.add_edge("merge_results", END)

    return graph.compile()
