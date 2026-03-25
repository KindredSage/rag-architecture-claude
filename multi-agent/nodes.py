"""
agents/master/nodes.py
----------------------
All LangGraph nodes for the Master Agent graph.

Node contract:
  • Accepts  MasterState
  • Returns  a PARTIAL dict (only keys it mutated)
  • Side-effects go in the trace list

Routing logic deliberately uses LLM reasoning (not if/else rules) so
new agents can be added to the registry without touching this file.
"""

from __future__ import annotations
import json
import time
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.master.state import MasterState
from app.core.config import get_settings
from app.core.logging import get_logger
from app.registry.agent_registry import registry
from app.utils.retry import with_retry

logger = get_logger(__name__)
settings = get_settings()


def _llm() -> ChatOpenAI:
    """Return a fresh LLM client (lightweight, stateless)."""
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )


def _trace(node: str, action: str, output: Any) -> dict:
    return {"node": node, "action": action, "output": output}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 – Intent Analyzer
# ─────────────────────────────────────────────────────────────────────────────

@with_retry(max_attempts=3, exceptions=(Exception,))
async def analyze_intent(state: MasterState) -> dict:
    """
    Use the LLM to extract a concise intent label from the user query.
    E.g. "retrieve trade details", "ask general question", "run analysis".
    """
    node_name = "IntentAnalyzer"
    logger.info("node_start", node=node_name, query=state["query"])

    system = SystemMessage(content=(
        "You are an expert intent classifier. "
        "Given a user query, respond with a single concise intent phrase "
        "(3-6 words, lowercase, no punctuation). "
        "Examples: 'retrieve trade details', 'get schema info', "
        "'answer general question', 'analyze data trends'."
    ))
    human = HumanMessage(content=f"Query: {state['query']}")

    response = await _llm().ainvoke([system, human])
    intent = response.content.strip().lower()

    logger.info("intent_extracted", node=node_name, intent=intent)
    return {
        "intent": intent,
        "trace": [_trace(node_name, "extracted_intent", intent)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 – Agent Selector (LLM-based, NOT rule-based)
# ─────────────────────────────────────────────────────────────────────────────

@with_retry(max_attempts=3, exceptions=(Exception,))
async def select_agent(state: MasterState) -> dict:
    """
    Feed the agent registry manifest + user intent to the LLM and ask it
    to choose the best agent (or 'none' for direct answers).

    This node is the heart of extensibility: adding a new agent to the
    registry automatically makes it available for LLM routing.
    """
    node_name = "AgentSelector"
    agents_manifest = json.dumps(registry.describe_all(), indent=2)

    system = SystemMessage(content=(
        "You are a smart orchestration router. "
        "You will be given:\n"
        "  1. A list of available agents with their names, descriptions and capabilities.\n"
        "  2. The user's query and its inferred intent.\n\n"
        "Your job:\n"
        "  • If the query is domain-specific and matches an agent's capabilities, "
        "    return the agent name.\n"
        "  • If the query is simple / conversational / general knowledge, "
        "    return 'none'.\n"
        "  • If the query is ambiguous and no agent clearly fits, return 'none'.\n\n"
        "Respond ONLY with valid JSON in this exact shape:\n"
        '{"selected_agent": "<name or none>", "reason": "<one sentence explanation>", '
        '"confidence": <float 0-1>}'
    ))
    human = HumanMessage(content=(
        f"Available agents:\n{agents_manifest}\n\n"
        f"User query: {state['query']}\n"
        f"Inferred intent: {state['intent']}"
    ))

    response = await _llm().ainvoke([system, human])

    # ── Parse LLM JSON ────────────────────────────────────────────────────────
    raw = response.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()

    parsed = json.loads(raw)
    selected = parsed.get("selected_agent", "none")
    reason = parsed.get("reason", "")
    confidence = float(parsed.get("confidence", 0.5))

    # Validate the agent actually exists
    if selected != "none" and selected not in registry:
        logger.warning("unknown_agent_selected", agent=selected)
        selected = "none"
        reason = f"LLM suggested '{selected}' which is not registered; falling back to direct."

    logger.info("agent_selected", node=node_name, selected=selected, reason=reason)
    return {
        "selected_agent": selected if selected != "none" else None,
        "routing_reason": reason,
        "confidence": confidence,
        "trace": [_trace(node_name, "agent_selected", {"agent": selected, "reason": reason})],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3a – Direct Answer (no agent needed)
# ─────────────────────────────────────────────────────────────────────────────

@with_retry(max_attempts=3, exceptions=(Exception,))
async def direct_answer(state: MasterState) -> dict:
    """Answer simple or general-knowledge queries directly via the LLM."""
    node_name = "DirectAnswer"
    logger.info("node_start", node=node_name)

    context_str = json.dumps(state.get("context", {})) if state.get("context") else "none"
    system = SystemMessage(content=(
        "You are a helpful assistant. Answer the user's question clearly and concisely. "
        "Use the provided context if relevant."
    ))
    human = HumanMessage(content=(
        f"Context: {context_str}\n\nQuestion: {state['query']}"
    ))

    response = await _llm().ainvoke([system, human])
    answer = response.content.strip()

    return {
        "final_answer": answer,
        "confidence": 0.9,
        "trace": [_trace(node_name, "direct_answer_generated", answer[:120] + "...")],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3b – Agent Executor
# ─────────────────────────────────────────────────────────────────────────────

async def execute_agent(state: MasterState) -> dict:
    """
    Look up the selected agent in the registry and run its LangGraph workflow.
    The agent's raw output dict is stored for the synthesiser to use.
    """
    node_name = "AgentExecutor"
    agent_name = state["selected_agent"]
    logger.info("node_start", node=node_name, agent=agent_name)

    agent = registry.get(agent_name)  # type: ignore[arg-type]
    if agent is None:
        return {
            "error": f"Agent '{agent_name}' not found in registry at execution time.",
            "final_answer": "Agent not found.",
            "confidence": 0.0,
            "trace": [_trace(node_name, "agent_not_found", agent_name)],
        }

    try:
        result = await agent.execute(state["query"], state.get("context", {}))
    except Exception as exc:
        logger.error("agent_execution_failed", agent=agent_name, error=str(exc))
        return {
            "error": str(exc),
            "final_answer": f"Agent '{agent_name}' failed: {exc}",
            "confidence": 0.0,
            "trace": [_trace(node_name, "agent_error", str(exc))],
        }

    logger.info("agent_executed", agent=agent_name)
    return {
        "agent_raw_output": result,
        "trace": [_trace(node_name, "agent_executed", {"agent": agent_name, "result_keys": list(result.keys())})],
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 – Response Synthesiser
# ─────────────────────────────────────────────────────────────────────────────

@with_retry(max_attempts=2, exceptions=(Exception,))
async def synthesize_response(state: MasterState) -> dict:
    """
    Take raw agent output and produce a polished, user-facing answer.
    Also propagates confidence and trace steps from the sub-agent.
    """
    node_name = "ResponseSynthesizer"
    logger.info("node_start", node=node_name)

    raw = state.get("agent_raw_output", {})
    agent_answer = raw.get("answer", "No answer returned by agent.")
    agent_steps = raw.get("steps", [])
    agent_confidence = float(raw.get("confidence", 0.5))

    # Give the LLM the raw agent answer to polish
    system = SystemMessage(content=(
        "You are a response formatter. Given a raw answer produced by a data agent, "
        "rewrite it to be clear, professional and well-structured. "
        "Preserve all factual content. Do not add information not present in the raw answer."
    ))
    human = HumanMessage(content=(
        f"Original user query: {state['query']}\n\n"
        f"Raw agent answer:\n{agent_answer}"
    ))

    response = await _llm().ainvoke([system, human])
    final_answer = response.content.strip()

    return {
        "final_answer": final_answer,
        "confidence": agent_confidence,
        "trace": [
            _trace(node_name, "response_synthesized", final_answer[:120] + "..."),
            *agent_steps,   # fold agent sub-steps into master trace
        ],
    }
