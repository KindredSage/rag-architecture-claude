"""
Central Agent Registry.

Agents self-register at import time. The Master Agent queries the registry
to build routing context for LLM-based agent selection.

Adding a new agent:
  1. Create agents/your_agent/ directory with graph.py, nodes/, etc.
  2. In agents/your_agent/__init__.py, call AgentRegistry.register(...)
  3. Import the agent module in agents/__init__.py
  4. Done -- the master agent auto-discovers it.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from models import AgentDescriptor

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Singleton registry for all available agents."""

    _agents: dict[str, tuple[AgentDescriptor, Callable]] = {}

    @classmethod
    def register(
        cls,
        descriptor: AgentDescriptor,
        graph_factory: Callable,
    ) -> None:
        """
        Register an agent.

        Args:
            descriptor: Agent metadata (used for routing decisions).
            graph_factory: A callable that accepts (settings, services_dict)
                          and returns a compiled LangGraph StateGraph.
        """
        if descriptor.agent_id in cls._agents:
            logger.warning("Agent %s already registered, overwriting", descriptor.agent_id)
        cls._agents[descriptor.agent_id] = (descriptor, graph_factory)
        logger.info(
            "Agent registered: %s (%s) - %d capabilities",
            descriptor.agent_id,
            descriptor.name,
            len(descriptor.capabilities),
        )

    @classmethod
    def unregister(cls, agent_id: str) -> None:
        cls._agents.pop(agent_id, None)

    @classmethod
    def get_all(cls) -> list[AgentDescriptor]:
        return [desc for desc, _ in cls._agents.values() if desc.enabled]

    @classmethod
    def get_descriptor(cls, agent_id: str) -> AgentDescriptor | None:
        entry = cls._agents.get(agent_id)
        return entry[0] if entry else None

    @classmethod
    def get_graph_factory(cls, agent_id: str) -> Callable | None:
        entry = cls._agents.get(agent_id)
        return entry[1] if entry else None

    @classmethod
    def has_agent(cls, agent_id: str) -> bool:
        return agent_id in cls._agents

    @classmethod
    def get_routing_context(cls) -> str:
        """
        Build a structured text block describing all agents.
        This is injected into the Master Agent's system prompt so the
        LLM can make informed routing decisions.
        """
        lines: list[str] = []
        for desc, _ in cls._agents.values():
            if not desc.enabled:
                continue
            capabilities_str = ", ".join(c.value for c in desc.capabilities)
            keywords_str = ", ".join(desc.domain_keywords[:15])
            examples_str = "; ".join(desc.example_queries[:3])
            lines.append(
                f"AGENT: {desc.agent_id}\n"
                f"  Name: {desc.name}\n"
                f"  Description: {desc.description}\n"
                f"  Capabilities: [{capabilities_str}]\n"
                f"  Domain Keywords: [{keywords_str}]\n"
                f"  Sub-Agents: {desc.sub_agents}\n"
                f"  Example Queries: [{examples_str}]\n"
                f"  Priority: {desc.priority}"
            )
        return "\n\n".join(lines) if lines else "No agents registered."

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations (useful for testing)."""
        cls._agents.clear()
