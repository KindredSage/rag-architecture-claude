"""
registry/agent_registry.py
---------------------------
Central registry that holds all available agents.

Design goals:
  • Plug-and-play: new agents self-register via `registry.register(MyAgent())`
  • The Master Agent queries the registry to discover what agents exist,
    then feeds that list to the LLM for dynamic routing.
  • Thread-safe singleton.
"""

from __future__ import annotations
from typing import Dict, Optional
from app.agents.base import BaseAgent
from app.core.logging import get_logger

logger = get_logger(__name__)


class AgentRegistry:
    """Singleton registry mapping agent names → agent instances."""

    _instance: Optional["AgentRegistry"] = None

    def __new__(cls) -> "AgentRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents: Dict[str, BaseAgent] = {}
        return cls._instance

    # ── Mutations ─────────────────────────────────────────────────────────────

    def register(self, agent: BaseAgent) -> None:
        """Register an agent. Idempotent – re-registering overwrites."""
        if agent.name in self._agents:
            logger.info("agent_re_registered", agent=agent.name)
        else:
            logger.info("agent_registered", agent=agent.name)
        self._agents[agent.name] = agent

    def unregister(self, name: str) -> None:
        """Remove an agent from the registry."""
        if name in self._agents:
            del self._agents[name]
            logger.info("agent_unregistered", agent=name)

    # ── Queries ───────────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[BaseAgent]:
        """Return an agent by name, or None if not registered."""
        return self._agents.get(name)

    def list_agents(self) -> list[BaseAgent]:
        """Return all registered agent instances."""
        return list(self._agents.values())

    def describe_all(self) -> list[dict]:
        """
        Return a list of agent metadata dicts.
        Used by the Master Agent to build its routing prompt.
        """
        return [a.describe() for a in self._agents.values()]

    def agent_names(self) -> list[str]:
        return list(self._agents.keys())

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents


# Module-level singleton – import this everywhere
registry = AgentRegistry()
