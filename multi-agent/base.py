"""
agents/base.py
--------------
Abstract base class for all agents in the registry.
Enforces a consistent interface so the Master Agent can invoke
any registered agent without knowing its internals.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """
    Every agent must:
      1. Declare a name (str) and description (str).
      2. List its capabilities so the Master Agent can reason about routing.
      3. Implement `execute(query, context) -> dict` (async).
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    name: str = "base_agent"
    description: str = "Abstract base agent"
    capabilities: list[str] = []

    # ── Contract ──────────────────────────────────────────────────────────────
    @abstractmethod
    async def execute(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Run the agent's LangGraph workflow.

        Returns a dict with at minimum:
            {
                "answer": str,
                "steps": list[dict],
                "confidence": float,
            }
        """
        ...

    def describe(self) -> dict[str, Any]:
        """Serialise metadata for the agent registry."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
        }
