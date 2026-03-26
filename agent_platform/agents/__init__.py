"""
Agents package.

Importing this module triggers self-registration of all agents.
To add a new agent, just add an import line here.
"""

# Each import triggers AgentRegistry.register() in the agent's __init__.py
import agents.trade  # noqa: F401
import agents.reporting  # noqa: F401
import agents.analytics  # noqa: F401

from agents.registry import AgentRegistry
from agents.master_agent import build_master_graph

__all__ = ["AgentRegistry", "build_master_graph"]
