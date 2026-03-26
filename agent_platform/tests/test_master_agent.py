"""
Tests for Master Agent: intent classification, agent selection, routing logic.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.registry import AgentRegistry
from models import AgentCapability, AgentDescriptor


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry is clean before each test."""
    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


@pytest.fixture
def registered_agents():
    """Register sample agents for routing tests."""
    trade_desc = AgentDescriptor(
        agent_id="trade_agent",
        name="Trade Agent",
        description="Analyzes trading data",
        capabilities=[AgentCapability.SQL_QUERY, AgentCapability.PLOTTING],
        domain_keywords=["trade", "PnL", "volume", "desk", "ticker"],
        sub_agents=["trade_analyst", "query_builder"],
        example_queries=["What was the PnL last month?"],
        priority=10,
    )
    report_desc = AgentDescriptor(
        agent_id="reporting_agent",
        name="Reporting Agent",
        description="Generates reports and sends emails",
        capabilities=[AgentCapability.REPORT_GENERATION, AgentCapability.EMAIL],
        domain_keywords=["report", "email", "PDF", "send", "export"],
        sub_agents=["report_planner", "email_dispatcher"],
        example_queries=["Email the monthly report"],
        priority=5,
    )
    analytics_desc = AgentDescriptor(
        agent_id="analytics_agent",
        name="Analytics Agent",
        description="Trend analysis and anomaly detection",
        capabilities=[AgentCapability.ANOMALY_DETECTION, AgentCapability.SUMMARY],
        domain_keywords=["trend", "anomaly", "pattern", "insight"],
        sub_agents=["trend_analyzer", "anomaly_detector"],
        example_queries=["Are there anomalies in trading volume?"],
        priority=5,
    )

    def _noop(**kwargs): return MagicMock()
    AgentRegistry.register(trade_desc, _noop)
    AgentRegistry.register(report_desc, _noop)
    AgentRegistry.register(analytics_desc, _noop)

    return [trade_desc, report_desc, analytics_desc]


# ── Registry Tests ───────────────────────────────────────────────


class TestAgentRegistry:
    def test_register_and_list(self, registered_agents):
        agents = AgentRegistry.get_all()
        assert len(agents) == 3
        ids = {a.agent_id for a in agents}
        assert ids == {"trade_agent", "reporting_agent", "analytics_agent"}

    def test_get_descriptor(self, registered_agents):
        desc = AgentRegistry.get_descriptor("trade_agent")
        assert desc is not None
        assert desc.name == "Trade Agent"
        assert desc.priority == 10

    def test_get_nonexistent(self):
        assert AgentRegistry.get_descriptor("nonexistent") is None
        assert AgentRegistry.has_agent("nonexistent") is False

    def test_routing_context_format(self, registered_agents):
        context = AgentRegistry.get_routing_context()
        assert "trade_agent" in context
        assert "reporting_agent" in context
        assert "Capabilities:" in context
        assert "Domain Keywords:" in context

    def test_clear(self, registered_agents):
        assert len(AgentRegistry.get_all()) == 3
        AgentRegistry.clear()
        assert len(AgentRegistry.get_all()) == 0

    def test_disabled_agent_excluded(self, registered_agents):
        desc = AgentRegistry.get_descriptor("trade_agent")
        desc.enabled = False
        agents = AgentRegistry.get_all()
        assert "trade_agent" not in {a.agent_id for a in agents}
        desc.enabled = True  # restore


# ── Intent Classification Tests ──────────────────────────────────


class TestIntentClassification:
    @pytest.mark.asyncio
    async def test_trade_query_classified(self, mock_llm, mock_llm_response, registered_agents):
        from agents.master_agent import classify_intent

        intent_json = json.dumps({
            "primary_domain": "trade",
            "intent": "query_data",
            "entities": ["PnL", "desk"],
            "desired_output": "table",
            "complexity": "simple",
            "requires_multi_agent": False,
            "sub_intents": [],
            "time_range": "last month",
            "filters": {},
            "ambiguity_notes": "",
        })
        mock_llm.ainvoke.return_value = mock_llm_response(intent_json)

        state = {
            "user_query": "What was the PnL by desk last month?",
            "user_context": {},
        }
        result = await classify_intent(state, llm=mock_llm)

        assert result["intent_analysis"]["primary_domain"] == "trade"
        assert result["intent_analysis"]["intent"] == "query_data"
        assert len(result["execution_trace"]) == 1
        assert result["execution_trace"][0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_classification_fallback_on_error(self, mock_llm, registered_agents):
        mock_llm.ainvoke.side_effect = Exception("LLM unavailable")

        state = {"user_query": "test query", "user_context": {}}
        result = await classify_intent(state, llm=mock_llm)

        # Should produce fallback intent, not crash
        assert result["intent_analysis"]["primary_domain"] == "general"
        assert result["execution_trace"][0]["status"] == "fallback"

    @pytest.mark.asyncio
    async def test_classification_handles_markdown_fences(self, mock_llm, mock_llm_response, registered_agents):
        from agents.master_agent import classify_intent

        # LLM wraps JSON in markdown
        fenced = '```json\n{"primary_domain":"trade","intent":"query_data","entities":[],"desired_output":"table","complexity":"simple","requires_multi_agent":false,"sub_intents":[],"time_range":null,"filters":{},"ambiguity_notes":""}\n```'
        mock_llm.ainvoke.return_value = mock_llm_response(fenced)

        state = {"user_query": "PnL query", "user_context": {}}
        result = await classify_intent(state, llm=mock_llm)
        assert result["intent_analysis"]["primary_domain"] == "trade"


# ── Agent Selection Tests ────────────────────────────────────────


class TestAgentSelection:
    @pytest.mark.asyncio
    async def test_forced_agent(self, mock_llm, registered_agents):
        from agents.master_agent import select_agents

        state = {
            "user_query": "test",
            "intent_analysis": {},
            "user_context": {"forced_agent": "trade_agent"},
        }
        result = await select_agents(state, llm=mock_llm)
        selected = result["routing_decision"]["selected_agents"]
        assert len(selected) == 1
        assert selected[0]["agent_id"] == "trade_agent"

    @pytest.mark.asyncio
    async def test_forced_invalid_agent_falls_through(self, mock_llm, mock_llm_response, registered_agents):
        from agents.master_agent import select_agents

        selection_json = json.dumps({
            "selected_agents": [{"agent_id": "trade_agent", "reason": "best match", "confidence": 0.9}],
            "execution_strategy": "sequential",
            "context_overrides": {},
        })
        mock_llm.ainvoke.return_value = mock_llm_response(selection_json)

        state = {
            "user_query": "test",
            "intent_analysis": {},
            "user_context": {"forced_agent": "nonexistent_agent"},
        }
        result = await select_agents(state, llm=mock_llm)
        # Should not use forced agent since it doesn't exist
        selected = result["routing_decision"]["selected_agents"]
        assert all(AgentRegistry.has_agent(s["agent_id"]) for s in selected)

    @pytest.mark.asyncio
    async def test_selection_fallback_on_error(self, mock_llm, registered_agents):
        from agents.master_agent import select_agents

        mock_llm.ainvoke.side_effect = Exception("LLM error")
        state = {
            "user_query": "test",
            "intent_analysis": {},
            "user_context": {},
        }
        result = await select_agents(state, llm=mock_llm)
        # Should fallback to highest-priority agent
        selected = result["routing_decision"]["selected_agents"]
        assert len(selected) >= 1
        assert result["execution_trace"][0]["status"] == "fallback"
