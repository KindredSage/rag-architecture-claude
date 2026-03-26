"""
Integration tests: end-to-end flow through the master agent pipeline.

Uses fully mocked services (no real LLM/CH/PG) but exercises the real
LangGraph graph compilation and invocation.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from agents.registry import AgentRegistry
from models import AgentCapability, AgentDescriptor


@pytest.fixture(autouse=True)
def clean_registry():
    AgentRegistry.clear()
    yield
    AgentRegistry.clear()


def _make_mock_llm_response(content: str):
    m = MagicMock()
    m.content = content
    return m


@pytest.fixture
def mock_llm_with_responses():
    """
    LLM mock that returns different responses based on call order:
      1st call -> intent classification
      2nd call -> agent selection
      3rd+ calls -> sub-agent responses
    """
    llm = AsyncMock()
    call_count = {"n": 0}

    intent = json.dumps({
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

    selection = json.dumps({
        "selected_agents": [{"agent_id": "test_agent", "reason": "best match", "confidence": 0.95}],
        "execution_strategy": "sequential",
        "context_overrides": {},
    })

    trade_context = json.dumps({
        "asset_class": "equities", "lifecycle_stage": "post-trade",
        "relevant_metrics": ["PnL"], "time_granularity": "daily",
        "domain_notes": "", "suggested_tables": ["trades"],
        "special_considerations": "",
    })

    parsed_intent = json.dumps({
        "operation": "AGGREGATE", "target_entities": ["trades"],
        "filters": [], "aggregations": [{"function": "SUM", "field": "pnl", "alias": "total_pnl"}],
        "group_by": ["desk"], "order_by": [{"field": "total_pnl", "direction": "DESC"}],
        "limit": 100, "time_range": {"start": "2024-01-01", "end": "2024-01-31"},
        "output_format": "table", "needs_join": False, "join_hint": "",
    })

    query_plan = json.dumps({
        "strategy": "single_query",
        "steps": [{"step": 1, "description": "Aggregate PnL by desk", "depends_on": [], "type": "main"}],
        "optimization_hints": ["PREWHERE on trade_date"],
        "estimated_complexity": "low", "needs_final": False,
        "use_prewhere_columns": ["trade_date"], "suggested_settings": {},
    })

    built_sql = json.dumps({
        "sql": "SELECT desk, SUM(pnl) AS total_pnl FROM trades PREWHERE trade_date >= '2024-01-01' GROUP BY desk ORDER BY total_pnl DESC LIMIT 100",
        "parameters": {}, "explanation": "Simple aggregation", "estimated_scan": "~500MB",
    })

    validation = json.dumps({
        "is_valid": True, "security_passed": True, "performance_score": 9,
        "issues": [], "suggested_fixes": [],
        "approved_sql": "SELECT desk, SUM(pnl) AS total_pnl FROM trades PREWHERE trade_date >= '2024-01-01' GROUP BY desk ORDER BY total_pnl DESC LIMIT 100",
    })

    analysis = json.dumps({
        "narrative": "Desk Alpha leads with $2.3M in PnL for January 2024.",
        "key_findings": ["Alpha is top performer", "3 desks analyzed"],
        "data_summary": {"total_records": 3, "key_metrics": {"total_pnl": 5050000}},
        "visualization_recommendations": [],
        "follow_up_questions": ["Break down by ticker?"],
        "caveats": [], "export_recommendation": "none", "confidence": 0.92,
    })

    merger = json.dumps({
        "answer": "Desk Alpha leads with $2.3M PnL.",
        "data_summary": {"top_desk": "Alpha"},
        "visualizations": [],
        "suggestions": ["Break down by ticker?"],
        "confidence": 0.92,
        "execution_summary": "Processed by test_agent",
    })

    responses = [
        intent, selection,                             # master: classify + select
        trade_context, parsed_intent, query_plan,      # trade sub-agents
        built_sql, validation, analysis,               # trade sub-agents cont
        merger,                                        # master: merge
    ]

    async def _side_effect(*args, **kwargs):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return _make_mock_llm_response(responses[idx])

    llm.ainvoke.side_effect = _side_effect
    return llm


@pytest.fixture
def mock_services_full(mock_llm_with_responses, mock_ch_service, mock_cache):
    llm_svc = MagicMock()
    llm_svc.get_model.return_value = mock_llm_with_responses
    llm_svc.primary = mock_llm_with_responses
    llm_svc.fast = mock_llm_with_responses

    return {
        "clickhouse": mock_ch_service,
        "llm": llm_svc,
        "cache": mock_cache,
        "ch_tools": [],
        "plotting_tools": [],
        "email_tools": [],
        "export_tools": [],
    }


class TestEndToEndFlow:
    @pytest.mark.asyncio
    async def test_full_trade_query_pipeline(self, mock_services_full, test_settings):
        """Test the complete flow: classify -> select -> execute trade agent -> merge."""
        from agents.trade.graph import build_trade_agent_graph
        from agents.master_agent import build_master_graph

        # Register a test trade agent
        desc = AgentDescriptor(
            agent_id="test_agent",
            name="Test Trade Agent",
            description="Test agent for integration tests",
            capabilities=[AgentCapability.SQL_QUERY],
            domain_keywords=["trade", "PnL"],
            sub_agents=["trade_analyst"],
            priority=10,
        )
        AgentRegistry.register(desc, build_trade_agent_graph)

        # Build and invoke the master graph
        master = build_master_graph(settings=test_settings, services=mock_services_full)

        input_state = {
            "user_query": "What was the total PnL by desk last month?",
            "user_context": {"history": []},
            "agent_results": {},
            "execution_trace": [],
            "artifacts": [],
            "error": None,
        }

        result = await master.ainvoke(input_state)

        # Verify the pipeline completed
        assert result.get("final_response") is not None
        assert result["final_response"].get("answer")
        assert len(result.get("execution_trace", [])) >= 4  # classify + select + execute + merge
        assert result.get("error") is None
