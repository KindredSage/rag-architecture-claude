"""
tests/test_pipeline.py
----------------------
Integration tests for the full Master → Trade pipeline.
Uses unittest.mock to patch LLM calls so no real API key is required.

Run:
    python -m pytest tests/test_pipeline.py -v
or directly:
    python tests/test_pipeline.py
"""

import asyncio
import json
import sys
import os
import unittest
from unittest.mock import AsyncMock, patch, MagicMock

# ── Ensure project root is on PYTHONPATH ─────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm_response(content: str):
    """Build a minimal mock that looks like a LangChain AIMessage."""
    msg = MagicMock()
    msg.content = content
    return msg


def _ainvoke_factory(content: str):
    """Return an async callable that always returns a mock AIMessage."""
    async def _ainvoke(messages):
        return _make_llm_response(content)
    return _ainvoke


# ─────────────────────────────────────────────────────────────────────────────
# Test: Agent Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentRegistry(unittest.TestCase):

    def setUp(self):
        # Fresh registry for each test
        from app.registry.agent_registry import AgentRegistry
        AgentRegistry._instance = None

    def test_register_and_retrieve(self):
        from app.registry.agent_registry import registry
        from app.agents.trade.agent import TradeAgent
        registry.register(TradeAgent())
        self.assertIn("trade_agent", registry)
        self.assertEqual(len(registry), 1)

    def test_describe_all(self):
        from app.registry.agent_registry import registry
        from app.agents.trade.agent import TradeAgent
        registry.register(TradeAgent())
        descriptions = registry.describe_all()
        self.assertEqual(len(descriptions), 1)
        self.assertIn("capabilities", descriptions[0])

    def test_unregister(self):
        from app.registry.agent_registry import registry
        from app.agents.trade.agent import TradeAgent
        registry.register(TradeAgent())
        registry.unregister("trade_agent")
        self.assertNotIn("trade_agent", registry)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Trade Agent nodes in isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestTradeAgentNodes(unittest.IsolatedAsyncioTestCase):

    async def test_query_analyst_extracts_entities(self):
        """QueryAnalyst should parse table_name and file_id from the query."""
        from app.agents.trade.nodes import query_analyst

        mock_response = json.dumps({
            "intent": "retrieve",
            "table_name": "My_Table",
            "file_id": "999",
            "filters": {},
            "columns_requested": ["*"],
            "requires_join": False,
            "summary": "Get trade details for file id 999 from My_Table",
        })

        with patch("app.agents.trade.nodes.ChatOpenAI") as MockLLM:
            instance = MagicMock()
            instance.ainvoke = _ainvoke_factory(mock_response)
            MockLLM.return_value = instance

            state = {
                "query": "Give me trade details for file id 999 from My_Table",
                "context": {},
                "query_analysis": {}, "table_name": None, "file_id": None,
                "filters": {}, "execution_plan": [], "schema_info": {},
                "raw_query": "", "validated_query": "", "validation_errors": [],
                "query_result": {}, "row_count": 0, "result_analysis": "",
                "answer": "", "confidence": 0.0, "steps": [], "error": None,
            }

            result = await query_analyst(state)

        self.assertEqual(result["table_name"], "My_Table")
        self.assertEqual(result["file_id"], "999")
        self.assertEqual(result["query_analysis"]["intent"], "retrieve")

    async def test_schema_analyzer_returns_columns(self):
        """SchemaAnalyzer should return My_Table columns without any LLM call."""
        from app.agents.trade.nodes import schema_analyzer

        state = {"table_name": "My_Table", "steps": []}
        result = await schema_analyzer(state)

        self.assertIn("columns", result["schema_info"])
        self.assertIn("file_id", result["schema_info"]["columns"])

    async def test_query_validator_rejects_delete(self):
        """Validator must block DELETE statements."""
        from app.agents.trade.nodes import query_validator

        # Patch LLM to avoid real API call for the structural check
        with patch("app.agents.trade.nodes.ChatOpenAI") as MockLLM:
            instance = MagicMock()
            instance.ainvoke = _ainvoke_factory(
                json.dumps({"valid": True, "issues": [], "corrected_sql": ""})
            )
            MockLLM.return_value = instance

            state = {
                "raw_query": "DELETE FROM My_Table WHERE file_id = 999",
                "steps": [], "validation_errors": [],
            }
            result = await query_validator(state)

        # Should have flagged the dangerous keyword before LLM check
        self.assertTrue(any("dangerous" in e.lower() for e in result["validation_errors"]))

    async def test_query_executor_returns_mock_row(self):
        """Executor should return a simulated row for any valid query."""
        from app.agents.trade.nodes import query_executor

        state = {
            "validated_query": "SELECT * FROM My_Table WHERE file_id = :file_id",
            "file_id": "999",
            "steps": [],
        }
        result = await query_executor(state)

        self.assertEqual(result["row_count"], 1)
        rows = result["query_result"]["rows"]
        self.assertEqual(rows[0]["file_id"], 999)
        self.assertEqual(rows[0]["status"], "SETTLED")


# ─────────────────────────────────────────────────────────────────────────────
# Test: Master Agent nodes in isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestMasterAgentNodes(unittest.IsolatedAsyncioTestCase):

    async def test_analyze_intent_returns_string(self):
        from app.agents.master.nodes import analyze_intent

        with patch("app.agents.master.nodes.ChatOpenAI") as MockLLM:
            instance = MagicMock()
            instance.ainvoke = _ainvoke_factory("retrieve trade details")
            MockLLM.return_value = instance

            state = {
                "query": "Give me trade details for file id 999 from My_Table",
                "context": {}, "intent": "", "selected_agent": None,
                "routing_reason": "", "plan": [], "intermediate_results": {},
                "agent_raw_output": {}, "final_answer": "", "confidence": 0.0,
                "trace": [], "error": None,
            }
            result = await analyze_intent(state)

        self.assertEqual(result["intent"], "retrieve trade details")

    async def test_select_agent_routes_to_trade_agent(self):
        from app.agents.master.nodes import select_agent
        from app.registry.agent_registry import AgentRegistry
        from app.agents.trade.agent import TradeAgent

        # Ensure registry has trade_agent
        AgentRegistry._instance = None
        from app.registry.agent_registry import registry
        registry.register(TradeAgent())

        routing_json = json.dumps({
            "selected_agent": "trade_agent",
            "reason": "Query asks for trade data retrieval.",
            "confidence": 0.97,
        })

        with patch("app.agents.master.nodes.ChatOpenAI") as MockLLM:
            instance = MagicMock()
            instance.ainvoke = _ainvoke_factory(routing_json)
            MockLLM.return_value = instance

            state = {
                "query": "Give me trade details for file id 999 from My_Table",
                "context": {}, "intent": "retrieve trade details",
                "selected_agent": None, "routing_reason": "", "plan": [],
                "intermediate_results": {}, "agent_raw_output": {},
                "final_answer": "", "confidence": 0.0, "trace": [], "error": None,
            }
            result = await select_agent(state)

        self.assertEqual(result["selected_agent"], "trade_agent")
        self.assertAlmostEqual(result["confidence"], 0.97)

    async def test_select_agent_falls_back_to_direct_for_general_query(self):
        from app.agents.master.nodes import select_agent
        from app.registry.agent_registry import AgentRegistry
        from app.agents.trade.agent import TradeAgent

        AgentRegistry._instance = None
        from app.registry.agent_registry import registry
        registry.register(TradeAgent())

        routing_json = json.dumps({
            "selected_agent": "none",
            "reason": "This is a general question with no specific agent match.",
            "confidence": 0.85,
        })

        with patch("app.agents.master.nodes.ChatOpenAI") as MockLLM:
            instance = MagicMock()
            instance.ainvoke = _ainvoke_factory(routing_json)
            MockLLM.return_value = instance

            state = {
                "query": "What is the capital of France?",
                "context": {}, "intent": "answer general question",
                "selected_agent": None, "routing_reason": "", "plan": [],
                "intermediate_results": {}, "agent_raw_output": {},
                "final_answer": "", "confidence": 0.0, "trace": [], "error": None,
            }
            result = await select_agent(state)

        # "none" should translate to None (direct answer path)
        self.assertIsNone(result["selected_agent"])


# ─────────────────────────────────────────────────────────────────────────────
# Test: Full End-to-End pipeline (mocked LLM)
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd(unittest.IsolatedAsyncioTestCase):
    """
    Full pipeline test: FastAPI → MasterGraph → TradeAgent → response.
    All LLM calls are mocked; only the graph wiring and state transitions
    are exercised.
    """

    def _patch_llm(self, responses: list[str]):
        """
        Returns a context manager that patches ChatOpenAI.ainvoke to cycle
        through the provided list of response strings.
        """
        call_count = {"n": 0}
        responses_cycle = responses

        async def _ainvoke(messages):
            idx = min(call_count["n"], len(responses_cycle) - 1)
            call_count["n"] += 1
            return _make_llm_response(responses_cycle[idx])

        mock_instance = MagicMock()
        mock_instance.ainvoke = _ainvoke
        return mock_instance

    async def test_trade_query_full_pipeline(self):
        """
        Simulates: 'Give me trade details for file id 999 from My_Table'
        flowing through the full Master + Trade graph with mocked LLMs.
        """
        from app.registry.agent_registry import AgentRegistry
        AgentRegistry._instance = None
        from app.registry.agent_registry import registry
        from app.agents.trade.agent import TradeAgent
        registry.register(TradeAgent())

        # ── Prepare ordered mock responses for every LLM call ─────────────────
        # Master nodes: intent → select
        # Trade nodes: query_analyst → query_planner → query_builder → query_validator → details_analyzer → result_synthesizer
        # Master: synthesize_response
        llm_responses = [
            # 1. Master: analyze_intent
            "retrieve trade details",
            # 2. Master: select_agent
            json.dumps({"selected_agent": "trade_agent", "reason": "trade data query", "confidence": 0.97}),
            # 3. Trade: query_analyst
            json.dumps({"intent": "retrieve", "table_name": "My_Table", "file_id": "999",
                        "filters": {}, "columns_requested": ["*"], "requires_join": False,
                        "summary": "Get trade details for file 999"}),
            # 4. Trade: query_planner
            json.dumps({"plan": ["Fetch schema", "Build query", "Execute", "Analyze results"]}),
            # 5. Trade: query_builder
            json.dumps({"sql": "SELECT * FROM My_Table WHERE file_id = :file_id LIMIT 100",
                        "params": {"file_id": 999}}),
            # 6. Trade: query_validator (LLM structural check)
            json.dumps({"valid": True, "issues": [], "corrected_sql": ""}),
            # 7. Trade: details_analyzer
            "The trade for file 999 is a SETTLED USD/EUR FX Forward with notional EUR 1,085,200.",
            # 8. Trade: result_synthesizer
            "Trade file 999 corresponds to a settled USD/EUR FX Forward...",
            # 9. Master: synthesize_response
            "Here are the trade details for file ID 999 from My_Table:\n\n"
            "- **Trade ID**: TRD-20241115-0042\n"
            "- **Instrument**: USD/EUR FX Forward\n"
            "- **Notional**: EUR 1,085,200.00\n"
            "- **Status**: SETTLED\n"
            "- **Counterparty**: DEUTSCHE BANK AG\n"
            "- **Trade Date**: 2024-11-15",
        ]

        call_idx = {"n": 0}

        async def _mock_ainvoke(messages):
            idx = min(call_idx["n"], len(llm_responses) - 1)
            call_idx["n"] += 1
            return _make_llm_response(llm_responses[idx])

        mock_llm_instance = MagicMock()
        mock_llm_instance.ainvoke = _mock_ainvoke

        with patch("app.agents.master.nodes.ChatOpenAI", return_value=mock_llm_instance), \
             patch("app.agents.trade.nodes.ChatOpenAI", return_value=mock_llm_instance):

            from app.agents.master.graph import master_graph

            initial_state = {
                "query": "Give me trade details for file id 999 from My_Table",
                "context": {},
                "intent": "",
                "selected_agent": None,
                "routing_reason": "",
                "plan": [],
                "intermediate_results": {},
                "agent_raw_output": {},
                "final_answer": "",
                "confidence": 0.0,
                "trace": [],
                "error": None,
            }

            final_state = await master_graph.ainvoke(initial_state)

        # ── Assertions ─────────────────────────────────────────────────────────
        self.assertEqual(final_state["selected_agent"], "trade_agent")
        self.assertIn("999", final_state["final_answer"] or "")
        self.assertGreater(final_state["confidence"], 0.5)
        self.assertGreater(len(final_state["trace"]), 3,
                           "Expected at least 4 trace steps")

        print("\n✅ End-to-end test passed!")
        print(f"   Agent used    : {final_state['selected_agent']}")
        print(f"   Confidence    : {final_state['confidence']:.2f}")
        print(f"   Trace steps   : {len(final_state['trace'])}")
        print(f"   Answer snippet: {final_state['final_answer'][:100]}...")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
