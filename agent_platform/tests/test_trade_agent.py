"""
Tests for Trade Agent sub-nodes and graph.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestTradeAnalyst:
    @pytest.mark.asyncio
    async def test_produces_domain_context(self, mock_llm, mock_llm_response):
        from agents.trade.nodes.trade_analyst import trade_analyst

        context_json = json.dumps({
            "asset_class": "equities",
            "lifecycle_stage": "post-trade",
            "relevant_metrics": ["PnL", "volume"],
            "time_granularity": "daily",
            "domain_notes": "Standard equity PnL analysis",
            "suggested_tables": ["trades"],
            "special_considerations": "",
        })
        mock_llm.ainvoke.return_value = mock_llm_response(context_json)

        state = {
            "user_query": "Show PnL by desk",
            "intent_analysis": {"primary_domain": "trade"},
        }
        result = await trade_analyst(state, llm=mock_llm)

        assert result["trade_context"]["asset_class"] == "equities"
        assert "PnL" in result["trade_context"]["relevant_metrics"]
        assert result["execution_trace"][0]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_handles_llm_failure(self, mock_llm):
        from agents.trade.nodes.trade_analyst import trade_analyst

        mock_llm.ainvoke.side_effect = Exception("timeout")
        state = {"user_query": "test", "intent_analysis": {}}
        result = await trade_analyst(state, llm=mock_llm)

        assert result["trade_context"]["asset_class"] == "unknown"
        assert result["execution_trace"][0]["status"] == "error"


class TestQueryValidator:
    @pytest.mark.asyncio
    async def test_blocks_unsafe_sql(self, mock_llm, mock_ch_service):
        from agents.trade.nodes.query_validator import query_validator

        mock_ch_service.validate_sql_safety.return_value = (
            False, ["Blocked keyword detected: DROP"]
        )

        state = {
            "user_query": "test",
            "generated_sql": "DROP TABLE trades",
            "schema_info": {},
            "sql_parameters": {},
        }
        result = await query_validator(state, llm=mock_llm, ch_service=mock_ch_service)

        assert result["validation_result"]["is_valid"] is False
        assert result["validation_result"]["security_passed"] is False
        assert result["needs_retry"] is True

    @pytest.mark.asyncio
    async def test_passes_safe_sql(self, mock_llm, mock_llm_response, mock_ch_service):
        from agents.trade.nodes.query_validator import query_validator

        mock_ch_service.validate_sql_safety.return_value = (True, [])

        validation_json = json.dumps({
            "is_valid": True,
            "security_passed": True,
            "performance_score": 9,
            "issues": [],
            "suggested_fixes": [],
            "approved_sql": "SELECT desk, SUM(pnl) FROM trades GROUP BY desk LIMIT 100",
        })
        mock_llm.ainvoke.return_value = mock_llm_response(validation_json)

        state = {
            "user_query": "PnL by desk",
            "generated_sql": "SELECT desk, SUM(pnl) FROM trades GROUP BY desk LIMIT 100",
            "schema_info": {},
            "sql_parameters": {},
        }
        result = await query_validator(state, llm=mock_llm, ch_service=mock_ch_service)

        assert result["validation_result"]["is_valid"] is True
        assert result["needs_retry"] is False


class TestQueryExecutor:
    @pytest.mark.asyncio
    async def test_skips_on_invalid_validation(self, mock_ch_service):
        from agents.trade.nodes.query_executor import query_executor

        state = {
            "generated_sql": "SELECT 1 LIMIT 1",
            "validation_result": {"is_valid": False, "approved_sql": ""},
            "sql_parameters": {},
            "retry_count": 0,
        }
        result = await query_executor(state, ch_service=mock_ch_service)

        assert result["query_results"]["success"] is False
        assert "validation" in result["query_results"]["error"].lower()
        mock_ch_service.execute_query.assert_not_called()

    @pytest.mark.asyncio
    async def test_executes_valid_query(self, mock_ch_service):
        from agents.trade.nodes.query_executor import query_executor

        state = {
            "generated_sql": "SELECT desk, SUM(pnl) FROM trades GROUP BY desk LIMIT 100",
            "validation_result": {
                "is_valid": True,
                "approved_sql": "SELECT desk, SUM(pnl) FROM trades GROUP BY desk LIMIT 100",
            },
            "sql_parameters": {},
            "retry_count": 0,
        }
        result = await query_executor(state, ch_service=mock_ch_service)

        assert result["query_results"]["success"] is True
        assert result["query_results"]["row_count"] == 3
        mock_ch_service.execute_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_execution_error(self, mock_ch_service):
        from agents.trade.nodes.query_executor import query_executor

        mock_ch_service.execute_query.return_value = {
            "success": False,
            "data": [],
            "columns": [],
            "row_count": 0,
            "execution_time_ms": 100,
            "bytes_read": 0,
            "truncated": False,
            "error": "Code: 62. DB::Exception: Syntax error",
        }

        state = {
            "generated_sql": "SELEC bad_sql LIMIT 10",
            "validation_result": {
                "is_valid": True,
                "approved_sql": "SELEC bad_sql LIMIT 10",
            },
            "sql_parameters": {},
            "retry_count": 0,
        }
        result = await query_executor(state, ch_service=mock_ch_service)

        assert result["query_results"]["success"] is False
        assert result["needs_retry"] is True
        assert result["retry_count"] == 1


class TestDetailsAnalyzer:
    @pytest.mark.asyncio
    async def test_produces_narrative(self, mock_llm, mock_llm_response):
        from agents.trade.nodes.details_analyzer import details_analyzer

        analysis_json = json.dumps({
            "narrative": "Desk Alpha leads with $2.3M PnL.",
            "key_findings": ["Alpha is top performer", "Gamma underperforming"],
            "data_summary": {"total_records": 3},
            "visualization_recommendations": [],
            "follow_up_questions": ["Drill into Alpha?"],
            "caveats": [],
            "export_recommendation": "none",
            "confidence": 0.9,
        })
        mock_llm.ainvoke.return_value = mock_llm_response(analysis_json)

        state = {
            "user_query": "PnL by desk",
            "intent_analysis": {"desired_output": "table"},
            "query_results": {
                "data": [{"desk": "Alpha", "pnl": 2300000}],
                "columns": [{"name": "desk", "type": "String"}],
                "row_count": 1,
                "execution_time_ms": 45,
            },
            "generated_sql": "SELECT desk, SUM(pnl) FROM trades GROUP BY desk LIMIT 10",
            "trade_context": {},
        }
        result = await details_analyzer(state, llm=mock_llm)

        assert "Alpha" in result["analysis"]["narrative"]
        assert len(result["analysis"]["key_findings"]) >= 1
        assert result["execution_trace"][0]["status"] == "completed"
