"""
Pytest fixtures for agent platform tests.

Provides mock services, test clients, and sample data.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures: Settings
# ---------------------------------------------------------------------------


@pytest.fixture
def test_settings():
    """Settings configured for testing (no real external connections)."""
    os.environ["AGENT_LLM_API_KEY"] = "test-key"
    os.environ["AGENT_PG_HOST"] = "localhost"
    os.environ["AGENT_ENVIRONMENT"] = "development"
    os.environ["AGENT_API_KEY"] = ""  # disable auth for tests

    from config import Settings
    return Settings(
        llm_api_key="test-key",
        ch_host="localhost",
        ch_database="test_db",
        pg_host="localhost",
        pg_database="test_agent_platform",
        redis_url="redis://localhost:6379/1",
        api_key="",
        mcp_enabled=False,
        artifact_dir="/tmp/test_agent_artifacts",
        allowed_ch_databases=["test_db", "trading_db"],
    )


# ---------------------------------------------------------------------------
# Fixtures: Mock Services
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_response():
    """Factory for creating mock LLM responses."""
    def _make(content: str):
        mock = MagicMock()
        mock.content = content
        return mock
    return _make


@pytest.fixture
def mock_llm(mock_llm_response):
    """Mock LLM that returns configurable JSON responses."""
    llm = AsyncMock()

    # Default: intent classification response
    default_intent = json.dumps({
        "primary_domain": "trade",
        "intent": "query_data",
        "entities": ["trades", "PnL"],
        "desired_output": "table",
        "complexity": "simple",
        "requires_multi_agent": False,
        "sub_intents": [],
        "time_range": "last 30 days",
        "filters": {},
        "ambiguity_notes": "",
    })
    llm.ainvoke.return_value = mock_llm_response(default_intent)
    return llm


@pytest.fixture
def mock_ch_service():
    """Mock ClickHouse service."""
    ch = MagicMock()
    ch.ping.return_value = True

    ch.validate_sql_safety.return_value = (True, [])

    ch.execute_query.return_value = {
        "success": True,
        "data": [
            {"desk": "Alpha", "total_pnl": 2300000},
            {"desk": "Beta", "total_pnl": 1800000},
            {"desk": "Gamma", "total_pnl": 950000},
        ],
        "columns": [
            {"name": "desk", "type": "String"},
            {"name": "total_pnl", "type": "Float64"},
        ],
        "row_count": 3,
        "execution_time_ms": 45.2,
        "bytes_read": 1024000,
        "truncated": False,
        "error": None,
    }

    ch.get_tables.return_value = [
        {
            "database": "trading_db",
            "name": "trades",
            "engine": "MergeTree",
            "partition_key": "toYYYYMM(trade_date)",
            "sorting_key": "trade_date, ticker",
            "total_rows": 150000000,
            "readable_size": "12.5 GiB",
        }
    ]

    ch.get_columns.return_value = [
        {"name": "trade_id", "type": "UInt64", "comment": "Unique trade ID",
         "is_in_partition_key": 0, "is_in_sorting_key": 0},
        {"name": "trade_date", "type": "Date", "comment": "Trade date",
         "is_in_partition_key": 1, "is_in_sorting_key": 1},
        {"name": "ticker", "type": "String", "comment": "Ticker symbol",
         "is_in_partition_key": 0, "is_in_sorting_key": 1},
        {"name": "desk", "type": "String", "comment": "Trading desk",
         "is_in_partition_key": 0, "is_in_sorting_key": 0},
        {"name": "pnl", "type": "Float64", "comment": "Profit and loss",
         "is_in_partition_key": 0, "is_in_sorting_key": 0},
        {"name": "volume", "type": "UInt64", "comment": "Trade volume",
         "is_in_partition_key": 0, "is_in_sorting_key": 0},
        {"name": "counterparty", "type": "String", "comment": "Counterparty name",
         "is_in_partition_key": 0, "is_in_sorting_key": 0},
    ]

    ch.get_full_schema_context.return_value = {
        "database": "trading_db",
        "tables": {
            "trades": {
                "engine": "MergeTree",
                "partition_key": "toYYYYMM(trade_date)",
                "sorting_key": "trade_date, ticker",
                "total_rows": 150000000,
                "columns": ch.get_columns.return_value,
            }
        },
    }

    return ch


@pytest.fixture
def mock_session_manager():
    """Mock PostgreSQL session manager."""
    sm = AsyncMock()
    sm.get_or_create_session.return_value = "test-session-123"
    sm.get_history.return_value = []
    sm.get_active_run_count.return_value = 0
    sm.create_run.return_value = None
    sm.complete_run.return_value = None
    sm.add_message.return_value = None
    sm.get_run.return_value = {
        "run_id": "test-run-456",
        "session_id": "test-session-123",
        "status": "completed",
        "user_query": "test query",
        "result": {"answer": "test answer"},
        "timing_ms": 100.0,
        "error": None,
        "created_at": None,
        "completed_at": None,
    }
    return sm


@pytest.fixture
def mock_cache():
    """Mock Redis cache service."""
    cache = AsyncMock()
    cache.available = True
    cache.ping.return_value = True
    cache.get.return_value = None
    cache.set.return_value = None
    cache.get_schema.return_value = None
    cache.is_duplicate_run.return_value = False
    return cache


@pytest.fixture
def mock_llm_service(mock_llm):
    """Mock LLM service."""
    svc = MagicMock()
    svc.get_model.return_value = mock_llm
    svc.primary = mock_llm
    svc.fast = mock_llm
    return svc


@pytest.fixture
def mock_services(
    mock_ch_service,
    mock_llm_service,
    mock_cache,
    mock_session_manager,
):
    """Complete mock services dict."""
    return {
        "clickhouse": mock_ch_service,
        "llm": mock_llm_service,
        "cache": mock_cache,
        "session_manager": mock_session_manager,
        "ch_tools": [],
        "plotting_tools": [],
        "email_tools": [],
        "export_tools": [],
    }


# ---------------------------------------------------------------------------
# Fixtures: Sample Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_trade_data():
    return [
        {"trade_id": 1, "desk": "Alpha", "ticker": "AAPL", "pnl": 50000,
         "volume": 1000, "trade_date": "2024-01-15"},
        {"trade_id": 2, "desk": "Alpha", "ticker": "GOOGL", "pnl": -12000,
         "volume": 500, "trade_date": "2024-01-15"},
        {"trade_id": 3, "desk": "Beta", "ticker": "MSFT", "pnl": 78000,
         "volume": 2000, "trade_date": "2024-01-16"},
    ]


@pytest.fixture
def sample_schema_info():
    return {
        "database": "trading_db",
        "tables": {
            "trades": {
                "engine": "MergeTree",
                "partition_key": "toYYYYMM(trade_date)",
                "sorting_key": "trade_date, ticker",
                "total_rows": 150000000,
                "readable_size": "12.5 GiB",
                "columns": [
                    {"name": "trade_id", "type": "UInt64"},
                    {"name": "trade_date", "type": "Date"},
                    {"name": "ticker", "type": "String"},
                    {"name": "desk", "type": "String"},
                    {"name": "pnl", "type": "Float64"},
                    {"name": "volume", "type": "UInt64"},
                ],
            }
        },
    }
