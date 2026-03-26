"""
Tests for PostgreSQL session manager.

These tests use mock asyncpg since they don't require a live database.
For integration tests with a live PG, see test_integration.py.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_pool():
    pool = AsyncMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=False)
    return pool, conn


class TestSessionManager:
    @pytest.mark.asyncio
    async def test_create_session(self, test_settings, mock_pool):
        from services.session_manager import SessionManager

        pool, conn = mock_pool
        sm = SessionManager(test_settings, "test-server")
        sm._pool = pool

        session_id = await sm.create_session(user_id="user-1")
        assert session_id  # UUID string
        conn.execute.assert_called_once()
        call_args = conn.execute.call_args
        assert "INSERT INTO agent_sessions" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_get_session_updates_server_id(self, test_settings, mock_pool):
        from services.session_manager import SessionManager

        pool, conn = mock_pool
        conn.fetchrow.return_value = {
            "session_id": "sess-123",
            "user_id": "user-1",
            "server_id": "old-server",
            "is_active": True,
        }

        sm = SessionManager(test_settings, "new-server")
        sm._pool = pool

        result = await sm.get_session("sess-123")
        assert result is not None
        # Should update server_id to current server
        update_call = conn.execute.call_args
        assert "new-server" in update_call[0]

    @pytest.mark.asyncio
    async def test_get_or_create_existing(self, test_settings, mock_pool):
        from services.session_manager import SessionManager

        pool, conn = mock_pool
        conn.fetchrow.return_value = {
            "session_id": "existing-sess",
            "user_id": "u1",
            "server_id": "s1",
            "is_active": True,
        }

        sm = SessionManager(test_settings, "test-server")
        sm._pool = pool

        sid = await sm.get_or_create_session("existing-sess", user_id="u1")
        assert sid == "existing-sess"

    @pytest.mark.asyncio
    async def test_get_or_create_new_when_missing(self, test_settings, mock_pool):
        from services.session_manager import SessionManager

        pool, conn = mock_pool
        conn.fetchrow.return_value = None  # session not found

        sm = SessionManager(test_settings, "test-server")
        sm._pool = pool

        sid = await sm.get_or_create_session("missing-sess", user_id="u1")
        assert sid != "missing-sess"  # new session created

    @pytest.mark.asyncio
    async def test_add_and_get_history(self, test_settings, mock_pool):
        from services.session_manager import SessionManager

        pool, conn = mock_pool

        sm = SessionManager(test_settings, "test-server")
        sm._pool = pool

        await sm.add_message("sess-1", "run-1", "user", {"query": "test"})
        conn.execute.assert_called_once()
        assert "INSERT INTO agent_messages" in conn.execute.call_args[0][0]

    @pytest.mark.asyncio
    async def test_run_lifecycle(self, test_settings, mock_pool):
        from services.session_manager import SessionManager

        pool, conn = mock_pool
        sm = SessionManager(test_settings, "test-server")
        sm._pool = pool

        await sm.create_run("run-1", "sess-1", "test query")
        create_call = conn.execute.call_args
        assert "INSERT INTO agent_runs" in create_call[0][0]

        await sm.complete_run(
            run_id="run-1",
            status="completed",
            result={"answer": "test"},
            timing_ms=150.0,
        )
        complete_call = conn.execute.call_args
        assert "UPDATE agent_runs" in complete_call[0][0]

    @pytest.mark.asyncio
    async def test_cleanup_expired(self, test_settings, mock_pool):
        from services.session_manager import SessionManager

        pool, conn = mock_pool
        conn.execute.return_value = "UPDATE 5"

        sm = SessionManager(test_settings, "test-server")
        sm._pool = pool

        count = await sm.cleanup_expired_sessions()
        assert count == 5

    @pytest.mark.asyncio
    async def test_pool_not_initialized_raises(self, test_settings):
        from services.session_manager import SessionManager

        sm = SessionManager(test_settings, "test-server")
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = sm.pool
