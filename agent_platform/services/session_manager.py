"""
PostgreSQL-backed session manager.

Handles:
- Session creation, lookup, and lifecycle
- Cross-server session continuity (any server can resume any session)
- Conversation history persistence
- Periodic cleanup of expired sessions
- LangGraph checkpoint storage via PostgresSaver

Schema is auto-created on startup via ensure_tables().
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg

from config import Settings

logger = logging.getLogger(__name__)


class SessionManager:
    """Async PostgreSQL session store for multi-server deployments."""

    def __init__(self, settings: Settings, server_id: str):
        self.settings = settings
        self.server_id = server_id
        self._pool: asyncpg.Pool | None = None

    # ── Lifecycle ────────────────────────────────────────────────

    async def initialize(self) -> None:
        self._pool = await asyncpg.create_pool(
            dsn=self.settings.pg_dsn,
            min_size=self.settings.pg_pool_min,
            max_size=self.settings.pg_pool_max,
            command_timeout=30,
        )
        await self.ensure_tables()
        logger.info("SessionManager initialized (server=%s)", self.server_id)

    async def shutdown(self) -> None:
        if self._pool:
            await self._pool.close()
            logger.info("SessionManager pool closed")

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("SessionManager not initialized. Call initialize() first.")
        return self._pool

    # ── Schema ───────────────────────────────────────────────────

    async def ensure_tables(self) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_sessions (
                    session_id      TEXT PRIMARY KEY,
                    user_id         TEXT,
                    server_id       TEXT NOT NULL,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_active     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    metadata        JSONB NOT NULL DEFAULT '{}',
                    is_active       BOOLEAN NOT NULL DEFAULT TRUE
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_user
                    ON agent_sessions (user_id) WHERE is_active = TRUE;
                CREATE INDEX IF NOT EXISTS idx_sessions_active
                    ON agent_sessions (last_active) WHERE is_active = TRUE;
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_messages (
                    id              BIGSERIAL PRIMARY KEY,
                    session_id      TEXT NOT NULL REFERENCES agent_sessions(session_id) ON DELETE CASCADE,
                    run_id          TEXT NOT NULL,
                    role            TEXT NOT NULL,  -- 'user' | 'assistant' | 'system'
                    content         JSONB NOT NULL,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON agent_messages (session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_messages_run
                    ON agent_messages (run_id);
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_runs (
                    run_id          TEXT PRIMARY KEY,
                    session_id      TEXT NOT NULL REFERENCES agent_sessions(session_id) ON DELETE CASCADE,
                    user_query      TEXT NOT NULL,
                    status          TEXT NOT NULL DEFAULT 'pending',
                    selected_agents TEXT[] NOT NULL DEFAULT '{}',
                    result          JSONB,
                    trace           JSONB NOT NULL DEFAULT '[]',
                    artifacts       JSONB NOT NULL DEFAULT '[]',
                    timing_ms       REAL,
                    error           TEXT,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    completed_at    TIMESTAMPTZ
                );

                CREATE INDEX IF NOT EXISTS idx_runs_session
                    ON agent_runs (session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_runs_status
                    ON agent_runs (status) WHERE status IN ('pending', 'running');
            """)
            # LangGraph checkpoint table for cross-server graph state
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS langgraph_checkpoints (
                    thread_id       TEXT NOT NULL,
                    checkpoint_ns   TEXT NOT NULL DEFAULT '',
                    checkpoint_id   TEXT NOT NULL,
                    parent_id       TEXT,
                    checkpoint      JSONB NOT NULL,
                    metadata        JSONB NOT NULL DEFAULT '{}',
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                );

                CREATE INDEX IF NOT EXISTS idx_checkpoints_thread
                    ON langgraph_checkpoints (thread_id, created_at DESC);
            """)
            logger.info("Database tables ensured")

    # ── Session CRUD ─────────────────────────────────────────────

    async def create_session(
        self,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        session_id = str(uuid.uuid4())
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_sessions (session_id, user_id, server_id, metadata)
                VALUES ($1, $2, $3, $4)
                """,
                session_id,
                user_id,
                self.server_id,
                json.dumps(metadata or {}),
            )
        logger.info("Session created: %s (user=%s)", session_id, user_id)
        return session_id

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM agent_sessions WHERE session_id = $1 AND is_active = TRUE",
                session_id,
            )
            if row:
                # Touch last_active and update server_id (session may have migrated)
                await conn.execute(
                    """
                    UPDATE agent_sessions
                    SET last_active = NOW(), server_id = $2
                    WHERE session_id = $1
                    """,
                    session_id,
                    self.server_id,
                )
                return dict(row)
        return None

    async def get_or_create_session(
        self,
        session_id: str | None,
        user_id: str | None = None,
    ) -> str:
        if session_id:
            existing = await self.get_session(session_id)
            if existing:
                return session_id
            logger.warning("Session %s not found or expired, creating new", session_id)
        return await self.create_session(user_id=user_id)

    async def close_session(self, session_id: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                "UPDATE agent_sessions SET is_active = FALSE WHERE session_id = $1",
                session_id,
            )

    # ── Message History ──────────────────────────────────────────

    async def add_message(
        self,
        session_id: str,
        run_id: str,
        role: str,
        content: dict[str, Any],
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_messages (session_id, run_id, role, content)
                VALUES ($1, $2, $3, $4)
                """,
                session_id,
                run_id,
                role,
                json.dumps(content),
            )

    async def get_history(
        self,
        session_id: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, content, created_at
                FROM agent_messages
                WHERE session_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                session_id,
                limit,
            )
            return [
                {
                    "role": r["role"],
                    "content": json.loads(r["content"]),
                    "timestamp": r["created_at"].isoformat(),
                }
                for r in reversed(rows)
            ]

    # ── Run Tracking ─────────────────────────────────────────────

    async def create_run(
        self,
        run_id: str,
        session_id: str,
        user_query: str,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_runs (run_id, session_id, user_query, status)
                VALUES ($1, $2, $3, 'running')
                """,
                run_id,
                session_id,
                user_query,
            )

    async def complete_run(
        self,
        run_id: str,
        status: str,
        result: dict[str, Any] | None = None,
        trace: list[dict] | None = None,
        artifacts: list[dict] | None = None,
        timing_ms: float | None = None,
        error: str | None = None,
        selected_agents: list[str] | None = None,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE agent_runs
                SET status = $2,
                    result = $3,
                    trace = $4,
                    artifacts = $5,
                    timing_ms = $6,
                    error = $7,
                    selected_agents = $8,
                    completed_at = NOW()
                WHERE run_id = $1
                """,
                run_id,
                status,
                json.dumps(result) if result else None,
                json.dumps(trace or []),
                json.dumps(artifacts or []),
                timing_ms,
                error,
                selected_agents or [],
            )

    async def get_run(self, run_id: str) -> dict[str, Any] | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM agent_runs WHERE run_id = $1", run_id
            )
            return dict(row) if row else None

    # ── Cleanup ──────────────────────────────────────────────────

    async def cleanup_expired_sessions(self) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(
            hours=self.settings.session_ttl_hours
        )
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE agent_sessions
                SET is_active = FALSE
                WHERE is_active = TRUE AND last_active < $1
                """,
                cutoff,
            )
            count = int(result.split()[-1])
            if count:
                logger.info("Cleaned up %d expired sessions", count)
            return count

    # ── Stats ────────────────────────────────────────────────────

    async def get_active_run_count(self) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) as cnt FROM agent_runs WHERE status = 'running'"
            )
            return row["cnt"] if row else 0

    async def get_session_count(self) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT COUNT(*) as cnt FROM agent_sessions WHERE is_active = TRUE"
            )
            return row["cnt"] if row else 0
