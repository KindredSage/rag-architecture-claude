"""
Human-in-the-Loop (HITL) Service.

Manages interrupt points where agent execution pauses and waits for human input.

Architecture:
  1. Agent node calls hitl.create_interrupt() with context (SQL, email, question)
  2. Run status is set to 'waiting_human'
  3. Graph state is checkpointed to PostgreSQL
  4. User sees pending interrupt via GET /interrupts/{run_id}
  5. User resolves via POST /interrupts/{interrupt_id}/resolve
  6. Graph resumes from checkpoint with user's input merged into state

PostgreSQL stores all interrupt state, so any server can handle the resolution.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg

from config import Settings
from models import (
    HITLConfig,
    InterruptInfo,
    InterruptRequest,
    InterruptResponse,
    InterruptStatus,
    InterruptType,
)

logger = logging.getLogger(__name__)


class HITLService:
    """PostgreSQL-backed interrupt queue for human-in-the-loop flows."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    async def ensure_tables(self) -> None:
        """Create the interrupts table if it doesn't exist."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_interrupts (
                    interrupt_id        TEXT PRIMARY KEY,
                    run_id              TEXT NOT NULL,
                    session_id          TEXT NOT NULL,
                    interrupt_type      TEXT NOT NULL,
                    status              TEXT NOT NULL DEFAULT 'pending',
                    node_name           TEXT NOT NULL,
                    agent_id            TEXT NOT NULL,
                    title               TEXT NOT NULL,
                    description         TEXT NOT NULL DEFAULT '',
                    payload             JSONB NOT NULL DEFAULT '{}',
                    resolution          JSONB,
                    auto_approve_seconds INTEGER,
                    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    resolved_at         TIMESTAMPTZ,
                    resolved_by         TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_interrupts_run
                    ON agent_interrupts (run_id) WHERE status = 'pending';
                CREATE INDEX IF NOT EXISTS idx_interrupts_session
                    ON agent_interrupts (session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_interrupts_auto_approve
                    ON agent_interrupts (created_at)
                    WHERE status = 'pending' AND auto_approve_seconds IS NOT NULL;
            """)
        logger.info("HITL tables ensured")

    # ── Create Interrupt ─────────────────────────────────────────

    async def create_interrupt(
        self,
        request: InterruptRequest,
    ) -> InterruptInfo:
        """
        Create a pending interrupt. Called by agent nodes when they
        need human input before proceeding.
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_interrupts
                    (interrupt_id, run_id, session_id, interrupt_type, status,
                     node_name, agent_id, title, description, payload,
                     auto_approve_seconds)
                VALUES ($1, $2, $3, $4, 'pending', $5, $6, $7, $8, $9, $10)
                """,
                request.interrupt_id,
                request.run_id,
                request.session_id,
                request.interrupt_type.value,
                request.node_name,
                request.agent_id,
                request.title,
                request.description,
                json.dumps(request.payload, default=str),
                request.auto_approve_seconds,
            )

            # Update run status to waiting_human
            await conn.execute(
                "UPDATE agent_runs SET status = 'waiting_human' WHERE run_id = $1",
                request.run_id,
            )

        logger.info(
            "Interrupt created: %s (type=%s, node=%s, run=%s)",
            request.interrupt_id,
            request.interrupt_type.value,
            request.node_name,
            request.run_id,
        )

        return InterruptInfo(
            interrupt_id=request.interrupt_id,
            run_id=request.run_id,
            session_id=request.session_id,
            interrupt_type=request.interrupt_type,
            status=InterruptStatus.PENDING,
            node_name=request.node_name,
            agent_id=request.agent_id,
            title=request.title,
            description=request.description,
            payload=request.payload,
            auto_approve_seconds=request.auto_approve_seconds,
            created_at=request.created_at,
        )

    # ── Resolve Interrupt ────────────────────────────────────────

    async def resolve_interrupt(
        self,
        response: InterruptResponse,
        resolved_by: str = "user",
    ) -> InterruptInfo | None:
        """
        Resolve a pending interrupt with the user's decision.
        Returns the updated interrupt info, or None if not found/already resolved.
        """
        async with self._pool.acquire() as conn:
            # Check it exists and is pending
            row = await conn.fetchrow(
                "SELECT * FROM agent_interrupts WHERE interrupt_id = $1",
                response.interrupt_id,
            )
            if not row:
                return None
            if row["status"] != "pending":
                logger.warning(
                    "Interrupt %s already resolved (status=%s)",
                    response.interrupt_id, row["status"],
                )
                return self._row_to_info(row)

            # Resolve
            resolution = {
                "action": response.action.value,
                "modifications": response.modifications,
                "comment": response.comment,
            }
            await conn.execute(
                """
                UPDATE agent_interrupts
                SET status = $2, resolution = $3, resolved_at = NOW(), resolved_by = $4
                WHERE interrupt_id = $1
                """,
                response.interrupt_id,
                response.action.value,
                json.dumps(resolution, default=str),
                resolved_by,
            )

            # If rejected, update run status to cancelled
            if response.action == InterruptStatus.REJECTED:
                await conn.execute(
                    "UPDATE agent_runs SET status = 'cancelled', error = 'Rejected by user' WHERE run_id = $1",
                    row["run_id"],
                )

            # Fetch updated
            updated = await conn.fetchrow(
                "SELECT * FROM agent_interrupts WHERE interrupt_id = $1",
                response.interrupt_id,
            )

        logger.info(
            "Interrupt resolved: %s -> %s (by %s)",
            response.interrupt_id, response.action.value, resolved_by,
        )
        return self._row_to_info(updated) if updated else None

    # ── Query Interrupts ─────────────────────────────────────────

    async def get_interrupt(self, interrupt_id: str) -> InterruptInfo | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM agent_interrupts WHERE interrupt_id = $1",
                interrupt_id,
            )
            return self._row_to_info(row) if row else None

    async def get_pending_for_run(self, run_id: str) -> list[InterruptInfo]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM agent_interrupts
                   WHERE run_id = $1 AND status = 'pending'
                   ORDER BY created_at""",
                run_id,
            )
            return [self._row_to_info(r) for r in rows]

    async def get_pending_for_session(self, session_id: str) -> list[InterruptInfo]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM agent_interrupts
                   WHERE session_id = $1 AND status = 'pending'
                   ORDER BY created_at DESC""",
                session_id,
            )
            return [self._row_to_info(r) for r in rows]

    async def get_all_for_run(self, run_id: str) -> list[InterruptInfo]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT * FROM agent_interrupts
                   WHERE run_id = $1 ORDER BY created_at""",
                run_id,
            )
            return [self._row_to_info(r) for r in rows]

    # ── Auto-Approve Expired ─────────────────────────────────────

    async def auto_approve_expired(self) -> int:
        """Auto-approve interrupts whose timeout has elapsed."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT interrupt_id, run_id, auto_approve_seconds, created_at
                FROM agent_interrupts
                WHERE status = 'pending' AND auto_approve_seconds IS NOT NULL
                """,
            )

            count = 0
            now = datetime.now(timezone.utc)
            for row in rows:
                created = row["created_at"]
                if created.tzinfo is None:
                    created = created.replace(tzinfo=timezone.utc)
                deadline = created + timedelta(seconds=row["auto_approve_seconds"])
                if now >= deadline:
                    await conn.execute(
                        """
                        UPDATE agent_interrupts
                        SET status = 'auto_approved',
                            resolution = '{"action": "auto_approved", "reason": "timeout"}',
                            resolved_at = NOW(),
                            resolved_by = 'system'
                        WHERE interrupt_id = $1 AND status = 'pending'
                        """,
                        row["interrupt_id"],
                    )
                    count += 1
                    logger.info("Auto-approved interrupt %s", row["interrupt_id"])

            return count

    # ── Wait for Resolution ──────────────────────────────────────

    async def wait_for_resolution(
        self,
        interrupt_id: str,
        timeout: int = 300,
        poll_interval: float = 2.0,
    ) -> InterruptInfo | None:
        """
        Poll until the interrupt is resolved or timeout expires.
        Used internally when the graph needs to block until user responds.

        Returns the resolved InterruptInfo, or None on timeout.
        """
        deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout)

        while datetime.now(timezone.utc) < deadline:
            info = await self.get_interrupt(interrupt_id)
            if info and info.status != InterruptStatus.PENDING:
                return info
            await asyncio.sleep(poll_interval)

        # Timeout: auto-expire
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE agent_interrupts
                SET status = 'expired', resolved_at = NOW(), resolved_by = 'system'
                WHERE interrupt_id = $1 AND status = 'pending'
                """,
                interrupt_id,
            )
        logger.warning("Interrupt %s expired after %ds", interrupt_id, timeout)
        return await self.get_interrupt(interrupt_id)

    # ── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _row_to_info(row: asyncpg.Record) -> InterruptInfo:
        payload = row.get("payload", "{}")
        if isinstance(payload, str):
            payload = json.loads(payload)

        resolution = row.get("resolution")
        if isinstance(resolution, str):
            resolution = json.loads(resolution)

        return InterruptInfo(
            interrupt_id=row["interrupt_id"],
            run_id=row["run_id"],
            session_id=row["session_id"],
            interrupt_type=InterruptType(row["interrupt_type"]),
            status=InterruptStatus(row["status"]),
            node_name=row["node_name"],
            agent_id=row["agent_id"],
            title=row["title"],
            description=row.get("description", ""),
            payload=payload,
            auto_approve_seconds=row.get("auto_approve_seconds"),
            created_at=row["created_at"],
            resolved_at=row.get("resolved_at"),
            resolution=resolution,
        )

    # ── HITL Config Helper ───────────────────────────────────────

    @staticmethod
    def should_interrupt(
        config: HITLConfig,
        interrupt_type: InterruptType,
        complexity: str = "simple",
    ) -> bool:
        """
        Determine if an interrupt should be created based on user's HITL config.
        """
        if not config.enabled:
            return False

        # Complexity threshold check
        complexity_order = {"simple": 0, "moderate": 1, "complex": 2}
        threshold = complexity_order.get(config.complexity_threshold, 0)
        current = complexity_order.get(complexity, 0)

        if current < threshold:
            return False

        if interrupt_type == InterruptType.APPROVAL:
            return config.require_sql_approval
        elif interrupt_type == InterruptType.CONFIRMATION:
            return config.require_email_confirmation
        elif interrupt_type == InterruptType.REVIEW:
            return config.require_export_confirmation
        elif interrupt_type == InterruptType.CLARIFICATION:
            return True  # always honor clarification requests
        elif interrupt_type == InterruptType.MODIFICATION:
            return config.require_sql_approval

        return False
