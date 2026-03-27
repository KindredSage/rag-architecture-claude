"""
Tests for Human-in-the-Loop service and gate nodes.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from models import (
    HITLConfig,
    InterruptRequest,
    InterruptResponse,
    InterruptStatus,
    InterruptType,
)
from services.hitl import HITLService


# ── HITLConfig Logic Tests ───────────────────────────────────────


class TestHITLConfig:
    def test_disabled_never_interrupts(self):
        config = HITLConfig(enabled=False)
        assert not HITLService.should_interrupt(config, InterruptType.APPROVAL)
        assert not HITLService.should_interrupt(config, InterruptType.CONFIRMATION)
        assert not HITLService.should_interrupt(config, InterruptType.CLARIFICATION)

    def test_enabled_approval_required(self):
        config = HITLConfig(enabled=True, require_sql_approval=True)
        assert HITLService.should_interrupt(config, InterruptType.APPROVAL)

    def test_enabled_approval_not_required(self):
        config = HITLConfig(enabled=True, require_sql_approval=False)
        assert not HITLService.should_interrupt(config, InterruptType.APPROVAL)

    def test_email_confirmation_required(self):
        config = HITLConfig(enabled=True, require_email_confirmation=True)
        assert HITLService.should_interrupt(config, InterruptType.CONFIRMATION)

    def test_email_confirmation_not_required(self):
        config = HITLConfig(enabled=True, require_email_confirmation=False)
        assert not HITLService.should_interrupt(config, InterruptType.CONFIRMATION)

    def test_clarification_always_honored(self):
        config = HITLConfig(enabled=True, require_sql_approval=False)
        assert HITLService.should_interrupt(config, InterruptType.CLARIFICATION)

    def test_complexity_threshold_simple(self):
        config = HITLConfig(enabled=True, complexity_threshold="moderate")
        # Simple query should NOT trigger interrupt
        assert not HITLService.should_interrupt(
            config, InterruptType.APPROVAL, complexity="simple"
        )
        # Moderate query should trigger
        assert HITLService.should_interrupt(
            config, InterruptType.APPROVAL, complexity="moderate"
        )
        # Complex should trigger
        assert HITLService.should_interrupt(
            config, InterruptType.APPROVAL, complexity="complex"
        )

    def test_complexity_threshold_complex_only(self):
        config = HITLConfig(enabled=True, complexity_threshold="complex")
        assert not HITLService.should_interrupt(
            config, InterruptType.APPROVAL, complexity="simple"
        )
        assert not HITLService.should_interrupt(
            config, InterruptType.APPROVAL, complexity="moderate"
        )
        assert HITLService.should_interrupt(
            config, InterruptType.APPROVAL, complexity="complex"
        )

    def test_export_confirmation(self):
        config = HITLConfig(enabled=True, require_export_confirmation=True)
        assert HITLService.should_interrupt(config, InterruptType.REVIEW)

    def test_modification_uses_sql_approval_setting(self):
        config = HITLConfig(enabled=True, require_sql_approval=True)
        assert HITLService.should_interrupt(config, InterruptType.MODIFICATION)

        config2 = HITLConfig(enabled=True, require_sql_approval=False)
        assert not HITLService.should_interrupt(config2, InterruptType.MODIFICATION)


# ── SQL Approval Gate Tests ──────────────────────────────────────


class TestSQLApprovalGate:
    @pytest.mark.asyncio
    async def test_skipped_when_hitl_disabled(self):
        from agents.hitl_gates import sql_approval_gate

        state = {
            "hitl_config": {"enabled": False},
            "generated_sql": "SELECT 1 LIMIT 1",
            "validation_result": {"performance_score": 9},
            "intent_analysis": {},
        }
        result = await sql_approval_gate(
            state,
            hitl_service=MagicMock(),
            run_id="run-1",
            session_id="sess-1",
        )
        assert result["hitl_skipped"] is True
        assert result["execution_trace"][0]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_skipped_below_complexity_threshold(self):
        from agents.hitl_gates import sql_approval_gate

        state = {
            "hitl_config": {"enabled": True, "complexity_threshold": "complex"},
            "generated_sql": "SELECT 1 LIMIT 1",
            "validation_result": {},
            "intent_analysis": {"complexity": "simple"},
        }
        result = await sql_approval_gate(
            state,
            hitl_service=MagicMock(),
            run_id="run-1",
            session_id="sess-1",
        )
        assert result["hitl_skipped"] is True

    @pytest.mark.asyncio
    async def test_creates_interrupt_when_enabled(self):
        from agents.hitl_gates import sql_approval_gate

        mock_hitl = AsyncMock()
        mock_interrupt_info = MagicMock()
        mock_interrupt_info.interrupt_id = "int-123"

        mock_resolved = MagicMock()
        mock_resolved.status = InterruptStatus.APPROVED
        mock_resolved.resolution = None

        mock_hitl.create_interrupt.return_value = mock_interrupt_info
        mock_hitl.wait_for_resolution.return_value = mock_resolved

        state = {
            "hitl_config": {"enabled": True, "require_sql_approval": True},
            "generated_sql": "SELECT desk, SUM(pnl) FROM trades GROUP BY desk LIMIT 100",
            "validation_result": {"performance_score": 9},
            "sql_parameters": {},
            "schema_info": {},
            "user_query": "PnL by desk",
            "intent_analysis": {"complexity": "simple"},
        }
        result = await sql_approval_gate(
            state,
            hitl_service=mock_hitl,
            run_id="run-1",
            session_id="sess-1",
            blocking=True,
        )

        mock_hitl.create_interrupt.assert_called_once()
        mock_hitl.wait_for_resolution.assert_called_once()
        assert result["hitl_response"]["action"] == "approved"

    @pytest.mark.asyncio
    async def test_rejection_stops_pipeline(self):
        from agents.hitl_gates import sql_approval_gate

        mock_hitl = AsyncMock()
        mock_info = MagicMock()
        mock_info.interrupt_id = "int-456"
        mock_hitl.create_interrupt.return_value = mock_info

        mock_resolved = MagicMock()
        mock_resolved.status = InterruptStatus.REJECTED
        mock_resolved.resolution = None
        mock_hitl.wait_for_resolution.return_value = mock_resolved

        state = {
            "hitl_config": {"enabled": True, "require_sql_approval": True},
            "generated_sql": "SELECT 1 LIMIT 1",
            "validation_result": {"performance_score": 5, "is_valid": True},
            "sql_parameters": {},
            "schema_info": {},
            "user_query": "test",
            "intent_analysis": {"complexity": "simple"},
        }
        result = await sql_approval_gate(
            state,
            hitl_service=mock_hitl,
            run_id="run-1",
            session_id="sess-1",
            blocking=True,
        )

        assert result["hitl_response"]["action"] == "rejected"
        assert result["error"] == "SQL query rejected by user."
        assert result["validation_result"]["is_valid"] is False

    @pytest.mark.asyncio
    async def test_modification_updates_sql(self):
        from agents.hitl_gates import sql_approval_gate

        mock_hitl = AsyncMock()
        mock_info = MagicMock()
        mock_info.interrupt_id = "int-789"
        mock_hitl.create_interrupt.return_value = mock_info

        mock_resolved = MagicMock()
        mock_resolved.status = InterruptStatus.MODIFIED
        mock_resolved.resolution = {
            "modifications": {"sql": "SELECT desk, SUM(pnl) FROM trades WHERE desk = 'Alpha' GROUP BY desk LIMIT 10"}
        }
        mock_hitl.wait_for_resolution.return_value = mock_resolved

        state = {
            "hitl_config": {"enabled": True, "require_sql_approval": True},
            "generated_sql": "SELECT desk, SUM(pnl) FROM trades GROUP BY desk LIMIT 100",
            "validation_result": {"performance_score": 9, "is_valid": True},
            "sql_parameters": {},
            "schema_info": {},
            "user_query": "PnL by desk",
            "intent_analysis": {"complexity": "simple"},
        }
        result = await sql_approval_gate(
            state,
            hitl_service=mock_hitl,
            run_id="run-1",
            session_id="sess-1",
            blocking=True,
        )

        assert "Alpha" in result["generated_sql"]
        assert result["validation_result"]["is_valid"] is True

    @pytest.mark.asyncio
    async def test_nonblocking_returns_pending(self):
        from agents.hitl_gates import sql_approval_gate

        mock_hitl = AsyncMock()
        mock_info = MagicMock()
        mock_info.interrupt_id = "int-nb"
        mock_hitl.create_interrupt.return_value = mock_info

        state = {
            "hitl_config": {"enabled": True, "require_sql_approval": True},
            "generated_sql": "SELECT 1 LIMIT 1",
            "validation_result": {},
            "sql_parameters": {},
            "schema_info": {},
            "user_query": "test",
            "intent_analysis": {"complexity": "simple"},
        }
        result = await sql_approval_gate(
            state,
            hitl_service=mock_hitl,
            run_id="run-1",
            session_id="sess-1",
            blocking=False,  # non-blocking mode
        )

        assert result["hitl_pending"]["status"] == "pending"
        assert result["hitl_pending"]["interrupt_id"] == "int-nb"
        # Should NOT have called wait_for_resolution
        mock_hitl.wait_for_resolution.assert_not_called()


# ── Clarification Gate Tests ─────────────────────────────────────


class TestClarificationGate:
    @pytest.mark.asyncio
    async def test_skipped_when_no_ambiguity(self):
        from agents.hitl_gates import clarification_gate

        state = {
            "hitl_config": {"enabled": True},
            "intent_analysis": {"ambiguity_notes": ""},
            "user_query": "PnL by desk",
        }
        # No ambiguity and no hitl_service = skip
        result = await clarification_gate(
            state,
            hitl_service=None,
            run_id="run-1",
            session_id="sess-1",
        )
        assert result["hitl_skipped"] is True

    @pytest.mark.asyncio
    async def test_clarification_augments_query(self):
        from agents.hitl_gates import clarification_gate

        mock_hitl = AsyncMock()
        mock_info = MagicMock()
        mock_info.interrupt_id = "clr-1"
        mock_hitl.create_interrupt.return_value = mock_info

        mock_resolved = MagicMock()
        mock_resolved.status = InterruptStatus.CLARIFIED
        mock_resolved.resolution = {
            "modifications": {"answer": "Only the Alpha desk"}
        }
        mock_hitl.wait_for_resolution.return_value = mock_resolved

        state = {
            "hitl_config": {"enabled": True},
            "intent_analysis": {"ambiguity_notes": "Which desk do you mean?"},
            "user_query": "Show me PnL by desk",
        }
        result = await clarification_gate(
            state,
            hitl_service=mock_hitl,
            run_id="run-1",
            session_id="sess-1",
            question="Which desk do you mean?",
        )

        assert "Alpha desk" in result["user_query"]
        assert result["hitl_response"]["action"] == "clarified"
