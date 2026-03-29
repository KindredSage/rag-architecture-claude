"""
Human-in-the-Loop gate nodes.

These nodes are inserted at critical points in agent graphs.
When HITL is enabled, they pause execution and wait for human input.

Gate types:
  - sql_approval_gate:   Before query execution (shows SQL, asks approve/modify)
  - email_confirm_gate:  Before sending email (shows recipients, asks confirm)
  - clarification_gate:  When intent is ambiguous (asks user to clarify)

Each gate follows this pattern:
  1. Check HITLConfig: should we interrupt?
  2. If no: pass through (no-op)
  3. If yes: create InterruptRequest in PG, then either:
     a. BLOCKING mode: poll until user resolves (for sync /execute)
     b. NON-BLOCKING mode: return with hitl_pending set (for /execute/stream)
  4. Merge user's modifications into state and continue
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from models import (
    HITLConfig,
    InterruptRequest,
    InterruptStatus,
    InterruptType,
)
from services.hitl import HITLService

logger = logging.getLogger(__name__)


async def sql_approval_gate(
    state: dict,
    *,
    hitl_service: HITLService,
    run_id: str,
    session_id: str,
    agent_id: str = "trade_agent",
    blocking: bool = True,
    timeout: int = 300,
) -> dict:
    """
    Gate before query execution: shows generated SQL and asks for approval.

    If user modifies the SQL, the modified version replaces generated_sql.
    If user rejects, the pipeline stops gracefully.
    """
    start = time.perf_counter()
    config = HITLConfig(**state.get("hitl_config", {}))
    sql = state.get("generated_sql", "")
    validation = state.get("validation_result", {})
    complexity = state.get("intent_analysis", {}).get("complexity", "simple")

    # Check if we should interrupt
    if not HITLService.should_interrupt(config, InterruptType.APPROVAL, complexity):
        return {
            "hitl_skipped": True,
            "execution_trace": [{
                "node": "sql_approval_gate",
                "status": "skipped",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "output_summary": "HITL disabled or below complexity threshold",
            }],
        }

    # Create interrupt
    interrupt = InterruptRequest(
        run_id=run_id,
        session_id=session_id,
        interrupt_type=InterruptType.APPROVAL,
        node_name="sql_approval_gate",
        agent_id=agent_id,
        title="SQL Query Approval Required",
        description=(
            f"The following SQL query has been generated and validated "
            f"(performance score: {validation.get('performance_score', '?')}/10). "
            f"Please review and approve, modify, or reject."
        ),
        payload={
            "sql": sql,
            "parameters": state.get("sql_parameters", {}),
            "validation": validation,
            "estimated_rows": state.get("schema_info", {}).get("estimated_rows"),
            "user_query": state.get("user_query", ""),
            "editable_fields": ["sql", "parameters"],
        },
        auto_approve_seconds=config.auto_approve_timeout,
    )

    info = await hitl_service.create_interrupt(interrupt)
    logger.info("SQL approval gate: interrupt %s created", info.interrupt_id)

    if not blocking:
        # Non-blocking: return immediately with pending status
        return {
            "hitl_pending": {
                "interrupt_id": info.interrupt_id,
                "type": InterruptType.APPROVAL.value,
                "status": "pending",
            },
            "execution_trace": [{
                "node": "sql_approval_gate",
                "status": "waiting_human",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "output_summary": f"Waiting for human approval (interrupt={info.interrupt_id})",
            }],
        }

    # Blocking: wait for resolution
    resolved = await hitl_service.wait_for_resolution(
        info.interrupt_id, timeout=timeout,
    )
    duration = (time.perf_counter() - start) * 1000

    if not resolved or resolved.status in (InterruptStatus.EXPIRED,):
        return {
            "hitl_response": {"action": "expired"},
            "error": "SQL approval timed out. Query was not executed.",
            "execution_trace": [{
                "node": "sql_approval_gate",
                "status": "expired",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
            }],
        }

    if resolved.status == InterruptStatus.REJECTED:
        return {
            "hitl_response": {"action": "rejected"},
            "error": "SQL query rejected by user.",
            "validation_result": {**validation, "is_valid": False},
            "execution_trace": [{
                "node": "sql_approval_gate",
                "status": "rejected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
                "output_summary": "User rejected the query",
            }],
        }

    # Approved or Modified
    result: dict[str, Any] = {
        "hitl_response": {"action": resolved.status.value},
        "hitl_skipped": False,
        "execution_trace": [{
            "node": "sql_approval_gate",
            "status": "approved",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round(duration, 1),
            "output_summary": f"User {resolved.status.value}",
        }],
    }

    # If user modified the SQL, apply changes
    if resolved.resolution and resolved.status == InterruptStatus.MODIFIED:
        mods = resolved.resolution.get("modifications", {})
        if "sql" in mods:
            result["generated_sql"] = mods["sql"]
            result["validation_result"] = {
                **validation,
                "approved_sql": mods["sql"],
                "is_valid": True,  # user approved = valid
            }
            logger.info("SQL modified by user: %s", mods["sql"][:100])
        if "parameters" in mods:
            result["sql_parameters"] = mods["parameters"]

    return result


async def email_confirm_gate(
    state: dict,
    *,
    hitl_service: HITLService,
    run_id: str,
    session_id: str,
    agent_id: str = "reporting_agent",
    blocking: bool = True,
    timeout: int = 300,
) -> dict:
    """
    Gate before sending email: shows recipients, subject, attachments.
    """
    start = time.perf_counter()
    config = HITLConfig(**state.get("hitl_config", {}))
    analysis = state.get("analysis", {})

    if not HITLService.should_interrupt(config, InterruptType.CONFIRMATION):
        return {
            "hitl_skipped": True,
            "execution_trace": [{
                "node": "email_confirm_gate",
                "status": "skipped",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    plan = state.get("report_plan", {})
    interrupt = InterruptRequest(
        run_id=run_id,
        session_id=session_id,
        interrupt_type=InterruptType.CONFIRMATION,
        node_name="email_confirm_gate",
        agent_id=agent_id,
        title="Email Confirmation Required",
        description="An email is about to be sent. Please confirm the details.",
        payload={
            "action": "send_email",
            "to": plan.get("email_to", []),
            "subject": f"Report: {plan.get('title', 'Report')}",
            "attachments": [a.get("name", "") for a in state.get("artifacts", [])],
            "body_preview": analysis.get("email_summary", analysis.get("narrative", ""))[:500],
        },
        auto_approve_seconds=config.auto_approve_timeout,
    )

    info = await hitl_service.create_interrupt(interrupt)

    if not blocking:
        return {
            "hitl_pending": {
                "interrupt_id": info.interrupt_id,
                "type": InterruptType.CONFIRMATION.value,
                "status": "pending",
            },
            "execution_trace": [{
                "node": "email_confirm_gate",
                "status": "waiting_human",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    resolved = await hitl_service.wait_for_resolution(info.interrupt_id, timeout=timeout)
    duration = (time.perf_counter() - start) * 1000

    if not resolved or resolved.status in (InterruptStatus.EXPIRED, InterruptStatus.REJECTED):
        action = resolved.status.value if resolved else "expired"
        return {
            "hitl_response": {"action": action},
            "email_result": {"sent": False, "reason": f"Email {action} by user"},
            "execution_trace": [{
                "node": "email_confirm_gate",
                "status": action,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
            }],
        }

    return {
        "hitl_response": {"action": "approved"},
        "execution_trace": [{
            "node": "email_confirm_gate",
            "status": "approved",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round(duration, 1),
        }],
    }


async def clarification_gate(
    state: dict,
    *,
    hitl_service: HITLService,
    run_id: str,
    session_id: str,
    agent_id: str = "trade_agent",
    question: str = "",
    options: list[str] | None = None,
    blocking: bool = True,
    timeout: int = 600,
) -> dict:
    """
    Gate for ambiguous queries: asks user to clarify before proceeding.
    Called when intent_analysis has non-empty ambiguity_notes.
    """
    start = time.perf_counter()
    config = HITLConfig(**state.get("hitl_config", {}))

    if not config.enabled:
        return {
            "hitl_skipped": True,
            "execution_trace": [{
                "node": "clarification_gate",
                "status": "skipped",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    intent = state.get("intent_analysis", {})
    if not question:
        question = intent.get("ambiguity_notes", "Could you clarify your request?")

    interrupt = InterruptRequest(
        run_id=run_id,
        session_id=session_id,
        interrupt_type=InterruptType.CLARIFICATION,
        node_name="clarification_gate",
        agent_id=agent_id,
        title="Clarification Needed",
        description=question,
        payload={
            "question": question,
            "options": options or [],
            "original_query": state.get("user_query", ""),
            "detected_ambiguity": intent.get("ambiguity_notes", ""),
        },
        auto_approve_seconds=None,  # never auto-approve clarifications
    )

    info = await hitl_service.create_interrupt(interrupt)

    if not blocking:
        return {
            "hitl_pending": {
                "interrupt_id": info.interrupt_id,
                "type": InterruptType.CLARIFICATION.value,
                "status": "pending",
            },
            "execution_trace": [{
                "node": "clarification_gate",
                "status": "waiting_human",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        }

    resolved = await hitl_service.wait_for_resolution(info.interrupt_id, timeout=timeout)
    duration = (time.perf_counter() - start) * 1000

    if not resolved or resolved.status == InterruptStatus.EXPIRED:
        return {
            "hitl_response": {"action": "expired"},
            "error": "Clarification request timed out.",
            "execution_trace": [{
                "node": "clarification_gate",
                "status": "expired",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
            }],
        }

    # Merge clarification into query context
    answer = ""
    if resolved.resolution:
        answer = resolved.resolution.get("modifications", {}).get("answer", "")

    result: dict[str, Any] = {
        "hitl_response": {"action": "clarified", "answer": answer},
        "execution_trace": [{
            "node": "clarification_gate",
            "status": "clarified",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round(duration, 1),
            "output_summary": f"User clarified: {answer[:100]}",
        }],
    }

    # Augment the user query with the clarification
    if answer:
        result["user_query"] = f"{state.get('user_query', '')} [Clarification: {answer}]"

    return result


async def failure_feedback_gate(
    state: dict,
    *,
    hitl_service: HITLService,
    run_id: str,
    session_id: str,
    agent_id: str = "trade_agent",
    blocking: bool = True,
    timeout: int = 600,
) -> dict:
    """
    Gate when query retries are exhausted. Asks the user what they expected,
    then routes back to query_analyst for a full re-assessment.
    """
    start = time.perf_counter()
    config = HITLConfig(**state.get("hitl_config", {}))

    if not config.enabled:
        return {
            "hitl_skipped": True,
            "execution_trace": [{
                "node": "failure_feedback_gate",
                "status": "skipped",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "output_summary": "HITL disabled; cannot ask user for feedback",
            }],
        }

    retry_feedback = state.get("retry_feedback", "Unknown error")
    generated_sql = state.get("generated_sql", "")
    validation_issues = state.get("validation_result", {}).get("issues", [])

    interrupt = InterruptRequest(
        run_id=run_id,
        session_id=session_id,
        interrupt_type=InterruptType.CLARIFICATION,
        node_name="failure_feedback_gate",
        agent_id=agent_id,
        title="Query Failed - Help Us Understand What You Need",
        description=(
            f"We tried {state.get('retry_count', 0)} times but could not build a valid query. "
            f"Last error: {retry_feedback}. "
            "Could you describe what result you expected?"
        ),
        payload={
            "question": "What result did you expect? Please describe the columns, filters, or format you need.",
            "failed_sql": generated_sql,
            "errors": [i.get("message", "") for i in validation_issues if i.get("severity") == "error"],
            "retry_count": state.get("retry_count", 0),
            "original_query": state.get("user_query", ""),
        },
        auto_approve_seconds=None,
    )

    info = await hitl_service.create_interrupt(interrupt)

    if not blocking:
        return {
            "hitl_pending": {
                "interrupt_id": info.interrupt_id,
                "type": InterruptType.CLARIFICATION.value,
                "status": "pending",
            },
            "execution_trace": [{
                "node": "failure_feedback_gate",
                "status": "waiting_human",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "output_summary": f"Waiting for user feedback (interrupt={info.interrupt_id})",
            }],
        }

    resolved = await hitl_service.wait_for_resolution(info.interrupt_id, timeout=timeout)
    duration = (time.perf_counter() - start) * 1000

    if not resolved or resolved.status == InterruptStatus.EXPIRED:
        return {
            "hitl_response": {"action": "expired"},
            "error": "Feedback request timed out. Query could not be completed.",
            "execution_trace": [{
                "node": "failure_feedback_gate",
                "status": "expired",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
            }],
        }

    if resolved.status == InterruptStatus.REJECTED:
        return {
            "hitl_response": {"action": "rejected"},
            "error": "User chose not to provide feedback.",
            "execution_trace": [{
                "node": "failure_feedback_gate",
                "status": "rejected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_ms": round(duration, 1),
            }],
        }

    # User provided feedback -- merge into state for query_analyst re-run
    answer = ""
    if resolved.resolution:
        answer = resolved.resolution.get("modifications", {}).get("answer", "")

    result: dict[str, Any] = {
        "hitl_response": {"action": "clarified", "answer": answer},
        "needs_retry": False,
        "retry_count": 0,
        "retry_feedback": "",
        "generated_sql": "",
        "validation_result": {},
        "execution_trace": [{
            "node": "failure_feedback_gate",
            "status": "clarified",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round(duration, 1),
            "output_summary": f"User feedback: {answer[:100]}",
        }],
    }

    if answer:
        result["user_query"] = f"{state.get('user_query', '')} [User clarification: {answer}]"

    return result
