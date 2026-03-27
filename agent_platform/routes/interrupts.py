from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from middleware.auth import verify_api_key
from models import InterruptResponse, InterruptStatus

router = APIRouter(tags=["interrupts"])


def _get_hitl(request: Request):
    svc = request.app.state.services.get("hitl")
    if not svc:
        raise HTTPException(501, "HITL not enabled")
    return svc


@router.get("/interrupts/run/{run_id}")
async def get_interrupts_for_run(
    run_id: str,
    request: Request,
    _: Any = Depends(verify_api_key),
):
    hitl = _get_hitl(request)
    pending = await hitl.get_pending_for_run(run_id)
    return {"run_id": run_id, "pending_count": len(pending), "interrupts": pending}


@router.get("/interrupts/session/{session_id}")
async def get_interrupts_for_session(
    session_id: str,
    request: Request,
    _: Any = Depends(verify_api_key),
):
    hitl = _get_hitl(request)
    pending = await hitl.get_pending_for_session(session_id)
    return {"session_id": session_id, "pending_count": len(pending), "interrupts": pending}


@router.get("/interrupts/{interrupt_id}")
async def get_interrupt_details(
    interrupt_id: str,
    request: Request,
    _: Any = Depends(verify_api_key),
):
    hitl = _get_hitl(request)
    info = await hitl.get_interrupt(interrupt_id)
    if not info:
        raise HTTPException(404, "Interrupt not found")
    return info


@router.post("/interrupts/{interrupt_id}/resolve")
async def resolve_interrupt(
    interrupt_id: str,
    body: dict,
    request: Request,
    _: Any = Depends(verify_api_key),
):
    """
    Resolve a pending interrupt.

    Body examples:
      Approve:  {"action": "approved"}
      Reject:   {"action": "rejected", "comment": "looks wrong"}
      Modify:   {"action": "modified", "modifications": {"sql": "SELECT ..."}}
      Clarify:  {"action": "clarified", "modifications": {"answer": "Only Alpha"}}
    """
    hitl = _get_hitl(request)

    try:
        response = InterruptResponse(
            interrupt_id=interrupt_id,
            action=InterruptStatus(body.get("action", "approved")),
            modifications=body.get("modifications", {}),
            comment=body.get("comment", ""),
        )
    except ValueError as e:
        raise HTTPException(400, f"Invalid action. Use: approved, rejected, modified, clarified. {e}")

    result = await hitl.resolve_interrupt(response)
    if not result:
        raise HTTPException(404, "Interrupt not found")

    return {"status": "resolved", "interrupt": result}
