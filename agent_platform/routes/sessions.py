from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from middleware.auth import verify_api_key

router = APIRouter(tags=["sessions"])


@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    request: Request,
    _: Any = Depends(verify_api_key),
):
    """Get session info and conversation history."""
    sm = request.app.state.services["session_manager"]
    session = await sm.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    history = await sm.get_history(session_id, limit=50)
    return {
        "session_id": session["session_id"],
        "user_id": session.get("user_id"),
        "server_id": session["server_id"],
        "created_at": session["created_at"].isoformat(),
        "last_active": session["last_active"].isoformat(),
        "history": history,
    }


@router.delete("/sessions/{session_id}")
async def close_session(
    session_id: str,
    request: Request,
    _: Any = Depends(verify_api_key),
):
    """Close a session (mark inactive)."""
    sm = request.app.state.services["session_manager"]
    await sm.close_session(session_id)
    return {"status": "closed", "session_id": session_id}
