from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from agents.registry import AgentRegistry
from models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    services = request.app.state.services
    ch = services["clickhouse"]
    cache = services["cache"]
    sm = services["session_manager"]

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        server_id=request.app.state.server_id,
        clickhouse="connected" if ch.ping() else "disconnected",
        postgres="connected",
        redis="connected" if await cache.ping() else "disconnected",
        active_runs=await sm.get_active_run_count(),
        registered_agents=len(AgentRegistry.get_all()),
    )
