"""
FastAPI dependency injection.

Provides typed access to shared services (ClickHouse, LLM, Redis, Sessions)
through FastAPI's Depends() system.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request

from config import Settings, get_settings
from services.cache_service import CacheService
from services.clickhouse_service import ClickHouseService
from services.llm_service import LLMService
from services.session_manager import SessionManager


def get_services(request: Request) -> dict[str, Any]:
    """Return the services dict from app.state."""
    return request.app.state.services


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.services["session_manager"]


def get_clickhouse(request: Request) -> ClickHouseService:
    return request.app.state.services["clickhouse"]


def get_cache(request: Request) -> CacheService:
    return request.app.state.services["cache"]


def get_llm(request: Request) -> LLMService:
    return request.app.state.services["llm"]


async def get_active_run_count(request: Request) -> int:
    sm: SessionManager = request.app.state.services["session_manager"]
    return await sm.get_active_run_count()


async def check_concurrency_limit(
    request: Request,
    settings: Settings = Depends(get_settings),
):
    """Reject requests if too many runs are active."""
    sm: SessionManager = request.app.state.services["session_manager"]
    active = await sm.get_active_run_count()
    if active >= settings.max_concurrent_runs:
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent runs ({active}/{settings.max_concurrent_runs}). Try again shortly.",
        )
