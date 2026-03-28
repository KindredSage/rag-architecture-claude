"""
Application entrypoint.

Responsibilities:
  - FastAPI app creation
  - Service initialization (lifespan)
  - Middleware setup
  - Router mounting

All route handlers live in routes/.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import Settings, get_settings
from middleware.rate_limiter import setup_rate_limiter
from routes import register_routes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent_platform")

SERVER_ID = os.getenv("SERVER_ID", f"server-{uuid.uuid4().hex[:8]}")


# ── Lifespan ─────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting %s v%s  server=%s  env=%s",
                settings.app_name, settings.app_version, SERVER_ID, settings.environment)

    services = await _init_services(settings)
    app.state.services = services
    app.state.server_id = SERVER_ID

    _register_agents()

    from agents.master_agent import build_master_graph
    app.state.master_graph = build_master_graph(settings=settings, services=services)
    logger.info("Master graph compiled")

    if services.get("session_manager"):
        cleanup = asyncio.create_task(
            _session_cleanup_loop(services["session_manager"], settings)
        )
    else:
        cleanup = None
    logger.info("Startup complete")

    yield

    if cleanup:
        cleanup.cancel()
        try:
            await cleanup
        except asyncio.CancelledError:
            pass
    await _shutdown_services(services)
    logger.info("Shutdown complete")


async def _init_services(settings: Settings) -> dict[str, Any]:
    from services.cache_service import CacheService
    from services.clickhouse_service import ClickHouseService
    from services.llm_service import LLMService
    from services.session_manager import SessionManager
    from services.hitl import HITLService
    from tools.mcp_client import MCPClientManager
    from tools.clickhouse_tools import create_clickhouse_tools
    from tools.email_tools import create_email_tools
    from tools.export_tools import create_export_tools
    from tools.plotting_tools import create_plotting_tools

    ch = ClickHouseService(settings)
    ch.initialize()

    llm = LLMService(settings)
    llm.initialize()

    cache = CacheService(settings)
    await cache.initialize()

    sm = SessionManager(settings, SERVER_ID)
    await sm.initialize()

    hitl = HITLService(sm.pool)
    await hitl.ensure_tables()

    mcp = MCPClientManager(settings)
    await mcp.initialize()

    os.makedirs(settings.artifact_dir, exist_ok=True)
    ch_tools = create_clickhouse_tools(ch)
    if mcp.available:
        mcp_ch = mcp.get_tools_by_prefix("clickhouse_")
        if mcp_ch:
            ch_tools = mcp_ch
            logger.info("Using MCP ClickHouse tools (%d)", len(mcp_ch))

    return {
        "clickhouse": ch,
        "llm": llm,
        "cache": cache,
        "session_manager": sm,
        "hitl": hitl,
        "mcp": mcp,
        "ch_tools": ch_tools,
        "plotting_tools": create_plotting_tools(settings.artifact_dir),
        "email_tools": create_email_tools(settings),
        "export_tools": create_export_tools(settings.artifact_dir),
    }


def _register_agents():
    import agents.trade      # noqa: F401
    import agents.reporting   # noqa: F401
    import agents.analytics   # noqa: F401
    from agents.registry import AgentRegistry
    logger.info("Agents: %s", [a.agent_id for a in AgentRegistry.get_all()])


async def _shutdown_services(services: dict[str, Any]):
    if services.get("mcp"):
        await services["mcp"].shutdown()
    if services.get("cache"):
        await services["cache"].shutdown()
    if services.get("session_manager"):
        await services["session_manager"].shutdown()
    if services.get("clickhouse"):
        services["clickhouse"].close()


async def _session_cleanup_loop(sm, settings: Settings):
    while True:
        try:
            await asyncio.sleep(settings.session_cleanup_interval_minutes * 60)
            await sm.cleanup_expired_sessions()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("Session cleanup error: %s", e)


# ── App ──────────────────────────────────────────────────────────

app = FastAPI(title="Agent Platform", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
setup_rate_limiter(app)
register_routes(app)
