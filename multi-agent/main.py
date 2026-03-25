"""
app/main.py
-----------
FastAPI application entry point.

Startup sequence (lifespan):
  1. Configure structured logging
  2. Register all agents into the AgentRegistry
  3. Mount API routers

The lifespan pattern (replacing deprecated @app.on_event) ensures clean
startup and shutdown for async resources.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.registry.agent_registry import registry

# ── Import and register concrete agents ───────────────────────────────────────
# Add new agents here – nothing else needs to change.
from app.agents.trade.agent import TradeAgent

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle manager."""
    # ── STARTUP ───────────────────────────────────────────────────────────────
    configure_logging()
    logger = get_logger("startup")

    logger.info("app_starting", env=settings.app_env, version=settings.app_version)

    # Register agents – plug-and-play: just add more lines here
    registry.register(TradeAgent())
    # registry.register(RiskAgent())      ← future agent
    # registry.register(PositionAgent())  ← future agent

    logger.info("agents_registered", count=len(registry), agents=registry.agent_names())

    yield  # ← application is running

    # ── SHUTDOWN ──────────────────────────────────────────────────────────────
    logger.info("app_shutting_down")


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    description=(
        "Production-grade multi-agent orchestration system. "
        "LangGraph-powered agents exposed via FastAPI."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS (adjust origins for production) ─────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
from app.api.routes.routes import agents_router, execute_router

app.include_router(agents_router)
app.include_router(execute_router)


# ── Health check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "healthy",
        "version": settings.app_version,
        "registered_agents": registry.agent_names(),
    })


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", tags=["System"])
async def root() -> JSONResponse:
    return JSONResponse({
        "message": "Multi-Agent Orchestration System",
        "docs": "/docs",
        "agents": "/agents",
        "execute": "POST /execute",
    })
