"""
Route registration. Called once from main.py during app setup.
"""

from fastapi import FastAPI

from routes.health import router as health_router
from routes.agents import router as agents_router
from routes.execute import router as execute_router
from routes.sessions import router as sessions_router
from routes.interrupts import router as interrupts_router


def register_routes(app: FastAPI) -> None:
    app.include_router(health_router)
    app.include_router(agents_router)
    app.include_router(execute_router)
    app.include_router(sessions_router)
    app.include_router(interrupts_router)
