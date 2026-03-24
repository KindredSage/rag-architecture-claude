import logging
from contextlib import asynccontextmanager
from typing import Optional

import httpx

from config import settings
from core.auth import token_manager

logger = logging.getLogger(__name__)

_client: Optional[httpx.AsyncClient] = None


def _build_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        cert=(settings.CLIENT_CERT, settings.CLIENT_KEY),
        verify=settings.CA_CERT,
        limits=httpx.Limits(
            max_keepalive_connections=20,
            max_connections=50,
            keepalive_expiry=30,        # seconds — match your backend's keepalive
        ),
        timeout=httpx.Timeout(
            connect=10.0,
            read=120.0,                 # long for streaming responses
            write=30.0,
            pool=5.0,
        ),
    )


async def get_client() -> httpx.AsyncClient:
    """Return the singleton mTLS client, rebuilding if closed."""
    global _client
    if _client is None or _client.is_closed:
        logger.warning("httpx client missing or closed — rebuilding")
        _client = _build_client()
    return _client


async def get_authed_headers() -> dict:
    """Return headers with a valid Bearer token. Token is refreshed lazily."""
    client = await get_client()
    token = await token_manager.get_token(client)
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


@asynccontextmanager
async def lifespan_client():
    """
    Use inside FastAPI lifespan to manage client lifecycle.
    Warms up the token on startup so the first real request has no cold start.
    """
    global _client
    logger.info("Building mTLS httpx client...")
    _client = _build_client()

    logger.info("Warming up token on startup...")
    await token_manager.get_token(_client)
    logger.info("Gateway ready.")

    yield

    logger.info("Shutting down — closing httpx client...")
    await _client.aclose()
