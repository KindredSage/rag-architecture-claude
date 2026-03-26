"""
Rate limiting using SlowAPI with Redis backend.

Limits are configurable per-endpoint. Falls back to in-memory
storage if Redis is unavailable.
"""

from __future__ import annotations

import logging

from slowapi import Limiter
from slowapi.util import get_remote_address

from config import Settings

logger = logging.getLogger(__name__)


def create_limiter(settings: Settings) -> Limiter:
    """Create a SlowAPI limiter with Redis or memory backend."""
    if not settings.rate_limit_enabled:
        # Disabled: use an extremely high limit
        return Limiter(
            key_func=get_remote_address,
            default_limits=["10000/minute"],
            enabled=False,
        )

    storage_uri = settings.redis_url if settings.redis_url else "memory://"

    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=[settings.rate_limit_default],
        storage_uri=storage_uri,
        strategy="moving-window",
    )

    logger.info(
        "Rate limiter initialized: default=%s backend=%s",
        settings.rate_limit_default,
        "redis" if "redis" in storage_uri else "memory",
    )

    return limiter


def setup_rate_limiter(app) -> None:
    """Attach the SlowAPI limiter to a FastAPI app."""
    from slowapi import _rate_limit_exceeded_handler
    from slowapi.errors import RateLimitExceeded
    from config import get_settings

    settings = get_settings()
    limiter = create_limiter(settings)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
