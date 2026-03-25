"""
utils/retry.py
--------------
Reusable retry decorators built on Tenacity.
Used by agent nodes that call external services (LLMs, databases).
"""

import asyncio
from functools import wraps
from typing import Callable, Type, Tuple
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


def with_retry(
    max_attempts: int | None = None,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
):
    """
    Decorator factory: wraps an async function with exponential-backoff retry.

    Usage:
        @with_retry(max_attempts=3, exceptions=(httpx.HTTPError,))
        async def call_llm(...): ...
    """
    attempts = max_attempts or settings.max_retries

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            delay = settings.retry_delay
            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt < attempts:
                        logger.warning(
                            "retry_scheduled",
                            func=func.__name__,
                            attempt=attempt,
                            max_attempts=attempts,
                            error=str(exc),
                        )
                        await asyncio.sleep(delay * attempt)
                    else:
                        logger.error(
                            "retry_exhausted",
                            func=func.__name__,
                            attempts=attempts,
                            error=str(exc),
                        )
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
