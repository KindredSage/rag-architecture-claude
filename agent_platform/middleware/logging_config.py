"""
Structured logging configuration using structlog.

Produces JSON logs in production, pretty-printed in development.
"""

from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog


def setup_logging(
    level: str = "INFO",
    environment: Literal["development", "staging", "production"] = "development",
    server_id: str = "unknown",
) -> None:
    """Configure structured logging for the application."""

    # Shared processors
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if environment == "production":
        # JSON output for production (machine parseable)
        renderer = structlog.processors.JSONRenderer()
    else:
        # Pretty console output for dev
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure stdlib logging to use structlog
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    if environment == "production":
        formatter = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s",'
            f'"server_id":"{server_id}","message":"%(message)s"}}'
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Suppress noisy loggers
    for noisy in ("httpx", "httpcore", "openai", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
