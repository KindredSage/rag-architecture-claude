"""
core/logging.py
---------------
Configures structlog for consistent, structured JSON logging throughout
the application. Every log entry carries a correlation context (trace_id)
so end-to-end request flows can be traced.
"""

import logging
import sys
import structlog
from app.core.config import get_settings


def configure_logging() -> None:
    """Wire structlog + stdlib logging together at startup."""
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Configure stdlib root logger (used by uvicorn, fastapi internals)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # Shared processors: timestamp, log level, caller info
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.CallsiteParameterAdder(
            [structlog.processors.CallsiteParameter.FUNC_NAME]
        ),
    ]

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a named structlog logger. Call anywhere in the codebase."""
    return structlog.get_logger(name)
