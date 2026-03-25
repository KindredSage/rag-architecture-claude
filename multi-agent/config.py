"""
core/config.py
--------------
Centralised application settings loaded from environment variables.
Pydantic-Settings validates and coerces types at startup.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # ── LLM ───────────────────────────────────────────────────────────────────
    openai_api_key: str = "sk-placeholder"
    openai_model: str = "gpt-4o-mini"

    # ── App ───────────────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"
    app_title: str = "Multi-Agent Orchestration System"
    app_version: str = "1.0.0"

    # ── Retry ─────────────────────────────────────────────────────────────────
    max_retries: int = 3
    retry_delay: float = 1.0

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache()
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()
