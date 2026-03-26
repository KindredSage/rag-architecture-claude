"""
Centralized configuration using Pydantic BaseSettings.
All values are overridable via environment variables prefixed with AGENT_.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────
    app_name: str = "Agent Platform"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    environment: Literal["development", "staging", "production"] = "development"

    # ── LLM ──────────────────────────────────────────────────────
    llm_provider: Literal["openai", "azure", "anthropic"] = "openai"
    llm_model: str = "gpt-4o"
    llm_api_key: SecretStr = SecretStr("")
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4096
    llm_request_timeout: int = 60

    # Fallback model for lightweight tasks (intent classification, validation)
    llm_fast_model: str = "gpt-4o-mini"

    # ── ClickHouse ───────────────────────────────────────────────
    ch_host: str = "localhost"
    ch_port: int = 8123
    ch_native_port: int = 9000
    ch_database: str = "trading_db"
    ch_user: str = "default"
    ch_password: SecretStr = SecretStr("")
    ch_secure: bool = False
    ch_query_timeout: int = 30
    ch_max_rows: int = 50_000
    ch_connect_timeout: int = 10
    ch_send_receive_timeout: int = 30

    # ── PostgreSQL (Session / State Management) ──────────────────
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "agent_platform"
    pg_user: str = "agent_user"
    pg_password: SecretStr = SecretStr("agent_pass")
    pg_pool_min: int = 5
    pg_pool_max: int = 20

    @property
    def pg_dsn(self) -> str:
        return (
            f"postgresql://{self.pg_user}:{self.pg_password.get_secret_value()}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    @property
    def pg_async_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.pg_user}:{self.pg_password.get_secret_value()}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_database}"
        )

    # ── Redis ────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl: int = 300
    rate_limit_enabled: bool = True
    rate_limit_default: str = "60/minute"

    # ── Email (SMTP) ─────────────────────────────────────────────
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: SecretStr = SecretStr("")
    smtp_from: str = ""
    smtp_use_tls: bool = True

    # ── MCP ──────────────────────────────────────────────────────
    mcp_enabled: bool = False
    mcp_server_urls: list[str] = Field(default_factory=list)

    # ── Storage / Artifacts ──────────────────────────────────────
    artifact_dir: str = "/tmp/agent_artifacts"
    max_artifact_size_mb: int = 50

    # ── Security ─────────────────────────────────────────────────
    api_key: SecretStr = SecretStr("")  # empty = no auth
    allowed_ch_databases: list[str] = Field(
        default_factory=lambda: ["trading_db"]
    )
    sql_blocked_keywords: list[str] = Field(
        default_factory=lambda: [
            "DROP", "TRUNCATE", "ALTER", "CREATE", "INSERT",
            "UPDATE", "DELETE", "GRANT", "REVOKE", "ATTACH",
            "DETACH", "RENAME", "OPTIMIZE", "KILL",
        ]
    )

    # ── Concurrency ──────────────────────────────────────────────
    max_concurrent_runs: int = 20
    run_timeout: int = 300  # 5 min max per run
    max_retry_per_node: int = 3

    # ── Session ──────────────────────────────────────────────────
    session_ttl_hours: int = 24
    session_cleanup_interval_minutes: int = 30


@lru_cache()
def get_settings() -> Settings:
    return Settings()
