"""
Redis-backed cache service.

Used for:
- Schema introspection caching (schemas rarely change)
- Query result caching (short TTL for repeated queries)
- Rate limiting state (via SlowAPI)
- Run deduplication
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

import redis.asyncio as aioredis

from config import Settings

logger = logging.getLogger(__name__)


class CacheService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._redis: aioredis.Redis | None = None

    async def initialize(self) -> None:
        self._redis = aioredis.from_url(
            self.settings.redis_url,
            decode_responses=True,
            max_connections=20,
        )
        try:
            await self._redis.ping()
            logger.info("Redis connected: %s", self.settings.redis_url)
        except Exception as e:
            logger.warning("Redis unavailable, caching disabled: %s", e)
            self._redis = None

    async def shutdown(self) -> None:
        if self._redis:
            await self._redis.close()

    @property
    def available(self) -> bool:
        return self._redis is not None

    async def ping(self) -> bool:
        if not self._redis:
            return False
        try:
            await self._redis.ping()
            return True
        except Exception:
            return False

    # ── Generic get/set ──────────────────────────────────────────

    async def get(self, key: str) -> Any | None:
        if not self._redis:
            return None
        try:
            val = await self._redis.get(key)
            if val is not None:
                return json.loads(val)
        except Exception as e:
            logger.debug("Cache get error for %s: %s", key, e)
        return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        if not self._redis:
            return
        try:
            await self._redis.set(
                key,
                json.dumps(value, default=str),
                ex=ttl or self.settings.cache_ttl,
            )
        except Exception as e:
            logger.debug("Cache set error for %s: %s", key, e)

    async def delete(self, key: str) -> None:
        if not self._redis:
            return
        try:
            await self._redis.delete(key)
        except Exception as e:
            logger.debug("Cache delete error for %s: %s", key, e)

    # ── Schema Cache ─────────────────────────────────────────────

    @staticmethod
    def schema_key(database: str) -> str:
        return f"schema:{database}"

    async def get_schema(self, database: str) -> dict | None:
        return await self.get(self.schema_key(database))

    async def set_schema(self, database: str, schema: dict) -> None:
        await self.set(self.schema_key(database), schema, ttl=600)

    # ── Query Result Cache ───────────────────────────────────────

    @staticmethod
    def query_key(sql: str, params: dict | None = None) -> str:
        raw = sql + json.dumps(params or {}, sort_keys=True)
        h = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"qcache:{h}"

    async def get_query_result(
        self, sql: str, params: dict | None = None
    ) -> dict | None:
        return await self.get(self.query_key(sql, params))

    async def set_query_result(
        self,
        sql: str,
        result: dict,
        params: dict | None = None,
        ttl: int = 120,
    ) -> None:
        await self.set(self.query_key(sql, params), result, ttl=ttl)

    # ── Run Dedup ────────────────────────────────────────────────

    async def is_duplicate_run(self, query_hash: str, window_seconds: int = 5) -> bool:
        """Prevent identical queries within a short window."""
        if not self._redis:
            return False
        key = f"dedup:{query_hash}"
        exists = await self._redis.get(key)
        if exists:
            return True
        await self._redis.set(key, "1", ex=window_seconds)
        return False
