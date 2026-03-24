import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import httpx

from config import settings

logger = logging.getLogger(__name__)


class TokenManager:
    """
    Async-safe token manager.
    Fetches a new token via mTLS client credentials and caches it.
    Refreshes automatically TOKEN_EXPIRY_BUFFER seconds before expiry.
    """

    def __init__(self):
        self._token: Optional[str] = None
        self._expiry: Optional[datetime] = None
        self._lock = asyncio.Lock()

    async def get_token(self, client: httpx.AsyncClient) -> str:
        async with self._lock:
            if self._is_valid():
                return self._token
            return await self._refresh(client)

    def _is_valid(self) -> bool:
        return (
            self._token is not None
            and self._expiry is not None
            and datetime.now() < self._expiry
        )

    async def _refresh(self, client: httpx.AsyncClient) -> str:
        logger.info("Token expired or missing — refreshing...")
        resp = await client.post(
            settings.TOKEN_URL,
            data={"grant_type": "client_credentials"},
            # Adjust above to match your token endpoint contract.
            # Add client_id/client_secret here if needed.
        )
        resp.raise_for_status()
        data = resp.json()

        self._token = data["access_token"]
        expires_in = data.get("expires_in", 3600)
        self._expiry = datetime.now() + timedelta(
            seconds=expires_in - settings.TOKEN_EXPIRY_BUFFER
        )
        logger.info(f"Token refreshed — valid until {self._expiry}")
        return self._token


# Singleton — imported everywhere
token_manager = TokenManager()
