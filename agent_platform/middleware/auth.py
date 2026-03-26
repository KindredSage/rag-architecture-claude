"""
API Key authentication middleware.

If AGENT_API_KEY is set, all requests must include it as a Bearer token
or X-API-Key header. If not set, authentication is disabled.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from config import Settings, get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


async def verify_api_key(
    settings: Settings = Depends(get_settings),
    api_key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> str | None:
    """
    Verify API key from header or Bearer token.
    Returns the key if valid, None if auth is disabled.
    """
    configured_key = settings.api_key.get_secret_value()
    if not configured_key:
        return None  # Auth disabled

    provided = api_key or (bearer.credentials if bearer else None)

    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide via X-API-Key header or Bearer token.",
        )

    if provided != configured_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return provided
