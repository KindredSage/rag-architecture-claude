import logging

import httpx
from fastapi import APIRouter, HTTPException

from config import settings
from core.client import get_authed_headers, get_client
from schemas.openai import ModelListResponse, ModelObject

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Static fallback model list.
# Edit this if your backend does not expose GET /v1/models.
# ---------------------------------------------------------------------------
STATIC_MODELS = [
    ModelObject(id="your-chat-model", owned_by="internal"),
    ModelObject(id="your-embed-model", owned_by="internal"),
]


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models():
    """
    Proxy GET /v1/models to the backend.
    Falls back to the static list above if the backend doesn't expose the endpoint.
    OpenWebUI hits this on load to populate the model dropdown.
    """
    client = await get_client()
    headers = await get_authed_headers()

    try:
        resp = await client.get(
            f"{settings.BACKEND_BASE_URL}/v1/models",
            headers=headers,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # Backend has no /models — return static list
            logger.info("Backend has no /v1/models — returning static model list")
            return ModelListResponse(data=STATIC_MODELS)
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        logger.warning(f"/v1/models backend call failed: {e} — using static fallback")
        return ModelListResponse(data=STATIC_MODELS)
