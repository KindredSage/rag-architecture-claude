import logging

import httpx
from fastapi import APIRouter, HTTPException

from config import settings
from core.client import get_authed_headers, get_client
from schemas.openai import EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(request: EmbeddingRequest):
    """
    Proxy POST /v1/embeddings to the backend.
    OpenWebUI RAG pipeline hits this for document and query embedding.
    Input can be a string, list of strings, or tokenized input.
    """
    client = await get_client()
    headers = await get_authed_headers()
    payload = request.model_dump(exclude_none=True)

    try:
        resp = await client.post(
            f"{settings.BACKEND_BASE_URL}/v1/embeddings",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Embedding backend error {e.response.status_code}: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        logger.error(f"Embedding request error: {e}")
        raise HTTPException(status_code=502, detail=f"Backend unreachable: {e}")
