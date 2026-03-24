import logging

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from config import settings
from core.client import get_authed_headers, get_client
from schemas.openai import ChatCompletionRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Proxy POST /v1/chat/completions to the backend.
    Handles both streaming and non-streaming responses.
    Tool calls, response_format, seed, etc. are passed through transparently.
    """
    client = await get_client()
    headers = await get_authed_headers()
    payload = request.model_dump(exclude_none=True)

    if request.stream:
        return StreamingResponse(
            _stream_response(client, headers, payload),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",   # disable nginx buffering for SSE
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # Non-streaming
    try:
        resp = await client.post(
            f"{settings.BACKEND_BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"Backend error {e.response.status_code}: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        raise HTTPException(status_code=502, detail=f"Backend unreachable: {e}")


async def _stream_response(client: httpx.AsyncClient, headers: dict, payload: dict):
    """
    Async generator that proxies SSE lines from the backend to the client.
    Properly forwards [DONE] and handles errors mid-stream.
    """
    try:
        async with client.stream(
            "POST",
            f"{settings.BACKEND_BASE_URL}/v1/chat/completions",
            headers=headers,
            json=payload,
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                logger.error(f"Stream error {resp.status_code}: {body}")
                yield f"data: {{\"error\": \"{resp.status_code}\"}}\n\n"
                return

            async for line in resp.aiter_lines():
                if line:
                    yield f"{line}\n\n"
    except httpx.RequestError as e:
        logger.error(f"Streaming request error: {e}")
        yield f"data: {{\"error\": \"upstream connection failed\"}}\n\n"
