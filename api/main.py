import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import settings
from core.client import lifespan_client
from routers import chat, embeddings, models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

security = HTTPBearer()


def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> str:
    """
    Validate the Bearer token sent by OpenWebUI (or any other client) to THIS gateway.
    Set GATEWAY_API_KEY in your .env — configure the same value in OpenWebUI.
    """
    if credentials.credentials != settings.GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with lifespan_client():
        yield


app = FastAPI(
    title="LLM Gateway",
    description="OpenAI-compatible proxy with mTLS + token auth to internal LLM backends.",
    version="1.0.0",
    lifespan=lifespan,
)

_auth = [Depends(verify_api_key)]

app.include_router(models.router, dependencies=_auth, tags=["Models"])
app.include_router(chat.router, dependencies=_auth, tags=["Chat"])
app.include_router(embeddings.router, dependencies=_auth, tags=["Embeddings"])


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}
