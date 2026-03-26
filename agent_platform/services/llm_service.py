"""
LLM provider abstraction.

Supports OpenAI, Azure OpenAI, and Anthropic.
Provides both a primary model (for complex reasoning) and a fast model
(for classification, validation, and lightweight tasks).
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from config import Settings

logger = logging.getLogger(__name__)


def build_llm(
    settings: Settings,
    *,
    fast: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """
    Build the appropriate LangChain chat model based on configuration.

    Args:
        settings: Application settings.
        fast: If True, use the lighter/faster model for cheap tasks.
        temperature: Override default temperature.
        max_tokens: Override default max_tokens.
    """
    model_name = settings.llm_fast_model if fast else settings.llm_model
    temp = temperature if temperature is not None else settings.llm_temperature
    tokens = max_tokens or settings.llm_max_tokens
    api_key = settings.llm_api_key.get_secret_value()

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=temp,
            max_tokens=tokens,
            request_timeout=settings.llm_request_timeout,
        )

    elif settings.llm_provider == "azure":
        from langchain_openai import AzureChatOpenAI

        llm = AzureChatOpenAI(
            deployment_name=model_name,
            api_key=api_key,
            temperature=temp,
            max_tokens=tokens,
            request_timeout=settings.llm_request_timeout,
        )

    elif settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model_name=model_name,
            api_key=api_key,
            temperature=temp,
            max_tokens=tokens,
            timeout=settings.llm_request_timeout,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")

    logger.info("LLM initialized: provider=%s model=%s", settings.llm_provider, model_name)
    return llm


class LLMService:
    """
    Holds both the primary and fast LLM instances.
    Injected into agents via FastAPI app.state.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.primary: BaseChatModel | None = None
        self.fast: BaseChatModel | None = None

    def initialize(self) -> None:
        self.primary = build_llm(self.settings, fast=False)
        self.fast = build_llm(self.settings, fast=True)

    def get_model(self, *, fast: bool = False) -> BaseChatModel:
        model = self.fast if fast else self.primary
        if model is None:
            raise RuntimeError("LLMService not initialized")
        return model
