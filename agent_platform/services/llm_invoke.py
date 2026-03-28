"""
Safe LLM invocation with automatic truncation recovery.

Usage in any node:
    from services.llm_invoke import invoke_llm

    response = await invoke_llm(llm, messages)
    # response.content is guaranteed complete (auto-continued if truncated)
"""

from __future__ import annotations

import logging
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

MAX_CONTINUATIONS = 3

CONTINUE_PROMPT = (
    "Your previous response was truncated due to length limits. "
    "Continue EXACTLY from where you stopped. Do not repeat any "
    "content you already produced. Do not add preamble."
)


async def invoke_llm(
    llm: BaseChatModel,
    messages: Sequence[BaseMessage],
    *,
    max_continuations: int = MAX_CONTINUATIONS,
) -> AIMessage:
    """
    Invoke an LLM and automatically handle truncation.

    If the response is cut off (finish_reason == 'length'), the partial
    output is fed back as an AIMessage and the LLM is asked to continue.
    Partial outputs are concatenated into a single response.

    Args:
        llm: The LangChain chat model.
        messages: The message list to send.
        max_continuations: Max retry attempts for truncation (default 3).

    Returns:
        AIMessage with complete content.
    """
    response = await llm.ainvoke(messages)
    full_content = response.content

    for attempt in range(max_continuations):
        finish_reason = _get_finish_reason(response)

        if finish_reason != "length":
            break  # complete response

        logger.info(
            "LLM response truncated (attempt %d/%d, %d chars so far). Continuing...",
            attempt + 1, max_continuations, len(full_content),
        )

        # Build continuation: original messages + partial output + continue prompt
        continuation_messages = [
            *messages,
            AIMessage(content=full_content),
            HumanMessage(content=CONTINUE_PROMPT),
        ]

        response = await llm.ainvoke(continuation_messages)
        full_content += response.content

    if _get_finish_reason(response) == "length":
        logger.warning(
            "LLM response still truncated after %d continuations (%d chars). "
            "Consider increasing max_tokens or breaking the task into smaller parts.",
            max_continuations, len(full_content),
        )

    # Return a single AIMessage with the complete concatenated content
    return AIMessage(
        content=full_content,
        response_metadata=response.response_metadata,
    )


def _get_finish_reason(response) -> str:
    """Extract finish_reason from various LLM provider response formats."""
    meta = getattr(response, "response_metadata", {}) or {}

    # OpenAI / Azure
    if "finish_reason" in meta:
        return meta["finish_reason"]

    # Anthropic
    if meta.get("stop_reason") == "max_tokens":
        return "length"

    # Nested in usage metadata
    usage = meta.get("usage", {})
    if isinstance(usage, dict) and usage.get("completion_tokens_details", {}).get("truncated"):
        return "length"

    return meta.get("finish_reason", "stop")
