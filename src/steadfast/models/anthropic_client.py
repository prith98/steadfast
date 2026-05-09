"""Anthropic client wrapper — Claude family.

Backed by ``anthropic.AsyncAnthropic``. Logprob-derived confidence is limited
on Anthropic's API; ``docs/METHODOLOGY.md`` §3.1 documents the asymmetry
across providers.
"""

from __future__ import annotations

from typing import Any, ClassVar

import anthropic

from steadfast.models.base import (
    BaseModelClient,
    ChatMessage,
    ChatResponse,
    TokenUsage,
)
from steadfast.models.pricing import compute_cost


class AnthropicClient(BaseModelClient):
    """:class:`BaseModelClient` implementation backed by ``anthropic.AsyncAnthropic``."""

    # gen_ai.provider.name canonical value per OTel semconv registry.
    PROVIDER_NAME: ClassVar[str] = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_concurrent: int = 5,
        max_retries: int = 5,
    ) -> None:
        super().__init__(max_concurrent=max_concurrent, max_retries=max_retries)
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    @classmethod
    def _is_retryable(cls, exc: BaseException) -> bool:
        if isinstance(exc, anthropic.RateLimitError | anthropic.APIConnectionError):
            return True
        if isinstance(exc, anthropic.APIStatusError):
            return exc.status_code >= 500
        return False

    async def _achat_provider(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> ChatResponse:
        # Anthropic separates the system prompt from the message list.
        system_msgs = [m.content for m in messages if m.role == "system"]
        user_assistant = [
            {"role": m.role, "content": m.content} for m in messages if m.role != "system"
        ]
        system = "\n\n".join(system_msgs) if system_msgs else None

        api_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": user_assistant,
            **kwargs,
        }
        if system:
            api_kwargs["system"] = system

        response = await self._client.messages.create(**api_kwargs)

        text = "".join(getattr(block, "text", "") for block in response.content)
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        cost = compute_cost(model, usage)

        return ChatResponse(
            text=text,
            usage=usage,
            cost_usd=cost,
            model=response.model,
            finish_reason=response.stop_reason,
            raw=response.model_dump(),
        )
