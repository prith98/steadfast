"""OpenAI client wrapper — GPT family + embeddings.

Used both as a target-model client (when ``--model`` is a GPT ID) and as
v0.1 benchmark infrastructure (paraphrase generation, ``text-embedding-3-large``
similarity, default rubric judge — see ``docs/adr/0001-infrastructure-model.md``).
"""

from __future__ import annotations

from typing import Any, ClassVar

import openai

from steadfast.models.base import (
    BaseModelClient,
    ChatMessage,
    ChatResponse,
    TokenUsage,
)
from steadfast.models.pricing import compute_cost


class OpenAIClient(BaseModelClient):
    """:class:`BaseModelClient` implementation backed by ``openai.AsyncOpenAI``."""

    # gen_ai.provider.name canonical value per OTel semconv registry.
    PROVIDER_NAME: ClassVar[str] = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        max_concurrent: int = 5,
        max_retries: int = 5,
    ) -> None:
        super().__init__(max_concurrent=max_concurrent, max_retries=max_retries)
        self._client = openai.AsyncOpenAI(api_key=api_key)

    @classmethod
    def _is_retryable(cls, exc: BaseException) -> bool:
        if isinstance(exc, openai.RateLimitError | openai.APIConnectionError):
            return True
        if isinstance(exc, openai.APIStatusError):
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
        api_messages = [{"role": m.role, "content": m.content} for m in messages]

        response = await self._client.chat.completions.create(
            model=model,
            messages=api_messages,  # type: ignore[arg-type]
            max_completion_tokens=max_tokens,
            **kwargs,
        )

        choice = response.choices[0]
        text = choice.message.content or ""
        if response.usage is None:
            # Silent zero-cost fallback would corrupt manifest cost accounting.
            # In non-streaming chat completions the API always returns usage;
            # if it ever doesn't, fail loudly per ADR-0002 §C.3.
            raise RuntimeError(
                "OpenAI response missing 'usage' field — cannot compute cost. "
                "If streaming, set stream_options={'include_usage': True}."
            )
        usage = TokenUsage(
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        cost = compute_cost(model, usage)

        return ChatResponse(
            text=text,
            usage=usage,
            cost_usd=cost,
            model=response.model,
            finish_reason=choice.finish_reason,
            raw=response.model_dump(),
        )
