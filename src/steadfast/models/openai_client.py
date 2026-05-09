"""OpenAI client wrapper — GPT family + embeddings.

Used both as a target-model client (when ``--model`` is a GPT ID) and as
v0.1 benchmark infrastructure (paraphrase generation, ``text-embedding-3-large``
similarity, default rubric judge — see ``docs/adr/0001-infrastructure-model.md``).
"""

from __future__ import annotations

from collections.abc import Sequence
from decimal import Decimal
from typing import Any, ClassVar

import openai
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from steadfast.models.base import (
    BaseModelClient,
    ChatMessage,
    ChatResponse,
    TokenUsage,
)
from steadfast.models.pricing import compute_cost
from steadfast.tracing import embeddings_span, record_embeddings_response

# Default infrastructure embedding model per ADR-0001 (locked for v0.1
# leaderboard comparability). Local users may pass a different model.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"


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

    async def aembed(
        self,
        texts: Sequence[str],
        *,
        model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> tuple[list[list[float]], TokenUsage, Decimal]:
        """Embed ``texts`` and return ``(embeddings, usage, cost_usd)``.

        Embeddings have no output tokens; ``usage.output_tokens`` is always
        zero, and pricing for ``text-embedding-3-large`` is input-only
        (per ``models/pricing.PRICING``). Concurrency is bounded by the
        same per-instance semaphore that ``achat`` uses, so embedding
        bursts cannot starve concurrent chat traffic on the same client.

        Wraps the network call in the same tenacity retry pattern as
        :meth:`BaseModelClient.achat` (ADR-0002 §B.1) — the embeddings
        endpoint is subject to the same rate limits as chat. Emits an
        ``embeddings {model}`` span (ADR-0004 §C / §I).

        Raises :class:`RuntimeError` if the API returns fewer vectors
        than texts — defensive against an undocumented partial response,
        which would otherwise propagate as an :class:`IndexError` from
        the cosine-similarity layer downstream.
        """
        text_list = list(texts)
        retryable = type(self)._is_retryable
        async with self._semaphore:
            with embeddings_span(provider=type(self).PROVIDER_NAME, model=model) as span:
                response = None
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential(multiplier=1, min=1, max=30),
                    retry=retry_if_exception(retryable),
                    reraise=True,
                ):
                    with attempt:
                        response = await self._client.embeddings.create(
                            model=model,
                            input=text_list,
                        )
                if response is None:  # pragma: no cover — AsyncRetrying returns or raises
                    raise RuntimeError("unreachable: AsyncRetrying returned without response")

                if response.usage is None:
                    raise RuntimeError(
                        "OpenAI embeddings response missing 'usage' field — cannot compute cost."
                    )
                if len(response.data) != len(text_list):
                    raise RuntimeError(
                        f"OpenAI embeddings returned {len(response.data)} vectors "
                        f"for {len(text_list)} inputs"
                    )
                usage = TokenUsage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=0,
                )
                cost = compute_cost(model, usage)
                record_embeddings_response(
                    span,
                    response_model=response.model,
                    input_tokens=usage.input_tokens,
                    cost_usd=cost,
                )
                vectors = [d.embedding for d in response.data]
        return vectors, usage, cost
