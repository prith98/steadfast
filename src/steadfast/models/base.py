"""``BaseModelClient`` — async interface common to all provider clients.

The base class owns retry, the per-provider semaphore, and the public
:meth:`achat` / :meth:`acomplete` surface. Provider subclasses implement
:meth:`_achat_provider` and override :meth:`_is_retryable` (which
exceptions are transient) and :data:`PROVIDER_NAME` (the canonical
``gen_ai.provider.name`` value for the provider).

Per Q1 from the Tuesday design (``docs/adr/0002-v01-core-abstractions.md``),
the ``raw: dict[str, Any]`` field on :class:`ChatResponse` is the only place
the public surface uses ``Any``. It carries provider-specific debug data and
is not part of the typed contract.

Per ADR-0003 §A.3, every public :meth:`achat` call emits exactly one
``chat`` span; retries become ``span.add_event("retry", ...)`` events. The
retry contract (ADR-0002 §B.1, §B.2) is unchanged — tracing is purely
observability. If tracing is not configured, the OTel API returns no-op
spans and the call sites still work.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import date
from decimal import Decimal
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from steadfast.tracing import chat_span, record_chat_response, record_retry_event


class ChatMessage(BaseModel):
    """One message in a chat-style request."""

    role: Literal["system", "user", "assistant"]
    content: str


class TokenUsage(BaseModel):
    """Input/output token counts for a single completion."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ModelPricing(BaseModel):
    """Per-million-token pricing snapshot for a model.

    ``dated_at`` lands in the run manifest so reproductions can verify the
    pricing assumption. Update via PR with a new CHANGELOG entry when pricing
    changes; large pricing structure changes warrant an ADR.
    """

    model_config = ConfigDict(frozen=True)

    input_per_mtok: Decimal
    output_per_mtok: Decimal
    dated_at: date


class ChatResponse(BaseModel):
    """Provider-normalized response from :meth:`BaseModelClient.achat`.

    ``raw`` is a deliberate carve-out from the "no ``dict[str, Any]`` in
    public APIs" rule — see ``docs/adr/0002-v01-core-abstractions.md`` Q1.
    Callers should not depend on its structure; it is provider-specific
    debug data, kept on spans and discarded from the run manifest.
    """

    text: str
    usage: TokenUsage
    cost_usd: Decimal
    model: str
    finish_reason: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class BaseModelClient(ABC):
    """Async LLM client base class.

    Subclasses implement :meth:`_achat_provider`, override
    :meth:`_is_retryable`, and set :data:`PROVIDER_NAME` to their canonical
    OTel ``gen_ai.provider.name`` value. The base class wraps
    ``_achat_provider`` with a tenacity retry layer, a per-instance
    semaphore, and a single ``chat`` span per public call.
    """

    # Canonical gen_ai.provider.name — see the OTel registry at
    # https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/.
    # Default is intentionally "unknown" so a subclass that forgets to set
    # it shows up as such in trace tooling, surfacing the bug without
    # raising at import time.
    PROVIDER_NAME: ClassVar[str] = "unknown"

    def __init__(self, *, max_concurrent: int = 5, max_retries: int = 5) -> None:
        # Internal — callers must not bypass the achat contract by acquiring
        # the semaphore directly.
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries

    @classmethod
    def _is_retryable(cls, exc: BaseException) -> bool:
        """Whether ``exc`` should trigger a retry. Default: never."""
        del exc
        return False

    @abstractmethod
    async def _achat_provider(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        **kwargs: Any,
    ) -> ChatResponse:
        """Provider-specific implementation. Must populate ``cost_usd`` via
        :func:`steadfast.models.pricing.compute_cost`.
        """

    async def achat(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        **kwargs: Any,
    ) -> ChatResponse:
        # Retry pattern: tenacity AsyncRetrying with provider-aware exception
        # classification via the ``_is_retryable`` classmethod. Backoff is
        # exponential with a 30-second cap, derived from the standard
        # retry-with-jitter recommendation for cloud APIs (AWS Architecture
        # Blog, "Exponential Backoff And Jitter", 2015 — and the tenacity
        # docs at https://tenacity.readthedocs.io).
        retryable = type(self)._is_retryable
        provider_name = type(self).PROVIDER_NAME

        # Pull request-shape attributes out of kwargs for the span. We
        # accept all three providers' max-tokens spellings — the right one
        # for this provider is already in kwargs and the others won't be.
        request_max_tokens = _coerce_int(
            kwargs.get("max_tokens")
            or kwargs.get("max_completion_tokens")
            or kwargs.get("max_output_tokens")
        )
        request_temperature = _coerce_float(kwargs.get("temperature"))
        request_top_p = _coerce_float(kwargs.get("top_p"))

        async with self._semaphore:
            with chat_span(
                provider=provider_name,
                model=model,
                max_tokens=request_max_tokens,
                temperature=request_temperature,
                top_p=request_top_p,
            ) as span:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self.max_retries),
                    wait=wait_exponential(multiplier=1, min=1, max=30),
                    retry=retry_if_exception(retryable),
                    reraise=True,
                    before_sleep=_make_before_sleep(span),
                ):
                    with attempt:
                        response = await self._achat_provider(messages, model=model, **kwargs)
                        record_chat_response(
                            span,
                            response_model=response.model,
                            input_tokens=response.usage.input_tokens,
                            output_tokens=response.usage.output_tokens,
                            finish_reason=response.finish_reason,
                            cost_usd=response.cost_usd,
                        )
                        return response
        raise RuntimeError("unreachable: AsyncRetrying with reraise=True returns or raises")

    async def acomplete(
        self,
        prompt: str,
        *,
        model: str,
        **kwargs: Any,
    ) -> ChatResponse:
        return await self.achat([ChatMessage(role="user", content=prompt)], model=model, **kwargs)


def _make_before_sleep(span: Any) -> Any:
    """Build a tenacity ``before_sleep`` callback that records retry events.

    ``before_sleep`` fires between a failed attempt and the next one — i.e.,
    only when a retry is actually going to happen. Terminal failures (max
    retries exhausted, or a non-retryable exception) propagate out of the
    ``chat_span`` and become the span's ERROR status; they are not recorded
    here to avoid double-counting.
    """

    def _cb(retry_state: RetryCallState) -> None:
        outcome = retry_state.outcome
        if outcome is None:
            return
        exc = outcome.exception()
        if exc is None:
            return
        record_retry_event(span, attempt=retry_state.attempt_number, exc=exc)

    return _cb


def _coerce_int(value: Any) -> int | None:
    """Best-effort int coercion for span attribute values."""
    if isinstance(value, bool):  # bool is an int subtype — exclude it.
        return None
    if isinstance(value, int):
        return value
    return None


def _coerce_float(value: Any) -> float | None:
    """Best-effort float coercion for span attribute values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None
