"""``BaseModelClient`` — async interface common to all provider clients.

The base class owns retry, the per-provider semaphore, and the public
:meth:`achat` / :meth:`acomplete` surface. Provider subclasses implement
:meth:`_achat_provider` and override :meth:`_is_retryable` to declare which
exceptions are transient.

Per Q1 from the Tuesday design (``docs/adr/0002-v01-core-abstractions.md``),
the ``raw: dict[str, Any]`` field on :class:`ChatResponse` is the only place
the public surface uses ``Any``. It carries provider-specific debug data and
is not part of the typed contract.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from datetime import date
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


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

    Subclasses implement :meth:`_achat_provider` and may override
    :meth:`_is_retryable`. The base class wraps ``_achat_provider`` with a
    tenacity retry layer and a per-instance semaphore.
    """

    def __init__(self, *, max_concurrent: int = 5, max_retries: int = 5) -> None:
        self.semaphore = asyncio.Semaphore(max_concurrent)
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
        retryable = type(self)._is_retryable
        async with self.semaphore:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=1, min=1, max=30),
                retry=retry_if_exception(retryable),
                reraise=True,
            ):
                with attempt:
                    return await self._achat_provider(messages, model=model, **kwargs)
        raise RuntimeError("unreachable: AsyncRetrying with reraise=True returns or raises")

    async def acomplete(
        self,
        prompt: str,
        *,
        model: str,
        **kwargs: Any,
    ) -> ChatResponse:
        return await self.achat([ChatMessage(role="user", content=prompt)], model=model, **kwargs)
