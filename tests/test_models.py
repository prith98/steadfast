"""Tests for steadfast.models — base client retry, pricing, ChatResponse."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest

from steadfast.models.base import (
    BaseModelClient,
    ChatMessage,
    ChatResponse,
    TokenUsage,
)
from steadfast.models.pricing import PRICING, compute_cost

# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


def test_pricing_table_known_models_present() -> None:
    assert "claude-opus-4-7" in PRICING
    assert "claude-sonnet-4-6" in PRICING
    assert "claude-haiku-4-5-20251001" in PRICING


def test_compute_cost_known_model() -> None:
    # 1M input + 1M output on Opus at 15/75 per million => $15 + $75 = $90
    usage = TokenUsage(input_tokens=1_000_000, output_tokens=1_000_000)
    cost = compute_cost("claude-opus-4-7", usage)
    assert cost == Decimal("90")


def test_compute_cost_partial_million() -> None:
    # 1k input + 500 output on Opus => 1000 * 15 / 1e6 + 500 * 75 / 1e6
    #                                = 0.015 + 0.0375 = 0.0525
    usage = TokenUsage(input_tokens=1000, output_tokens=500)
    cost = compute_cost("claude-opus-4-7", usage)
    assert cost == Decimal("0.0525")


def test_compute_cost_unknown_model_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        compute_cost("not-a-real-model", TokenUsage(input_tokens=1, output_tokens=1))


def test_pricing_dated_at_present_on_all_entries() -> None:
    for model_id, pricing in PRICING.items():
        assert pricing.dated_at is not None, f"{model_id} missing dated_at"


# ---------------------------------------------------------------------------
# ChatResponse round-trip
# ---------------------------------------------------------------------------


def test_chat_response_serializes_decimal_as_string_in_json() -> None:
    response = ChatResponse(
        text="hi",
        usage=TokenUsage(input_tokens=1, output_tokens=2),
        cost_usd=Decimal("0.000017"),
        model="claude-opus-4-7",
        finish_reason="stop",
        raw={"provider_field": "value"},
    )
    payload = response.model_dump_json()
    rebuilt = ChatResponse.model_validate_json(payload)
    assert rebuilt.cost_usd == Decimal("0.000017")
    assert rebuilt.raw == {"provider_field": "value"}


# ---------------------------------------------------------------------------
# BaseModelClient retry behavior
# ---------------------------------------------------------------------------


class _TransientError(Exception):
    """Treated as retryable by _RetryClient."""


class _PermanentError(Exception):
    """Treated as non-retryable by _RetryClient."""


class _RetryClient(BaseModelClient):
    """Fakes a flaky provider: fails ``fail_count`` times before succeeding."""

    def __init__(
        self,
        *,
        fail_count: int,
        exc: type[Exception] = _TransientError,
        max_retries: int = 5,
    ) -> None:
        super().__init__(max_concurrent=2, max_retries=max_retries)
        self.fail_count = fail_count
        self.exc = exc
        self.attempts = 0

    @classmethod
    def _is_retryable(cls, exc: BaseException) -> bool:
        return isinstance(exc, _TransientError)

    async def _achat_provider(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        **kwargs: Any,
    ) -> ChatResponse:
        del messages, kwargs
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise self.exc("transient" if self.exc is _TransientError else "permanent")
        return ChatResponse(
            text="ok",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            cost_usd=Decimal("0.00001"),
            model=model,
            finish_reason="stop",
        )


@pytest.fixture(autouse=True)
def _instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force retry waits to zero so backoff doesn't slow the suite."""
    from tenacity import wait_none

    monkeypatch.setattr("steadfast.models.base.wait_exponential", lambda **_kw: wait_none())


def test_base_client_retries_on_retryable_until_success() -> None:
    client = _RetryClient(fail_count=3, exc=_TransientError)
    response = asyncio.run(client.acomplete("hi", model="claude-opus-4-7"))
    assert response.text == "ok"
    assert client.attempts == 4  # 3 failures + 1 success


def test_base_client_does_not_retry_on_non_retryable() -> None:
    client = _RetryClient(fail_count=1, exc=_PermanentError)
    with pytest.raises(_PermanentError):
        asyncio.run(client.acomplete("hi", model="claude-opus-4-7"))
    assert client.attempts == 1  # immediate raise; no retry


def test_base_client_exhausts_retries_and_reraises() -> None:
    client = _RetryClient(fail_count=10, exc=_TransientError, max_retries=3)
    with pytest.raises(_TransientError):
        asyncio.run(client.acomplete("hi", model="claude-opus-4-7"))
    assert client.attempts == 3  # max_retries=3 -> 3 attempts total
