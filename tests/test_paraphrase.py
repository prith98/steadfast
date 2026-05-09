"""Tests for steadfast.perturbations.paraphrase — generation + validation + retry."""

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
from steadfast.perturbations.paraphrase import (
    PARAPHRASE_PROMPT_VERSION,
    ParaphraseError,
    _render_generator_prompt,
    _render_validator_prompt,
    generate_paraphrases,
)


@pytest.fixture(autouse=True)
def _instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force tenacity backoff to zero so retries are instant in tests."""
    from tenacity import wait_none

    monkeypatch.setattr("steadfast.models.base.wait_exponential", lambda **_kw: wait_none())


class _ScriptedClient(BaseModelClient):
    """Returns canned outputs in sequence — generator and validator share the wire."""

    PROVIDER_NAME = "test"

    def __init__(self, *, outputs: list[str]) -> None:
        super().__init__(max_concurrent=1, max_retries=1)
        self._outputs = list(outputs)
        self.calls: list[str] = []

    async def _achat_provider(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        **kwargs: Any,
    ) -> ChatResponse:
        del kwargs
        self.calls.append(messages[0].content)
        text = self._outputs.pop(0) if self._outputs else "{}"
        return ChatResponse(
            text=text,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            cost_usd=Decimal("0"),
            model=model,
            finish_reason="stop",
        )


# ---------------------------------------------------------------------------
# Prompt rendering — single-pass, collision-safe
# ---------------------------------------------------------------------------


def test_generator_prompt_substitutes_k_and_original() -> None:
    rendered = _render_generator_prompt(
        template="Make {k} of: {original}",
        k=3,
        original="the dog barked",
    )
    assert rendered == "Make 3 of: the dog barked"


def test_generator_prompt_no_double_substitution() -> None:
    """Original containing literal '{k}' must NOT trigger second pass."""
    rendered = _render_generator_prompt(
        template="K={k} text={original}",
        k=5,
        original="format string {k} is preserved",
    )
    assert rendered == "K=5 text=format string {k} is preserved"


def test_validator_prompt_renders() -> None:
    rendered = _render_validator_prompt(
        template="orig={original} para={paraphrase}",
        original="hello",
        paraphrase="hi there",
    )
    assert rendered == "orig=hello para=hi there"


# ---------------------------------------------------------------------------
# generate_paraphrases — happy path, validator rejection, retry exhaustion
# ---------------------------------------------------------------------------


def _gen_response(paraphrases: list[str]) -> str:
    """Build a generator-shaped JSON response."""
    items = ", ".join(f'"{p}"' for p in paraphrases)
    return f'{{"paraphrases": [{items}]}}'


def _yes(reason: str = "equivalent") -> str:
    return f'{{"equivalent": true, "reason": "{reason}"}}'


def _no(reason: str = "not equivalent") -> str:
    return f'{{"equivalent": false, "reason": "{reason}"}}'


def test_happy_path_first_attempt() -> None:
    """Generator returns K paraphrases, validator approves all → no retry."""
    client = _ScriptedClient(
        outputs=[
            _gen_response(["p1", "p2", "p3"]),
            _yes(),
            _yes(),
            _yes(),
        ]
    )
    result = asyncio.run(
        generate_paraphrases(original="orig", k=3, client=client, model="gpt-test")
    )
    assert result.paraphrases == ["p1", "p2", "p3"]
    assert result.requested_k == 3
    assert result.accepted == 3
    assert result.rejected == 0
    assert result.rejection_rate == 0.0
    assert result.prompt_version == PARAPHRASE_PROMPT_VERSION


def test_validator_rejection_triggers_regeneration() -> None:
    """First gen produces 3, validator rejects 1; second gen produces the missing 1."""
    client = _ScriptedClient(
        outputs=[
            _gen_response(["p1", "p2", "p3"]),
            _yes(),
            _no(),  # p2 rejected
            _yes(),
            _gen_response(["p4"]),  # second gen for the missing one
            _yes(),
        ]
    )
    result = asyncio.run(
        generate_paraphrases(original="orig", k=3, client=client, model="gpt-test")
    )
    assert len(result.paraphrases) == 3
    assert "p2" not in result.paraphrases
    assert result.accepted == 3
    assert result.rejected == 1
    assert result.rejection_rate == pytest.approx(1 / 4)


def test_max_retries_exhausted_raises() -> None:
    """Validator rejects everything for max_retries passes → ParaphraseError."""
    # Generator yields 3 paraphrases per pass; validator says no every time.
    client = _ScriptedClient(
        outputs=[
            _gen_response(["a", "b", "c"]),
            _no(),
            _no(),
            _no(),
            _gen_response(["d", "e", "f"]),
            _no(),
            _no(),
            _no(),
        ]
    )
    with pytest.raises(ParaphraseError, match="could not generate"):
        asyncio.run(
            generate_paraphrases(
                original="orig", k=3, client=client, model="gpt-test", max_retries=2
            )
        )


def test_unparseable_generator_output_skips_to_next_attempt() -> None:
    """Generator returns garbage → that attempt yields zero paraphrases; next attempt salvages."""
    client = _ScriptedClient(
        outputs=[
            "not json at all",  # first gen junk → 0 candidates
            _gen_response(["x", "y"]),  # second gen succeeds
            _yes(),
            _yes(),
        ]
    )
    result = asyncio.run(
        generate_paraphrases(original="orig", k=2, client=client, model="gpt-test", max_retries=3)
    )
    assert result.paraphrases == ["x", "y"]


def test_unparseable_validator_treats_as_rejection() -> None:
    """Validator returns garbage → conservative reject; metric absorbs as rejected."""
    client = _ScriptedClient(
        outputs=[
            _gen_response(["p1", "p2"]),
            "junk validator output",  # treated as reject
            _yes(),  # p2 ok
            _gen_response(["p3"]),
            _yes(),
        ]
    )
    result = asyncio.run(
        generate_paraphrases(original="orig", k=2, client=client, model="gpt-test", max_retries=2)
    )
    assert len(result.paraphrases) == 2
    assert result.rejected >= 1


def test_invalid_k_raises() -> None:
    client = _ScriptedClient(outputs=[])
    with pytest.raises(ValueError, match="k must be"):
        asyncio.run(generate_paraphrases(original="x", k=0, client=client, model="gpt-test"))


def test_invalid_max_retries_raises() -> None:
    client = _ScriptedClient(outputs=[])
    with pytest.raises(ValueError, match="max_retries"):
        asyncio.run(
            generate_paraphrases(original="x", k=2, client=client, model="gpt-test", max_retries=0)
        )
