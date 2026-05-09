"""Tests for steadfast.perturbations.confidence — parser + agent integration."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest

from steadfast.agent import AgentResponse, SimplePromptingAgent, Task
from steadfast.models.base import (
    BaseModelClient,
    ChatMessage,
    ChatResponse,
    TokenUsage,
)
from steadfast.perturbations.confidence import (
    CONFIDENCE_PROMPT_VERSION,
    ParsedConfidence,
    _normalize_confidence_value,
    load_confidence_suffix_v1,
    parse_verbalized_confidence,
)


@pytest.fixture(autouse=True)
def _instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force tenacity backoff to zero so retries are instant in tests."""
    from tenacity import wait_none

    monkeypatch.setattr("steadfast.models.base.wait_exponential", lambda **_kw: wait_none())


# ---------------------------------------------------------------------------
# Frozen prompt
# ---------------------------------------------------------------------------


def test_frozen_prompt_loads() -> None:
    suffix = load_confidence_suffix_v1()
    assert "ANSWER:" in suffix
    assert "CONFIDENCE:" in suffix
    assert "REFUSE" in suffix
    # The version constant should match the filename suffix.
    assert CONFIDENCE_PROMPT_VERSION == "v1"


def test_frozen_prompt_is_cached() -> None:
    a = load_confidence_suffix_v1()
    b = load_confidence_suffix_v1()
    # Identity equality — module-level cache, not a re-read each call.
    assert a is b


# ---------------------------------------------------------------------------
# _normalize_confidence_value — accepts a few forms; rejects out of range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0.85", 0.85),
        (".85", 0.85),
        ("0,85", 0.85),  # comma decimal
        ("85%", 0.85),
        ("85.0%", 0.85),
        ("85", 0.85),  # bare percent integer
        ("0", 0.0),
        ("1", 1.0),
        ("1.0", 1.0),
        ("0.0", 0.0),
    ],
)
def test_normalize_confidence_accepted(raw: str, expected: float) -> None:
    assert _normalize_confidence_value(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "abc", "1.5", "-0.1", "150", "150%"])
def test_normalize_confidence_rejected(raw: str) -> None:
    assert _normalize_confidence_value(raw) is None


# ---------------------------------------------------------------------------
# parse_verbalized_confidence — happy path and edge cases
# ---------------------------------------------------------------------------


def test_parse_basic_happy_path() -> None:
    text = "Some preamble.\n\nANSWER: 30 days for unopened items.\nCONFIDENCE: 0.92"
    result = parse_verbalized_confidence(text)
    assert isinstance(result, ParsedConfidence)
    assert result.parse_ok is True
    assert result.answer == "30 days for unopened items."
    assert result.confidence == pytest.approx(0.92)
    assert result.refused is False


def test_parse_returns_none_on_missing_confidence() -> None:
    text = "ANSWER: I think it's 30 days.\n(No confidence emitted.)"
    result = parse_verbalized_confidence(text)
    assert result.parse_ok is False
    assert result.confidence is None
    # answer falls back to the raw text so callers can still surface something.
    assert "30 days" in result.answer


def test_parse_returns_none_on_out_of_range() -> None:
    text = "ANSWER: foo\nCONFIDENCE: 1.7"
    result = parse_verbalized_confidence(text)
    assert result.parse_ok is False
    assert result.confidence is None
    assert "foo" in result.answer


def test_parse_handles_multi_line_answer() -> None:
    text = (
        "ANSWER: The return window is 30 days.\n"
        "This applies to unopened items only.\n"
        "CONFIDENCE: 0.9"
    )
    result = parse_verbalized_confidence(text)
    assert result.parse_ok is True
    assert "30 days" in result.answer
    assert "unopened items only" in result.answer
    assert result.confidence == pytest.approx(0.9)


def test_parse_last_label_wins() -> None:
    """Models that echo the format header in prose must not shadow the tail."""
    text = (
        "I'll write the answer on the ANSWER: line and the probability on CONFIDENCE:.\n\n"
        "ANSWER: 42\n"
        "CONFIDENCE: 0.6"
    )
    result = parse_verbalized_confidence(text)
    assert result.parse_ok is True
    assert result.answer == "42"
    assert result.confidence == pytest.approx(0.6)


def test_parse_refuse_token() -> None:
    text = "ANSWER: REFUSE\nCONFIDENCE: 0.0"
    result = parse_verbalized_confidence(text)
    assert result.refused is True
    assert result.parse_ok is True
    assert result.confidence == 0.0


def test_parse_refuse_with_punctuation() -> None:
    text = 'ANSWER: "REFUSE."\nCONFIDENCE: 0.0'
    result = parse_verbalized_confidence(text)
    assert result.refused is True


def test_parse_lowercase_labels() -> None:
    """Real-world models often emit lowercase labels; we accept them."""
    text = "answer: foo bar\nconfidence: 0.5"
    result = parse_verbalized_confidence(text)
    assert result.parse_ok is True
    assert result.answer == "foo bar"
    assert result.confidence == pytest.approx(0.5)


def test_parse_percent_form() -> None:
    text = "ANSWER: 30 days\nCONFIDENCE: 92%"
    result = parse_verbalized_confidence(text)
    assert result.parse_ok is True
    assert result.confidence == pytest.approx(0.92)


def test_parse_empty_returns_none() -> None:
    result = parse_verbalized_confidence("")
    assert result.parse_ok is False
    assert result.answer == ""
    assert result.confidence is None
    assert result.refused is False


def test_parse_falls_back_to_prefix_when_no_answer_label() -> None:
    """Some models emit only CONFIDENCE: without an ANSWER: label.

    We fall back to "everything before CONFIDENCE: is the answer" so we
    don't lose information when only the confidence label is present.
    """
    text = "30 days for unopened items.\nCONFIDENCE: 0.7"
    result = parse_verbalized_confidence(text)
    assert result.parse_ok is True
    assert "30 days" in result.answer
    assert result.confidence == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# SimplePromptingAgent integration — confidence + retry-once-then-soft-fail
# ---------------------------------------------------------------------------


class _ScriptedClient(BaseModelClient):
    """Returns canned chat outputs in sequence; records call count."""

    PROVIDER_NAME = "test"

    def __init__(self, *, outputs: list[str], avg_logprob: float | None = None) -> None:
        super().__init__(max_concurrent=1, max_retries=1)
        self._outputs = list(outputs)
        self._avg_logprob = avg_logprob
        self.calls: list[str] = []
        self.last_kwargs: dict[str, Any] | None = None

    async def _achat_provider(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        **kwargs: Any,
    ) -> ChatResponse:
        self.calls.append(messages[0].content)
        self.last_kwargs = kwargs
        text = self._outputs.pop(0) if self._outputs else "(scripted exhausted)"
        return ChatResponse(
            text=text,
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            cost_usd=Decimal("0"),
            model=model,
            finish_reason="stop",
            avg_logprob=self._avg_logprob,
        )


def test_agent_no_confidence_suffix_returns_raw_answer() -> None:
    """No suffix → no parsing; answer is the raw text and confidence is None."""
    client = _ScriptedClient(outputs=["just the answer"])
    agent = SimplePromptingAgent(client=client, model="gpt-test")
    task = Task(id="t1", domain="d", input="What is 2+2?")
    response = asyncio.run(agent.arun(task))
    assert response.answer == "just the answer"
    assert response.confidence is None
    assert response.refused is False
    assert len(client.calls) == 1


def test_agent_parses_confidence_when_suffix_set() -> None:
    client = _ScriptedClient(
        outputs=["ANSWER: 4\nCONFIDENCE: 0.95"],
        avg_logprob=-0.3,
    )
    agent = SimplePromptingAgent(client=client, model="gpt-test")
    task = Task(
        id="t1",
        domain="d",
        input="What is 2+2?",
        confidence_suffix=load_confidence_suffix_v1(),
    )
    response = asyncio.run(agent.arun(task))
    assert response.answer == "4"
    assert response.confidence == pytest.approx(0.95)
    assert response.refused is False
    assert response.logprob_avg == pytest.approx(-0.3)
    assert len(client.calls) == 1
    assert response.metadata["elicitation_attempts"] == 1
    assert response.metadata["elicitation_parse_ok"] is True


def test_agent_passes_logprobs_kwarg_through_to_client() -> None:
    """SimplePromptingAgent must pass logprobs=True so OpenAI populates them."""
    client = _ScriptedClient(outputs=["ANSWER: foo\nCONFIDENCE: 0.5"])
    agent = SimplePromptingAgent(client=client, model="gpt-test")
    task = Task(id="t1", domain="d", input="x", confidence_suffix="suffix")
    asyncio.run(agent.arun(task))
    assert client.last_kwargs is not None
    assert client.last_kwargs.get("logprobs") is True


def test_agent_retries_once_on_parse_failure_then_succeeds() -> None:
    client = _ScriptedClient(
        outputs=[
            "I think it's 4 but I'm not sure",  # no CONFIDENCE label
            "ANSWER: 4\nCONFIDENCE: 0.85",
        ]
    )
    agent = SimplePromptingAgent(client=client, model="gpt-test")
    task = Task(id="t1", domain="d", input="x", confidence_suffix="suffix")
    response = asyncio.run(agent.arun(task))
    assert response.confidence == pytest.approx(0.85)
    assert len(client.calls) == 2
    # The retry call must include the stricter reminder.
    assert "previous output did not include" in client.calls[1]
    assert response.metadata["elicitation_attempts"] == 2
    assert response.metadata["elicitation_parse_ok"] is True


def test_agent_soft_fails_after_two_failures() -> None:
    """Two parse failures → rep stays COMPLETED with confidence=None."""
    client = _ScriptedClient(
        outputs=[
            "no labels first attempt",
            "no labels second attempt either",
        ]
    )
    agent = SimplePromptingAgent(client=client, model="gpt-test")
    task = Task(id="t1", domain="d", input="x", confidence_suffix="suffix")
    response = asyncio.run(agent.arun(task))
    assert response.confidence is None
    assert response.refused is False
    assert len(client.calls) == 2  # one initial + one retry, no third attempt
    assert response.metadata["elicitation_attempts"] == 2
    assert response.metadata["elicitation_parse_ok"] is False
    # Raw text preserved so the rep still contributes to consistency etc.
    assert "second attempt" in response.answer


def test_agent_propagates_refusal() -> None:
    client = _ScriptedClient(outputs=["ANSWER: REFUSE\nCONFIDENCE: 0.0"])
    agent = SimplePromptingAgent(client=client, model="gpt-test")
    task = Task(id="t1", domain="d", input="x", confidence_suffix="suffix")
    response = asyncio.run(agent.arun(task))
    assert response.refused is True
    assert response.confidence == 0.0


def test_agent_response_round_trips_with_new_fields() -> None:
    """logprob_avg and refused must serialize and round-trip."""
    r = AgentResponse(answer="ok", confidence=0.7, refused=False, logprob_avg=-0.5)
    rebuilt = AgentResponse.model_validate_json(r.model_dump_json())
    assert rebuilt.refused is False
    assert rebuilt.logprob_avg == pytest.approx(-0.5)


def test_task_difficulty_default_is_normal() -> None:
    t = Task(id="t1", domain="d", input="x")
    assert t.difficulty == "normal"


def test_task_difficulty_hard_round_trip() -> None:
    t = Task(id="t1", domain="d", input="x", difficulty="hard")
    rebuilt = Task.model_validate_json(t.model_dump_json())
    assert rebuilt.difficulty == "hard"
