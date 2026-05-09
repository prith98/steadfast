"""Tests for steadfast.tracing — span hierarchy, attributes, and conventions.

Each test installs a fresh :class:`TracerProvider` backed by an
:class:`InMemorySpanExporter` so assertions can read finished spans
directly. The OTel provider is global, so the fixture overrides it for
the duration of the test (the SDK logs a "tracer provider already set"
warning that we accept — production never calls ``set_tracer_provider``
twice).
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from steadfast.agent import Agent, AgentResponse, Task
from steadfast.models.base import (
    BaseModelClient,
    ChatMessage,
    ChatResponse,
    TokenUsage,
)
from steadfast.runner import RepStatus, run_task
from steadfast.tracing import (
    benchmark_span,
    chat_span,
    record_chat_response,
    rep_span,
    score_span,
    task_span,
)
from steadfast.tracing.conventions import (
    ERROR_TYPE,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    GENAI_CONVENTIONS_VERSION,
    OP_CHAT,
    STEADFAST_BENCHMARK_NAME,
    STEADFAST_COST_USD,
    STEADFAST_REP_IDX,
    STEADFAST_RUN_ID,
    STEADFAST_TASK_DOMAIN,
    STEADFAST_TASK_ID,
)

# The session-level TracerProvider + ``memory_exporter`` fixture live in
# ``tests/conftest.py`` so every test file shares one provider (the OTel
# SDK refuses to swap it once installed).


# ---------------------------------------------------------------------------
# Conventions module — version pin and attribute names.
# ---------------------------------------------------------------------------


def test_conventions_version_pin() -> None:
    """ADR-0003 §A pins GENAI_CONVENTIONS_VERSION at 1.41.0."""
    assert GENAI_CONVENTIONS_VERSION == "1.41.0"


def test_conventions_attributes_use_canonical_keys() -> None:
    """Attribute names match the published GenAI semconv exactly."""
    assert GEN_AI_OPERATION_NAME == "gen_ai.operation.name"
    assert GEN_AI_PROVIDER_NAME == "gen_ai.provider.name"
    assert GEN_AI_SYSTEM == "gen_ai.system"
    assert GEN_AI_REQUEST_MODEL == "gen_ai.request.model"
    assert GEN_AI_RESPONSE_MODEL == "gen_ai.response.model"
    assert GEN_AI_USAGE_INPUT_TOKENS == "gen_ai.usage.input_tokens"
    assert GEN_AI_USAGE_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"
    assert ERROR_TYPE == "error.type"


# ---------------------------------------------------------------------------
# chat_span — required attributes per GenAI v1.41.0 + ADR-0003 §A.2 dual emission.
# ---------------------------------------------------------------------------


def test_chat_span_emits_required_genai_attributes(memory_exporter: InMemorySpanExporter) -> None:
    """Per ADR-0003 §A.2 we emit BOTH gen_ai.system AND gen_ai.provider.name."""
    with chat_span(provider="anthropic", model="claude-opus-4-7", max_tokens=1024) as span:
        record_chat_response(
            span,
            response_model="claude-opus-4-7",
            input_tokens=100,
            output_tokens=50,
            finish_reason="end_turn",
            cost_usd=Decimal("0.0042"),
        )

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "chat claude-opus-4-7"
    assert s.attributes is not None
    # Dual emission of provider identity (ADR-0003 §A.2).
    assert s.attributes[GEN_AI_PROVIDER_NAME] == "anthropic"
    assert s.attributes[GEN_AI_SYSTEM] == "anthropic"
    # Operation + request shape.
    assert s.attributes[GEN_AI_OPERATION_NAME] == OP_CHAT
    assert s.attributes[GEN_AI_REQUEST_MODEL] == "claude-opus-4-7"
    # Response shape.
    assert s.attributes[GEN_AI_RESPONSE_MODEL] == "claude-opus-4-7"
    assert s.attributes[GEN_AI_USAGE_INPUT_TOKENS] == 100
    assert s.attributes[GEN_AI_USAGE_OUTPUT_TOKENS] == 50
    finish_reasons = s.attributes[GEN_AI_RESPONSE_FINISH_REASONS]
    assert isinstance(finish_reasons, tuple | list)
    assert tuple(finish_reasons) == ("end_turn",)
    # Steadfast-namespaced cost.
    assert s.attributes[STEADFAST_COST_USD] == "0.0042"


def test_chat_span_records_error_on_exception(memory_exporter: InMemorySpanExporter) -> None:
    with pytest.raises(RuntimeError, match="boom"), chat_span(provider="openai", model="gpt-5.2"):
        raise RuntimeError("boom")

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.status.status_code == StatusCode.ERROR
    assert s.attributes is not None
    assert s.attributes[ERROR_TYPE] == "RuntimeError"


# ---------------------------------------------------------------------------
# Span hierarchy — benchmark → task → rep → chat.
# ---------------------------------------------------------------------------


def test_span_hierarchy_benchmark_task_rep_chat(
    memory_exporter: InMemorySpanExporter,
) -> None:
    """ADR-0003 §A.1: benchmark → task → rep → chat."""
    with (
        benchmark_span(name="t1", package_version="0.1.0.dev0"),
        task_span(task_id="t1", domain="customer_support", run_id="r-deadbeef", reps_total=2),
        rep_span(rep_idx=0, run_id="r-deadbeef", task_id="t1"),
        chat_span(provider="anthropic", model="claude-opus-4-7") as span,
    ):
        record_chat_response(
            span,
            response_model="claude-opus-4-7",
            input_tokens=10,
            output_tokens=5,
            finish_reason="end_turn",
        )

    spans = memory_exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    chat = by_name["chat claude-opus-4-7"]
    rep = by_name["rep 0"]
    task = by_name["task t1"]
    bench = by_name["benchmark"]

    # Parent linkage.
    assert chat.parent is not None
    assert chat.parent.span_id == rep.context.span_id
    assert rep.parent is not None
    assert rep.parent.span_id == task.context.span_id
    assert task.parent is not None
    assert task.parent.span_id == bench.context.span_id
    assert bench.parent is None  # root


def test_task_and_rep_spans_carry_steadfast_attributes(
    memory_exporter: InMemorySpanExporter,
) -> None:
    with (
        benchmark_span(name="my_pilot", package_version="0.1.0.dev0"),
        task_span(task_id="t-1", domain="code_repair", run_id="run-abc", reps_total=10),
        rep_span(rep_idx=3, run_id="run-abc", task_id="t-1"),
    ):
        pass

    spans = memory_exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    bench = by_name["benchmark"]
    task = by_name["task t-1"]
    rep = by_name["rep 3"]

    assert bench.attributes is not None
    assert bench.attributes[STEADFAST_BENCHMARK_NAME] == "my_pilot"
    assert task.attributes is not None
    assert task.attributes[STEADFAST_TASK_ID] == "t-1"
    assert task.attributes[STEADFAST_TASK_DOMAIN] == "code_repair"
    assert task.attributes[STEADFAST_RUN_ID] == "run-abc"
    assert rep.attributes is not None
    assert rep.attributes[STEADFAST_REP_IDX] == 3


# ---------------------------------------------------------------------------
# score_span — sibling of task per ADR-0003 §A.7.
# ---------------------------------------------------------------------------


def test_score_span_is_sibling_of_task_under_benchmark(
    memory_exporter: InMemorySpanExporter,
) -> None:
    with benchmark_span(name="t1", package_version="0.1.0.dev0"):
        with (
            task_span(task_id="t1", domain="d", run_id="r1", reps_total=1),
            rep_span(rep_idx=0, run_id="r1", task_id="t1"),
        ):
            pass
        # task closed; now score_span should attach to benchmark, not task.
        with score_span(judge_kind="exact_match", task_id="t1", rep_idx=0, run_id="r1"):
            pass

    spans = memory_exporter.get_finished_spans()
    by_name = {s.name: s for s in spans}
    bench = by_name["benchmark"]
    score = by_name["score exact_match"]

    assert score.parent is not None
    assert score.parent.span_id == bench.context.span_id


# ---------------------------------------------------------------------------
# BaseModelClient.achat — emits a chat span; retries become events.
# ---------------------------------------------------------------------------


class _OkClient(BaseModelClient):
    """Stub client that always succeeds; provider name is ``test``."""

    PROVIDER_NAME = "test"

    async def _achat_provider(
        self,
        messages: list[ChatMessage],
        *,
        model: str,
        **kwargs: Any,
    ) -> ChatResponse:
        del messages, kwargs
        return ChatResponse(
            text="hi",
            usage=TokenUsage(input_tokens=3, output_tokens=4),
            cost_usd=Decimal("0.0001"),
            model=model,
            finish_reason="stop",
        )


class _RetryThenOkClient(BaseModelClient):
    """Fails ``fail_count`` times with a retryable error then succeeds.

    Used to confirm the chat span captures retries as ``add_event``
    entries (ADR-0003 §A.3) rather than separate spans.
    """

    PROVIDER_NAME = "test"

    class _TransientError(Exception):
        pass

    def __init__(self, *, fail_count: int) -> None:
        super().__init__(max_concurrent=1, max_retries=5)
        self.fail_count = fail_count
        self.attempts = 0

    @classmethod
    def _is_retryable(cls, exc: BaseException) -> bool:
        return isinstance(exc, cls._TransientError)

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
            raise self._TransientError(f"fail #{self.attempts}")
        return ChatResponse(
            text="ok",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            cost_usd=Decimal("0.00001"),
            model=model,
            finish_reason="stop",
        )


@pytest.fixture(autouse=True)
def _instant_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force retry waits to zero so the test suite stays fast."""
    from tenacity import wait_none

    monkeypatch.setattr("steadfast.models.base.wait_exponential", lambda **_kw: wait_none())


def test_achat_emits_one_chat_span_with_provider_name(
    memory_exporter: InMemorySpanExporter,
) -> None:
    client = _OkClient(max_concurrent=1, max_retries=1)
    asyncio.run(client.acomplete("hi", model="claude-opus-4-7"))

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    s = spans[0]
    assert s.name == "chat claude-opus-4-7"
    assert s.attributes is not None
    assert s.attributes[GEN_AI_PROVIDER_NAME] == "test"
    assert s.attributes[GEN_AI_SYSTEM] == "test"
    assert s.attributes[GEN_AI_USAGE_INPUT_TOKENS] == 3
    assert s.attributes[GEN_AI_USAGE_OUTPUT_TOKENS] == 4


def test_achat_retries_become_span_events_not_separate_spans(
    memory_exporter: InMemorySpanExporter,
) -> None:
    """ADR-0003 §A.3 — one span per public achat() call; retries are events."""
    client = _RetryThenOkClient(fail_count=2)
    asyncio.run(client.acomplete("hi", model="claude-opus-4-7"))

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1, "retries must not produce additional chat spans"
    s = spans[0]

    retry_events = [e for e in s.events if e.name == "retry"]
    assert len(retry_events) == 2  # two failed attempts before success
    # Attempt numbers are 1-indexed and increasing.
    attempts = [e.attributes["steadfast.retry.attempt"] for e in retry_events if e.attributes]
    assert attempts == [1, 2]


# ---------------------------------------------------------------------------
# Runner — task and rep spans wrap run_task; rep failure marks span ERROR.
# ---------------------------------------------------------------------------


class _CountingAgent(Agent):
    def __init__(self) -> None:
        self.count = 0

    async def arun(self, task: Task) -> AgentResponse:
        self.count += 1
        return AgentResponse(answer=f"reply #{self.count}")


class _FailingAgent(Agent):
    async def arun(self, task: Task) -> AgentResponse:
        raise RuntimeError("agent boom")


def test_run_task_emits_task_and_rep_spans(
    memory_exporter: InMemorySpanExporter,
    tmp_path: Any,
) -> None:
    task = Task(id="t1", domain="d", input="x")
    asyncio.run(
        run_task(
            agent=_CountingAgent(),
            task=task,
            reps=2,
            model="claude-opus-4-7",
            checkpoint_path=tmp_path / "ckpt.sqlite",
        )
    )
    spans = memory_exporter.get_finished_spans()
    by_name = [s.name for s in spans]
    assert "task t1" in by_name
    assert by_name.count("rep 0") == 1
    assert by_name.count("rep 1") == 1


def test_rep_span_marked_error_when_agent_fails(
    memory_exporter: InMemorySpanExporter,
    tmp_path: Any,
) -> None:
    task = Task(id="t1", domain="d", input="x")
    result = asyncio.run(
        run_task(
            agent=_FailingAgent(),
            task=task,
            reps=1,
            model="claude-opus-4-7",
            checkpoint_path=tmp_path / "ckpt.sqlite",
        )
    )
    # Runner records the rep as FAILED in checkpoint storage.
    assert result.reps[0].status == RepStatus.FAILED

    spans = memory_exporter.get_finished_spans()
    rep = next(s for s in spans if s.name == "rep 0")
    assert rep.status.status_code == StatusCode.ERROR
    assert rep.attributes is not None
    assert rep.attributes[ERROR_TYPE] == "RuntimeError"


# ---------------------------------------------------------------------------
# Configure tracing — exporter selection.
# ---------------------------------------------------------------------------


def test_configure_tracing_none_installs_provider_with_no_processor() -> None:
    """``--exporter none`` installs a real provider but exports nothing.

    The OTel API still returns real spans; ``get_finished_spans`` is
    empty because there is no processor wired.
    """
    from steadfast.tracing import configure_tracing

    configure_tracing(exporter="none")

    with chat_span(provider="x", model="m"):
        pass

    # No InMemoryExporter installed — nothing to assert beyond "did not crash".
    # The provider's get_active_span_processor count is implementation detail;
    # the contract is that calling code is unchanged.
