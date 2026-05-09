"""Span helpers for the Steadfast hierarchy.

Per ADR-0003 §A.1, the span tree is::

    benchmark                              (CLI root)
    └── task {task.id}                     (runner)
        ├── rep {idx}                      (runner)
        │   ├── chat {model}               (model client)
        │   └── execute_tool {tool}        (week 2)
        └── ...
    └── score {judge_kind}                 (CLI post-run; sibling of task)
        └── chat {judge_model}             (rubric judges only)

This module exposes context-manager helpers that wrap the OTel API and
populate Steadfast-canonical attributes from
:mod:`steadfast.tracing.conventions`. Span attributes are typed via the
``conventions`` constants — there are no string literals for attribute
keys here.

Required attributes for the ``chat`` span follow the GenAI semconv
v1.41.0 (``gen_ai.operation.name``, ``gen_ai.provider.name``,
``gen_ai.request.model``, etc.) plus a back-compat ``gen_ai.system``
emission per ADR-0003 §A.2.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from decimal import Decimal
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from steadfast.tracing.conventions import (
    ERROR_TYPE,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROVIDER_NAME,
    GEN_AI_REQUEST_MAX_TOKENS,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_REQUEST_TEMPERATURE,
    GEN_AI_REQUEST_TOP_P,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_RESPONSE_ID,
    GEN_AI_RESPONSE_MODEL,
    GEN_AI_SYSTEM,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
    OP_CHAT,
    OP_EMBEDDINGS,
    SPAN_BENCHMARK,
    SPAN_REP_PREFIX,
    SPAN_SCORE_PREFIX,
    SPAN_TASK_PREFIX,
    STEADFAST_BENCHMARK_NAME,
    STEADFAST_COST_USD,
    STEADFAST_JUDGE_KIND,
    STEADFAST_JUDGE_MODEL,
    STEADFAST_LOGPROB_AVG,
    STEADFAST_PACKAGE_VERSION,
    STEADFAST_REP_IDX,
    STEADFAST_REPS_TOTAL,
    STEADFAST_RUN_ID,
    STEADFAST_TASK_DOMAIN,
    STEADFAST_TASK_ID,
    STEADFAST_VERDICT_PASSED,
    STEADFAST_VERDICT_SCORE,
)

# Tracer name follows the OTel-Python convention of using the importing
# package's name. Phoenix uses this as the "Instrumentation" facet.
_TRACER_NAME = "steadfast"


def _tracer() -> trace.Tracer:
    """Resolve the tracer at call-time so test fixtures that swap providers
    are picked up. (``trace.get_tracer`` proxies to the current provider.)
    """
    return trace.get_tracer(_TRACER_NAME)


def _record_exception(span: Span, exc: BaseException) -> None:
    """Mark ``span`` as ERROR and record ``exc`` per OTel conventions."""
    span.set_attribute(ERROR_TYPE, type(exc).__qualname__)
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))


@contextmanager
def _start_span(
    *,
    name: str,
    kind: SpanKind,
    attributes: Mapping[str, Any],
) -> Iterator[Span]:
    """Open a span, yield it, and on exception mark ERROR before re-raising.

    Centralizes the try/except boilerplate every public span helper would
    otherwise duplicate.
    """
    with _tracer().start_as_current_span(name, kind=kind, attributes=dict(attributes)) as span:
        try:
            yield span
        except BaseException as exc:
            _record_exception(span, exc)
            raise


@contextmanager
def benchmark_span(
    *,
    name: str,
    package_version: str,
    extra_attributes: Mapping[str, Any] | None = None,
) -> Iterator[Span]:
    """Root span for one ``steadfast bench`` invocation.

    ``name`` is a human label (e.g., ``"customer_support_pilot"`` or just
    the task ID for single-task runs). Lands as ``steadfast.benchmark.name``.
    """
    attributes: dict[str, Any] = {STEADFAST_BENCHMARK_NAME: name}
    if extra_attributes:
        attributes.update(extra_attributes)
    with _start_span(name=SPAN_BENCHMARK, kind=SpanKind.INTERNAL, attributes=attributes) as span:
        # package_version also lives on the resource, but copying it onto
        # the root span keeps single-span exports (collectors that strip
        # resource attrs) usable.
        span.set_attribute(STEADFAST_PACKAGE_VERSION, package_version)
        yield span


@contextmanager
def task_span(
    *,
    task_id: str,
    domain: str,
    run_id: str,
    reps_total: int,
) -> Iterator[Span]:
    """Per-task span — child of the benchmark span when one is active."""
    attributes: dict[str, Any] = {
        STEADFAST_TASK_ID: task_id,
        STEADFAST_TASK_DOMAIN: domain,
        STEADFAST_RUN_ID: run_id,
        STEADFAST_REPS_TOTAL: reps_total,
    }
    with _start_span(
        name=f"{SPAN_TASK_PREFIX} {task_id}",
        kind=SpanKind.INTERNAL,
        attributes=attributes,
    ) as span:
        yield span


@contextmanager
def rep_span(
    *,
    rep_idx: int,
    run_id: str,
    task_id: str,
) -> Iterator[Span]:
    """Per-rep span — child of the task span. Records FAILED status on raise."""
    attributes: dict[str, Any] = {
        STEADFAST_REP_IDX: rep_idx,
        STEADFAST_RUN_ID: run_id,
        STEADFAST_TASK_ID: task_id,
    }
    with _start_span(
        name=f"{SPAN_REP_PREFIX} {rep_idx}",
        kind=SpanKind.INTERNAL,
        attributes=attributes,
    ) as span:
        yield span


@contextmanager
def chat_span(
    *,
    provider: str,
    model: str,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> Iterator[Span]:
    """LLM-call span — emitted from :class:`BaseModelClient.achat`.

    Span name follows the GenAI v1.41.0 convention ``{op_name} {model}`` so
    Phoenix renders it as e.g. ``chat claude-opus-4-7``. Sets request
    attributes up-front; :func:`record_chat_response` populates response
    attributes after the provider call returns.
    """
    attributes: dict[str, Any] = {
        GEN_AI_OPERATION_NAME: OP_CHAT,
        GEN_AI_PROVIDER_NAME: provider,
        # Legacy attribute — kept in lockstep with provider.name per ADR-0003 §A.2.
        GEN_AI_SYSTEM: provider,
        GEN_AI_REQUEST_MODEL: model,
    }
    if max_tokens is not None:
        attributes[GEN_AI_REQUEST_MAX_TOKENS] = max_tokens
    if temperature is not None:
        attributes[GEN_AI_REQUEST_TEMPERATURE] = temperature
    if top_p is not None:
        attributes[GEN_AI_REQUEST_TOP_P] = top_p

    with _start_span(
        name=f"{OP_CHAT} {model}",
        kind=SpanKind.CLIENT,
        attributes=attributes,
    ) as span:
        yield span


def record_chat_response(
    span: Span,
    *,
    response_model: str,
    input_tokens: int,
    output_tokens: int,
    finish_reason: str | None,
    response_id: str | None = None,
    cost_usd: Decimal | None = None,
    avg_logprob: float | None = None,
) -> None:
    """Populate response-side attributes on a ``chat`` span after the call returns.

    Cost is recorded under the Steadfast-namespaced
    :data:`STEADFAST_COST_USD` (not part of the OTel GenAI spec). Decimal
    is stringified to preserve precision — OTel attribute values are
    primitives only, and ``Decimal`` doesn't survive the SDK's int/float
    coercion.

    ``avg_logprob`` populates :data:`STEADFAST_LOGPROB_AVG` (reserved
    Wednesday in ADR-0003 §A.4 and populated Friday per ADR-0005 §A) when
    the provider exposed per-token logprobs. ``None`` callers leave the
    attribute absent — downstream consumers treat absence as N/A.
    """
    span.set_attribute(GEN_AI_RESPONSE_MODEL, response_model)
    span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    span.set_attribute(GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
    if finish_reason is not None:
        # finish_reasons is a list[string] in the spec.
        span.set_attribute(GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason])
    if response_id is not None:
        span.set_attribute(GEN_AI_RESPONSE_ID, response_id)
    if cost_usd is not None:
        span.set_attribute(STEADFAST_COST_USD, str(cost_usd))
    if avg_logprob is not None:
        span.set_attribute(STEADFAST_LOGPROB_AVG, avg_logprob)


@contextmanager
def embeddings_span(
    *,
    provider: str,
    model: str,
) -> Iterator[Span]:
    """Embedding-call span — analog of :func:`chat_span` for embedding models.

    Span name follows the GenAI convention ``{op_name} {model}`` so Phoenix
    renders it as e.g. ``embeddings text-embedding-3-large``.
    Embeddings have no output tokens or finish reasons, so the response
    helper :func:`record_embeddings_response` only records input tokens
    and (optionally) cost.
    """
    attributes: dict[str, Any] = {
        GEN_AI_OPERATION_NAME: OP_EMBEDDINGS,
        GEN_AI_PROVIDER_NAME: provider,
        GEN_AI_SYSTEM: provider,
        GEN_AI_REQUEST_MODEL: model,
    }
    with _start_span(
        name=f"{OP_EMBEDDINGS} {model}",
        kind=SpanKind.CLIENT,
        attributes=attributes,
    ) as span:
        yield span


def record_embeddings_response(
    span: Span,
    *,
    response_model: str,
    input_tokens: int,
    cost_usd: Decimal | None = None,
) -> None:
    """Populate response-side attributes on an :func:`embeddings_span`.

    Embeddings have no ``output_tokens`` or ``finish_reasons`` per the
    OpenAI / Cohere / Voyage embedding conventions; the chat-only
    attributes are intentionally omitted.
    """
    span.set_attribute(GEN_AI_RESPONSE_MODEL, response_model)
    span.set_attribute(GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
    if cost_usd is not None:
        span.set_attribute(STEADFAST_COST_USD, str(cost_usd))


def record_retry_event(span: Span, *, attempt: int, exc: BaseException) -> None:
    """Record a retry attempt as a span event (ADR-0003 §A.3).

    Per the ADR, we keep one span per public ``achat()`` call; retries are
    events on that span rather than separate spans. ``attempt`` is the
    attempt number that just failed (1-indexed).
    """
    span.add_event(
        "retry",
        attributes={
            "steadfast.retry.attempt": attempt,
            ERROR_TYPE: type(exc).__qualname__,
            "exception.message": str(exc),
        },
    )


@contextmanager
def score_span(
    *,
    judge_kind: str,
    task_id: str,
    rep_idx: int,
    run_id: str,
    judge_model: str | None = None,
) -> Iterator[Span]:
    """Per-(rep, judge) span emitted by the post-run scoring phase.

    Sibling of the ``task`` span under ``benchmark`` (ADR-0003 §A.7).
    ``judge_model`` is the model identifier when the judge is a
    :class:`steadfast.judges.rubric.RubricJudge`; ``None`` for
    deterministic judges like :class:`ExactMatchJudge`.
    """
    attributes: dict[str, Any] = {
        STEADFAST_JUDGE_KIND: judge_kind,
        STEADFAST_TASK_ID: task_id,
        STEADFAST_REP_IDX: rep_idx,
        STEADFAST_RUN_ID: run_id,
    }
    if judge_model is not None:
        attributes[STEADFAST_JUDGE_MODEL] = judge_model

    with _start_span(
        name=f"{SPAN_SCORE_PREFIX} {judge_kind}",
        kind=SpanKind.INTERNAL,
        attributes=attributes,
    ) as span:
        yield span


def record_verdict(span: Span, *, score: float, passed: bool) -> None:
    """Record verdict outcome on a score span."""
    span.set_attribute(STEADFAST_VERDICT_SCORE, score)
    span.set_attribute(STEADFAST_VERDICT_PASSED, passed)
