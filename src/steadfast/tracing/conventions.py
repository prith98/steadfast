"""OpenTelemetry GenAI semantic-convention attribute names and version pin.

Single source of truth for every ``gen_ai.*`` and ``steadfast.*`` attribute
key emitted by the harness. No string literals for attribute keys outside
this module — a future spec bump (1.42.0, 2.x) is then a single-file edit
and protects against typos that would silently break downstream ingest.

Spec: https://opentelemetry.io/docs/specs/semconv/gen-ai/

Notable points:

* The current spec uses ``gen_ai.provider.name`` for the provider identifier,
  but earlier drafts (and ``docs/WEEK_1.md``'s required-attribute list, plus
  some downstream ingest paths) still key on ``gen_ai.system``. ADR-0003 §A.2
  decides we emit *both* until the ecosystem completes the migration. Both
  keys live here; helpers populate them in lockstep.
* Span name conventions follow ``{operation_name} {request.model}``
  (per ``docs/specs/semconv/gen-ai/gen-ai-spans/``). Steadfast-owned
  spans (``benchmark``, ``task``, ``rep``, ``score``) are *not* part of
  the GenAI spec and use ``steadfast.*`` attributes.
* The agentic-systems extension (semconv issue #2664) is still draft and
  the constants below cover only the parts we emit today; tool spans
  land in week 2 alongside the framework adapters.

Verify ``GENAI_CONVENTIONS_VERSION`` against the latest stable semconv
release before any methodology-version event; any change is tracked in
``CHANGELOG.md``.
"""

from __future__ import annotations

from typing import Final

GENAI_CONVENTIONS_VERSION: Final[str] = "1.41.0"

# ---------------------------------------------------------------------------
# gen_ai.* attribute keys (from the current GenAI semconv spec).
# ---------------------------------------------------------------------------

# Operation identity. Required on every gen_ai span.
GEN_AI_OPERATION_NAME: Final[str] = "gen_ai.operation.name"

# Provider identity — current spec.
GEN_AI_PROVIDER_NAME: Final[str] = "gen_ai.provider.name"
# Provider identity — legacy. Emitted alongside GEN_AI_PROVIDER_NAME for
# back-compat with older Phoenix/Langfuse mappings (ADR-0003 §A.2).
GEN_AI_SYSTEM: Final[str] = "gen_ai.system"

# Request shape.
GEN_AI_REQUEST_MODEL: Final[str] = "gen_ai.request.model"
GEN_AI_REQUEST_MAX_TOKENS: Final[str] = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_TEMPERATURE: Final[str] = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P: Final[str] = "gen_ai.request.top_p"

# Response shape.
GEN_AI_RESPONSE_MODEL: Final[str] = "gen_ai.response.model"
GEN_AI_RESPONSE_ID: Final[str] = "gen_ai.response.id"
GEN_AI_RESPONSE_FINISH_REASONS: Final[str] = "gen_ai.response.finish_reasons"

# Token usage.
GEN_AI_USAGE_INPUT_TOKENS: Final[str] = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS: Final[str] = "gen_ai.usage.output_tokens"

# Agent / framework attributes (used when adapters land in week 2).
GEN_AI_AGENT_NAME: Final[str] = "gen_ai.agent.name"
GEN_AI_AGENT_ID: Final[str] = "gen_ai.agent.id"
GEN_AI_TOOL_NAME: Final[str] = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID: Final[str] = "gen_ai.tool.call.id"

# Standard error attribute (semconv core, not gen_ai).
ERROR_TYPE: Final[str] = "error.type"

# ---------------------------------------------------------------------------
# Operation-name values used in {op_name} {model} span names.
# ---------------------------------------------------------------------------

OP_CHAT: Final[str] = "chat"
OP_EMBEDDINGS: Final[str] = "embeddings"
OP_EXECUTE_TOOL: Final[str] = "execute_tool"
OP_INVOKE_AGENT: Final[str] = "invoke_agent"

# ---------------------------------------------------------------------------
# steadfast.* extensions — Steadfast-specific attributes on Steadfast-owned
# spans. Distinguished from gen_ai.* by namespace; these are NOT part of the
# OTel spec and downstream consumers should treat them as opaque.
# ---------------------------------------------------------------------------

STEADFAST_RUN_ID: Final[str] = "steadfast.run_id"
STEADFAST_TASK_ID: Final[str] = "steadfast.task.id"
STEADFAST_TASK_DOMAIN: Final[str] = "steadfast.task.domain"
STEADFAST_REP_IDX: Final[str] = "steadfast.rep.idx"
STEADFAST_REPS_TOTAL: Final[str] = "steadfast.reps.total"
STEADFAST_BENCHMARK_NAME: Final[str] = "steadfast.benchmark.name"
STEADFAST_PACKAGE_VERSION: Final[str] = "steadfast.package_version"
STEADFAST_GENAI_CONVENTIONS_VERSION: Final[str] = "steadfast.genai_conventions_version"

# Cost is best-effort; lives on the chat span when the provider returns usage
# we can price (per models/pricing.py).
STEADFAST_COST_USD: Final[str] = "steadfast.cost_usd"

# Calibration span attribute — average per-token logprob, populated by
# the calibration metric layer when the provider SDK exposes logprobs.
# Reserved here so the spec is stable when consumers attach (ADR-0003 §A.4).
STEADFAST_LOGPROB_AVG: Final[str] = "steadfast.logprob_avg"

# Judging extensions.
STEADFAST_JUDGE_KIND: Final[str] = "steadfast.judge.kind"
STEADFAST_JUDGE_MODEL: Final[str] = "steadfast.judge.model"
STEADFAST_VERDICT_SCORE: Final[str] = "steadfast.verdict.score"
STEADFAST_VERDICT_PASSED: Final[str] = "steadfast.verdict.passed"

# ---------------------------------------------------------------------------
# Steadfast span names (not part of the GenAI spec; chosen for readability
# in Phoenix / Langfuse / Datadog trace trees).
# ---------------------------------------------------------------------------

SPAN_BENCHMARK: Final[str] = "benchmark"
SPAN_TASK_PREFIX: Final[str] = "task"
SPAN_REP_PREFIX: Final[str] = "rep"
SPAN_SCORE_PREFIX: Final[str] = "score"
