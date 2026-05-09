# ADR-0003: OTel GenAI tracing model + outcome judges (Wednesday)

- **Status:** Accepted
- **Date:** 2026-05-08
- **Methodology version:** v0.1
- **Supersedes:** none
- **Superseded by:** none

## Context

Wednesday of week 1 (`docs/WEEK_1.md` §"Wednesday") introduces two
contract-level systems used by every subsequent module:

1. **OTel GenAI tracing** — span hierarchy, attribute conventions, and
   exporter wiring. Required for Phoenix / Langfuse / Datadog ingest and
   for the leaderboard's reproducibility manifest.
2. **Outcome judges** — `ExactMatchJudge` and `RubricJudge`, returning a
   `Verdict {score, passed, reason}`. The first scoring layer; every
   downstream metric (consistency, calibration, robustness, safety)
   consumes this surface.

Per ADR-0002 §F we bundle both into one ADR for now; either section can
be promoted to its own ADR if it evolves materially in v0.2.

## A. Tracing

### A.1 — Span hierarchy

```
benchmark                                (CLI root, INTERNAL kind)
└── task {task.id}                       (runner, INTERNAL)
│   ├── rep 0                            (runner, INTERNAL)
│   │   ├── chat {model}                 (model client, CLIENT)
│   │   └── execute_tool {tool}          (future; tool spans, INTERNAL)
│   ├── rep 1, ...
└── score {judge_kind}                   (CLI post-run, INTERNAL)
    ├── (per-rep judging)
    └── chat {judge_model}               (rubric judge only, CLIENT)
```

Names follow the [GenAI semantic conventions
v1.41.0](https://opentelemetry.io/docs/specs/semconv/gen-ai/) span-name
pattern `{operation_name} {request.model}` for inference spans.
Steadfast-owned spans (`benchmark`, `task`, `rep`, `score`) are
namespaced via `steadfast.*` attributes.

### A.2 — `gen_ai.system` AND `gen_ai.provider.name`

The conventions evolved between drafts: `gen_ai.system` (the name
required by `docs/WEEK_1.md`) was renamed to `gen_ai.provider.name` in
the recent semconv revision. Some ingest paths (older Phoenix versions,
existing Langfuse mappings) still key on the legacy name.

**Decision:** emit *both* on every LLM-call span. The cost is one extra
attribute per span; the benefit is forward compatibility with the spec
*and* backward compatibility with downstream consumers we don't control.
When the ecosystem completes the migration, this dual emission becomes
a single edit in `tracing/conventions.py`.

### A.3 — One span per `achat()`, not per retry attempt

`BaseModelClient.achat` wraps `_achat_provider` in a tenacity retry loop
(ADR-0002 §B.1). Two options for instrumentation:

1. **One span per public call**, retry attempts recorded as
   `span.add_event("retry", {...})`.
2. **One span per attempt**, parent span linking them.

We pick (1). The retry contract (ADR-0002 §B.2) stays untouched — tracing
is purely observability. Phoenix and Langfuse render retry events
inline; per-attempt spans would clutter the trace tree without adding
diagnostic value at v0.1 scale. Revisit if a real run shows retry-storm
patterns we need to inspect attempt-by-attempt.

### A.4 — Logprob attribute deferred to Friday

Auto-memory's open methodology question #3 (Anthropic logprob asymmetry)
intersects this Wednesday work — the LLM-call span is the natural place
to record per-token logprobs. We **defer** the surface to Friday for two
reasons:

1. The downstream consumer (calibration `Brier` / `ECE` metrics) lands
   Friday. Adding the span attribute Wednesday would create a
   write-without-readers situation that's hard to test.
2. The presentation question — "verbalized as headline, logprob as
   secondary, or both with explicit holes" — is a Friday calibration
   table decision, not a tracing decision. Tracing should record what's
   available; Friday decides how to display it.

Wednesday names a constant `STEADFAST_LOGPROB_AVG` in
`tracing/conventions.py` so Friday's PR is purely additive, but does not
populate it.

### A.5 — Phoenix default endpoint

When `--exporter otlp` is set and `OTEL_EXPORTER_OTLP_ENDPOINT` is not,
default to `http://localhost:6006/v1/traces` (Phoenix's HTTP collector
on its default port). Documented in the CLI help. Users running
Langfuse, Datadog, or a custom collector set the env var.

### A.6 — `tracing/conventions.py` is the single source of truth

All `gen_ai.*` and `steadfast.*` attribute names are declared there as
typed string constants. No string literals for attribute keys outside
that module. This makes a future spec bump (1.42.0, 2.x) a single-file
edit and protects against typos that would silently break downstream
ingest.

### A.7 — Score subtree is a sibling of `task`, not a child of `rep`

By the time judging runs, the rep spans are already closed (we call
`run_task` to completion, then iterate verdicts). Reattaching to a
closed span's context would require span linking, which Phoenix's
visualization handles inconsistently. Cleaner: judge spans live under
their own `score` subtree, child of `benchmark`, with attributes
linking back via `steadfast.task.id` and `steadfast.rep.idx`. This
matches the operational sequence and renders predictably in every
ingest path we tested.

## B. Judges

### B.1 — `Verdict` shape

```python
class Verdict(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reason: str
```

Frozen via `model_config = ConfigDict(frozen=True)`. The `score`/`passed`
split exists because some metrics want a continuous rubric score (output
consistency, refusal calibration nuance) and others want a binary signal
(catastrophic-failure rate, format consistency). Carrying both
unconditionally avoids a downstream metric having to redo the threshold
decision.

### B.2 — `Judge` ABC

```python
class Judge(ABC):
    @abstractmethod
    async def ajudge(self, task: Task, response: AgentResponse) -> Verdict: ...
```

Async because rubric judges hit an LLM. Sync wrapper not provided —
every caller (runner, CLI, future ensemble) is already in async context.
The ensemble path from ADR-0001's "Path to v0.2" implements this same
ABC over `[Judge, ...]`.

### B.3 — `ExactMatchJudge` canonicalization

Rules, applied in order:

1. NFKC unicode normalization (idempotent; collapses widths and
   compatibility forms).
2. Lowercase (`.casefold()` for unicode-correct lowercasing).
3. Collapse runs of whitespace to a single space.
4. Strip leading/trailing whitespace.
5. Strip trailing punctuation in `{".", ",", "!", "?", ";", ":"}`.

After canonicalization, **substring containment** is the match operator:
`canonical(ground_truth.value) in canonical(response.answer)`. Score is
1.0 on match, 0.0 otherwise; `passed = score == 1.0`. Reason carries
both canonical forms for debug.

The substring rule (vs strict equality) is deliberate. The pilot ground
truth is `"30 days"`; agents reasonably answer "The return window is 30
days for unopened items." Strict equality would reject every realistic
answer; the rubric judge handles harder semantic cases. Substring is the
right cheap check.

### B.4 — `RubricJudge`: JSON-only + 1 retry + raise on second failure

The judge prompt instructs JSON-only output matching the `Verdict`
schema. Output is parsed with `Verdict.model_validate_json`. On
`ValidationError`, retry once with a stricter "your previous output
failed validation; emit *only* the JSON object" instruction. On a second
`ValidationError`, raise `JudgeParseError`.

We deliberately do **not** silently produce a default "verdict failed"
result. Per ADR-0002 §D.1, failures are signal; a soft-failed verdict
would corrupt downstream metric distributions exactly the same way an
auto-retry of a failed rep would corrupt rep distributions. The CLI
catches `JudgeParseError` and reports it; the rep stays scored as `None`
in the result JSON, distinguishable from a `score=0.0` real failure.

### B.5 — Default rubric model = `gpt-5.2` (per ADR-0001)

Hardcoded default; configurable via constructor. The infrastructure-model
lock (ADR-0001) requires that `RubricJudge` defaults to `gpt-5.2` for
v0.1 leaderboard comparability. Local users running cheaper inner-loop
experiments may pass `model="gpt-5-mini"` or similar — that's
encouraged, but their numbers don't go on the leaderboard.

### B.6 — Frozen prompt at `prompts/rubric_v1.txt`

The prompt is content-addressed by filename suffix (`_v1`). Any change
to scoring semantics gets a new file (`_v2`) and an ADR per
`docs/METHODOLOGY.md` §"Versioning". Prompt content includes:

- Role: an impartial expert grader.
- Inputs: task input, ground truth (rubric criteria), agent answer.
- Output schema: explicit `{"score": float [0..1], "passed": bool, "reason": str}`.
- Calibration guidance: 0.0 = wrong, 0.5 = partially correct,
  1.0 = fully correct, `passed = score >= 0.6`.
- Bias guard: prompt does *not* reveal which model produced the answer.

## Consequences

**Positive**

- Every Wednesday-level decision in one place.
- Span attribute names are typed constants, mypy-checked.
- Verdict shape is the same surface every metric will consume.
- ADR-0001's v0.2 ensemble path is unblocked: plug into `Judge` ABC.

**Negative**

- Dual emission of `gen_ai.system` + `gen_ai.provider.name` doubles the
  attribute count on LLM spans (mitigated: the rest of each span is
  unchanged; the bytes are negligible).
- `RubricJudge` raising on parse failure means the CLI must catch and
  surface it explicitly — opaque crashes would be worse, but it's
  another error path to maintain.

## Update history

- **2026-05-09 (ADR-0004 §I)**: added an ``embeddings {model}`` CLIENT
  span (sibling-of-chat under whatever parent is active) for
  :meth:`OpenAIClient.aembed`. Span name follows the same
  ``{op_name} {model}`` convention; the helper is
  :func:`steadfast.tracing.embeddings_span`.

## Path to v0.2

- **Logprob span attribute** populated by Friday's calibration work; the
  constant is reserved Wednesday so the change is purely additive.
- **Tool execution spans** (`execute_tool {tool}`) when the LangGraph /
  OpenAI Agents SDK adapters land in week 2. Conventions are already
  set in `tracing/conventions.py`; the spans just need to be created.
- **Ensemble judging** per ADR-0001 §"Path to v0.2".
- **Per-attempt span on retry** if a real run shows we need it.

## References

- `docs/METHODOLOGY.md` §1.1, §"Known limitations and threats to
  validity".
- `docs/WEEK_1.md` §"Wednesday".
- ADR-0001 — infrastructure-model lock (default rubric judge model).
- ADR-0002 — Tuesday core abstractions (Agent, ModelClient, Runner).
- OTel GenAI conventions (v1.41.0):
  https://opentelemetry.io/docs/specs/semconv/gen-ai/
- Agent / framework spans:
  https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/
- Agentic-systems extension (draft):
  https://github.com/open-telemetry/semantic-conventions/issues/2664
- Pydantic Evals LLM-as-judge guide:
  https://pydantic.dev/articles/llm-as-a-judge
