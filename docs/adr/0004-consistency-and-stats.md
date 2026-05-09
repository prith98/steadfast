# ADR-0004: Consistency dimension + statistical primitives (Thursday)

- **Status:** Accepted
- **Date:** 2026-05-09
- **Methodology version:** v0.1
- **Supersedes:** none
- **Superseded by:** none

## Context

Thursday of week 1 (per `docs/WEEK_1.md`) ships the first reliability
**dimension** — Consistency — and the statistical primitives every
subsequent dimension will use. Two systems land:

1. **Consistency metrics** (`src/steadfast/metrics/consistency.py`):
   output, trajectory, and format consistency per
   `docs/METHODOLOGY.md` §1.
2. **Stats primitives** (`src/steadfast/stats/{bootstrap,wilson}.py`):
   BCa bootstrap CI and Wilson interval — the only sanctioned entry
   points for confidence intervals across the codebase.

Thursday also adds a paraphrase generator
(`src/steadfast/perturbations/paraphrase.py`), an embedding API surface
on the OpenAI client, and a `Task.output_schema` field so format
consistency knows what to validate against. ADRs 0001-0003 stand; this
ADR adds, doesn't supersede.

## Decisions

### A. Resolve open methodology Q1 — K=5 paraphrases vs N=10 reps

`docs/METHODOLOGY.md` §"Multi-run by default" specifies N=10, but §1.1
specifies K=5 paraphrases for output consistency. The auto-memory's
open methodology Q1 flagged this apparent inconsistency.

**Decision:** K=5 is correct as-is. Paraphrases are *different* inputs
each run once, not the same input run N times — the N=10 commitment is
about distributional measurement of a fixed input. Add a single
clarifying sentence to METHODOLOGY §1.1 documenting this; computation
unchanged so this is a "typo and clarification fix" per
§"Versioning", not a methodology version event. Auto-memory's Q1 is
closed by this ADR.

### B. Output-consistency rubric — 0-4 Likert with normalization

Methodology §1.1 specifies a 0-4 Likert scale for the rubric judge.
Wednesday's `RubricJudge` (ADR-0003 §B) returns a Verdict with `score`
in [0, 1].

**Decision:** A new pair-rubric LLM call (not a `Judge` subclass — it
takes two answers, not one) lives in `metrics/consistency.py`. The
prompt asks for a 0-4 integer; the metric divides by 4 to normalize to
[0, 1] before reporting. Frozen prompt at
`prompts/consistency_rubric_v1.txt`.

The Likert anchors are not a presentation gimmick — they let a human
auditor disagree with the LLM judge in human-interpretable terms (vs
"score 0.43"). Methodology preserves the Likert structure; the
normalization is purely for downstream aggregation.

### C. Embedding caching — defer to v0.2

A full leaderboard run hits ~1,250 embedding calls (50 tasks × 5
models × K=5 paraphrases). At
~500 tokens × $0.13 / M tokens for `text-embedding-3-large` that is
**$0.08 total**. Caching adds storage surface, key-collision risk, and
a re-derivation path for stale entries.

**Decision:** No cache for v0.1. Embedding calls flow through
`OpenAIClient.aembed` and emit their own `embeddings {model}` span via
the tracing module. Revisit if a real run shows the cost has grown by
an order of magnitude (unlikely at 50-task scale).

### D. `Task.output_schema: str | None = None`

Format consistency (§1.3) measures the rate at which an agent produces
schema-valid output. The metric needs to know the expected schema.

**Decision:** Add `Task.output_schema: str | None = None` — a JSON
Schema string (validated lazily by `jsonschema` when the metric runs).
Storing the schema as a string rather than a `dict[str, Any]` keeps
the public contract Pydantic-typed (no `Any` in the public surface
per Q5 from project kickoff). Tasks without a schema return `N/A`
from format consistency. Additive backwards-compatible change to
ADR-0002 §A — CHANGELOG entry, no ADR-0002 amendment.

### E. Three standalone measurement functions

Three pure-ish functions in `metrics/consistency.py`, each independent
of the runner and each returning a typed result Pydantic model:

```python
async def measure_output_consistency(
    *, task, agent, infra_client, k=5, seed=0,
) -> OutputConsistencyResult: ...

def measure_trajectory_consistency(reps: list[RepRecord]) -> TrajectoryConsistencyResult: ...

def measure_format_consistency(reps: list[RepRecord], schema: str) -> FormatConsistencyResult: ...
```

`measure_output_consistency` does **not** go through `run_task` — K=5
*different* paraphrased Tasks run once each is conceptually orthogonal
to the runner's N=10 same-input reps. Trajectory and format consume
reps the runner already produced. CLI orchestration that runs both
N=10 *and* K=5 for one task is a Friday concern; Thursday delivers the
building blocks.

### F. Dependencies — `agentevals` and `jsonschema`

Methodology §1.2 explicitly mandates `agentevals` trajectory matchers
in superset mode; no in-tree alternative satisfies the methodology.
Format consistency needs schema validation; `jsonschema>=4` is the
standard. Both deps are pre-justified by the methodology.

**`Levenshtein` is *not* added.** Wagner-Fischer edit distance is a
~15-line textbook DP algorithm; a dep for one routine is dependency
creep. The hand-rolled implementation lives in `metrics/consistency.py`
with a citation and is cross-checked in tests against
`difflib.SequenceMatcher`-derived expected values.

### G. Trajectory consistency on empty trajectories

Per ADR-0002 §A.2 (which partially resolved auto-memory Q2),
`AgentResponse.trajectory` is optional and trajectory consistency
returns `N/A` on empty trajectories.

**Decision:** Implementation returns
`TrajectoryConsistencyResult(value=None, reason="trajectory not exposed
by agent")` when *all* completed reps have empty trajectories. Mixed
case (some reps have trajectories, some don't) treats empty
trajectories as length-0 sequences and Levenshtein-compares them to
the populated ones — they will score low, which is the right
calibration signal.

### H. Bootstrap edge cases

`scipy.stats.bootstrap` is the underlying engine (no hand-rolled
bootstrap per CLAUDE.md "Tech stack"). Three edge cases need explicit
handling:

* Empty data → raise `ValueError("bootstrap requires non-empty data")`.
* `N < 2` → raise `ValueError("bootstrap requires N >= 2 samples")`.
* Zero-variance data (all identical) → scipy emits a `DegenerateDataWarning`
  and may return NaN for the CI. The wrapper detects this and returns
  `(point_estimate, point_estimate)` with `degenerate=True` flagged on
  the result.

These behaviors are exercised by `tests/test_bootstrap.py` against
`scipy.stats.bootstrap` directly.

### I. `aembed` lives on `OpenAIClient`, not `BaseModelClient`

Per ADR-0001 the embedding model (`text-embedding-3-large`) is
OpenAI-only for v0.1. Adding an abstract `aembed` to `BaseModelClient`
would force the Anthropic and Google clients to implement a method
they will never use.

**Decision:** `aembed(texts, *, model)` is a concrete method on
`OpenAIClient`. Future provider embedding support adds the method
where appropriate; v0.1 doesn't need an abstraction with one
implementation.

The pricing table gains a `text-embedding-3-large` entry with
`output_per_mtok=Decimal("0")` (embeddings have no output tokens).
`compute_cost` works unchanged — `output_tokens=0 × 0 = 0`.

## Consequences

**Positive**

- Open methodology Q1 closed.
- All four bootstrap-needing measurements (output consistency mean,
  format pass-rate, trajectory similarity, future Brier/ECE) now share
  one entry point.
- ADR-0001's infrastructure-model lock now has a concrete consumer:
  `OpenAIClient.aembed` for embeddings; the rubric-judge path was the
  only consumer before today.

**Negative**

- Two new runtime deps (`agentevals`, `jsonschema`). Mitigation: both
  are well-maintained and pre-justified; `Levenshtein` was rejected
  for dependency hygiene.
- `Task.output_schema` adds a new public field; downstream code that
  introspects the Task model needs to handle the new attribute. The
  default `None` makes existing tasks pass through unchanged.
- The output-consistency rubric is a third LLM-as-judge prompt to
  maintain (alongside the rubric judge and paraphrase validator); each
  is its own quality risk that needs ensemble-judge follow-up in v0.2.

## Path to v0.2

- **Embedding cache** in `src/steadfast/storage/` keyed by
  `sha256(text) + model_id` — when a real run shows it's needed.
- **Ensemble judge** (per ADR-0001 §"Path to v0.2") for the
  output-consistency rubric — same generalization as the outcome
  rubric judge.
- **Tool-args normalization for trajectory consistency** beyond
  `agentevals` superset mode: a future ADR can promote richer
  argument-equivalence rules (numeric tolerance, semantic equivalence
  of strings, etc.).
- **Long-context handling** in `compute_cost` — embedding text may be
  long; current pricing assumption is uniform.

## References

- `docs/METHODOLOGY.md` §1 (output / trajectory / format consistency),
  §"Statistical conventions", §"Versioning".
- `docs/WEEK_1.md` §"Thursday".
- ADR-0001 — infrastructure-model lock (gpt-5.2 + text-embedding-3-large).
- ADR-0002 — Tuesday core abstractions; §A.2 (trajectory optionality),
  §B.1 (BaseModelClient).
- ADR-0003 — Wednesday tracing + judges; §B.4 (judge retry policy).
- `agentevals` — https://github.com/langchain-ai/agentevals
- Wagner & Fischer (1974), "The string-to-string correction problem",
  *J. ACM* 21(1), 168-173.
- Wilson (1927), "Probable inference, the law of succession, and
  statistical inference", *JASA* 22(158), 209-212.
- Brown, Cai & DasGupta (2001), "Interval estimation for a binomial
  proportion", *Statistical Science* 16(2), 101-133.
- Efron & Tibshirani (1993), *An Introduction to the Bootstrap.*
