# ADR-0002: v0.1 core abstractions — Agent, ModelClient, Runner

- **Status:** Accepted
- **Date:** 2026-05-08
- **Methodology version:** v0.1
- **Supersedes:** none
- **Superseded by:** none

## Context

Tuesday of week 1 (per `docs/WEEK_1.md`) introduces three contract-level
abstractions used by every subsequent module:

1. The **Agent ABC** (and its companion Pydantic types `Task`, `AgentResponse`,
   `ToolCall`, `GroundTruth`) — the public surface that user-supplied agents
   conform to.
2. **`BaseModelClient`** and the per-provider implementations (Anthropic,
   OpenAI, Google) — the async LLM interface used by the harness and by
   built-in agents.
3. **The runner** (`run_task`, `RepStatus`, `RepRecord`, `RunResult`) — the
   N-repetition executor with deterministic resumption.

Each carries decisions that affect every downstream module. We commit them
here so future work can reference the rationale (and so a v0.2 successor knows
exactly what to supersede).

## Decisions

### A. Agent contract

**A.1 — Confidence elicitation contract (carry-over from project kickoff Q1).**
The primary contract requires the user-supplied Agent to populate
`AgentResponse.confidence` itself. The harness sets
`Task.confidence_suffix` to a frozen elicitation prompt; the agent must
concatenate it into its prompt and parse the resulting confidence into a
float in `[0, 1]`.

A clearly-labeled "post-hoc confidence" variant is supported for black-box
agents that can't be modified — it makes a second LLM call after `arun`
returns and is reported under a separate column on the leaderboard, never
silently mixed with primary-contract results.

*Why:* joint generation of answer + confidence preserves calibration
semantics (`docs/METHODOLOGY.md` §3.1). Post-hoc elicitation is
methodologically weaker; transparency about which mode produced each
calibration number is essential.

**A.2 — Optional confidence and trajectory.**
`confidence: float | None` and `trajectory: list[ToolCall] = []` are both
nullable / empty-by-default. Calibration metrics skip None-confidence reps
with a logged warning; trajectory consistency returns N/A on empty
trajectories. This unblocks toolless agents and agents that don't expose
confidence, with the cost reflected in their leaderboard scores.

**A.3 — Metadata typing (carry-over from project kickoff Q5).**
`Task.metadata`, `ToolCall.args`, and `AgentResponse.metadata` are
`dict[str, str | int | float | bool]` — the scalar metadata union. Richer
shapes belong on subclasses, not the public dict bag.

*Why:* `CLAUDE.md` forbids `dict[str, Any]` in public APIs. The scalar
union is the pragmatic middle ground.

### B. Model client surface

**B.1 — `achat` is the abstract contract.** `acomplete(prompt)` is sugar
that wraps a single user message. Subclasses implement
`_achat_provider`; the base class owns the per-instance asyncio semaphore
and the tenacity retry layer.

**B.2 — Subclasses declare `_is_retryable(exc)`.** The base default is
"never retry". Each provider client returns `True` for its own rate-limit
and 5xx status errors. This keeps the retry policy provider-aware while
the orchestration lives in one place.

**B.3 — `raw: dict[str, Any]` carve-out on `ChatResponse`.**
`ChatResponse.raw` carries the provider-specific full response payload.
This is the single place in the public surface where we use
`dict[str, Any]`. It exists for two reasons: it lands on OTel spans for
debug, and it would be lossy to serialize as a string.

*Why this is acceptable:* `raw` is documented as opaque, callers must not
depend on its structure, and it is excluded from the leaderboard
manifest. Q5's prohibition on `Any` in public APIs is preserved in
spirit by keeping `raw` strictly informational.

### C. Pricing and cost

**C.1 — Decimal everywhere monetary.** `cost_usd`, `input_per_mtok`,
and `output_per_mtok` are all `Decimal`. Float drift across thousands of
calls per leaderboard run would compound.

**C.2 — `dated_at` on every `ModelPricing` entry.** Lands in the run
manifest. Reproductions verify the pricing assumption against this date.

**C.3 — `compute_cost` raises `KeyError` on unknown models.** Silent
zero-cost fallback would mask a real bug in the leaderboard's
reproducibility claims.

### D. Runner

**D.1 — Failed-rep retry policy: don't auto-retry.**
When a rep fails (parse error, malformed model output, persistent 5xx),
mark it `FAILED` and continue. A future `--retry-failed` flag handles
deliberate re-runs.

*Why:* failures are signal. Auto-retry hides them and biases the rep
distribution toward whatever happened to succeed on retry, which is
exactly the kind of selection effect that breaks reliability
measurement.

**D.2 — Deterministic `run_id`.** `run_id = sha256(canonical_inputs)[:16]`
where `canonical_inputs` is a sorted JSON of `{task_id,
task_content_sha256, agent_class, model, reps, package_version}`.
Identical configurations produce identical IDs, so a re-invocation of
`steadfast bench` automatically resumes any prior incomplete run.
Editing a task's content invalidates the `run_id`, forcing a fresh
execution.

**D.3 — Storage is dumb persistence.** `CheckpointStore` takes/returns
primitives and JSON strings. `RepRecord` ↔ `RepRow` conversion lives in
the runner, not in storage. This avoids circular imports and keeps the
SQLite schema decoupled from the Pydantic surface.

### E. Real model IDs (Tuesday Q3)

`docs/WEEK_1.md` originally specified `--model claude-opus-4.5` for
Tuesday's DOD invocation. The current Anthropic API doesn't expose that
ID; the closest real model is `claude-opus-4-7`. We chose to use real
model IDs throughout (code, spec, manifest) rather than maintain an
alt-history alias map. Methodology stays portable to whatever frontier
models are current at launch; documentation is updated to match.

### F. ADR structure (Tuesday Q4)

Three substantive contracts (Agent, ModelClient, Runner) collapse into
this single ADR rather than three. Each contract gets a top-level section.
If any one section evolves substantially in v0.2, that section gets
promoted to its own ADR, leaving 0002 as the historical reference.

## Consequences

**Positive**

- One place to read every Tuesday-level decision.
- Tests can assert against documented behavior (e.g., the
  `test_run_task_does_not_retry_failed` test enforces D.1).
- Future ADRs can supersede individual sections by reference.

**Negative**

- A single document grows with every contract revision; section
  numbering and consequences may drift. Mitigation: when a section's
  successor ADR is written, mark the section "Superseded by ADR-N" in
  this document rather than deleting it.

## Path to v0.2

- **Confidence post-hoc variant**: implement Friday alongside the
  calibration metrics. Will become a separate "post-hoc confidence"
  ADR if the leaderboard presentation needs material discussion.
- **Trajectory contract for non-tool agents**: open methodology question
  recorded in auto-memory. Resolve before the LangGraph adapter (week
  2) and either amend section A.2 or create ADR-0003.
- **`--retry-failed` flag**: add when a real run encounters transient
  failures we want to surgically re-run. Not blocking v0.1.
- **Ensemble judging** (per ADR-0001) and **OTel tracing** are the next
  two large architectural decisions; they will be ADR-0003 and ADR-0004
  in some order.

## References

- `docs/SPEC.md` — what we're building.
- `docs/METHODOLOGY.md` §1 (consistency), §3 (calibration).
- `docs/WEEK_1.md` §"Tuesday".
- ADR-0001 — infrastructure-model lock.
- `notes/monday_writeup.md` (gitignored, local) — captures the original
  Q1-Q5 framing carried into Tuesday.
