# ADR-0005: Calibration dimension + confidence elicitation contract (Friday)

- **Status:** Accepted
- **Date:** 2026-05-09
- **Methodology version:** v0.1
- **Supersedes:** none
- **Superseded by:** none

## Context

Friday of week 1 (`docs/WEEK_1.md` §"Friday") ships the **calibration**
reliability dimension end-to-end:

1. **Confidence elicitation** — the harness must surface, parse, and store
   self-reported confidence from the agent under test, plus opportunistic
   logprob-derived confidence where the provider's API supports it.
2. **Calibration metrics** — Brier score, ECE (15 equal-mass bins),
   refusal calibration confusion matrix, and overconfidence rate per
   `docs/METHODOLOGY.md` §3.
3. **Pilot run plumbing** — multi-model, multi-task CLI surface so the
   end-of-week pilot run actually executes against `claude-opus-4-7`,
   `gpt-5.2`, and `gemini-2.5-pro` from a single command.

Friday closes the third — and final — open methodology question for v0.1
(Anthropic logprob asymmetry; auto-memory Q3). Each decision section below
is independently superseded-by-section-reference if it evolves in v0.2,
following the precedent of ADR-0002 §F.

## Decisions

### A. Resolve open methodology Q3 — Anthropic logprob asymmetry

`docs/METHODOLOGY.md` §3.1 commits us to "verbalized and (where the API
supports it) logprob-derived confidence are recorded." Anthropic's public
API surface does not expose per-token logprobs in any officially supported
form; OpenAI does (`logprobs=True, top_logprobs=N` on `chat.completions`);
Google's Gemini API exposes `responseLogprobs` on a subset of models
behind a config flag.

Three presentation options were considered:

1. **Verbalized-only headline; logprob-derived as a clearly-marked
   secondary column with explicit "N/A" cells where unavailable.**
2. **Two parallel calibration leaderboards** — one ranking on verbalized
   confidence (universal coverage), one on logprob (OpenAI-only for v0.1).
3. **Drop logprob confidence from v0.1 entirely**; deliver only verbalized
   confidence and defer logprob to v0.2.

**Decision: option 1.** Reasoning:

- Honest about the asymmetry. METHODOLOGY §"Known limitations and threats
  to validity" already commits us to documenting the verbalized/logprob
  divergence; surfacing both columns with explicit holes operationalizes
  that commitment.
- Single headline number per model preserves leaderboard ergonomics. A
  reader scanning "Calibration: Brier=X" gets one comparable number across
  every model; the deeper logprob column is one click away.
- v0.2 already has a path: when Anthropic exposes logprobs (or a research
  workaround crystallizes), populate the column for Claude and re-run.
  Until then, "N/A" is not a defect — it's a documented provider-API
  feature.
- Option 2 fragments the leaderboard story ("which model is best
  calibrated?" gets two answers) and is closer to what we'd ship in v0.2,
  not v0.1.
- Option 3 throws away signal we already have for OpenAI and (partially)
  Google, which is exactly the kind of "delete the data because the
  table looks ugly" move that would damage methodological credibility.

**Implementation contract:**

* `AgentResponse.logprob_avg: float | None` — mean per-token logprob over
  the model's response. `None` when the provider doesn't expose
  per-token logprobs OR when the agent doesn't request them.
* `ChatResponse.avg_logprob: float | None` — same field at the model-
  client layer, populated by provider clients that can supply it.
* `OpenAIClient._achat_provider` accepts a Steadfast-internal
  `logprobs: bool = False` kwarg; when true, sets the OpenAI SDK's
  `logprobs=True, top_logprobs=0`, averages per-chosen-token logprobs,
  and populates `ChatResponse.avg_logprob`.
* `AnthropicClient` and `GoogleClient` accept the kwarg and silently
  ignore it (consume + drop) — `avg_logprob` stays `None`. Google's
  partial logprob support is **deferred to v0.2**: the API surface is
  experimental and only on a subset of models, and shipping it for some
  Gemini variants but not others would create exactly the
  partial-coverage confusion option 1 was chosen to avoid for Anthropic.
* The "implied probability" transform from `avg_logprob` to a [0, 1]
  scalar is `exp(avg_logprob)` — the geometric mean of per-token
  probabilities, used as a calibration heuristic by Kadavath et al. 2022
  ("Language models (mostly) know what they know"). Documented in
  METHODOLOGY §3.1 with the citation; the metric layer applies the
  transform.
* Tracing populates `STEADFAST_LOGPROB_AVG` on the `chat` span when the
  provider supplied a value (the constant was reserved Wednesday in
  ADR-0003 §A.4).

Auto-memory's Q3 is **closed** by this decision.

### B. Confidence-elicitation prompt contract

The harness sets `Task.confidence_suffix` to a frozen prompt suffix; the
user-supplied agent (or the built-in `SimplePromptingAgent`) concatenates
it onto its prompt, prompts the model, and parses the model's output into
a structured `(answer, confidence, refused)` triple. The frozen prompt is
versioned via filename suffix at `prompts/confidence_v1.txt`.

The prompt instructs the model to emit a two-line structured tail:

```
ANSWER: <answer-text or the literal word REFUSE>
CONFIDENCE: <float in [0, 1]>
```

**Why a structured tail rather than free-form prose:** parsing a stated
probability out of free-form text ("I'm pretty sure", "definitely") is
itself a calibration question we'd be asking an LLM judge to answer.
Constraining the surface lets us measure what the model claims directly,
without an additional infrastructure judge sitting between the agent and
the metric. The 0-1 scale matches the Brier formulation and avoids a
post-hoc rescaling step.

**Why include `REFUSE`:** refusal calibration (METHODOLOGY §3.4) needs a
boolean flag per rep. Two alternatives were considered:

1. Heuristic regex for "I don't know" / "I'm not sure" phrases.
2. A second LLM call classifying refusal post-hoc.

Both are weaker. (1) is brittle — agents express refusal in many ways
("I'm not the right person to answer this"), and a regex bank
inevitably skews the metric toward whatever phrasing the bank
recognizes. (2) introduces another infrastructure judge (cost,
calibration risk, prompt-engineering surface). The structured-tail
approach makes refusal a **first-class part of the elicitation
contract**: the agent says yes-or-no on the wire, and the metric reads
the bit.

A REFUSE response is paired with `CONFIDENCE: 0.0` per the prompt; the
metric layer does not consume confidence on refused reps (refusal
calibration treats refusal as a separate signal — see §E below).

### C. Verbalized-confidence parser — retry-once-then-soft-fail

Friday's parser:

1. Strips the response text to find the last `ANSWER:` and `CONFIDENCE:`
   labels (case-insensitive). The "last" rule handles the case where the
   model echoes the format header or includes the words mid-prose; only
   the trailing structured tail is binding.
2. Reads everything between `ANSWER:` and `CONFIDENCE:` as the answer.
3. Parses `CONFIDENCE:` as a `float` in [0, 1]. Accepts `0.85`,
   `85%`, and `85` (latter two normalized to `0.85`).
4. Detects `REFUSE` (case-insensitive, surrounded by optional
   whitespace and trailing punctuation) on the answer line.

On parse failure (no recognizable `CONFIDENCE:` line, or out-of-range
value), the agent retries the LLM call **once** with a stricter "your
previous output did not include a CONFIDENCE: line; emit only the two
required lines" reminder. On a second failure, the agent populates
`AgentResponse.answer` with the raw response text, leaves
`confidence=None`, and sets `refused=False`. The metric layer skips
None-confidence reps with a logged warning per ADR-0002 §A.2; the rep
stays `COMPLETED` so it still contributes to consistency, format, and
trajectory metrics.

This matches the precedent in ADR-0003 §B.4 (rubric judge: one retry,
then signal failure; never silent fallback) but bends in one place: the
**rep stays completed** rather than failing. Reasoning: a fully-failed
rep would also lose the rep's contribution to consistency / trajectory
/ format dimensions, all of which are valid even when calibration data
is missing. Calibration's None-handling per ADR-0002 §A.2 is exactly
the right surface; we use it.

### D. Brier and ECE — pooling and bootstrap conventions

The bootstrap unit for the **aggregate** (model-level) Brier and ECE
metrics is the **pooled set of all (task, rep) squared errors** for that
model — i.e., a Brier resample is `np.random.choice(squared_errors,
size=N, replace=True)` over the flat pool.

Two alternatives were considered:

1. **Per-task Brier, bootstrap over tasks** — collapse N=10 reps into a
   per-task Brier, then bootstrap over the per-task means. Reflects
   task-level uncertainty.
2. **Cluster bootstrap** — resample tasks with replacement, then for
   each resampled task take all N=10 reps. Standard for hierarchical
   data (Field & Welsh 2007); the methodologically correct CI for
   clustered prediction data.
3. **Pooled bootstrap** (chosen) — flatten `(task, rep)` pairs and
   bootstrap over the pool.

**Decision: pooled bootstrap for v0.1, cluster bootstrap as a v0.2
upgrade.**

Reasoning:

- For the headline aggregate Brier, the 50-task leaderboard run's
  per-task variance and per-rep within-task variance are not
  separately the quantity we want to communicate; the leaderboard
  number is "how close is the agent's stated probability to the
  outcome, on a representative draw from this benchmark distribution."
  Pooling matches that framing.
- The pooled bootstrap **understates** the CI relative to cluster
  bootstrap when task-level variance dominates. We document this in
  METHODOLOGY §3.2 and §"Path to v0.2" so leaderboard readers know the
  CI is a lower bound on true uncertainty.
- 5-task pilot doesn't have enough tasks for cluster bootstrap to be
  meaningfully different from pooled bootstrap. The leaderboard run
  (50 tasks) is where the divergence matters; we have time before
  launch to add cluster bootstrap as a separate metric (`brier_v2` or
  `brier_clustered`), which the v0.1 pre-registration in METHODOLOGY
  §"Versioning" fully supports.
- Per-task Brier remains reported as a **secondary table** for diagnostics
  ("which tasks were poorly calibrated for this model"); only the
  aggregate is the leaderboard headline.

**ECE:** equal-mass binning is computed by sorting all (confidence,
outcome) pairs by confidence, partitioning into 15 contiguous chunks of
floor/ceil(N/15) samples each, then computing
`Σ_b (n_b / N) · |acc_b − conf_b|` per Guo et al. 2017 / Nixon et al.
2019. ECE is reported with a bootstrap CI over the same pool used for
Brier (each resample re-bins). When the pool has fewer than 15
forecasts (which the 5-task pilot can hit if some reps fail or refuse),
the metric falls back to `floor(N / 3)` bins and surfaces a warning
in the result; below 3 bins it returns `None` with a reason.

**Overconfidence rate:** Wilson 95% CI on `count(incorrect ∧ confidence
≥ 0.9) / count(answered)`. The pool is "answered" reps (refused reps
excluded) per METHODOLOGY §3.5.

### E. Refusal calibration — `Task.difficulty` field + confusion matrix

Refusal calibration (METHODOLOGY §3.4) requires distinguishing tasks
that are "hard or unanswerable" (where a well-calibrated agent should
hedge or refuse) from "normal" tasks (where it should answer).

**Decision:** Add a typed first-class field
`Task.difficulty: Literal["normal", "hard"] = "normal"` (default
`normal` for backwards compatibility with existing pilot tasks). The
existing `metadata.difficulty` string in `pilot_001.json` is
informational and remains as metadata; the new typed field drives the
metric.

Three alternatives were considered:

1. **Task metadata flag** (`metadata["hard"] = True`) — works, but
   the metric layer would need a magic key string and the Task contract
   wouldn't make refusal-eligibility legible.
2. **Separate "hard task" file or directory** — duplicates task content
   if a task ever needs to participate in both pools (which it
   shouldn't, but task-set evolution surface should be conservative).
3. **Dedicated typed field** (chosen) — Pydantic-typed Literal,
   ergonomic for the confusion matrix construction, additive change
   to ADR-0002 §A (CHANGELOG entry, no ADR-0002 amendment).

The confusion matrix is reported as a 2×2 with each cell carrying its
Wilson 95% CI:

|              | hard tasks | normal tasks |
| ------------ | ---------- | ------------ |
| **refused**  | TR (good)  | FR (bad)     |
| **answered** | FA (bad)   | TA (good)    |

Headline scalars: refusal sensitivity (TR / (TR + FA)) and refusal
specificity (TA / (TA + FR)), each with a Wilson CI. v0.2 may
introduce an F1-style aggregate; v0.1 keeps the cells visible.

### F. Pilot benchmark task surface — `customer_support_pilot`

The Friday pilot run targets a 5-task `customer_support_pilot`
benchmark. Tasks are hand-authored under
`benchmarks/customer_support/pilot_*.json`; the existing `pilot_001`
seed is one of the five (its `judge: exact_match` form is preserved).

The CLI's `--benchmark customer_support_pilot` resolves to "every
`benchmarks/customer_support/pilot_*.json` file", sorted lexicographically
by ID. v0.2 will introduce a `manifest.json` per benchmark directory if
non-glob inclusion becomes useful.

**Hard task representation:** at least one of the five carries
`difficulty: "hard"`. The pilot uses a deliberately
under-specified question whose ground truth is "this cannot be
answered without information not in the prompt"; the rubric judge
scores hedging-or-refusal answers as passing.

### G. CLI surface — `--benchmark`, `--metrics`, `--models`

Three CLI extensions land Friday so the end-of-week pilot run is one
command:

1. **`--benchmark NAME`** — directory-resolution as in §F. Mutually
   exclusive with `--task`.
2. **`--models a,b,c`** — comma-separated list of target model IDs.
   Each model gets its own checkpoint subdirectory and its own
   per-task `RunResult` JSON. The CLI iterates models sequentially
   (parallelism is per-rep within a model, bounded by the existing
   per-client semaphore — adding cross-model parallelism would multiply
   API spend without improving statistics).
3. **`--metrics consistency,calibration`** — comma-separated metric
   dimensions. After a model's reps complete and are judged, the CLI
   computes the requested metrics from the already-persisted
   `RunResult`s. Each (model, metric, task) triple writes a Pydantic-
   typed result JSON; the HTML report aggregates them.

`--task --model --reps` continues to work as the single-task surface
for inner-loop development; `--benchmark --models` is the leaderboard-
shaped surface.

### H. Pilot-run cost guardrails

The Friday pilot run hits real APIs. Estimated cost ceiling for the
spec'd command (5 tasks × 3 frontier models × 10 reps × ~500 token
average request and response, plus K=5 paraphrases × pairwise rubric
calls for consistency, plus rubric judge per rep, plus embeddings):

* Target-model spend: 150 reps × ~$0.003-0.015 = ~$0.45-$2.25.
* Paraphrase + paraphrase-validator: 5 tasks × (~5 generator + ~5
  validator) × $0.0006 = ~$0.03.
* Output-consistency rubric judge: 5 tasks × C(5,2)=10 pairs × 3
  models × $0.0006 = ~$0.09.
* Outcome rubric judge: 150 reps × $0.0006 = ~$0.09.
* Embeddings: trivial (<$0.01).

Order-of-magnitude estimate: **$1-3** for the full Friday pilot. Well
below the $50-150/week ceiling in WEEK_1.md §"What can go wrong" #5.

The CLI does **not** add a budget-cap flag in v0.1 — the cost is small
and surfacing a flag would imply an enforcement mechanism we don't have
(post-hoc cost computation only). v0.2 may add a pre-flight estimator
under `steadfast estimate-cost`.

## Consequences

**Positive**

- Last open methodology question for v0.1 is closed; launch is unblocked
  on the methodology axis.
- One headline number per (model, calibration metric) keeps the
  leaderboard surface honest and ergonomic.
- Friday's PR is purely additive: every new field defaults to a
  no-change value, and existing tests stay green.

**Negative**

- Pooled bootstrap on Brier/ECE understates CI relative to cluster
  bootstrap (Field & Welsh 2007). Mitigation: documented in METHODOLOGY
  §3.2; cluster bootstrap is on the v0.2 roadmap with a clear migration
  path (new metric name per METHODOLOGY §"Versioning").
- Logprob coverage is 1/3 of frontier providers in v0.1 (OpenAI only);
  Anthropic and Google show "N/A". Mitigation: explicit holes in the
  reported table; verbalized headline keeps every model comparable.
- The structured ANSWER/CONFIDENCE tail is a small but real prompting
  intervention that nudges the model toward a particular shape of
  output. We don't know whether nudging the format also nudges the
  underlying confidence distribution. METHODOLOGY §3.1 documents this
  as a known limitation; v0.2 may compare elicitation prompts.

## Path to v0.2

- **Cluster bootstrap** for aggregate Brier / ECE — separate metric
  name (`brier_v2` / `brier_clustered`), full leaderboard re-run.
- **Logprob coverage parity** — track Anthropic and Google API
  evolution; populate the column when supported on a per-provider basis.
- **Confidence-elicitation prompt comparison** — measure how much the
  prompt shape moves the calibration numbers. Likely a methodology
  ablation rather than a leaderboard change.
- **Refusal F1 / Matthews correlation** — single-scalar refusal
  calibration once we have leaderboard data showing whether the 2×2
  matrix is too granular for headline display.
- **Pre-flight cost estimator** under `steadfast estimate-cost`.

## References

- `docs/METHODOLOGY.md` §3 (calibration), §3.1 (elicitation),
  §"Statistical conventions", §"Known limitations".
- `docs/WEEK_1.md` §"Friday".
- ADR-0001 — infrastructure-model lock (gpt-5.2 default rubric / paraphrase).
- ADR-0002 — Tuesday core abstractions; §A.1 (confidence contract),
  §A.2 (None-confidence handling), §A.3 (metadata typing), §B.3 (raw
  carve-out), §C.3 (loud failures).
- ADR-0003 — Wednesday tracing + judges; §A.4 (logprob span attribute
  reservation), §B.4 (judge retry policy).
- ADR-0004 — Thursday consistency + stats; §H (bootstrap edge cases).
- Tian et al. (2023), *Just Ask for Calibration*,
  https://arxiv.org/abs/2305.14975 — verbalized-confidence
  miscalibration in current LLMs.
- Kadavath et al. (2022), *Language Models (Mostly) Know What They Know*,
  https://arxiv.org/abs/2207.05221 — `exp(avg_logprob)` as a calibration
  heuristic.
- Guo et al. (2017), *On Calibration of Modern Neural Networks*,
  https://arxiv.org/abs/1706.04599 — original ECE formulation.
- Nixon et al. (2019), *Measuring Calibration in Deep Learning*,
  https://arxiv.org/abs/1904.01685 — equal-mass binning argument.
- Field & Welsh (2007), *Bootstrapping clustered data*, *JRSS-B* 69(3),
  369-390 — cluster bootstrap baseline for v0.2 path.
- OpenAI Chat Completions logprobs:
  https://platform.openai.com/docs/api-reference/chat/create#chat-create-logprobs
- Anthropic API surface (logprobs not in public response shape):
  https://docs.anthropic.com/en/api/messages
- Google Gemini logprobs (responseLogprobs flag):
  https://ai.google.dev/gemini-api/docs/text-generation#logprobs
