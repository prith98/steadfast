# ADR-0006: Robustness dimension methodology + paired bootstrap (Week 2)

- **Status:** Accepted
- **Date:** 2026-05-11
- **Methodology version:** v0.1
- **Supersedes:** none
- **Superseded by:** none

## Context

Week 2 of the v0.1 build (per `docs/WEEK_2.md`) ships the **robustness**
reliability dimension end-to-end. Where ADR-0005 closed every open
calibration question for v0.1 launch, METHODOLOGY §2 left several
robustness sub-decisions under-specified — sufficient for the metric
*output* shape to be pinned, but not the implementation. This ADR
commits each.

Bundled into one ADR (per the ADR-0002 §F precedent — single ADR per
day's worth of decisions, sectioned, individual sections superseded by
reference if they evolve in v0.2). Sections cover:

1. Closure of the LangGraph facet of auto-memory open methodology Q2
   (the trajectory contract for the LangGraph adapter scaffold landing
   Thursday).
2. Perturbation seed strategy across all four robustness sub-metrics.
3. Distractor bank source / curation / versioning.
4. Contradiction label schema + classifier choice (rule-based vs
   LLM-judged).
5. Long-context curve fit + slope interpretation.
6. Paired bootstrap CI on robustness deltas — the new statistical
   primitive.
7. Triage of the 2026-05-10 pilot's v0.2 backlog items between v0.1
   clarification fixes and v0.2-proper metric-version events.

## A. Resolve auto-memory Q2 — LangGraph trajectory contract

**Status entering this ADR:** ADR-0002 §A.2 partially resolved Q2 —
`AgentResponse.trajectory` is optional, trajectory consistency returns
N/A on empty trajectories. The remaining facet is whether the
`LangGraph` adapter (landing Thursday per `docs/WEEK_2.md`) must
*enforce* trajectory exposure or may pass through agents that don't
expose tool calls.

**Decision:** the LangGraph adapter populates `trajectory` from
`state.messages` when the compiled graph exposes tool calls (the stable
LangGraph surface for `ToolMessage` / `AIMessage.tool_calls`); when no
tool calls are present, `trajectory` is the empty list `[]`, *not*
`None`. Trajectory consistency falls through to the same N/A path
toolless agents already use per ADR-0004 §G.

Three alternatives were considered:

1. **Adapter MUST expose trajectory; raise on empty.** Rejected:
   fragile across graph shapes (subgraphs, custom reducers, bare
   chat-only graphs). Rejects valid use cases.
2. **Empty trajectory is `None`, not `[]`.** Rejected: `None` would
   require type-narrowing every consumer; the empty-list contract from
   ADR-0002 §A.3 is already the surface trajectory consistency
   exercises.
3. **Empty list when absent (chosen).** Matches the OpenAI Agents SDK
   adapter precedent; the N/A path is exercised by the consistency
   dimension; the explicit "empty list, not None" rule keeps the
   public type contract stable.

**Implementation contract:**

* `LangGraphAdapter.arun(task)` invokes the compiled graph, awaits
  completion, extracts the final answer from the last AI-message text,
  extracts trajectory from any `tool_calls` on intermediate messages
  (preserved in invocation order).
* Returns `AgentResponse(answer, confidence=…, trajectory=[…], …)` —
  confidence comes from the standard `perturbations.confidence` parser
  applied to the final message text per ADR-0005 §B-C.
* Adapter docstring documents the assumed `state.messages` shape and
  notes that exotic graph shapes (subgraphs, custom reducers) may
  require subclassing the adapter.

Auto-memory Q2 is **closed** by this decision in full (Q1 closed by
ADR-0004 §A; Q3 closed by ADR-0005 §A; Q2's LangGraph facet closed
here).

## B. Perturbation seed strategy

METHODOLOGY §"Statistical conventions" already commits to "all non-API
stochasticity (paraphrase generation, perturbation sampling, bootstrap
resampling) uses seeds derived from the task ID to ensure
reproducibility across runs." This ADR makes that concrete for the
four robustness perturbations.

**Decision:** each perturbation derives its seed from
`sha256(f"{task.id}:{perturbation_kind}:v1".encode())[:8]`, interpreted
as a big-endian unsigned integer to seed `random.Random` /
`numpy.random.Generator`. The per-perturbation kind suffix
(`"typo"`, `"distractor"`, `"long_context"`, `"contradiction"`)
ensures different perturbations on the same task draw different
randomness while remaining reproducible. The `:v1` suffix lets a
future seed-strategy change (extremely unlikely, but possible if a
perturbation's RNG consumption pattern changes incompatibly) bump to
`:v2` without invalidating existing leaderboard entries that were
computed at `:v1`.

For the contradiction perturbation, which decides whether to corrupt
on a per-tool-call basis (METHODOLOGY §2.3: "with probability 0.3"),
the per-call seed extends to
`sha256(f"{task.id}:contradiction:v1:tool{tool_call_idx}".encode())[:8]`
so reordering tool calls on the agent side doesn't shift the
corruption pattern across calls. The `tool` prefix matches the
`rep{rep_idx}` style used for the per-rep extension below — both
extensions are tagged with their kind so the suffix is self-describing
and the two extensions can in principle stack (a per-rep contradiction
perturbation would suffix `:rep{i}:tool{j}`).

The seed also rides on the rep itself when the perturbation has a
per-rep stochastic dimension (e.g., the typo perturbation's character-
position selection); reps within a single (task, perturbation) get
distinct draws via
`sha256(f"{task.id}:{kind}:v1:rep{rep_idx}".encode())[:8]`. This
preserves N=10 distributional measurement per METHODOLOGY §"Multi-run
by default" — without per-rep seeding, all 10 reps would receive the
identical perturbed input, collapsing the distribution to a point.

**Why not just `random.seed(task.id)`:** Python's `random.seed(str)`
hashes the input to an integer in an implementation-defined way; CPython
3.11 guarantees stability via `hash(str)` only because `PYTHONHASHSEED`
is fixed for the seeded RNG context, but relying on this is brittle
across implementations and unnecessary when sha256 gives us a stable,
explicit derivation.

## C. Distractor bank — generation, curation, freezing

METHODOLOGY §2.2 specifies "200-800 tokens of plausible-but-irrelevant
context drawn from a curated bank that's topically adjacent" but does
not specify how that bank is produced or where it lives.

**Decision:** LLM-generated bank with a mandatory human-review pass,
frozen as a versioned JSON artifact at
`benchmarks/<domain>/distractors_v1.json`. Per-domain because
"topically adjacent" is domain-specific (a customer-support distractor
should read like a customer-support adjacency, not a code-repair one).

Generation pipeline (one-shot per domain, run via
`scripts/generate_distractor_bank.py`):

1. Read every task in the domain.
2. Call GPT-5.2 with the frozen prompt at
   `prompts/distractor_bank_v1.txt`. The prompt asks for N=50 prose
   snippets that are topically adjacent to customer-support themes
   (returns / shipping / billing) but answer-irrelevant — i.e., a
   reader would say "this is about customer support" but no snippet
   contains the answer to any task in the bank.
3. **Manual review pass.** The script writes a draft to
   `benchmarks/<domain>/distractors_v1.draft.json` and exits with a
   prompt to the operator: "review the draft, delete or rewrite any
   snippet that subtly contradicts a task's ground truth, then
   `mv draft.json v1.json` to commit." Without the rename step, the
   bank is not picked up by the metric — fail-loud rather than
   silently shipping unaudited LLM output.
4. Once committed, `distractors_v1.json` is content-addressed by the
   `_v1` suffix per the existing `prompts/*_v1.*` pattern. Any
   regeneration creates `_v2`, requiring a new metric name and a
   leaderboard re-run per METHODOLOGY §"Versioning".

Per-task selection at runtime: K=1 distractor per (task, rep),
deterministically chosen by index `seed % len(bank)` from the seeded
RNG of §B above. Token-count gating ensures the chosen snippet falls
in the 200-800 token range; if not, increment the index (still
deterministic) until a snippet fits.

Three alternatives were considered:

1. **Hand-curate ~50 entries per domain.** Higher quality bar, slower
   to author, harder to scale across the 3 v0.1 domains. Rejected
   primarily on the calendar — Tuesday's deliverable would slip into
   Wednesday and force contradiction handling later in the week.
2. **Sample Wikipedia paragraphs from related categories.**
   Reproducible (with a Wikipedia dump pin) but heavyweight (need a
   dump artifact in the repo or downloaded via a build step) and
   harder to control for "topical adjacency" — Wikipedia categories
   are noisy proxies for the semantic property we want.
3. **Generate at runtime per task** from a frozen prompt. Reproducible
   via seed but adds infrastructure-LLM cost on every benchmark run
   (the per-task generator + per-task validator at 50 tasks × N=10
   reps = 1000 calls per leaderboard model). Rejected on cost.
4. **LLM-generated + human-reviewed + frozen JSON (chosen).** One-time
   generation cost (~$0.50 per domain), human review is the quality
   gate, version suffix on the JSON file enforces v0.2 re-generation
   as a metric-version event.

**K=1 distractor per task:** METHODOLOGY says "200-800 tokens of
plausible-but-irrelevant context"; one snippet hits that target
naturally and avoids stitching seams between snippets that could
themselves become a confound. v0.2 may add a multi-snippet variant if
single-snippet distractor robustness ceilings out across all frontier
models.

## D. Contradiction labels + classifier

METHODOLOGY §2.3 specifies the metric is "a 3-way categorical, not a
single scalar" but does not pin (a) the precise label definitions or
(b) how an agent's behavior is mapped to a label.

**Label schema:**

| Label | Definition (an agent earns this label when…) |
| --- | --- |
| `detected` | The agent emits the literal `REFUSE` token (per ADR-0005 §B), or the answer text contains a phrase from the frozen detection-phrase list at `prompts/contradiction_detection_phrases_v1.txt` ("conflicting", "inconsistent", "I cannot reconcile", "the data appears contradictory", "I need to flag…"). |
| `retried_or_escalated` | Not `detected`, but the trajectory shows at least one tool call repeated with the same args after a corrupted response, OR the answer text contains an escalation phrase from the frozen list ("escalating to a human", "I need clarification before…", "passing this to…"). |
| `hallucinated` | Neither of the above — the agent produced a confident-looking answer despite the corrupted tool input. |

The decision rules are evaluated in order: `detected` wins over
`retried_or_escalated`, which wins over `hallucinated`. The frozen
phrase lists are versioned via `_v1` suffix; any change requires a
new metric name per METHODOLOGY §"Versioning".

**Classifier choice — rule-based, not LLM-judged.** Three alternatives
were considered:

1. **Dedicated `ContradictionRubricJudge` LLM judge per rep.** Highest
   quality ceiling — an LLM can read the agent's prose and decide
   whether it flagged the contradiction in unanticipated phrasing.
   Cost: a fourth infrastructure-LLM judge surface to maintain
   (alongside paraphrase, paraphrase-validator, rubric judge per
   ADR-0001), with attendant ensemble-judge follow-up obligations and
   bias-surface concerns.
2. **Hybrid: rule-based for "obviously detected" (REFUSE token) and
   "obviously hallucinated" (no retry, no flagging language); LLM
   judge for the residual.** Adds branching surface; complicates the
   reproducibility manifest (the rule-based path doesn't need an
   infra LLM but the residual does, so the run cost is data-dependent).
3. **Rule-based only, with the phrase list pinned via the same
   `_v1` versioning the metric infrastructure already uses (chosen).**

**Why the lean:** keeps the v0.1 infrastructure-LLM surface to three
(paraphrase, paraphrase-validator, rubric judge) — a fourth is real
architectural weight (cost surface, bias surface, ensemble path
obligation) and the rule-based approach exercises the contradiction
perturbation end-to-end with no new judge. The known-precision-loss
on the `detected` bucket (an agent that prose-flags a contradiction
without using our keyword list will be misclassified `hallucinated`)
is the v0.1 cost; v0.2's `ContradictionRubricJudge` upgrade has a
clean drop-in path because the
`classify_contradiction_response(task, response, corrupted_calls)`
signature is judge-shape compatible.

**N/A path for toolless agents:** the contradiction perturbation
requires intercepting a tool call to corrupt. For agents whose
trajectory is empty (no tools called for this task), the metric
returns `ContradictionResult(value=None, reason="agent did not call any
tools")` — same N/A pattern trajectory consistency uses per
ADR-0004 §G.

**Reporting shape:** marginal proportions
`(p_detect, p_retry, p_halluc)` with Wilson 95% CIs per cell. The
three CIs are not jointly bounded (sum-to-1 only at the point
estimate); the result's `notes` field documents this honestly. v0.2
may add a Dirichlet-multinomial CI if the marginal Wilson surface
proves insufficient for leaderboard interpretation.

## E. Long-context fit

METHODOLOGY §2.4 specifies a "logistic fit" reporting "the slope
coefficient" but does not specify which fitter, the exact functional
form, the CI method, or what derived quantity (if any) to surface
alongside the slope.

**Decision:**

* **Functional form:**
  `p(L) = 1 / (1 + exp(-(a + b · log10(L))))` — the logit of success
  probability is linear in `log10(tokens)`. Taking `log10(L)` rather
  than `L` directly puts the four METHODOLOGY-specified context
  lengths (4k, 16k, 64k, 128k) on an evenly-spaced x-axis (each
  4×-jump is one unit on `log10`), which is both ergonomic for the
  fit and statistically motivated (token degradation is widely
  observed to scale on the log axis, e.g., Liu et al. 2024 *Lost in
  the Middle*).
* **Fitter:** `scipy.optimize.curve_fit` with default Levenberg-
  Marquardt — no new dependency. (Considered `statsmodels.GLM` with
  `Binomial()` family + logit link, which would handle per-rep
  observations natively rather than aggregated rates and provide
  built-in standard errors. Rejected for v0.1: adds a ~150 MB
  transitive closure for one consumer. The CLAUDE.md "dependency
  tree as quality signal" commitment is the deciding factor; v0.2
  may bring `statsmodels` in if a second consumer materializes.)
* **CI method:** bootstrap over tasks (resample tasks with their N=10
  reps as a unit; refit per resample; take the 2.5/97.5 percentiles of
  the resampled `slope` and `l50`). Reuses
  `stats.bootstrap.bootstrap_ci` with a custom statistic function that
  accepts the per-task aggregated arrays. 10,000 resamples per
  METHODOLOGY §"Statistical conventions".
* **Derived quantity:** `l50 = 10^(-a/b)` — the token count where
  predicted success probability drops to 0.5. Reported alongside the
  raw slope coefficient `b` for two-audience clarity (stat-literate
  readers want the slope; everyone wants the `l50` half-life).
* **Convergence-failure fallback:** when `curve_fit` raises
  `OptimizeWarning` / `RuntimeError` (typically because the empirical
  curve is nearly flat — the model degrades gracefully across all four
  tiers), the metric reports `slope = slope_ci = l50 = l50_ci = None`,
  sets `fit_converged = False`, and surfaces the empirical curve only.
  The empirical curve is the methodologically primary artifact; the
  fit is a summarization aid. This matches the spirit of ADR-0005's
  ECE small-N fallback (graceful degradation with a `reason` field
  rather than crashing).

**Slope interpretation rule for the report:** negative `b` means the
agent degrades with length (the expected case); positive `b` means
the agent gets *better* with more context (a real surprise worth
flagging in the report rather than silently presenting). The
sanity-check rubric in WEEK_2.md §"Friday" lists positive slope as a
"warrants investigation" signal.

## F. Paired bootstrap for robustness deltas

METHODOLOGY §2 requires "95% CI on the delta itself, not the two
endpoints separately." This commits us to a paired-data CI method but
does not specify which one.

**Decision:** paired bootstrap, resampling **tasks** (with both arms'
N=10 rep arrays carried as a unit). The statistic is

```
delta_per_task[i] = perturbed_success_rate[i] - clean_success_rate[i]
delta_aggregate   = mean(delta_per_task)
```

Each bootstrap resample draws task indices with replacement, computes
`delta_per_task` over the resampled task set, takes the mean. The
2.5/97.5 percentiles of the 10,000 resampled means are the 95% CI on
the aggregate delta.

Three alternatives were considered:

1. **Resample (task, rep) pairs flatly across both arms** — closer to
   pooled bootstrap; ignores task-level variance; understates CI.
   This is the same pattern ADR-0005 §D documented as "understates the
   CI relative to cluster bootstrap" for Brier / ECE. The
   understatement is more severe here because the per-task pairing
   carries critical signal (a task that's noisy in both arms shouldn't
   get its noise double-counted into the CI).
2. **Resample each arm independently and subtract endpoint CIs.** Loses
   the pairing entirely. The CI on the delta would be much wider than
   reality justifies because the within-task correlation between
   clean and perturbed outcomes (driven by per-task difficulty)
   doesn't get factored out.
3. **Resample tasks-with-both-arms (chosen).** The paired bootstrap.
   Standard for paired-sample CIs; preserves the within-task
   correlation; cleanly composes with `scipy.stats.bootstrap` via a
   custom statistic function.

**Why paired here but pooled in calibration (ADR-0005 §D):** Brier /
ECE forecasts are single-arm — there's no natural "before / after"
pairing per rep, only aggregate-level claims about distributional
calibration. Robustness deltas are *fundamentally* paired: the
methodology speaks of "success-rate delta on the same task set"
(METHODOLOGY §2). Without per-task pairing, the delta loses its
methodological meaning, so paired bootstrap is required from day one
— there is no "v0.2 cluster upgrade" path because the pairing is
intrinsic.

**Implementation:** new module `stats/paired_bootstrap.py` exposing
`paired_bootstrap_ci(clean_rates, perturbed_rates, *, n_resamples=10_000,
ci_level=0.95)`. Internally calls `scipy.stats.bootstrap` with
`method="BCa"` (matching the existing `stats/bootstrap.py` defaults)
on the per-task delta array. Edge cases (empty input, N<2, zero-
variance) raise `ValueError` consistently with `stats/bootstrap.py`
per ADR-0004 §H.

The hand-computed test cases verify:

* Identical arms → delta = 0, CI brackets 0.
* All-clean-pass / all-perturbed-fail → delta = -1, CI is degenerate
  at -1 (`scipy.stats.bootstrap` returns NaN for zero-variance; we
  surface `delta_low = delta_high = -1.0` per ADR-0004 §H precedent).
* A small synthetic 3-task case where the per-task delta is hand-
  computable: clean = `[1.0, 0.8, 0.6]`, perturbed = `[0.7, 0.5, 0.4]`,
  delta_per_task = `[-0.3, -0.3, -0.2]`, mean delta = `-0.267`.

Citation for paired bootstrap: Efron & Tibshirani (1993),
*An Introduction to the Bootstrap* §10 ("Confidence intervals based on
bootstrap percentiles") for the standard percentile / BCa method;
Field & Welsh (2007) for the paired/clustered extension that
underwrites the choice to resample at the task (cluster) level.

## G. Triage of v0.2 backlog items between week 2 and v0.2 proper

The 2026-05-10 pilot run surfaced four concrete items now tracked in
auto-memory `project_v02_backlog.md`. Each requires a clean classification
between "ships this week as a clarification fix" and "rides v0.2 as a
metric-version event."

| # | Backlog item | This-week disposition | Rationale |
| --- | --- | --- | --- |
| 1 | `ExactMatchJudge.canonicalize` doesn't normalize hyphenation | **Ships this week as clarification fix.** | The original ADR-0003 §B.3 substring-containment intent was hyphenation-insensitive; the omission was a bug relative to that intent. Score flips are one-directional (more matches, never fewer). The pilot_001 ground-truth tightening (`"30 days"` → `"30-day"`) is the second half of the same fix and rides as its own commit for reviewability. Both shipped 2026-05-11. |
| 2 | Empty / parse-failed responses counted as "answered" in refusal calibration | **Defers to v0.2 (`refusal_v2`).** | This changes the denominator of the refusal sensitivity scalar — model rankings on the refusal cell can flip in either direction depending on which models had empty responses on hard tasks. New metric name + full leaderboard re-run required per METHODOLOGY §"Versioning". |
| 3 | `measure_output_consistency` returns spurious 1.0 when `n_empty == k` | **Defers to v0.2 (`consistency_v2`).** | This changes the consistency scalar from "spurious 1.0" to "honest N/A" — same direction of impact for all affected (model, task) cells, but the change in surface (a previously-numeric cell becomes a None) requires consumers (the HTML report, the leaderboard schema) to handle the new shape. Better landed alongside the leaderboard schema work in v0.2. |
| 4 | Document Gemini bursty-paraphrase content-filter behavior | **Shipped 2026-05-11 morning.** | Pure docs amendment to METHODOLOGY §"Known limitations". No code, no metric event. |

The criterion separating items 1 from items 2 and 3 is **direction of
impact + intent of original spec**: a clarification fix only flips
scores in one direction and was already implied by the original spec;
a metric-version event flips scores in either direction (or changes
the surface shape) and represents a genuine semantic change.

## Consequences

**Positive**

- Auto-memory Q2 closed in full. All three v0.1 open methodology
  questions now have ADR-codified resolutions (Q1: ADR-0004 §A;
  Q2: ADR-0002 §A.2 + this ADR §A; Q3: ADR-0005 §A).
- Every Mon-Fri week-2 methodological choice is in one document with
  alternatives + rationale recorded.
- The v0.2 backlog triage criterion (§G) generalizes — future pilots
  that surface similar issues can be classified by the same rule.
- `stats/paired_bootstrap.py` adds the third statistical primitive
  (alongside `bootstrap.py` and `wilson.py`) needed for v0.1 metrics.
  The reliability dimensions now cover all three primitive shapes
  (single-arm bootstrap for calibration; binomial Wilson for
  proportions; paired bootstrap for robustness deltas).

**Negative**

- The rule-based contradiction classifier has known-loose precision on
  the `detected` bucket — agents that prose-flag contradictions in
  phrasings outside the frozen list are misclassified `hallucinated`.
  Mitigation: the `_v1` versioning on
  `prompts/contradiction_detection_phrases_v1.txt` allows targeted
  expansion as v0.1 pilots surface false negatives;
  `ContradictionRubricJudge` is the v0.2 upgrade path.
- The distractor bank's quality is gated by the human-review pass.
  An incomplete review pass that lets a subtle ground-truth-
  contradicting snippet ship would bias the distractor metric toward
  "model is brittle" when the real failure is bank contamination.
  Mitigation: §C's draft-then-rename gate is fail-loud; the v0.2
  path is to introduce automated bank-validation (LLM verifier
  comparing each snippet against every task's ground truth).
- The long-context fit's convergence-failure fallback means that
  exceptionally robust models ("flat curve across 4k-128k") get a
  `slope = None` row in the report. This is an honest signal but may
  read as a "we don't know" gap to first-time leaderboard readers;
  the report copy needs to explicitly distinguish "no degradation
  detected" from "fit failed."

## Path to v0.2

- **`ContradictionRubricJudge`** — LLM-judged classifier upgrade for
  the contradiction metric (per §D). Drop-in via the existing
  `classify_contradiction_response` signature.
- **`statsmodels.GLM`-based long-context fit** — if a second consumer
  for the GLM surface materializes (e.g., per-domain robustness GLMs
  in v0.2 analysis), promote the long-context fit to the GLM path
  with built-in standard errors.
- **Multi-snippet distractor bank variant** — if single-snippet
  distractor robustness ceilings out across frontier models in v0.1.
- **Dirichlet-multinomial CI** for the contradiction 3-vector — if
  the marginal Wilson cells prove insufficient for leaderboard
  interpretation.
- **Automated distractor-bank validation** — LLM verifier comparing
  each snippet against every task's ground truth, removing the
  human-review gate.

## References

- `docs/SPEC.md` — what we're building.
- `docs/METHODOLOGY.md` §2 (robustness), §"Statistical conventions",
  §"Versioning".
- `docs/WEEK_2.md` — the daily breakdown that this ADR codifies.
- ADR-0001 — infrastructure-model lock (constrains §C / §D from adding
  judges casually).
- ADR-0002 §A.2 — trajectory optionality (partially resolved Q2;
  this ADR §A closes the LangGraph facet).
- ADR-0002 §F — single-ADR-per-day-of-decisions precedent.
- ADR-0003 §B — Verdict surface; §B.3 substring-containment intent
  underwriting §G item 1's "clarification fix" classification.
- ADR-0004 §G — N/A handling for empty trajectories (the pattern
  this ADR §A and §D reuse).
- ADR-0004 §H — bootstrap edge-case handling (the pattern this ADR
  §F reuses for paired bootstrap).
- ADR-0005 §B — REFUSE token elicitation contract (consumed by §D
  classifier).
- ADR-0005 §D — pooled bootstrap rationale and the cluster-bootstrap
  v0.2 path; this ADR §F's "paired here vs pooled there" contrast.
- Auto-memory `project_open_methodology_questions.md` (Q2 closure) and
  `project_v02_backlog.md` (§G triage source).
- Efron & Tibshirani (1993), *An Introduction to the Bootstrap*
  (paired-bootstrap baseline).
- Field & Welsh (2007), *Bootstrapping clustered data*, *JRSS-B*
  69(3), 369-390 (cluster-bootstrap baseline; same reference as
  ADR-0005 §D).
- Liu et al. (2024), *Lost in the Middle: How Language Models Use Long
  Contexts*, *TACL* 12, 157-173 (motivates `log10(L)` x-axis in §E).
- Ribeiro et al. (2020), *Beyond Accuracy: Behavioral Testing of NLP
  Models with CheckList*, *ACL* (typo-perturbation prior art per
  METHODOLOGY §2.1).
