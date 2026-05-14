# ADR-0008: Benchmark expansion to ~50 tasks across 3 domains (Week 4)

- **Status:** Accepted
- **Date:** 2026-05-14
- **Methodology version:** v0.1
- **Supersedes:** none
- **Superseded by:** none

## Context

Week 4 of the v0.1 build (per `docs/WEEK_4.md`) grows the benchmark
from the 5-task `customer_support_pilot` to the SPEC v0.1 target of
~50 tasks across 3 domains. Weeks 1-3 shipped the four reliability
dimensions (consistency, calibration, robustness, safety) on a
single 5-task pilot domain; the dimensions are methodologically
complete, but the bank size makes cross-model claims indefensible
(per `notes/week2_findings.md` §"Cost reality check" and
`notes/week3_findings.md` Finding 3 — bank-size-10 Wilson width
[0.000, 0.278] on a 0/10 catastrophic rate blocked all pairwise
model-ranking claims on the safety pilot).

METHODOLOGY §§1-4 already pin every per-metric contract; this ADR
commits **benchmark composition** — the cross-domain task counts,
per-domain judge contracts, ground-truth construction conventions,
difficulty distribution, and the operator-audit gate on the new
banks. Bundled into one ADR per the ADR-0002 §F / ADR-0006 /
ADR-0007 precedent — single ADR per week, sectioned, individual
sections superseded by reference if they later evolve in v0.2.

Sections cover:

- **§A** — Per-domain task counts and overall mix
- **§B** — customer_support expansion contract (extension of the
  existing pilot pattern)
- **§C** — code_repair contract: **rubric-judged diffs**, no
  executable sandbox in v0.1 (mirrors ADR-0007 §B's prompt-only
  threat-model rationale)
- **§D** — multi_hop_research contract: **self-contained
  synthesized prompts** (multi-hop reasoning over evidence given
  in the prompt; no live web/search tool fixture in v0.1)
- **§E** — Ground-truth distribution: target ~40% exact_match /
  ~60% rubric across the full benchmark
- **§F** — Operator-audit gate on the new tasks via per-domain
  `_review.json` manifest (mirrors ADR-0006 §C / ADR-0007 §G's
  `review_status` pattern, scoped to a directory rather than a
  single file)
- **§G** — Versioning + v0.2 paths

## A. Per-domain task counts — ~17/domain, ~50 total

**Decision:** Target ~17 reviewed tasks per domain × 3 domains ≈
50 tasks total. Tracks the SPEC v0.1 commitment ("50+ tasks
across 3 domains"). Customer_support grows from 5 to ~17;
code_repair and multi_hop_research bring up to ~17 each from
zero.

Three alternatives were considered:

1. **30 tasks across 3 domains (~10 each).** Faster to ship,
   lighter rubric judge cost on every pilot. Rejected: CI width
   at n=10 is the same problem that blocked pairwise rankings on
   the week-3 safety pilot (per ADR-0007 §E's Wilson [0.000,
   0.278] on a 0/10 rate). Under-sized banks make the dimension
   measurement uninterpretable, not just imprecise.
2. **~17/domain ≈ 50 total (chosen).** Matches SPEC, is the
   minimum bank size that produces interpretable cross-model CIs
   on a single dimension at the leaderboard scale, is the
   maximum one week of authoring + audit can realistically
   deliver.
3. **30/domain ≈ 100 total.** Highest fidelity; ~2x the rubric
   judge cost on every pilot; slipped delivery by a week+.
   Defers to v0.2 if v0.1 leaderboard CIs prove too wide on
   important comparisons (which they shouldn't at n=17 per the
   safety-pilot evidence — the catastrophic-rate Wilson width on
   a 0/17 rate is roughly [0.000, 0.184], approximately
   tightenable for pairwise model claims).

**Implementation contract:**

- Customer_support: extend the existing `pilot_*.json` set with
  `cs_*.json` files, keeping the pilot tasks as the "smoke" subset
  (used by `--benchmark customer_support_pilot` for CI smoke runs)
  and the full reviewed set under `--benchmark customer_support`.
- code_repair: new directory `benchmarks/code_repair/` with
  `cr_*.json` task files.
- multi_hop_research: new directory `benchmarks/multi_hop_research/`
  with `mhr_*.json` task files.

## B. customer_support expansion contract

**Decision:** Extend the existing customer_support pilot pattern.
The 5 `pilot_*.json` tasks remain (smoke + back-compat with
existing CI runs); ~10-12 new tasks ship under `cs_*.json`
covering broader policy and customer-interaction surfaces.

**Coverage targets per the difficulty + judge mix in §D and §E:**

- ≥1 additional hard task complementing `pilot_005`'s "missing
  information" trigger — pick a different hard-task trigger
  category (contradictory premises, genuinely under-determined,
  or expertise-out-of-scope) so the refusal-calibration signal
  doesn't collapse to a single failure mode.
- ~50/50 mix of exact_match and rubric judges (customer_support
  skews exact because policy answers are concrete: "30-day",
  "yes", "$40"; rubric covers the multi-sentence policy
  applications).

Why extend rather than replace: the existing pilot's content has
been operator-reviewed (week 1) and validated against three
frontier models (weeks 1-3). The cleanup-vs-extend tradeoff
favored extend because reviewing 5 known-good tasks to vouch
they're still good is cheaper than re-authoring them from
scratch.

## C. code_repair contract — rubric-judged diff, no sandbox

**Decision:** Each code_repair task ships a triple:

1. **`prompt`** containing the broken code snippet plus a
   plain-English description of intended behavior;
2. **`ground_truth`** of kind `"rubric"` carrying an explicit
   description of what a correct fix must do (and what
   technically-passing-but-wrong fixes should be flagged for);
3. **Optional `metadata.expected_fix_sketch`** (str) — a
   reference fix the operator wrote during authoring, kept only
   for human review; no metric-time role.

Agent emits its fix as the answer body; the existing
`RubricJudge` adjudicates against the rubric. **No executable
sandbox in v0.1.**

Three alternatives were considered:

1. **Sandboxed test execution** (SWE-bench-Live / SWE-Sandbox /
   Princeton's SWE harness style). Highest-fidelity measurement
   of code correctness. Rejected for v0.1 by the same reasoning
   ADR-0007 §B used to reject tool-execution safety traps:
   building a sandboxed test fixture environment is a multi-week
   task on its own; the resulting metric measures *fixture
   quality* as much as model code-repair ability; existing
   benchmarks (SWE-bench, SWE-bench-Live, Princeton's harness)
   already own this space. Per CLAUDE.md "build on, don't
   reinvent": if we want sandboxed code_repair in v0.2 we should
   wrap an existing harness, not build our own.
2. **Pattern-match exact_match on key fix tokens.** Lowest-
   fidelity; pattern-fragile (`x += 1` vs `x = x + 1` vs
   `x = 1 + x` all correct but exact_match catches only one);
   rejected.
3. **Rubric-judged diff (chosen).** Reuses the existing
   `RubricJudge` infrastructure with the same prompt-only
   threat-model rationale as the safety dimension. v0.2 path:
   wrap SWE-bench-Live's harness.

**Cost.** v0.1 code_repair scores measure *what the model
emitted*, not *whether the emitted code passes tests*. The
correlation between the two is the v0.2 validation target.
METHODOLOGY §"Known limitations" gains a bullet: *Steadfast
code_repair v0.1 measures rubric-judged code-fix quality, not
executed-test pass-rate; sandboxed-execution evaluation is a
v0.2 path via a wrapped SWE-bench harness.*

**Rubric construction guidance** (informs the operator-audit
gate per §F): each rubric must specify (a) the minimal correct
behavior change, (b) at least one common-but-wrong fix pattern
to score 0, (c) any partial-credit conditions (matching the
existing `pilot_003` style on customer_support). The rubric is
the metric's source of truth; vagueness inflates judge
variance.

## D. multi_hop_research contract — self-contained synthesized prompts

**Decision:** Each multi_hop_research task embeds 2-4 evidence
snippets in the `prompt` and asks for a conclusion that
requires combining them across at least two reasoning hops.
**No live web/search tool fixture in v0.1.** Tests *multi-hop
reasoning over given evidence*, not *retrieval*. The "research"
in the domain name applies to the multi-step inference
discipline, not the source-finding step.

Three alternatives were considered:

1. **Live tool calls** (Search tool with deterministic fixture
   set). Highest fidelity; requires a per-task search index;
   lots of moving parts; couples the metric to fixture
   reliability. Out of scope for v0.1 alongside the OpenAI
   Agents SDK adapter — the right v0.2 path is to layer in a
   Search tool once the adapter scaffold is robust.
2. **Self-contained synthesized (chosen).** Reuses the
   existing rubric judge; no new infrastructure surface; the
   one-step prompt shape matches every other Steadfast task.
3. **Hybrid:** ~half live-tool, ~half synthesized. Adds
   branching without clearly improving the signal at the v0.1
   bank size.

**Mining angle.** This is the same prompt-only philosophy
ADR-0007 §B applied to safety. The v0.1 commitment is "measure
reliability of the response"; isolating reasoning from
retrieval is the cleaner v0.1 measurement.

**Implementation contract:**

- Prompts embed evidence as numbered bullets (`Fact 1: ...
  Fact 2: ...`) so the multi-hop nature is unambiguous to a
  reader.
- At least one task per domain bank should require a hop the
  model has to *combine* (not just retrieve and emit) — e.g.,
  "given X happened in 2023 and Y costs Z per unit, compute
  total cost over the period."
- Hard tasks (the §D minimum of ~10%) should be cases where
  the evidence is genuinely *insufficient* to reach a single
  answer; refusal/hedging is the correct behavior. Mirrors
  `pilot_005`'s pattern at the multi-hop level.

## E. Ground-truth distribution — ~40% exact_match / ~60% rubric

**Decision:** Across the full ~50-task benchmark, target
~40% exact_match / ~60% rubric. The split falls out naturally
from the task-content distribution rather than being enforced
domain-by-domain:

| Domain                 | Approx exact / rubric split |
|------------------------|------------------------------|
| customer_support       | ~60% exact / ~40% rubric    |
| code_repair            | ~10% exact / ~90% rubric    |
| multi_hop_research     | ~40% exact / ~60% rubric    |

**Rationale:**

- exact_match is faster and cheaper but only works on
  canonicalizable single-token / single-fact answers (after the
  week-2 hyphenation clarification fix; per ADR-0003 §B.3).
- rubric handles multi-sentence policy applications, code
  diffs, and reasoning-chain conclusions.
- Customer_support skews exact because policy answers are
  concrete; code_repair skews rubric because diffs require
  semantic judgment; multi_hop_research is mixed depending on
  whether the conclusion is a single fact or a derivation.

**Why not enforce a per-domain ratio:** the target ratio is
prescriptive at the bank level, descriptive at the domain
level. Forcing customer_support to a 40/60 split would mean
authoring rubric tasks where exact_match works fine (rubric
judge cost without rubric judge value); forcing code_repair
to a 40/60 split would mean inventing exact_match-judgeable
fix patterns where the natural shape is "diff-against-rubric."

## F. Operator-audit gate via per-domain `_review.json` manifest

**Decision:** Each benchmark domain ships a `_review.json`
manifest at the directory root recording which tasks the
operator has audited:

```json
{
  "version": "v1",
  "review_status": "partial",
  "reviewed_tasks": ["pilot_001", "pilot_002", ...],
  "draft_tasks": ["cs_006", "cs_007", ...],
  "notes": "..."
}
```

CLI's task loader (`resolve_benchmark` + a new internal helper)
refuses to surface a task whose ID isn't in `reviewed_tasks`.
Same fail-loud philosophy as `load_safety_bank()` per ADR-0007
§G and `load_distractor_bank()` per ADR-0006 §C, but scoped to a
**directory** rather than a single file so the audit can happen
in batches.

Three alternatives were considered:

1. **Per-task `review_status: Literal["draft", "reviewed"]`
   field inside each task's JSON.** Inline with the task
   content; consistent with the safety bank's per-case kind
   pattern. Rejected: every task JSON gains a new required
   field; the v0.1 pilot tasks need a migration script; the
   audit becomes per-file rather than per-batch which is the
   wrong granularity for what operators actually do (review a
   batch of new tasks against each other for coverage and
   coherence).
2. **Per-batch `_review.json` manifest (chosen).** Audit
   tracking lives separately from task content; existing
   pilot tasks need no migration (they auto-promote to
   `reviewed` by being listed in the manifest); new domains
   start with everything `draft` until audited.
3. **No gate; trust the commit history.** What the
   customer_support pilot used. Acceptable at N=5; untenable at
   N=50+ where the operator can't remember which tasks they
   personally vouched for.

**Manifest semantics:**

- `version`: schema version, matches the v0.1.x convention.
- `review_status`: one of `"draft"` (no tasks audited),
  `"partial"` (some tasks audited), `"complete"` (every task
  in the directory is in `reviewed_tasks`). Computed
  consistency check at load time.
- `reviewed_tasks`: list of task IDs (not paths) that the
  operator has audited. Loader compares against the loaded
  `Task.id` field rather than the filename so renames don't
  invalidate an audit.
- `draft_tasks`: list of task IDs the operator authored but
  hasn't yet audited. Both lists together should equal the
  union of task IDs in the directory; load-time consistency
  check fails loud if a task file is in neither list (that's
  the "did the operator forget to update the manifest"
  failure mode).
- `notes`: free-text audit-trail field. Operator-only;
  metric-time ignored.

**Audit checklist for each new task** (parallel to ADR-0007
§G's safety-bank checklist):

1. **Schema validity** — the JSON loads as a `Task` without
   raising; the `judge` field's referenced judge is
   resolvable; `ground_truth` matches the judge's `kind`
   contract.
2. **Domain coherence** — the task fits the domain's scope
   per §B/§C/§D; cross-domain leakage (e.g., a code_repair
   task that's actually a customer_support refund question)
   should be moved to the right domain.
3. **Ground-truth specificity** — exact_match tasks have a
   canonical answer that the canonicalizer can reach; rubric
   tasks have a rubric specific enough that the
   `RubricJudge`'s 0-4 Likert scale lands on the same score
   the operator would.
4. **Difficulty labeling** — tasks marked `"hard"` either have
   no determinable ground truth or are genuinely under-
   specified; tasks marked `"normal"` have a clear correct
   answer (per ADR-0005 §E).

## G. Path to v0.2

**v0.2 events** queued from this ADR:

- **`code_repair_v2`: sandboxed execution.** Wrap SWE-bench-Live
  or SWE-Sandbox; metric becomes test-pass-rate-on-fix-diff
  rather than rubric-on-diff. Triggers a leaderboard re-run per
  METHODOLOGY §"Versioning".
- **`multi_hop_research_v2`: live tool calls.** Layer in a
  deterministic Search tool fixture (mirrors the safety
  dimension's tool-execution `safety_v2` path); makes the
  multi-hop measurement actually depend on retrieval quality
  rather than reasoning-only. Triggers re-run.
- **Bank growth toward 100 tasks (~30/domain).** If v0.1
  leaderboard CIs prove too wide on cross-model claims for a
  given dimension, doubling the bank tightens the CI. Same
  ADR-0007 §H bank-growth path applied to the cross-domain
  bank.
- **Per-domain refusal-calibration breakdown.** If the v0.1
  pilot shows that refusal calibration is materially
  domain-conditional (e.g., a model refuses appropriately on
  customer_support hard tasks but confabulates on
  multi_hop_research hard tasks), the leaderboard could
  surface a per-domain refusal-sensitivity column. v0.2
  presentation upgrade, not a new metric.
- **Per-task field migration** of the audit gate (per the §F
  alternative 1). If the directory-level manifest becomes
  unwieldy or operator workflows favor inline annotation, a
  one-time migration script promotes the gate to a per-task
  field.

All four are v0.2 events that either change a metric's
computation (the first two — new metric names) or change the
benchmark surface (the second two — bank composition or audit
mechanism). Per METHODOLOGY §"Versioning", the first two trigger
a leaderboard re-run obligation; the second two are
presentation/operational changes that ride into v0.2 without a
re-run.

## References

- METHODOLOGY §1-§4 (per-dimension contracts — unaffected by
  this ADR; benchmark expansion changes *what* gets measured,
  not *how*).
- METHODOLOGY §3.4 (refusal-calibration ≥10% hard rule — drives
  §A's difficulty distribution).
- METHODOLOGY §"Statistical conventions" (paired bootstrap on
  robustness deltas — §A's task-count target is calibrated to
  produce interpretable CIs at this bootstrap).
- METHODOLOGY §"Versioning" (clarification-fix vs metric-version
  rules — §G's v0.2 paths follow the metric-version branch).
- ADR-0001 §"Path to v0.2" (ensemble-judge upgrade — applies
  to code_repair's rubric judge if leaderboard variance proves
  judge-dominated).
- ADR-0003 §B.3 (exact_match canonicalization contract — drives
  §E's exact_match shape).
- ADR-0006 §C (distractor-bank `review_status` pattern — §F's
  audit-gate precedent).
- ADR-0007 §B (prompt-only threat model — §C's rubric-on-diff
  rationale is the exact analogue).
- ADR-0007 §G (safety-bank operator-audit gate — §F mirrors the
  pattern at the directory level).
- SWE-bench / SWE-bench-Live (sandboxed code-repair benchmarks
  Steadfast deliberately does NOT compete with in v0.1; §C's
  v0.2 path wraps one of them).
- `notes/week3_findings.md` §"v0.2 implications surfaced by
  this run" (the bank-size-10 → bank-size-50 evidence for §A's
  task-count target).
