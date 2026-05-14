# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Per `docs/METHODOLOGY.md` §"Versioning", any change to a metric's computation
requires a new metric name and an ADR — typo and clarification fixes are tracked
here.

## [Unreleased]

### Added

- **Benchmark expansion to 51 tasks across 3 domains** per ADR-0008.
  Three domains at 17 tasks each (= ADR-0008 §A target):
  customer_support extended from 5 to 17 tasks (5 pilots +
  `cs_001.json` through `cs_012.json`); new `benchmarks/code_repair/`
  directory with 17 tasks (`cr_001.json` through `cr_017.json`)
  under the rubric-judged-diff contract from ADR-0008 §C (no
  executable sandbox in v0.1, mirroring ADR-0007 §B's prompt-only
  threat-model rationale); new `benchmarks/multi_hop_research/`
  directory with 17 tasks (`mhr_001.json` through `mhr_017.json`)
  under the self-contained-synthesized-prompts contract from
  ADR-0008 §D (no live web/search tool fixture in v0.1). Per-domain
  `_review.json` audit manifests per ADR-0008 §F gate the bank;
  end state is 32 reviewed + 19 drafted across the three manifests
  pending operator second-pass audit. Hard-task density is 2/17 =
  11.8% per domain (above METHODOLOGY §3.4's 10% floor in each),
  with diverse hard-task triggers across the 6 hard tasks
  (`missing_information` on pilot_005 + cs_005 + mhr_007;
  `contradictory_premises` / `under_specified` on cr_009 + cr_010 +
  mhr_010).
- **Per-domain audit-gate enforcement in `resolve_benchmark`** per
  ADR-0008 §F. Bare-domain slugs (`customer_support`, `code_repair`,
  `multi_hop_research`) now apply the `_review.json` manifest filter;
  the `_pilot` slug bypasses the gate for back-compat with
  weeks-1-3 invocations. Drafted tasks are filtered with a stderr
  warning; an all-drafted directory fails loud per the existing
  `load_safety_bank` / `load_distractor_bank` precedents
  (ADR-0007 §G / ADR-0006 §C). New `BenchmarkAuditManifest` Pydantic
  model exported from `steadfast.cli`. Glob exclusions for
  manifest/metadata convention (`_*.json`), draft bank files
  (`*.draft.json`), and frozen banks (`distractors_v*.json`,
  `cases_v*.json`) so they don't leak into the task surface.
- **METHODOLOGY §"Benchmark composition" subsection** linking
  ADR-0008's per-domain contracts, judge mix targets, difficulty
  distribution, and operator-audit gate. Plus two new
  §"Known limitations" bullets covering code_repair's
  rubric-judged-not-test-executed scope and multi_hop_research's
  reasoning-only-not-retrieval-grounded scope.
- **v0.1.x clarification fix: cost aggregation in safety summary**
  (week-3 pilot v0.1.x backlog item 1 per `notes/tradeoffs_log.md`
  P4 / ADR-0008 §G item 1). `SafetyCaseResult` carries a new
  `cost_usd: Decimal | None` field; `SafetyDimension.total_cost_usd`
  property aggregates across cases; `_summarize_safety_run` prints
  the total. METHODOLOGY §"Versioning" clarification-fix exemption
  applies; no metric semantic change.
- **v0.1.x clarification fix: per-category catastrophic-rate
  breakdown in HTML report** (week-3 pilot v0.1.x backlog item 2).
  `_render_safety_section` now appends a "Catastrophic rate by harm
  category" sub-table with Wilson 95% CI per (model, category) cell.
  Surfaces the week-3 pilot's "data_destruction dominates failures"
  finding from `notes/week3_findings.md` §"Finding 1" that's
  currently buried in the per-case JSON. Pure presentation; data
  already on `SafetyDimension.per_case`.
- **Robustness dimension — contradiction handling sub-metric**
  (`docs/METHODOLOGY.md` §2.3 / ADR-0006 §D). Third robustness sub-metric;
  3-way categorical (`{detected, retried_or_escalated, hallucinated}`)
  with per-cell Wilson 95% CIs (the three CIs are not jointly bounded —
  documented in `ContradictionResult.notes`). New
  `src/steadfast/perturbations/contradiction.py` exposes the per-call
  primitives a tool-using agent wires into its own tool loop:
  `should_corrupt(task_id, tool_call_idx, probability=0.3)` (Bernoulli
  coin seeded by ADR-0006 §B's `:tool{idx}` extension),
  `corrupt_tool_result(...)` (programmatic strategy dispatch:
  `negate_number`, `flip_boolean`, `replace_with_plausible`,
  `swap_entities`), and the `load_corruption_strategies` /
  `load_detection_phrases` loaders for the two new frozen prompt files.
  Strategy registry is fail-loud against unknown names per ADR-0006 §C
  precedent. **Strategies are programmatic, not LLM-driven** —
  ADR-0006 §D's "no fourth infrastructure-LLM judge" rationale extended
  symmetrically to the corruption side; v0.2 may introduce an
  `LLMCorruptor`.
- **Rule-based contradiction classifier**
  (`src/steadfast/metrics/robustness.py::classify_contradiction_response`).
  Three decision rules in ADR-0006 §D priority order: `detected` if
  `response.refused` or any phrase from
  `prompts/contradiction_detection_phrases_v1.txt` appears in the
  (lowercased) answer; `retried_or_escalated` if the trajectory shows a
  same-`(name, args)` tool call after a corrupted call OR an escalation
  phrase appears; `hallucinated` otherwise. The retry rule's "after"
  semantics use the earliest-corruption index — duplicate calls before
  any corruption do NOT count as retries.
- **Contradiction signaling convention** — agents wire corrupted-call
  indices into `response.metadata["steadfast.contradiction.corrupted_call_indices"]`
  as a JSON-encoded `list[int]`, exported via
  `CORRUPTED_CALLS_METADATA_KEY` from
  `steadfast.perturbations.contradiction`. The metric's
  `_extract_corrupted_calls` helper parses and slices `trajectory`;
  malformed metadata silently degrades to an empty list (rep is still
  counted, but the retry rule simply can't fire).
- **`measure_contradiction_handling`**
  (`src/steadfast/metrics/robustness.py`). Returns
  `ContradictionResult` with the three marginal proportions, three
  Wilson 95% CIs, `n_reps_with_tools` denominator,
  `per_task: list[ContradictionTaskResult]` for diagnostic drill-down,
  and `value: Literal["measured"] | None` for the N/A surface.
  Toolless agents — no rep across any task had a non-empty trajectory —
  return `ContradictionResult(value=None, reason="agent did not call any
  tools")` per ADR-0004 §G. Reps with non-empty trajectory but zero
  corrupted calls (probabilistic — 0.3 per call) still count toward
  `n_reps_with_tools` and fall through to the `hallucinated` bucket.
- **`RobustnessDimension.sub_metrics` typing widened** to
  `dict[str, RobustnessSubMetricResult | ContradictionResult]`.
  Pydantic 2 smart-union discrimination via the narrower `kind` Literal
  on each member (`"typo" | "distractor"` vs `"contradiction"`).
  Round-trips cleanly through `model_dump_json` /
  `model_validate_json` — verified by
  `test_measure_robustness_dimension_roundtrips_through_json`.
- **CLI surface** — `--robustness-types contradiction` is now
  accepted; `_VALID_ROBUSTNESS_TYPES` derives from the metric layer's
  `SUPPORTED_KINDS` (no CLI edit needed). Customer-support pilot
  invocations land on the N/A path (`SimplePromptingAgent` is toolless);
  a real cross-model contradiction pilot waits on the multi-hop research
  domain (week 3 or 4) per WEEK_2 §Friday.
- **HTML report contradiction cell** (`reporting/html.py`). New
  `_render_contradiction_cell` — three labeled lines (`detect`, `retry`,
  `halluc`) with per-cell Wilson CIs, plus an `n=…` footer line. The
  N/A path renders the `reason` text in warn style. Section copy
  amended to document the marginal-CI semantics.
- **`prompts/contradiction_corruptions_v1.txt`** — frozen registry of
  the four programmatic corruption strategies (one strategy per
  non-comment line, `name: description` format). Loader gates names
  against the registry per ADR-0006 §C precedent.
- **`prompts/contradiction_detection_phrases_v1.txt`** — frozen
  detection / escalation phrase lists in `[detection]` and
  `[escalation]` sections. Phrases are lowercased on load (the
  classifier matches lowered answer text).
- **Tool-using agent fixture**
  (`tests/fixtures/contradiction_agents.py`). New
  `EchoToolAgent(behavior, corruption_probability)` — first
  tool-using agent in the codebase. Calls a single synthetic
  `lookup_policy` tool, wires the contradiction perturbation into its
  own tool loop, populates the metadata convention, and exhibits one of
  the three classifier-relevant behaviors per its `behavior`
  parameter. The fixture's `_invoke_tool` returns
  `tuple[str, bool]` so corruption is reported by `should_corrupt`'s
  authoritative flag rather than inferred from string equality
  against the ground truth.
- **Tests**: `tests/test_perturbations_contradiction.py` (35 tests —
  `should_corrupt` determinism / 0.3-convergence / probability bounds;
  per-strategy `corrupt_tool_result` behavior including zero-numeric
  fallthrough and word-boundary matching for `flip_boolean`; loader
  gates and section-header parsing). `tests/test_metrics_robustness.py`
  extended with 25 tests — 12 hand-computed classifier rule cases
  (priority order, refusal short-circuit, pre-corruption-duplicate
  rejection, structural args equality), 8 metric integration tests
  against the `EchoToolAgent` fixture (N/A on toolless agent;
  three-vector convergence per behavior at `p=1.0`; per-task
  diagnostics; default-phrases load; arun-failure tolerance), and 3
  `measure_robustness` bundling tests including JSON round-trip.
  `tests/test_reporting_html.py` extended with 3 contradiction render
  tests (3-bar marginal cell, N/A cell with reason, mixed-kind row).
  `tests/test_cli.py` updated — contradiction is now a valid
  `--robustness-types` value.
- **HTML report `write_text` now passes `encoding="utf-8"`** so
  reports with non-ASCII content (model names with accents, task
  prose) are byte-identical across platforms — Windows would
  otherwise default to the system code page.

- **Robustness dimension — typo + distractor sub-metrics**
  (`docs/METHODOLOGY.md` §2.1-§2.2 / ADR-0006). New
  `src/steadfast/metrics/robustness.py` exposing
  `RobustnessTaskResult`, `RobustnessSubMetricResult`, `RobustnessDimension`,
  `measure_typo_robustness`, `measure_distractor_robustness`, and the
  bundling `measure_robustness` wrapper. The clean arm reuses the
  per-task `RunResult` from the main bench loop (no re-spend); the
  perturbed arm runs N=10 distinct perturbations through `agent.arun`
  via `asyncio.gather` (mirrors `measure_output_consistency` for
  multi-input metric paths). Cross-task aggregation calls
  `paired_bootstrap_ci` from `stats/paired_bootstrap.py`. Single-task
  invocations (`--task pilot_001`) populate the point estimate but
  N/A the CI with a `reason` field per ADR-0004 §G.
- **Per-rep distinct perturbation seeds** (ADR-0006 §B). New
  `src/steadfast/perturbations/_seed.py::derive_seed(task_id, kind, *,
  rep_idx=None, tool_call_idx=None)` exposing the
  `sha256(f"{task_id}:{kind}:v1[:rep{idx}]")[:8]` derivation as a
  shared helper across the four robustness perturbations. Re-exported
  at `steadfast.perturbations.derive_seed`.
- **`perturbations/typo.py::perturb_typo`** — character-level
  substitution at the 5%-rate / 25%-per-word-cap defaults from
  METHODOLOGY §2.1. Substitution-only (no insert / delete / swap) so
  word lengths are preserved and the per-word cap is trivially correct.
  Cites Ribeiro et al. 2020 (CheckList) per METHODOLOGY §2.1.
- **`perturbations/distractor.py`** — `DistractorBank` /
  `DistractorSnippet` Pydantic models, `load_distractor_bank` /
  `write_distractor_bank` IO helpers, and the
  `pick_distractor` / `apply_distractor` / `perturb_distractor`
  surface. Token-count gating (200-800 token range per METHODOLOGY
  §2.2) walks the bank deterministically from `seed % len(bank)`;
  `DistractorBankExhaustedError` is raised loudly if no snippet fits.
  Frozen delimiter contract: `--- background reading --- … --- task ---`.
- **`scripts/generate_distractor_bank.py`** — one-shot operator
  script per ADR-0006 §C. Reads tasks in a domain, calls the
  generator (GPT-5.2, ADR-0001 lock) at temperature 0.7 with the
  frozen prompt at `prompts/distractor_bank_v1.txt`, tokenizes via
  `tiktoken cl100k_base`, dedupes, and writes
  `benchmarks/<domain>/distractors_v1.draft.json`. Manual review pass
  + `mv` to `distractors_v1.json` is the methodology gate before the
  metric picks up the bank. Cost ~$0.50 per domain.
- **`prompts/distractor_bank_v1.txt`** — frozen generator prompt.
  `{n}` / `{domain}` / `{tasks}` placeholders; demands JSON-only
  output; gives the model a corpus of in-domain tasks to avoid
  contradicting.
- **CLI surface** — `--metrics` accepts `robustness`; new
  `--robustness-types` Option (default: all supported kinds; v0.1 /
  Tuesday surface = `typo,distractor`). New `parse_robustness_types`
  helper. `_run_one_model` dispatches a per-model robustness pass
  after the task loop and writes `<model_dir>/robustness.json`.
  Distractor banks are loaded per-domain at CLI time; missing-bank
  domains warn-and-skip rather than aborting.
- **HTML report robustness section** (`reporting/html.py`). New
  `_render_robustness_section` renders a per-(model, kind) table with
  the cross-task delta + paired-bootstrap CI; clean / perturbed
  endpoint means appear in a subtle line for context. Single-task /
  N/A surfaces render the point estimate with N/A on the CI. Footer
  amended with an ADR-0006 reference.
- **Tests**: `tests/test_perturbations_typo.py` (12 tests — determinism,
  per-word cap floor semantics, length / boundary preservation, rate
  convergence, edge cases); `tests/test_perturbations_distractor.py`
  (12 tests — pick determinism, gating walk, exhaustion,
  delimiter contract, JSON round-trip); `tests/test_metrics_robustness.py`
  (15 tests — perturbed-arm fanout, delta = 0 / -1 / partial fixtures,
  n_tasks=1 N/A surface, cross-task aggregation with non-degenerate CI,
  distractor missing-bank skip, kind subsetting, unknown-kind raise).
  Plus a robustness section to `tests/test_reporting_html.py` and a
  `parse_robustness_types` block to `tests/test_cli.py`.
- **`tiktoken` direct dependency** — promoted from transitive (was
  pulled in via the `agentevals` chain) to declared. Justified per
  CLAUDE.md "dependency tree as quality signal": tiktoken is small
  (~1 MB), already in the OpenAI ecosystem we depend on, and required
  by the bank generator's token-count precomputation per ADR-0006 §C.
- **Customer-support distractor bank draft** at
  `benchmarks/customer_support/distractors_v1.draft.json` (38 snippets,
  token range 270-430, generator: gpt-5.2 at temperature 0.7). Carries
  `review_status: "draft"`; does **not** load via
  `load_distractor_bank` until an operator audits the snippets for
  ground-truth contradictions vs. the pilot tasks, edits the field to
  `"reviewed"`, and renames to `distractors_v1.json` per ADR-0006 §C.
  The metric's distractor sub-metric will skip-with-warning until that
  operator gate clears.
- **`stats/paired_bootstrap.py`** (`docs/adr/0006-robustness-and-paired-bootstrap.md`
  §F): paired-delta bootstrap CI primitive for robustness sub-metrics.
  Resamples per-task delta vector via the existing
  `stats/bootstrap.bootstrap_ci` (BCa, 10k resamples by default — same
  defaults as the canonical entry point per METHODOLOGY §"Statistical
  conventions"). Returns frozen `PairedBootstrapCI` with both arms'
  means, the delta point estimate, and the bootstrap CI on the delta.
  Edge cases mirror `stats/bootstrap.py` per ADR-0004 §H: mismatched
  arm lengths / empty input / `N < 2` raise `ValueError`; zero-variance
  delta vectors flag `degenerate=True` and collapse the CI to
  `(delta, delta)`.
- 11 hand-computed tests covering: identical arms (delta = 0,
  degenerate); worst-case (clean=1.0 / perturbed=0.0 → delta = -1.0,
  degenerate at -1.0); the ADR-0006 §F worked example
  (clean=[1.0,0.8,0.6], perturbed=[0.7,0.5,0.4] → delta ≈ -0.267 with
  realizable [-0.3, -0.2] CI bounds); seed reproducibility; frozen
  Pydantic contract; default-inheritance from `stats/bootstrap.py`;
  edge-case raises.

- **ADR-0006 — Robustness dimension methodology + paired bootstrap.**
  Bundles every Mon-Fri week-2 methodological choice into one ADR per
  the ADR-0002 §F precedent. Sections cover: closure of auto-memory Q2
  (LangGraph trajectory contract — §A); per-perturbation seed strategy
  (§B); distractor bank generation + freezing (§C); contradiction label
  schema + rule-based classifier (§D); long-context sigmoid fit + L_50
  (§E); paired bootstrap as the CI primitive for robustness deltas (§F);
  v0.2 backlog triage criterion (§G).

- **`docs/WEEK_2.md`** — daily Mon-Fri plan for the robustness
  dimension and the LangGraph adapter scaffold; modeled on
  `docs/WEEK_1.md`. Open methodological choices §O.1-§O.7 record the
  alternatives + rationale that ADR-0006 codifies.

### Fixed

- **`ExactMatchJudge.canonicalize` hyphenation insensitivity** (clarification
  fix per METHODOLOGY §"Versioning"; not a metric-version event because the
  original ADR-0003 §B.3 substring-containment intent was hyphenation-
  insensitive). New rule between casefold and whitespace-collapse: ASCII
  hyphens between word characters are replaced with a single space, so
  "30-day" canonicalizes to "30 day" and a ground truth phrased as
  "30-day" matches both hyphenated singular ("Our store offers a 30-day
  return window") and plural ("The return window is 30 days") agent
  answers. Surfaces the regression test using GPT-5.2's actual 2026-05-10
  `pilot_001` response text. Item 1 of the v0.2 backlog
  (`auto-memory: project_v02_backlog.md`) — hyphenation alone was
  insufficient; the pilot_001 ground-truth tightening to `"30-day"` is
  the second half of the fix and lands in a follow-up commit.
- **`benchmarks/customer_support/pilot_001.json` ground truth tightened**
  from `"30 days"` to `"30-day"`. Paired with the canonicalize hyphen
  fix above, the new GT canonicalizes to `"30 day"` and matches both
  hyphenated-singular ("Our store offers a 30-day return window for
  unopened items.") and plural ("The return window is 30 days.") agent
  phrasings via substring containment — both forms are semantically
  equivalent and both should pass. Pre-launch in-place benchmark edit;
  the README's "tasks immutable once published" rule applies to v0.1
  launch onward. Provenance row in `benchmarks/customer_support/README.md`
  updated with the date + reason.

### Changed

- **GitHub Actions bumped to Node-24-compatible majors**:
  `actions/checkout@v4 → v6` and `astral-sh/setup-uv@v3 → v7` in
  `.github/workflows/ci.yml`. Pre-emptive of GitHub forcing Node 24 by
  default on 2026-06-02 (Node 20 removed from runners 2026-09-16); the
  v4 / v3 majors used Node 20. Note: the original 2026-05-10 push of
  this bump used `setup-uv@v8`, which fails to resolve because
  `astral-sh/setup-uv` only publishes moving majors up to `v7` (v8 has
  point releases only). Corrected here to `@v7`, which already runs on
  Node 24 and satisfies the deadline.

### Added (week 1 calibration — earlier in the same Unreleased block)

- **Confidence-elicitation contract** (`docs/adr/0005-calibration-and-confidence.md`
  §B-C): frozen prompt suffix at `prompts/confidence_v1.txt` and
  `perturbations.confidence.parse_verbalized_confidence` parser. The agent
  emits a structured `ANSWER:` / `CONFIDENCE:` two-line tail; the parser
  accepts case-insensitive labels, multi-line answers, percent or decimal
  confidence forms, and a literal `REFUSE` token on the answer line. The
  parser uses last-occurrence semantics so prose mentions of either label
  earlier in the response don't shadow the trailing structured tail.
- **`SimplePromptingAgent` confidence integration**: when
  `Task.confidence_suffix` is set, the agent appends the suffix, requests
  logprobs from clients that support them (`logprobs=True` Steadfast-internal
  kwarg), parses the response, retries once on parse failure with a stricter
  reminder, and on second failure soft-fails (rep stays `COMPLETED`,
  `confidence=None`) so consistency / format / trajectory metrics still
  consume the rep (per ADR-0005 §C).
- **`AgentResponse.refused: bool`** and **`AgentResponse.logprob_avg: float | None`**
  — additive fields driving refusal calibration (METHODOLOGY §3.4) and the
  secondary logprob-derived calibration column (METHODOLOGY §3.1).
- **`Task.difficulty: Literal["normal", "hard"]`** — typed first-class
  field driving refusal calibration. Default `"normal"`; existing tasks
  pass through unchanged.
- **`ChatResponse.avg_logprob: float | None`** plus per-provider plumbing:
  `OpenAIClient` populates the field when `logprobs=True` is requested
  (averaging chosen-token logprobs from `choice.logprobs.content[i].logprob`);
  `AnthropicClient` and `GoogleClient` accept the kwarg and silently drop
  it (per ADR-0005 §A — Anthropic has no public per-token logprobs;
  Gemini support is deferred to v0.2 to avoid partial coverage confusion).
- **OTel logprob span attribute**: `record_chat_response` populates the
  reserved `STEADFAST_LOGPROB_AVG` (`steadfast.logprob_avg`) on the `chat`
  span when the provider supplied a value; `BaseModelClient.achat` wires
  the response field through.
- 38 new tests covering parser happy path / edge cases (multi-line,
  last-label-wins, refusal token, case-insensitive labels, percent and
  decimal forms, out-of-range rejection), agent retry-once-then-soft-fail
  logic, agent passes the `logprobs` kwarg through, and round-tripping of
  the new `AgentResponse` / `Task` fields.

- **Calibration dimension** (`docs/adr/0005-calibration-and-confidence.md`
  §D-E): four measurement functions in `metrics/calibration.py` —
  Brier (pooled-bootstrap squared errors with parallel verbalized +
  logprob columns), ECE (15 equal-mass bins per Nixon et al. 2019, with
  the `floor(N/3)` small-N fallback documented in METHODOLOGY §3.3),
  refusal calibration (2x2 confusion matrix on `(task.difficulty,
  response.refused)` with Wilson cell CIs and sensitivity / specificity
  scalars), and overconfidence rate (Wilson 95% CI on
  `count(incorrect ∧ confidence ≥ 0.9) / count(answered)`).
  All four return frozen Pydantic result models; `measure_calibration`
  bundles them into a single `CalibrationDimension` for HTML report
  consumption.
- **`stats/calibration.py`** math primitives: `brier_squared_errors`,
  `brier_score`, `equal_mass_bin_indices`, `expected_calibration_error`.
  Cited Brier (1950), Guo et al. (2017), Nixon et al. (2019). Hand-
  verified test values for both Brier (perfect / worst / uniform-half /
  three-point hand-computed) and ECE (perfect-calibration-by-construction,
  total-miscalibration, two-bin closed-form `2/15`).
- 43 new tests covering Brier perfect / worst-case / uniform-half /
  filter-refused-and-none, ECE default-15-bins / fallback / too-small /
  perfect-by-construction, refusal calibration perfect / worst / one-row-
  empty, overconfidence threshold-inclusivity / refused-excluded /
  invalid-threshold, end-to-end `measure_calibration` bundling, and
  `reps_from_run_results` filtering.

- **CLI multi-model + benchmark + metrics surface** (ADR-0005 §G):
  `steadfast bench --benchmark NAME` resolves a benchmark name to every
  matching `benchmarks/<domain>/pilot_*.json` (or `*.json` for bare
  domain names); `--models a,b,c` runs sequentially with per-model
  output subdirectories; `--metrics consistency,calibration` dispatches
  the corresponding measurement after each model's reps complete and
  are judged. The single-task / single-model surface (`--task --model`)
  remains for inner-loop development.
- **5-task customer-support pilot benchmark** under
  `benchmarks/customer_support/pilot_*.json` — `pilot_001` (existing
  trivial seed), `pilot_002` (tier-table lookup), `pilot_003`
  (conditional-policy application), `pilot_004` (binary eligibility),
  and `pilot_005` (hard / hedge-appropriate; `difficulty: "hard"`).
  Mixes exact-match and rubric judges. README documents provenance,
  judge type, and difficulty distribution.
- **HTML report** (`reporting.html.write_html_report`) — single-file
  self-contained HTML with inline CSS. Sections: per-model calibration
  table (Brier verbalized + logprob, ECE, refusal sens / spec,
  overconfidence rate; CIs in subdued styling), per-(model, task)
  consistency table, per-task pass-rate matrix, run header (benchmark
  name + models + date + package version), reproducibility footer
  (methodology version + ADR pointers). User-supplied strings are
  HTML-escaped; missing-data cells render as N/A rather than crashing.
- 19 new tests covering CLI helpers (`parse_metrics`, `parse_models`,
  `resolve_benchmark`, `_apply_confidence_suffix`), pilot-task
  difficulty distribution / ground-truth presence, and HTML report
  rendering (calibration / consistency / pass-rate sections,
  missing-files graceful handling, HTML escaping of user input).

- **OpenAI rate-limit-tier env knobs**: `STEADFAST_OPENAI_MAX_CONCURRENT`
  and `STEADFAST_OPENAI_MAX_RETRIES` override the `OpenAIClient`
  constructor defaults (5 / 5). The gpt-5.2 free tier is 3 RPM, which
  collides with the default 5-way concurrent fanout (paraphrase
  generator + validator + pairwise rubric + outcome judge); setting
  `MAX_CONCURRENT=1` and `MAX_RETRIES=10` makes the run finish on the
  free tier at the cost of wall-clock time. Documented in
  `.env.example`.
- **Empty-answer handling in `measure_output_consistency`**: real-world
  Gemini target runs hit safety filters on paraphrased inputs and
  return empty `response.text`; OpenAI's embedding endpoint rejects
  empty strings (HTTP 400) and crashed the entire consistency
  measurement. Empty answers are now substituted with a placeholder
  `"(no answer)"` before the embedding / rubric calls; the count
  rides on `OutputConsistencyResult.n_empty_answers` so the report
  surfaces the target-model issue. Known limitation for v0.2: when
  `n_empty == k` (all paraphrase responses empty), the metric currently
  reports spurious 1.0 agreement (all placeholders are identical) —
  fix is to return `None` with a `reason` field; deferred so v0.1
  schema stays compatible with the Friday pilot report.

### Resolved methodology questions

- **Q3 (Anthropic logprob)** — closed by ADR-0005 §A.
  `docs/METHODOLOGY.md` §3.1 commits to verbalized confidence as the
  leaderboard headline calibration column with a clearly-marked secondary
  logprob column carrying explicit `N/A` cells. v0.1 logprob coverage is
  OpenAI only; Anthropic and Google show `N/A`. Constant
  `STEADFAST_LOGPROB_AVG` reserved 2026-05-08 in ADR-0003 §A.4 is now
  populated.

### Earlier — Wednesday + Thursday + Tuesday + Monday

- Project skeleton: package layout, Typer CLI shell (`steadfast --help`,
  `steadfast bench --help`), CI workflow (ruff format check + ruff check +
  `mypy --strict` + pytest), Apache 2.0 license, smoke tests.
- `docs/adr/0001-infrastructure-model.md`: lock GPT-5.2 and
  `text-embedding-3-large` as benchmark infrastructure models for v0.1.
- `tracing/conventions.GENAI_CONVENTIONS_VERSION` pinned to OTel semconv
  **1.41.0** (latest stable as of April 2026).
- **v0.1 core abstractions** (`docs/adr/0002-v01-core-abstractions.md`):
  `Agent` ABC, `Task`/`AgentResponse`/`ToolCall`/`GroundTruth` Pydantic models,
  built-in `SimplePromptingAgent`.
- **Model clients**: `BaseModelClient` with tenacity-backed retry +
  per-instance asyncio semaphore; provider implementations for Anthropic
  (`claude-*`), OpenAI (`gpt-*`), and Google Gemini (`gemini-*`).
- **Pricing table**: `models/pricing.PRICING` with Anthropic, OpenAI, and
  Google entries snapshotted 2026-05-08, plus `provider_for_model` helper
  for CLI routing.
- **N-rep executor** (`runner.run_task`): async, deterministic `run_id`,
  automatic resumption of pending/in-flight reps, no auto-retry of failed
  reps. SQLite checkpointing via `aiosqlite`.
- **CLI wiring**: `steadfast bench --task --model --reps --output` runs the
  pilot end-to-end and writes a `RunResult` JSON.
- **Pilot task**: `benchmarks/customer_support/pilot_001.json` (trivial
  hand-authored task; exercises the harness end-to-end).
- **OTel GenAI tracing** (`docs/adr/0003-tracing-and-judges.md` §A):
  span hierarchy ``benchmark → task → rep → chat {model}`` plus per-rep
  ``score {kind}`` siblings. Required ``gen_ai.*`` attributes (operation
  name, provider name [dual-emitted as both ``gen_ai.provider.name`` and
  legacy ``gen_ai.system``], request/response model, usage tokens,
  finish reasons) plus Steadfast-namespaced ``steadfast.*`` extensions
  declared in ``tracing/conventions.py``. ``BaseModelClient.achat``
  emits exactly one ``chat`` span per public call; tenacity retries
  become ``span.add_event("retry", …)`` events rather than separate
  spans. Exporters: ``console`` (default), ``otlp`` (HTTP, defaulting
  to Phoenix at ``http://localhost:6006/v1/traces``), and ``none``.
- **Outcome judges** (ADR-0003 §B): ``Verdict`` Pydantic shape (frozen,
  ``score: float in [0,1]``, ``passed: bool``, ``reason: str``),
  ``Judge`` ABC with async ``ajudge``, ``ExactMatchJudge`` (NFKC +
  casefold + whitespace-collapse + trailing-punct strip + substring
  containment), ``RubricJudge`` (frozen prompt at
  ``prompts/rubric_v1.txt``, default model ``gpt-5.2`` per ADR-0001,
  Pydantic-validated JSON output, 1 retry on parse failure then raise).
- **Verdict on `RepRecord`**: completed reps carry their judge verdict
  in the result JSON (not persisted to SQLite — re-judging is cheap and
  re-runnable).
- **`steadfast bench --exporter`**: ``console`` | ``otlp`` | ``none``
  selects the OTel exporter; the run is wrapped in a ``benchmark`` span
  and judging is dispatched after ``run_task`` returns.
- **Consistency dimension** (``docs/adr/0004-consistency-and-stats.md``):
  three measurement functions in ``metrics/consistency.py`` — output
  consistency (K=5 paraphrases x pairwise embedding cosine + 0-4 Likert
  rubric normalized to [0,1]; mean rubric reported with bootstrap CI),
  trajectory consistency (pairwise normalized Wagner-Fischer Levenshtein
  over tool-name sequences, plus ``agentevals`` superset arg-equivalence
  rate), and format consistency (JSON-schema pass-rate with Wilson 95%
  CI). All three return frozen Pydantic result models.
- **Paraphrase generator** (``perturbations/paraphrase.py``): K=5 by
  default with frozen ``prompts/paraphrase_v1.txt``, validator second
  pass via ``prompts/paraphrase_validate_v1.txt``, up to 3 retries on
  rejected paraphrases, rejection rate tracked as a quality signal.
- **Statistical primitives** (``stats/{bootstrap,wilson}.py``): canonical
  entry points for confidence intervals. ``bootstrap_ci`` wraps
  ``scipy.stats.bootstrap`` with the methodology defaults (BCa, 10k
  resamples, 95% CI) and explicit handling of empty / N<2 / zero-variance
  edge cases. ``wilson_ci`` wraps ``scipy.stats.binomtest.proportion_ci``.
- **Embedding API** (``OpenAIClient.aembed``): batched call returning
  ``(vectors, usage, cost)``; emits an ``embeddings {model}`` span via the
  new ``embeddings_span`` / ``record_embeddings_response`` helpers in
  ``tracing/spans.py``. Default model ``text-embedding-3-large`` per
  ADR-0001; pricing entry added to ``models/pricing.PRICING``.
- **`Task.output_schema: str | None`** — JSON-schema string consumed by
  format consistency. Tasks without a schema return N/A.
- **`_llm_parsing.try_parse_strict`** — shared helper for parsing LLM
  JSON output as Pydantic models; ``RubricJudge`` was refactored to use
  it (removing its private ``_try_parse``).
- **Frozen prompts**: ``prompts/{paraphrase_v1.txt, paraphrase_validate_v1.txt,
  consistency_rubric_v1.txt}``.
- 57 new tests covering Levenshtein, cosine, Wilson, bootstrap edge
  cases, paraphrase happy/retry/exhaustion paths, format-consistency
  pass-rate, trajectory-consistency hand-computed similarity values,
  and end-to-end output consistency on stub clients.
- Runtime deps: ``agentevals``, ``jsonschema``.

### Resolved methodology questions
- **Q1 (K=5 vs N=10)** — ``docs/METHODOLOGY.md`` §1.1 now carries a
  one-sentence clarification (per ADR-0004 §A): paraphrases are
  *different* inputs each run once, distinct from the N=10 commitment
  about distributional measurement of a fixed input. Computation
  unchanged; classified as a "typo and clarification fix" per
  §"Versioning".

### Changed

- `docs/SPEC.md` and `README.md`: v0.1 ships **all 14** reliability sub-metrics
  (was "~10 of 14" / "10 sub-metrics").
- `docs/WEEK_1.md`: Tuesday DOD model identifier updated from
  `claude-opus-4.5` to the real `claude-opus-4-7` (per ADR-0002 §E).
- `pyproject.toml`: hatchling now force-includes the repo's `prompts/`
  directory under `steadfast/prompts` in the wheel so judge prompts
  resolve via `importlib.resources` for installed users.
- `BaseModelClient.achat`: now wraps the retry loop in a `chat {model}`
  span; the retry contract (ADR-0002 §B.1, §B.2) is unchanged.
- `BaseModelClient.PROVIDER_NAME`: required ClassVar override on each
  provider client (`anthropic`, `openai`, `gcp.gemini`).
