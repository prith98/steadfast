# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Per `docs/METHODOLOGY.md` §"Versioning", any change to a metric's computation
requires a new metric name and an ADR — typo and clarification fixes are tracked
here.

## [Unreleased]

### Changed

- **GitHub Actions bumped to Node-24-compatible majors**:
  `actions/checkout@v4 → v6` and `astral-sh/setup-uv@v3 → v8` in
  `.github/workflows/ci.yml`. Pre-emptive of GitHub forcing Node 24 by
  default on 2026-06-02 (Node 20 removed from runners 2026-09-16); the
  v4 / v3 majors used Node 20.

### Added

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
