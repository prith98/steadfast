# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Per `docs/METHODOLOGY.md` §"Versioning", any change to a metric's computation
requires a new metric name and an ADR — typo and clarification fixes are tracked
here.

## [Unreleased]

### Added

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
