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
- Runtime deps: `tenacity`, `aiosqlite`.

### Changed

- `docs/SPEC.md` and `README.md`: v0.1 ships **all 14** reliability sub-metrics
  (was "~10 of 14" / "10 sub-metrics").
- `docs/WEEK_1.md`: Tuesday DOD model identifier updated from
  `claude-opus-4.5` to the real `claude-opus-4-7` (per ADR-0002 §E).
