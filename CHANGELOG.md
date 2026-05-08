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
  **1.41.0** (latest stable as of April 2026; verified before Wednesday's
  tracing implementation).

### Changed

- `docs/SPEC.md` and `README.md`: v0.1 ships **all 14** reliability sub-metrics
  (was "~10 of 14" / "10 sub-metrics").
