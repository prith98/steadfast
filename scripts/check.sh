#!/usr/bin/env bash
# Mirror of `.github/workflows/ci.yml`'s lint-type-test job.
#
# Run this before `git push` so CI-only failures (e.g., a `ruff format`
# diff that local `ruff check` misses) get caught on the local machine
# instead of in CI. Exits non-zero on the first failing step, matching
# CI's behavior under `set -e`.
#
# Usage: bash scripts/check.sh
set -euo pipefail

echo "==> ruff format --check"
uv run ruff format --check

echo "==> ruff check"
uv run ruff check

echo "==> mypy (strict via pyproject)"
uv run mypy src/

echo "==> pytest (skip slow + live)"
uv run pytest -m "not slow and not live"

echo "==> all CI checks passed"
