# Contributing to Steadfast

Thanks for the interest. Steadfast is in early development; expect breaking
changes through v0.1. See `docs/ROADMAP.md` for what's coming.

## Before you start

Please read these documents in order:

1. [`docs/SPEC.md`](docs/SPEC.md) — what we're building and what we're not.
2. [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — the canonical reference for
   any methodological question. If code disagrees with this document, the
   document is correct and the code is a bug.
3. [`CLAUDE.md`](CLAUDE.md) — operating principles, code style, and our working
   agreement.

If you're proposing a new metric, perturbation, or task domain, please open an
issue first to align on scope and methodology before writing code.

## Local development

```bash
uv sync
uv run pytest
uv run mypy src/
uv run ruff check
uv run ruff format --check
```

## What we welcome

- New tasks in underrepresented domains, with verified ground truth and
  documented provenance.
- New framework adapters (initial scope: OpenAI Agents SDK, LangGraph).
- New sub-metrics with rigorous statistical justification and a citation.
- Reproductions of published leaderboard results.

## What we don't welcome

- Closed-source dependencies.
- Metrics without confidence intervals or established statistical properties.
- "Add my model to the leaderboard" PRs that don't rerun the full suite for
  *all* listed models — see `docs/METHODOLOGY.md` §"Reproducibility is a
  first-class feature".

## Commit messages

Conventional Commits: `feat(metrics): add Brier score with bootstrap CI`,
`fix(runner): handle SQLite checkpoint race`, `docs(adr): add ADR-0002`.

## License

Apache 2.0. By contributing, you agree your contributions will be licensed
under the same terms.
