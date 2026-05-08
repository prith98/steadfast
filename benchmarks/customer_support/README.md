# Customer Support Tasks

Curated customer-support benchmark tasks for Steadfast.

## Versioning

Tasks are immutable once published; corrections create a `_v2` suffix. The
task-set version (committed in this directory) lands in the run manifest so
reproductions can pin task content.

## Provenance

| Task | Origin | Verified by |
| --- | --- | --- |
| `pilot_001` | Hand-authored 2026-05-08 for the Tuesday harness pilot. Trivial by design — exercises the bench pipeline end to end. | Author (Prith) |

The full v0.1 customer-support set (target: ~17 tasks) lands by end of week
2 per `docs/WEEK_1.md`. Friday's pilot run uses 5 tasks (`pilot_001` …
`pilot_005`) hand-authored with verified ground truth.

## Schema

Each task is a JSON object conforming to `steadfast.agent.Task`. See
`docs/METHODOLOGY.md` and `src/steadfast/agent.py` for field semantics.
