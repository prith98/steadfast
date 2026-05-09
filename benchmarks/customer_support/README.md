# Customer Support Tasks

Curated customer-support benchmark tasks for Steadfast.

## Versioning

Tasks are immutable once published; corrections create a `_v2` suffix. The
task-set version (committed in this directory) lands in the run manifest so
reproductions can pin task content.

## Provenance

| Task         | Difficulty | Judge        | Origin / verification                                                                                                                                      |
| ------------ | ---------- | ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pilot_001`  | normal     | exact_match  | Hand-authored 2026-05-08 for the Tuesday harness pilot. Trivial by design — exercises the bench pipeline end to end. Verified by author (Prith).            |
| `pilot_002`  | normal     | rubric       | Hand-authored 2026-05-09 for the Friday calibration pilot. Tier-table lookup (2 lb in the 1.01-3 lb tier → $40). Verified by author against the rate table. |
| `pilot_003`  | normal     | rubric       | Hand-authored 2026-05-09. Conditional-policy application: order is 5 days late so the 7-day discount threshold does not fire. Verified by author.           |
| `pilot_004`  | normal     | exact_match  | Hand-authored 2026-05-09. Binary eligibility — 9 days old, 14-day window → "yes". Exact-match canonicalization handles "Yes." / "yes, eligible" answers.   |
| `pilot_005`  | **hard**   | rubric       | Hand-authored 2026-05-09. Hedge-appropriate: the prompt omits the information needed (store hours / location). Refusal or hedging is the correct behavior. |

The full v0.1 customer-support set (target: ~17 tasks) lands by end of week
2 per `docs/WEEK_1.md`. Friday's pilot run uses the five `pilot_*.json`
tasks here; the CLI's `--benchmark customer_support_pilot` resolves to
this glob (sorted lexicographically).

## Difficulty distribution

`pilot_005` carries `difficulty: "hard"` (1 of 5 = 20%); the rest are
`"normal"`. METHODOLOGY §3.4 calls for ≥10% hard tasks per benchmark
to drive refusal calibration; the pilot is intentionally over-weighted
toward hard so the small N=10 reps yield interpretable refusal-cell
counts.

## Schema

Each task is a JSON object conforming to `steadfast.agent.Task`. See
`docs/METHODOLOGY.md` and `src/steadfast/agent.py` for field semantics.
The `difficulty` field is the v0.1 typed first-class signal for refusal
calibration (per ADR-0005 §E); the `metadata.expected_difficulty` string
is informational only and does not drive any metric.
