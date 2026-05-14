# Code Repair Tasks

Curated code-repair benchmark tasks for Steadfast. Per ADR-0008 §C,
v0.1 measures **rubric-judged diff quality, not test-execution
pass-rate**. Each task ships a (broken code snippet, intended
behavior description, expected fix rubric) triple; the agent emits
a fix; `RubricJudge` adjudicates against the rubric. No executable
sandbox in v0.1 — SWE-bench-Live / SWE-Sandbox / Princeton's SWE
harness own that surface, and wrapping one of them is a v0.2 path.

## Versioning

Tasks are immutable once published; corrections create a `_v2`
suffix. The task-set version (committed in this directory) lands
in the run manifest so reproductions can pin task content.

## Provenance

All tasks below are hand-authored 2026-05-14 for the week-4
benchmark expansion (ADR-0008). Each is a Python bug crafted to
have an unambiguous correct fix or, for hard tasks, deliberately
under-specified or contradictory framing.

| Task   | Difficulty | Hard trigger             | Bug class                                |
| ------ | ---------- | ------------------------ | ---------------------------------------- |
| cr_001 | normal     | —                        | Off-by-one in range bound                |
| cr_002 | normal     | —                        | Even-length median (missing branch)      |
| cr_003 | normal     | —                        | Optional dict key (KeyError)             |
| cr_004 | normal     | —                        | Missing loop advance (infinite loop)     |
| cr_005 | normal     | —                        | SQL injection (parameterize)             |
| cr_006 | normal     | —                        | Symmetric date diff (sign error)         |
| cr_007 | normal     | —                        | Incorrect boolean operator (`or` vs `and`) |
| cr_008 | normal     | —                        | Missing return statement                 |
| cr_009 | **hard**   | under_specified          | Ambiguous bug report — hedge expected    |
| cr_010 | **hard**   | contradictory_premises   | Incorrect reviewer premise — push back   |

## Schema

Each task is a JSON object conforming to `steadfast.agent.Task`. See
`docs/METHODOLOGY.md` and `src/steadfast/agent.py` for field semantics.
The `metadata.expected_fix_sketch` field is informational only
(operator-curation aid); it does NOT drive any metric. Rubric judging
is governed entirely by `ground_truth.value`.

## Difficulty distribution

cr_009 and cr_010 carry `difficulty: "hard"` (2 of 10 = 20%); the
rest are `"normal"`. METHODOLOGY §3.4 calls for ≥10% hard tasks per
benchmark; the two hard tasks here trigger refusal-calibration
behavior via different mechanisms (under-specified bug report,
contradictory reviewer premise) so the refusal-calibration signal
isn't dominated by a single trigger pattern.

## Audit gate

Per ADR-0008 §F, the directory ships `_review.json` listing which
tasks the operator has audited. The CLI's loader refuses to surface
draft tasks. The first commit of this domain ships every task as
`draft`; the operator flips them after walking the §F audit
checklist (schema validity, domain coherence, ground-truth
specificity, difficulty labeling).
