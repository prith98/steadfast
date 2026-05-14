# Multi-Hop Research Tasks

Curated multi-hop reasoning benchmark tasks for Steadfast. Per
ADR-0008 §D, v0.1 ships **self-contained synthesized prompts**:
each task embeds 2-4 evidence snippets and asks for a conclusion
that requires combining them across at least two reasoning hops.
**No live web/search tool fixture in v0.1.** Tests multi-hop
reasoning over given evidence, not retrieval. Layering in a Search
tool fixture is a v0.2 path alongside the OpenAI Agents SDK
adapter.

## Versioning

Tasks are immutable once published; corrections create a `_v2`
suffix. The task-set version (committed in this directory) lands
in the run manifest so reproductions can pin task content.

## Provenance

All tasks below are hand-authored 2026-05-14 for the week-4
benchmark expansion (ADR-0008). The names, dates, and figures
in each task are fictional and constructed specifically for the
benchmark — using fictional content avoids the benchmark-
contamination concern from METHODOLOGY §"Known limitations"
(frontier models may have seen public-source content during
training).

| Task    | Difficulty | Hops | Judge       | Trigger                  | Topic                              |
| ------- | ---------- | ---- | ----------- | ------------------------ | ---------------------------------- |
| mhr_001 | normal     | 2    | exact_match | —                        | Library founding-date arithmetic   |
| mhr_002 | normal     | 3    | exact_match | —                        | Engineering-team size growth       |
| mhr_003 | normal     | 2    | exact_match | —                        | Satellite-launch temporal chain    |
| mhr_004 | normal     | 3    | exact_match | —                        | Clinical-trial birth-year          |
| mhr_005 | normal     | 3    | exact_match | —                        | Grant allocation + division        |
| mhr_006 | normal     | 3    | exact_match | —                        | Forklift capacity with safety margin |
| mhr_007 | **hard**   | 2    | rubric      | missing_information      | Lecture-count under-specified      |
| mhr_008 | normal     | 4    | rubric      | —                        | Scholarship eligibility logic      |
| mhr_009 | normal     | 3    | exact_match | —                        | Flight + timezone arithmetic       |
| mhr_010 | **hard**   | 3    | rubric      | contradictory_premises   | Contradictory historical sources   |

## Schema

Each task is a JSON object conforming to `steadfast.agent.Task`. The
`metadata.n_hops` field documents the minimum reasoning chain length
the task requires; it's a curation aid, not a metric input. Tasks
prefix each evidence snippet with a `Fact N:` label so the multi-hop
nature is unambiguous to a reader and the judge can reason about
which facts the agent combined.

## Difficulty distribution

mhr_007 and mhr_010 carry `difficulty: "hard"` (2 of 10 = 20%); the
rest are `"normal"`. The two hard tasks trigger refusal-calibration
behavior via different mechanisms (missing information,
contradictory premises) per METHODOLOGY §3.4's diverse-trigger
requirement; this is symmetric with code_repair's hard-task pair.

## Judge mix

6 of 10 are `exact_match` (single integers / dates / dollar amounts
that survive canonicalization); 4 of 10 are `rubric` (logic
explanations, hedging on hard tasks). The exact_match skew on this
domain reflects that derivations from multi-hop evidence often
produce a single specific numeric or temporal answer; rubric is
reserved for tasks where reasoning quality is the metric, not the
final scalar.

## Audit gate

Per ADR-0008 §F, the directory ships `_review.json` listing which
tasks the operator has audited. The CLI's loader refuses to surface
draft tasks. The first commit of this domain ships every task as
`draft`; the operator flips them after walking the §F audit
checklist (schema validity, domain coherence, ground-truth
specificity, difficulty labeling) and verifying the embedded
arithmetic on each exact_match task.
