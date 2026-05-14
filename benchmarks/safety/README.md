# `benchmarks/safety/` — safety dimension trap + benign case bank

This directory hosts the curated case bank for the Steadfast safety
dimension (METHODOLOGY §4, ADR-0007). Each case is one prompt the
benchmark feeds to the target agent; trap cases test catastrophic-rate
resistance (§4.1) and benign cases populate the should-not-refuse half
of refusal F1 (§4.2).

## Files

- **`cases_v1.json`** — the bank file. Contains 10 trap cases + 10
  benign cases in a single :class:`~steadfast.metrics.safety.SafetyBank`.
  The trap cases span the 4-category v0.1 harm taxonomy from ADR-0007
  §C; the benign cases mirror the trap surface domains so the
  refusal-F1 confusion matrix isn't dominated by category-domain
  mismatch.
- **This README.** Per-bank documentation: audit checklist, audit-gate
  flip command, and per-case notes (the latter inlined into each
  case's `notes` field in `cases_v1.json` for proximity).

## The audit gate (read this before running anything)

Per ADR-0007 §G, `cases_v1.json` ships with `review_status: "draft"`.
`load_safety_bank()` refuses to load anything other than `"reviewed"`:
any `--benchmark safety` invocation against the draft bank fails loud
with a `ValueError` explaining the gate. This is intentional — the
distractor bank's analogous gate was load-bearing in the 2026-05-10
pilot (`notes/week2_findings.md` §"Distractor bank operator-audit gate
is load-bearing").

### Audit checklist (ADR-0007 §G)

Before flipping `review_status` to `"reviewed"`, walk every case and
confirm:

1. **Trap-case clarity.** The trap's `expected_harmful_artifact`
   describes a concrete textual artifact that a credulous model would
   produce in a usable form. Borderline / interpretive cases inflate
   judge variance; the bank's purpose is to span the failure modes
   unambiguously.
2. **Benign-case legitimacy.** Each benign prompt is one a reasonable
   user with the stated role would actually ask. Refusing it would be
   over-refusal, not principled caution. The benigns are the
   over-refusal test — if a model refuses these, refusal F1 drops
   even though catastrophic rate stays at 0.
3. **Category coverage.** Each of the four harm categories
   (`data_destruction`, `illegal_advice`, `pii_exfiltration`,
   `social_engineering`) has at least 2 trap cases. Benign cases
   span the same surface domains (shell/SQL/Python; tax/billing;
   employee data; IT/medical/sales comms) so the F1 confusion matrix
   isn't dominated by category-domain mismatch.

### Flipping the gate

After the audit, edit `cases_v1.json` and change:

```diff
- "review_status": "draft",
+ "review_status": "reviewed",
```

Commit the change with a message like:

```
chore(benchmarks/safety): mark cases_v1 reviewed after operator audit
```

## Running the safety benchmark

After the audit gate is flipped, run the safety dimension end-to-end
across three frontier models with:

```bash
set -a && source .env && set +a && uv run steadfast bench \
  --benchmark safety \
  --models claude-opus-4-7,gpt-5.2,gemini-2.5-pro \
  --metrics safety \
  --output results/safety_pilot_001 \
  --exporter none
```

The `--exporter none` is recommended for cost-mode pilots so OTel
overhead doesn't add to the per-call latency. Switch to `console` or
`otlp` when iterating on the metric itself.

Outputs land under `results/safety_pilot_001/<model_slug>/safety.json`
and the cross-model HTML report at
`results/safety_pilot_001/report.html`.

## v0.1 scope reminders (don't expand without an ADR)

- **Prompt-only threat model.** The agent isn't connected to an
  executable tool sandbox; the harmful artifact is the text itself
  (ADR-0007 §B). Tool-execution traps are a `safety_v2` path.
- **10 cases is small.** The Wilson 95% CI on 0/10 trap cases is
  [0.000, 0.278] — a perfect catastrophic rate carries a wide
  interval. v0.1 commits to surfacing the width honestly per
  METHODOLOGY §"Known limitations"; v0.2 grows the bank past 50.
- **4-category taxonomy is a spanning set.** Not exhaustive.
  AILuminate / HarmBench taxonomies are the v0.2 path (ADR-0007 §H).
- **Single judge, no ensemble.** Ensemble safety judging queues post-
  leaderboard per ADR-0007 §H, matching the ADR-0001 outcome-rubric
  ensemble path.
