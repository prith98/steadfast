# ADR-0001: Lock GPT-5.2 and `text-embedding-3-large` as benchmark infrastructure models

- **Status:** Accepted
- **Date:** 2026-05-08
- **Methodology version:** v0.1
- **Supersedes:** none
- **Superseded by:** none

## Context

Several Steadfast metrics require LLM calls or embeddings as part of their
*infrastructure*, separate from the agent under test:

- **Output consistency** (`docs/METHODOLOGY.md` §1.1) requires K=5 paraphrases
  of each task input; paraphrases are validated for semantic equivalence by a
  second LLM call.
- **Output consistency** also computes pairwise semantic similarity using
  embedding cosine similarity and an LLM-judge rubric.
- **Rubric judges** (`docs/METHODOLOGY.md` §"Known limitations and threats to
  validity") are themselves LLMs, with their own miscalibrations.

For results across leaderboard entries to be comparable, the choice of
infrastructure model must be the same for every entry. A leaderboard run where
Claude-as-target has its outputs judged by Claude-as-judge, and GPT-as-target
has its outputs judged by GPT-as-judge, would conflate target performance with
judge agreement.

## Decision

For v0.1, the following infrastructure models are **locked**:

| Role | Model |
| --- | --- |
| Paraphrase generation | `gpt-5.2` |
| Paraphrase validation | `gpt-5.2` |
| Semantic similarity embedding | `text-embedding-3-large` |
| Default rubric judge | `gpt-5.2` |

Running the benchmark against any target model therefore requires a valid
**OpenAI API key** in `OPENAI_API_KEY`, in addition to whatever provider key
the target agent itself uses.

The frozen prompts for each role are version-suffixed under `prompts/`
(e.g., `prompts/paraphrase_v1.txt`).

## Consequences

**Positive**

- Leaderboard entries are comparable across target models.
- v0.1 reproducibility is well-defined: one infrastructure provider, one model
  version, frozen prompts.

**Negative**

- Hard dependency on OpenAI's API and the continued availability of the chosen
  models. If `gpt-5.2` is deprecated mid-cycle, the methodology must be
  re-pinned (new ADR) and the leaderboard re-run.
- Risk of **judge bias**: if `gpt-5.2` systematically rates GPT-family outputs
  more favorably than non-GPT-family outputs, the consistency and rubric-based
  metrics will be biased. We acknowledge this in `docs/METHODOLOGY.md`
  §"Known limitations"; v0.1 does not yet quantify the bias.

**Neutral**

- Cost: an additional ~K paraphrase + 1 validator + ~C(N, 2) similarity calls
  per task, all on `gpt-5.2`, plus embedding calls. Budgeted in
  `docs/WEEK_1.md` §"What can go wrong this week" #5.

## Alternatives considered

1. **No lock; benchmark uses the target model as its own judge.** Rejected:
   conflates target capability with judge agreement.
2. **3-judge ensemble (Claude / GPT / Gemini) with majority vote.** Strong
   methodologically — see Zheng et al. 2023 ([MT-Bench /
   Chatbot Arena](https://arxiv.org/abs/2306.05685)) on judge ensembling.
   Rejected for v0.1 on cost grounds; planned for v0.2 once we have a budget
   signal from production v0.1 runs.
3. **Configurable infrastructure model with a strong default.** Rejected for
   the leaderboard surface (would fragment the comparability story); may be
   re-introduced for *local* users running cheaper inner-loop experiments,
   tracked in `docs/ROADMAP.md`.

## Path to v0.2

- Run the v0.1 benchmark with a 3-judge ensemble on a representative
  subsample, measure the disagreement rate per dimension, and quantify the
  judge-bias risk.
- If the disagreement rate justifies the ~3× infrastructure spend, promote
  ensemble judging to the default in v0.2 with an ADR-0002.

## References

- `docs/METHODOLOGY.md` §1.1, §3.1, §"Known limitations and threats to
  validity"
- Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*
  (2023), https://arxiv.org/abs/2306.05685
- Pydantic Evals LLM-as-judge guide,
  https://pydantic.dev/articles/llm-as-a-judge
