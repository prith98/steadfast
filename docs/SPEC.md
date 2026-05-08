# Steadfast — Project Specification

## What we're building

Steadfast is an open-source benchmark and evaluation harness that measures the **reliability** of AI agents — not their accuracy. It is a Python package + CLI + public leaderboard that lets anyone:

1. Wrap an existing agent (built on the OpenAI Agents SDK or LangGraph)
2. Run it against a curated benchmark of 50+ tasks across 3 domains
3. Get rigorous reliability scores along four dimensions: **consistency, robustness, calibration, safety**
4. Emit OpenTelemetry GenAI-compliant traces that flow into Langfuse, Phoenix, or Datadog

## Why this exists

Agent capabilities improved dramatically in 2025–2026, but **reliability lagged**. The Princeton paper _"Towards a Science of AI Agent Reliability"_ (Kapoor, Narayanan et al., 2026) showed that on agentic benchmarks, reliability improvements were **half** the rate of accuracy improvements; on customer-service benchmarks, **one-seventh**.

Production agents that succeed on 90% of tasks but fail unpredictably on 10% are useful assistants but unacceptable autonomous systems. The 95% × 20-step trajectory math is unforgiving: end-to-end success drops to 36%.

Today's eval ecosystem (DeepEval, Ragas, agentevals, MLflow, Phoenix) ships strong primitives for _outcome quality_ — faithfulness, answer relevance, tool-call correctness. None of them ship a reproducible, opinionated implementation of the **four-dimensional reliability framework** the field is converging on.

That's the gap Steadfast fills. We're not the platform. We're the ruler.

## The four dimensions

These follow the Princeton framework, broken into 14 sub-metrics.

### 1. Consistency

_Does the agent produce semantically equivalent outputs and trajectories when given semantically equivalent inputs?_

Sub-metrics:

- **Output consistency**: paraphrased queries → semantically equivalent answers (embedding similarity + LLM-judge rubric)
- **Trajectory consistency**: same task across N runs → similar tool-call sequences
- **Format consistency**: structured output adherence across runs

### 2. Robustness

_Does the agent maintain performance under realistic perturbations of input or environment?_

Sub-metrics:

- **Typo robustness**: success-rate delta with character-level noise
- **Distractor robustness**: success-rate delta with irrelevant context injected
- **Contradiction handling**: behavior when tool outputs contradict each other
- **Long-context degradation**: success-rate vs. context length

### 3. Calibration

_Does the agent's stated confidence correlate with actual correctness?_

Sub-metrics:

- **Brier score** on self-reported confidence
- **Expected Calibration Error (ECE)** with proper binning
- **Refusal calibration**: appropriate hedging on hard tasks
- **Overconfidence rate**: high-confidence wrong answers

### 4. Safety

_When the agent fails, how catastrophic are the failures?_

Sub-metrics:

- **Catastrophic failure rate** on trap cases (irreversible-action prompts)
- **Refusal accuracy** on prompts that should be refused
- **Escalation behavior** on out-of-scope tasks

## What "reliability" specifically means here

Reliability is **not the same as average accuracy**. We compute every metric across **N=10 runs** of each task, with proper statistics:

- Point estimates with bootstrapped 95% confidence intervals
- Variance reported alongside means
- Per-task and per-domain breakdowns, not just global averages

This is the central methodological commitment of Steadfast.

## Scope (v0.1)

In:

- 4 dimensions, ~10 of the 14 sub-metrics
- 50+ tasks across 3 domains: customer support, code repair, multi-hop research
- 2 framework adapters: OpenAI Agents SDK, LangGraph
- 5 model integrations: Claude Opus 4.5, GPT-5.2, Gemini 3 Pro, Llama-4, Mistral Large
- OpenTelemetry GenAI semantic convention compliance
- Public static leaderboard (Next.js)
- Published methodology writeup

Out (deferred to v0.2+):

- Real-time monitoring / streaming evals
- Custom dashboard / observability UI
- Multi-agent / agent-to-agent reliability
- Multimodal benchmarks
- More than 3 domains
- Fine-tuning or model-improvement loops

## Success criteria

A v0.1 ship is successful if:

1. A reader can clone the repo, run `steadfast bench --agent ./my_agent.py --reps 10` against any compliant agent, and get a reliability report.
2. The methodology page is rigorous enough that the Princeton authors would not object to it.
3. The leaderboard publishes results across at least 5 frontier models with confidence intervals.
4. The writeup is shareable on Hacker News without methodological embarrassment.

## Non-goals

- Becoming a full observability platform (use Langfuse/Phoenix/LangSmith)
- Replacing existing eval libraries (we build on them)
- Training or fine-tuning models
- "Solving" reliability — we measure it
- Closed/proprietary anything

## Prior art and influences

- [Princeton: Towards a Science of AI Agent Reliability (2026)](https://fortune.com/2026/03/24/ai-agents-are-getting-more-capable-but-reliability-is-lagging-narayanan-kapoor/)
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [LangChain agentevals](https://github.com/langchain-ai/agentevals)
- [Confident AI DeepEval](https://github.com/confident-ai/deepeval)
- [Pydantic Evals (LLM-as-judge patterns)](https://pydantic.dev/articles/llm-as-a-judge)
- [Arize Phoenix](https://www.digitalapplied.com/blog/agent-observability-platforms-langsmith-langfuse-arize-2026)
- τ-bench, AgentBench, GAIA (task design inspiration)

## License

Apache 2.0 (consistent with the OpenTelemetry / observability ecosystem we want to plug into).
