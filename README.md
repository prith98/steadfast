# Steadfast

**Reliability benchmarking for AI agents.**

Most agent eval tools measure whether an agent gets the right answer once. Steadfast measures whether it gets the right answer _every_ time, hedges appropriately when it isn't sure, and degrades gracefully under realistic perturbation.

Steadfast implements the four-dimensional reliability framework from Princeton's _Towards a Science of AI Agent Reliability_ (Kapoor, Narayanan et al., 2026) as a reproducible Python package, a curated benchmark, and a public leaderboard.

> ⚠️ Status: **v0.1 in progress.** Expect breaking changes. See `docs/ROADMAP.md`.

## Why this exists

Agent accuracy improved 2x in 2025. Agent reliability did not. A 95% reliable step chained 20 deep is a 36% reliable system.

Existing eval libraries (DeepEval, Ragas, agentevals) ship excellent primitives for _outcome quality_. None ship an opinionated, reproducible implementation of the reliability framework that distinguishes "useful assistant" from "deployable autonomous system."

Steadfast does that, and only that.

## The four dimensions

| Dimension       | Question                         | Key sub-metrics                                            |
| --------------- | -------------------------------- | ---------------------------------------------------------- |
| **Consistency** | Same input → same output?        | Output similarity, trajectory similarity, format stability |
| **Robustness**  | Survives realistic perturbation? | Typo, distractor, contradiction, long-context              |
| **Calibration** | Knows what it knows?             | Brier score, ECE, refusal calibration                      |
| **Safety**      | Fails gracefully?                | Catastrophic failure rate, refusal accuracy                |

Every metric is reported with bootstrapped 95% confidence intervals across N=10 runs.

## Quickstart

```bash
pip install steadfast

# wrap your agent
cat > my_agent.py <<EOF
from steadfast import Agent

class MyAgent(Agent):
    def run(self, task):
        # your agent logic
        return {"answer": "...", "confidence": 0.8}
EOF

# run the benchmark
steadfast bench --agent my_agent.MyAgent --reps 10 --output results/

# open the report
open results/report.html
```

OpenTelemetry GenAI traces are emitted by default and can be ingested by [Langfuse](https://langfuse.com), [Phoenix](https://phoenix.arize.com), or [Datadog LLM Observability](https://www.datadoghq.com/product/llm-observability/).

## Leaderboard

Latest results across frontier models: [steadfast.dev/leaderboard](https://steadfast.dev/leaderboard) _(placeholder)_

## What's in v0.1

- ✅ 4 dimensions, 14 sub-metrics
- ✅ 50+ tasks across customer support, code repair, multi-hop research
- ✅ Adapters for OpenAI Agents SDK and LangGraph
- ✅ Reference results for Claude Opus 4.5, GPT-5.2, Gemini 3 Pro, Llama-4, Mistral Large
- ✅ OpenTelemetry GenAI semantic convention compliance
- ✅ Static leaderboard

## What's not in v0.1

- ❌ Real-time / streaming eval
- ❌ Custom UI / dashboard
- ❌ Multi-agent system evaluation
- ❌ Multimodal tasks
- ❌ Fine-tuning loops

See `docs/ROADMAP.md` for what's coming in v0.2.

## Methodology

Read [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full statistical methodology, task design rationale, and reproducibility instructions.

## Contributing

We welcome:

- New tasks (especially in underrepresented domains)
- New framework adapters
- New sub-metrics with rigorous justification
- Reproductions of leaderboard results

We don't welcome:

- Closed-source dependencies
- Metrics without statistical justification
- "I added GPT-X to the leaderboard" PRs without rerunning the full suite

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citation

If you use Steadfast in research or production, please cite the writeup _(link forthcoming after launch)_.

## License

Apache 2.0
