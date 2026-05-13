# Steadfast

**An end-to-end harness for stress-testing AI agents before they go to production.**

Steadfast runs your agent through a curated benchmark, captures every model call and tool invocation as OpenTelemetry GenAI traces, and produces a single HTML report grading how reliably it behaves under realistic conditions.

The stack:

- **Agent framework adapters** — LangGraph today, OpenAI Agents SDK next; any agent runs through the same interface
- **OpenTelemetry GenAI semantic conventions** — traces your existing observability stack ([Langfuse](https://langfuse.com), [Phoenix](https://phoenix.arize.com), [Datadog LLM Obs](https://www.datadoghq.com/product/llm-observability/)) already understands
- **Async clients for Claude, GPT, and Gemini** with rate-limiting, retries, and resumable runs — open-weight models via Together/Groq on the roadmap
- **Reproducible from a single CLI command** — seeded, versioned, every run re-runnable end-to-end

> ⚠️ Status: **v0.1 in progress.** Expect breaking changes. See `docs/ROADMAP.md`.

## What it tests

Most "agent evals" check whether the agent got one right answer. Steadfast checks the things you actually want to know before shipping:

| What                       | Practitioner question                                              | Sub-metrics                                                |
| -------------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------- |
| **Consistency**            | Same input → same output, twice in a row?                          | Output similarity, trajectory similarity, format stability |
| **Robustness**             | Survives a typo, a distractor, a contradiction, a longer document? | Typo / distractor / contradiction / long-context           |
| **Calibration**            | Knows when it's right — and when it isn't?                         | Brier score, ECE, refusal calibration                      |
| **Safety** *(in progress)* | Fails gracefully? Refuses what it should, without over-refusing?   | Catastrophic failure rate, refusal accuracy                |

> Safety is the last v0.1 metric still in progress — see `docs/ROADMAP.md`.

Every metric ships with bootstrapped 95% confidence intervals across N=10 runs, because a single number isn't an evaluation.

## Quickstart

```bash
# install from the repo while v0.1 is in progress; PyPI release lands at v0.1
pip install git+https://github.com/prith98/steadfast.git

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

## Why it exists

Agent accuracy improved 2× in 2025. Agent reliability did not. A 95%-reliable step chained 20 deep is a 36%-reliable system — and most teams shipping LLM agents today have no systematic way to measure where in that chain things break.

Existing eval libraries (DeepEval, Ragas, agentevals) ship excellent primitives for *outcome quality*. None ship an opinionated, end-to-end harness for the things that decide whether an agent is *deployable*. Steadfast does that, and only that. The four-dimension framework comes from Princeton's *Towards a Science of AI Agent Reliability* (Kapoor, Narayanan et al., 2026).

## Leaderboard

Latest results across frontier models: [steadfast.dev/leaderboard](https://steadfast.dev/leaderboard) _(placeholder)_

## v0.1 status

**Shipped:**

- ✅ Three of four reliability dimensions — Consistency, Calibration, and Robustness (with sub-metrics for typo, distractor, contradiction, and long-context perturbations)
- ✅ Outcome scoring + LLM-as-judge
- ✅ OpenTelemetry GenAI semantic convention compliance
- ✅ Async clients for Claude, GPT, and Gemini with rate-limiting, retries, and resumable runs
- ✅ LangGraph adapter
- ✅ Static HTML report — cross-model views, per-task drill-downs, inline SVG long-context curves
- ✅ Robustness pilot across three frontier models (full coverage for typo / contradiction; long-context on gpt-5.2 and gemini-2.5-pro)

**In progress for v0.1:**

- 🚧 Safety dimension — catastrophic failure rate, refusal accuracy
- 🚧 OpenAI Agents SDK adapter
- 🚧 50-task benchmark across customer support, code repair, multi-hop research
- 🚧 Full benchmark run across 5 models (Claude, GPT, Gemini, plus open-weight models via Together / Groq)
- 🚧 Public leaderboard site at `steadfast.dev/leaderboard`
- 🚧 Methodology writeup

**Not in v0.1:**

- ❌ Real-time / streaming eval
- ❌ Custom UI / dashboard
- ❌ Multi-agent system evaluation
- ❌ Multimodal tasks
- ❌ Fine-tuning loops

See `docs/ROADMAP.md` for what's planned in v0.2.

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
