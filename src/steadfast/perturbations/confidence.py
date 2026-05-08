"""Confidence elicitation — frozen prompt suffix + parser.

Per Q1 (project kickoff), the **primary contract** requires the integrator to
populate ``AgentResponse.confidence`` using the helper exposed here:

    suffix = load_confidence_suffix_v1()  # from prompts/confidence_v1.txt
    confidence = parse_verbalized_confidence(model_output)

A clearly-labeled **post-hoc variant** runs a second LLM call after
``Agent.arun`` returns; results from that variant are reported under a separate
column on the leaderboard, never silently mixed with primary-contract results.

Logprob-derived confidence (where the provider exposes it — OpenAI yes,
Anthropic limited, Google check) is recorded alongside verbalized confidence
per ``docs/METHODOLOGY.md`` §3.1.

Implementation in ``docs/WEEK_1.md`` §"Friday".
"""

from __future__ import annotations
