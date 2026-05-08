"""LLM-as-judge with a Pydantic-validated output schema.

Defaults to ``gpt-5.2`` per ADR-0001 (``docs/adr/0001-infrastructure-model.md``).
The judge receives a frozen rubric prompt plus the task and response, and
returns a structured verdict — Pydantic validates the model's JSON output
before it touches the metric pipeline.

Pattern adapted from the Pydantic Evals LLM-as-judge guide
(https://pydantic.dev/articles/llm-as-a-judge); ensemble variant in
``ensemble.py`` covers the residual judge-bias risk noted in
``docs/METHODOLOGY.md`` §"Known limitations".

Implementation in ``docs/WEEK_1.md`` §"Wednesday".
"""

from __future__ import annotations
