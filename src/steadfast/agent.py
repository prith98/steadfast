"""Agent ABC and Pydantic models for ``Task`` and ``AgentResponse``.

Per the Tuesday design (``docs/WEEK_1.md`` §"Tuesday — Agent abstraction"), this
module will define:

* ``Agent`` — abstract base class with ``arun(task: Task) -> AgentResponse``
  and a sync ``run`` wrapper.
* ``Task`` — Pydantic model for a single benchmark task (id, input, ground
  truth, ground-truth checker spec, optional ``confidence_suffix``).
* ``AgentResponse`` — Pydantic model for a single attempt: ``answer``,
  optional ``confidence ∈ [0, 1]``, ``trajectory`` (list of tool calls), and
  ``metadata: dict[str, str | int | float | bool]`` (per Q5 type policy).

**Confidence elicitation contract (Q1 decision).** The primary contract
requires the integrator to populate ``AgentResponse.confidence`` using the
helper in ``steadfast.perturbations.confidence``. A clearly-labeled
"post-hoc confidence" variant exists for black-box agents; results from that
variant are reported under a separate column on the leaderboard, never
silently mixed.

Empty stub on Monday. The 1-page design lives in a forthcoming ADR before
implementation begins on Tuesday.
"""

from __future__ import annotations
