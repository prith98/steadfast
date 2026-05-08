"""Steadfast — reliability benchmarking for AI agents.

The artifact has four parts: a frozen statistical methodology
(``docs/METHODOLOGY.md``), a Python reference implementation, a curated
benchmark suite, and a public leaderboard. See ``docs/SPEC.md`` for scope.

The public surface (per ADR-0002):

* ``Agent`` — abstract base class for benchmarkable agents.
* ``Task`` / ``GroundTruth`` / ``ToolCall`` / ``AgentResponse`` — Pydantic
  contract types.
* ``SimplePromptingAgent`` — built-in single-shot agent used by the CLI when
  no ``--agent`` is provided.
"""

from steadfast._version import __version__
from steadfast.agent import (
    Agent,
    AgentResponse,
    GroundTruth,
    SimplePromptingAgent,
    Task,
    ToolCall,
)

__all__ = [
    "Agent",
    "AgentResponse",
    "GroundTruth",
    "SimplePromptingAgent",
    "Task",
    "ToolCall",
    "__version__",
]
