"""Steadfast — reliability benchmarking for AI agents.

The artifact has four parts: a frozen statistical methodology
(``docs/METHODOLOGY.md``), a Python reference implementation, a curated
benchmark suite, and a public leaderboard. See ``docs/SPEC.md`` for scope.

The Tuesday design pass (``docs/WEEK_1.md``) extends the public surface with
``Agent``, ``Task``, and ``AgentResponse``.
"""

from steadfast._version import __version__

__all__ = ["__version__"]
