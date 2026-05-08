"""Single source of truth for the package version.

Bump in lockstep with ``pyproject.toml``. The leaderboard's manifest records
this string verbatim so reproductions can pin it.
"""

from __future__ import annotations

from typing import Final

__version__: Final[str] = "0.1.0.dev0"
