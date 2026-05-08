"""Judge protocol and ``Verdict`` Pydantic model.

Per ``docs/WEEK_1.md`` §"Wednesday", a ``Verdict`` carries
``score: float``, ``passed: bool``, and ``reason: str``. The ``Judge``
protocol exposes a single ``judge(task, response) -> Verdict`` method.

Implementation Wednesday.
"""

from __future__ import annotations
