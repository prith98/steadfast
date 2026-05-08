"""Frozen reproducibility manifest writer.

Per ``docs/METHODOLOGY.md`` §"Reproducibility is a first-class feature", every
benchmark run produces a ``manifest.json`` recording:

* package version (``steadfast.__version__``)
* model identifiers and provider SDK versions
* framework adapter versions
* task-set version + per-task content hashes
* random seed
* infrastructure-model identifiers (per ADR-0001)
* timestamps and host metadata

Implemented late week 1 / early week 2. Empty stub on Monday.
"""

from __future__ import annotations
