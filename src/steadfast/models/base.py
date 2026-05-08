"""``BaseModelClient`` — async interface common to all provider clients.

Provides:

* an async ``acomplete`` / ``achat`` surface
* token usage tracking (input/output) and cost computation via
  ``steadfast.models.pricing``
* exponential-backoff retry (``tenacity``, added Tuesday)
* a per-provider semaphore for rate-limit-aware concurrency

Implementation in ``docs/WEEK_1.md`` §"Tuesday".
"""

from __future__ import annotations
