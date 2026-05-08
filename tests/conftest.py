"""Pytest fixtures shared across the test tree.

Monday's surface is intentionally minimal: tests that need live LLM APIs are
marked ``@pytest.mark.live`` and deselected by default. As the test tree grows
(Tuesday onward), this file will gain async-event-loop scope, API-key
fixtures backed by ``pytest-env``, and a deterministic-seed fixture.
"""

from __future__ import annotations
