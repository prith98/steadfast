"""Adapter for the OpenAI Agents SDK.

Wraps an OpenAI-Agents-SDK agent so it conforms to ``steadfast.agent.Agent``.
Hands off tool-call extraction to ``steadfast.tracing.spans``.

Implementation in week 1 (Tuesday for the trivial wrapper, Wednesday once the
trace plumbing exists).
"""

from __future__ import annotations
