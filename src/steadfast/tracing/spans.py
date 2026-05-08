"""Span helpers for the Steadfast hierarchy: benchmark → task → rep → LLM call → tool call.

Per ``docs/WEEK_1.md`` §"Wednesday", required attributes include
``gen_ai.system``, ``gen_ai.request.model``, ``gen_ai.usage.input_tokens``,
``gen_ai.usage.output_tokens``, plus the agentic-systems attributes from the
draft proposal at https://github.com/open-telemetry/semantic-conventions/issues/2664.

Implementation pending Wednesday.
"""

from __future__ import annotations
