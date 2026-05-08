"""OpenTelemetry GenAI semantic-convention attribute names and version pin.

This module is the single source of truth for the ``gen_ai.*`` attribute names
that Steadfast emits. Per the Q4 decision (project kickoff), we own the pin
rather than depending on the experimental ``opentelemetry-instrumentation-genai``
package — that package's surface is still in flux and pinning a spec version
gives us a stable, auditable target.

Spec: https://opentelemetry.io/docs/specs/semconv/gen-ai/
Agentic systems extension (draft): https://github.com/open-telemetry/semantic-conventions/issues/2664

Verify ``GENAI_CONVENTIONS_VERSION`` against the latest stable semconv release
before Wednesday's tracing implementation; any change is a methodology event
and needs a CHANGELOG entry.

Wednesday's work (``docs/WEEK_1.md`` §"Wednesday") populates the attribute
table and the span-helper module.
"""

from __future__ import annotations

from typing import Final

GENAI_CONVENTIONS_VERSION: Final[str] = "1.41.0"
