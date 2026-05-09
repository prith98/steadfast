"""OpenTelemetry GenAI tracing — span helpers and exporter wiring.

Public surface for the rest of the harness:

* :func:`configure_tracing` — install a :class:`TracerProvider` on the
  global OTel registry. Called once by the CLI.
* :func:`benchmark_span`, :func:`task_span`, :func:`rep_span`,
  :func:`chat_span`, :func:`score_span` — context managers for the
  Steadfast span hierarchy.
* :func:`record_chat_response`, :func:`record_retry_event`,
  :func:`record_verdict` — attribute-population helpers.

See ``docs/adr/0003-tracing-and-judges.md`` for the design rationale.
"""

from steadfast.tracing.exporters import (
    PHOENIX_DEFAULT_ENDPOINT,
    ExporterKind,
    configure_tracing,
)
from steadfast.tracing.spans import (
    benchmark_span,
    chat_span,
    embeddings_span,
    record_chat_response,
    record_embeddings_response,
    record_retry_event,
    record_verdict,
    rep_span,
    score_span,
    task_span,
)

__all__ = [
    "PHOENIX_DEFAULT_ENDPOINT",
    "ExporterKind",
    "benchmark_span",
    "chat_span",
    "configure_tracing",
    "embeddings_span",
    "record_chat_response",
    "record_embeddings_response",
    "record_retry_event",
    "record_verdict",
    "rep_span",
    "score_span",
    "task_span",
]
