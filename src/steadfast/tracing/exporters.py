"""Tracer-provider configuration and exporter wiring.

Per ADR-0003 §A, the harness supports three exporter modes:

* ``console`` (default): :class:`ConsoleSpanExporter` — useful for local
  development. Spans print to stdout via the SDK's default
  :class:`BatchSpanProcessor`.
* ``otlp``: :class:`OTLPSpanExporter` (HTTP/protobuf) — for Phoenix /
  Langfuse / Datadog ingest. Defaults to Phoenix's local endpoint
  (``http://localhost:6006/v1/traces``) when ``OTEL_EXPORTER_OTLP_ENDPOINT``
  is unset.
* ``none``: install a :class:`TracerProvider` with no span processor.
  Spans are still created (the API is no-op-safe) but nothing is exported.
  Used by tests and by users who want a quiet ``steadfast bench``.

A single call to :func:`configure_tracing` installs the provider on the
global OTel registry. Subsequent calls override (the SDK logs a warning
about the override; we accept that — re-configuring across invocations
is unusual outside tests, and tests build their own provider rather than
calling :func:`configure_tracing`).
"""

from __future__ import annotations

import os
from typing import Literal

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

from steadfast import __version__
from steadfast.tracing.conventions import (
    GENAI_CONVENTIONS_VERSION,
    STEADFAST_GENAI_CONVENTIONS_VERSION,
    STEADFAST_PACKAGE_VERSION,
)

ExporterKind = Literal["console", "otlp", "none"]

# Phoenix's default local OTLP/HTTP collector endpoint. Used when the user
# selects ``--exporter otlp`` and has not set OTEL_EXPORTER_OTLP_ENDPOINT.
PHOENIX_DEFAULT_ENDPOINT: str = "http://localhost:6006/v1/traces"


def _build_resource() -> Resource:
    """Resource attributes attached to every span emitted by Steadfast.

    ``service.name`` and ``service.version`` are the OTel-canonical fields
    used by Phoenix to group runs into a project. The Steadfast-specific
    keys are namespaced via ``steadfast.*`` so they don't collide with
    other instrumentation in the same collector.
    """
    return Resource.create(
        {
            "service.name": "steadfast",
            "service.version": __version__,
            STEADFAST_PACKAGE_VERSION: __version__,
            STEADFAST_GENAI_CONVENTIONS_VERSION: GENAI_CONVENTIONS_VERSION,
        }
    )


def _otlp_endpoint() -> str:
    """Resolve the OTLP/HTTP endpoint, defaulting to Phoenix when unset."""
    return os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or PHOENIX_DEFAULT_ENDPOINT


def configure_tracing(*, exporter: ExporterKind = "none") -> TracerProvider:
    """Install a :class:`TracerProvider` configured for ``exporter`` mode.

    Returns the installed provider so callers can hold a reference and
    invoke ``.shutdown()`` / ``.force_flush()`` (the CLI does this on exit
    so async batch exporters drain before process termination).

    See ADR-0003 §A.5 for the Phoenix endpoint default.
    """
    provider = TracerProvider(resource=_build_resource())

    if exporter == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=_otlp_endpoint())))
    elif exporter == "none":
        # No processor; spans are dropped on close. The API still works
        # (start_span returns a real span), so call sites are unchanged.
        pass
    else:  # pragma: no cover — Literal type guards this at the call site
        raise ValueError(f"unknown exporter: {exporter!r}")

    trace.set_tracer_provider(provider)
    return provider
