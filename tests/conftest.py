"""Pytest fixtures shared across the test tree.

* :func:`memory_exporter` — yields a freshly-cleared in-memory span exporter
  for tests that need to assert on emitted spans.

OpenTelemetry's :class:`TracerProvider` is a process-global singleton; the
SDK refuses to swap it after one is set. We therefore install ONE provider
for the entire test session here, with a single
:class:`InMemorySpanExporter` attached via :class:`SimpleSpanProcessor`, and
every test that needs span assertions takes the exporter from this fixture
(which clears it before yielding).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

_TEST_EXPORTER: InMemorySpanExporter | None = None


@pytest.fixture(scope="session", autouse=True)
def _session_tracing() -> Iterator[None]:
    """Install one TracerProvider + InMemorySpanExporter for the whole session."""
    global _TEST_EXPORTER
    _TEST_EXPORTER = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_TEST_EXPORTER))
    trace.set_tracer_provider(provider)
    yield
    provider.shutdown()


@pytest.fixture
def memory_exporter() -> Iterator[InMemorySpanExporter]:
    """Yield a freshly-cleared in-memory span exporter for one test."""
    assert _TEST_EXPORTER is not None, "session fixture must run first"
    _TEST_EXPORTER.clear()
    yield _TEST_EXPORTER
    _TEST_EXPORTER.clear()
