"""Per-model pricing snapshot for cost computation.

The :data:`PRICING` table is the canonical source of truth for cost
computation across the harness. ``dated_at`` on every entry lands in the run
manifest so reproductions can verify the pricing assumption. Update via PR
with a CHANGELOG entry when prices change.

**Verify before publishing leaderboard results.** Prices below are a snapshot
informed by published rates as of 2026-05-08. Sources:

* Anthropic — anthropic.com/pricing
* OpenAI — developers.openai.com/api/docs/pricing (current flagship: GPT-5.4)
* Google — ai.google.dev/gemini-api/docs/pricing (Gemini 2.5 Pro tiered by context length)

For Gemini's tiered pricing (≤200K vs >200K tokens), v0.1 records the
≤200K-context tier; reproductions exceeding 200K context will under-report
cost. A future :func:`compute_cost` revision will accept a ``context_tokens``
hint to resolve this; tracked in auto-memory.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Final

from steadfast.models.base import ModelPricing, TokenUsage

_SNAPSHOT = date(2026, 5, 8)

PRICING: Final[dict[str, ModelPricing]] = {
    # ---- Anthropic ----
    "claude-opus-4-7": ModelPricing(
        input_per_mtok=Decimal("15"),
        output_per_mtok=Decimal("75"),
        dated_at=_SNAPSHOT,
    ),
    "claude-sonnet-4-6": ModelPricing(
        input_per_mtok=Decimal("3"),
        output_per_mtok=Decimal("15"),
        dated_at=_SNAPSHOT,
    ),
    "claude-haiku-4-5-20251001": ModelPricing(
        input_per_mtok=Decimal("0.80"),
        output_per_mtok=Decimal("4"),
        dated_at=_SNAPSHOT,
    ),
    # ---- OpenAI ----
    # GPT-5.4 (current flagship as of April 2026).
    "gpt-5.4": ModelPricing(
        input_per_mtok=Decimal("2.50"),
        output_per_mtok=Decimal("15.00"),
        dated_at=_SNAPSHOT,
    ),
    # GPT-5.2 (referenced in docs/SPEC.md as the v0.1 leaderboard target).
    "gpt-5.2": ModelPricing(
        input_per_mtok=Decimal("1.25"),
        output_per_mtok=Decimal("10.00"),
        dated_at=_SNAPSHOT,
    ),
    "gpt-5": ModelPricing(
        input_per_mtok=Decimal("1.25"),
        output_per_mtok=Decimal("10.00"),
        dated_at=_SNAPSHOT,
    ),
    "gpt-5-mini": ModelPricing(
        input_per_mtok=Decimal("0.25"),
        output_per_mtok=Decimal("2.00"),
        dated_at=_SNAPSHOT,
    ),
    "gpt-5.4-nano": ModelPricing(
        input_per_mtok=Decimal("0.20"),
        output_per_mtok=Decimal("1.25"),
        dated_at=_SNAPSHOT,
    ),
    # ---- Google ----
    # Gemini 2.5 Pro pricing for ≤200K-context tier.
    "gemini-2.5-pro": ModelPricing(
        input_per_mtok=Decimal("1.25"),
        output_per_mtok=Decimal("10.00"),
        dated_at=_SNAPSHOT,
    ),
    # Gemini 2.5 Flash (cost-optimized).
    "gemini-2.5-flash": ModelPricing(
        input_per_mtok=Decimal("0.30"),
        output_per_mtok=Decimal("2.50"),
        dated_at=_SNAPSHOT,
    ),
}

_PER_MTOK = Decimal("1000000")


def compute_cost(model: str, usage: TokenUsage) -> Decimal:
    """Compute USD cost from token usage, looking up pricing by model ID.

    Raises ``KeyError`` if the model is not registered. Silent zero-cost
    fallback would mask a real bug in the leaderboard's reproducibility
    claims, so we fail loudly.
    """
    if model not in PRICING:
        raise KeyError(f"unknown model for pricing: {model!r}")
    p = PRICING[model]
    return (
        Decimal(usage.input_tokens) * p.input_per_mtok
        + Decimal(usage.output_tokens) * p.output_per_mtok
    ) / _PER_MTOK


def provider_for_model(model: str) -> str:
    """Infer the provider from a model identifier.

    Used by the CLI to choose which :class:`BaseModelClient` subclass to
    instantiate. Naming conventions: ``claude-*`` → Anthropic, ``gpt-*`` →
    OpenAI, ``gemini-*`` → Google. Raises ``ValueError`` on unknown prefixes.
    """
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gpt-"):
        return "openai"
    if model.startswith("gemini-"):
        return "google"
    raise ValueError(
        f"cannot infer provider from model {model!r} — "
        "Steadfast recognizes 'claude-*', 'gpt-*', and 'gemini-*' prefixes"
    )
