"""Tests for steadfast.perturbations.long_context — determinism + token targeting.

The methodology contract is in METHODOLOGY §2.4 and ADR-0006 §E. The
unit-level surface tested here:

* Token-target accuracy at every standard tier (4k / 16k / 64k / 128k).
* Determinism under identical (text, seed, target_tokens) inputs.
* Per-rep seeding produces distinct filler windows.
* Tiling on requested windows that exceed corpus length.
* Delimiter presence and structural shape.
* Edge cases (target too small, empty filler).

The integration with the metric layer is exercised in
``tests/test_metrics_robustness.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from steadfast.perturbations import derive_seed
from steadfast.perturbations.long_context import (
    DEFAULT_FILLER_PATH,
    count_tokens,
    perturb_long_context,
)

_TASK_TEXT = "What is the return window for unopened items at our store?"


# ---------------------------------------------------------------------------
# Token targeting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_tokens", [4_000, 16_000, 64_000, 128_000])
def test_perturbation_tokens_within_tolerance_of_target(target_tokens: int) -> None:
    """Each tier hits ``target_tokens`` within a small tolerance.

    Tokenization is not strictly injective across encode/decode for
    cl100k_base — boundary tokens at the filler/delimiter seam can
    re-tokenize. The contract is "close to target", not "exactly
    target." Per-tier tolerance of ±5 tokens covers the boundary
    effect at every tier in the standard ladder.
    """
    seed = derive_seed("pilot_001", "long_context", rep_idx=0, tool_call_idx=0)
    out = perturb_long_context(_TASK_TEXT, target_tokens=target_tokens, seed=seed)
    actual = count_tokens(out)
    assert abs(actual - target_tokens) <= 5, (
        f"target={target_tokens} but got {actual} tokens (delta={actual - target_tokens})"
    )


def test_perturbation_includes_delimiter_and_original_text() -> None:
    """Output ends with the delimiter + original task text."""
    seed = derive_seed("t", "long_context", rep_idx=0)
    out = perturb_long_context(_TASK_TEXT, target_tokens=4_000, seed=seed)
    assert "--- task ---" in out
    assert out.endswith(_TASK_TEXT), "original task must appear at the end"


def test_perturbation_filler_precedes_task_delimiter() -> None:
    """Filler comes first, then the delimiter, then the task — in that order."""
    seed = derive_seed("t", "long_context", rep_idx=0)
    out = perturb_long_context(_TASK_TEXT, target_tokens=4_000, seed=seed)
    delim_idx = out.index("--- task ---")
    task_idx = out.index(_TASK_TEXT)
    # Filler occupies [0, delim_idx); delimiter line is around delim_idx;
    # task starts after the delimiter.
    assert delim_idx > 0
    assert task_idx > delim_idx


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_perturbation_is_deterministic_given_seed() -> None:
    """Same (text, seed, target) → byte-identical output."""
    seed = derive_seed("pilot_001", "long_context", rep_idx=3)
    a = perturb_long_context(_TASK_TEXT, target_tokens=4_000, seed=seed)
    b = perturb_long_context(_TASK_TEXT, target_tokens=4_000, seed=seed)
    assert a == b


def test_different_seeds_yield_different_outputs() -> None:
    """Per-rep seeds (ADR-0006 §B) must pick distinct filler windows."""
    seeds = [derive_seed("pilot_001", "long_context", rep_idx=i) for i in range(5)]
    outputs = {perturb_long_context(_TASK_TEXT, target_tokens=4_000, seed=s) for s in seeds}
    # All five reps must produce distinct outputs; the corpus is large
    # enough relative to the window that collisions are not expected.
    assert len(outputs) == 5


def test_different_tiers_get_different_seeds() -> None:
    """The metric layer rides the tier index on the tool_call_idx slot.

    Two reps at the same rep_idx but different tiers must pull different
    seeds (which the metric uses to draw distinct filler windows even
    though both reps share rep_idx). This test asserts the seed
    derivation behavior the metric depends on.
    """
    s_tier0 = derive_seed("pilot_001", "long_context", rep_idx=0, tool_call_idx=0)
    s_tier1 = derive_seed("pilot_001", "long_context", rep_idx=0, tool_call_idx=1)
    assert s_tier0 != s_tier1


# ---------------------------------------------------------------------------
# Tiling — when requested window exceeds corpus length
# ---------------------------------------------------------------------------


def test_perturbation_tiles_when_target_exceeds_corpus() -> None:
    """128k target requires ~34x tiling of the ~3.8k-token corpus."""
    seed = derive_seed("pilot_001", "long_context", rep_idx=0)
    out = perturb_long_context(_TASK_TEXT, target_tokens=128_000, seed=seed)
    actual = count_tokens(out)
    assert abs(actual - 128_000) <= 5
    # Sanity: the output is large enough that the corpus must have been
    # tiled (otherwise the corpus is way bigger than expected and the
    # docstring's tile-rate analysis is wrong).
    corpus_len = count_tokens(Path(DEFAULT_FILLER_PATH).read_text(encoding="utf-8"))
    assert actual > corpus_len, "tiling should have produced a window larger than the corpus"


def test_perturbation_window_advances_through_corpus_by_seed() -> None:
    """Two seeds an even N apart pick predictably-different starts.

    Documents the deterministic-offset contract (seed % corpus_len)
    without coupling to the exact filler content.
    """
    s1 = derive_seed("a", "long_context", rep_idx=0)
    s2 = derive_seed("b", "long_context", rep_idx=0)
    out1 = perturb_long_context(_TASK_TEXT, target_tokens=4_000, seed=s1)
    out2 = perturb_long_context(_TASK_TEXT, target_tokens=4_000, seed=s2)
    assert out1 != out2


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------


def test_perturbation_rejects_target_too_small() -> None:
    """``target_tokens`` smaller than (task + delimiter) must raise."""
    seed = 0
    # The task tokenizes to ~13 tokens; delimiter to ~6. target_tokens=5
    # cannot fit either.
    with pytest.raises(ValueError, match="leaves no room"):
        perturb_long_context(_TASK_TEXT, target_tokens=5, seed=seed)


def test_perturbation_rejects_zero_target() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        perturb_long_context(_TASK_TEXT, target_tokens=0, seed=0)


def test_perturbation_rejects_negative_target() -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        perturb_long_context(_TASK_TEXT, target_tokens=-100, seed=0)


def test_perturbation_rejects_empty_filler_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        perturb_long_context(_TASK_TEXT, target_tokens=4_000, seed=0, filler_path=empty)


def test_perturbation_uses_custom_filler_path(tmp_path: Path) -> None:
    """Override filler_path with a small fixture corpus."""
    fixture = tmp_path / "tiny.txt"
    fixture.write_text("THE QUICK BROWN FOX " * 200)
    seed = derive_seed("t", "long_context", rep_idx=0)
    out = perturb_long_context(_TASK_TEXT, target_tokens=500, seed=seed, filler_path=fixture)
    assert "THE QUICK BROWN FOX" in out
    assert _TASK_TEXT in out
    assert "--- task ---" in out
