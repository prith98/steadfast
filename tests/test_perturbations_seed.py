"""Tests for steadfast.perturbations._seed — sha256-based seed derivation.

Per ADR-0006 §B, the seed-derivation primitive is the reproducibility
foundation for every robustness sub-metric. Hand-computed expected
values pin the formula so any future change (algorithm swap, byte slice,
or version-suffix bump) trips a test rather than silently invalidating
prior leaderboard entries.
"""

from __future__ import annotations

import hashlib

import pytest

from steadfast.perturbations._seed import SEED_VERSION, derive_seed


def _ref(payload: str) -> int:
    """Reference implementation: ADR-0006 §B verbatim."""
    return int.from_bytes(
        hashlib.sha256(payload.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )


# ---------------------------------------------------------------------------
# Hand-computed expected values
# ---------------------------------------------------------------------------


def test_base_seed_matches_adr_formula() -> None:
    """Base form: ``sha256("{task_id}:{kind}:v1")[:8]`` (per ADR-0006 §B)."""
    assert derive_seed("pilot_001", "typo") == _ref("pilot_001:typo:v1")
    assert derive_seed("pilot_001", "typo") == 14783675440936682357


def test_rep_extension_matches_adr_formula() -> None:
    """Per-rep: ``sha256("{task_id}:{kind}:v1:rep{idx}")[:8]``."""
    assert derive_seed("pilot_001", "typo", rep_idx=0) == _ref("pilot_001:typo:v1:rep0")
    assert derive_seed("pilot_001", "typo", rep_idx=0) == 8062193515252073443
    assert derive_seed("pilot_001", "typo", rep_idx=1) == 18395401399291058478


def test_tool_call_extension_matches_internal_format() -> None:
    """Per-tool-call: ``sha256("{task_id}:contradiction:v1:tool{idx}")[:8]``.

    The ``tool`` prefix matches the ``rep{idx}`` style of the per-rep
    extension; both extensions are prefixed for self-consistency. ADR-0006
    §B's example formula in the prose uses a bare ``{tool_call_idx}``,
    which is being amended to match this implementation.
    """
    assert derive_seed("pilot_001", "contradiction", tool_call_idx=0) == _ref(
        "pilot_001:contradiction:v1:tool0"
    )
    assert derive_seed("pilot_001", "contradiction", tool_call_idx=0) == 10830967691823974649


def test_distractor_per_rep_matches_adr_formula() -> None:
    assert derive_seed("pilot_002", "distractor") == _ref("pilot_002:distractor:v1")
    assert derive_seed("pilot_002", "distractor") == 12898619318418955593
    assert derive_seed("pilot_002", "distractor", rep_idx=3) == _ref("pilot_002:distractor:v1:rep3")


# ---------------------------------------------------------------------------
# Discrimination invariants (different inputs → different outputs)
# ---------------------------------------------------------------------------


def test_different_task_ids_yield_different_seeds() -> None:
    a = derive_seed("pilot_001", "typo")
    b = derive_seed("pilot_002", "typo")
    assert a != b


def test_different_kinds_yield_different_seeds() -> None:
    a = derive_seed("pilot_001", "typo")
    b = derive_seed("pilot_001", "distractor")
    assert a != b


def test_different_rep_indices_yield_different_seeds() -> None:
    seeds = {derive_seed("pilot_001", "typo", rep_idx=i) for i in range(10)}
    assert len(seeds) == 10  # all distinct


def test_base_seed_differs_from_rep0_seed() -> None:
    """Base form (no ``:rep`` suffix) and ``rep_idx=0`` are different keys."""
    base = derive_seed("pilot_001", "typo")
    rep0 = derive_seed("pilot_001", "typo", rep_idx=0)
    assert base != rep0


# ---------------------------------------------------------------------------
# Stability invariants
# ---------------------------------------------------------------------------


def test_seed_is_deterministic_across_calls() -> None:
    a = derive_seed("pilot_005", "long_context", rep_idx=7)
    b = derive_seed("pilot_005", "long_context", rep_idx=7)
    assert a == b


def test_seed_fits_in_unsigned_64_bit() -> None:
    """The seed is the first 8 bytes of a SHA-256 digest, big-endian unsigned."""
    seed = derive_seed("pilot_005", "long_context", rep_idx=7)
    assert 0 <= seed < 2**64


def test_seed_version_constant_is_v1() -> None:
    """ADR-0006 §B locks v0.1 to the ``:v1`` version suffix."""
    assert SEED_VERSION == "v1"


# ---------------------------------------------------------------------------
# Custom version suffix
# ---------------------------------------------------------------------------


def test_custom_version_changes_seed() -> None:
    """A future ``:v2`` derivation must produce a different seed than ``:v1``."""
    v1 = derive_seed("pilot_001", "typo", version="v1")
    v2 = derive_seed("pilot_001", "typo", version="v2")
    assert v1 != v2
    assert v2 == _ref("pilot_001:typo:v2")


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rep_idx": 0},
        {"rep_idx": 999},
        {"tool_call_idx": 0},
        {"tool_call_idx": 5},
    ],
)
def test_extension_kwargs_accepted(kwargs: dict[str, int]) -> None:
    """Each extension parameter alone is a legal call."""
    seed = derive_seed("pilot_001", "typo", **kwargs)
    assert isinstance(seed, int)
