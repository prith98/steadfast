"""Tests for steadfast.perturbations.typo — character-level noise.

Per METHODOLOGY §2.1 and ADR-0006 §B: deterministic character-level
substitution at rate 5% with a 25% per-word cap, seeded so reps within
a single (task, kind) get distinct draws.
"""

from __future__ import annotations

import re
import statistics

import pytest

from steadfast.perturbations import derive_seed
from steadfast.perturbations.typo import (
    DEFAULT_MAX_WORD_CORRUPTION,
    DEFAULT_RATE,
    perturb_typo,
)

_WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


def _word_diff_counts(original: str, perturbed: str) -> list[tuple[int, int]]:
    """Per-word ``(word_len, n_diff_chars)`` pairs.

    Substitution-only mutation preserves length, so per-word indices align
    one-to-one between ``original`` and ``perturbed``. Word boundaries are
    the same alphanumeric runs the perturbation uses.
    """
    out: list[tuple[int, int]] = []
    o_words = list(_WORD_RE.finditer(original))
    p_words = list(_WORD_RE.finditer(perturbed))
    assert len(o_words) == len(p_words)
    for om, pm in zip(o_words, p_words, strict=True):
        assert om.span() == pm.span()
        diffs = sum(1 for o, p in zip(om.group(), pm.group(), strict=True) if o != p)
        out.append((len(om.group()), diffs))
    return out


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_same_output() -> None:
    text = "What is the return window for unopened items at our store?"
    a = perturb_typo(text, seed=42)
    b = perturb_typo(text, seed=42)
    assert a == b


def test_different_seed_different_output() -> None:
    text = "What is the return window for unopened items at our store?"
    a = perturb_typo(text, seed=42)
    b = perturb_typo(text, seed=43)
    assert a != b


def test_per_rep_seed_helper_yields_distinct_outputs() -> None:
    """ADR-0006 §B: per-rep seeding gives N distinct perturbed inputs.

    Without per-rep seeding, the N=10 perturbed reps would collapse to a
    single perturbed string repeated ten times, which would defeat
    distributional measurement.
    """
    text = "Return windows are 30 days for our customers; please verify with us."
    outputs = [
        perturb_typo(text, seed=derive_seed("pilot_001", "typo", rep_idx=i)) for i in range(10)
    ]
    distinct = set(outputs)
    # We expect ~all 10 to be distinct; allow at most 1 collision in case of
    # bad luck with low-rate perturbations on short text. Determinism keeps
    # this number stable across runs.
    assert len(distinct) >= 8, f"only {len(distinct)} distinct outputs from 10 reps"


# ---------------------------------------------------------------------------
# Length and word-boundary invariants
# ---------------------------------------------------------------------------


def test_length_preserved() -> None:
    """Substitution-only mutation must not change total length."""
    text = "Hello World — 30-day windows!"
    out = perturb_typo(text, seed=7)
    assert len(out) == len(text)


def test_punctuation_and_whitespace_unchanged() -> None:
    """Only characters inside word runs (``\\w+``) are mutation candidates."""
    text = "A!B@C#D — 30-day, 90-day; OK?"
    out = perturb_typo(text, seed=11, rate=1.0, max_word_corruption=1.0)
    # Iterate non-word characters; both strings must have them at the same
    # positions and with the same content.
    for i, ch in enumerate(text):
        if not ch.isalnum():
            assert out[i] == ch, f"non-word char at index {i} mutated: {out!r}"


def test_word_boundaries_unchanged() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    out = perturb_typo(text, seed=99, rate=0.5, max_word_corruption=0.5)
    o_spans = [m.span() for m in _WORD_RE.finditer(text)]
    p_spans = [m.span() for m in _WORD_RE.finditer(out)]
    assert o_spans == p_spans


# ---------------------------------------------------------------------------
# Per-word cap (METHODOLOGY §2.1)
# ---------------------------------------------------------------------------


def test_per_word_cap_floor_semantics() -> None:
    """No word may be mutated above ``floor(max_word_corruption * len(word))``.

    With max_word_corruption=0.25, a 4-char word gets at most 1 mutation;
    a 3-char or shorter word gets 0. Asserting on a forced 100% rate (so
    the per-word cap is the only effective limit) makes the test exercise
    the cap directly rather than the rate.
    """
    text = "The quick brown fox jumps over the lazy dog and many other animals."
    out = perturb_typo(text, seed=3, rate=1.0, max_word_corruption=0.25)
    diffs = _word_diff_counts(text, out)
    for word_len, n_diffs in diffs:
        cap = int(0.25 * word_len)
        assert n_diffs <= cap, f"word of length {word_len} had {n_diffs} mutations; cap={cap}"


def test_short_words_immune_at_default_cap() -> None:
    """``floor(0.25 * 3) == 0`` — three-letter words must remain unchanged."""
    text = "The cat sat on the red mat the cat sat the cat"
    out = perturb_typo(text, seed=5, rate=1.0, max_word_corruption=0.25)
    diffs = _word_diff_counts(text, out)
    for word_len, n_diffs in diffs:
        if word_len < 4:
            assert n_diffs == 0, f"short word (len {word_len}) was mutated; floor cap should be 0"


def test_max_word_corruption_zero_means_no_mutations() -> None:
    text = "The quick brown fox jumps over the lazy dog."
    out = perturb_typo(text, seed=1, rate=1.0, max_word_corruption=0.0)
    assert out == text


# ---------------------------------------------------------------------------
# Rate convergence at large N
# ---------------------------------------------------------------------------


def test_empirical_rate_close_to_target_at_large_n() -> None:
    """Across a long input, the empirical mutation rate should approach the target.

    The per-word cap can suppress the rate when many words are short
    (which is realistic for natural prose). We assert a generous lower
    bound — the rate-floor is essentially "per-word cap is the binding
    constraint" — and an upper bound at the target.
    """
    # Long-ish text with a mix of word lengths; rate=0.10 with the default
    # cap of 0.25 means longer words contribute most of the mutation budget.
    text = (
        "The reliability benchmark measures whether autonomous agents produce "
        "consistent answers across repeated identical prompts and across "
        "perturbed variants of those prompts that should not change the answer. "
        "Calibration measurement separately evaluates whether the agent's "
        "stated confidence correlates with actual correctness across the "
        "benchmark suite, with refusal calibration on a curated subset."
    ) * 4

    rates: list[float] = []
    for seed in range(20):
        out = perturb_typo(text, seed=seed, rate=0.10, max_word_corruption=0.25)
        diffs = _word_diff_counts(text, out)
        n_alphanum = sum(wl for wl, _ in diffs)
        n_diffs = sum(d for _, d in diffs)
        rates.append(n_diffs / n_alphanum)

    mean_rate = statistics.mean(rates)
    # Average rate should sit between the per-word-cap floor and the
    # nominal target. Floor is ~5-7% on natural English with 0.25 cap;
    # target is 10%.
    assert 0.04 <= mean_rate <= 0.10 + 1e-9, (
        f"empirical mean rate {mean_rate:.3f} outside expected band [0.04, 0.10]"
    )


def test_zero_rate_returns_original() -> None:
    text = "Returns are accepted within 30 days of purchase."
    out = perturb_typo(text, seed=0, rate=0.0)
    assert out == text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_text_returns_empty() -> None:
    assert perturb_typo("", seed=0) == ""


def test_text_with_no_word_chars_returns_unchanged() -> None:
    text = "...!!!---"
    assert perturb_typo(text, seed=0, rate=1.0) == text


def test_unicode_letter_preserved_within_word_boundaries() -> None:
    """Words may contain non-ASCII letters; the perturbation may
    substitute them with ASCII letters but must not break the word
    boundary."""
    text = "café résumé naïve"
    out = perturb_typo(text, seed=2, rate=1.0, max_word_corruption=0.5)
    o_spans = [m.span() for m in _WORD_RE.finditer(text)]
    p_spans = [m.span() for m in _WORD_RE.finditer(out)]
    assert o_spans == p_spans


def test_invalid_rate_raises() -> None:
    with pytest.raises(ValueError, match="rate must be"):
        perturb_typo("hello", seed=0, rate=-0.1)
    with pytest.raises(ValueError, match="rate must be"):
        perturb_typo("hello", seed=0, rate=1.5)


def test_invalid_max_word_corruption_raises() -> None:
    with pytest.raises(ValueError, match="max_word_corruption must be"):
        perturb_typo("hello", seed=0, max_word_corruption=-0.1)
    with pytest.raises(ValueError, match="max_word_corruption must be"):
        perturb_typo("hello", seed=0, max_word_corruption=1.5)


# ---------------------------------------------------------------------------
# Methodology defaults
# ---------------------------------------------------------------------------


def test_methodology_defaults() -> None:
    """METHODOLOGY §2.1 specifies rate=5%, per-word cap=25%."""
    assert DEFAULT_RATE == 0.05
    assert DEFAULT_MAX_WORD_CORRUPTION == 0.25
