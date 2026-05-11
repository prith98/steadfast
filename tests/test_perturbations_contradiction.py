"""Tests for steadfast.perturbations.contradiction — runtime primitives.

The contradiction perturbation is consumed by tool-using agents (the
fixture in ``tests/fixtures/contradiction_agents.py``); these tests pin
the deterministic behavior of the per-call coin (:func:`should_corrupt`),
the corruption transform (:func:`corrupt_tool_result`), and the loaders
for the two frozen prompt files.

End-to-end metric behavior is exercised in
``tests/test_metrics_robustness.py`` against the fixture; here we cover
the primitive surface so a regression in the corruption transforms or
the file format trips a unit test, not an integration test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from steadfast.perturbations.contradiction import (
    CORRUPTED_CALLS_METADATA_KEY,
    DEFAULT_CORRUPTION_PROBABILITY,
    corrupt_tool_result,
    encode_corrupted_call_indices,
    load_corruption_strategies,
    load_detection_phrases,
    should_corrupt,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_default_corruption_probability_matches_methodology() -> None:
    """METHODOLOGY §2.3 pins the per-call rate at 0.3."""
    assert DEFAULT_CORRUPTION_PROBABILITY == 0.3


def test_metadata_key_is_namespaced() -> None:
    """The metadata key is namespaced so other perturbations can coexist."""
    assert CORRUPTED_CALLS_METADATA_KEY.startswith("steadfast.contradiction.")


# ---------------------------------------------------------------------------
# should_corrupt
# ---------------------------------------------------------------------------


def test_should_corrupt_deterministic_per_seed() -> None:
    """Same (task_id, tool_call_idx) → same boolean."""
    a = should_corrupt(task_id="pilot_001", tool_call_idx=0, probability=0.3)
    b = should_corrupt(task_id="pilot_001", tool_call_idx=0, probability=0.3)
    assert a == b


def test_should_corrupt_distinct_calls_get_distinct_coins() -> None:
    """Different tool_call_idx → different RNG seed → not all coins identical."""
    coins = {
        should_corrupt(task_id="pilot_001", tool_call_idx=i, probability=0.5) for i in range(20)
    }
    assert coins == {True, False}, "should see both outcomes across 20 distinct calls"


def test_should_corrupt_probability_zero_never_fires() -> None:
    for i in range(50):
        assert should_corrupt(task_id="t", tool_call_idx=i, probability=0.0) is False


def test_should_corrupt_probability_one_always_fires() -> None:
    for i in range(50):
        assert should_corrupt(task_id="t", tool_call_idx=i, probability=1.0) is True


def test_should_corrupt_proportion_converges_to_target() -> None:
    """At large N, the empirical hit rate is within 2.5pp of the target.

    Stable test (deterministic seeds, no actual randomness) but documents
    the convergence behavior we rely on for the metric's interpretation.
    """
    n = 5000
    hits = sum(
        1
        for i in range(n)
        if should_corrupt(task_id="convergence_check", tool_call_idx=i, probability=0.3)
    )
    ratio = hits / n
    assert abs(ratio - 0.3) < 0.025, f"empirical ratio {ratio:.4f} too far from 0.3"


def test_should_corrupt_rejects_invalid_probability() -> None:
    with pytest.raises(ValueError, match="probability must be in"):
        should_corrupt(task_id="t", tool_call_idx=0, probability=1.5)
    with pytest.raises(ValueError, match="probability must be in"):
        should_corrupt(task_id="t", tool_call_idx=0, probability=-0.1)


# ---------------------------------------------------------------------------
# corrupt_tool_result — strategies are deterministic given seed
# ---------------------------------------------------------------------------


def test_corrupt_tool_result_deterministic() -> None:
    a = corrupt_tool_result("Return window is 30 days", task_id="t", tool_call_idx=0)
    b = corrupt_tool_result("Return window is 30 days", task_id="t", tool_call_idx=0)
    assert a == b


def test_corrupt_tool_result_changes_input() -> None:
    """The corruption strategies always produce a different string."""
    original = "Return window is 30 days for unopened items"
    # Try a few seeds — each strategy is deterministic, so a single seed
    # might land on a strategy that no-ops on this particular text. Across
    # several seeds at least one must produce a change.
    changes = {corrupt_tool_result(original, task_id=f"t{i}", tool_call_idx=0) for i in range(20)}
    assert original not in changes


def test_corrupt_tool_result_with_explicit_strategies() -> None:
    """Passing a single-strategy list pins which transform is applied."""
    out = corrupt_tool_result(
        "We charge $40 for shipping",
        task_id="strategies_pin",
        tool_call_idx=0,
        strategies=["negate_number"],
    )
    # negate_number replaces the first numeric token; the result must
    # differ at the digit position(s).
    assert out != "We charge $40 for shipping"
    assert "$" in out  # the surrounding text is preserved


def test_corrupt_tool_result_negate_number_changes_first_numeric() -> None:
    """The negate_number transform changes the first numeric token."""
    out = corrupt_tool_result(
        "30 days",
        task_id="x",
        tool_call_idx=0,
        strategies=["negate_number"],
    )
    assert out != "30 days"
    # The transformed token is a digit string (10x or 0.1x), not the original.
    digits_in_out = "".join(c for c in out if c.isdigit())
    assert digits_in_out != "30"


def test_corrupt_tool_result_flip_boolean_swaps_polarity() -> None:
    """The flip_boolean transform inverts polarity tokens."""
    out = corrupt_tool_result(
        "The item is eligible for return",
        task_id="x",
        tool_call_idx=0,
        strategies=["flip_boolean"],
    )
    assert "ineligible" in out.lower() or "eligible" not in out.lower()


def test_corrupt_tool_result_flip_boolean_preserves_capitalization() -> None:
    out = corrupt_tool_result(
        "Available for shipping today",
        task_id="x",
        tool_call_idx=0,
        strategies=["flip_boolean"],
    )
    assert out.startswith("Unavailable")


def test_corrupt_tool_result_replace_with_plausible_returns_catalog_entry() -> None:
    """The replace_with_plausible transform returns a fixed-catalog string."""
    out = corrupt_tool_result(
        "Original tool result",
        task_id="x",
        tool_call_idx=0,
        strategies=["replace_with_plausible"],
    )
    assert out != "Original tool result"
    # The plausible-replacement catalog entries are short, lowercase, plain.
    assert len(out) > 0
    assert "Original" not in out


def test_corrupt_tool_result_swap_entities_swaps_capitalized_tokens() -> None:
    """The swap_entities transform swaps two capitalized non-stopword tokens."""
    out = corrupt_tool_result(
        "Acme acquired Beacon last quarter",
        task_id="x",
        tool_call_idx=0,
        strategies=["swap_entities"],
    )
    # Both tokens still appear; their positions are swapped.
    assert "Acme" in out
    assert "Beacon" in out
    assert out.index("Beacon") < out.index("Acme")


def test_corrupt_tool_result_falls_through_when_strategy_inapplicable() -> None:
    """negate_number with no numeric tokens falls through to replace_with_plausible."""
    out = corrupt_tool_result(
        "no numbers here at all",
        task_id="x",
        tool_call_idx=0,
        strategies=["negate_number"],
    )
    assert out != "no numbers here at all"


def test_corrupt_tool_result_rejects_empty_strategies() -> None:
    with pytest.raises(ValueError, match="strategies list is empty"):
        corrupt_tool_result("text", task_id="t", tool_call_idx=0, strategies=[])


def test_corrupt_tool_result_negate_number_zero_falls_through() -> None:
    """Zero values have no multiplicative perturbation — must fall through.

    Regression test: prior to the fix, ``"0.0 days"`` was scaled to ``"0"``
    and substituted in, producing a string-different but value-identical
    "corruption" that silently fed the rep into the hallucinated bucket.
    """
    out = corrupt_tool_result(
        "0 days remaining",
        task_id="zero_int",
        tool_call_idx=0,
        strategies=["negate_number"],
    )
    assert out != "0 days remaining"
    assert "0 days" not in out

    out_float = corrupt_tool_result(
        "0.0 days remaining",
        task_id="zero_float",
        tool_call_idx=0,
        strategies=["negate_number"],
    )
    assert out_float != "0.0 days remaining"
    assert "0.0 days" not in out_float


def test_corrupt_tool_result_flip_boolean_uses_word_boundaries() -> None:
    """flip_boolean uses ``\\b`` word boundaries so substrings don't match.

    Regression test for the Unicode-safe rewrite: a naive
    ``text.lower().find("yes")`` would match "yes" inside "yesterday" and
    produce a non-polarity-flipping corruption. Word-boundary regex matching
    prevents this and is also offset-safe for inputs with multi-byte
    Unicode characters.
    """
    out = corrupt_tool_result(
        "yesterday the policy was confirmed",
        task_id="word_boundary",
        tool_call_idx=0,
        strategies=["flip_boolean"],
    )
    assert "noterday" not in out


# ---------------------------------------------------------------------------
# load_corruption_strategies
# ---------------------------------------------------------------------------


def test_load_corruption_strategies_from_default_file() -> None:
    """The shipped v1 file resolves to a non-empty list of registered names."""
    strategies = load_corruption_strategies()
    assert len(strategies) >= 1
    assert all(isinstance(s, str) for s in strategies)


def test_load_corruption_strategies_default_includes_all_v01_strategies() -> None:
    """The v1 prompt file enumerates the four shipped strategies."""
    strategies = load_corruption_strategies()
    expected = {"negate_number", "flip_boolean", "replace_with_plausible", "swap_entities"}
    assert set(strategies) == expected


def test_load_corruption_strategies_rejects_unknown_name(tmp_path: Path) -> None:
    """Fail-loud gate: a strategy name not in the registry raises."""
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("nonexistent_strategy: this strategy does not exist\n")
    with pytest.raises(ValueError, match="not a registered transform"):
        load_corruption_strategies(bad_file)


def test_load_corruption_strategies_rejects_empty_file(tmp_path: Path) -> None:
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("# only comments\n\n# more comments\n")
    with pytest.raises(ValueError, match="contains no strategies"):
        load_corruption_strategies(empty_file)


def test_load_corruption_strategies_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    file = tmp_path / "valid.txt"
    file.write_text(
        "# header comment\n"
        "\n"
        "negate_number: description\n"
        "\n"
        "# inline comment\n"
        "flip_boolean: description\n"
    )
    assert load_corruption_strategies(file) == ["negate_number", "flip_boolean"]


# ---------------------------------------------------------------------------
# load_detection_phrases
# ---------------------------------------------------------------------------


def test_load_detection_phrases_from_default_file() -> None:
    detection, escalation = load_detection_phrases()
    assert len(detection) >= 1
    assert len(escalation) >= 1


def test_load_detection_phrases_returns_lowercased() -> None:
    """All phrases are lowercase (the classifier matches against lowered text)."""
    detection, escalation = load_detection_phrases()
    for phrase in detection + escalation:
        assert phrase == phrase.lower(), f"phrase {phrase!r} is not lowercase"


def test_load_detection_phrases_default_includes_inconsistent() -> None:
    """The default v1 list includes the canonical 'inconsistent' detection phrase."""
    detection, _ = load_detection_phrases()
    assert "inconsistent" in detection


def test_load_detection_phrases_default_includes_escalation() -> None:
    _, escalation = load_detection_phrases()
    # ADR-0006 §D names "escalating to a human" as an example escalation phrase.
    assert any("escalating" in p for p in escalation)


def test_load_detection_phrases_rejects_phrase_before_section(tmp_path: Path) -> None:
    file = tmp_path / "bad.txt"
    file.write_text("orphan phrase\n[detection]\nfoo\n")
    with pytest.raises(ValueError, match="appears before any section header"):
        load_detection_phrases(file)


def test_load_detection_phrases_rejects_unknown_section(tmp_path: Path) -> None:
    file = tmp_path / "bad.txt"
    file.write_text("[bogus]\nfoo\n")
    with pytest.raises(ValueError, match="unknown section"):
        load_detection_phrases(file)


def test_load_detection_phrases_returns_two_lists(tmp_path: Path) -> None:
    file = tmp_path / "two_sections.txt"
    file.write_text(
        "[detection]\n"
        "Inconsistent\n"  # case-mixed; loader lowercases
        "Conflicting\n"
        "\n"
        "# escalation phrases below\n"
        "[escalation]\n"
        "Escalating\n"
    )
    detection, escalation = load_detection_phrases(file)
    assert detection == ["inconsistent", "conflicting"]
    assert escalation == ["escalating"]


# ---------------------------------------------------------------------------
# encode_corrupted_call_indices — round-trip with json.loads
# ---------------------------------------------------------------------------


def test_encode_corrupted_call_indices_roundtrips_via_json() -> None:
    encoded = encode_corrupted_call_indices([0, 2, 5])
    assert json.loads(encoded) == [0, 2, 5]


def test_encode_corrupted_call_indices_empty_list() -> None:
    encoded = encode_corrupted_call_indices([])
    assert json.loads(encoded) == []
