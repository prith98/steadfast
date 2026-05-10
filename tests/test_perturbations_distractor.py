"""Tests for steadfast.perturbations.distractor — bank loader, gating, apply.

Per METHODOLOGY §2.2 and ADR-0006 §C: deterministic snippet selection
from a frozen per-domain bank with a 200-800 token gate; bank-exhaustion
is a loud failure mode.
"""

from __future__ import annotations

import json

import pytest

from steadfast.perturbations.distractor import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    DistractorBank,
    DistractorBankExhaustedError,
    DistractorSnippet,
    apply_distractor,
    load_distractor_bank,
    perturb_distractor,
    pick_distractor,
    write_distractor_bank,
)


def _snippet(*, snippet_id: str, tokens: int, text: str | None = None) -> DistractorSnippet:
    return DistractorSnippet(
        id=snippet_id,
        text=text if text is not None else f"[snippet {snippet_id} body]",
        tokens=tokens,
    )


def _bank(snippets: list[DistractorSnippet], *, domain: str = "test_domain") -> DistractorBank:
    return DistractorBank(
        domain=domain,
        encoding="cl100k_base",
        prompt_version="v1",
        review_status="reviewed",
        snippets=snippets,
    )


# ---------------------------------------------------------------------------
# pick_distractor
# ---------------------------------------------------------------------------


def test_pick_returns_snippet_at_seed_modulo() -> None:
    """Seed=2 picks index 2 when in range; small bank where every snippet fits."""
    bank = _bank([_snippet(snippet_id=f"id{i}", tokens=400) for i in range(5)])
    chosen = pick_distractor(bank, seed=2)
    assert chosen.id == "id2"


def test_pick_walks_to_next_in_range_when_first_too_short() -> None:
    """Snippet at seed%len is below min_tokens → walk forward until in-range."""
    bank = _bank(
        [
            _snippet(snippet_id="a", tokens=50),  # too short
            _snippet(snippet_id="b", tokens=100),  # too short
            _snippet(snippet_id="c", tokens=400),  # in range
            _snippet(snippet_id="d", tokens=500),  # in range
        ]
    )
    # seed % 4 == 0 → start at 'a'; walk to 'b' (still short), then 'c' (fit).
    chosen = pick_distractor(bank, seed=0)
    assert chosen.id == "c"


def test_pick_walks_when_first_too_long() -> None:
    bank = _bank(
        [
            _snippet(snippet_id="huge1", tokens=900),
            _snippet(snippet_id="huge2", tokens=1500),
            _snippet(snippet_id="ok", tokens=300),
        ]
    )
    chosen = pick_distractor(bank, seed=0)
    assert chosen.id == "ok"


def test_pick_wraps_around_bank() -> None:
    """Seed past the in-range snippets must wrap modulo bank length."""
    bank = _bank(
        [
            _snippet(snippet_id="ok", tokens=400),  # idx 0 — fits
            _snippet(snippet_id="big", tokens=900),  # idx 1 — too long
        ]
    )
    # seed % 2 == 1 → start at 'big' (out of range); walk forward, wrap to 'ok'.
    chosen = pick_distractor(bank, seed=1)
    assert chosen.id == "ok"


def test_pick_is_deterministic() -> None:
    bank = _bank([_snippet(snippet_id=f"s{i}", tokens=400) for i in range(7)])
    a = pick_distractor(bank, seed=12345)
    b = pick_distractor(bank, seed=12345)
    assert a.id == b.id


def test_pick_raises_when_bank_empty() -> None:
    bank = _bank([])
    with pytest.raises(DistractorBankExhaustedError, match="empty"):
        pick_distractor(bank, seed=0)


def test_pick_raises_when_no_snippet_fits() -> None:
    bank = _bank(
        [
            _snippet(snippet_id="too-short", tokens=100),
            _snippet(snippet_id="too-long", tokens=1000),
        ]
    )
    with pytest.raises(DistractorBankExhaustedError, match=r"no snippet in \[200, 800\]"):
        pick_distractor(bank, seed=0)


def test_pick_raises_on_invalid_token_range() -> None:
    bank = _bank([_snippet(snippet_id="x", tokens=400)])
    with pytest.raises(ValueError, match="min_tokens must be > 0"):
        pick_distractor(bank, seed=0, min_tokens=0)
    with pytest.raises(ValueError, match=r"max_tokens \(\d+\) must be >= min_tokens"):
        pick_distractor(bank, seed=0, min_tokens=500, max_tokens=200)


# ---------------------------------------------------------------------------
# apply_distractor — frozen delimiter contract
# ---------------------------------------------------------------------------


def test_apply_uses_frozen_delimiters() -> None:
    """The on-disk delimiter format is part of the metric contract."""
    text = "What is the return window?"
    distractor = _snippet(snippet_id="x", tokens=300, text="Background context here.")
    result = apply_distractor(text, distractor)
    assert "--- background reading ---" in result
    assert "--- task ---" in result
    assert distractor.text in result
    assert text in result
    # Order: background fence → snippet → task fence → original task.
    bg_idx = result.index("--- background reading ---")
    task_idx = result.index("--- task ---")
    assert bg_idx < task_idx
    assert result.index(distractor.text) < task_idx
    assert result.index(text) > task_idx


def test_apply_preserves_original_text() -> None:
    """The original task input must appear verbatim after the delimiter."""
    text = "Original task text — verbatim."
    distractor = _snippet(snippet_id="y", tokens=300)
    result = apply_distractor(text, distractor)
    # Suffix: original text appears at the end.
    assert result.endswith(text)


# ---------------------------------------------------------------------------
# perturb_distractor — picks + applies in one call
# ---------------------------------------------------------------------------


def test_perturb_distractor_uses_seed_to_pick() -> None:
    bank = _bank(
        [
            _snippet(snippet_id="alpha", tokens=400, text="ALPHA-BODY"),
            _snippet(snippet_id="beta", tokens=400, text="BETA-BODY"),
        ]
    )
    a = perturb_distractor("orig", bank=bank, seed=0)
    b = perturb_distractor("orig", bank=bank, seed=1)
    assert "ALPHA-BODY" in a
    assert "BETA-BODY" in b


# ---------------------------------------------------------------------------
# Bank file IO round-trip
# ---------------------------------------------------------------------------


def test_bank_round_trip_via_disk(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bank = _bank(
        [
            _snippet(snippet_id="a", tokens=300),
            _snippet(snippet_id="b", tokens=500),
        ]
    )
    path = tmp_path / "distractors_v1.json"
    write_distractor_bank(bank, path)
    reloaded = load_distractor_bank(path)
    assert reloaded.domain == bank.domain
    assert reloaded.encoding == bank.encoding
    assert [s.id for s in reloaded.snippets] == ["a", "b"]
    assert reloaded.snippets[0].tokens == 300


def test_load_missing_bank_raises_with_helpful_message(tmp_path) -> None:  # type: ignore[no-untyped-def]
    missing = tmp_path / "no_such_bank.json"
    with pytest.raises(FileNotFoundError, match="distractor bank not found"):
        load_distractor_bank(missing)


def test_load_rejects_draft_review_status(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """ADR-0006 §C fail-loud gate: a draft bank must not load silently.

    The generator script writes ``review_status="draft"`` and prints
    instructions for the operator to flip it after auditing. If the
    operator skips that step, ``load_distractor_bank`` must refuse the
    load — this test pins the contract.
    """
    draft_bank = DistractorBank(
        domain="test_domain",
        encoding="cl100k_base",
        prompt_version="v1",
        review_status="draft",
        snippets=[_snippet(snippet_id="x", tokens=400)],
    )
    path = tmp_path / "distractors_v1.json"
    write_distractor_bank(draft_bank, path)
    with pytest.raises(ValueError, match="review_status='draft'"):
        load_distractor_bank(path)


def test_bank_json_layout_is_human_readable(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The on-disk JSON should be indented and have non-ASCII passthrough.

    A reviewer reads this file by hand to look for ground-truth
    contradictions; minified or escaped JSON would make that impractical.
    """
    bank = _bank([_snippet(snippet_id="x", tokens=300, text="Café résumé naïve.")])
    path = tmp_path / "distractors_v1.json"
    write_distractor_bank(bank, path)
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text)
    assert "\n  " in text  # indented
    assert payload["snippets"][0]["text"] == "Café résumé naïve."  # not escape-encoded


# ---------------------------------------------------------------------------
# Methodology defaults
# ---------------------------------------------------------------------------


def test_methodology_defaults() -> None:
    """METHODOLOGY §2.2 specifies the 200-800 token range."""
    assert DEFAULT_MIN_TOKENS == 200
    assert DEFAULT_MAX_TOKENS == 800
