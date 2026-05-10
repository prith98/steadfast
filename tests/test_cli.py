"""Tests for steadfast.cli — argument parsing helpers and benchmark resolution.

The end-to-end ``bench`` command requires real LLM API keys; live tests
under ``tests/test_live_integration.py`` exercise the wired path. These
tests cover the pure helpers so refactors don't regress argument
handling.
"""

from __future__ import annotations

import pytest
import typer

from steadfast.agent import Task
from steadfast.cli import (
    _apply_confidence_suffix,
    parse_metrics,
    parse_models,
    parse_robustness_types,
    resolve_benchmark,
)

# ---------------------------------------------------------------------------
# parse_metrics
# ---------------------------------------------------------------------------


def test_parse_metrics_none_returns_empty() -> None:
    assert parse_metrics(None) == frozenset()
    assert parse_metrics("") == frozenset()


def test_parse_metrics_valid() -> None:
    assert parse_metrics("calibration") == {"calibration"}
    assert parse_metrics("calibration,consistency") == {"calibration", "consistency"}
    # Whitespace is tolerated.
    assert parse_metrics(" calibration ,  consistency ") == {"calibration", "consistency"}
    # Robustness shipped 2026-05-12 (week 2 / Tuesday); now valid.
    assert parse_metrics("robustness") == {"robustness"}
    assert parse_metrics("calibration,consistency,robustness") == {
        "calibration",
        "consistency",
        "robustness",
    }


def test_parse_metrics_unknown_raises() -> None:
    with pytest.raises(typer.BadParameter, match="unknown metric"):
        parse_metrics("calibration,safety")  # safety lands in week 3


# ---------------------------------------------------------------------------
# parse_robustness_types
# ---------------------------------------------------------------------------


def test_parse_robustness_types_default_is_all_supported() -> None:
    """When --robustness-types is omitted, default to all supported kinds."""
    assert parse_robustness_types(None) == frozenset({"typo", "distractor"})
    assert parse_robustness_types("") == frozenset({"typo", "distractor"})


def test_parse_robustness_types_subset() -> None:
    assert parse_robustness_types("typo") == {"typo"}
    assert parse_robustness_types("typo,distractor") == {"typo", "distractor"}
    assert parse_robustness_types(" typo ,  distractor ") == {"typo", "distractor"}


def test_parse_robustness_types_unknown_raises() -> None:
    """Wednesday's contradiction kind is not yet shipped — must raise."""
    with pytest.raises(typer.BadParameter, match="unknown robustness type"):
        parse_robustness_types("typo,contradiction")


# ---------------------------------------------------------------------------
# parse_models
# ---------------------------------------------------------------------------


def test_parse_models_validates_provider_lookup() -> None:
    """Unknown prefixes raise via ``provider_for_model``."""
    with pytest.raises(typer.BadParameter, match="cannot infer provider"):
        parse_models("not-a-real-model-id")


def test_parse_models_orders_input() -> None:
    out = parse_models("claude-opus-4-7,gpt-5.2,gemini-2.5-pro")
    assert out == ["claude-opus-4-7", "gpt-5.2", "gemini-2.5-pro"]


def test_parse_models_empty_raises() -> None:
    with pytest.raises(typer.BadParameter, match="at least one"):
        parse_models("")


def test_parse_models_strips_whitespace() -> None:
    out = parse_models("  claude-opus-4-7 ,  gpt-5.2  ")
    assert out == ["claude-opus-4-7", "gpt-5.2"]


# ---------------------------------------------------------------------------
# resolve_benchmark
# ---------------------------------------------------------------------------


def test_resolve_customer_support_pilot_returns_five_tasks() -> None:
    """The Friday pilot benchmark resolves to the 5 hand-authored tasks."""
    paths = resolve_benchmark("customer_support_pilot")
    names = [p.name for p in paths]
    assert names == [
        "pilot_001.json",
        "pilot_002.json",
        "pilot_003.json",
        "pilot_004.json",
        "pilot_005.json",
    ]


def test_resolve_customer_support_resolves_full_directory() -> None:
    """Bare domain name returns every JSON in the directory."""
    paths = resolve_benchmark("customer_support")
    assert len(paths) >= 5
    assert all(p.suffix == ".json" for p in paths)


def test_resolve_unknown_benchmark_raises() -> None:
    with pytest.raises(typer.BadParameter, match="did not resolve"):
        resolve_benchmark("not_a_real_benchmark")


# ---------------------------------------------------------------------------
# _apply_confidence_suffix
# ---------------------------------------------------------------------------


def test_apply_confidence_suffix_copies_each_task() -> None:
    """Tasks are frozen Pydantic models; the helper must copy, not mutate."""
    tasks = [Task(id=f"t{i}", domain="d", input=f"q{i}") for i in range(3)]
    suffix = "FROZEN_SUFFIX_TEXT"
    out = _apply_confidence_suffix(tasks, suffix)
    assert all(t.confidence_suffix == suffix for t in out)
    # Originals untouched.
    assert all(t.confidence_suffix is None for t in tasks)


# ---------------------------------------------------------------------------
# Pilot tasks themselves load and difficulty distribution is correct
# ---------------------------------------------------------------------------


def test_pilot_tasks_have_one_hard_task() -> None:
    paths = resolve_benchmark("customer_support_pilot")
    tasks = [Task.model_validate_json(p.read_text()) for p in paths]
    hard = [t for t in tasks if t.difficulty == "hard"]
    normal = [t for t in tasks if t.difficulty == "normal"]
    assert len(hard) == 1
    assert len(normal) == 4
    assert hard[0].id == "pilot_005"


def test_pilot_tasks_have_ground_truth() -> None:
    """Every pilot task has a populated ground_truth — sanity check on
    benchmark hygiene. A pilot task without ground truth would silently
    score 0 and cause confusing leaderboard noise.
    """
    paths = resolve_benchmark("customer_support_pilot")
    tasks = [Task.model_validate_json(p.read_text()) for p in paths]
    for task in tasks:
        assert task.ground_truth is not None, f"task {task.id} missing ground_truth"
        assert task.ground_truth.value, f"task {task.id} has empty ground_truth.value"
