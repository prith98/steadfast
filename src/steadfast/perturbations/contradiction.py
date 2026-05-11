"""Contradiction perturbation — corrupt tool outputs at probability 0.3 per call.

Per ``docs/METHODOLOGY.md`` §2.3 and ADR-0006 §D: when an agent calls a tool,
this perturbation may replace the tool's response with a corrupted variant,
seeded deterministically per (task_id, tool_call_idx) via ADR-0006 §B's
``:tool{idx}`` extension. The downstream metric measures whether the agent
detects the contradiction, retries/escalates, or hallucinates a coherent
answer despite the corrupted input.

**The perturbation does NOT live inside the harness.** Steadfast's
:class:`~steadfast.agent.Agent` ABC makes tool execution the agent's
responsibility — :meth:`Agent.arun` returns a complete
:class:`~steadfast.agent.AgentResponse` after the agent has run its own
tool loop. There is no harness-side place to "intercept" a tool call.
This module therefore exposes *primitives* (a per-call coin flip and a
deterministic corruption transform) that a tool-using agent (or fixture,
or future v0.2 LangGraph adapter) wires into its own tool execution.

**Corruption signaling.** Agents that wire this perturbation set
``response.metadata[CORRUPTED_CALLS_METADATA_KEY]`` to a JSON-encoded
``list[int]`` of indices into ``response.trajectory`` whose
``ToolCall.result`` was replaced. The metric layer reads this convention
to build the ``corrupted_calls`` argument for
:func:`~steadfast.metrics.robustness.classify_contradiction_response`.
The metadata bag's value type is constrained to
:class:`~steadfast.agent.MetadataValue` (``str | int | float | bool``)
per ADR-0002 §A.3 / Q5, so the indices ride as a JSON string.

**Why programmatic strategies, not LLM-driven.** ADR-0006 §D rejects a
fourth infrastructure-LLM surface for the *classifier*; the same
rationale (cost, bias, ensemble obligation) applies symmetrically to
the corruption side. The strategies in
``prompts/contradiction_corruptions_v1.txt`` are deterministic Python
transforms; v0.2 may introduce an ``LLMCorruptor`` that uses each
strategy's description as an LLM prompt.

References:

* ADR-0006 §B (per-call seed), §D (label classifier + N/A path).
* ``prompts/contradiction_corruptions_v1.txt`` — frozen strategy registry.
* ``prompts/contradiction_detection_phrases_v1.txt`` — frozen phrase lists
  consumed by the classifier in :mod:`steadfast.metrics.robustness`.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable
from pathlib import Path
from typing import Final

from steadfast.perturbations._seed import derive_seed

# Public methodology constants ----------------------------------------------

DEFAULT_CORRUPTION_PROBABILITY: Final[float] = 0.3
"""Per-call probability of corruption per METHODOLOGY §2.3."""

CORRUPTED_CALLS_METADATA_KEY: Final[str] = "steadfast.contradiction.corrupted_call_indices"
"""``AgentResponse.metadata`` key under which the agent reports corruption.

The value is a JSON-encoded ``list[int]`` of indices into
``response.trajectory`` whose ``ToolCall.result`` was replaced by the
contradiction perturbation. The string-encoding is required because
``MetadataValue`` is the scalar union (no nested ``list``) per ADR-0002 §A.3.
"""

# Decorrelation salts so the should-corrupt coin, the strategy choice, and
# the inner-transform RNG don't share a draw stream — same precedent the
# typo perturbation set with (_POS_SEED_SALT, _SUB_SEED_SALT). Without
# decorrelation, a change in one consumer's draw count would shift another
# consumer's outputs.
_SHOULD_CORRUPT_SALT: Final[int] = 0xCA1F_00D5_C0FF_EE01
_STRATEGY_SALT: Final[int] = 0xDEAD_BEEF_1234_5678
_TRANSFORM_SALT: Final[int] = 0x1337_CAFE_BABE_0042

# Default prompt-file paths. Both live under ``<repo>/prompts/`` per the
# existing ``prompts/*_v1.*`` versioning convention.
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_DEFAULT_CORRUPTION_PROMPT: Final[Path] = (
    _REPO_ROOT / "prompts" / "contradiction_corruptions_v1.txt"
)
_DEFAULT_DETECTION_PROMPT: Final[Path] = (
    _REPO_ROOT / "prompts" / "contradiction_detection_phrases_v1.txt"
)


# Programmatic corruption strategies ----------------------------------------

_NUMBER_RE: Final[re.Pattern[str]] = re.compile(r"\b\d+(?:\.\d+)?\b")


def _negate_number(text: str, rng: random.Random) -> str:
    """Replace the first numeric token in ``text`` with a 10x-off variant.

    "30 days" -> "300 days" or "3 days"; "$40" -> "$400" or "$4". When no
    numeric token is found OR the original token is zero (which has no
    multiplicative perturbation), falls through to
    :func:`_replace_with_plausible` so the strategy always produces a
    corrupted string (never a no-op).
    """
    match = _NUMBER_RE.search(text)
    if match is None:
        return _replace_with_plausible(text, rng)
    original = match.group()
    try:
        original_value = float(original)
    except ValueError:
        return _replace_with_plausible(text, rng)
    if original_value == 0:
        # 0 * 10 == 0 and 0 / 10 == 0 — any multiplicative perturbation is
        # vacuous on zero. Fall through rather than produce a fake "change".
        return _replace_with_plausible(text, rng)
    multiplier_up = rng.random() < 0.5
    try:
        if "." in original:
            value = original_value * (10.0 if multiplier_up else 0.1)
            new_token = f"{value:g}"
        else:
            scaled = int(original) * 10 if multiplier_up else int(original) // 10
            new_token = str(scaled)
    except (ValueError, OverflowError):
        return _replace_with_plausible(text, rng)
    if new_token == original:
        # int(N) // 10 == 0 when N < 10; the literal-string equality check
        # also catches float precision quirks like "1.0" * 10 -> "10". Fall
        # through to the plausible-replacement so the corruption is
        # guaranteed-different rather than silently no-op.
        return _replace_with_plausible(text, rng)
    return text[: match.start()] + new_token + text[match.end() :]


_BOOLEAN_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("yes", "no"),
    ("true", "false"),
    ("available", "unavailable"),
    ("eligible", "ineligible"),
    ("enabled", "disabled"),
    ("allowed", "disallowed"),
    ("approved", "denied"),
    ("active", "inactive"),
)


def _flip_boolean(text: str, rng: random.Random) -> str:
    """Swap polarity tokens; preserves case (uppercase / capitalized / lower).

    Walks ``_BOOLEAN_PAIRS`` in registry order and, for the first pair whose
    either form appears in ``text`` (case-insensitive search), swaps to the
    other form. Falls through to :func:`_replace_with_plausible` when no
    boolean-like token is found.

    Implementation note: uses :func:`re.search` with the ``IGNORECASE`` flag
    so the match is byte-offset-correct even for inputs containing Unicode
    characters that change length when lowercased (e.g., ``"Ⅱ".lower() ==
    "ii"``). Doing ``text.lower().find(...)`` and indexing back into ``text``
    would miscount in those cases.
    """
    for a, b in _BOOLEAN_PAIRS:
        for original, replacement in ((a, b), (b, a)):
            match = re.search(rf"\b{re.escape(original)}\b", text, flags=re.IGNORECASE)
            if match is None:
                continue
            actual = match.group()
            if actual.isupper():
                cased = replacement.upper()
            elif actual[:1].isupper():
                cased = replacement.capitalize()
            else:
                cased = replacement
            return text[: match.start()] + cased + text[match.end() :]
    return _replace_with_plausible(text, rng)


_PLAUSIBLE_REPLACEMENTS: Final[tuple[str, ...]] = (
    "no record found in system",
    "policy not currently active",
    "data temporarily unavailable",
    "the requested resource has been archived",
    "no matching entries returned by the lookup",
    "this query exceeds the per-call rate limit",
)


def _replace_with_plausible(text: str, rng: random.Random) -> str:
    """Return a fixed plausible-but-wrong record, deterministically chosen.

    ``text`` is intentionally ignored: the strategy is the "obviously wrong
    record" corruption, the catalog is the source of truth.
    """
    del text  # marker: we replace the entire result
    return rng.choice(_PLAUSIBLE_REPLACEMENTS)


_CAPITALIZED_RE: Final[re.Pattern[str]] = re.compile(r"\b[A-Z][A-Za-z]{2,}\b")
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "The",
        "And",
        "Or",
        "But",
        "For",
        "Nor",
        "Yet",
        "With",
        "From",
        "Into",
        "Onto",
        "Upon",
        "About",
        "After",
        "Before",
        "Between",
        "During",
        "Against",
    }
)


def _swap_entities(text: str, rng: random.Random) -> str:
    """Swap two capitalized non-stopword tokens in ``text``.

    Falls through to :func:`_replace_with_plausible` when ``text`` has fewer
    than two distinct capitalized non-stopword tokens (the strategy can't
    produce a meaningful swap).
    """
    matches = [m for m in _CAPITALIZED_RE.finditer(text) if m.group() not in _STOPWORDS]
    if len(matches) < 2:
        return _replace_with_plausible(text, rng)
    a_idx, b_idx = sorted(rng.sample(range(len(matches)), 2))
    ma, mb = matches[a_idx], matches[b_idx]
    return (
        text[: ma.start()]
        + mb.group()
        + text[ma.end() : mb.start()]
        + ma.group()
        + text[mb.end() :]
    )


# Strategy name -> transform function. The registry is the source of truth
# for which strategy names are valid; ``load_corruption_strategies`` raises
# on any prompt-file name not present here.
_STRATEGIES: Final[dict[str, Callable[[str, random.Random], str]]] = {
    "negate_number": _negate_number,
    "flip_boolean": _flip_boolean,
    "replace_with_plausible": _replace_with_plausible,
    "swap_entities": _swap_entities,
}


# Loaders --------------------------------------------------------------------


def load_corruption_strategies(path: str | Path | None = None) -> list[str]:
    """Load and validate strategy names from the frozen prompt file.

    Skips empty / comment lines. Each remaining line is parsed as
    ``"<name>: <description>"`` (description is documentation only — the
    runtime uses ``name``). Raises :class:`ValueError` if any name is not
    in the registered transform table; the implementation in this module
    is the authoritative list.

    Returns the list in file order (the order matters because the runtime
    picks ``strategies[seed % len(strategies)]`` — reordering the file
    silently invalidates seeded reproducibility).
    """
    p = Path(path) if path is not None else _DEFAULT_CORRUPTION_PROMPT
    names: list[str] = []
    for line_no, raw_line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split(":", 1)[0].strip()
        if name not in _STRATEGIES:
            raise ValueError(
                f"corruption strategy {name!r} on line {line_no} of {p} is not a "
                f"registered transform; valid: {sorted(_STRATEGIES)}. Add the "
                "implementation in src/steadfast/perturbations/contradiction.py "
                "before referencing the strategy in the prompt file."
            )
        names.append(name)
    if not names:
        raise ValueError(f"corruption-strategies file {p} contains no strategies")
    return names


def load_detection_phrases(path: str | Path | None = None) -> tuple[list[str], list[str]]:
    """Return ``(detection_phrases, escalation_phrases)`` from the frozen file.

    File format: two sections marked ``[detection]`` and ``[escalation]``;
    one phrase per non-empty / non-comment line; whitespace is trimmed.
    All phrases are lowercased on load (the classifier uses lowercase
    substring matching).

    Raises :class:`ValueError` on unknown section names or phrases that
    appear before any section header (fail-loud per the prompt-file
    versioning discipline).
    """
    p = Path(path) if path is not None else _DEFAULT_DETECTION_PROMPT
    sections: dict[str, list[str]] = {"detection": [], "escalation": []}
    current: str | None = None
    for line_no, raw_line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section_name = line[1:-1].strip().lower()
            if section_name not in sections:
                raise ValueError(
                    f"unknown section {section_name!r} on line {line_no} of {p}; "
                    "expected [detection] or [escalation]"
                )
            current = section_name
            continue
        if current is None:
            raise ValueError(
                f"phrase {line!r} on line {line_no} of {p} appears before any "
                "section header; add a [detection] or [escalation] header first"
            )
        sections[current].append(line.lower())
    return sections["detection"], sections["escalation"]


# Runtime primitives ---------------------------------------------------------


def should_corrupt(
    *,
    task_id: str,
    tool_call_idx: int,
    probability: float = DEFAULT_CORRUPTION_PROBABILITY,
) -> bool:
    """Deterministic Bernoulli draw for whether to corrupt this tool call.

    Same ``(task_id, tool_call_idx)`` always yields the same result. Across
    distinct ``tool_call_idx`` values, the proportion of ``True`` returns
    converges to ``probability`` at large N (verified by
    :mod:`tests.test_perturbations_contradiction`).

    Parameters
    ----------
    task_id:
        The :attr:`~steadfast.agent.Task.id` whose tool call is being
        decided. Different tasks get different corruption patterns.
    tool_call_idx:
        Zero-based index of the tool call in the agent's trajectory for
        this rep. Reordering tool calls on the agent side shifts the
        corruption pattern — that's intentional, since tool-call order
        is part of the agent's behavior under measurement.
    probability:
        Per-call corruption probability. Default per METHODOLOGY §2.3.

    Raises
    ------
    ValueError
        If ``probability`` is not in ``[0, 1]``.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1]; got {probability}")
    seed = derive_seed(task_id, "contradiction", tool_call_idx=tool_call_idx)
    rng = random.Random(seed ^ _SHOULD_CORRUPT_SALT)
    return rng.random() < probability


def corrupt_tool_result(
    result: str,
    *,
    task_id: str,
    tool_call_idx: int,
    strategies: list[str] | None = None,
) -> str:
    """Apply a deterministically-selected corruption strategy to ``result``.

    Strategy choice: ``strategies[seed_idx % len(strategies)]`` where
    ``seed_idx`` derives from ``(task_id, tool_call_idx)`` via the same
    ADR-0006 §B seed helper (with a decorrelation salt so the choice
    stream is independent of :func:`should_corrupt`'s coin stream).

    Parameters
    ----------
    result:
        The original tool response. Strategies that depend on the content
        (``negate_number``, ``flip_boolean``, ``swap_entities``) operate on
        this string; ``replace_with_plausible`` ignores it.
    task_id, tool_call_idx:
        See :func:`should_corrupt`. The same seed inputs produce the same
        corrupted output.
    strategies:
        Optional list of strategy names. Defaults to
        :func:`load_corruption_strategies` (the frozen prompt-file list).

    Raises
    ------
    ValueError
        If ``strategies`` is empty.
    """
    if strategies is None:
        strategies = load_corruption_strategies()
    if not strategies:
        raise ValueError("strategies list is empty; nothing to corrupt with")
    seed = derive_seed(task_id, "contradiction", tool_call_idx=tool_call_idx)
    choice_rng = random.Random(seed ^ _STRATEGY_SALT)
    name = strategies[choice_rng.randrange(len(strategies))]
    transform = _STRATEGIES[name]
    transform_rng = random.Random(seed ^ _TRANSFORM_SALT)
    return transform(result, transform_rng)


def encode_corrupted_call_indices(indices: list[int]) -> str:
    """JSON-encode a list of indices for the metadata convention.

    Helper so agents writing the metadata key don't have to import ``json``
    just for the encoding format; matches the decoding rules used by the
    metric layer's ``_extract_corrupted_calls``.
    """
    return json.dumps(list(indices))


__all__ = [
    "CORRUPTED_CALLS_METADATA_KEY",
    "DEFAULT_CORRUPTION_PROBABILITY",
    "corrupt_tool_result",
    "encode_corrupted_call_indices",
    "load_corruption_strategies",
    "load_detection_phrases",
    "should_corrupt",
]
