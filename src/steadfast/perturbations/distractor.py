"""Distractor perturbation — prepend topically-adjacent but answer-irrelevant text.

Per ``docs/METHODOLOGY.md`` §2.2 and ADR-0006 §C: a curated per-domain
bank of LLM-generated, human-reviewed snippets is committed at
``benchmarks/<domain>/distractors_v1.json``. At measurement time the
metric picks one snippet per (task, rep) deterministically from the seed
and prepends it to the task input with a clear delimiter so the agent
can in principle distinguish background from task — a robust agent
ignores the distractor; a brittle one is led astray by it.

Snippet selection is deterministic via ``seed % len(bank)`` plus
walk-until-fit on the 200-800 token gate (METHODOLOGY §2.2). If the
walk wraps the entire bank without finding a fitting snippet, the
function raises — better a loud failure than silently shipping
out-of-spec distractors that bias the metric.

References:

* ADR-0006 §C — bank generation / curation / freezing.
* METHODOLOGY §2.2 — perturbation contract (200-800 tokens; one snippet
  per (task, rep); deterministic selection).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MIN_TOKENS: Final[int] = 200
DEFAULT_MAX_TOKENS: Final[int] = 800
DEFAULT_ENCODING: Final[str] = "cl100k_base"

# Frozen delimiter format per ``docs/WEEK_2.md`` §Tuesday item 3 / ADR-0006 §C.
# Both fences are visible to the agent so a robust agent has a fighting
# chance to identify the boundary; brittle agents that ignore the fences
# show up as the failure mode the metric is designed to detect.
_DISTRACTOR_DELIMITER_TEMPLATE: Final[str] = (
    "--- background reading ---\n{distractor}\n--- task ---\n{original}"
)


class DistractorSnippet(BaseModel):
    """One snippet in the per-domain distractor bank.

    ``tokens`` is precomputed at bank-generation time so the runtime
    selection path doesn't tokenize on the hot path. ``id`` is a stable
    short identifier (SHA-256-prefix of ``text``) — useful for
    diagnostics ("which snippet was prepended on rep 7?") without
    serializing the full text into the run manifest.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    text: str
    tokens: int = Field(gt=0)


class DistractorBank(BaseModel):
    """The frozen, content-addressed distractor bank for one domain.

    The on-disk JSON file at
    ``benchmarks/<domain>/distractors_v1.json`` is a Pydantic round-trip
    of this model. The ``_v1`` suffix is the version contract (ADR-0006
    §C); regenerating the bank creates ``_v2`` and triggers a metric-
    version event per METHODOLOGY §"Versioning".
    """

    model_config = ConfigDict(frozen=True)

    domain: str
    encoding: str = DEFAULT_ENCODING
    prompt_version: str = "v1"
    generator_model: str | None = None
    generated_at: str | None = None
    # Two-state gate per ADR-0006 §C: the bank generator writes "draft"
    # and ``load_distractor_bank`` refuses to load anything else. The
    # operator flips this to "reviewed" only after auditing snippets for
    # ground-truth contradictions vs. the domain's tasks.
    review_status: Literal["draft", "reviewed"] = "draft"
    snippets: list[DistractorSnippet]


class DistractorBankExhaustedError(RuntimeError):
    """Raised when no snippet in the bank fits the requested token range.

    Either the bank is too small (regenerate with a larger ``n``) or the
    requested range is too narrow (widen it, or audit the bank's token
    distribution). Either way, this is a bank-quality issue and should
    surface loudly rather than be silently absorbed by the metric.
    """


def load_distractor_bank(path: str | Path) -> DistractorBank:
    """Load a frozen distractor bank from disk.

    Raises :class:`FileNotFoundError` if the file does not exist,
    :class:`pydantic.ValidationError` if the file is malformed, and
    :class:`ValueError` if ``review_status != "reviewed"`` — the
    fail-loud gate from ADR-0006 §C. An operator who renames a draft
    without flipping the field gets a hard error rather than a silent
    benchmark run against unaudited LLM output.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"distractor bank not found at {p} — generate one with "
            "`scripts/generate_distractor_bank.py` and review per ADR-0006 §C "
            "before committing."
        )
    bank = DistractorBank.model_validate_json(p.read_text(encoding="utf-8"))
    if bank.review_status != "reviewed":
        raise ValueError(
            f"distractor bank at {p} has review_status={bank.review_status!r}; "
            "edit it to 'reviewed' (after auditing snippets per ADR-0006 §C) "
            "before using the bank in a benchmark run."
        )
    return bank


def pick_distractor(
    bank: DistractorBank,
    *,
    seed: int,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> DistractorSnippet:
    """Deterministically pick one snippet from ``bank``.

    Per METHODOLOGY §2.2: start at index ``seed % len(bank)``; if the
    snippet at that index is outside the [min_tokens, max_tokens] range,
    walk forward (modulo bank size) until a fit is found. If the walk
    returns to the starting index, the bank has no snippet that fits —
    raise :class:`DistractorBankExhaustedError`.

    The walk preserves determinism: two calls with the same ``seed`` and
    ``bank`` always pick the same snippet.
    """
    if min_tokens <= 0:
        raise ValueError(f"min_tokens must be > 0; got {min_tokens}")
    if max_tokens < min_tokens:
        raise ValueError(f"max_tokens ({max_tokens}) must be >= min_tokens ({min_tokens})")
    n = len(bank.snippets)
    if n == 0:
        raise DistractorBankExhaustedError(f"distractor bank for domain={bank.domain!r} is empty")

    start_idx = seed % n
    for offset in range(n):
        candidate = bank.snippets[(start_idx + offset) % n]
        if min_tokens <= candidate.tokens <= max_tokens:
            return candidate

    # Walked the whole bank without finding a fit. Surface bank-quality
    # issue loudly — this means either the bank was generated outside the
    # target range or the metric's range is wrong.
    token_range = (
        f"min={min(s.tokens for s in bank.snippets)}, max={max(s.tokens for s in bank.snippets)}"
    )
    raise DistractorBankExhaustedError(
        f"distractor bank for domain={bank.domain!r} has no snippet in "
        f"[{min_tokens}, {max_tokens}] tokens (bank's actual range: {token_range}). "
        "Regenerate the bank or widen the requested range."
    )


def apply_distractor(text: str, distractor: DistractorSnippet) -> str:
    """Prepend ``distractor.text`` to ``text`` with the frozen delimiter."""
    return _DISTRACTOR_DELIMITER_TEMPLATE.format(distractor=distractor.text, original=text)


def perturb_distractor(
    text: str,
    *,
    bank: DistractorBank,
    seed: int,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Pick a snippet from ``bank`` per ``seed`` and prepend it to ``text``.

    Convenience composition of :func:`pick_distractor` and
    :func:`apply_distractor`. The metric layer uses this for a single
    perturbed-input string; the diagnostic surface (which snippet was
    used) is recovered by calling :func:`pick_distractor` directly.
    """
    chosen = pick_distractor(bank, seed=seed, min_tokens=min_tokens, max_tokens=max_tokens)
    return apply_distractor(text, chosen)


def write_distractor_bank(bank: DistractorBank, path: str | Path) -> None:
    """Write a bank JSON to disk with a stable, human-readable layout.

    Used by ``scripts/generate_distractor_bank.py`` to write the draft
    artifact and (after operator rename) the committed
    ``distractors_v1.json``. Encoding kept explicit (UTF-8) so non-ASCII
    snippets round-trip on Windows.
    """
    payload = bank.model_dump(mode="json")
    Path(path).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "DEFAULT_ENCODING",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MIN_TOKENS",
    "DistractorBank",
    "DistractorBankExhaustedError",
    "DistractorSnippet",
    "apply_distractor",
    "load_distractor_bank",
    "perturb_distractor",
    "pick_distractor",
    "write_distractor_bank",
]
