"""Deterministic exact-match judge with canonicalization.

Per ADR-0003 §B.3, canonicalization rules are applied in order:

1. NFKC unicode normalization.
2. Casefold (unicode-correct lowercase).
3. ASCII hyphens between word characters → single space (clarification
   fix 2026-05-11; see CHANGELOG / `docs/WEEK_2.md` §"Open methodological
   choices" §O.1). The original substring-containment intent was
   hyphenation-insensitive; treating "30-day" and "30 day" as the same
   canonical form makes the judge match the way a reader expects.
4. Collapse runs of whitespace to a single space.
5. Strip leading/trailing whitespace.
6. Strip trailing punctuation in ``{".", ",", "!", "?", ";", ":"}``.

After canonicalization, ``canonical(ground_truth.value)`` must appear as
a *substring* of ``canonical(response.answer)`` to pass — strict equality
would reject every realistic free-form answer (e.g., "The return window
is 30 days" against ground truth "30 days"). The rubric judge handles
harder semantic cases.
"""

from __future__ import annotations

import re
import unicodedata

from steadfast.agent import AgentResponse, Task
from steadfast.judges.base import Judge, Verdict

_TRAILING_PUNCT: frozenset[str] = frozenset({".", ",", "!", "?", ";", ":"})
_WHITESPACE_RUN = re.compile(r"\s+")
# Hyphen between two word characters — `\w` is Unicode-aware in Python 3,
# so this also catches hyphens between Unicode letters / digits after the
# NFKC + casefold passes have run. Lookbehind / lookahead means the hyphen
# itself is replaced without consuming the surrounding chars, so a chain
# like "state-of-the-art" gets every internal hyphen replaced in a single
# `sub` pass.
_INTERNAL_HYPHEN = re.compile(r"(?<=\w)-(?=\w)")


def canonicalize(s: str) -> str:
    """Apply ADR-0003 §B.3 canonicalization rules to ``s``.

    Exposed at module scope so tests (and a future debug-report layer)
    can verify the canonical form independently of judge dispatch.

    ``casefold`` rather than ``.lower()`` because casefold handles
    German eszett (ß → ss) and other unicode equivalents that ``.lower``
    leaves alone.
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.casefold()
    s = _INTERNAL_HYPHEN.sub(" ", s)
    s = _WHITESPACE_RUN.sub(" ", s)
    s = s.strip()
    while s and s[-1] in _TRAILING_PUNCT:
        s = s[:-1].rstrip()
    return s


class ExactMatchJudge(Judge):
    """Substring-containment judge over canonicalized ground-truth and answer.

    Used when ``task.judge == "exact_match"`` and the task has a
    :class:`~steadfast.agent.GroundTruth` with ``kind="exact"``. Raises
    :class:`ValueError` if the task is malformed for this judge — surfacing
    the misconfiguration rather than silently scoring 0.
    """

    async def ajudge(self, task: Task, response: AgentResponse) -> Verdict:
        if task.ground_truth is None or task.ground_truth.kind != "exact":
            raise ValueError(
                f"ExactMatchJudge requires task.ground_truth.kind='exact'; "
                f"task {task.id!r} has ground_truth={task.ground_truth!r}"
            )

        canonical_gt = canonicalize(task.ground_truth.value)
        canonical_answer = canonicalize(response.answer)
        passed = bool(canonical_gt) and canonical_gt in canonical_answer

        reason = (
            f"canonical ground_truth={canonical_gt!r} "
            f"{'in' if passed else 'NOT in'} canonical answer={canonical_answer!r}"
        )
        return Verdict(score=1.0 if passed else 0.0, passed=passed, reason=reason)
