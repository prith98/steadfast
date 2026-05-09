"""Confidence elicitation — frozen prompt suffix + structured-tail parser.

Per ADR-0005 §B, the harness sets :attr:`Task.confidence_suffix` to the
frozen prompt suffix at ``prompts/confidence_v1.txt`` and the agent
populates :attr:`AgentResponse.confidence` and :attr:`AgentResponse.refused`
by parsing the model's response into a structured ``(answer, confidence,
refused)`` triple.

The prompt requires the model to emit a two-line tail::

    ANSWER: <answer-text or the literal word REFUSE>
    CONFIDENCE: <float in [0, 1]>

Why structured rather than free-form prose: parsing a probability out of
free-form text ("I'm pretty sure", "definitely") would require an LLM
judge to assign a probability — exactly the calibration question we're
trying to measure directly. The structured surface avoids that meta-
calibration step.

Why include ``REFUSE``: refusal calibration (METHODOLOGY §3.4) needs a
boolean per rep. Heuristic regexes on free-form refusal phrasing are
brittle; a second LLM-judge call to classify refusal post-hoc adds
infrastructure cost and yet another calibration risk. Encoding refusal
in the elicitation contract directly is the simplest correct surface.

This module is the single source of truth for the elicitation protocol;
the :class:`SimplePromptingAgent` (and any user-supplied agent that opts
in) consumes :func:`load_confidence_suffix_v1` and
:func:`parse_verbalized_confidence`.
"""

from __future__ import annotations

import re
from typing import Final

from pydantic import BaseModel, ConfigDict

from steadfast._llm_parsing import load_prompt

CONFIDENCE_PROMPT_VERSION: Final[str] = "v1"
CONFIDENCE_PROMPT_FILENAME: Final[str] = "confidence_v1.txt"

# Regex anchors. Case-insensitive and line-anchored — the prompt requires
# uppercase but real model outputs are noisy.
#
# We match the *last* ANSWER and CONFIDENCE labels so prose mentions of
# either word earlier in the response don't shadow the trailing structured
# tail. The label regex matches only the ``LABEL :`` header (up to the
# colon-and-whitespace), not the value — the value is sliced positionally
# between header.end() and the next boundary so multi-line answers survive.
_ANSWER_HEADER = re.compile(r"^[ \t]*ANSWER[ \t]*:[ \t]*", re.IGNORECASE | re.MULTILINE)
_CONFIDENCE_LABEL = re.compile(
    r"^[ \t]*CONFIDENCE[ \t]*:[ \t]*([0-9.,%]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
# Match a REFUSE token surrounded by optional whitespace / punctuation.
# We accept ``REFUSE`` and ``REFUSE.`` and ``"REFUSE"`` etc.
_REFUSE_TOKEN = re.compile(r"^['\" .]*REFUSE['\" .]*$", re.IGNORECASE)


class ParsedConfidence(BaseModel):
    """Structured tail extracted from a model response.

    ``answer`` is the parsed answer text (always populated, even on parse
    failure — falls back to the raw text). ``confidence`` is ``None`` when
    the CONFIDENCE label is missing or out of range; the metric layer skips
    None-confidence reps with a warning per ADR-0002 §A.2 / ADR-0005 §C.
    ``refused`` is ``True`` iff the answer line is the REFUSE token.
    """

    model_config = ConfigDict(frozen=True)

    answer: str
    confidence: float | None
    refused: bool
    parse_ok: bool


def load_confidence_suffix_v1() -> str:
    """Return the frozen ``prompts/confidence_v1.txt`` text.

    Cached on first call — the prompt is read once per process. The
    caching cuts a few microseconds per agent invocation but more
    importantly makes the prompt a stable string that other call sites
    (tracing, manifest writers) can identity-compare against.
    """
    return _CACHED_SUFFIX


def _normalize_confidence_value(raw: str) -> float | None:
    """Parse a CONFIDENCE value into a float in [0, 1], or ``None`` on failure.

    Accepts ``0.85``, ``.85``, ``85%``, ``85.0%``, and **bare integers
    without a decimal point** in [0, 100] (interpreted as percent).
    Comma decimals (``0,85``) are normalized to dots. Out-of-range
    values return ``None`` rather than being clamped — clamping would
    mask "model emitted 1.5" type bugs.

    The "bare integer interpreted as percent" rule is deliberately
    narrow: we accept ``85`` (no decimal point), but ``1.5`` is rejected
    as out-of-range rather than rescaled to ``0.015``. Models that emit
    a malformed float trigger the parser retry path, which is the right
    behavior.
    """
    candidate = raw.strip().rstrip(".")
    candidate = candidate.replace(",", ".")
    is_percent = candidate.endswith("%")
    if is_percent:
        candidate = candidate[:-1].strip()
    has_decimal = "." in candidate
    try:
        value = float(candidate)
    except ValueError:
        return None
    if is_percent:
        value = value / 100.0
    elif value > 1.0 and not has_decimal and value <= 100.0:
        # Bare integer (no decimal point) in (1, 100] → interpret as
        # percent. With a decimal point ("1.5"), keep as-is so the
        # range check below rejects it.
        value = value / 100.0
    if not (0.0 <= value <= 1.0):
        return None
    return value


def parse_verbalized_confidence(text: str) -> ParsedConfidence:
    """Parse a model response into ``(answer, confidence, refused)``.

    Looks for the **last** ``ANSWER:`` and ``CONFIDENCE:`` labels in
    ``text`` (case-insensitive, line-anchored). The "last" rule handles
    models that echo the format header or include the words mid-prose;
    only the trailing structured tail is binding.

    On parse failure (missing CONFIDENCE label or unparseable value),
    returns ``ParsedConfidence(answer=text, confidence=None, refused=False,
    parse_ok=False)``. The caller decides whether to retry (the agent
    layer does, once) or accept the soft failure.
    """
    if not text:
        return ParsedConfidence(answer="", confidence=None, refused=False, parse_ok=False)

    confidence_matches = list(_CONFIDENCE_LABEL.finditer(text))
    if not confidence_matches:
        return ParsedConfidence(answer=text.strip(), confidence=None, refused=False, parse_ok=False)
    confidence_match = confidence_matches[-1]
    confidence_value = _normalize_confidence_value(confidence_match.group(1))
    parse_ok = confidence_value is not None

    # Find the last ANSWER header that appears BEFORE the chosen CONFIDENCE
    # label — filtering by position avoids accidentally pairing an early
    # ANSWER with a late CONFIDENCE when the model emits multiple
    # structured-tail headers (rare but seen when a model "thinks" before
    # answering).
    answer_headers = [
        m for m in _ANSWER_HEADER.finditer(text) if m.start() < confidence_match.start()
    ]
    if not answer_headers:
        # No ANSWER label before the CONFIDENCE — fall back to "everything
        # before the CONFIDENCE line is the answer". This is the natural
        # behavior for models that emit only the CONFIDENCE label after
        # their prose.
        answer_text = text[: confidence_match.start()].rstrip()
    else:
        answer_header = answer_headers[-1]
        # Slice from end-of-header (right after the colon-and-whitespace)
        # to start-of-CONFIDENCE-label. Captures multi-line answers cleanly.
        answer_text = text[answer_header.end() : confidence_match.start()].rstrip()

    refused = bool(_REFUSE_TOKEN.match(answer_text.strip()))

    return ParsedConfidence(
        answer=answer_text,
        confidence=confidence_value,
        refused=refused,
        parse_ok=parse_ok,
    )


# Module-level cache populated at import. Loading the prompt eagerly
# surfaces a missing-prompt error at import time, not at first agent
# invocation deep inside an asyncio.gather call.
_CACHED_SUFFIX: Final[str] = load_prompt(CONFIDENCE_PROMPT_FILENAME)


__all__ = [
    "CONFIDENCE_PROMPT_FILENAME",
    "CONFIDENCE_PROMPT_VERSION",
    "ParsedConfidence",
    "load_confidence_suffix_v1",
    "parse_verbalized_confidence",
]
