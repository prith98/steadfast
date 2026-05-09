"""Shared helpers for LLM JSON parsing and frozen-prompt loading.

LLMs frequently wrap structured output in triple-backtick code fences,
emit prose alongside the JSON object, or otherwise diverge from
"JSON-only" prompting. The helpers here centralize the resilience
patterns so judges, validators, and perturbation generators can stay
focused on their domain logic.

Private module — callers should not depend on the parsing semantics
beyond what these helpers expose.
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# Greedy match for an outermost ``{...}`` JSON object. Used to extract a
# JSON payload from LLM output that wraps it in surrounding prose
# ("Here is your verdict: {...}\n\nNote: ..."). Greedy is correct here —
# we want the *outermost* braces, not the first inner pair.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def strip_json_fences(text: str) -> str:
    """Strip ``\\`\\`\\`...\\`\\`\\``` code fences and leading/trailing whitespace."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    return cleaned


def try_parse_strict(text: str, model_cls: type[T]) -> T | None:
    """Parse ``text`` as ``model_cls`` after stripping common LLM envelopes.

    Resilience layers, in order:

    1. Strip code fences and outer whitespace.
    2. Attempt ``model_validate_json`` directly.
    3. On failure, extract the outermost ``{...}`` JSON object substring
       and retry — handles cases where the LLM emits prose adjacent to
       the JSON ("Here is the verdict: {...}\\n\\nNote: ...") despite
       prompting otherwise.
    4. Return ``None`` if both attempts fail; callers decide whether to
       retry with a stricter prompt or raise a domain error.
    """
    cleaned = strip_json_fences(text)
    try:
        return model_cls.model_validate_json(cleaned)
    except ValidationError:
        pass
    # Extract the outermost JSON object and retry. Without this, an LLM
    # that emits ``Note: this is correct.`` after the JSON silently fails
    # parse and downstream callers default to a fallback (e.g., the rubric
    # judge's 0.5 mid-scale), biasing the metric without any signal.
    match = _JSON_OBJECT_RE.search(cleaned)
    if match is None:
        return None
    try:
        return model_cls.model_validate_json(match.group(0))
    except ValidationError:
        return None


def load_prompt(filename: str) -> str:
    """Load a frozen prompt from ``prompts/`` (wheel or repo-relative).

    Tries the installed-wheel layout first
    (``importlib.resources.files("steadfast").joinpath("prompts/...")``),
    falls back to the repo-relative ``<repo>/prompts/<filename>`` path so
    editable installs work without requiring ``prompts/`` inside the
    package directory.

    Raises :class:`FileNotFoundError` if neither location resolves.
    """
    try:
        ref = resources.files("steadfast").joinpath(f"prompts/{filename}")
        if ref.is_file():
            return ref.read_text(encoding="utf-8")
    except (ModuleNotFoundError, FileNotFoundError):
        pass
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / "prompts" / filename
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"prompt {filename!r} not found in package data or repo-relative path; "
        "ensure prompts/* are shipped with the wheel."
    )
