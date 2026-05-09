"""LLM-as-judge with a Pydantic-validated output schema.

Per ADR-0003 §B.4, the judge prompt instructs JSON-only output matching
the :class:`~steadfast.judges.base.Verdict` schema. We parse with
:meth:`Verdict.model_validate_json`; on :class:`pydantic.ValidationError`
we retry once with a stricter "JSON only" reminder, then raise
:class:`JudgeParseError`. We deliberately do *not* produce a soft-failed
verdict — failures are signal (cf. ADR-0002 §D.1).

Default model is ``gpt-5.2`` per ADR-0001 (infrastructure-model lock).
Local users running cheaper inner-loop experiments may pass a different
model in the constructor; only ``gpt-5.2`` is acceptable for v0.1
leaderboard publication.

Pattern adapted from the Pydantic Evals LLM-as-judge guide
(https://pydantic.dev/articles/llm-as-a-judge); ensemble variant in
``ensemble.py`` covers the residual judge-bias risk noted in
``docs/METHODOLOGY.md`` §"Known limitations".
"""

from __future__ import annotations

import re
from importlib import resources
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from steadfast.agent import AgentResponse, Task
from steadfast.judges.base import Judge, JudgeParseError, Verdict
from steadfast.models.base import BaseModelClient

# Frozen prompt at prompts/rubric_v1.txt. Versioned via filename suffix
# (``_v1``); any change to scoring semantics → new file (``_v2``) + new
# methodology version per docs/METHODOLOGY.md §"Versioning".
RUBRIC_PROMPT_VERSION: Final[str] = "v1"
DEFAULT_RUBRIC_MODEL: Final[str] = "gpt-5.2"


def _load_rubric_prompt() -> str:
    """Load the frozen rubric prompt template from ``prompts/rubric_v1.txt``.

    Tries the installed-package layout first (``importlib.resources``),
    falls back to the repo-relative path so editable installs and
    development checkouts work without running ``pip install -e``.
    """
    # Try package data first — works when prompts/ is shipped with the wheel.
    try:
        ref = resources.files("steadfast").joinpath("prompts/rubric_v1.txt")
        if ref.is_file():
            return ref.read_text(encoding="utf-8")
    except (ModuleNotFoundError, FileNotFoundError):
        pass

    # Repo-relative fallback for editable / source-tree runs. The file is
    # at ``<repo>/prompts/rubric_v1.txt``; this module is at
    # ``<repo>/src/steadfast/judges/rubric.py`` → 4 parents up.
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "prompts" / "rubric_v1.txt"
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8")

    raise FileNotFoundError(
        "rubric_v1.txt not found in package data or repo-relative path; "
        "ensure prompts/rubric_v1.txt is shipped with the wheel."
    )


_RETRY_REMINDER = (
    "Your previous output failed JSON validation. Emit ONLY the JSON object "
    "described in the schema — no prose, no code fences, no commentary."
)


# Single-pass placeholder regex. Substituting via three sequential
# ``str.replace`` calls would re-substitute placeholders that appear
# inside earlier-substituted values (e.g., a ``task.input`` containing
# the literal ``{rubric}`` would have the rubric injected at two
# locations). ``re.sub`` with a dispatch dict resolves each match
# against the original template only.
_PLACEHOLDER_RE = re.compile(r"\{(task_input|rubric|answer)\}")


def _render_prompt(*, template: str, task: Task, response: AgentResponse) -> str:
    """Substitute task / rubric / answer into the frozen template.

    Uses ``re.sub`` (not ``str.format``) because the JSON schema portion
    of the template contains literal braces. The placeholders
    ``{task_input}``, ``{rubric}``, and ``{answer}`` are matched against
    the original template only; user-provided content cannot trigger
    follow-on substitutions.
    """
    rubric_text = (
        task.ground_truth.value if task.ground_truth is not None else "(no rubric provided)"
    )
    values = {
        "task_input": task.input,
        "rubric": rubric_text,
        "answer": response.answer,
    }
    return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template)


class RubricJudge(Judge):
    """LLM-as-judge — frozen rubric prompt, Pydantic-validated JSON output.

    Construction takes a :class:`BaseModelClient` (the v0.1 default per
    ADR-0001 is :class:`OpenAIClient`) and a model identifier. The judge
    holds the loaded prompt template at construction so per-judge calls
    don't repeatedly hit the filesystem.
    """

    def __init__(
        self,
        *,
        client: BaseModelClient,
        model: str = DEFAULT_RUBRIC_MODEL,
    ) -> None:
        self._client = client
        self._model = model
        self._template = _load_rubric_prompt()

    @property
    def model(self) -> str:
        """Judge model identifier — used by tracing to populate
        :data:`~steadfast.tracing.conventions.STEADFAST_JUDGE_MODEL`.
        """
        return self._model

    async def ajudge(self, task: Task, response: AgentResponse) -> Verdict:
        prompt = _render_prompt(template=self._template, task=task, response=response)

        # First attempt.
        text = (await self._client.acomplete(prompt, model=self._model)).text
        verdict = _try_parse(text)
        if verdict is not None:
            return verdict

        # Second attempt with a stricter reminder. Per ADR-0003 §B.4, this
        # is the only retry — exhaustion raises rather than soft-failing.
        retry_prompt = f"{prompt}\n\n{_RETRY_REMINDER}"
        text = (await self._client.acomplete(retry_prompt, model=self._model)).text
        verdict = _try_parse(text)
        if verdict is not None:
            return verdict

        raise JudgeParseError(
            f"RubricJudge ({self._model}) produced unparseable output twice; "
            f"last output (truncated): {text[:200]!r}"
        )


def _try_parse(text: str) -> Verdict | None:
    """Parse ``text`` as a :class:`Verdict`, stripping common envelopes.

    Returns the verdict on success or ``None`` on validation failure
    (caller decides whether to retry or raise). Handles two real-world
    deviations from "JSON-only" prompting:

    1. Triple-backtick code fences (``\\`\\`\\`json ... \\`\\`\\``).
    2. Surrounding whitespace.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip a leading fence (with optional language tag) and a trailing fence.
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
    try:
        return Verdict.model_validate_json(cleaned)
    except ValidationError:
        return None
