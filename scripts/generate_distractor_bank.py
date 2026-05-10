"""Generate a draft distractor bank for one domain.

Per ADR-0006 §C, the bank generation pipeline is:

1. Read every task in ``benchmarks/<domain>/`` (excluding any pre-existing
   distractor JSONs).
2. Render the frozen prompt at ``prompts/distractor_bank_v1.txt`` with the
   domain name + the corpus of task inputs as context.
3. Call GPT-5.2 (per ADR-0001's infrastructure-LLM lock) at temperature
   0.7 — the high temperature is intentional, the bank should be diverse.
4. Parse the JSON-only response into a snippet list, dedupe by content
   hash, tokenize each surviving snippet with ``tiktoken cl100k_base``,
   and serialize as a :class:`steadfast.perturbations.distractor.DistractorBank`
   in ``review_status="draft"``.
5. Write to ``benchmarks/<domain>/distractors_v1.draft.json`` and exit
   with operator review instructions. The metric only loads
   ``distractors_v1.json`` (no draft suffix); the rename is the
   methodology gate that prevents unaudited LLM output from biasing
   benchmark results.

Usage::

    set -a && source .env && set +a && \\
        uv run python scripts/generate_distractor_bank.py --domain customer_support

Cost estimate: ~$0.50 per domain on gpt-5.2 at the default ``--n 50``.

References:

* ADR-0006 §C — bank generation / curation / freezing.
* METHODOLOGY §2.2 — perturbation contract.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import tiktoken
from pydantic import BaseModel

from steadfast._llm_parsing import load_prompt, try_parse_strict
from steadfast.agent import Task
from steadfast.models.openai_client import OpenAIClient
from steadfast.perturbations.distractor import (
    DEFAULT_ENCODING,
    DistractorBank,
    DistractorSnippet,
    write_distractor_bank,
)

DEFAULT_GENERATOR_MODEL: Final[str] = "gpt-5.2"
DEFAULT_N: Final[int] = 50
DEFAULT_TEMPERATURE: Final[float] = 0.7
PROMPT_FILE: Final[str] = "distractor_bank_v1.txt"

# Single-pass placeholder regex (same pattern as ``perturbations/paraphrase.py``)
# — protects against task-input text that contains literal ``{n}`` /
# ``{domain}`` placeholders triggering double substitution.
_PLACEHOLDER_RE = re.compile(r"\{(n|domain|tasks)\}")

_log = logging.getLogger("steadfast.scripts.generate_distractor_bank")


class _SnippetList(BaseModel):
    """Schema for the generator's JSON output."""

    snippets: list[str]


def _render_prompt(*, template: str, n: int, domain: str, tasks_block: str) -> str:
    values = {"n": str(n), "domain": domain, "tasks": tasks_block}
    return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template)


def _format_tasks_block(tasks: list[Task]) -> str:
    """Format the corpus of task inputs for inclusion in the generator prompt.

    The format is intentionally low-structure: an LLM reading this block
    just needs to see "what topics / policies the tasks cover" so it can
    avoid contradicting them. JSON or YAML would add noise without
    helping.
    """
    return "\n\n---\n\n".join(f"TASK {t.id}:\n{t.input}" for t in tasks)


def _load_tasks(domain_dir: Path) -> list[Task]:
    """Load every ``Task`` JSON in ``domain_dir`` excluding distractor banks."""
    paths = sorted(p for p in domain_dir.glob("*.json") if "distractors" not in p.stem)
    if not paths:
        raise SystemExit(f"error: no task JSONs found in {domain_dir}; nothing to generate against")
    return [Task.model_validate_json(p.read_text(encoding="utf-8")) for p in paths]


async def _call_generator(
    *,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> _SnippetList:
    """Call the OpenAI client once and parse the response.

    Single-call for v0.1 — if the generator returns fewer than ``n``
    snippets the script reports the shortfall as a warning rather than
    looping. The operator can rerun the script for more snippets if
    needed.

    ``max_tokens`` must be large enough for ``n * ~700 tokens`` of
    snippet text plus JSON overhead — the default OpenAI client cap of
    1024 truncates after one or two snippets and breaks JSON parsing.
    The CLI sizes this against the requested ``--n``.
    """
    client = OpenAIClient()
    response = await client.acomplete(
        prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    parsed = try_parse_strict(response.text, _SnippetList)
    if parsed is None:
        raise SystemExit(
            "error: generator output failed to parse as a JSON snippet list. "
            f"finish_reason={response.finish_reason!r}; "
            "if 'length', increase --max-tokens.\n"
            "First 400 chars of response:\n" + response.text[:400]
        )
    return parsed


def _build_bank(
    *,
    domain: str,
    raw_snippets: list[str],
    generator_model: str,
    encoding_name: str,
) -> DistractorBank:
    """Tokenize, dedupe by content hash, and assemble a draft bank model."""
    enc = tiktoken.get_encoding(encoding_name)
    seen_ids: set[str] = set()
    snippets: list[DistractorSnippet] = []
    for raw in raw_snippets:
        text = raw.strip()
        if not text:
            continue
        snippet_id = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        if snippet_id in seen_ids:
            continue
        seen_ids.add(snippet_id)
        tokens = len(enc.encode(text))
        snippets.append(DistractorSnippet(id=snippet_id, text=text, tokens=tokens))

    return DistractorBank(
        domain=domain,
        encoding=encoding_name,
        prompt_version="v1",
        generator_model=generator_model,
        generated_at=datetime.now(UTC).isoformat(),
        review_status="draft",
        snippets=snippets,
    )


def _print_review_instructions(*, draft_path: Path, target_path: Path) -> None:
    """Print operator-facing review instructions to stderr.

    These instructions are the manual review gate from ADR-0006 §C —
    deliberately fail-loud (the metric won't pick up the draft file) so
    a hurried operator can't accidentally ship unaudited LLM output.
    """
    print(file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print("DRAFT WRITTEN. Manual review required per ADR-0006 §C.", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    print(f"  draft:  {draft_path}", file=sys.stderr)
    print(f"  target: {target_path}", file=sys.stderr)
    print(file=sys.stderr)
    print("Review checklist (per ADR-0006 §C):", file=sys.stderr)
    print(
        "  1. No snippet restates a policy from any task in this domain "
        "(e.g., 'X-day return window').",
        file=sys.stderr,
    )
    print(
        "  2. No snippet contradicts a policy from any task "
        "(e.g., a different return-window number).",
        file=sys.stderr,
    )
    print(
        "  3. Every snippet reads as plausible domain-adjacent prose.",
        file=sys.stderr,
    )
    print(
        "  4. Token distribution covers the 200-800 range (check tokens fields).",
        file=sys.stderr,
    )
    print(file=sys.stderr)
    print(
        "When done: edit ``review_status`` from 'draft' to 'reviewed' in the JSON",
        file=sys.stderr,
    )
    print("and rename:", file=sys.stderr)
    print(f"  mv {draft_path} {target_path}", file=sys.stderr)
    print("=" * 72, file=sys.stderr)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate a draft distractor bank for one Steadfast domain."
    )
    ap.add_argument(
        "--domain",
        required=True,
        help="Domain name (e.g., 'customer_support'); resolves to benchmarks/<domain>/.",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"Number of snippets to request (default {DEFAULT_N}).",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_GENERATOR_MODEL,
        help=(f"Generator model (default {DEFAULT_GENERATOR_MODEL}; ADR-0001 v0.1 lock)."),
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default {DEFAULT_TEMPERATURE}).",
    )
    ap.add_argument(
        "--encoding",
        default=DEFAULT_ENCODING,
        help=(
            f"tiktoken encoding for token counts (default {DEFAULT_ENCODING}). "
            "If you change this, the bank's gating range may shift across "
            "encodings; a fresh _v2 bank is the right move."
        ),
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help=(
            "Max output tokens for the generator. Default: ``n * 750`` "
            "(rough budget for n snippets averaging ~700 tokens of body "
            "text + JSON overhead). Set explicitly to override."
        ),
    )
    return ap.parse_args()


async def _main_async() -> int:
    args = _parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    domain_dir = repo_root / "benchmarks" / args.domain
    if not domain_dir.is_dir():
        print(
            f"error: benchmarks directory for domain {args.domain!r} not found at {domain_dir}",
            file=sys.stderr,
        )
        return 2

    target_path = domain_dir / "distractors_v1.json"
    if target_path.exists():
        print(
            f"error: {target_path} already exists. To regenerate the bank, "
            "remove it first (and bump the metric to _v2 per METHODOLOGY §"
            '"Versioning" if this is a leaderboard-grade change).',
            file=sys.stderr,
        )
        return 2

    tasks = _load_tasks(domain_dir)
    print(f"loaded {len(tasks)} task(s) from {domain_dir}", file=sys.stderr)

    template = load_prompt(PROMPT_FILE)
    prompt = _render_prompt(
        template=template,
        n=args.n,
        domain=args.domain,
        tasks_block=_format_tasks_block(tasks),
    )

    max_tokens = args.max_tokens or args.n * 750
    print(
        f"calling {args.model} for {args.n} snippet(s) at "
        f"temperature={args.temperature} (max_tokens={max_tokens})...",
        file=sys.stderr,
    )
    parsed = await _call_generator(
        prompt=prompt,
        model=args.model,
        temperature=args.temperature,
        max_tokens=max_tokens,
    )

    if len(parsed.snippets) < args.n:
        print(
            f"warning: generator returned {len(parsed.snippets)} snippet(s); "
            f"requested {args.n}. The bank may be smaller than ideal — rerun "
            "the script for more, or proceed with the smaller bank if it's "
            "still useful.",
            file=sys.stderr,
        )

    bank = _build_bank(
        domain=args.domain,
        raw_snippets=parsed.snippets,
        generator_model=args.model,
        encoding_name=args.encoding,
    )

    if not bank.snippets:
        print(
            "error: no snippets survived dedup/empty-strip — bank would be useless. Aborting.",
            file=sys.stderr,
        )
        return 1

    draft_path = domain_dir / "distractors_v1.draft.json"
    write_distractor_bank(bank, draft_path)

    print(
        f"wrote draft with {len(bank.snippets)} snippet(s); token range "
        f"[{min(s.tokens for s in bank.snippets)}, "
        f"{max(s.tokens for s in bank.snippets)}]",
        file=sys.stderr,
    )
    _print_review_instructions(draft_path=draft_path, target_path=target_path)
    return 0


def main() -> None:
    """Console entry point — wraps :func:`_main_async`."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    rc = asyncio.run(_main_async())
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
