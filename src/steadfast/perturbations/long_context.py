"""Long-context degradation perturbation.

Per ``docs/METHODOLOGY.md`` §2.4 and ADR-0006 §E: prepend neutral filler
to the task input until the tokenized prompt reaches a target token
count. The metric layer runs the agent at each tier in the standard
``[4k, 16k, 64k, 128k]`` ladder and reports the empirical success curve
plus a logistic fit on ``log10(tokens)``.

Filler is **prepended**, not mid-inserted (ADR-0006 §E rationale): the
prepend shape matches the most common production deployment (long system
prompt or RAG context followed by the user query at the tail) and avoids
the haystack-position confound that mid-insertion would introduce. A
single ``--- task ---`` delimiter separates the filler from the original
task so a robust agent can in principle locate the task boundary.

The filler corpus lives at ``prompts/longcontext_filler_v1.txt`` —
bland, topic-neutral encyclopedic prose with no semantic overlap with
the v0.1 benchmark domains. To reach 128k tokens with a small repo
footprint the corpus is **tiled** (concat with itself) as many times as
needed. The tile seam is a small acknowledged confound (a model could
recognize the repetition pattern); the empirical evidence from prior
long-context work is that the seam does not materially shift the
degradation curve at the tiers we measure.

Determinism: the seed selects a starting token offset into the corpus;
ten reps over one task at one tier each get a distinct window. Seeds
are derived by the metric layer via
:func:`steadfast.perturbations.derive_seed` with the kind
``"long_context"``.

References:

* Liu et al. (2024), "Lost in the Middle: How Language Models Use Long
  Contexts", *TACL* 12, 157-173 — motivates the ``log10(L)`` ladder and
  the general degradation-with-length methodology.
* ADR-0006 §E — fit + CI contract.
* METHODOLOGY §2.4 — tier ladder and reporting contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import tiktoken

DEFAULT_ENCODING: Final[str] = "cl100k_base"

# Resolves to ``<repo>/prompts/longcontext_filler_v1.txt`` in editable
# installs; the wheel build copies ``prompts/`` into the package data so
# this path resolves there too (per pyproject.toml [tool.hatch] config).
DEFAULT_FILLER_PATH: Final[Path] = (
    Path(__file__).resolve().parents[3] / "prompts" / "longcontext_filler_v1.txt"
)

# Frozen single-fence delimiter per ADR-0006 §E. The leading newline keeps
# the filler text and the delimiter visually separated when the filler
# happens to end mid-sentence (which it usually will, since the window is
# token-aligned, not sentence-aligned).
_DELIMITER: Final[str] = "\n--- task ---\n"

# Module-level caches: tokenizer initialization is non-trivial, and the
# filler corpus is read + tokenized once per (filler_path, encoding) pair.
# Both caches are unbounded — long_context is called at most O(n_tiers x
# n_tasks x n_reps) per benchmark run and the cache key set is tiny.
_encoding_cache: dict[str, tiktoken.Encoding] = {}
_filler_token_cache: dict[tuple[str, str], list[int]] = {}


def _get_encoding(encoding: str) -> tiktoken.Encoding:
    cached = _encoding_cache.get(encoding)
    if cached is None:
        cached = tiktoken.get_encoding(encoding)
        _encoding_cache[encoding] = cached
    return cached


def _load_filler_tokens(filler_path: Path, encoding: str) -> list[int]:
    key = (str(filler_path), encoding)
    cached = _filler_token_cache.get(key)
    if cached is None:
        text = Path(filler_path).read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"filler file is empty: {filler_path}")
        cached = _get_encoding(encoding).encode(text)
        if not cached:
            raise ValueError(f"filler file tokenizes to empty token list: {filler_path}")
        _filler_token_cache[key] = cached
    return cached


def perturb_long_context(
    text: str,
    *,
    target_tokens: int,
    filler_path: str | Path = DEFAULT_FILLER_PATH,
    seed: int,
    encoding: str = DEFAULT_ENCODING,
) -> str:
    """Prepend deterministic filler so the perturbed prompt tokenizes to ~``target_tokens``.

    Parameters
    ----------
    text:
        The original task input.
    target_tokens:
        Desired total token count of the perturbed prompt (filler +
        delimiter + ``text``). Must leave room for the task and the
        delimiter; otherwise :class:`ValueError` is raised. The actual
        token count of the returned string may differ from
        ``target_tokens`` by a small number of tokens because
        ``encode(decode(tokens))`` is not strictly identity on the
        ``cl100k_base`` BPE — boundary tokens at the seam between
        filler and delimiter can re-tokenize differently. Tests assert
        the returned string is within a small tolerance of
        ``target_tokens``.
    filler_path:
        Path to the frozen filler file. Defaults to
        :data:`DEFAULT_FILLER_PATH`. The file is read once per process
        and cached.
    seed:
        Integer seed from
        :func:`steadfast.perturbations.derive_seed` (typically with
        ``kind="long_context"`` and ``rep_idx=rep_idx``). The seed
        picks a starting offset into the (tokenized) corpus; ten reps
        with distinct seeds get ten distinct prefix windows. The same
        seed produces byte-identical output across runs.
    encoding:
        :mod:`tiktoken` encoding name. Default ``cl100k_base`` matches
        the distractor-bank convention.

    Returns
    -------
    str
        ``"<filler-window-text>\\n--- task ---\\n<original-text>"``.
        The filler window is a contiguous slice of the tokenized
        corpus starting at ``seed % corpus_len`` and tiling forward
        (wrapping around the corpus end as many times as needed).

    Raises
    ------
    ValueError
        If ``target_tokens`` is too small to fit the task and delimiter,
        if the filler file is empty, or if ``target_tokens`` is not
        positive.

    Notes
    -----
    Tiling: when the requested window exceeds the corpus length, the
    perturbation concatenates the corpus with itself as many times as
    needed. The seam at each tile boundary is an acknowledged confound
    (a model could recognize the repetition). The corpus at
    ``prompts/longcontext_filler_v1.txt`` is ~3.8k tokens, so reaching
    128k tokens requires roughly 34 tiles. Per ADR-0006 §E the seam is
    a v0.1 cost we accept in exchange for a small repo footprint; v0.2
    may ship a longer corpus if seam effects are detected empirically.
    """
    if target_tokens <= 0:
        raise ValueError(f"target_tokens must be > 0; got {target_tokens}")

    enc = _get_encoding(encoding)
    text_tokens = enc.encode(text)
    delimiter_tokens = enc.encode(_DELIMITER)

    n_filler_needed = target_tokens - len(text_tokens) - len(delimiter_tokens)
    if n_filler_needed <= 0:
        raise ValueError(
            f"target_tokens={target_tokens} leaves no room for filler: "
            f"task tokenizes to {len(text_tokens)} tokens, delimiter to "
            f"{len(delimiter_tokens)} tokens. Need target_tokens >= "
            f"{len(text_tokens) + len(delimiter_tokens) + 1}."
        )

    filler_corpus = _load_filler_tokens(Path(filler_path), encoding)
    corpus_len = len(filler_corpus)

    # `seed` is an unsigned 64-bit int from derive_seed; Python's % is
    # well-defined on it.
    start = seed % corpus_len

    if n_filler_needed <= corpus_len - start:
        # Single contiguous slice fits within the remainder of the corpus.
        window: list[int] = list(filler_corpus[start : start + n_filler_needed])
    else:
        # Tile: take from `start` to end, then wrap full copies, then a
        # final partial slice. Allocate the target list up front to avoid
        # repeated re-allocation as window grows.
        window = list(filler_corpus[start:])
        remaining = n_filler_needed - len(window)
        while remaining >= corpus_len:
            window.extend(filler_corpus)
            remaining -= corpus_len
        if remaining > 0:
            window.extend(filler_corpus[:remaining])

    filler_text = enc.decode(window)
    return f"{filler_text}{_DELIMITER}{text}"


def count_tokens(text: str, *, encoding: str = DEFAULT_ENCODING) -> int:
    """Return the tokenized length of ``text`` under ``encoding``.

    Thin wrapper around the cached tokenizer; consumed by tests and by
    the metric layer's per-tier diagnostics.
    """
    return len(_get_encoding(encoding).encode(text))


__all__ = [
    "DEFAULT_ENCODING",
    "DEFAULT_FILLER_PATH",
    "count_tokens",
    "perturb_long_context",
]
