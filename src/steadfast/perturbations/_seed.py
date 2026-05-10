"""Deterministic seed derivation for robustness perturbations.

Per ``docs/adr/0006-robustness-and-paired-bootstrap.md`` §B, every
robustness perturbation derives its RNG seed from
``sha256(f"{task.id}:{kind}:v1"...)`` so re-runs with the same task
content produce the same perturbed inputs. Per-rep extension
(``:rep{idx}``) gives N=10 distinct perturbations within a single (task,
kind), preserving the distributional measurement that METHODOLOGY
§"Multi-run by default" requires — without per-rep variation, the N=10
perturbed reps would collapse to one input x ten retries.

The ``:v1`` suffix lets a future incompatible change to the seed
derivation bump to ``:v2`` without invalidating leaderboard entries
computed against ``:v1`` (per the same versioning discipline that
``prompts/*_v1.*`` files follow).

References:

* ADR-0006 §B — perturbation seed strategy.
* METHODOLOGY §"Statistical conventions" — reproducibility commitment.
"""

from __future__ import annotations

import hashlib
from typing import Final

SEED_VERSION: Final[str] = "v1"


def derive_seed(
    task_id: str,
    kind: str,
    *,
    rep_idx: int | None = None,
    tool_call_idx: int | None = None,
    version: str = SEED_VERSION,
) -> int:
    """Return a stable 64-bit seed for a perturbation draw.

    ``kind`` is the perturbation name (e.g., ``"typo"``, ``"distractor"``,
    ``"long_context"``, ``"contradiction"``). ``rep_idx`` extends the seed
    when the perturbation has a per-rep stochastic dimension (typo's
    character positions, distractor's bank-snippet selection, long-
    context's filler choice). ``tool_call_idx`` extends it for the
    contradiction perturbation, which decides per-tool-call whether to
    corrupt the response.

    Both extensions can stack (a contradiction perturbation that also
    varies per rep would pass both), but in v0.1 only one is set at a
    time per the four perturbations as specified.

    Returns an unsigned 64-bit integer suitable for seeding
    :class:`random.Random` and :class:`numpy.random.Generator` — both
    accept arbitrary integer seeds without truncation up to 64 bits.

    Examples
    --------
    >>> derive_seed("pilot_001", "typo", rep_idx=0) != derive_seed("pilot_001", "typo", rep_idx=1)
    True
    >>> derive_seed("pilot_001", "typo") == derive_seed("pilot_001", "typo")
    True
    """
    parts = [task_id, kind, version]
    if rep_idx is not None:
        parts.append(f"rep{rep_idx}")
    if tool_call_idx is not None:
        parts.append(f"tool{tool_call_idx}")
    payload = ":".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


__all__ = ["SEED_VERSION", "derive_seed"]
