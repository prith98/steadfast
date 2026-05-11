"""Input perturbations.

Paraphrase, typo, distractor, contradiction, long-context, plus the
confidence-elicitation prompt suffix used by the calibration dimension.

Robustness perturbations (typo, distractor, contradiction, long-context)
share a deterministic seed-derivation helper exposed here at package
scope; see :mod:`steadfast.perturbations._seed`.
"""

from steadfast.perturbations._seed import SEED_VERSION, derive_seed
from steadfast.perturbations.long_context import (
    DEFAULT_FILLER_PATH as LONG_CONTEXT_DEFAULT_FILLER_PATH,
)
from steadfast.perturbations.long_context import (
    count_tokens,
    perturb_long_context,
)

__all__ = [
    "LONG_CONTEXT_DEFAULT_FILLER_PATH",
    "SEED_VERSION",
    "count_tokens",
    "derive_seed",
    "perturb_long_context",
]
