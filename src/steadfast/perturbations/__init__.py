"""Input perturbations.

Paraphrase, typo, distractor, contradiction, long-context, plus the
confidence-elicitation prompt suffix used by the calibration dimension.

Robustness perturbations (typo, distractor, contradiction, long-context)
share a deterministic seed-derivation helper exposed here at package
scope; see :mod:`steadfast.perturbations._seed`.
"""

from steadfast.perturbations._seed import SEED_VERSION, derive_seed

__all__ = ["SEED_VERSION", "derive_seed"]
