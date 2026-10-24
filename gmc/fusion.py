"""Late fusion with the host RMOT model (Eq. 3 of the paper):

    s_final = s_host + alpha_c * s_gmc

alpha_c is alpha_mot for MOVING and STATIC expressions and alpha_oth for
APPEARANCE expressions (gmc/expressions.classify). A track the module has no
score for (still warming up) keeps s_final = s_host. alpha = 0 reproduces
the host exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

from .expressions import classify


@dataclass(frozen=True)
class FusionWeights:
    alpha_mot: float
    alpha_oth: float

    def for_expression(self, expression: str) -> float:
        return self.alpha_oth if classify(expression) == "APPEARANCE" else self.alpha_mot


# Selected by leave-one-sequence-out cross-validation (paper, Table 1).
PAPER_WEIGHTS = {
    "ikun": FusionWeights(alpha_mot=1.0, alpha_oth=0.1),
    "transrmot": FusionWeights(alpha_mot=0.7, alpha_oth=0.5),
    "flexhook_v1": FusionWeights(alpha_mot=7.0, alpha_oth=7.0),
    "flexhook_v2": FusionWeights(alpha_mot=5.0, alpha_oth=5.0),
}


def fuse(s_host: float, s_gmc: float | None, alpha: float) -> float:
    return s_host if s_gmc is None else s_host + alpha * s_gmc
