"""The GMC module: camera-motion-compensated motion cues for referring
multi-object tracking (RMOT), added to a host model's scores by late fusion."""

from .aligner import MotionLanguageAligner, load_aligner
from .ego import EgoMotionEstimator
from .expressions import classify
from .fusion import PAPER_WEIGHTS, FusionWeights, fuse
from .manager import FRAME_GAPS, GMCManager

__all__ = [
    "FRAME_GAPS",
    "PAPER_WEIGHTS",
    "EgoMotionEstimator",
    "FusionWeights",
    "GMCManager",
    "MotionLanguageAligner",
    "classify",
    "fuse",
    "load_aligner",
]
__version__ = "1.0.0"
