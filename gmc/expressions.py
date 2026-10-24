"""Keyword rules over referring expressions.

Two independent uses:

* `classify` sorts a test expression into MOVING / STATIC / APPEARANCE. It
  picks the fusion weight (MOVING and STATIC use alpha_mot, APPEARANCE uses
  alpha_oth) and defines the moving class that moving-class HOTA is computed
  over. Direction and turning expressions ("turning cars", "cars faster than
  ours") are APPEARANCE by design.

* `is_motion_expression` and `motion_type_group` choose the training
  expressions and give each one of six coarse motion-type labels.
"""

from __future__ import annotations

# ---------------------------------------------------------------- test-time classes

MOVING_KEYWORDS = [
    # explicit motion
    "moving", "in motion", "driving", "walking", "running", "jogging", "crossing",
    "riding", "travelling", "traveling",
    # velocity changes
    "braking", "brake", "accelerat", "decelerat", "slowing down", "speeding up",
    # relative motion
    "approaching", "overtaking", "receding",
]
STATIC_KEYWORDS = ["parking", "parked", "stopped", "stop", "stand", "static", "stationary"]

CLASSES = ("MOVING", "STATIC", "APPEARANCE")


def classify(text: str) -> str:
    """'MOVING' | 'STATIC' | 'APPEARANCE'. Slugs ("moving-cars") and sentences
    ("moving cars") give the same answer. STATIC wins a tie."""
    t = text.lower().replace("-", " ")
    if any(k in t for k in STATIC_KEYWORDS):
        return "STATIC"
    if any(k in t for k in MOVING_KEYWORDS):
        return "MOVING"
    return "APPEARANCE"


# ---------------------------------------------------------------- training selection

# An expression is used for training only if it mentions motion at all;
# appearance-only sentences ("red cars") cannot be learned from a motion vector.
MOTION_KEYWORDS = [
    # motion state
    "moving", "in motion", "driving", "parking", "parked", "at rest",
    "stationary", "still", "stopping", "stopped",
    # direction
    "turning", "counter direction", "same direction", "opposite direction",
    "contrary direction", "reverse direction", "horizon direction", "heading",
    "in front of",
    # speed
    "braking", "brake", "slower", "faster", "speedier", "accelerat", "decelerat",
    "slowing",
    # interaction
    "following", "approaching", "overtaking", "travelling", "traveling",
]

# First match wins, so the more specific keywords come first
# ("moving turning cars" is TURNING, not MOVING).
MOTION_TYPE_GROUPS = {
    "braking": 0,
    "slower": 0,
    "turning": 1,
    "parking": 2,
    "counter direction": 3,
    "contrary direction": 3,
    "reverse direction": 3,
    "opposite direction": 3,
    "horizon direction": 4,
    "in front of": 4,
    "same direction": 5,
    "moving": 5,
    "in motion": 5,
    "driving": 5,
    "following": 5,
    "approaching": 5,
    "travelling": 5,
    "traveling": 5,
    "overtaking": 5,
    "accelerat": 5,
    "faster": 5,
    "speedier": 5,
}
MOTION_TYPE_NAMES = {0: "braking", 1: "turning", 2: "parking", 3: "counter_dir", 4: "approaching", 5: "moving"}


def is_motion_expression(sentence: str) -> bool:
    lower = sentence.lower()
    return any(k in lower for k in MOTION_KEYWORDS)


def motion_type_group(sentence: str) -> int | None:
    """Motion-type label (0-5) of a training sentence, or None if it has none."""
    lower = sentence.lower()
    for keyword, group in MOTION_TYPE_GROUPS.items():
        if keyword in lower:
            return group
    return None
