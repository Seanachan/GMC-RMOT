import os

import pytest

from gmc.expressions import classify, is_motion_expression, motion_type_group

SEQMAP = os.path.join(os.path.dirname(__file__), "..", "seqmaps", "refer_kitti_v1_test_official_150.txt")


@pytest.mark.parametrize("text,expected", [
    ("moving-cars", "MOVING"),
    ("moving cars", "MOVING"),
    ("braked cars", "MOVING"),
    ("pedestrians-who-are-walking", "MOVING"),
    ("left-cars-which-are-parking", "STATIC"),
    ("standing-females", "STATIC"),
    ("turning-cars", "APPEARANCE"),
    ("cars-which-are-faster-than-ours", "APPEARANCE"),
    ("women-carrying-a-bag", "APPEARANCE"),
    ("black-cars-in-the-left", "APPEARANCE"),
])
def test_classify(text, expected):
    assert classify(text) == expected


def test_static_wins_a_tie():
    assert classify("moving then stopped cars") == "STATIC"


def test_official_test_list_class_sizes():
    """21 of the 150 official V1 test expressions are moving-class (paper, Sec. 4.1)."""
    with open(SEQMAP) as f:
        exprs = [line.split("+", 1)[1] for line in f.read().split()]
    assert len(exprs) == 150
    counts = {c: sum(classify(e) == c for e in exprs) for c in ("MOVING", "STATIC", "APPEARANCE")}
    assert counts == {"MOVING": 21, "STATIC": 12, "APPEARANCE": 117}


@pytest.mark.parametrize("sentence,group", [
    ("cars which are braking", 0),
    ("moving turning cars", 1),
    ("left vehicles which are parking", 2),
    ("cars in the counter direction of ours", 3),
    ("cars in front of ours", 4),
    ("moving cars", 5),
    ("parked cars", None),
    ("red cars", None),
])
def test_motion_type_group(sentence, group):
    assert motion_type_group(sentence) == group


def test_is_motion_expression():
    assert is_motion_expression("Cars which are STOPPED")
    assert is_motion_expression("cars in front of ours")
    assert not is_motion_expression("women in black")
