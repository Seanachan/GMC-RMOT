import pytest

from gmc import PAPER_WEIGHTS, FusionWeights, fuse


def test_paper_weights_match_table_1():
    assert {k: (w.alpha_mot, w.alpha_oth) for k, w in PAPER_WEIGHTS.items()} == {
        "ikun": (1.0, 0.1),
        "transrmot": (0.7, 0.5),
        "flexhook_v1": (7.0, 7.0),
        "flexhook_v2": (5.0, 5.0),
    }


def test_routing_by_expression_class():
    w = FusionWeights(alpha_mot=1.0, alpha_oth=0.1)
    assert w.for_expression("moving-cars") == 1.0
    assert w.for_expression("parked-cars") == 1.0
    assert w.for_expression("red-cars") == 0.1


@pytest.mark.parametrize("s_host,s_gmc,alpha,expected", [
    (0.3, 0.5, 1.0, 0.8),
    (0.3, -0.5, 1.0, -0.2),
    (0.3, 0.5, 0.0, 0.3),
    (-0.4, None, 7.0, -0.4),
])
def test_fuse(s_host, s_gmc, alpha, expected):
    assert fuse(s_host, s_gmc, alpha) == pytest.approx(expected)


def test_zero_alpha_reproduces_host_exactly():
    for s_host in (-1.3, 0.0, 2.25):
        assert fuse(s_host, 0.73, 0.0) == s_host
