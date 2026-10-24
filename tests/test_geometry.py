import numpy as np
import pytest

from gmc.geometry import VELOCITY_SCALE, normalize_velocity, warp_points


def test_warp_identity():
    pts = np.array([[10.0, 20.0], [300.0, 5.5]])
    np.testing.assert_allclose(warp_points(pts, np.eye(3)), pts)


def test_warp_translation_and_scale():
    H = np.array([[2.0, 0, 5.0], [0, 2.0, -3.0], [0, 0, 1.0]])
    np.testing.assert_allclose(warp_points(np.array([[1.0, 1.0]]), H), [[7.0, -1.0]])


def test_warp_projective_divides_by_w():
    H = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 2.0]])
    np.testing.assert_allclose(warp_points(np.array([[4.0, 6.0]]), H), [[2.0, 3.0]])


def test_warp_empty():
    assert warp_points(np.empty((0, 2)), np.eye(3)).shape == (0, 2)
    assert warp_points(None, np.eye(3)).shape == (0, 2)


def test_normalize_velocity_units():
    v = normalize_velocity(np.array([12.42, -3.75]), (375, 1242))
    np.testing.assert_allclose(v, [0.01 * VELOCITY_SCALE, -0.01 * VELOCITY_SCALE], rtol=1e-6)
    assert v.dtype == np.float32


@pytest.mark.parametrize("scale", [0.5, 2.0, 3.0])
def test_normalize_velocity_is_resolution_independent(scale):
    base = normalize_velocity(np.array([30.0, 9.0]), (375, 1242))
    scaled = normalize_velocity(np.array([30.0 * scale, 9.0 * scale]), (375 * scale, 1242 * scale))
    np.testing.assert_allclose(base, scaled, rtol=1e-6)
