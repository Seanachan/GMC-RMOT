import numpy as np
import pytest

from gmc.ego import EgoMotionEstimator, _clear_boxes
from gmc.geometry import warp_points
from tests.synthetic import H, W, frame_at, world_texture


@pytest.fixture(scope="module")
def world():
    return world_texture()


def _translation_of(Hm: np.ndarray) -> np.ndarray:
    """Mean displacement of a grid of points under H."""
    pts = np.array([[x, y] for x in (50, 320, 600) for y in (130, 200)], dtype=np.float64)
    return (warp_points(pts, Hm) - pts).mean(axis=0)


@pytest.mark.parametrize("shift", [3, 8, -5])
def test_road_homography_recovers_camera_pan(world, shift):
    prev, curr = frame_at(world, 100), frame_at(world, 100 + shift)
    Hm = EgoMotionEstimator().road_homography(prev, curr)
    assert Hm is not None
    # the camera moves right by `shift`, so the scene moves left by `shift`
    np.testing.assert_allclose(_translation_of(Hm), [-shift, 0.0], atol=0.3)


def test_global_homography_recovers_camera_pan(world):
    prev, curr = frame_at(world, 100), frame_at(world, 106, offset_y=42)
    Hm = EgoMotionEstimator().global_homography(prev, curr)
    np.testing.assert_allclose(_translation_of(Hm), [-6.0, -2.0], atol=0.3)


def test_road_fit_fails_on_blank_road_and_estimate_falls_back(world):
    prev, curr = frame_at(world, 100), frame_at(world, 104)
    prev[H // 2:] = 128
    curr[H // 2:] = 128
    ego = EgoMotionEstimator()
    assert ego.road_homography(prev, curr) is None
    np.testing.assert_allclose(_translation_of(ego.estimate(prev, curr)), [-4.0, 0.0], atol=0.5)


def test_global_homography_is_identity_without_texture():
    blank = np.full((H, W, 3), 90, dtype=np.uint8)
    np.testing.assert_array_equal(EgoMotionEstimator().global_homography(blank, blank), np.eye(3))


def test_road_band_ignores_upper_half(world):
    """Moving only the top half of the image must not change the road fit."""
    prev, curr = frame_at(world, 100), frame_at(world, 105)
    scrambled = curr.copy()
    scrambled[: H // 2] = np.roll(scrambled[: H // 2], 40, axis=1)
    ego = EgoMotionEstimator()
    np.testing.assert_allclose(
        _translation_of(ego.road_homography(prev, scrambled)),
        _translation_of(ego.road_homography(prev, curr)),
        atol=0.2,
    )


def test_clear_boxes_clips_to_frame():
    mask = np.full((10, 20), 255, dtype=np.uint8)
    _clear_boxes(mask, [(-5, 2, 4, 6), (18, 8, 40, 30), (3, 3, 3, 9)])
    assert (mask[2:6, 0:4] == 0).all()
    assert (mask[8:10, 18:20] == 0).all()
    assert mask[0, 10] == 255 and mask[5, 3] == 0
    assert mask.sum() == 255 * (200 - 16 - 4)
