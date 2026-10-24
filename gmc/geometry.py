"""Point warping and resolution-independent velocity normalization."""

import numpy as np

# Normalized velocities are multiplied by this factor so the aligner sees
# inputs of magnitude ~1 instead of ~0.01.
VELOCITY_SCALE = 100


def warp_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N, 2) points. Returns (N, 2) points."""
    if points is None or len(points) == 0:
        return np.empty((0, 2))
    homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    warped = homogeneous @ homography.T
    return warped[:, :2] / warped[:, 2:3]


def normalize_velocity(v: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
    """Divide a pixel displacement [dx, dy] by the frame size and scale by
    VELOCITY_SCALE, so the same physical motion gives the same value at any
    resolution."""
    if len(v) == 0:
        return v
    h, w = frame_shape
    v = np.array(v, dtype=np.float32)
    v = v / np.array([w, h], dtype=np.float32)
    v *= VELOCITY_SCALE
    return v
