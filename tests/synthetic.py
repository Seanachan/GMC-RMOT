"""Synthetic moving-camera footage for tests: a fixed textured world seen
through a window that pans by a known number of pixels per frame."""

import cv2
import numpy as np

H, W = 240, 640


def world_texture(height: int = H + 80, width: int = W + 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    tex = cv2.GaussianBlur(noise, (0, 0), 2.0)
    tex = cv2.normalize(tex, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(tex, cv2.COLOR_GRAY2BGR)


def frame_at(world: np.ndarray, offset_x: int, offset_y: int = 40) -> np.ndarray:
    """Camera view whose top-left corner sits at (offset_x, offset_y) in the
    world; a world point X shows up at image x = X - offset_x."""
    return world[offset_y:offset_y + H, offset_x:offset_x + W].copy()
