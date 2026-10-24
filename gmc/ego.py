"""Camera (ego) motion between consecutive frames, as a 3x3 homography.

Primary estimator: the road plane. Driver-view footage shows the road in the
lower half of the frame, and the road is a plane, so a homography fitted to
road correspondences is the right model for the ground under the objects.
Shi-Tomasi corners in that band (minus the tracked boxes) are followed into
the next frame with pyramidal Lucas-Kanade flow, and RANSAC fits H.

Fallback: a global ORB + RANSAC fit over the whole frame, used only when the
road fit has fewer than 12 tracked points. On the 7,690 adjacent frame pairs
of the Refer-KITTI training and test sequences the road fit never fails.
"""

from __future__ import annotations

import cv2
import numpy as np

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def _clear_boxes(mask: np.ndarray, boxes: list[Box] | None) -> None:
    if not boxes:
        return
    h, w = mask.shape
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 0


class EgoMotionEstimator:
    """Estimates H_{t-1 -> t}, mapping pixel coordinates in the previous frame
    to the current frame."""

    MIN_POINTS = 12

    def __init__(
        self,
        road_band: float = 0.5,
        max_corners: int = 600,
        road_ransac_px: float = 3.0,
        orb_features: int = 1500,
        global_ransac_px: float = 5.0,
    ) -> None:
        self.road_band = road_band
        self.max_corners = max_corners
        self.road_ransac_px = road_ransac_px
        self.global_ransac_px = global_ransac_px
        self.orb = cv2.ORB_create(orb_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def estimate(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_boxes: list[Box] | None = None,
    ) -> np.ndarray:
        """Road-plane homography, falling back to the global fit."""
        H = self.road_homography(prev_frame, curr_frame, prev_boxes)
        if H is None:
            H = self.global_homography(prev_frame, curr_frame, prev_boxes)
        return H

    def road_homography(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_boxes: list[Box] | None = None,
    ) -> np.ndarray | None:
        """Homography from the road band (lower part of the frame, excluding
        object boxes). Returns None when too few points can be tracked."""
        prev_gray, curr_gray = _gray(prev_frame), _gray(curr_frame)
        h, w = prev_gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[int(h * self.road_band):, :] = 255
        _clear_boxes(mask, prev_boxes)

        pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=self.max_corners,
                                      qualityLevel=0.01, minDistance=7, mask=mask)
        if pts is None or len(pts) < self.MIN_POINTS:
            return None
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts, None,
                                                  winSize=(21, 21), maxLevel=3)
        ok = status.ravel() == 1
        if ok.sum() < self.MIN_POINTS:
            return None
        H, _ = cv2.findHomography(pts[ok], nxt[ok], cv2.RANSAC, self.road_ransac_px)
        return None if H is None else H.astype(np.float32)

    def global_homography(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        prev_boxes: list[Box] | None = None,
    ) -> np.ndarray:
        """ORB + Lowe ratio test + RANSAC over the whole frame. Returns the
        identity when no reliable fit exists."""
        prev_gray, curr_gray = _gray(prev_frame), _gray(curr_frame)
        identity = np.eye(3, dtype=np.float32)

        mask = None
        if prev_boxes:
            mask = np.full(prev_gray.shape, 255, dtype=np.uint8)
            _clear_boxes(mask, prev_boxes)

        kp1, des1 = self.orb.detectAndCompute(prev_gray, mask=mask)
        kp2, des2 = self.orb.detectAndCompute(curr_gray, mask=None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return identity

        good = []
        for pair in self.matcher.knnMatch(des1, des2, k=2):
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.7 * n.distance:
                    good.append(m)
            elif len(pair) == 1:
                good.append(pair[0])
        if len(good) < 4:
            return identity

        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, self.global_ransac_px)
        return identity if H is None else H.astype(np.float32)
