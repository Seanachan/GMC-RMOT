"""Online inference: per-track residual-motion features and match scores.

For every frame the manager

1. estimates the camera motion H_{t-1 -> t} (gmc/ego.py) and composes it into
   accumulated homographies H_{t-k -> t} for k up to the largest gap;
2. keeps each track's *original* (never warped) centroid history and warps
   the old centroid once with the accumulated homography;
3. builds the 12-D motion feature
       [dx_s, dy_s, dx_m, dy_m, dx_l, dy_l, dw, dh, cx, cy, w, h]
   where (dx_g, dy_g) is the residual velocity over gap g in {2, 5, 10}:
   raw centroid displacement minus the displacement the camera alone would
   have caused (Eq. 2 of the paper);
4. scores the feature against an expression embedding with the aligner and
   returns the raw cosine similarity s_gmc in [-1, 1].

A track gets a score only once its history covers the largest gap (11
contiguous frames). Before that the module abstains and the host score
passes through unchanged; see `GMCManager.ready_ids`.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .aligner import MotionLanguageAligner, load_aligner
from .ego import EgoMotionEstimator
from .geometry import VELOCITY_SCALE, normalize_velocity, warp_points

FRAME_GAPS = (2, 5, 10)

# (track_id, x, y, w, h): top-left corner and size in pixels, MOT format.
Track = tuple[int, float, float, float, float]


class GMCManager:
    def __init__(
        self,
        weights: str | MotionLanguageAligner | None = None,
        device: str = "cpu",
        frame_gaps: tuple[int, ...] = FRAME_GAPS,
        compensate_ego: bool = True,
        ego: EgoMotionEstimator | None = None,
    ) -> None:
        """
        Args:
            weights: checkpoint path from gmc.train, an aligner instance, or
                None (features only; `score` is unavailable).
            frame_gaps: (short, mid, long) gaps. dw/dh use the mid gap.
            compensate_ego: False keeps the raw image-plane velocity (the
                "- ego" ablation). The aligner must be trained the same way.
        """
        if len(frame_gaps) != 3:
            raise ValueError("frame_gaps must have exactly three entries")
        self.device = device
        self.frame_gaps = tuple(frame_gaps)
        self.compensate_ego = compensate_ego
        self.ego = ego or EgoMotionEstimator()
        if isinstance(weights, str):
            self.aligner = load_aligner(weights, device)
        elif weights is not None:
            self.aligner = weights.to(device).eval()
        else:
            self.aligner = None
        config = getattr(self.aligner, "train_config", {})
        if "frame_gaps" in config and tuple(config["frame_gaps"]) != self.frame_gaps:
            raise ValueError(f"weights were trained with gaps {config['frame_gaps']}, got {self.frame_gaps}")
        if "compensate_ego" in config and bool(config["compensate_ego"]) != compensate_ego:
            raise ValueError(f"weights were trained with compensate_ego={config['compensate_ego']}")
        self.history_len = max(self.frame_gaps) + 1
        self.reset()

    def reset(self) -> None:
        """Forget all tracks and camera motion (call between videos)."""
        self.prev_frame = None
        self.prev_boxes = None
        # ego_buffer[i] maps frame (t - (len-1-i)) to the current frame t.
        self.ego_buffer: deque = deque(maxlen=self.history_len)
        self.centroid_history: dict[int, deque] = {}
        self.size_history: dict[int, deque] = {}

    @property
    def ready_ids(self) -> set[int]:
        """Tracks whose history covers the largest gap, i.e. whose features
        are fully defined. Only these are scored by `process_frame`."""
        return {tid for tid, hist in self.centroid_history.items() if len(hist) == self.history_len}

    def update(self, frame: np.ndarray, tracks: list[Track]) -> dict[int, np.ndarray]:
        """Advance one frame. Returns {track_id: 12-D float32 feature} for
        every track in `tracks` (features of tracks that are not yet in
        `ready_ids` are only partially defined).

        A frame with no tracks is skipped entirely: camera motion is then
        composed across it on the next call.
        """
        if not tracks:
            return {}

        img_h, img_w = frame.shape[:2]
        frame_shape = (img_h, img_w)
        boxes = [(x, y, x + w, y + h) for _, x, y, w, h in tracks]

        # Accumulate camera motion: every stored H_{t-k -> t-1} becomes
        # H_{t-k -> t}, and the current frame enters with the identity.
        if self.prev_frame is not None:
            step = self.ego.estimate(self.prev_frame, frame, self.prev_boxes)
            composed = deque(maxlen=self.history_len)
            for H_old in self.ego_buffer:
                composed.append(step @ H_old)
            composed.append(np.eye(3, dtype=np.float32))
            self.ego_buffer = composed
        else:
            self.ego_buffer.append(np.eye(3, dtype=np.float32))
        self.prev_frame = frame.copy()
        self.prev_boxes = boxes

        short, mid, long = self.frame_gaps
        features: dict[int, np.ndarray] = {}
        for (tid, x, y, w, h), (x1, y1, x2, y2) in zip(tracks, boxes):
            centroid = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)
            size = np.array([x2 - x1, y2 - y1], dtype=np.float64)
            if tid not in self.centroid_history:
                self.centroid_history[tid] = deque(maxlen=self.history_len)
                self.size_history[tid] = deque(maxlen=self.history_len)
            self.centroid_history[tid].append(centroid)
            self.size_history[tid].append(size)

            pts = list(self.centroid_history[tid])
            sizes = list(self.size_history[tid])
            T = len(pts)
            velocity = np.zeros(8, dtype=np.float32)
            if T > 1:
                ego_Hs = list(self.ego_buffer)[-T:]
                while len(ego_Hs) < T:
                    ego_Hs.insert(0, np.eye(3, dtype=np.float32))

                residuals = []
                for gap in self.frame_gaps:
                    if T > gap:
                        old = pts[-(gap + 1)]
                        v_raw = normalize_velocity(pts[-1] - old, frame_shape)
                        if self.compensate_ego:
                            warped = warp_points(np.array([old]), ego_Hs[T - 1 - gap])[0]
                            v_ego = normalize_velocity(warped - old, frame_shape)
                            residuals.append(v_raw - v_ego)
                        else:
                            residuals.append(v_raw)
                    else:
                        residuals.append(np.zeros(2, dtype=np.float32))

                d_size = sizes[-1] - sizes[-(mid + 1)] if T > mid else sizes[-1] - sizes[0]
                velocity = np.array([
                    residuals[0][0], residuals[0][1],
                    residuals[1][0], residuals[1][1],
                    residuals[2][0], residuals[2][1],
                    d_size[0] / float(img_w) * VELOCITY_SCALE,
                    d_size[1] / float(img_h) * VELOCITY_SCALE,
                ], dtype=np.float32)

            features[tid] = np.array([
                *velocity,
                centroid[0] / float(img_w), centroid[1] / float(img_h),
                size[0] / float(img_w), size[1] / float(img_h),
            ], dtype=np.float32)

        # A track missing from this frame loses its history.
        for tid in set(self.centroid_history) - set(features):
            del self.centroid_history[tid]
            del self.size_history[tid]
        return features

    @torch.no_grad()
    def embed_motion(self, features: dict[int, np.ndarray]) -> tuple[list[int], torch.Tensor]:
        """Aligner embeddings (N, 256) for a frame's features, in track order."""
        if self.aligner is None:
            raise RuntimeError("GMCManager was created without weights")
        ids = list(features)
        motion = torch.tensor(np.array([features[t] for t in ids]), dtype=torch.float32).to(self.device)
        return ids, self.aligner.encode_motion(motion)

    @torch.no_grad()
    def embed_lang(self, lang_embedding: torch.Tensor) -> torch.Tensor:
        """(1, 384) sentence embedding -> (1, 256) aligner embedding."""
        if self.aligner is None:
            raise RuntimeError("GMCManager was created without weights")
        return self.aligner.encode_lang(lang_embedding.to(self.device))

    def score(self, features: dict[int, np.ndarray], lang_embedding: torch.Tensor) -> dict[int, float]:
        """Raw cosine s_gmc for every track in `features`."""
        if not features:
            return {}
        ids, motion_emb = self.embed_motion(features)
        cos = (motion_emb @ self.embed_lang(lang_embedding).t()).flatten().cpu().numpy()
        return {tid: float(c) for tid, c in zip(ids, cos)}

    def process_frame(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        lang_embedding: torch.Tensor,
    ) -> dict[int, float]:
        """`update` + `score`, restricted to `ready_ids`. Tracks without a
        score should keep their host score unchanged."""
        scores = self.score(self.update(frame, tracks), lang_embedding)
        ready = self.ready_ids
        return {tid: s for tid, s in scores.items() if tid in ready}
