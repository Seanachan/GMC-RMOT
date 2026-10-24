"""Training pairs for the aligner, built from Refer-KITTI ground truth.

For every training expression that mentions motion, each referred object at
each frame gives one positive pair (12-D motion feature, sentence embedding).
Negatives come from the rest of the mini-batch, so only positives are stored.

The feature matches gmc/manager.py, with two differences that make it a
training signal rather than an online one: velocities look *forward* from the
anchor frame (frame t to t + g, the next annotated frame at least g ahead and
at most 2g ahead), and the target point gets +-2 px uniform jitter.

Expected layout under `data_root` (the Refer-KITTI release):

    expression/<seq>/*.json                          {"sentence", "label": {frame: [track ids]}}
    KITTI/labels_with_ids/image_02/<seq>/<frame>.txt class track x1 y1 w h  (normalized)
    KITTI/training/image_02/<seq>/<frame:06d>.png
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .ego import EgoMotionEstimator
from .expressions import MOTION_TYPE_NAMES, is_motion_expression, motion_type_group
from .geometry import VELOCITY_SCALE, warp_points
from .manager import FRAME_GAPS

KITTI_FRAME_SHAPE = (375, 1242)

V1_TRAIN_SEQS = [
    "0001", "0002", "0003", "0004", "0006",
    "0007", "0008", "0009", "0010", "0012",
    "0014", "0015", "0016", "0018", "0020",
]


class MotionLanguageDataset(Dataset):
    def __init__(self, motion: np.ndarray, language: np.ndarray, labels: np.ndarray) -> None:
        assert len(motion) == len(language) == len(labels)
        self.motion = torch.as_tensor(motion, dtype=torch.float32)
        self.language = torch.as_tensor(language, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return self.motion[idx], self.language[idx], self.labels[idx]


# ------------------------------------------------------------------ loaders


def load_expressions(expression_dir: str) -> list[dict]:
    out = []
    for name in sorted(os.listdir(expression_dir)):
        if name.endswith(".json"):
            with open(os.path.join(expression_dir, name), encoding="utf-8") as f:
                out.append(json.load(f))
    return out


def load_labels_with_ids(labels_dir: str) -> dict[int, list[dict]]:
    """{frame: [{class_id, track_id, x1_n, y1_n, w_n, h_n}, ...]}"""
    labels = {}
    for name in sorted(os.listdir(labels_dir)):
        if not name.endswith(".txt"):
            continue
        rows = []
        with open(os.path.join(labels_dir, name), encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if parts:
                    rows.append({
                        "class_id": int(parts[0]),
                        "track_id": int(parts[1]),
                        "x1_n": float(parts[2]),
                        "y1_n": float(parts[3]),
                        "w_n": float(parts[4]),
                        "h_n": float(parts[5]),
                    })
        labels[int(os.path.splitext(name)[0])] = rows
    return labels


def target_boxes(
    labels_by_frame: dict[int, list[dict]],
    label_map: dict[str, list[int]],
    frame_shape: tuple[int, int],
) -> dict[int, dict[int, tuple[float, float, float, float]]]:
    """{track_id: {frame: (cx, cy, w, h)}} in pixels, for the tracks an
    expression refers to."""
    h, w = frame_shape
    out: dict[int, dict] = {}
    for fid in sorted(int(f) for f in label_map):
        wanted = set(label_map[str(fid)])
        for det in labels_by_frame.get(fid, []):
            tid = det["track_id"]
            if tid in wanted:
                x1, y1 = det["x1_n"] * w, det["y1_n"] * h
                bw, bh = det["w_n"] * w, det["h_n"] * h
                out.setdefault(tid, {})[fid] = (x1 + bw / 2.0, y1 + bh / 2.0, bw, bh)
    return out


# ------------------------------------------------------------------ features


def find_future_frame(frames: list[int], start: int, gap: int) -> int | None:
    """Index of the first annotated frame at least `gap` after frames[start],
    or None if there is none within 2 * gap."""
    curr = frames[start]
    for j in range(start + 1, len(frames)):
        if frames[j] >= curr + gap:
            return None if frames[j] - curr > 2 * gap else j
    return None


class RoadChain:
    """Camera motion between two frames of one sequence, composed from cached
    frame-to-frame road-plane homographies (no box masking at train time)."""

    def __init__(self, frame_dir: str, ego: EgoMotionEstimator) -> None:
        self.frame_dir = frame_dir
        self.ego = ego
        self.steps: dict[int, np.ndarray | None] = {}
        self.spans: dict[tuple[int, int], np.ndarray | None] = {}

    def _step(self, a: int) -> np.ndarray | None:
        if a not in self.steps:
            ia = cv2.imread(os.path.join(self.frame_dir, f"{a:06d}.png"))
            ib = cv2.imread(os.path.join(self.frame_dir, f"{a + 1:06d}.png"))
            self.steps[a] = None if ia is None or ib is None else self.ego.estimate(ia, ib)
        return self.steps[a]

    def between(self, start: int, end: int) -> np.ndarray | None:
        if (start, end) not in self.spans:
            H = np.eye(3, dtype=np.float32)
            for a in range(start, end):
                step = self._step(a)
                if step is None:
                    H = None
                    break
                H = step @ H
            self.spans[(start, end)] = H
        return self.spans[(start, end)]


def residual_velocity(
    boxes: dict[int, tuple[float, float, float, float]],
    start: int,
    end: int,
    frame_shape: tuple[int, int],
    chain: RoadChain | None,
    compensate_ego: bool = True,
) -> tuple[float, float]:
    """Normalized residual velocity of one track from frame `start` to `end`."""
    h, w = frame_shape
    cx1, cy1, _, _ = boxes[start]
    cx2, cy2, _, _ = boxes[end]
    jx = cx2 + np.random.uniform(-2.0, 2.0)
    jy = cy2 + np.random.uniform(-2.0, 2.0)
    raw_dx = (jx - cx1) / w * VELOCITY_SCALE
    raw_dy = (jy - cy1) / h * VELOCITY_SCALE
    if not compensate_ego:
        return raw_dx, raw_dy
    ego_dx, ego_dy = 0.0, 0.0
    H = chain.between(start, end) if chain is not None else None
    if H is not None:
        warped = warp_points(np.array([[cx1, cy1]], dtype=np.float32), H)[0]
        ego_dx = (warped[0] - cx1) / w * VELOCITY_SCALE
        ego_dy = (warped[1] - cy1) / h * VELOCITY_SCALE
    return raw_dx - ego_dx, raw_dy - ego_dy


def positive_pairs(
    track_boxes: dict[int, dict[int, tuple[float, float, float, float]]],
    frame_shape: tuple[int, int],
    chain: RoadChain | None,
    frame_gaps: tuple[int, ...] = FRAME_GAPS,
    compensate_ego: bool = True,
) -> list[np.ndarray]:
    """One 12-D feature per (track, anchor frame) with at least one valid gap."""
    h, w = frame_shape
    mid = frame_gaps[1]
    out = []
    for boxes in track_boxes.values():
        frames = sorted(boxes)
        for i, fid in enumerate(frames):
            vel, any_valid = [], False
            for gap in frame_gaps:
                j = find_future_frame(frames, i, gap)
                if j is None:
                    vel.append((0.0, 0.0))
                else:
                    vel.append(residual_velocity(boxes, fid, frames[j], frame_shape, chain, compensate_ego))
                    any_valid = True
            if not any_valid:
                continue

            cx1, cy1, bw1, bh1 = boxes[fid]
            j = find_future_frame(frames, i, mid)
            if j is not None:
                _, _, bw2, bh2 = boxes[frames[j]]
                dw = (bw2 + np.random.uniform(-2.0, 2.0) - bw1) / w * VELOCITY_SCALE
                dh = (bh2 + np.random.uniform(-2.0, 2.0) - bh1) / h * VELOCITY_SCALE
            else:
                dw, dh = 0.0, 0.0

            out.append(np.array([
                vel[0][0], vel[0][1], vel[1][0], vel[1][1], vel[2][0], vel[2][1],
                dw, dh, cx1 / w, cy1 / h, bw1 / w, bh1 / h,
            ], dtype=np.float32))
    return out


# ------------------------------------------------------------------ builder


def _cache_path(cache_dir: str, key: dict) -> str:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"train_{digest}.npz")


def build_training_data(
    data_root: str,
    sequences: list[str],
    text_encoder: Any,
    frame_gaps: tuple[int, ...] = FRAME_GAPS,
    compensate_ego: bool = True,
    cache_dir: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (motion (N, 12), language (N, D), motion-type label (N,)).

    With `cache_dir`, the result is stored once per configuration and reused
    (the jitter drawn on the first build is then shared by later runs).
    """
    key = {
        "data_root": os.path.abspath(data_root),
        "sequences": sorted(sequences),
        "frame_gaps": list(frame_gaps),
        "compensate_ego": compensate_ego,
        "text_encoder": getattr(text_encoder, "model_name", "unknown"),
    }
    if cache_dir:
        path = _cache_path(cache_dir, key)
        if os.path.exists(path):
            print(f"  [cache] loading {path}")
            d = np.load(path)
            return d["motion"], d["language"], d["labels"]

    expressions = []
    for seq in sequences:
        expr_dir = os.path.join(data_root, "expression", seq)
        if not os.path.isdir(expr_dir):
            print(f"  skipping {seq}: no expression directory")
            continue
        for expr in load_expressions(expr_dir):
            expressions.append({"sentence": expr["sentence"], "label": expr["label"], "seq": seq})

    sentences = sorted({e["sentence"] for e in expressions})
    print(f"  encoding {len(sentences)} unique sentences")
    emb = text_encoder.encode(sentences, batch_size=64, show_progress_bar=True)
    sentence_emb = {s: emb[i].cpu().numpy() for i, s in enumerate(sentences)}

    expressions = [e for e in expressions if is_motion_expression(e["sentence"])]
    expressions.sort(key=lambda e: e["seq"])
    print(f"  {len(expressions)} motion expressions")

    ego = EgoMotionEstimator()
    seq_state: dict[str, tuple] = {}
    motion, language, labels = [], [], []
    for expr in expressions:
        group = motion_type_group(expr["sentence"])
        if group is None:
            continue
        seq = expr["seq"]
        if seq not in seq_state:
            frame_dir = os.path.join(data_root, "KITTI", "training", "image_02", seq)
            shape = KITTI_FRAME_SHAPE
            pngs = [f for f in os.listdir(frame_dir) if f.endswith(".png")] if os.path.isdir(frame_dir) else []
            if pngs:
                sample = cv2.imread(os.path.join(frame_dir, pngs[0]))
                if sample is not None:
                    shape = sample.shape[:2]
            labels_dir = os.path.join(data_root, "KITTI", "labels_with_ids", "image_02", seq)
            seq_labels = load_labels_with_ids(labels_dir) if os.path.isdir(labels_dir) else {}
            seq_state[seq] = (shape, RoadChain(frame_dir, ego), seq_labels)
        shape, chain, seq_labels = seq_state[seq]

        boxes = target_boxes(seq_labels, expr["label"], shape)
        feats = positive_pairs(boxes, shape, chain, frame_gaps, compensate_ego)
        motion.extend(feats)
        language.extend([sentence_emb[expr["sentence"]]] * len(feats))
        labels.extend([group] * len(feats))

    counts = {MOTION_TYPE_NAMES[g]: labels.count(g) for g in sorted(set(labels))}
    print(f"  {len(labels)} positive pairs, by motion type: {counts}")
    motion_arr = np.stack(motion).astype(np.float32)
    language_arr = np.stack(language).astype(np.float32)
    labels_arr = np.array(labels, dtype=np.int64)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(path, motion=motion_arr, language=language_arr, labels=labels_arr)
        print(f"  [cache] saved {path}")
    return motion_arr, language_arr, labels_arr
