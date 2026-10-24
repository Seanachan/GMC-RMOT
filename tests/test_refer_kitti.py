"""Tests on the real Refer-KITTI data. Skipped unless REFER_KITTI_ROOT points
at a Refer-KITTI (V1) copy; the dataset cannot be redistributed, so CI skips
them."""

import json
import os

import cv2
import numpy as np
import pytest
import torch

from gmc import EgoMotionEstimator, GMCManager, MotionLanguageAligner
from gmc.dataset import build_training_data, load_labels_with_ids
from tests.test_dataset import FakeEncoder

ROOT = os.environ.get("REFER_KITTI_ROOT", "")
pytestmark = pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, "expression")),
                                reason="needs Refer-KITTI: set REFER_KITTI_ROOT")
SEQMAP = os.path.join(os.path.dirname(__file__), "..", "seqmaps", "refer_kitti_v1_test_official_150.txt")


def _frames(seq):
    d = os.path.join(ROOT, "KITTI", "training", "image_02", seq)
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".png")]


def test_official_test_expressions_exist():
    with open(SEQMAP) as f:
        for line in f.read().split():
            seq, expr = line.split("+", 1)
            assert os.path.exists(os.path.join(ROOT, "expression", seq, expr + ".json")), line


@pytest.mark.parametrize("seq", ["0005", "0011", "0013"])
def test_road_fit_never_falls_back_on_test_sequences(seq):
    ego, prev = EgoMotionEstimator(), None
    for path in _frames(seq):
        curr = cv2.imread(path)
        if prev is not None:
            assert ego.road_homography(prev, curr) is not None, path
        prev = curr


def _speeds(seq, expression):
    """Median long-gap speed of the referred tracks: (raw, compensated)."""
    labels = load_labels_with_ids(os.path.join(ROOT, "KITTI", "labels_with_ids", "image_02", seq))
    with open(os.path.join(ROOT, "expression", seq, expression + ".json")) as f:
        referred = {int(k): set(v) for k, v in json.load(f)["label"].items()}
    raw, comp = GMCManager(compensate_ego=False), GMCManager()
    raw_speed, comp_speed = [], []
    for fid, path in enumerate(_frames(seq)):
        frame = cv2.imread(path)
        h, w = frame.shape[:2]
        tracks = [(r["track_id"], r["x1_n"] * w, r["y1_n"] * h, r["w_n"] * w, r["h_n"] * h)
                  for r in labels.get(fid, [])]
        f_raw, f_comp = raw.update(frame, tracks), comp.update(frame, tracks)
        for tid in referred.get(fid, set()) & comp.ready_ids:
            raw_speed.append(np.hypot(*f_raw[tid][4:6]))
            comp_speed.append(np.hypot(*f_comp[tid][4:6]))
    assert len(comp_speed) > 50
    return np.median(raw_speed), np.median(comp_speed)


def test_compensation_moves_apparent_speed_toward_true_motion():
    """Seq 0005: in raw image motion the parked cars look faster than the
    moving ones (the camera drives past the former and follows the latter).
    Compensation shrinks the parked cars' speed and raises the moving cars'.
    It does not reach zero for parked cars: box centres sit above the road
    plane, so some parallax remains."""
    parked_raw, parked_comp = _speeds("0005", "left-cars-which-are-parking")
    moving_raw, moving_comp = _speeds("0005", "moving-cars")
    assert parked_raw > moving_raw
    assert parked_comp < 0.75 * parked_raw
    assert moving_comp > 1.5 * moving_raw


def test_training_pairs_on_one_sequence():
    np.random.seed(0)
    motion, language, labels = build_training_data(ROOT, ["0001"], FakeEncoder())
    assert len(motion) > 1000
    assert np.isfinite(motion).all()
    assert set(labels.tolist()) <= set(range(6))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_cuda_scores_match_cpu():
    torch.manual_seed(0)
    a_cpu = MotionLanguageAligner()
    a_gpu = MotionLanguageAligner()
    a_gpu.load_state_dict(a_cpu.state_dict())
    cpu, gpu = GMCManager(a_cpu), GMCManager(a_gpu, device="cuda")
    tracks = [(1, 100.0, 180.0, 60.0, 40.0), (2, 700.0, 170.0, 80.0, 50.0)]
    lang = torch.randn(1, 384)
    for path in _frames("0005")[:12]:
        frame = cv2.imread(path)
        s_cpu = cpu.score(cpu.update(frame, tracks), lang)
        s_gpu = gpu.score(gpu.update(frame, tracks), lang)
    for tid in s_cpu:
        assert s_gpu[tid] == pytest.approx(s_cpu[tid], abs=1e-5)
