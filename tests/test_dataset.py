import json

import cv2
import numpy as np
import pytest
import torch

from gmc.dataset import (
    MotionLanguageDataset,
    build_training_data,
    find_future_frame,
    load_labels_with_ids,
)
from tests.synthetic import H, W, frame_at, world_texture

PAN = 4
N_FRAMES = 16


class FakeEncoder:
    """Deterministic stand-in for the sentence encoder."""

    model_name = "fake"

    def encode(self, texts, batch_size=32, show_progress_bar=False):
        return torch.stack([torch.full((8,), float(len(t))) for t in texts])


@pytest.fixture(scope="module")
def mini_refer_kitti(tmp_path_factory):
    """One sequence: a camera panning right over a static world, with track 1
    parked (static in the world) and track 2 driving right at 3 px/frame."""
    root = tmp_path_factory.mktemp("refer-kitti")
    world = world_texture()
    frames = root / "KITTI" / "training" / "image_02" / "0001"
    labels = root / "KITTI" / "labels_with_ids" / "image_02" / "0001"
    exprs = root / "expression" / "0001"
    for d in (frames, labels, exprs):
        d.mkdir(parents=True)
    for t in range(N_FRAMES):
        offset = 100 + PAN * t
        cv2.imwrite(str(frames / f"{t:06d}.png"), frame_at(world, offset))
        rows = []
        for tid, world_x in ((1, 300.0), (2, 200.0 + 3.0 * t)):
            x = world_x - offset
            rows.append(f"2 {tid} {x / W} {150 / H} {40 / W} {30 / H}")
        (labels / f"{t:06d}.txt").write_text("\n".join(rows) + "\n")
    all_frames = {str(t): [1] for t in range(N_FRAMES)}
    moving = {str(t): [2] for t in range(N_FRAMES)}
    for name, sentence, label in [
        ("parked", "left cars which are parking", all_frames),
        ("moving", "moving cars", moving),
        ("appearance", "black cars", all_frames),
        ("ungrouped", "cars which are stopped", all_frames),
    ]:
        (exprs / f"{name}.json").write_text(json.dumps({"sentence": sentence, "label": label}))
    return root


def test_find_future_frame():
    frames = [0, 1, 2, 5, 9, 30]
    assert find_future_frame(frames, 0, 2) == 2
    assert find_future_frame(frames, 2, 2) == 3  # frame 5, within 2 * gap
    assert find_future_frame(frames, 3, 2) == 4  # frame 9 is exactly 2 * gap ahead
    assert find_future_frame(frames, 3, 1) is None  # frame 9 is 4 > 2 * 1 ahead
    assert find_future_frame(frames, 4, 10) is None  # frame 30 is 21 > 2 * 10 ahead
    assert find_future_frame(frames, 4, 11) == 5
    assert find_future_frame(frames, 5, 2) is None


def test_load_labels_with_ids(mini_refer_kitti):
    labels = load_labels_with_ids(str(mini_refer_kitti / "KITTI" / "labels_with_ids" / "image_02" / "0001"))
    assert sorted(labels) == list(range(N_FRAMES))
    assert [r["track_id"] for r in labels[0]] == [1, 2]


def test_build_training_data(mini_refer_kitti, tmp_path):
    np.random.seed(0)
    motion, language, labels = build_training_data(str(mini_refer_kitti), ["0001"], FakeEncoder(),
                                                   cache_dir=str(tmp_path))
    # "black cars" has no motion keyword and "stopped" has no motion-type group
    assert sorted(set(labels.tolist())) == [2, 5]
    # the last two anchor frames have no frame >= 2 ahead, so they give no pair
    assert (labels == 2).sum() == N_FRAMES - 2 and (labels == 5).sum() == N_FRAMES - 2
    assert motion.shape == (2 * (N_FRAMES - 2), 12) and motion.dtype == np.float32
    assert language.shape == (len(labels), 8)

    jitter = 2.0 / W * 100 + 0.05  # +-2 px jitter, plus fit error
    parked, driving = motion[labels == 2], motion[labels == 5]
    first = 0  # anchor frame 0: all three gaps are defined
    for i, gap in enumerate((2, 5, 10)):
        assert abs(parked[first, 2 * i]) < jitter
        assert driving[first, 2 * i] == pytest.approx(3.0 * gap / W * 100, abs=jitter)

    np.random.seed(1)
    again = build_training_data(str(mini_refer_kitti), ["0001"], FakeEncoder(), cache_dir=str(tmp_path))
    np.testing.assert_array_equal(again[0], motion)  # served from the cache


def test_no_ego_keeps_camera_motion(mini_refer_kitti):
    np.random.seed(0)
    motion, _, labels = build_training_data(str(mini_refer_kitti), ["0001"], FakeEncoder(), compensate_ego=False)
    parked = motion[labels == 2]
    assert parked[0, 4] == pytest.approx(-PAN * 10 / W * 100, abs=2.0 / W * 100)


def test_dataset_items():
    ds = MotionLanguageDataset(np.zeros((3, 12)), np.ones((3, 384)), np.array([0, 1, 2]))
    m, lang, y = ds[1]
    assert len(ds) == 3 and m.shape == (12,) and lang.shape == (384,) and int(y) == 1
