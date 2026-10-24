import numpy as np
import pytest
import torch

from gmc import GMCManager, MotionLanguageAligner
from gmc.geometry import VELOCITY_SCALE
from tests.synthetic import H, W, frame_at, world_texture

PAN = 4  # camera pans right by 4 px per frame


@pytest.fixture(scope="module")
def world():
    return world_texture()


def _box_at(world_x: float, offset_x: float, y: float = 150.0, w: float = 40.0, h: float = 30.0, tid: int = 1):
    """Track tuple for an object whose left edge sits at world_x."""
    return (tid, world_x - offset_x, y, w, h)


def _run(manager, world, n_frames, object_speed=0.0, pan=PAN):
    feats = None
    for t in range(n_frames):
        offset = 100 + pan * t
        feats = manager.update(frame_at(world, offset), [_box_at(300 + object_speed * t, offset)])
    return feats[1]


def test_static_object_under_camera_pan_has_near_zero_residual(world):
    feat = _run(GMCManager(), world, 12)
    assert np.abs(feat[:6]).max() < 0.1, feat[:6]


def test_raw_velocity_without_compensation_sees_camera_motion(world):
    feat = _run(GMCManager(compensate_ego=False), world, 12)
    # image displacement over gap g is -PAN * g pixels
    for i, gap in enumerate((2, 5, 10)):
        assert feat[2 * i] == pytest.approx(-PAN * gap / W * VELOCITY_SCALE, rel=1e-5)
        assert feat[2 * i + 1] == pytest.approx(0.0, abs=1e-6)


def test_moving_object_on_static_camera(world):
    feat = _run(GMCManager(), world, 12, object_speed=3.0, pan=0)
    for i, gap in enumerate((2, 5, 10)):
        assert feat[2 * i] == pytest.approx(3.0 * gap / W * VELOCITY_SCALE, rel=1e-3)


def test_moving_object_under_pan_recovers_own_motion(world):
    feat = _run(GMCManager(), world, 12, object_speed=3.0)
    for i, gap in enumerate((2, 5, 10)):
        assert feat[2 * i] == pytest.approx(3.0 * gap / W * VELOCITY_SCALE, abs=0.1)


def test_feature_layout(world):
    m = GMCManager()
    m.update(frame_at(world, 100), [(7, 64.0, 48.0, 32.0, 24.0)])
    feat = m.update(frame_at(world, 100), [(7, 64.0, 48.0, 32.0, 24.0)])[7]
    assert feat.shape == (12,) and feat.dtype == np.float32
    np.testing.assert_allclose(feat[:8], 0.0, atol=1e-5)
    np.testing.assert_allclose(feat[8:], [80 / W, 60 / H, 32 / W, 24 / H], rtol=1e-6)


def test_box_growth_gives_dw_dh_over_mid_gap(world):
    m = GMCManager()
    for t in range(7):
        feat = m.update(frame_at(world, 100), [(1, 100.0, 100.0, 40.0 + t, 30.0 + 2 * t)])[1]
    np.testing.assert_allclose(feat[6:8], [5 / W * VELOCITY_SCALE, 10 / H * VELOCITY_SCALE], rtol=1e-5)


def test_warmup_abstains_until_history_covers_largest_gap(world):
    m = GMCManager(MotionLanguageAligner())
    lang = torch.randn(1, 384)
    for t in range(11):
        scores = m.process_frame(frame_at(world, 100 + PAN * t), [_box_at(300, 100 + PAN * t)], lang)
        assert (1 in scores) == (t == 10)
        assert m.ready_ids == ({1} if t == 10 else set())
    assert -1.0 <= scores[1] <= 1.0


def test_missing_track_loses_history(world):
    m = GMCManager()
    for t in range(11):
        m.update(frame_at(world, 100), [(1, 50.0, 100.0, 30.0, 30.0), (2, 300.0, 100.0, 30.0, 30.0)])
    assert m.ready_ids == {1, 2}
    m.update(frame_at(world, 100), [(2, 300.0, 100.0, 30.0, 30.0)])
    assert m.ready_ids == {2}
    m.update(frame_at(world, 100), [(1, 50.0, 100.0, 30.0, 30.0), (2, 300.0, 100.0, 30.0, 30.0)])
    assert len(m.centroid_history[1]) == 1


def test_frame_without_tracks_is_skipped(world):
    m = GMCManager()
    m.update(frame_at(world, 100), [(1, 50.0, 100.0, 30.0, 30.0)])
    state = (len(m.ego_buffer), m.prev_frame.copy())
    assert m.update(frame_at(world, 140), []) == {}
    assert len(m.ego_buffer) == state[0]
    np.testing.assert_array_equal(m.prev_frame, state[1])


def test_ready_ids_match_offline_replay():
    """`ready_ids` must equal a replay of contiguous presence counts: a track
    is ready once seen in >= 11 consecutive processed frames."""
    rng = np.random.default_rng(3)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    m = GMCManager()
    counts: dict[int, int] = {}
    for _ in range(200):
        present = [tid for tid in range(6) if rng.random() < 0.93]
        if not present:
            continue
        m.update(frame, [(tid, 20.0 * tid, 50.0, 10.0, 10.0) for tid in present])
        counts = {tid: counts.get(tid, 0) + 1 for tid in present}
        assert m.ready_ids == {tid for tid, c in counts.items() if c >= 11}


def test_score_matches_aligner(world):
    aligner = MotionLanguageAligner()
    m = GMCManager(aligner)
    feats = m.update(frame_at(world, 100), [(1, 10.0, 10.0, 20.0, 20.0), (2, 200.0, 90.0, 50.0, 40.0)])
    lang = torch.randn(1, 384)
    scores = m.score(feats, lang)
    expected = aligner(torch.tensor(np.stack([feats[1], feats[2]])), lang).flatten()
    np.testing.assert_allclose([scores[1], scores[2]], expected.detach().numpy(), rtol=1e-6)


def test_score_without_weights_raises(world):
    m = GMCManager()
    feats = m.update(frame_at(world, 100), [(1, 10.0, 10.0, 20.0, 20.0)])
    with pytest.raises(RuntimeError):
        m.score(feats, torch.randn(1, 384))


def test_checkpoint_settings_must_match(tmp_path):
    path = tmp_path / "w.pth"
    torch.save({"model": MotionLanguageAligner().state_dict(), "frame_gaps": [5, 5, 5],
                "compensate_ego": True}, path)
    with pytest.raises(ValueError, match="gaps"):
        GMCManager(str(path))
    GMCManager(str(path), frame_gaps=(5, 5, 5))
