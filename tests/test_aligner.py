import torch

from gmc import MotionLanguageAligner, load_aligner
from gmc.losses import InfoNCELoss

# Parameter names of the released checkpoints; renaming any of them breaks loading.
CHECKPOINT_KEYS = {
    "motion_adapter.weight", "motion_adapter.bias",
    "lang_adapter.weight", "lang_adapter.bias",
    "shared_weight.0.weight", "shared_weight.0.bias",
    "shared_weight.2.weight", "shared_weight.2.bias",
    "shared_weight.4.weight", "shared_weight.4.bias",
    "shared_weight.5.weight", "shared_weight.5.bias",
}


def test_state_dict_keys_are_stable():
    assert set(MotionLanguageAligner().state_dict()) == CHECKPOINT_KEYS


def test_parameter_count():
    n = sum(p.numel() for p in MotionLanguageAligner().parameters())
    assert n == (12 * 256 + 256) + (384 * 256 + 256) + (256 * 512 + 512) + (512 * 512 + 512) \
        + (512 * 256 + 256) + 2 * 256


def test_embeddings_are_unit_norm_and_shared_space():
    torch.manual_seed(0)
    model = MotionLanguageAligner().eval()
    m, lang = model.encode(torch.randn(5, 12), torch.randn(3, 384))
    assert m.shape == (5, 256) and lang.shape == (3, 256)
    torch.testing.assert_close(m.norm(dim=1), torch.ones(5))
    torch.testing.assert_close(lang.norm(dim=1), torch.ones(3))
    sim = model(torch.randn(5, 12), torch.randn(3, 384))
    assert sim.shape == (5, 3) and sim.abs().max() <= 1.0 + 1e-6


def test_load_aligner_roundtrip(tmp_path):
    model = MotionLanguageAligner()
    path = tmp_path / "ckpt.pth"
    torch.save({"model": model.state_dict(), "motion_dim": 12, "lang_dim": 384, "seed": 2}, path)
    loaded = load_aligner(str(path))
    assert not loaded.training
    assert loaded.train_config["seed"] == 2
    x, lang = torch.randn(4, 12), torch.randn(2, 384)
    torch.testing.assert_close(loaded(x, lang), model.eval()(x, lang))


def test_infonce_matches_manual_cross_entropy():
    sim = torch.tensor([[0.9, 0.1, -0.2], [0.0, 0.8, 0.3], [0.2, -0.1, 0.7]])
    logits = sim / 0.07
    t = torch.arange(3)
    expected = (torch.nn.functional.cross_entropy(logits, t) + torch.nn.functional.cross_entropy(logits.t(), t)) / 2
    torch.testing.assert_close(InfoNCELoss(0.07)(sim), expected)


def test_infonce_prefers_aligned_pairs():
    loss = InfoNCELoss()
    aligned = torch.eye(8) * 2 - 1
    shuffled = aligned[torch.randperm(8, generator=torch.Generator().manual_seed(0))]
    assert loss(aligned) < loss(shuffled)


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    model = MotionLanguageAligner()
    motion, lang = torch.randn(32, 12), torch.randn(32, 384)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = InfoNCELoss()
    first = loss_fn(model(motion, lang)).item()
    for _ in range(30):
        opt.zero_grad()
        loss = loss_fn(model(motion, lang))
        loss.backward()
        opt.step()
    assert loss.item() < first
