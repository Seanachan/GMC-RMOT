"""Motion-language aligner: projects a 12-D motion feature and a sentence
embedding into one 256-D space, where cosine similarity is the match score."""

import torch
import torch.nn.functional as F
from torch import nn


class MotionLanguageAligner(nn.Module):
    """Two linear adapters (motion 12 -> 256, language 384 -> 256) feeding one
    shared MLP (256 -> 512 -> 512 -> 256, LayerNorm), then L2 normalization.

    The language encoder itself (MiniLM) stays frozen; only this module is
    trained, with a symmetric InfoNCE loss (see gmc/losses.py).
    """

    def __init__(self, motion_dim: int = 12, lang_dim: int = 384, embed_dim: int = 256) -> None:
        super().__init__()
        self.motion_dim = motion_dim
        self.lang_dim = lang_dim
        self.embed_dim = embed_dim
        self.motion_adapter = nn.Linear(motion_dim, embed_dim)
        self.lang_adapter = nn.Linear(lang_dim, embed_dim)
        self.shared_weight = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def encode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        """(N, motion_dim) -> (N, embed_dim), unit norm."""
        return F.normalize(self.shared_weight(self.motion_adapter(motion)), p=2, dim=-1)

    def encode_lang(self, lang: torch.Tensor) -> torch.Tensor:
        """(M, lang_dim) -> (M, embed_dim), unit norm."""
        return F.normalize(self.shared_weight(self.lang_adapter(lang)), p=2, dim=-1)

    def encode(self, motion: torch.Tensor, lang: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_motion(motion), self.encode_lang(lang)

    def forward(self, motion: torch.Tensor, lang: torch.Tensor) -> torch.Tensor:
        """Cosine similarity matrix (N, M) in [-1, 1]."""
        motion_emb, lang_emb = self.encode(motion, lang)
        return motion_emb @ lang_emb.t()


def load_aligner(path: str, device: str = "cpu") -> MotionLanguageAligner:
    """Load a checkpoint written by gmc.train: {"model": state_dict, plus the
    settings it was trained with}. The settings are kept on `.train_config`."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    config = {k: v for k, v in ckpt.items() if k != "model"}
    model = MotionLanguageAligner(
        motion_dim=int(config.get("motion_dim", 12)),
        lang_dim=int(config.get("lang_dim", 384)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.train_config = config
    return model
