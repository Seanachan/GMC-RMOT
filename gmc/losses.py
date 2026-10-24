"""Contrastive loss for the motion-language aligner."""

import torch
import torch.nn.functional as F
from torch import nn


class InfoNCELoss(nn.Module):
    """Symmetric (CLIP-style) InfoNCE with a fixed temperature.

    Row i of the (B, B) similarity matrix is motion sample i against every
    language sample in the batch; the diagonal holds the positive pairs and
    every off-diagonal entry is a negative. The loss is the mean of the
    motion->language and language->motion cross-entropies.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        logits = sim_matrix / self.temperature
        targets = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        m2l = F.cross_entropy(logits, targets)
        l2m = F.cross_entropy(logits.t(), targets)
        return (m2l + l2m) / 2.0
