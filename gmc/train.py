"""Train the motion-language aligner on the Refer-KITTI V1 training split.

    python -m gmc.train --data-root /path/to/refer-kitti --seed 0 \
        --save-path weights/gmc_aligner_seed0.pth

Paper setting: AdamW, lr 1e-3 with cosine decay to 1e-5, batch 256, 100
epochs, symmetric InfoNCE at temperature 0.07, seeds 0/1/2.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from .aligner import MotionLanguageAligner
from .dataset import V1_TRAIN_SEQS, MotionLanguageDataset, build_training_data
from .losses import InfoNCELoss
from .manager import FRAME_GAPS
from .text import DEFAULT_TEXT_ENCODER, TextEncoder


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> tuple[float, float]:
    """Returns (mean loss, motion->language retrieval accuracy), where a
    retrieval counts as correct if the nearest sentence in the batch has the
    same motion-type label."""
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for motion, lang, labels in loader:
        motion, lang, labels = motion.to(device), lang.to(device), labels.to(device)
        sim = model(motion, lang)
        loss = loss_fn(sim)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            correct += (labels[sim.argmax(dim=1)] == labels).sum().item()
            seen += labels.numel()
    return total_loss / len(loader), correct / max(seen, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Train the GMC motion-language aligner")
    p.add_argument("--data-root", required=True, help="Refer-KITTI (V1) root directory")
    p.add_argument("--save-path", default="weights/gmc_aligner.pth")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--text-encoder", default=DEFAULT_TEXT_ENCODER)
    p.add_argument("--gaps", default=",".join(map(str, FRAME_GAPS)),
                   help="three frame gaps; '5,5,5' is the single-gap ablation")
    p.add_argument("--no-ego", action="store_true",
                   help="raw image-plane velocity, no camera-motion compensation (ablation)")
    p.add_argument("--cache-dir", default="cache", help="training-pair cache ('' to disable)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    gaps = tuple(int(g) for g in args.gaps.split(","))
    print(f"device={device} seed={args.seed} gaps={gaps} ego={'off' if args.no_ego else 'road'}")

    encoder = TextEncoder(args.text_encoder, device=str(device))
    motion, language, labels = build_training_data(
        args.data_root, V1_TRAIN_SEQS, encoder,
        frame_gaps=gaps, compensate_ego=not args.no_ego, cache_dir=args.cache_dir or None,
    )
    loader = DataLoader(
        MotionLanguageDataset(motion, language, labels),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
        pin_memory=device.type == "cuda",
    )

    model = MotionLanguageAligner(motion_dim=motion.shape[1], lang_dim=language.shape[1]).to(device)
    loss_fn = InfoNCELoss(args.temperature)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history = []
    for epoch in range(args.epochs):
        loss, acc = train_one_epoch(model, loader, optimizer, loss_fn, device)
        scheduler.step()
        history.append({"epoch": epoch + 1, "loss": loss, "acc": acc})
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:3d}/{args.epochs}  loss {loss:.4f}  acc {acc:.2%}  "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}")

    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "motion_dim": int(motion.shape[1]),
        "lang_dim": int(language.shape[1]),
        "text_encoder": args.text_encoder,
        "frame_gaps": list(gaps),
        "compensate_ego": not args.no_ego,
        "seed": args.seed,
    }, args.save_path)
    with open(os.path.splitext(args.save_path)[0] + "_log.json", "w") as f:
        json.dump(history, f, indent=1)
    print(f"saved {args.save_path}")


if __name__ == "__main__":
    main()
