"""Compute s_gmc for every (expression, frame, track) of the Refer-KITTI V1
test sequences, on the host tracker's tracks (NeuralSORT for iKUN).

    python scripts/build_gmc_scores.py --data-root /data/refer-kitti \
        --tracks /data/NeuralSORT --weights weights/gmc_aligner_seed0.pth \
        --out scores/seed0

Writes <out>/<seq>.json = {expression: {frame: {track_id: s_gmc}}}. Tracks the
module abstains on (history shorter than 11 frames) are absent; the fusion
step then keeps the host score.

Camera motion and motion features do not depend on the expression, so they
are computed once per sequence and every expression is scored against them.
"""

import argparse
import json
import os
import sys

import cv2
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gmc import GMCManager  # noqa: E402
from gmc.io import load_neuralsort  # noqa: E402
from gmc.text import DEFAULT_TEXT_ENCODER, TextEncoder  # noqa: E402

TEST_SEQS = ["0005", "0011", "0013"]


def build(seq, args, encoder, device) -> None:
    out_path = os.path.join(args.out, f"{seq}.json")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"[{seq}] {out_path} exists, skipping")
        return

    tracks = load_neuralsort(args.tracks, seq)
    expr_dir = os.path.join(args.data_root, "expression", seq)
    expressions = sorted(f[:-5] for f in os.listdir(expr_dir) if f.endswith(".json"))
    frame_dir = os.path.join(args.frames or os.path.join(args.data_root, "KITTI", "training", "image_02"), seq)
    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith((".png", ".jpg")))

    # Pass 1: camera motion + motion embeddings, once per sequence.
    manager = GMCManager(args.weights, device=device)
    per_frame = []  # (frame id, track ids, motion embeddings (N, 256), ready ids)
    for f0, name in enumerate(tqdm(frame_files, desc=f"{seq} motion")):
        fid = f0 + 1  # NeuralSORT frames are 1-indexed
        dets = tracks.get(fid, [])
        if not dets:
            continue
        frame = cv2.imread(os.path.join(frame_dir, name))
        if frame is None:
            continue
        features = manager.update(frame, dets)
        ids, motion_emb = manager.embed_motion(features)
        per_frame.append((fid, ids, motion_emb, manager.ready_ids))

    # Pass 2: every expression against the stored embeddings.
    scores = {}
    for expr in tqdm(expressions, desc=f"{seq} expressions"):
        lang_emb = manager.embed_lang(encoder.encode(expr.replace("-", " ")))
        per_expr = {}
        for fid, ids, motion_emb, ready in per_frame:
            cos = (motion_emb @ lang_emb.t()).flatten().cpu().numpy()
            frame_scores = {str(t): float(c) for t, c in zip(ids, cos) if t in ready}
            if frame_scores:
                per_expr[str(fid)] = frame_scores
        scores[expr] = per_expr

    os.makedirs(args.out, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scores, f)
    print(f"[{seq}] {len(expressions)} expressions -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-root", required=True, help="Refer-KITTI root (expression/ and KITTI/)")
    p.add_argument("--frames", default=None, help="frame root, default <data-root>/KITTI/training/image_02")
    p.add_argument("--tracks", required=True, help="NeuralSORT results: <seq>/{car,pedestrian}/predict.txt")
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seqs", default=",".join(TEST_SEQS))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = TextEncoder(DEFAULT_TEXT_ENCODER, device=device)
    for seq in args.seqs.split(","):
        build(seq, args, encoder, device)


if __name__ == "__main__":
    main()
