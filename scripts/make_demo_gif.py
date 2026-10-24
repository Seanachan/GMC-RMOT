"""Demo GIF: image-plane motion vs. camera-compensated motion.

    python scripts/make_demo_gif.py --data-root /data/refer-kitti --seq 0001 \
        --expression parking-cars --start 195 --end 245 --out assets/demo.gif

Both panels show the same ground-truth boxes. Each arrow is the object's
displacement over the mid gap (5 frames). Top: raw image motion, where the
camera's own motion dominates. Bottom: the residual after removing camera
motion (Eq. 2 of the paper), which is what the GMC module matches against the
expression. Boxes the expression refers to are green, the others red.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gmc import GMCManager  # noqa: E402
from gmc.dataset import load_labels_with_ids  # noqa: E402
from gmc.geometry import VELOCITY_SCALE  # noqa: E402

GREEN, RED, WHITE, BLACK = (70, 200, 70), (60, 60, 235), (255, 255, 255), (0, 0, 0)


def draw_panel(frame, tracks, feats, ready, referred, title):
    img = frame.copy()
    h, w = img.shape[:2]
    for tid, x, y, bw, bh in tracks:
        color = GREEN if tid in referred else RED
        cv2.rectangle(img, (int(x), int(y)), (int(x + bw), int(y + bh)), color, 2)
        if tid not in ready:
            continue
        cx, cy = x + bw / 2, y + bh / 2
        dx = feats[tid][2] * w / VELOCITY_SCALE
        dy = feats[tid][3] * h / VELOCITY_SCALE
        cv2.arrowedLine(img, (int(cx), int(cy)), (int(cx + dx), int(cy + dy)), color, 3, tipLength=0.25)
    cv2.rectangle(img, (0, 0), (w, 36), BLACK, -1)
    cv2.putText(img, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
    return img


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data-root", required=True, help="Refer-KITTI root")
    p.add_argument("--seq", default="0001")
    p.add_argument("--expression", default="parking-cars", help="expression file name without .json")
    p.add_argument("--start", type=int, default=195)
    p.add_argument("--end", type=int, default=245)
    p.add_argument("--width", type=int, default=640, help="output width in pixels")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--out", default="assets/demo.gif")
    args = p.parse_args()

    frame_dir = os.path.join(args.data_root, "KITTI", "training", "image_02", args.seq)
    labels = load_labels_with_ids(os.path.join(args.data_root, "KITTI", "labels_with_ids", "image_02", args.seq))
    with open(os.path.join(args.data_root, "expression", args.seq, args.expression + ".json")) as f:
        expr = json.load(f)

    raw, comp = GMCManager(compensate_ego=False), GMCManager()
    images = []
    for fid in range(args.end):
        frame = cv2.imread(os.path.join(frame_dir, f"{fid:06d}.png"))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        tracks = [(r["track_id"], r["x1_n"] * w, r["y1_n"] * h, r["w_n"] * w, r["h_n"] * h)
                  for r in labels.get(fid, [])]
        raw_feats, comp_feats = raw.update(frame, tracks), comp.update(frame, tracks)
        if fid < args.start:
            continue  # warm up the motion history before recording
        referred = set(expr["label"].get(str(fid), []))
        ready = comp.ready_ids
        top = draw_panel(frame, tracks, raw_feats, ready, referred, "Image-plane motion (camera + object)")
        bottom = draw_panel(frame, tracks, comp_feats, ready, referred, "After global motion compensation")
        caption = np.zeros((44, w, 3), dtype=np.uint8)
        cv2.putText(caption, f'"{expr["sentence"]}"   green = referred   red = not referred   arrows = 5-frame motion',
                    (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
        stacked = np.vstack([top, bottom, caption])
        size = (args.width, int(stacked.shape[0] * args.width / stacked.shape[1]))
        stacked = cv2.resize(stacked, size, interpolation=cv2.INTER_AREA)
        images.append(Image.fromarray(cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB)))

    palette = images[len(images) // 2].quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps),
                   loop=0, optimize=True)
    print(f"{len(frames)} frames -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
