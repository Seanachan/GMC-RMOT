"""Fuse s_gmc into iKUN and report HOTA on Refer-KITTI V1 (official
150-expression test list): pooled, and per expression class.

    python scripts/eval_ikun.py --ikun-root ~/iKUN --data-root /data/refer-kitti \
        --tracks /data/NeuralSORT --ikun-scores ikun_results.json \
        --text-feat text_feat.json --gt-template /data/refer-kitti/gt_template_old \
        --trackeval ~/TrackEval --gmc-scores scores/seed0 --out runs/ikun_seed0

Host score = iKUN cascade score + similarity-calibration bias (iKUN's own
test-time procedure). Fusion: s_final = s_host + alpha_c * s_gmc, and a track
is kept iff s_final > 0. `--alpha-mot 0 --alpha-oth 0` reproduces iKUN.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gmc.expressions import CLASSES, classify  # noqa: E402
from gmc.fusion import PAPER_WEIGHTS, FusionWeights, fuse  # noqa: E402
from gmc.io import load_neuralsort  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OFFICIAL_SEQMAP = os.path.join(REPO, "seqmaps", "refer_kitti_v1_test_official_150.txt")
FRAME_RANGE = {"0005": (0, 296), "0011": (0, 372), "0013": (0, 339)}
# iKUN similarity calibration constants (iKUN test-time defaults).
SIM_A, SIM_B, SIM_TAU = 8.0, -0.1, 100.0


def similarity_bias(text_feat: dict, expressions: list[str], expression_conversion) -> dict[str, float]:
    """iKUN's similarity calibration: a per-expression bias from the
    similarity-weighted training-set positive rate."""
    train, test = text_feat["train"], text_feat["test"]
    keys = list(train)
    feats = np.array([train[k]["feature"] for k in keys])
    probs = np.array([train[k]["probability"] for k in keys])
    bias = {}
    for expr in expressions:
        name = expression_conversion(expr)
        table = test if name in test else train
        if name not in table:
            bias[expr] = 0.0
            continue
        sim = (np.array(table[name]["feature"])[None, :] @ feats.T)[0]
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-12)
        w = np.exp(SIM_TAU * sim)
        w = w / w.sum()
        bias[expr] = SIM_A * float((w * probs).sum()) + SIM_B
    return bias


def ikun_scores(all_scores: dict, seq: str, expr: str) -> dict[int, dict[int, float]]:
    """{frame: {track: mean logit}} from iKUN's video -> track -> frame -> expression file."""
    out: dict[int, dict[int, float]] = defaultdict(dict)
    for oid, frames in all_scores.get(seq, {}).items():
        for fid, by_expr in frames.items():
            if expr in by_expr:
                out[int(fid)][int(oid)] = float(np.mean(by_expr[expr]))
    return out


def write_predictions(args, weights: FusionWeights, entries: list[tuple[str, str]], res_dir: str) -> None:
    sys.path.insert(0, os.path.expanduser(args.ikun_root))
    from utils import expression_conversion  # iKUN's utils.py

    with open(args.text_feat) as f:
        text_feat = json.load(f)
    with open(args.ikun_scores) as f:
        all_ikun = json.load(f)

    by_seq = defaultdict(list)
    for seq, expr in entries:
        by_seq[seq].append(expr)
    for seq, exprs in by_seq.items():
        tracks = load_neuralsort(args.tracks, seq)
        bias = similarity_bias(text_feat, exprs, expression_conversion)
        gmc_seq = {}
        if args.gmc_scores:
            with open(os.path.join(args.gmc_scores, f"{seq}.json")) as f:
                gmc_seq = json.load(f)
        lo, hi = FRAME_RANGE[seq]
        for expr in exprs:
            out_dir = os.path.join(res_dir, seq, expr)
            os.makedirs(out_dir, exist_ok=True)
            gt = os.path.join(args.gt_template, seq, expr, "gt.txt")
            if os.path.exists(gt):
                shutil.copy2(gt, os.path.join(out_dir, "gt.txt"))
            else:
                open(os.path.join(out_dir, "gt.txt"), "w").close()

            host = ikun_scores(all_ikun, seq, expr)
            gmc = gmc_seq.get(expr, {})
            alpha = weights.for_expression(expr)
            b = bias.get(expr, 0.0)
            with open(os.path.join(out_dir, "predict.txt"), "w") as f:
                for fid, dets in tracks.items():
                    if not lo < fid < hi:
                        continue
                    for oid, x, y, w, h in dets:
                        cs = host.get(fid, {}).get(oid)
                        if cs is None:
                            continue
                        s_gmc = gmc.get(str(fid), {}).get(str(oid))
                        if fuse(cs + b, s_gmc, alpha) > 0.0:
                            f.write(f"{fid},{oid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")


def run_trackeval(trackeval_root: str, seqmap: str, res_dir: str) -> float | None:
    summary = os.path.join(res_dir, "pedestrian_summary.txt")
    if os.path.exists(summary):
        os.remove(summary)
    script = os.path.join(os.path.expanduser(trackeval_root), "scripts", "run_mot_challenge.py")
    cmd = [sys.executable, script,
           "--METRICS", "HOTA",
           "--SEQMAP_FILE", os.path.abspath(seqmap),
           "--SKIP_SPLIT_FOL", "True",
           "--GT_FOLDER", os.path.abspath(res_dir),
           "--TRACKERS_FOLDER", os.path.abspath(res_dir),
           "--GT_LOC_FORMAT", "{gt_folder}/{video_id}/{expression_id}/gt.txt",
           "--TRACKERS_TO_EVAL", os.path.abspath(res_dir),
           "--USE_PARALLEL", "False", "--PLOT_CURVES", "False", "--PRINT_CONFIG", "False"]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(script))
    if not os.path.exists(summary):
        sys.stderr.write(f"TrackEval failed (rc={proc.returncode}):\n{proc.stderr[-1500:]}\n")
        return None
    with open(summary) as f:
        return float(f.read().splitlines()[1].split()[0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ikun-root", required=True, help="iKUN checkout (for utils.expression_conversion)")
    p.add_argument("--data-root", required=True, help="Refer-KITTI root")
    p.add_argument("--tracks", required=True, help="NeuralSORT results directory")
    p.add_argument("--ikun-scores", required=True, help="results.json from iKUN's test.py (cascade attention, without calibration)")
    p.add_argument("--text-feat", required=True, help="iKUN's textual_features.json (for similarity calibration)")
    p.add_argument("--gt-template", required=True, help="Refer-KITTI gt_template in NeuralSORT's frame numbering")
    p.add_argument("--trackeval", required=True, help="TrackEval copy that reads seq+expression seqmaps (e.g. TempRMOT's)")
    p.add_argument("--gmc-scores", default=None, help="output of build_gmc_scores.py; omit for iKUN alone")
    p.add_argument("--alpha-mot", type=float, default=PAPER_WEIGHTS["ikun"].alpha_mot)
    p.add_argument("--alpha-oth", type=float, default=PAPER_WEIGHTS["ikun"].alpha_oth)
    p.add_argument("--seqmap", default=OFFICIAL_SEQMAP)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.seqmap) as f:
        entries = [tuple(line.split("+", 1)) for line in f.read().split() if line]
    weights = FusionWeights(args.alpha_mot, args.alpha_oth)

    res_dir = os.path.join(args.out, "results")
    shutil.rmtree(res_dir, ignore_errors=True)
    write_predictions(args, weights, entries, res_dir)

    result = {"alpha_mot": weights.alpha_mot, "alpha_oth": weights.alpha_oth,
              "gmc_scores": args.gmc_scores, "n_expressions": len(entries)}
    groups = {"pooled": entries, **{c.lower(): [e for e in entries if classify(e[1]) == c] for c in CLASSES}}
    for name, group in groups.items():
        seqmap = os.path.join(args.out, f"seqmap_{name}.txt")
        with open(seqmap, "w") as f:
            f.write("\n".join(f"{s}+{e}" for s, e in group) + "\n")
        result[name] = run_trackeval(args.trackeval, seqmap, res_dir) if group else None
        print(f"{name:>10s} HOTA {result[name]}  ({len(group)} expressions)")

    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
