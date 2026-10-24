"""Readers for host tracker output."""

import os
from collections import defaultdict

import numpy as np


def load_mot_tracks(path: str) -> dict[int, list[tuple]]:
    """MOT-format predict.txt (frame,id,x,y,w,h,...) -> {frame: [(id, x, y, w, h)]}"""
    out = defaultdict(list)
    if os.path.getsize(path) == 0:  # e.g. a sequence with no pedestrians
        return out
    for row in np.loadtxt(path, delimiter=",", ndmin=2):
        out[int(row[0])].append((int(row[1]), row[2], row[3], row[4], row[5]))
    return out


def load_neuralsort(track_dir: str, seq: str) -> dict[int, list[tuple]]:
    """Car and pedestrian NeuralSORT tracks of one sequence, as used by iKUN.
    Pedestrian ids are offset past the largest car id so they never collide."""
    car = load_mot_tracks(os.path.join(track_dir, seq, "car", "predict.txt"))
    ped = load_mot_tracks(os.path.join(track_dir, seq, "pedestrian", "predict.txt"))
    max_car = max((oid for dets in car.values() for oid, *_ in dets), default=0)
    merged = defaultdict(list)
    for fid, dets in car.items():
        merged[fid].extend(dets)
    for fid, dets in ped.items():
        merged[fid].extend((oid + max_car, x, y, w, h) for oid, x, y, w, h in dets)
    return merged
