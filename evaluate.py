#!/usr/bin/env python3
"""
Simple evaluator for a single prediction vs ground-truth solution.

Usage: python data_raw/evaluate.py --gt path/to/gt_solution.json --pred path/to/pred_solution.json 

Prints path lengths and off-road violations (point and segment-level) when a map mask is available.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def load_solution(p: Path):
    data = json.loads(p.read_text())
    if isinstance(data, list):
        return {"path": data}
    return data


def path_length(path: List[Tuple[int, int]]) -> float:
    return sum(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(path, path[1:]))


def rasterize_segment(a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
    # simple integer interpolation along the longest axis
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    steps = max(abs(dx), abs(dy), 1)
    pts = []
    for i in range(steps + 1):
        t = i / float(steps)
        x = int(round(a[0] + dx * t))
        y = int(round(a[1] + dy * t))
        pts.append((x, y))
    return pts


def load_mask(p: Path):
    im = Image.open(p).convert("L")
    arr = im.load()
    w, h = im.size
    def check(x, y):
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        return arr[x, y] > 127
    return check, (w, h)


def count_violations(path: List[Tuple[int, int]], mask_check) -> Tuple[int, int]:
    # returns (offroad_point_count, offroad_segment_count)
    off_pts = 0
    off_segs = 0
    for (x, y) in path:
        if not mask_check(x, y):
            off_pts += 1
    # segments
    for a, b in zip(path, path[1:]):
        pts = rasterize_segment(a, b)
        if any(not mask_check(x, y) for x, y in pts):
            off_segs += 1
    return off_pts, off_segs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt", required=True)
    p.add_argument("--pred", required=True)
    p.add_argument("--map", required=False)
    args = p.parse_args()

    gt = load_solution(Path(args.gt))
    pred = load_solution(Path(args.pred))

    gt_path = gt.get("path") or gt.get("points")
    pred_path = pred.get("path") or pred.get("points")
    if not gt_path or not pred_path:
        print("Error: both gt and pred must contain a 'path' field (list of [x,y])")
        raise SystemExit(2)

    gt_len = path_length(gt_path)
    pred_len = path_length(pred_path)

    print(f"GT length: {gt_len:.2f}")
    print(f"Pred length: {pred_len:.2f}")
    print(f"Length diff: {pred_len - gt_len:.2f}")

    if args.map:
        mask_check, (w, h) = load_mask(Path(args.map))
        gt_off_pts, gt_off_segs = count_violations(gt_path, mask_check)
        pred_off_pts, pred_off_segs = count_violations(pred_path, mask_check)
        print(f"GT off-road points: {gt_off_pts}, segments: {gt_off_segs}")
        print(f"Pred off-road points: {pred_off_pts}, segments: {pred_off_segs}")
        total_violations = pred_off_pts + pred_off_segs
        print(f"Pred total violations: {total_violations}")
    else:
        print("Map not provided; skipping off-road checks. Provide --map to enable.")


if __name__ == "__main__":
    main()
