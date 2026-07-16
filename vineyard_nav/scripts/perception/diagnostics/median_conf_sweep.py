#!/usr/bin/env python3
"""Supplementary median-based conf sweep on validation (feeds F007 discussion).

For each conf in the D030 grid, compute BOTH mean and median per-frame rasterised
fg IoU over the 46 val frames, plus the catastrophic-frame count (fg IoU < 0.1).
Reports mean-based conf* (D030 primary) and median-based conf*; if they differ,
tabulates mean/median/catastrophic-count at both. VAL ONLY; no test access; no
supersede of D030 (mean-based conf* = 0.25 stays primary).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy("file_system")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from ultralytics import YOLO
from segmentation.yolo_binary.visualize import polygons_to_mask, yolo_lines_to_polygons

YD = REPO / "data/yolo_binary"
RUN = REPO / "results/runs/phase_b_yolo_binary"
GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
CATASTROPHIC = 0.1


def fg_iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    u = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / u) if u else float("nan")


def main():
    model = YOLO(str(RUN / "weights/best.pt"))
    device = 0 if torch.cuda.is_available() else "cpu"
    idir, ldir = YD / "images/val", YD / "labels/val"

    per_image = []
    for r in model.predict(source=str(idir), conf=min(GRID), half=True,
                           device=device, verbose=False, stream=True):
        h, w = r.orig_shape
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.array([])
        polys = list(r.masks.xy) if r.masks is not None else []
        gt_lines = (ldir / f"{Path(r.path).stem}.txt").read_text().splitlines() \
            if (ldir / f"{Path(r.path).stem}.txt").exists() else []
        gt = polygons_to_mask(yolo_lines_to_polygons(gt_lines, w, h), h, w)
        per_image.append((h, w, list(zip(confs, polys)), gt))

    n = len(per_image)
    rows = {}
    for t in GRID:
        ious = []
        for h, w, dets, gt in per_image:
            kept = [p for c, p in dets if c >= t]
            ious.append(fg_iou(polygons_to_mask(kept, h, w), gt))
        a = np.array(ious, float)
        rows[t] = {"mean": float(np.nanmean(a)), "median": float(np.nanmedian(a)),
                   "catastrophic_lt_0.1": int((a < CATASTROPHIC).sum())}

    mean_star = max(GRID, key=lambda t: rows[t]["mean"])
    median_star = max(GRID, key=lambda t: rows[t]["median"])

    out = {"grid": GRID, "n_val": n, "catastrophic_threshold": CATASTROPHIC,
           "per_conf": rows, "mean_conf_star": mean_star, "median_conf_star": median_star,
           "differ": mean_star != median_star}
    if mean_star != median_star:
        out["comparison_at_conf_stars"] = {
            str(mean_star): rows[mean_star], str(median_star): rows[median_star]}
    (RUN / "val_conf_sweep_median.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(GRID, [rows[t]["mean"] for t in GRID], "-o", color="#3a6b35", label="mean fg IoU")
    ax.plot(GRID, [rows[t]["median"] for t in GRID], "-s", color="#9c6f18", label="median fg IoU")
    ax.axvline(mean_star, ls="--", color="#3a6b35", alpha=.5)
    ax.axvline(median_star, ls=":", color="#9c6f18", alpha=.7)
    ax.set_xlabel("confidence threshold"); ax.set_ylabel("val fg IoU (46 scenes)")
    ax.set_title(f"Phase B val fg IoU — mean vs median (mean* {mean_star}, median* {median_star})")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(RUN / "val_conf_sweep_median.png", dpi=130)

    print(f"Median-based conf sweep on val (n={n}), half=True")
    print(f"  {'conf':>6}{'mean':>10}{'median':>10}{'catastrophic(<0.1)':>20}")
    for t in GRID:
        print(f"  {t:>6.2f}{rows[t]['mean']:>10.4f}{rows[t]['median']:>10.4f}"
              f"{rows[t]['catastrophic_lt_0.1']:>20}")
    print(f"  mean-based   conf* = {mean_star}  (D030 primary)")
    print(f"  median-based conf* = {median_star}  {'(DIFFERS)' if out['differ'] else '(same)'}")
    if out["differ"]:
        for t in (mean_star, median_star):
            r = rows[t]
            print(f"    @conf {t}: mean {r['mean']:.4f}  median {r['median']:.4f}  "
                  f"catastrophic {r['catastrophic_lt_0.1']}")
    print(f"  -> {RUN / 'val_conf_sweep_median.json'}")


if __name__ == "__main__":
    main()
