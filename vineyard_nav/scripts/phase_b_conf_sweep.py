#!/usr/bin/env python3
"""Phase B confidence-threshold sweep on validation (D030).

Selects the operating-point confidence conf* for Phase B's rasterised per-frame
foreground IoU (the cross-arm perception metric, F005) by argmax mean fg IoU over
the 46 validation scenes. Aligns Phase B's operating-point selection with Phase
C's val-based T-sweep. VAL ONLY — no test access (rule 5).

Efficiency: one predict pass at the lowest grid conf (0.10) captures every
detection with its confidence; higher thresholds are obtained by filtering. This
is exact — NMS never lets a lower-confidence box suppress a higher-confidence one,
so {detections with conf>=t that survive NMS over conf>=0.10} == {... over conf>=t}.

Precision matches training/eval (half=True, D029).
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

REPO = Path("/workspaces/dissertation/vineyard_nav")
sys.path.insert(0, str(REPO))
from ultralytics import YOLO
from segmentation.yolo_binary.visualize import polygons_to_mask, yolo_lines_to_polygons

YOLO_DATA = REPO / "data/yolo_binary"
RUN_DIR = REPO / "results/runs/phase_b_yolo_binary"
CONF_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
SWEEP_MIN = min(CONF_GRID)


def fg_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool); gt = gt.astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return (inter / union) if union else float("nan")   # union==0 impossible (all val frames have GT fg)


def main() -> None:
    model = YOLO(str(RUN_DIR / "weights" / "best.pt"))
    device = 0 if torch.cuda.is_available() else "cpu"
    images_dir = YOLO_DATA / "images" / "val"
    labels_dir = YOLO_DATA / "labels" / "val"

    # Per-image: (list of (conf, polygon)), GT mask. One predict pass at SWEEP_MIN.
    per_image = []
    for r in model.predict(source=str(images_dir), conf=SWEEP_MIN, half=True,
                           device=device, verbose=False, stream=True):
        h, w = r.orig_shape
        confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.array([])
        polys = list(r.masks.xy) if r.masks is not None else []
        stem = Path(r.path).stem
        gt_lines = (labels_dir / f"{stem}.txt").read_text().splitlines() \
            if (labels_dir / f"{stem}.txt").exists() else []
        gt_mask = polygons_to_mask(yolo_lines_to_polygons(gt_lines, w, h), h, w)
        per_image.append((h, w, list(zip(confs, polys)), gt_mask))

    n = len(per_image)
    curve = {}
    for t in CONF_GRID:
        ious = []
        for h, w, dets, gt in per_image:
            kept = [p for c, p in dets if c >= t]
            pred_mask = polygons_to_mask(kept, h, w)
            ious.append(fg_iou(pred_mask, gt))
        curve[t] = float(np.nanmean(ious))

    conf_star = max(curve, key=curve.get)

    # --- persist: json + sensitivity curve ---
    out = {"grid": CONF_GRID, "n_val_scenes": n,
           "mean_val_fg_iou": curve, "conf_star": conf_star,
           "selection": "argmax mean per-frame rasterised fg IoU over val (D030)"}
    (RUN_DIR / "val_conf_sweep.json").write_text(json.dumps(out, indent=2))

    xs = CONF_GRID
    ys = [curve[t] for t in xs]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(xs, ys, "-o", color="#3a6b35")
    ax.plot(conf_star, curve[conf_star], "*", ms=16, color="#b8791f",
            label=f"conf* = {conf_star} ({curve[conf_star]:.4f})")
    ax.set_xlabel("confidence threshold"); ax.set_ylabel("mean val fg IoU (46 scenes)")
    ax.set_title("Phase B — val rasterised fg IoU vs conf")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(RUN_DIR / "val_conf_sweep.png", dpi=120)

    print(f"Phase B conf sweep on val (n={n} scenes), half=True")
    print(f"  {'conf':>6}  {'mean val fg IoU':>16}")
    for t in CONF_GRID:
        mark = "  <- conf*" if t == conf_star else ""
        print(f"  {t:>6.2f}  {curve[t]:>16.4f}{mark}")
    spread = max(ys) - min(ys)
    print(f"  spread (max-min) = {spread:.4f}  "
          f"({'FLAT (insensitive)' if spread < 0.02 else 'peaked (sensitive)'})")
    print(f"  conf* = {conf_star}")
    print(f"  -> {RUN_DIR / 'val_conf_sweep.json'}")
    print(f"  -> {RUN_DIR / 'val_conf_sweep.png'}")


if __name__ == "__main__":
    main()
