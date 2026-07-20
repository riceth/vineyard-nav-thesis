#!/usr/bin/env python3
"""High-res visualisations of the 6799 YOLO failure (diagnostic only).

Produces three figures in results/runs/phase_b_yolo_binary/diagnostic/6799_visualisation/:
  1_raw.png                 raw image, no annotations
  2_gt_vs_yolo_union.png    GT foreground (green) vs YOLO union @conf0.25 (red)
  3_per_detection.png       each detection colour-coded + labelled conf/area; blob in magenta

Labels use ACTUAL measured values (best.pt, conf>=0.25, deterministic across FP16/FP32:
13 detections; blob conf 0.406, 76,837 px). No retrain, no test metric, no commit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch

torch.multiprocessing.set_sharing_strategy("file_system")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from ultralytics import YOLO
from scripts.perception.segmentation.yolo_binary.visualize import polygons_to_mask, yolo_lines_to_polygons

YD = REPO / "data/yolo_binary"
RUN = REPO / "results/runs/phase_b_yolo_binary"
OUT = RUN / "diagnostic/6799_visualisation"
FN = "color_image_6799_png.rf.f15c54ed282871cb6b824e4e111ec031.jpg"
CONF = 0.25


def main():
    global RUN, OUT
    import argparse
    ap = argparse.ArgumentParser(description="6799 viz for a Phase B run (default seed 42).")
    ap.add_argument("--run-dir", default=str(RUN), help="results/runs/<phase_b run dir>")
    RUN = Path(ap.parse_args().run_dir)
    OUT = RUN / "diagnostic/6799_visualisation"
    OUT.mkdir(parents=True, exist_ok=True)
    img_path = (YD / "images/test" / FN).resolve()
    rgb = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    stem = Path(FN).stem
    gt = polygons_to_mask(
        yolo_lines_to_polygons((YD / "labels/test" / f"{stem}.txt").read_text().splitlines(), w, h),
        h, w).astype(bool)

    r = YOLO(str(RUN / "weights/best.pt")).predict(
        source=str(img_path), conf=CONF, half=True,
        device=0 if torch.cuda.is_available() else "cpu", verbose=False)[0]
    confs = r.boxes.conf.cpu().numpy()
    dets = []
    for i, poly in enumerate(r.masks.xy):
        m = polygons_to_mask([poly], h, w).astype(bool)
        dets.append({"conf": float(confs[i]), "area": int(m.sum()), "mask": m})
    dets.sort(key=lambda d: -d["area"])
    blob_i = 0  # largest by area
    union = np.zeros((h, w), bool)
    for d in dets:
        union |= d["mask"]

    # ---- 1: raw ----
    fig = plt.figure(figsize=(9, 9)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(rgb); fig.savefig(OUT / "1_raw.png", dpi=170); plt.close(fig)

    # ---- 2: GT (green) vs YOLO union (red) ----
    fig, ax = plt.subplots(figsize=(9, 9)); ax.imshow(rgb); ax.axis("off")
    g = np.zeros((h, w, 4)); g[gt] = [0, 0.8, 0, 0.45]; ax.imshow(g)
    rr = np.zeros((h, w, 4)); rr[union] = [1, 0, 0, 0.40]; ax.imshow(rr)
    ax.set_title(f"6799 — GT foreground (green, {int(gt.sum())} px) vs "
                 f"YOLO union @conf{CONF} (red, {int(union.sum())} px)", fontsize=11)
    ax.legend(handles=[Patch(color=(0, 0.8, 0), label="ground truth (trunk+pole)"),
                       Patch(color=(1, 0, 0), label="YOLO rasterised union")],
              loc="lower left", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "2_gt_vs_yolo_union.png", dpi=170); plt.close(fig)

    # ---- 3: per-detection, colour-coded + labelled ----
    fig, ax = plt.subplots(figsize=(13, 9)); ax.imshow(rgb); ax.axis("off")
    cmap = plt.get_cmap("tab20")
    handles = []
    for idx, d in enumerate(dets):
        is_blob = (idx == blob_i)
        color = (1.0, 0.0, 1.0) if is_blob else cmap(idx % 20)[:3]
        ov = np.zeros((h, w, 4)); ov[d["mask"]] = [*color, 0.55 if is_blob else 0.65]
        ax.imshow(ov)
        ys, xs = np.nonzero(d["mask"]); cy, cx = ys.mean(), xs.mean()
        ax.text(cx, cy, str(idx), color="white", fontsize=9, ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="circle", fc="black", ec=color, alpha=0.7, lw=1.5))
        handles.append(Patch(color=color,
                       label=f"#{idx}  conf {d['conf']:.3f}  {d['area']:,} px"
                             + ("   <-- BLOB (false positive)" if is_blob else "")))
    # arrow to blob
    bys, bxs = np.nonzero(dets[blob_i]["mask"])
    ax.annotate("BLOB\nconf 0.406\n76,837 px",
                xy=(bxs.mean(), bys.mean()), xytext=(0.62, 0.93), textcoords="axes fraction",
                color="magenta", fontsize=12, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color="magenta", lw=2.5))
    ax.set_title(f"6799 — YOLO best.pt detections at conf>={CONF} "
                 f"({len(dets)} detections; blob = 21x total GT area)", fontsize=11)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=8, title="detections (by area)")
    fig.tight_layout(); fig.savefig(OUT / "3_per_detection.png", dpi=170,
                                    bbox_inches="tight"); plt.close(fig)

    print(f"n_detections@conf{CONF}: {len(dets)}")
    print(f"blob: conf={dets[blob_i]['conf']:.3f}  area={dets[blob_i]['area']:,} px  "
          f"(GT total={int(gt.sum()):,} px, union={int(union.sum()):,} px)")
    for p in ("1_raw.png", "2_gt_vs_yolo_union.png", "3_per_detection.png"):
        print(f"  -> {OUT / p}")


if __name__ == "__main__":
    main()
