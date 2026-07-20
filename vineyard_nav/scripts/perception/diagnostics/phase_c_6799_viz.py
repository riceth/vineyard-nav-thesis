#!/usr/bin/env python3
"""Phase C 6799 check + visualisation (F007 informant). Same format as Phase B.

Predicts test scene 6799 with the LOCKED Phase C best.pt at conf 0.25 and reports
whether the F007 canopy-blob failure recurs. Renders three panels (raw, GT vs
YOLO union, per-detection class-coloured) into
results/runs/phase_c_yolo_multiclass/diagnostic/6799_visualisation/.
Diagnostic on the already-locked test read; no new metric committed.
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
torch.multiprocessing.set_sharing_strategy("file_system")
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
from ultralytics import YOLO
from scripts.perception.segmentation.yolo_binary.visualize import polygons_to_mask, yolo_lines_to_polygons

YD = REPO / "data/yolo_multiclass"; RUN = REPO / "results/runs/phase_c_yolo_multiclass"
OUT = RUN / "diagnostic/6799_visualisation"
FN = "color_image_6799_png.rf.f15c54ed282871cb6b824e4e111ec031.jpg"
CONF = 0.25
NAMES = {0: "trunk", 1: "pole"}


def main():
    global RUN, OUT
    import argparse
    ap = argparse.ArgumentParser(description="Phase C 6799 viz for a run (default seed 42).")
    ap.add_argument("--run-dir", default=str(RUN), help="results/runs/<phase_c run dir>")
    RUN = Path(ap.parse_args().run_dir)
    OUT = RUN / "diagnostic/6799_visualisation"
    OUT.mkdir(parents=True, exist_ok=True)
    p = (YD / "images/test" / FN).resolve()
    rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB); h, w = rgb.shape[:2]
    stem = Path(FN).stem
    gt = polygons_to_mask(yolo_lines_to_polygons(
        (YD / "labels/test" / f"{stem}.txt").read_text().splitlines(), w, h), h, w).astype(bool)

    r = YOLO(str(RUN / "weights/best.pt")).predict(source=str(p), conf=CONF, half=True,
                                                   device=0 if torch.cuda.is_available() else "cpu",
                                                   verbose=False)[0]
    confs = r.boxes.conf.cpu().numpy(); clss = r.boxes.cls.cpu().numpy().astype(int)
    dets = []
    for i, poly in enumerate(r.masks.xy):
        m = polygons_to_mask([poly], h, w).astype(bool)
        dets.append({"conf": float(confs[i]), "cls": int(clss[i]), "area": int(m.sum()), "mask": m})
    dets.sort(key=lambda d: -d["area"])
    union = np.zeros((h, w), bool)
    for d in dets:
        union |= d["mask"]
    max_area = max((d["area"] for d in dets), default=0)
    blob = max_area > 10000
    inter = np.logical_and(union, gt).sum(); uni = np.logical_or(union, gt).sum()
    fg_iou = inter / uni if uni else float("nan")

    print(f"Phase C 6799: {len(dets)} detections at conf>={CONF} "
          f"({sum(d['cls']==0 for d in dets)} trunk, {sum(d['cls']==1 for d in dets)} pole)")
    print(f"  GT union: {int(gt.sum())} px | pred union: {int(union.sum())} px | fg IoU: {fg_iou:.4f}")
    print(f"  largest mask: {max_area} px | BLOB(>10000px): {blob}")

    # 1 raw
    fig = plt.figure(figsize=(9, 9)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(rgb); fig.savefig(OUT / "1_raw.png", dpi=170); plt.close(fig)
    # 2 GT vs union
    fig, ax = plt.subplots(figsize=(9, 9)); ax.imshow(rgb); ax.axis("off")
    g = np.zeros((h, w, 4)); g[gt] = [0, .8, 0, .45]; ax.imshow(g)
    rr = np.zeros((h, w, 4)); rr[union] = [1, 0, 0, .40]; ax.imshow(rr)
    ax.set_title(f"Phase C 6799 — GT (green, {int(gt.sum())} px) vs YOLO union @conf{CONF} "
                 f"(red, {int(union.sum())} px)  fg IoU={fg_iou:.3f}", fontsize=10)
    ax.legend(handles=[Patch(color=(0, .8, 0), label="ground truth"),
                       Patch(color=(1, 0, 0), label="YOLO union (trunk+pole)")], loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "2_gt_vs_yolo_union.png", dpi=170); plt.close(fig)
    # 3 per-detection, class-coloured
    fig, ax = plt.subplots(figsize=(13, 9)); ax.imshow(rgb); ax.axis("off")
    reds = plt.get_cmap("autumn"); blues = plt.get_cmap("winter"); handles = []
    nt = np_ = 0
    for idx, d in enumerate(dets):
        if d["cls"] == 0:
            color = reds(0.15 + 0.6 * (nt % 4) / 4)[:3]; nt += 1
        else:
            color = blues(0.15 + 0.6 * (np_ % 4) / 4)[:3]; np_ += 1
        ov = np.zeros((h, w, 4)); ov[d["mask"]] = [*color, 0.6]; ax.imshow(ov)
        ys, xs = np.nonzero(d["mask"])
        ax.text(xs.mean(), ys.mean(), str(idx), color="white", fontsize=8, ha="center", va="center",
                fontweight="bold", bbox=dict(boxstyle="circle", fc="black", ec=color, alpha=.7))
        handles.append(Patch(color=color, label=f"#{idx} {NAMES[d['cls']]} c{d['conf']:.2f} {d['area']:,}px"
                             + ("  <-- BLOB" if d["area"] > 10000 else "")))
    ax.set_title(f"Phase C 6799 — {len(dets)} detections (trunk=red-family, pole=blue-family); "
                 f"largest {max_area:,} px {'= BLOB' if blob else '(no blob)'}", fontsize=10)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7,
              title="detections (by area)")
    fig.tight_layout(); fig.savefig(OUT / "3_per_detection.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    for f in ("1_raw.png", "2_gt_vs_yolo_union.png", "3_per_detection.png"):
        print(f"  -> {OUT / f}")


if __name__ == "__main__":
    main()
