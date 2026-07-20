#!/usr/bin/env python3
"""Forensic diagnostic on canopy test scene 6799 (YOLO fg IoU collapse).

Steps (diagnostic only; locked checkpoints; no test metric committed, no retrain):
  1. Raw detection dump at conf>=0.05 (bbox, conf, class, mask area px, centroid)
     + per-detection colour-coded mask visualisation (not the union).
  2. Rasterised fg IoU vs conf for 6799 alone.
  3. 6799 vs 6766 distinguishing features (both canopy; wildly different YOLO IoU).
  4. last.pt vs best.pt on 6799.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy("file_system")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from ultralytics import YOLO
from scripts.perception.segmentation.yolo_binary.visualize import polygons_to_mask, yolo_lines_to_polygons

YOLO_DATA = REPO / "data/yolo_binary"
RUN = REPO / "results/runs/phase_b_yolo_binary"
OUT = RUN / "diagnostic_panels"
S6799 = "color_image_6799_png.rf.f15c54ed282871cb6b824e4e111ec031.jpg"
S6766 = "color_image_6766_png.rf.86a42af939959fa05e72e2b5ca7c0674.jpg"


def load(fn):
    p = (YOLO_DATA / "images" / "test" / fn).resolve()
    bgr = cv2.imread(str(p)); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    stem = Path(fn).stem
    gt_lines = (YOLO_DATA / "labels" / "test" / f"{stem}.txt").read_text().splitlines()
    gt = polygons_to_mask(yolo_lines_to_polygons(gt_lines, w, h), h, w)
    return rgb, gt, h, w, len(gt_lines)


def fg_iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    u = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / u) if u else float("nan")


def detections(model, fn, conf):
    p = (YOLO_DATA / "images" / "test" / fn).resolve()
    r = model.predict(source=str(p), conf=conf, half=True,
                      device=0 if torch.cuda.is_available() else "cpu",
                      verbose=False)[0]
    dets = []
    if r.masks is not None:
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        boxes = r.boxes.xyxy.cpu().numpy()
        h, w = r.orig_shape
        for i, poly in enumerate(r.masks.xy):
            m = polygons_to_mask([poly], h, w)
            ys, xs = np.nonzero(m)
            cen = (float(xs.mean()), float(ys.mean())) if len(xs) else (float("nan"),) * 2
            dets.append({"conf": float(confs[i]), "cls": int(clss[i]),
                         "box": [round(float(b), 1) for b in boxes[i]],
                         "area_px": int(m.sum()), "centroid_xy": [round(c, 1) for c in cen],
                         "poly": poly})
    return dets, r.orig_shape


def scene_features(rgb, gt, n_gt):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, w = rgb.shape[:2]; npx = h * w
    green = ((hsv[..., 0] > 25) & (hsv[..., 0] < 95) & (hsv[..., 1] > 40)).mean()
    sky = ((hsv[..., 0] > 95) & (hsv[..., 0] < 135) & (hsv[..., 2] > 120)).mean()
    return {"gt_fg_frac": round(float(gt.mean()), 4), "n_gt_instances": n_gt,
            "mean_V": round(float(hsv[..., 2].mean()), 1),
            "green_frac": round(float(green), 3), "sky_frac": round(float(sky), 3)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    best = YOLO(str(RUN / "weights" / "best.pt"))

    # ---- Step 1: raw detection dump (conf>=0.05) + per-detection viz ----
    rgb, gt, h, w, n_gt = load(S6799)
    dets, _ = detections(best, S6799, 0.05)
    dets.sort(key=lambda d: -d["area_px"])
    print(f"=== STEP 1: 6799 detections at conf>=0.05 (best.pt), {len(dets)} dets, "
          f"GT fg area={int(gt.sum())} px ===")
    print(f"{'#':>2}{'conf':>7}{'cls':>4}{'area_px':>9}{'centroid(x,y)':>16}   box(xyxy)")
    for i, d in enumerate(dets):
        print(f"{i:>2}{d['conf']:>7.3f}{d['cls']:>4}{d['area_px']:>9}"
              f"{str(tuple(d['centroid_xy'])):>16}   {d['box']}")

    viz = rgb.copy().astype(np.float32)
    rng = np.linspace(0, 179, max(1, len(dets)), dtype=np.uint8)
    for i, d in enumerate(dets):
        color = cv2.cvtColor(np.uint8([[[rng[i], 220, 255]]]), cv2.COLOR_HSV2RGB)[0, 0].astype(float)
        m = polygons_to_mask([d["poly"]], h, w).astype(bool)
        viz[m] = 0.5 * viz[m] + 0.5 * color
    cv2.imwrite(str(OUT / "6799_detections_individual.png"),
                cv2.cvtColor(viz.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    # ---- Step 2: conf response on 6799 ----
    print("\n=== STEP 2: 6799 rasterised fg IoU vs conf (best.pt) ===")
    base_dets, _ = detections(best, S6799, 0.05)  # all, then filter
    for t in [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75]:
        kept = [d["poly"] for d in base_dets if d["conf"] >= t]
        iou = fg_iou(polygons_to_mask(kept, h, w), gt)
        print(f"  conf>={t:.2f}: fg IoU={iou:.4f}  (n_kept={len(kept)})")

    # ---- Step 3: 6799 vs 6766 features ----
    print("\n=== STEP 3: 6799 vs 6766 scene features (both canopy) ===")
    rgb2, gt2, h2, w2, n_gt2 = load(S6766)
    f1 = scene_features(rgb, gt, n_gt); f2 = scene_features(rgb2, gt2, n_gt2)
    d6799_25 = [d["poly"] for d in base_dets if d["conf"] >= 0.25]
    d6766, _ = detections(best, S6766, 0.05)
    d6766_25 = [d["poly"] for d in d6766 if d["conf"] >= 0.25]
    print(f"  {'feature':<18}{'6799':>12}{'6766':>12}")
    for k in f1:
        print(f"  {k:<18}{str(f1[k]):>12}{str(f2[k]):>12}")
    print(f"  {'n_det@0.05':<18}{len(base_dets):>12}{len(d6766):>12}")
    print(f"  {'n_det@0.25':<18}{len(d6799_25):>12}{len(d6766_25):>12}")
    print(f"  {'max_mask_px@0.05':<18}{max((d['area_px'] for d in base_dets), default=0):>12}"
          f"{max((d['area_px'] for d in d6766), default=0):>12}")
    print(f"  {'fgIoU@0.25':<18}{fg_iou(polygons_to_mask(d6799_25,h,w),gt):>12.4f}"
          f"{fg_iou(polygons_to_mask(d6766_25,h2,w2),gt2):>12.4f}")
    cmp = np.hstack([rgb, np.full((h, 12, 3), 255, np.uint8), rgb2])
    cv2.imwrite(str(OUT / "6799_vs_6766_raw.png"), cv2.cvtColor(cmp, cv2.COLOR_RGB2BGR))

    # ---- Step 4: last.pt vs best.pt on 6799 ----
    print("\n=== STEP 4: last.pt vs best.pt on 6799 ===")
    last = YOLO(str(RUN / "weights" / "last.pt"))
    for name, mdl in [("best.pt", best), ("last.pt", last)]:
        dd, _ = detections(mdl, S6799, 0.05)
        for t in [0.25, 0.50]:
            kept = [d["poly"] for d in dd if d["conf"] >= t]
            print(f"  {name} conf>={t:.2f}: fg IoU={fg_iou(polygons_to_mask(kept,h,w),gt):.4f}"
                  f"  n_det={len(kept)}  max_mask_px={max((d['area_px'] for d in dd if d['conf']>=t), default=0)}")
    print(f"\n-> {OUT}/6799_detections_individual.png, 6799_vs_6766_raw.png")


if __name__ == "__main__":
    main()
