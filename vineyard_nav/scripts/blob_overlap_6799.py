#!/usr/bin/env python3
"""Cross-run 6799 blob-overlap analysis (F007 / O009 informant).

Regenerates the cross-arm / cross-seed blob-geometry comparison on test scene
color_image_6799: for each supplied run, predicts 6799 with that run's LOCKED
weights/best.pt at conf 0.25 (half=True, D029), takes the largest-area instance
mask (the blob when present), then computes pairwise mask IoU, centroid distance,
bounding boxes, and mutual coverage across all runs. Writes overlay PNGs and a
JSON summary to results/runs/phase_c_blob_overlap_6799/.

Diagnostic on the already-locked test read; no new metric committed, no retrain,
no test re-evaluation (rule 5 preserved — each best.pt is only *read*).

Reproduces the O009 4-way result:
  python scripts/blob_overlap_6799.py \
    --runs phase_b_yolo_binary phase_b_yolo_binary_seed43 \
           phase_c_yolo_multiclass_seed43 phase_c_yolo_multiclass_seed44

Default runs (no --runs) = the four blobbing runs of the O009 pass.
"""
from __future__ import annotations
import sys, json, itertools
from pathlib import Path
import numpy as np, cv2, torch
torch.multiprocessing.set_sharing_strategy("file_system")
REPO = Path("/workspaces/dissertation/vineyard_nav"); sys.path.insert(0, str(REPO))
from ultralytics import YOLO
from segmentation.yolo_binary.visualize import polygons_to_mask

FN = "color_image_6799_png.rf.f15c54ed282871cb6b824e4e111ec031.jpg"
IMG = REPO / "data/yolo_binary/images/test" / FN          # same 6799 frame across data dirs
OUT = REPO / "results/runs/phase_c_blob_overlap_6799"
CONF = 0.25
DEFAULT_RUNS = ["phase_b_yolo_binary", "phase_b_yolo_binary_seed43",
                "phase_c_yolo_multiclass_seed43", "phase_c_yolo_multiclass_seed44"]


def largest_mask(run: str, rgb, h, w):
    """Largest-area instance mask from run/weights/best.pt on 6799 at conf 0.25."""
    r = YOLO(str(REPO / "results/runs" / run / "weights/best.pt")).predict(
        source=str(IMG.resolve()), conf=CONF, half=True,
        device=0 if torch.cuda.is_available() else "cpu", verbose=False)[0]
    masks = [polygons_to_mask([poly], h, w).astype(bool) for poly in r.masks.xy]
    m = max(masks, key=lambda a: int(a.sum()))
    ys, xs = np.nonzero(m)
    return {"mask": m, "area": int(m.sum()),
            "centroid": (float(xs.mean()), float(ys.mean())),
            "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Cross-run 6799 blob-overlap analysis.")
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                    help="run dir names under results/runs/ (each with weights/best.pt)")
    runs = ap.parse_args().runs
    OUT.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(cv2.imread(str(IMG.resolve())), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    B = {run: largest_mask(run, rgb, h, w) for run in runs}
    print("largest-mask area per run:", {k: B[k]["area"] for k in B})

    pairs = []
    for a, b in itertools.combinations(runs, 2):
        ma, mb = B[a]["mask"], B[b]["mask"]
        inter = int(np.logical_and(ma, mb).sum()); uni = int(np.logical_or(ma, mb).sum())
        iou = inter / uni if uni else float("nan")
        (ax, ay), (bx, by) = B[a]["centroid"], B[b]["centroid"]
        dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
        pairs.append({"a": a, "b": b, "iou": iou, "centroid_dist_px": dist,
                      "cov_a_in_b": inter / B[a]["area"], "cov_b_in_a": inter / B[b]["area"]})
        print(f"{a} vs {b}: IoU {iou:.4f} | centroid {dist:.1f}px | "
              f"cov {100*inter/B[a]['area']:.0f}%/{100*inter/B[b]['area']:.0f}%")
        ov = rgb.astype(np.float32).copy()
        ov[ma & ~mb] = 0.5 * ov[ma & ~mb] + 0.5 * np.array([255, 0, 0])
        ov[mb & ~ma] = 0.5 * ov[mb & ~ma] + 0.5 * np.array([0, 80, 255])
        ov[ma & mb] = 0.4 * ov[ma & mb] + 0.6 * np.array([255, 255, 0])
        cv2.imwrite(str(OUT / f"overlap_{a}_{b}.png"),
                    cv2.cvtColor(ov.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

    ious = [p["iou"] for p in pairs]
    summary = {"conf": CONF, "runs": runs,
               "areas": {k: B[k]["area"] for k in B},
               "bboxes": {k: B[k]["bbox"] for k in B},
               "pairs": pairs,
               "iou_min": min(ious), "iou_max": max(ious), "iou_mean": sum(ious) / len(ious)}
    (OUT / "overlap_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"pairwise IoU: min {min(ious):.4f} max {max(ious):.4f} mean {sum(ious)/len(ious):.4f}")
    print(f"  -> {OUT/'overlap_summary.json'}")


if __name__ == "__main__":
    main()
