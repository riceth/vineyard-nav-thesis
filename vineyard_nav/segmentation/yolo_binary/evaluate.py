#!/usr/bin/env python3
"""Phase B evaluation (PHASE_B_SPEC section 7).

Loads a locked best.pt, runs ultralytics `model.val()` on a split, computes
overall + canopy-stratified metrics, optionally saves prediction visualisations,
and writes <split>_metrics.json.

Canopy stratification (§7.1): ultralytics has no native stratified eval, so we
run val three times — once on the full split, once on the bare-vine subset, once
on the canopy subset — using temporary data.yaml files whose split path points at
a txt list of the stratum's images (canopy from data/yolo_binary/canopy_state_map.json).

TEST-SET GUARDRAIL (rule 5 / §7.2): `--split` defaults to `test`; the test split is
evaluated ONCE after best.pt is locked. The reproduce-on-val step uses `--split val`.
Output filename is split-named, so a val reproduction never occupies test_metrics.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch

# 64 MB /dev/shm cannot back worker IPC in this container.
torch.multiprocessing.set_sharing_strategy("file_system")

from ultralytics import YOLO

from .visualize import polygons_to_mask, yolo_lines_to_polygons, save_gt_pred_panel

REPO = Path("/workspaces/dissertation/vineyard_nav")
YOLO_DATA = REPO / "data/yolo_binary"
CANOPY_STATES = ("bare_vine", "canopy")
PREDICT_CONF = 0.25          # operating-point confidence for per-frame pixel metrics/overlays
_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-fA-F]+\.[A-Za-z0-9]+$")
PER_FRAME_COLUMNS = ["filename", "scene_id", "canopy_state", "iou_foreground",
                     "iou_background", "precision_foreground", "recall_foreground",
                     "f1_foreground"]


def _pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Per-frame binary foreground pixel metrics (parallels Phase A per-frame)."""
    pred = pred.astype(bool); gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    div = lambda a, b: (a / b) if b else float("nan")
    return {
        "iou_foreground": div(tp, tp + fp + fn),
        "iou_background": div(tn, tn + fp + fn),
        "precision_foreground": div(tp, tp + fp),
        "recall_foreground": div(tp, tp + fn),
        "f1_foreground": div(2 * tp, 2 * tp + fp + fn),
    }


def metric_block(res) -> Dict[str, float]:
    """Pull mask (segmentation) + box metrics from an ultralytics val result.
    Mask metrics are primary for this segmentation arm; box included for context."""
    return {
        "map50": float(res.seg.map50),
        "map50_95": float(res.seg.map),
        "precision": float(res.seg.mp),
        "recall": float(res.seg.mr),
        "box_map50": float(res.box.map50),
        "box_map50_95": float(res.box.map),
    }


def write_split_yaml(tmp: Path, split_key: str, target: str, tag: str) -> Path:
    """data.yaml with all splits present (ultralytics requires train+val); only
    `split_key` is pointed at `target` (dir or txt list), others at defaults."""
    defaults = {"train": "images/train", "val": "images/val", "test": "images/test"}
    defaults[split_key] = target
    y = tmp / f"data_{split_key}_{tag}.yaml"
    y.write_text(
        f"path: {YOLO_DATA.resolve()}\n"
        f"train: {defaults['train']}\n"
        f"val: {defaults['val']}\n"
        f"test: {defaults['test']}\n"
        "nc: 1\nnames:\n  0: crop\n"
    )
    return y


def image_list(images_dir: Path, filenames: List[str], out: Path) -> Path:
    # Use the path UNDER data/yolo_binary/images/... (do NOT resolve the symlink):
    # ultralytics derives the label path by replacing 'images'->'labels' in the
    # image path, so it must stay inside the yolo_binary tree, not the symlink
    # target in SemanticBLT (which has no labels/ sibling).
    out.write_text("\n".join(str(images_dir / fn) for fn in filenames) + "\n")
    return out


def evaluate(run_dir: Path, split: str, weights: str, save_predictions: bool) -> dict:
    split_key = "val" if split in ("val", "valid") else "test"   # ultralytics split name
    images_dir = YOLO_DATA / "images" / split_key
    all_images = sorted(p.name for p in images_dir.iterdir() if p.suffix == ".jpg")
    canopy_map = json.loads((YOLO_DATA / "canopy_state_map.json").read_text())

    model = YOLO(str(run_dir / "weights" / weights))
    device = 0 if torch.cuda.is_available() else "cpu"
    tmp = Path(tempfile.mkdtemp(prefix="yolo_eval_"))

    def run(target: str, tag: str):
        y = write_split_yaml(tmp, split_key, target, tag)
        res = model.val(data=str(y), split=split_key, project=str(tmp),
                        name=f"val_{tag}", device=device, plots=False,
                        verbose=False, save_json=False, workers=0,   # 64 MB /dev/shm
                        half=True)   # FP16 to match training-time AMP validation
        # (verified: half=True reproduces the training epoch-86 val mask mAP@50
        #  0.629; FP32 gives 0.603. AMP-consistent eval, matches D004/training.)
        return metric_block(res)

    try:
        result = {"overall": run(f"images/{split_key}", "overall")}
        n_frames = {"overall": len(all_images)}
        for cstate in CANOPY_STATES:
            subset = [fn for fn in all_images if canopy_map.get(fn) == cstate]
            n_frames[cstate] = len(subset)
            lst = image_list(images_dir, subset, tmp / f"{cstate}.txt")
            result[cstate] = run(str(lst), cstate)
        result["n_frames"] = n_frames
        result["_meta"] = {"split": split_key, "weights": weights,
                           "run_dir": str(run_dir)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out_path = run_dir / f"{split_key}_metrics.json"
    out_path.write_text(json.dumps(result, indent=2))
    result["_meta"]["metrics_path"] = str(out_path)

    if save_predictions:
        pred_dir = run_dir / f"predictions_{split_key}"
        pred_dir.mkdir(parents=True, exist_ok=True)
        labels_dir = YOLO_DATA / "labels" / split_key
        canopy_map = json.loads((YOLO_DATA / "canopy_state_map.json").read_text())
        rows: List[dict] = []
        # One deterministic predict pass at the operating-point confidence. Rasterise
        # predicted instance masks -> binary foreground; compare to GT binary mask.
        for r in model.predict(source=str(images_dir), conf=PREDICT_CONF, half=True,
                               device=device, verbose=False, stream=True):
            fn = Path(r.path).name
            stem = Path(fn).stem
            h, w = r.orig_shape
            pred_polys = list(r.masks.xy) if r.masks is not None else []
            pred_mask = polygons_to_mask(pred_polys, h, w)
            gt_lines = (labels_dir / f"{stem}.txt").read_text().splitlines() \
                if (labels_dir / f"{stem}.txt").exists() else []
            gt_mask = polygons_to_mask(yolo_lines_to_polygons(gt_lines, w, h), h, w)
            rgb = cv2.cvtColor(cv2.imread(r.path), cv2.COLOR_BGR2RGB)
            save_gt_pred_panel(str(pred_dir / f"{stem}.png"), rgb, gt_mask, pred_mask)
            row = {"filename": fn, "scene_id": _RF_SUFFIX.sub("", fn),
                   "canopy_state": canopy_map.get(fn, "unknown")}
            row.update(_pixel_metrics(pred_mask, gt_mask))
            rows.append(row)

        rows.sort(key=lambda x: x["filename"])
        per_frame_path = run_dir / f"{split_key}_per_frame_metrics.csv"
        with open(per_frame_path, "w", newline="") as f:
            w_ = csv.DictWriter(f, fieldnames=PER_FRAME_COLUMNS)
            w_.writeheader(); w_.writerows(rows)
        result["_meta"]["predictions_dir"] = str(pred_dir)
        result["_meta"]["per_frame_csv"] = str(per_frame_path)
        result["_meta"]["predict_conf"] = PREDICT_CONF

    return result


def print_summary(result: dict) -> None:
    m = result["_meta"]
    print(f"\nPhase B eval — split={m['split']} | weights={m['weights']}")
    hdr = f"  {'stratum':<10}{'n':>5}{'mask_mAP50':>12}{'mask_mAP50-95':>15}{'mask_P':>9}{'mask_R':>9}{'box_mAP50':>11}"
    print(hdr)
    for key in ("overall", "bare_vine", "canopy"):
        b = result[key]; n = result["n_frames"][key]
        print(f"  {key:<10}{n:>5}{b['map50']:>12.4f}{b['map50_95']:>15.4f}"
              f"{b['precision']:>9.4f}{b['recall']:>9.4f}{b['box_map50']:>11.4f}")
    print(f"  -> {m['metrics_path']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase B YOLO evaluation (§7).")
    ap.add_argument("--run-dir", default=str(REPO / "results/runs/phase_b_yolo_binary"))
    ap.add_argument("--split", default="test", choices=["train", "val", "valid", "test"],
                    help="Split to evaluate. Test is the ONE locked evaluation (rule 5).")
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--no-predictions", action="store_true")
    args = ap.parse_args()

    if args.split == "test":
        print("[GUARDRAIL] Evaluating the TEST split — single locked Phase B "
              "evaluation (rule 5). Do not re-tune and re-run.")

    result = evaluate(Path(args.run_dir), args.split, args.weights,
                      save_predictions=not args.no_predictions)
    print_summary(result)


if __name__ == "__main__":
    main()
