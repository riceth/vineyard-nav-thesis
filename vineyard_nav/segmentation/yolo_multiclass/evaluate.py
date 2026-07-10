#!/usr/bin/env python3
"""Phase C multiclass evaluation (PHASE_C_SPEC section 9).

Faithful copy of segmentation/yolo_binary/evaluate.py, repointed at
data/yolo_multiclass (nc=2, trunk=0/pole=1). Identical eval logic so B<->C is a
controlled comparison. mAP metrics are the class-mean (with per-class available in
the ultralytics result); the per-frame rasterised fg IoU rasterises ALL detected
instance masks (trunk + pole) into one binary foreground union — the class-agnostic
cross-arm-comparable metric (F005), directly parallel to Phase A/B.

Generic rasterisation/render helpers are imported read-only from the Phase B
visualiser (not modified). TEST-SET GUARDRAIL (rule 5 / §9.2): `--split` defaults
to `test`, evaluated once; reproduce-on-val uses `--split val`.
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
import yaml

# 64 MB /dev/shm cannot back worker IPC in this container.
torch.multiprocessing.set_sharing_strategy("file_system")

from ultralytics import YOLO

from segmentation.yolo_binary.visualize import (polygons_to_mask, yolo_lines_to_polygons,
                                                save_gt_pred_panel)

REPO = Path("/workspaces/dissertation/vineyard_nav")
YOLO_DATA = REPO / "data/yolo_multiclass"
CANOPY_STATES = ("bare_vine", "canopy")
PREDICT_CONF = 0.25          # fallback; real value config-driven (eval.predict_conf)
_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-fA-F]+\.[A-Za-z0-9]+$")
PER_FRAME_COLUMNS = ["filename", "scene_id", "canopy_state", "iou_foreground",
                     "iou_background", "precision_foreground", "recall_foreground",
                     "f1_foreground"]


def _pixel_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Per-frame binary foreground pixel metrics (class-agnostic union; parallels A/B)."""
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
    """Mask (segmentation) + box metrics; map50 is the class-mean over trunk+pole.
    per_class holds trunk/pole AP (verified ultralytics API: res.seg.ap50/ap,
    res.box.ap50 are per-class arrays indexed by res.ap_class_index)."""
    per_class = {}
    for i, ci in enumerate(res.ap_class_index):
        per_class[res.names[int(ci)]] = {
            "mask_map50": float(res.seg.ap50[i]),
            "mask_map50_95": float(res.seg.ap[i]),
            "box_map50": float(res.box.ap50[i]),
        }
    return {
        "map50": float(res.seg.map50),
        "map50_95": float(res.seg.map),
        "precision": float(res.seg.mp),
        "recall": float(res.seg.mr),
        "box_map50": float(res.box.map50),
        "box_map50_95": float(res.box.map),
        "per_class": per_class,
    }


def write_split_yaml(tmp: Path, split_key: str, target: str, tag: str) -> Path:
    """data.yaml with all splits present (ultralytics requires train+val); only
    `split_key` points at `target`. nc=2 multiclass (trunk=0, pole=1)."""
    defaults = {"train": "images/train", "val": "images/val", "test": "images/test"}
    defaults[split_key] = target
    y = tmp / f"data_{split_key}_{tag}.yaml"
    y.write_text(
        f"path: {YOLO_DATA.resolve()}\n"
        f"train: {defaults['train']}\n"
        f"val: {defaults['val']}\n"
        f"test: {defaults['test']}\n"
        "nc: 2\nnames:\n  0: trunk\n  1: pole\n"
    )
    return y


def image_list(images_dir: Path, filenames: List[str], out: Path) -> Path:
    # Path UNDER data/yolo_multiclass/images/... (do NOT resolve the symlink), so
    # ultralytics derives labels/ by the images->labels string swap inside the tree.
    out.write_text("\n".join(str(images_dir / fn) for fn in filenames) + "\n")
    return out


def evaluate(run_dir: Path, split: str, weights: str, save_predictions: bool,
             predict_conf: float = PREDICT_CONF) -> dict:
    split_key = "val" if split in ("val", "valid") else "test"
    images_dir = YOLO_DATA / "images" / split_key
    all_images = sorted(p.name for p in images_dir.iterdir() if p.suffix == ".jpg")
    canopy_map = json.loads((YOLO_DATA / "canopy_state_map.json").read_text())

    model = YOLO(str(run_dir / "weights" / weights))
    device = 0 if torch.cuda.is_available() else "cpu"
    tmp = Path(tempfile.mkdtemp(prefix="yoloC_eval_"))

    def run(target: str, tag: str):
        y = write_split_yaml(tmp, split_key, target, tag)
        res = model.val(data=str(y), split=split_key, project=str(tmp),
                        name=f"val_{tag}", device=device, plots=False,
                        verbose=False, save_json=False, workers=0,
                        half=True)   # FP16 to match training-time AMP validation (D029)
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
        result["_meta"] = {"split": split_key, "weights": weights, "run_dir": str(run_dir)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out_path = run_dir / f"{split_key}_metrics.json"
    out_path.write_text(json.dumps(result, indent=2))
    result["_meta"]["metrics_path"] = str(out_path)

    if save_predictions:
        pred_dir = run_dir / f"predictions_{split_key}"
        pred_dir.mkdir(parents=True, exist_ok=True)
        labels_dir = YOLO_DATA / "labels" / split_key
        rows: List[dict] = []
        for r in model.predict(source=str(images_dir), conf=predict_conf, half=True,
                               device=device, verbose=False, stream=True):
            fn = Path(r.path).name
            stem = Path(fn).stem
            h, w = r.orig_shape
            pred_polys = list(r.masks.xy) if r.masks is not None else []   # all classes -> union
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
        result["_meta"]["predict_conf"] = predict_conf

    return result


def print_summary(result: dict) -> None:
    m = result["_meta"]
    print(f"\nPhase C eval — split={m['split']} | weights={m['weights']} (mAP = trunk+pole mean)")
    hdr = f"  {'stratum':<10}{'n':>5}{'mask_mAP50':>12}{'mask_mAP50-95':>15}{'mask_P':>9}{'mask_R':>9}{'box_mAP50':>11}"
    print(hdr)
    for key in ("overall", "bare_vine", "canopy"):
        b = result[key]; n = result["n_frames"][key]
        print(f"  {key:<10}{n:>5}{b['map50']:>12.4f}{b['map50_95']:>15.4f}"
              f"{b['precision']:>9.4f}{b['recall']:>9.4f}{b['box_map50']:>11.4f}")
    print(f"  -> {m['metrics_path']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase C YOLO multiclass evaluation (§9).")
    ap.add_argument("--run-dir", default=str(REPO / "results/runs/phase_c_yolo_multiclass"))
    ap.add_argument("--split", default="test", choices=["train", "val", "valid", "test"],
                    help="Split to evaluate. Test is the ONE locked evaluation (rule 5).")
    ap.add_argument("--weights", default="best.pt")
    ap.add_argument("--config", default=str(REPO / "configs/phase_c_yolo_multiclass_train.yaml"),
                    help="Source of eval.predict_conf (perception operating point).")
    ap.add_argument("--no-predictions", action="store_true")
    args = ap.parse_args()

    predict_conf = PREDICT_CONF
    try:
        cfg = yaml.safe_load(Path(args.config).read_text())
        predict_conf = float(cfg.get("eval", {}).get("predict_conf", PREDICT_CONF))
    except (OSError, ValueError, TypeError):
        pass
    print(f"[operating point] predict_conf = {predict_conf} (from {args.config})")

    if args.split == "test":
        print("[GUARDRAIL] Evaluating the TEST split — single locked Phase C "
              "evaluation (rule 5). Do not re-tune and re-run.")

    result = evaluate(Path(args.run_dir), args.split, args.weights,
                      save_predictions=not args.no_predictions, predict_conf=predict_conf)
    print_summary(result)


if __name__ == "__main__":
    main()
