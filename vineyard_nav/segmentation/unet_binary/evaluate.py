#!/usr/bin/env python3
"""Phase A evaluation (PHASE_A_SPEC section 9).

Loads a locked checkpoint, runs inference on a split, computes perception
metrics overall and stratified by canopy state, saves a GT-vs-prediction panel
for every frame, and writes <split>_metrics.json.

TEST-SET GUARDRAIL (rule 5 / PHASE_A_SPEC 9.2): the test split is evaluated
ONCE, only after training is complete and best.pt is locked, and results are not
iterated on. `--split` defaults to `test`; the dry-run during development passes
`--split valid`. Output is written to `<split>_metrics.json`, so a validation
dry-run never occupies the `test_metrics.json` filename.

Validation/test are evaluated on representative frames only (one per scene, D028),
which the dataset enforces by default for those splits.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict

import torch
import yaml
from torch.utils.data import DataLoader

from .dataset import SemanticBLTBinaryDataset, eval_transform
from .metrics import SegmentationMetrics
from .model import build_model
from .visualize import denormalize_to_uint8, save_gt_pred_panel

CLASS_NAMES = ["background", "foreground"]
CANOPY_STATES = ("bare_vine", "canopy")


def _spec_block(metrics: SegmentationMetrics) -> Dict[str, float]:
    """Map a SegmentationMetrics result to the PHASE_A_SPEC 9.1 JSON schema."""
    m = metrics.compute()
    return {
        "miou": m["miou"],
        "iou_foreground": m["iou"]["foreground"],
        "iou_background": m["iou"]["background"],
        "precision_foreground": m["precision"]["foreground"],
        "recall_foreground": m["recall"]["foreground"],
        "f1_foreground": m["f1"]["foreground"],
    }


@torch.no_grad()
def evaluate(run_dir: str, split: str, checkpoint: str = "best.pt",
             alpha: float = 0.5, save_predictions: bool = True) -> dict:
    ckpt_path = os.path.join(run_dir, "checkpoints", checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    d = config["data"]
    ds = SemanticBLTBinaryDataset(d["root"], split, d["split_manifest"],
                                  transform=eval_transform())   # representative-only for valid/test
    loader = DataLoader(ds, batch_size=config["train"]["batch_size"], shuffle=False,
                        num_workers=0, pin_memory=True)

    overall = SegmentationMetrics(num_classes=2, class_names=CLASS_NAMES)
    per_canopy = {c: SegmentationMetrics(num_classes=2, class_names=CLASS_NAMES)
                  for c in CANOPY_STATES}
    n_frames = defaultdict(int)

    pred_dir = os.path.join(run_dir, f"predictions_{split}")
    if save_predictions:
        os.makedirs(pred_dir, exist_ok=True)

    amp = device.type == "cuda"
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        canopy = batch["canopy_state"]           # list[str], length B
        filenames = batch["filename"]

        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
        preds = logits.argmax(dim=1)

        overall.update(preds, masks)
        n_frames["overall"] += len(filenames)
        for cstate in CANOPY_STATES:
            idx = [i for i, c in enumerate(canopy) if c == cstate]
            if idx:
                sel = torch.tensor(idx, device=device)
                per_canopy[cstate].update(preds.index_select(0, sel),
                                          masks.index_select(0, sel))
                n_frames[cstate] += len(idx)

        if save_predictions:
            preds_cpu = preds.cpu().numpy()
            masks_cpu = masks.cpu().numpy()
            for i, fn in enumerate(filenames):
                rgb = denormalize_to_uint8(images[i])
                out = os.path.join(pred_dir, f"{os.path.splitext(fn)[0]}.png")
                save_gt_pred_panel(out, rgb, masks_cpu[i], preds_cpu[i], alpha)

    result = {
        "overall": _spec_block(overall),
        "bare_vine": _spec_block(per_canopy["bare_vine"]),
        "canopy": _spec_block(per_canopy["canopy"]),
        "n_frames": {"overall": n_frames["overall"],
                     "bare_vine": n_frames["bare_vine"],
                     "canopy": n_frames["canopy"]},
        "_meta": {
            "split": split,
            "checkpoint": checkpoint,
            "checkpoint_epoch": ckpt.get("epoch"),
            "checkpoint_val_miou": ckpt.get("val_miou"),
            "git_commit": ckpt.get("git_commit"),
        },
    }

    out_path = os.path.join(run_dir, f"{split}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    result["_meta"]["metrics_path"] = out_path
    result["_meta"]["predictions_dir"] = pred_dir if save_predictions else None
    return result


def _print_summary(result: dict) -> None:
    m = result["_meta"]
    print(f"\nEvaluation — split={m['split']} | checkpoint={m['checkpoint']} "
          f"(epoch {m['checkpoint_epoch']}, val_miou {m['checkpoint_val_miou']:.4f})")
    hdr = f"  {'stratum':<10}{'n':>5}{'mIoU':>9}{'IoU_fg':>9}{'P_fg':>9}{'R_fg':>9}{'F1_fg':>9}"
    print(hdr)
    for key in ("overall", "bare_vine", "canopy"):
        b = result[key]
        n = result["n_frames"][key]
        print(f"  {key:<10}{n:>5}{b['miou']:>9.4f}{b['iou_foreground']:>9.4f}"
              f"{b['precision_foreground']:>9.4f}{b['recall_foreground']:>9.4f}"
              f"{b['f1_foreground']:>9.4f}")
    print(f"  -> {result['_meta']['metrics_path']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase A evaluation (PHASE_A_SPEC 9).")
    ap.add_argument("--run-dir", required=True, help="results/runs/<run> directory.")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"],
                    help="Split to evaluate. Test is the ONE locked evaluation (rule 5).")
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--no-predictions", action="store_true",
                    help="Skip saving per-frame prediction panels.")
    args = ap.parse_args()

    if args.split == "test":
        print("[GUARDRAIL] Evaluating the TEST split — this is the single locked "
              "Phase A test evaluation (rule 5). Do not re-tune and re-run.")

    result = evaluate(args.run_dir, args.split, args.checkpoint, args.alpha,
                      save_predictions=not args.no_predictions)
    _print_summary(result)


if __name__ == "__main__":
    main()
