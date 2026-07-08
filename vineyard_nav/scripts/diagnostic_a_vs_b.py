#!/usr/bin/env python3
"""Diagnostic: Phase A (U-Net) vs Phase B (YOLO) foreground prediction on test.

Renders, per selected test image, a 4-panel strip:
  [ image | GT (green) | Phase A U-Net (red) | Phase B YOLO rasterised @conf (blue) ]
to inspect WHY Phase A's rasterised fg IoU (~0.72) exceeds Phase B's (~0.56).

Diagnostic only: uses the LOCKED checkpoints, produces NEW panels under
diagnostic_panels/. Does NOT recompute or overwrite any committed test metric
(rule 5 — no test-set metric is produced here, only visualisations + per-image
IoU printed to console for context).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

torch.multiprocessing.set_sharing_strategy("file_system")

REPO = Path("/workspaces/dissertation/vineyard_nav")
sys.path.insert(0, str(REPO))
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2

from segmentation.unet_binary.model import UNetBinary
from segmentation.unet_binary.dataset import IMAGENET_MEAN, IMAGENET_STD
from segmentation.yolo_binary.visualize import (polygons_to_mask, yolo_lines_to_polygons,
                                                overlay_mask, _banner)

SEMANTICBLT = Path("/workspaces/dissertation/SemanticBLT.v1-2024-june.coco-segmentation")
YOLO_DATA = REPO / "data/yolo_binary"
PHASE_A_RUN = REPO / "results/runs/phase_a_unet_binary_20260704_004105"
PHASE_B_RUN = REPO / "results/runs/phase_b_yolo_binary"
OUT_DIR = PHASE_B_RUN / "diagnostic_panels"
CONF = 0.25
GT_C, A_C, B_C = (0, 200, 0), (255, 0, 0), (0, 90, 255)


def fg_iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    u = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / u) if u else float("nan")


def unet_mask(model, rgb, device):
    """Phase A U-Net foreground mask (argmax) at native 640, eval transform."""
    tf = A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
    x = tf(image=rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import json
    canopy = json.loads((YOLO_DATA / "canopy_state_map.json").read_text())
    test_imgs = sorted(p.name for p in (YOLO_DATA / "images" / "test").iterdir()
                       if p.suffix == ".jpg")
    bare = [f for f in test_imgs if canopy[f] == "bare_vine"][:2]
    can = [f for f in test_imgs if canopy[f] == "canopy"][:2]
    picks = bare + can

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNetBinary(encoder_weights=None).to(device).eval()
    ck = torch.load(PHASE_A_RUN / "checkpoints" / "best.pt", map_location=device,
                    weights_only=False)
    unet.load_state_dict(ck["model_state_dict"])
    yolo = YOLO(str(PHASE_B_RUN / "weights" / "best.pt"))

    print(f"{'image':<48}{'canopy':<11}{'A_IoU':>8}{'B_IoU':>8}")
    for fn in picks:
        stem = Path(fn).stem
        # image lives in the symlink target (SemanticBLT) via yolo images dir
        img_path = (YOLO_DATA / "images" / "test" / fn).resolve()
        bgr = cv2.imread(str(img_path)); rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        gt_lines = (YOLO_DATA / "labels" / "test" / f"{stem}.txt").read_text().splitlines()
        gt_mask = polygons_to_mask(yolo_lines_to_polygons(gt_lines, w, h), h, w)

        a_mask = unet_mask(unet, rgb, device)

        r = yolo.predict(source=str(img_path), conf=CONF, half=True, device=0
                         if device.type == "cuda" else "cpu", verbose=False)[0]
        b_polys = list(r.masks.xy) if r.masks is not None else []
        b_mask = polygons_to_mask(b_polys, h, w)

        panel = np.hstack([
            _banner(rgb, "image"),
            _banner(overlay_mask(rgb, gt_mask, GT_C), "GT"),
            _banner(overlay_mask(rgb, a_mask, A_C), f"A U-Net {fg_iou(a_mask, gt_mask):.2f}"),
            _banner(overlay_mask(rgb, b_mask, B_C), f"B YOLO {fg_iou(b_mask, gt_mask):.2f}"),
        ])
        cv2.imwrite(str(OUT_DIR / f"{canopy[fn]}__{stem}.png"),
                    cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        print(f"{fn[:46]:<48}{canopy[fn]:<11}{fg_iou(a_mask, gt_mask):>8.3f}"
              f"{fg_iou(b_mask, gt_mask):>8.3f}")
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
