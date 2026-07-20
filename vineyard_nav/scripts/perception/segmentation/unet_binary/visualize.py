#!/usr/bin/env python3
"""Mask overlay rendering for Phase A (PHASE_A_SPEC section 4.5 / section 2).

Renders a foreground mask as a semi-transparent colour overlay on the RGB frame.
Used by evaluate.py to save a GT-vs-prediction panel for every evaluated frame,
and reusable for ad-hoc inspection. No training/eval logic lives here.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np
import torch

from .dataset import IMAGENET_MEAN, IMAGENET_STD

GT_COLOR = (0, 200, 0)       # green  = ground truth
PRED_COLOR = (255, 0, 0)     # red    = prediction


def denormalize_to_uint8(image: torch.Tensor) -> np.ndarray:
    """ImageNet-normalised [3, H, W] tensor -> HxWx3 uint8 RGB."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = (image.detach().cpu() * std + mean).clamp(0, 1)
    return (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray,
                 color: Tuple[int, int, int] = PRED_COLOR, alpha: float = 0.5) -> np.ndarray:
    """Composite `mask` (HxW, any truthy dtype) onto `rgb` (HxWx3 uint8) as a
    semi-transparent colour overlay. Returns a new uint8 array."""
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    c = np.asarray(color, dtype=np.float32)
    out[m] = (1.0 - alpha) * out[m] + alpha * c
    return out.clip(0, 255).astype(np.uint8)


def _banner(img: np.ndarray, text: str) -> np.ndarray:
    """Draw a small label banner in the top-left corner (returns a copy)."""
    img = img.copy()
    cv2.rectangle(img, (0, 0), (len(text) * 11 + 10, 24), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return img


def side_by_side(panels: Sequence[np.ndarray], labels: Sequence[str]) -> np.ndarray:
    """Horizontally stack labelled panels (all same HxWx3)."""
    assert len(panels) == len(labels)
    return np.hstack([_banner(p, l) for p, l in zip(panels, labels)])


def save_gt_pred_panel(path: str, rgb: np.ndarray, gt_mask: np.ndarray,
                       pred_mask: np.ndarray, alpha: float = 0.5) -> None:
    """Save a [GT | Pred] side-by-side overlay panel (GT green, Pred red)."""
    gt_ov = overlay_mask(rgb, gt_mask, GT_COLOR, alpha)
    pr_ov = overlay_mask(rgb, pred_mask, PRED_COLOR, alpha)
    panel = side_by_side([gt_ov, pr_ov], ["GT", "Pred"])
    cv2.imwrite(path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    # Self-test on synthetic data: shapes/colours correct, file writes.
    import tempfile, os
    rgb = np.full((32, 48, 3), 100, np.uint8)
    gt = np.zeros((32, 48), np.uint8); gt[:, :24] = 1
    pred = np.zeros((32, 48), np.uint8); pred[:, 12:36] = 1
    ov = overlay_mask(rgb, pred, PRED_COLOR, 0.5)
    assert ov.shape == rgb.shape and ov.dtype == np.uint8
    assert ov[0, 20, 0] > rgb[0, 20, 0]        # red channel raised where pred==1
    panel = side_by_side([rgb, ov], ["a", "b"])
    assert panel.shape == (32, 96, 3)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.png")
        save_gt_pred_panel(p, rgb, gt, pred)
        assert os.path.getsize(p) > 0
    print("visualize.py self-test passed.")
