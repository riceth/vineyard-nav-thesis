#!/usr/bin/env python3
"""Detection/segmentation overlay rendering for Phase B (PHASE_B_SPEC section 2).

Parallels segmentation/unet_binary/visualize.py: renders a GT-vs-prediction panel
(GT green, Pred red) so Phase B prediction visualisations are directly comparable
to Phase A's. Also provides polygon->mask rasterisation used by evaluate.py to
compute per-frame pixel-level foreground metrics. No model/eval logic here.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

GT_COLOR = (0, 200, 0)       # green = ground truth
PRED_COLOR = (255, 0, 0)     # red   = prediction


def polygons_to_mask(polygons: Sequence[np.ndarray], height: int, width: int) -> np.ndarray:
    """Rasterise polygons (each an Nx2 array of pixel coords) to a HxW uint8 {0,1}
    foreground mask (union of all polygons)."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        p = np.asarray(poly, dtype=np.float32)
        if p.ndim == 2 and p.shape[0] >= 3:
            cv2.fillPoly(mask, [p.round().astype(np.int32)], 1)
    return mask


def yolo_lines_to_polygons(lines: Sequence[str], width: int, height: int) -> List[np.ndarray]:
    """Parse YOLO-seg label lines ('<cls> x1 y1 ...' normalised) to pixel polygons."""
    polys = []
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        v = ln.split()
        coords = np.asarray(v[1:], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= width
        coords[:, 1] *= height
        polys.append(coords)
    return polys


def overlay_mask(rgb: np.ndarray, mask: np.ndarray,
                 color: Tuple[int, int, int] = PRED_COLOR, alpha: float = 0.5) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    out[m] = (1.0 - alpha) * out[m] + alpha * np.asarray(color, dtype=np.float32)
    return out.clip(0, 255).astype(np.uint8)


def _banner(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (len(text) * 11 + 10, 24), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)
    return img


def save_gt_pred_panel(path: str, rgb: np.ndarray, gt_mask: np.ndarray,
                       pred_mask: np.ndarray, alpha: float = 0.5) -> None:
    """Save a [GT | Pred] side-by-side overlay panel (GT green, Pred red)."""
    gt_ov = _banner(overlay_mask(rgb, gt_mask, GT_COLOR, alpha), "GT")
    pr_ov = _banner(overlay_mask(rgb, pred_mask, PRED_COLOR, alpha), "Pred")
    panel = np.hstack([gt_ov, pr_ov])
    cv2.imwrite(path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    import tempfile, os
    rgb = np.full((32, 48, 3), 90, np.uint8)
    gt = polygons_to_mask([np.array([[0, 0], [20, 0], [20, 30], [0, 30]])], 32, 48)
    pred = polygons_to_mask([np.array([[10, 0], [36, 0], [36, 30], [10, 30]])], 32, 48)
    assert gt.sum() > 0 and pred.sum() > 0
    ov = overlay_mask(rgb, pred, PRED_COLOR)
    assert ov.shape == rgb.shape and ov[0, 15, 0] > rgb[0, 15, 0]
    polys = yolo_lines_to_polygons(["0 0.0 0.0 0.5 0.0 0.5 0.5 0.0 0.5"], 48, 32)
    assert len(polys) == 1 and polys[0].shape == (4, 2)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.png"); save_gt_pred_panel(p, rgb, gt, pred)
        assert os.path.getsize(p) > 0
    print("visualize.py self-test passed.")
