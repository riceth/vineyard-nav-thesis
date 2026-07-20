#!/usr/bin/env python3
"""Class-coloured spot-check of Phase C multiclass YOLO labels (PHASE_C_SPEC 4.5).

Overlays ground-truth YOLO labels: trunk (class 0) = red, pole (class 1) = blue.
3-4 samples per canopy state; includes the Phase B diagnostic frames (6799, 6766,
april_104) for direct binary<->multiclass label cross-reference. Reads labels
from data/yolo_multiclass/labels/test (ground truth, not predictions).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.perception.segmentation.yolo_binary.visualize import polygons_to_mask

YD = REPO / "data/yolo_multiclass"
OUT = REPO / "results/runs/phase_c_yolo_multiclass_labels_spotcheck"
TRUNK_C, POLE_C = (255, 0, 0), (0, 80, 255)   # red, blue
ALPHA = 0.5
# Ensure the Phase B diagnostic frames are included for cross-reference.
MUST_INCLUDE = ["april_color_image_104", "color_image_6766", "color_image_6799"]


def class_polys(lines, cls, w, h):
    polys = []
    for ln in lines:
        v = ln.split()
        if v and int(v[0]) == cls:
            c = np.asarray(v[1:], float).reshape(-1, 2)
            c[:, 0] *= w; c[:, 1] *= h
            polys.append(c)
    return polys


def banner(img, text):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (min(img.shape[1], len(text) * 10 + 12), 26), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmap = json.loads((YD / "canopy_state_map.json").read_text())
    test = sorted(p.name for p in (YD / "images/test").iterdir() if p.suffix == ".jpg")

    def pick(state, n=4):
        must = [f for f in test if cmap[f] == state and any(m in f for m in MUST_INCLUDE)]
        rest = [f for f in test if cmap[f] == state and f not in must]
        return (must + rest)[:n]

    picks = pick("bare_vine", 4) + pick("canopy", 4)
    print(f"trunk=red  pole=blue  |  {len(picks)} frames")
    for fn in picks:
        stem = Path(fn).stem
        bgr = cv2.imread(str((YD / "images/test" / fn).resolve()))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        lines = (YD / "labels/test" / f"{stem}.txt").read_text().splitlines()
        trunk = polygons_to_mask(class_polys(lines, 0, w, h), h, w).astype(bool)
        pole = polygons_to_mask(class_polys(lines, 1, w, h), h, w).astype(bool)
        ov = rgb.astype(np.float32)
        ov[trunk] = (1 - ALPHA) * ov[trunk] + ALPHA * np.array(TRUNK_C, float)
        ov[pole] = (1 - ALPHA) * ov[pole] + ALPHA * np.array(POLE_C, float)
        nt = sum(1 for l in lines if l.startswith("0 "))
        npo = sum(1 for l in lines if l.startswith("1 "))
        out = banner(ov.clip(0, 255).astype(np.uint8),
                     f"{cmap[fn]}  trunk(red)={nt} pole(blue)={npo}")
        cv2.imwrite(str(OUT / f"{cmap[fn]}__{stem}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        print(f"  {cmap[fn]:<10} {fn[:46]:<48} trunk={nt} pole={npo}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
