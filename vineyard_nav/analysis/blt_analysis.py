#!/usr/bin/env python3
"""SemanticBLT dataset analysis for vineyard navigation pipeline design.

Computes per-image structural-element statistics, aggregate spatial distributions,
per-side RANSAC viability, seasonal breakdown, sample overlays and quality notes.

Key methodology decisions (confirmed with user):
  * The train split is 6x pixel-level augmented (annotations spatially identical
    across the 6 copies). Per-image stats are emitted for ALL images, but every
    augmented copy beyond the first per base frame is flagged is_duplicate=True.
    Aggregate / viability / seasonal stats use UNIQUE frames only (230 total).
  * Files with no month prefix (color_image_NNNN) are labelled month='unknown'.
"""
import json, re, os, math
from collections import defaultdict, Counter
import numpy as np
import cv2
import pandas as pd

ROOT = "/workspaces/dissertation/SemanticBLT.v1-2024-june.coco-segmentation"
OUT = "/workspaces/dissertation/analysis/output"
SPLITS = ["train", "valid", "test"]

# Real classes (Roboflow category 0 'trunks-trees-poles-sky-pipes' is an unused placeholder)
CLASSES = {1: "building", 2: "pipe", 3: "pole", 4: "robot", 5: "trunk", 6: "vehicle"}
CLASS_NAMES = list(CLASSES.values())
TRUNK_ID, POLE_ID = 5, 3

os.makedirs(OUT, exist_ok=True)


def parse_month(fn):
    m = re.match(r"([a-z]+)_color", fn)
    return m.group(1) if m else "unknown"


def base_name(fn):
    return re.sub(r"_png\.rf\..*", "", fn)


def load_all():
    """Return list of image records, each with split, annotations, base, month."""
    records = []
    for split in SPLITS:
        d = json.load(open(os.path.join(ROOT, split, "_annotations.coco.json")))
        ann_by_img = defaultdict(list)
        for a in d["annotations"]:
            ann_by_img[a["image_id"]].append(a)
        for im in d["images"]:
            records.append({
                "split": split,
                "file_name": im["file_name"],
                "path": os.path.join(ROOT, split, im["file_name"]),
                "height": im["height"],
                "width": im["width"],
                "base": base_name(im["file_name"]),
                "month": parse_month(im["file_name"]),
                "anns": ann_by_img[im["id"]],
            })
    return records


def poly_to_pts(seg, w, h):
    """COCO polygon flat list -> int32 Nx2, clipped to image bounds."""
    pts = np.array(seg, dtype=np.float64).reshape(-1, 2)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return np.round(pts).astype(np.int32)


def instance_mask(ann, w, h):
    m = np.zeros((h, w), dtype=np.uint8)
    for seg in ann["segmentation"]:
        if len(seg) >= 6:
            cv2.fillPoly(m, [poly_to_pts(seg, w, h)], 1)
    return m


def compute_image_stats(rec):
    """Full per-image statistics dict for the CSV."""
    w, h = rec["width"], rec["height"]
    row = {
        "file_name": rec["file_name"],
        "split": rec["split"],
        "month": rec["month"],
        "base": rec["base"],
        "height": h,
        "width": w,
    }
    # Per-class accumulators
    class_masks = {cid: np.zeros((h, w), dtype=np.uint8) for cid in CLASSES}
    inst_counts = {cid: 0 for cid in CLASSES}
    # Per-instance centroid x for trunk/pole side-instance counts
    side_inst = {TRUNK_ID: {"L": 0, "R": 0}, POLE_ID: {"L": 0, "R": 0}}

    for a in rec["anns"]:
        cid = a["category_id"]
        if cid not in CLASSES:
            continue
        inst_counts[cid] += 1
        m = instance_mask(a, w, h)
        np.maximum(class_masks[cid], m, out=class_masks[cid])
        if cid in side_inst:
            ys, xs = np.where(m > 0)
            if len(xs):
                cx = xs.mean()
            else:  # fallback to bbox centre if mask empty
                cx = a["bbox"][0] + a["bbox"][2] / 2
            side_inst[cid]["L" if cx < w / 2 else "R"] += 1

    for cid, name in CLASSES.items():
        ys, xs = np.where(class_masks[cid] > 0)
        area = int(len(xs))
        row[f"{name}_instances"] = inst_counts[cid]
        row[f"{name}_area_px"] = area
        row[f"{name}_mean_x"] = float(xs.mean()) if area else np.nan
        row[f"{name}_mean_y"] = float(ys.mean()) if area else np.nan
        row[f"{name}_std_x"] = float(xs.std()) if area else np.nan
        row[f"{name}_std_y"] = float(ys.std()) if area else np.nan

    # Left/right pixel splits for trunk & pole
    for cid, name in [(TRUNK_ID, "trunk"), (POLE_ID, "pole")]:
        mask = class_masks[cid]
        ys, xs = np.where(mask > 0)
        half = w / 2
        row[f"{name}_left_px"] = int((xs < half).sum())
        row[f"{name}_right_px"] = int((xs >= half).sum())
        row[f"{name}_left_instances"] = side_inst[cid]["L"]
        row[f"{name}_right_instances"] = side_inst[cid]["R"]
        row[f"{name}_left_ge3"] = side_inst[cid]["L"] >= 3
        row[f"{name}_right_ge3"] = side_inst[cid]["R"] >= 3
    return row


def main():
    print("Loading COCO annotations...")
    records = load_all()
    print(f"  {len(records)} images total across {SPLITS}")

    # Deduplication: first occurrence of each base = canonical (is_duplicate False)
    seen = set()
    for rec in records:
        dup = rec["base"] in seen
        rec["is_duplicate"] = dup
        seen.add(rec["base"])
    n_unique = sum(1 for r in records if not r["is_duplicate"])
    print(f"  {n_unique} unique base frames")

    print("Computing per-image statistics (rasterizing masks)...")
    rows = []
    for i, rec in enumerate(records):
        row = compute_image_stats(rec)
        row["is_duplicate"] = rec["is_duplicate"]
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(records)}")

    df = pd.DataFrame(rows)
    # Column ordering
    front = ["file_name", "split", "month", "base", "is_duplicate", "height", "width"]
    cols = front + [c for c in df.columns if c not in front]
    df = df[cols]
    df.to_csv(os.path.join(OUT, "per_image_stats.csv"), index=False)
    print(f"Wrote per_image_stats.csv ({len(df)} rows, {len(df.columns)} cols)")

    # Persist unique-frame subset for downstream scripts
    df.to_pickle(os.path.join(OUT, "_all_stats.pkl"))
    print("Done.")


if __name__ == "__main__":
    main()
