#!/usr/bin/env python3
"""SemanticBLT binary dataset for Phase A (U-Net binary baseline).

Contract: docs/PHASE_A_SPEC.md section 4. Split source: the scene-level resplit
manifest produced by scripts/perception/pipeline/resplit_dataset.py (decision D028).

Label-collapsing rule (LOCKED, D006 / PHASE_A_SPEC 4.2):
  foreground (mask == 1) = any pixel covered by a COCO annotation whose
  category_id is in {3 (pole), 5 (trunk)}; everything else is background
  (mask == 0), including pipe (2), building (1), robot (4), vehicle (6) and
  unannotated pixels. Overlapping foreground annotations simply union, so
  "foreground wins" is automatic.

Split semantics (D028 consumption rule):
  * train         -> all frames in the split (Roboflow augmentations included;
                     further on-the-fly augmentation applied by `train_transform`)
  * valid / test  -> representative frames only (one per scene), so perception
                     metrics are computed over independent scenes.
  Override via `representative_only`.

Category IDs, image size (640x640, native) and the manifest<->COCO filename
join were verified against the export before this module was written.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2
from pycocotools import mask as mask_utils

# --- Constants (all verified against the SemanticBLT COCO export) -------------
FOREGROUND_CATEGORY_IDS = (3, 5)          # 3 = pole, 5 = trunk (D006)
IMAGE_SIZE = 640                          # native resolution, no downsampling (D005)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
COCO_ANNOTATION_FILE = "_annotations.coco.json"


def canopy_state_from_filename(filename: str) -> str:
    """Canopy state from filename prefix (PHASE_A_SPEC 4.3, LOCKED).

    Bare-vine prefixes are checked first so `april_color_image_*` resolves to
    bare_vine (not canopy). No silent default.
    """
    name = filename.lower()
    if name.startswith("march_") or name.startswith("april_"):
        return "bare_vine"
    if name.startswith("may_") or name.startswith("color_image_"):
        return "canopy"
    raise ValueError(f"Cannot determine canopy state from filename: {filename!r}")


# --- Transforms (PHASE_A_SPEC 4.4, D013) --------------------------------------
def train_transform(seed: Optional[int] = None) -> A.Compose:
    """Train-split augmentation. No vertical flip, no cropping (rows have
    orientation; downstream needs full-frame geometry).

    `seed` seeds the pipeline's internal RNG. This is REQUIRED for reproducibility:
    albumentations 2.x does NOT honour random.seed()/np.random.seed() globals, so
    without Compose(seed=...) augmentations vary run-to-run even at fixed global
    seeds (verified). A seeded Compose is reproducible across runs yet still varies
    augmentations per sample/epoch (rule 7 / D016 / PHASE_A_SPEC 8.3).
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ], seed=seed)


def eval_transform() -> A.Compose:
    """Validation/test transform: normalisation only (no augmentation)."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _segmentation_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    """Rasterise one COCO annotation's segmentation to a HxW uint8 {0,1} mask.

    Mirrors pycocotools' COCO.annToMask logic, covering polygon lists, uncompressed
    RLE, and compressed RLE. SemanticBLT annotations are polygon lists.
    """
    if isinstance(segmentation, list):                      # polygon(s)
        rles = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation.get("counts"), list):      # uncompressed RLE
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:                                                   # already compressed RLE
        rle = segmentation
    return mask_utils.decode(rle).astype(np.uint8)


class SemanticBLTBinaryDataset(Dataset):
    """Binary (trunk+pole -> foreground) SemanticBLT dataset for a resplit split."""

    def __init__(
        self,
        root: str,
        split: str,
        split_manifest: str,
        transform: Optional[A.Compose] = None,
        representative_only: Optional[bool] = None,
    ):
        """
        Args:
            root: dataset root containing train/ valid/ test/ source folders
                  (e.g. .../vineyard_nav/data/semanticblt). Images and COCO
                  annotations are read from root/<orig_split>/.
            split: "train" | "valid" | "test" (the *new* resplit split).
            split_manifest: path to the D028 resplit manifest JSON.
            transform: albumentations pipeline ending in ToTensorV2. If None,
                       defaults to train_transform() for the train split and
                       eval_transform() otherwise (enforces "no aug on val/test").
            representative_only: if None, defaults to False for train and True
                       for valid/test (D028 consumption rule).
        """
        if split not in ("train", "valid", "test"):
            raise ValueError(f"split must be train|valid|test, got {split!r}")

        self.root = root
        self.split = split
        self.split_manifest = split_manifest
        self.transform = transform if transform is not None else (
            train_transform(seed=42) if split == "train" else eval_transform()  # 42 = D016 project seed
        )
        self.representative_only = (
            (split != "train") if representative_only is None else representative_only
        )

        with open(split_manifest) as f:
            manifest = json.load(f)

        rows = [r for r in manifest["images"] if r["split"] == split]
        if self.representative_only:
            rows = [r for r in rows if r["is_representative"]]
        if not rows:
            raise RuntimeError(f"No rows for split={split!r} in {split_manifest}")

        # Per-source-folder COCO index: file_name -> image record, image_id -> [fg segs].
        self._coco_images: Dict[str, Dict[str, dict]] = {}
        self._fg_segs: Dict[str, Dict[int, List]] = {}
        for orig_split in sorted({r["orig_split"] for r in rows}):
            self._load_coco_index(orig_split)

        # Materialise per-sample records with the geometry needed at __getitem__.
        self.records: List[dict] = []
        for r in rows:
            img_rec = self._coco_images[r["orig_split"]][r["filename"]]
            # Manifest is authoritative for canopy; assert it matches the LOCKED
            # filename rule so a corrupt manifest cannot silently mislabel.
            assert r["canopy_state"] == canopy_state_from_filename(r["filename"]), (
                f"Canopy mismatch for {r['filename']}")
            self.records.append({
                "filename": r["filename"],
                "orig_split": r["orig_split"],
                "canopy_state": r["canopy_state"],
                "scene_id": r["scene_id"],
                "is_representative": r["is_representative"],
                "image_id": int(img_rec["id"]),
                "height": int(img_rec["height"]),
                "width": int(img_rec["width"]),
            })

    def _load_coco_index(self, orig_split: str) -> None:
        ann_path = os.path.join(self.root, orig_split, COCO_ANNOTATION_FILE)
        with open(ann_path) as f:
            coco = json.load(f)
        self._coco_images[orig_split] = {im["file_name"]: im for im in coco["images"]}
        fg: Dict[int, List] = defaultdict(list)
        for a in coco["annotations"]:
            if a["category_id"] in FOREGROUND_CATEGORY_IDS:
                fg[a["image_id"]].append(a["segmentation"])
        self._fg_segs[orig_split] = fg

    def __len__(self) -> int:
        return len(self.records)

    def _binary_mask(self, rec: dict) -> np.ndarray:
        h, w = rec["height"], rec["width"]
        mask = np.zeros((h, w), dtype=np.uint8)
        for seg in self._fg_segs[rec["orig_split"]].get(rec["image_id"], []):
            mask |= _segmentation_to_mask(seg, h, w)
        return mask

    def _load_rgb(self, rec: dict) -> np.ndarray:
        path = os.path.join(self.root, rec["orig_split"], rec["filename"])
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        image = self._load_rgb(rec)                 # HxWx3 uint8 RGB
        mask = self._binary_mask(rec)               # HxW uint8 {0,1}

        out = self.transform(image=image, mask=mask)
        image_t, mask_t = out["image"], out["mask"]
        mask_t = mask_t.long()                      # int64 {0,1}, shape HxW

        return {
            "image": image_t,                       # float32 [3, H, W], normalised
            "mask": mask_t,                         # int64 [H, W] in {0, 1}
            "canopy_state": rec["canopy_state"],    # "bare_vine" | "canopy"
            "image_id": rec["image_id"],
            "filename": rec["filename"],
        }

    # --- Visualisation (PHASE_A_SPEC 3 step 1 / 4.5) --------------------------
    def overlay(self, idx: int, alpha: float = 0.5) -> np.ndarray:
        """Return the raw RGB frame with the foreground mask as a semi-transparent
        red overlay (uint8 HxWx3). Independent of `self.transform` so it always
        shows the true image and label, for the manual spot-check gate.
        """
        rec = self.records[idx]
        image = self._load_rgb(rec).astype(np.float32)
        mask = self._binary_mask(rec).astype(bool)
        red = np.array([255.0, 0.0, 0.0], dtype=np.float32)
        image[mask] = (1.0 - alpha) * image[mask] + alpha * red
        return image.clip(0, 255).astype(np.uint8)


def _spotcheck(manifest: str, root: str, per_canopy: int = 5) -> None:
    """Save foreground-overlay images for a few frames per canopy state to
    results/runs/dataset_spotcheck_<timestamp>/ for the manual verification gate.
    """
    from datetime import datetime

    ds = SemanticBLTBinaryDataset(root=root, split="train",
                                  split_manifest=manifest, representative_only=False)
    picks: Dict[str, List[int]] = defaultdict(list)
    for i, rec in enumerate(ds.records):
        if len(picks[rec["canopy_state"]]) < per_canopy:
            picks[rec["canopy_state"]].append(i)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, "..", "..", "results", "runs",
                           f"dataset_spotcheck_{ts}")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    n = 0
    for canopy, idxs in sorted(picks.items()):
        for i in idxs:
            rec = ds.records[i]
            ov = ds.overlay(i)
            mask = ds._binary_mask(rec)
            fg_frac = float(mask.mean())
            fname = f"{canopy}__{rec['scene_id']}__fg{fg_frac:.3f}.png"
            cv2.imwrite(os.path.join(out_dir, fname),
                        cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
            n += 1
    print(f"Wrote {n} spot-check overlays to {out_dir}")
    print("Manual gate: confirm red covers trunks + poles and EXCLUDES pipes, "
          "then approve before model.py (PHASE_A_SPEC 3 step 2).")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Dataset spot-check (PHASE_A_SPEC 3.2).")
    ap.add_argument("--root",
                    default="/workspaces/dissertation/vineyard_nav/data/semanticblt")
    ap.add_argument("--manifest",
                    default="/workspaces/dissertation/vineyard_nav/data/splits/resplit_70_20_10.json")
    ap.add_argument("--per-canopy", type=int, default=5)
    args = ap.parse_args()
    _spotcheck(args.manifest, args.root, args.per_canopy)
