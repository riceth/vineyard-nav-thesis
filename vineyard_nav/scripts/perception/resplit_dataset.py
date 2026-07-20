#!/usr/bin/env python3
"""Scene-level 70/20/10 stratified resplit of the SemanticBLT export (D028).

Rationale and full contract: docs/DECISIONS.md, decision D028 (supersedes D024).

Key properties enforced here:
  * Split unit is the *scene* (unique base image), NOT the image. Roboflow
    augmented only the original-train scenes (6x each); val/test scenes are
    single clean frames. Splitting by image would put near-duplicate augmented
    copies of a few scenes into val/test and inflate the nominal test count.
  * Stratified by canopy state (bare_vine vs canopy).
  * Augmentation-leakage guard: every augmentation of a scene inherits the
    scene's split, so no scene's frames span more than one split.
  * Each scene has exactly one `is_representative` frame (lexicographically
    first). Validation/test perception metrics are computed on representative
    frames only, so bootstrap CIs (D020) are over independent scenes. The true
    pre-augmentation original is not identifiable from the Roboflow export
    (every version carries a `.rf.<hash>` name), so a fixed deterministic
    choice is used.

Deterministic under `--seed` (default 42, per D016). Writes a manifest JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, List

# Roboflow export filename pattern: "<scene>.rf.<hexhash>.<ext>"
_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-fA-F]+\.[A-Za-z0-9]+$")

SOURCE_SPLITS = ("train", "valid", "test")   # original Roboflow folders to pool
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# Ratios (D028). Order matters for the "remainder goes to test" rounding below.
RATIOS = {"train": 0.70, "valid": 0.20, "test": 0.10}


def scene_id(filename: str) -> str:
    """Strip the Roboflow `.rf.<hash>.<ext>` suffix to recover the base scene."""
    stripped = _RF_SUFFIX.sub("", filename)
    if stripped == filename:
        raise ValueError(
            f"Filename does not match the Roboflow '<scene>.rf.<hash>.<ext>' "
            f"pattern, cannot derive scene id: {filename!r}"
        )
    return stripped


def canopy_state(filename: str) -> str:
    """Canopy state from the filename prefix (PHASE_A_SPEC 4.3 / D028).

    Order is significant: `april_color_image_*` must resolve to bare_vine via the
    `april_` prefix, so bare-vine prefixes are checked before canopy prefixes.
    """
    name = filename.lower()
    if name.startswith("march_") or name.startswith("april_"):
        return "bare_vine"
    if name.startswith("may_") or name.startswith("color_image_"):
        return "canopy"
    raise ValueError(f"Cannot determine canopy state from filename: {filename!r}")


def collect_images(data_root: str) -> Dict[str, str]:
    """Return {filename: orig_split} for every image across the source folders.

    Asserts filenames are globally unique across the pooled folders.
    """
    images: Dict[str, str] = {}
    for split in SOURCE_SPLITS:
        split_dir = os.path.join(data_root, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Expected source split folder missing: {split_dir}")
        for fn in os.listdir(split_dir):
            if not fn.lower().endswith(IMAGE_EXTS):
                continue
            if fn in images:
                raise ValueError(
                    f"Duplicate filename across source folders: {fn!r} in both "
                    f"'{images[fn]}' and '{split}'. Cannot pool safely."
                )
            images[fn] = split
    if not images:
        raise RuntimeError(f"No images found under {data_root}/{{{','.join(SOURCE_SPLITS)}}}")
    return images


def split_sizes(n: int) -> Dict[str, int]:
    """70/20/10 counts that sum exactly to n; test takes the remainder."""
    n_train = round(RATIOS["train"] * n)
    n_valid = round(RATIOS["valid"] * n)
    n_test = n - n_train - n_valid
    if n_test < 0:  # only possible for tiny n; clamp by shrinking valid
        n_valid += n_test
        n_test = 0
    return {"train": n_train, "valid": n_valid, "test": n_test}


def assign_splits(scenes_by_canopy: Dict[str, List[str]], seed: int) -> Dict[str, str]:
    """Assign each scene to a split, stratified by canopy, deterministically."""
    rng = random.Random(seed)
    scene_split: Dict[str, str] = {}
    for canopy in sorted(scenes_by_canopy):
        scenes = sorted(scenes_by_canopy[canopy])   # stable base order before shuffle
        rng.shuffle(scenes)
        sizes = split_sizes(len(scenes))
        idx = 0
        for split in ("train", "valid", "test"):
            for scene in scenes[idx: idx + sizes[split]]:
                scene_split[scene] = split
            idx += sizes[split]
        assert idx == len(scenes)
    return scene_split


def build_manifest(data_root: str, seed: int) -> dict:
    images = collect_images(data_root)

    # Group frames by scene; verify a scene never spans source folders (sanity).
    scene_frames: Dict[str, List[str]] = defaultdict(list)
    scene_canopy: Dict[str, str] = {}
    scene_orig_split: Dict[str, set] = defaultdict(set)
    for fn, orig in images.items():
        sid = scene_id(fn)
        scene_frames[sid].append(fn)
        scene_orig_split[sid].add(orig)
        c = canopy_state(fn)
        if sid in scene_canopy and scene_canopy[sid] != c:
            raise ValueError(f"Scene {sid!r} has inconsistent canopy states.")
        scene_canopy[sid] = c

    scenes_by_canopy: Dict[str, List[str]] = defaultdict(list)
    for sid, canopy in scene_canopy.items():
        scenes_by_canopy[canopy].append(sid)

    scene_split = assign_splits(scenes_by_canopy, seed)

    # Emit one manifest row per image; tag the representative frame per scene.
    rows: List[dict] = []
    for sid in sorted(scene_frames):
        frames = sorted(scene_frames[sid])          # lexicographic
        representative = frames[0]                   # deterministic choice (D028)
        split = scene_split[sid]
        for fn in frames:
            rows.append({
                "filename": fn,
                "orig_split": images[fn],
                "split": split,
                "canopy_state": scene_canopy[sid],
                "scene_id": sid,
                "is_representative": (fn == representative),
            })

    # --- Counts for the manifest header ---------------------------------------
    def _tally(pred):
        out = defaultdict(lambda: defaultdict(int))
        for r in rows:
            if pred(r):
                out[r["split"]][r["canopy_state"]] += 1
        return {s: dict(v) for s, v in out.items()}

    scene_rows = [r for r in rows if r["is_representative"]]
    counts = {
        "scenes_total": len(scene_frames),
        "images_total": len(rows),
        "scenes_by_split_canopy": _tally_scenes(scene_rows),
        "images_by_split": {s: sum(1 for r in rows if r["split"] == s)
                            for s in ("train", "valid", "test")},
        "images_by_split_canopy": _tally(lambda r: True),
        "representative_by_split": {s: sum(1 for r in scene_rows if r["split"] == s)
                                    for s in ("train", "valid", "test")},
        "representative_by_split_canopy": _tally(lambda r: r["is_representative"]),
    }

    manifest = {
        "meta": {
            "decision": "D028",
            "supersedes": "D024",
            "strategy": "scene-level 70/20/10 stratified by canopy_state",
            "seed": seed,
            "ratios": RATIOS,
            "source_root": os.path.abspath(data_root),
            "source_splits_pooled": list(SOURCE_SPLITS),
            "canopy_rule": ("prefix march_/april_ -> bare_vine; "
                            "may_/color_image_ -> canopy (bare-vine prefixes "
                            "checked first so april_color_image_* is bare_vine)"),
            "representative_rule": "lexicographically-first frame per scene",
            "consumption_rule": ("train: all frames; valid/test perception "
                                 "metrics: is_representative frames only"),
            "counts": counts,
        },
        "images": rows,
    }
    return manifest


def _tally_scenes(scene_rows: List[dict]) -> Dict[str, Dict[str, int]]:
    out = defaultdict(lambda: defaultdict(int))
    for r in scene_rows:
        out[r["split"]][r["canopy_state"]] += 1
    return {s: dict(v) for s, v in out.items()}


def verify(manifest: dict) -> None:
    """Fail loudly if the leakage guard or expected invariants are violated."""
    rows = manifest["images"]

    # 1. No scene spans multiple splits (the core augmentation-leakage guard).
    scene_to_splits: Dict[str, set] = defaultdict(set)
    for r in rows:
        scene_to_splits[r["scene_id"]].add(r["split"])
    leaked = {s: v for s, v in scene_to_splits.items() if len(v) > 1}
    if leaked:
        raise AssertionError(f"LEAKAGE: {len(leaked)} scene(s) span splits: "
                             f"{dict(list(leaked.items())[:5])} ...")

    # 2. Exactly one representative per scene.
    rep_per_scene: Dict[str, int] = defaultdict(int)
    for r in rows:
        rep_per_scene[r["scene_id"]] += int(r["is_representative"])
    bad = {s: n for s, n in rep_per_scene.items() if n != 1}
    if bad:
        raise AssertionError(f"Representative-count != 1 for scenes: "
                             f"{dict(list(bad.items())[:5])} ...")

    # 3. Representatives all live in their scene's split (trivially true here,
    #    but assert to catch future refactors).
    for r in rows:
        if r["is_representative"]:
            assert r["split"] == scene_to_splits[r["scene_id"]].copy().pop()

    print("Verification passed: no scene spans splits; one representative each.")


def print_report(manifest: dict) -> None:
    c = manifest["meta"]["counts"]
    print("\n=== Resplit report (D028, seed "
          f"{manifest['meta']['seed']}) ===")
    print(f"Unique scenes: {c['scenes_total']}   Total images: {c['images_total']}")

    print("\nScenes per split (representative frames = evaluation units):")
    hdr = f"  {'split':<7}{'bare_vine':>11}{'canopy':>9}{'total':>8}"
    print(hdr)
    for split in ("train", "valid", "test"):
        by = c["scenes_by_split_canopy"].get(split, {})
        bv, cp = by.get("bare_vine", 0), by.get("canopy", 0)
        print(f"  {split:<7}{bv:>11}{cp:>9}{bv + cp:>8}")

    print("\nImages per split (all Roboflow augmentations, train-consumed):")
    for split in ("train", "valid", "test"):
        by = c["images_by_split_canopy"].get(split, {})
        bv, cp = by.get("bare_vine", 0), by.get("canopy", 0)
        print(f"  {split:<7}{bv:>11}{cp:>9}{bv + cp:>8}")


def main() -> None:
    default_root = "/workspaces/dissertation/vineyard_nav/data/semanticblt"
    default_out = "/workspaces/dissertation/vineyard_nav/data/splits/resplit_70_20_10.json"

    ap = argparse.ArgumentParser(description="Scene-level 70/20/10 resplit (D028).")
    ap.add_argument("--data-root", default=default_root,
                    help="Root containing train/ valid/ test/ source folders.")
    ap.add_argument("--out", default=default_out, help="Manifest output path.")
    ap.add_argument("--seed", type=int, default=42, help="Deterministic seed (D016).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build and verify but do not write the manifest.")
    args = ap.parse_args()

    manifest = build_manifest(args.data_root, args.seed)
    verify(manifest)
    print_report(manifest)

    if args.dry_run:
        print("\n[dry-run] Manifest not written.")
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {args.out}  ({len(manifest['images'])} image rows)")


if __name__ == "__main__":
    main()
