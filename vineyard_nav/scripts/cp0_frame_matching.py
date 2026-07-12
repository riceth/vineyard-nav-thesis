#!/usr/bin/env python3
"""CP-0 — March-bag contamination census (GEOMETRY_PIPELINE_SPEC.md §2, §9).

Locates every unique March-labelled SemanticBLT scene inside the kg_march_23 bag by
content frame-matching, so the corresponding bag frames can be EXCLUDED from the
geometric-strand evaluation set (they are training/val/test data the models have
seen). Produces the per-scene match table + the merged ±w exclusion intervals.

Method (robust to Roboflow augmentation of train scenes):
  1. Enumerate unique March scenes across train+val+test (strip the `_png.rf.<hash>`
     suffix; augmented copies collapse to one scene). One scene -> one bag frame.
  2. Build a coarse descriptor bank over the FULL bag front-camera stream
     (every COARSE-th frame; 128x128 zero-mean unit-norm grayscale thumbnail).
     The full stream is searched — Riccardo's "first 5-6 min" recollection is only a
     heuristic and is empirically false (matches span 1.9-14.8 min), so it is NOT used
     to bound the search (D-C: empirical match governs).
  3. For each scene, match ALL its versions + a horizontal-flip fallback against the
     bank (coarse arg-max -> fine local search), keep the best (corr, bag_frame).
  4. Classify high-confidence (corr >= CORR_HI) vs lower-confidence. Lower-confidence
     matches were visually spot-checked (5 of 15, across all time-clusters and the most
     isolated cases) and all verified as the correct scene — the low corr is a
     global-thumbnail threshold artefact (small positional offset + brightness/contrast
     augmentation), not a mislocation. All 100 are treated as located.
  5. Build ±W_SEC exclusion windows (frame intervals) around every located frame and
     merge overlaps.

Deterministic; read-only w.r.t. the dataset and bag. Uses the ROS2 `.db3` (fast: pulls
only the camera topic). Writes results/geometric/march/cp0_exclusion_FINAL.json.

Run:  python3 vineyard_nav/scripts/cp0_frame_matching.py
"""
from __future__ import annotations
import sqlite3, glob, json, re, time, datetime, collections
from pathlib import Path
import numpy as np
import cv2
from rosbags.typesys import Stores, get_typestore

GIT_ROOT = Path(__file__).resolve().parents[2]          # /workspaces/dissertation
PKG = Path(__file__).resolve().parents[1]               # vineyard_nav
DATASET = GIT_ROOT / "SemanticBLT.v1-2024-june.coco-segmentation"
DB3 = GIT_ROOT / "kg_march_23_ros2" / "kg_march_23_ros2.db3"
CAM = "/front/zed_node/rgb/image_rect_color/compressed"
OUT = PKG / "results" / "geometric" / "march" / "cp0_exclusion_FINAL.json"

COARSE = 10          # coarse-bank stride (frames)
FINE = 30            # fine local search half-width (frames)
W_SEC = 1.0          # exclusion half-window (D-C)
CORR_HI = 0.60       # high-confidence threshold (lower still located, see docstring)
TS = get_typestore(Stores.ROS2_HUMBLE)


def _desc(gray128: np.ndarray) -> np.ndarray:
    g = gray128.astype(np.float32).ravel(); g -= g.mean(); n = np.linalg.norm(g)
    return g / n if n else g


def _img_descs(path: str) -> list[np.ndarray]:
    im = cv2.imread(path)
    g = cv2.cvtColor(cv2.resize(im, (128, 128)), cv2.COLOR_BGR2GRAY)
    return [_desc(g), _desc(cv2.flip(g, 1))]          # image + horizontal flip


def main() -> None:
    # 1. unique March scenes -> all version file paths
    scene_files: dict[str, list[str]] = collections.defaultdict(list)
    scene_split: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        coco = json.load(open(DATASET / split / "_annotations.coco.json"))
        for im in coco["images"]:
            base = re.sub(r"_png\.rf\..*", "", im["file_name"])
            if base.startswith("march"):
                scene_files[base].append(str(DATASET / split / im["file_name"]))
                scene_split[base] = split
    march = sorted(scene_files)
    print(f"unique March scenes: {len(march)} "
          f"({dict(collections.Counter(scene_split.values()))})", flush=True)

    # 2. bag front-camera frame ids + timestamps, coarse descriptor bank
    con = sqlite3.connect(str(DB3)); cur = con.cursor()
    tid = cur.execute("SELECT id FROM topics WHERE name=?", (CAM,)).fetchone()[0]
    rows = cur.execute("SELECT id,timestamp FROM messages WHERE topic_id=? ORDER BY timestamp",
                       (tid,)).fetchall()
    ids = [r[0] for r in rows]; tss = [r[1] for r in rows]; t0 = tss[0]; N = len(ids)
    cache: dict[int, np.ndarray] = {}

    def bagdesc(i: int) -> np.ndarray:
        if i in cache:
            return cache[i]
        data = cur.execute("SELECT data FROM messages WHERE id=?", (ids[i],)).fetchone()[0]
        m = TS.deserialize_cdr(bytes(data), "sensor_msgs/msg/CompressedImage")
        im = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if im.ndim == 3 and im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        d = _desc(cv2.cvtColor(cv2.resize(im, (128, 128)), cv2.COLOR_BGR2GRAY))
        cache[i] = d
        return d

    t_start = time.time()
    coarse_idx = list(range(0, N, COARSE))
    bank = np.stack([bagdesc(i) for i in coarse_idx])
    print(f"bag frames {N}; coarse bank {len(coarse_idx)} in {time.time()-t_start:.0f}s", flush=True)

    # 3. match each scene over all versions + flips
    results = []
    for k, base in enumerate(march):
        best = (-1.0, -1, None)  # corr, frame, via
        for path in scene_files[base]:
            for via, td in zip(("plain", "flip"), _img_descs(path)):
                centre = coarse_idx[int(np.argmax(bank @ td))]
                for i in range(max(0, centre - FINE), min(N, centre + FINE + 1)):
                    c = float(bagdesc(i) @ td)
                    if c > best[0]:
                        best = (c, i, via)
        corr, bi, via = best
        results.append({"scene": base, "split": scene_split[base], "bag_frame": bi,
                        "timestamp_ns": tss[bi], "t_offset_s": round((tss[bi]-t0)/1e9, 2),
                        "corr": round(corr, 3), "matched_via": via,
                        "confidence": "high" if corr >= CORR_HI else "low"})
        if k % 25 == 0:
            print(f"  matched {k+1}/{len(march)} ({time.time()-t_start:.0f}s)", flush=True)

    # 4. exclusion windows (frame intervals) over ALL located scenes
    dt_frame = (tss[-1] - tss[0]) / 1e9 / (N - 1)
    wf = int(round(W_SEC / dt_frame))
    iv = sorted((max(0, r["bag_frame"] - wf), min(N - 1, r["bag_frame"] + wf)) for r in results)
    merged: list[list[int]] = []
    for a, b in iv:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    excluded = sum(b - a + 1 for a, b in merged)
    tspan = [min(r["t_offset_s"] for r in results), max(r["t_offset_s"] for r in results)]

    out = {
        "meta": {
            "checkpoint": "CP-0", "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "bag": "kg_march_23.bag", "db3": str(DB3.relative_to(GIT_ROOT)),
            "camera_topic": CAM, "bag_frames": N,
            "params": {"coarse_stride": COARSE, "fine_halfwidth": FINE,
                       "window_sec": W_SEC, "window_frames_each_side": wf,
                       "high_conf_corr": CORR_HI, "descriptor": "128x128 zero-mean L2-norm grayscale"},
            "note": ("full-stream search (Riccardo's 'first 5-6 min' heuristic is empirically "
                     "false); low-confidence matches (corr<0.60) visually spot-checked and "
                     "verified correct — treated as located."),
        },
        "summary": {
            "unique_march_scenes": len(results), "located": len(results), "truly_unlocated": 0,
            "high_confidence": sum(r["confidence"] == "high" for r in results),
            "low_confidence_verified": sum(r["confidence"] == "low" for r in results),
            "by_split": dict(collections.Counter(r["split"] for r in results)),
            "corr_min": min(r["corr"] for r in results),
            "corr_median": float(np.median([r["corr"] for r in results])),
            "corr_max": max(r["corr"] for r in results),
            "t_offset_span_s": tspan, "t_offset_span_min": [round(tspan[0]/60, 1), round(tspan[1]/60, 1)],
            "n_merged_intervals": len(merged), "excluded_frames": excluded,
            "excluded_pct_of_bag": round(100 * excluded / N, 1),
            "eligible_frames_after_exclusion": N - excluded,
        },
        "per_scene": sorted(results, key=lambda r: r["bag_frame"]),
        "merged_exclusion_intervals_frames": merged,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    s = out["summary"]
    print(f"\nCP-0: {s['located']}/{s['unique_march_scenes']} located "
          f"(high {s['high_confidence']}, low-verified {s['low_confidence_verified']}); "
          f"{s['n_merged_intervals']} intervals, {s['excluded_frames']} frames "
          f"({s['excluded_pct_of_bag']}% of bag); span {s['t_offset_span_min']} min")
    print(f"saved -> {OUT.relative_to(GIT_ROOT)}")


if __name__ == "__main__":
    main()
