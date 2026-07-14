#!/usr/bin/env python3
"""CP-1 manifest builder + pass-level val/test split (GEOMETRY_PIPELINE_SPEC.md §3, D-D; D033).

Pairs every kg_march_23 bag camera frame with its /robot_pose, flags contamination
(CP-0 exclusion intervals), stationary (smoothed v < V_MIN) and headland (not in an in-row
pass), assigns each in-row frame a corridor and a PASS id (individual corridor traversal),
and splits the 11 passes val/test at the PASS level (D033, supersedes the earlier
corridor-level split). Marks the Delta_s = 1.5 m independence subsample per split.

Deterministic, read-only w.r.t. dataset/bag. Writes:
  results/geometric/march/dataset_manifest.json       (all 16,656 frames + flags/split/subsample)
  results/geometric/march/val_test_split_summary.json  (counts per pass / per split x corridor)

Run:  python3 vineyard_nav/scripts/geometric/dataset_split.py
"""
from __future__ import annotations
import sqlite3, json, itertools, collections
from pathlib import Path
import numpy as np
from rosbags.typesys import Stores, get_typestore

GIT = Path(__file__).resolve().parents[3]; PKG = Path(__file__).resolve().parents[2]
DB3 = GIT / "kg_march_23_ros2" / "kg_march_23_ros2.db3"
CP0 = PKG / "results/geometric/march/contamination_census_exclusions.json"
OUT = PKG / "results/geometric/march/dataset_manifest.json"
SUMMARY = PKG / "results/geometric/march/val_test_split_summary.json"
CAM = "/front/zed_node/rgb/image_rect_color/compressed"
TS = get_typestore(Stores.ROS2_HUMBLE)

V_MIN, VY_INROW, PASS_MIN_Y, DS_SUB = 0.10, 0.30, 10.0, 1.5
# Approved pass-level assignment (D033); passes are numbered in time order (p0 = first).
VAL_PASSES = [2, 4, 5, 6, 7, 8, 10]
TEST_PASSES = [0, 1, 3, 9]


def main() -> None:
    con = sqlite3.connect(str(DB3)); cur = con.cursor()
    tid = cur.execute("SELECT id FROM topics WHERE name=?", (CAM,)).fetchone()[0]
    cam = np.array([r[0] for r in cur.execute(
        "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,))])
    N = len(cam)
    # poses
    ptid = cur.execute("SELECT id FROM topics WHERE name='/robot_pose'").fetchone()[0]
    pts, px, py = [], [], []
    for ts_, data in cur.execute("SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp", (ptid,)):
        m = TS.deserialize_cdr(bytes(data), "geometry_msgs/msg/Pose")
        pts.append(ts_); px.append(m.position.x); py.append(m.position.y)
    pts = np.array(pts); px = np.array(px); py = np.array(py)
    j = np.clip(np.searchsorted(pts, cam), 1, len(pts) - 1)
    jbest = np.where(np.abs(cam - pts[j - 1]) <= np.abs(cam - pts[j]), j - 1, j)
    x, y = px[jbest], py[jbest]; t = (cam - cam[0]) / 1e9
    pair_off_ms = float(np.abs(cam - pts[jbest]).max() / 1e6)

    dt = np.diff(t, prepend=t[0] - 1 / 14.77)
    ds = np.hypot(np.diff(x, prepend=x[0]), np.diff(y, prepend=y[0]))
    vs = np.convolve(ds / np.maximum(dt, 1e-6), np.ones(15) / 15, mode="same")
    vy = np.convolve(np.gradient(y, t), np.ones(15) / 15, mode="same")

    # in-row passes -> corridor + pass id (time order)
    inrow = np.zeros(N, bool); corridor = np.full(N, -1, int); pass_id = np.full(N, -1, int)
    passes = []; mask = np.abs(vy) > VY_INROW; i = 0
    while i < N:
        if mask[i]:
            k = i
            while k < N and mask[k]:
                k += 1
            if abs(y[k - 1] - y[i]) > PASS_MIN_Y:
                inrow[i:k] = True
                passes.append((i, k, float(np.median(x[i:k]))))
            i = k
        else:
            i += 1
    xs = sorted(p[2] for p in passes); cors = []
    for xm in xs:
        if cors and abs(xm - cors[-1][-1]) < 1.2:
            cors[-1].append(xm)
        else:
            cors.append([xm])
    centres = [float(np.mean(c)) for c in cors]
    for pid, (a, b, xm) in enumerate(passes):
        corridor[a:b] = int(np.argmin([abs(xm - c) for c in centres]))
        pass_id[a:b] = pid
    assert len(passes) == 11, f"expected 11 passes, got {len(passes)}"

    # contamination (CP-0)
    contaminated = np.zeros(N, bool)
    for a, b in json.load(open(CP0))["merged_exclusion_intervals_frames"]:
        contaminated[a:b + 1] = True
    stationary = vs < V_MIN; headland = ~inrow
    eligible = inrow & ~stationary & ~contaminated

    # pass-level split (D033)
    assert set(VAL_PASSES) | set(TEST_PASSES) == set(range(11)), "pass assignment must cover all 11"
    split = np.array(["excluded"] * N, dtype=object)
    for idx in range(N):
        if eligible[idx]:
            split[idx] = "val" if pass_id[idx] in VAL_PASSES else "test"

    # Delta_s = 1.5 m subsample per split (greedy over time order)
    subsample = np.zeros(N, bool)
    for sp in ("val", "test"):
        last = None
        for idx in range(N):
            if split[idx] == sp and (last is None or np.hypot(x[idx] - x[last], y[idx] - y[last]) >= DS_SUB):
                subsample[idx] = True; last = idx

    # summary: per-pass and per-split x corridor
    per_pass = []
    for pid, (a, b, xm) in enumerate(passes):
        seg = slice(a, b)
        per_pass.append({"pass": pid, "corridor": int(corridor[a]),
                         "dir": "down" if y[b - 1] < y[a] else "up",
                         "t0_s": round(float(t[a]), 1), "t1_s": round(float(t[b - 1]), 1),
                         "eligible": int(eligible[seg].sum()),
                         "split": "val" if pid in VAL_PASSES else "test"})
    def split_corr(sp):
        d = collections.Counter()
        for idx in range(N):
            if split[idx] == sp:
                d[int(corridor[idx])] += 1
        return dict(sorted(d.items()))
    vc, tc = split_corr("val"), split_corr("test")
    summ = {
        "raw_frames": N, "pose_pair_max_offset_ms": round(pair_off_ms, 1),
        "contamination_excluded": int(contaminated.sum()),
        "stationary": int(stationary.sum()), "headland": int(headland.sum()),
        "headland_or_stationary_excl_noncontam": int(((headland | stationary) & ~contaminated).sum()),
        "in_row": int(inrow.sum()), "eligible": int(eligible.sum()),
        "n_passes": len(passes), "n_corridors": len(centres),
        "corridor_centres_x": [round(c, 2) for c in centres],
        "val_passes": VAL_PASSES, "test_passes": TEST_PASSES,
        "val_frames": int((split == "val").sum()), "test_frames": int((split == "test").sum()),
        "val_corridor_frames": vc, "test_corridor_frames": tc,
        "val_max_corridor_pct": round(100 * max(vc.values()) / sum(vc.values()), 1),
        "test_max_corridor_pct": round(100 * max(tc.values()) / sum(tc.values()), 1),
        "val_subsample_1p5m": int(((split == "val") & subsample).sum()),
        "test_subsample_1p5m": int(((split == "test") & subsample).sum()),
        "path_length_m": round(float(ds.sum()), 1),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "meta": {"checkpoint": "CP-1", "bag": "kg_march_23.bag", "frames": N,
                 "split_unit": "pass (D033)", "params": {"v_min": V_MIN, "vy_inrow": VY_INROW,
                 "pass_min_y_m": PASS_MIN_Y, "subsample_ds_m": DS_SUB,
                 "val_passes": VAL_PASSES, "test_passes": TEST_PASSES}},
        "summary": summ,
        "frames": [{"i": idx, "timestamp_ns": int(cam[idx]), "t_offset_s": round(float(t[idx]), 3),
                    "x": round(float(x[idx]), 3), "y": round(float(y[idx]), 3),
                    "speed": round(float(vs[idx]), 3), "corridor": int(corridor[idx]),
                    "pass_id": int(pass_id[idx]), "contaminated": bool(contaminated[idx]),
                    "stationary": bool(stationary[idx]), "headland": bool(headland[idx]),
                    "eligible": bool(eligible[idx]), "split": split[idx],
                    "subsample_1p5m": bool(subsample[idx])} for idx in range(N)],
    }, indent=2))
    SUMMARY.write_text(json.dumps({"summary": summ, "per_pass": per_pass}, indent=2))

    print(f"frames {N} | pose-pair max offset {pair_off_ms:.1f} ms | passes {len(passes)} "
          f"| corridors {len(centres)} at x={[round(c,1) for c in centres]}")
    for p in per_pass:
        print(f"  p{p['pass']}: cor{p['corridor']} {p['dir']:4s} t{p['t0_s']:5.0f}-{p['t1_s']:5.0f}s "
              f"eligible {p['eligible']:4d} -> {p['split']}")
    print(f"\nVAL {summ['val_frames']} frames (cor {vc}, max {summ['val_max_corridor_pct']}%), "
          f"subsample {summ['val_subsample_1p5m']}")
    print(f"TEST {summ['test_frames']} frames (cor {tc}, max {summ['test_max_corridor_pct']}%), "
          f"subsample {summ['test_subsample_1p5m']}")
    print(f"saved {OUT.relative_to(GIT)} + {SUMMARY.relative_to(GIT)}")


if __name__ == "__main__":
    main()
