#!/usr/bin/env python3
"""CP-1 frame-manifest builder (GEOMETRY_PIPELINE_SPEC.md §3; D033 passes; D040 whole-bag; D041).
Bag-parametrised.

Pairs every bag camera frame with its /robot_pose, flags contamination (CP-0
exclusion intervals), stationary (smoothed v < V_MIN) and headland (not in an in-row pass),
assigns each in-row frame a corridor and a PASS id (individual corridor traversal), and marks
the Delta_s = 1.5 m spatial-independence subsample. Under the whole-bag treatment (D040) there is
NO val/test split: `split` is a single canonical marker ("eligible" / "excluded") and the
subsample is a single greedy pass over ALL eligible frames (not per-split). The eligible set is the
evaluated whole-bag dataset (D041); pass-level structure (D033) is retained for the block-bootstrap.

Deterministic, read-only w.r.t. dataset/bag. Writes:
  results/geometric/{bag}/dataset_manifest.json    (all bag frames + flags/marker/subsample)
  results/geometric/{bag}/manifest_summary.json     (whole-bag counts per pass / per corridor)

Run:  python3 scripts/geometric/frame_manifest_build.py --bag april
"""
from __future__ import annotations
import sys, sqlite3, json, collections
from pathlib import Path
import numpy as np
from rosbags.typesys import Stores, get_typestore

GIT = Path(__file__).resolve().parents[3]; PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
from bag_config import parse_bag
CAM = "/front/zed_node/rgb/image_rect_color/compressed"
TS = get_typestore(Stores.ROS2_HUMBLE)

V_MIN, VY_INROW, PASS_MIN_Y, DS_SUB = 0.10, 0.30, 10.0, 1.5


def main() -> None:
    B = parse_bag()
    bag, DB3, CP0 = B["bag"], B["db3"], B["census"]
    OUT, SUMMARY = B["manifest"], B["manifest_summary"]
    if not DB3.exists():
        raise SystemExit(f"ROS2 bag not found: {DB3}\n"
                         f"Convert it first:  python3 scripts/geometric/convert_bag.py --bag {bag}")
    if not CP0.exists():
        raise SystemExit(f"CP-0 census not found: {CP0}\n"
                         f"Run it first:  python3 scripts/geometric/contamination_census.py --bag {bag}")

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
    exp = B["expected_passes"]
    if exp is not None:
        assert len(passes) == exp, f"[{bag}] expected {exp} in-row passes, got {len(passes)}"
    else:
        print(f"[{bag}] {len(passes)} in-row passes detected (no expected count configured for this bag)")

    # contamination (CP-0)
    contaminated = np.zeros(N, bool)
    for a, b in json.load(open(CP0))["merged_exclusion_intervals_frames"]:
        contaminated[a:b + 1] = True
    stationary = vs < V_MIN; headland = ~inrow
    eligible = inrow & ~stationary & ~contaminated

    # whole-bag marker (D040): single canonical "eligible" / "excluded" (no val/test split)
    split = np.where(eligible, "eligible", "excluded").astype(object)

    # Delta_s = 1.5 m subsample: single greedy pass over ALL eligible frames (whole-bag, D040)
    subsample = np.zeros(N, bool)
    last = None
    for idx in range(N):
        if eligible[idx] and (last is None or np.hypot(x[idx] - x[last], y[idx] - y[last]) >= DS_SUB):
            subsample[idx] = True; last = idx

    # summary: per-pass and per-corridor (whole-bag; eligible frames)
    per_pass = []
    for pid, (a, b, xm) in enumerate(passes):
        seg = slice(a, b)
        per_pass.append({"pass": pid, "corridor": int(corridor[a]),
                         "dir": "down" if y[b - 1] < y[a] else "up",
                         "t0_s": round(float(t[a]), 1), "t1_s": round(float(t[b - 1]), 1),
                         "eligible": int(eligible[seg].sum())})
    corr_frames = dict(sorted(collections.Counter(
        int(corridor[idx]) for idx in range(N) if eligible[idx]).items()))
    summ = {
        "raw_frames": N, "pose_pair_max_offset_ms": round(pair_off_ms, 1),
        "contamination_excluded": int(contaminated.sum()),
        "stationary": int(stationary.sum()), "headland": int(headland.sum()),
        "headland_or_stationary_excl_noncontam": int(((headland | stationary) & ~contaminated).sum()),
        "in_row": int(inrow.sum()), "eligible": int(eligible.sum()),
        "n_passes": len(passes), "n_corridors": len(centres),
        "corridor_centres_x": [round(c, 2) for c in centres],
        "eligible_corridor_frames": corr_frames,
        "max_corridor_pct": round(100 * max(corr_frames.values()) / sum(corr_frames.values()), 1),
        "subsample_1p5m": int(subsample.sum()),
        "path_length_m": round(float(ds.sum()), 1),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "meta": {"checkpoint": "CP-1", "bag": B["src_bag"].stem, "frames": N,
                 "eval_unit": "whole-bag eligible (D040)", "params": {"v_min": V_MIN, "vy_inrow": VY_INROW,
                 "pass_min_y_m": PASS_MIN_Y, "subsample_ds_m": DS_SUB}},
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
              f"eligible {p['eligible']:4d}")
    print(f"\nELIGIBLE {summ['eligible']} frames (whole-bag; cor {corr_frames}, max {summ['max_corridor_pct']}%), "
          f"Delta_s=1.5m subsample {summ['subsample_1p5m']}")
    print(f"saved {OUT.relative_to(GIT)} + {SUMMARY.relative_to(GIT)}")


if __name__ == "__main__":
    main()
