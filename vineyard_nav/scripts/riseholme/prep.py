"""Riseholme CP-0 + CP-1.

CP-1 (frame_manifest_build) is UNCHANGED from scripts/geometric/prep.py -- same pose pairing,
same smoothing windows, same V_MIN / VY_INROW / PASS_MIN_Y / DS_SUB, same corridor assignment and
same Delta_s subsample. Only the camera topic differs, and CP-0 is vacuous because SemanticBLT
holds no Riseholme imagery (verified -- see RISEHOLME.md 12.5 item 7).

The pass detector assumes row traverses dominate the map-frame Y axis. That holds here: Riseholme
rows run at 171.5 deg, so a 16.2 m traverse gives |dy| ~ 16.0 m, clearing PASS_MIN_Y = 10.0 m.

  python3 scripts/riseholme/prep.py --bag part2
"""
from __future__ import annotations
import sys, sqlite3, json, re, time, datetime, collections
import argparse
from pathlib import Path
import numpy as np
import cv2
from rosbags.typesys import Stores, get_typestore

GIT_ROOT = Path(__file__).resolve().parents[3]          # /workspaces/dissertation
GIT = GIT_ROOT                                           # alias: the CP-1 body refers to this name
PKG = Path(__file__).resolve().parents[2]               # vineyard_nav
sys.path.insert(0, str(PKG / "scripts" / "riseholme"))
from bag_config import resolve                            # noqa: E402
# scene_attribution is NOT imported here: CP-0 is vacuous for Riseholme (no SemanticBLT
# imagery from this site), so the D048 unattributed-scene gate has nothing to match against.

DATASET = GIT_ROOT / "SemanticBLT.v1-2024-june.coco-segmentation"
CAM = "/camera_link_rear/color/image_raw"   # RH publishes raw Image, not CompressedImage
TS = get_typestore(Stores.ROS2_HUMBLE)

# CP-0 (contamination census) params
COARSE = 10          # coarse-bank stride (frames)
FINE = 30            # fine local search half-width (frames)
W_SEC = 1.0          # exclusion half-window (D-C)
CORR_HI = 0.60       # high-confidence threshold (lower still located, see contamination_census docstring)

# CP-1 (frame manifest) params
V_MIN, VY_INROW, PASS_MIN_Y, DS_SUB = 0.10, 0.30, 10.0, 1.5


# ================================================================================================
# CP-0 — contamination census (GEOMETRY_PIPELINE_SPEC.md §2, §9)
# ================================================================================================
# Method (robust to Roboflow augmentation of train scenes):
#   1. Enumerate unique month scenes across train+val+test (strip `_png.rf.<hash>`; augmented copies
#      collapse to one scene). One scene -> one bag frame.
#   2. Build a coarse descriptor bank over the FULL bag front-camera stream (every COARSE-th frame;
#      128x128 zero-mean unit-norm grayscale thumbnail).
#   3. For each scene, match ALL its versions + a horizontal-flip fallback against the bank (coarse
#      arg-max -> fine local search), keep the best (corr, bag_frame).
#   4. Classify high-confidence (corr >= CORR_HI) vs lower-confidence (spot-checked, all verified).
#   5. Build +/-W_SEC exclusion windows (frame intervals) around every located frame and merge.
#   The D048 gate then scores the 90 unattributed scenes; present folded in, needs_review blocks CP-1.
def _desc(gray128: np.ndarray) -> np.ndarray:
    g = gray128.astype(np.float32).ravel(); g -= g.mean(); n = np.linalg.norm(g)
    return g / n if n else g


def _img_descs(path: str) -> list[np.ndarray]:
    im = cv2.imread(path)
    g = cv2.cvtColor(cv2.resize(im, (128, 128)), cv2.COLOR_BGR2GRAY)
    return [_desc(g), _desc(cv2.flip(g, 1))]          # image + horizontal flip


def contamination_census(B) -> dict:
    """CP-0 for Riseholme: VACUOUS BY CONSTRUCTION, and recorded as such rather than skipped.

    SemanticBLT contains no Riseholme imagery. BLT itself covers both sites (Ktima 2022, Riseholme
    2023), so this was checked rather than assumed: the 405 month-less SemanticBLT images resolve to
    exactly 90 source scenes -- matching the 90 D048 unattributed scenes -- and rendering them shows
    Mediterranean stone buildings, arid ground and red row-end roses, matching the Ktima july2023
    frames, with none of Riseholme's glasshouse or water tank. See docs/RISEHOLME.md section 12.5,
    verification item 7.

    Riseholme is therefore genuinely out-of-distribution for the perception models, with no training
    contamination. An empty exclusion list is written so downstream stages read the same structure
    they read for Ktima."""
    # schema matches scripts/geometric/prep.py exactly, so CP-1 reads it unchanged
    out = {
        "meta": {"bag": B["bag"], "scene_prefix": B["scene_prefix"],
                 "status": "vacuous_no_scenes_from_site",
                 "note": "SemanticBLT contains no Riseholme imagery; site is out-of-distribution. "
                         "Verified by inspection of all 90 unattributed scenes "
                         "(RISEHOLME.md 12.5 item 7), not assumed."},
        "summary": {"scenes_located": 0, "scenes_total": 0,
                    "d048_unattributed_total": 90, "d048_present": 0,
                    "d048_needs_review": 0, "d048_absent": 90,
                    "excluded_frames": 0, "excluded_pct": 0.0},
        "per_scene": [],
        "merged_exclusion_intervals_frames": [],
    }
    B["census"].parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(B["census"], "w"), indent=2)
    print(f"CP-0 [{B['bag']}]: vacuous - SemanticBLT has no scenes from this site "
          f"(verified, not assumed). 0 frames excluded.")
    return out


def frame_manifest_build(B) -> None:
    """CP-1. Writes B['manifest'] (all bag frames + flags/marker/subsample) and B['manifest_summary']
    (whole-bag counts per pass / per corridor). Reads the CP-0 census for the exclusion intervals."""
    bag, DB3, CP0 = B["bag"], B["db3"], B["census"]
    OUT, SUMMARY = B["manifest"], B["manifest_summary"]
    if not DB3.exists():
        raise SystemExit(f"ROS2 bag not found: {DB3}\n"
                         f"Convert it first:  rosbags-convert --src <bag> --dst {bag}_ros2")
    if not CP0.exists():
        raise SystemExit(f"CP-0 census not found: {CP0}\n"
                         f"Run it first:  python3 scripts/riseholme/prep.py --bag {bag}")

    census = json.load(open(CP0))
    # D048 needs_review is enforced by main() before this function is called (it inspects the census
    # `status` in memory and stops the run without building the manifest).

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
    for a, b in census["merged_exclusion_intervals_frames"]:
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


# ================================================================================================
# CLI — CP-0 then CP-1, with the D048 needs_review block between them
# ================================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="CP-0 contamination census + CP-1 frame manifest (replaces 2 scripts).")
    ap.add_argument("--bag", default="march", help="bag name (default: march)")
    ap.add_argument("--scope", default="eligible", choices=["eligible", "non_in_row"],
                    help="accepted for symmetry with the pipeline; does not affect these per-bag outputs")
    a = ap.parse_args()
    B = resolve(a.bag, a.scope)

    census = contamination_census(B)

    # D048 blocking rule: a census in the 40-200 review band stops the run — do NOT build the manifest.
    if census.get("status") == "needs_review":
        nr = census.get("d048_gate", {}).get("needs_review", [])
        listed = ", ".join(f"{r['scene']}({r['inliers']})" for r in nr[:8]) + (" ..." if len(nr) > 8 else "")
        raise SystemExit(
            f"CP-1 [{B['bag']}] is BLOCKED (D048): {len(nr)} unattributed scene(s) scored in the 40-200 "
            f"review band and must be visually confirmed before evaluation — manifest NOT built.\n"
            f"  Review 'd048_gate.needs_review' in {B['census'].relative_to(GIT_ROOT)}\n"
            f"  Confirm each present (add to the exclusion set) or absent (clear it), then re-run CP-0/CP-1:\n"
            f"    python3 scripts/geometric/prep.py --bag {B['bag']}\n"
            f"  Scenes: {listed}")

    frame_manifest_build(B)


if __name__ == "__main__":
    main()
