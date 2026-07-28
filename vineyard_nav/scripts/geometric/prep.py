#!/usr/bin/env python3
"""CP-0 + CP-1 preparation driver — consolidated (replaces two leaf scripts).

Runs the two per-bag PREP checkpoints of the geometric pipeline in sequence, replacing
contamination_census.py (CP-0) and frame_manifest_build.py (CP-1), whose logic now lives VERBATIM
in `contamination_census(B)` and `frame_manifest_build(B)` here.

  CP-0  contamination_census(B)  -> results/geometric/{bag}/contamination_census_exclusions.json
        Locates every unique SemanticBLT scene labelled from THIS bag's month by content
        frame-matching (prefix matcher), then the D048 gate (scene_attribution.py) ORB+RANSAC-scores
        the 90 unattributed `color_image_*` scenes against this bag: >=200 inliers -> present (folded
        into the exclusion set); 40-200 -> needs_review; <=40 -> absent. Emits the per-scene match
        table + the merged +/-w exclusion intervals and a `status` ("clear" | "needs_review").

  CP-1  frame_manifest_build(B)  -> results/geometric/{bag}/dataset_manifest.json
                                 -> results/geometric/{bag}/manifest_summary.json
        Pairs every camera frame with /robot_pose, flags contamination (CP-0 intervals) / stationary /
        headland, assigns corridor + PASS id to in-row frames, and marks the Delta_s=1.5 m subsample.
        Whole-bag treatment (D040): no val/test split.

BLOCKING RULE (D048): after CP-0 writes the census, if its `status == "needs_review"` (any
unattributed scene in the 40-200 review band) the manifest is NOT built — the run stops with the
review message. CP-1 runs only when the census is "clear". The check reads the in-memory census dict
returned by CP-0.

Deterministic, read-only w.r.t. dataset/bag; reads the ROS2 `.db3`. `--scope` is accepted for
symmetry with the other pipeline scripts but does not affect these outputs (census / manifest /
manifest_summary are per-bag, not per-scope).

Run:  python3 scripts/geometric/prep.py --bag march
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
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
from bag_config import resolve                            # noqa: E402
import scene_attribution as SA                            # noqa: E402  D048 unattributed-scene gate (shared module)

DATASET = GIT_ROOT / "SemanticBLT.v1-2024-june.coco-segmentation"
CAM = "/front/zed_node/rgb/image_rect_color/compressed"
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
    """CP-0. Writes B['census'] and returns the census dict (so main() can read `status`)."""
    bag, prefix, DB3, OUT = B["bag"], B["scene_prefix"], B["db3"], B["census"]
    if not DB3.exists():
        raise SystemExit(f"ROS2 bag not found: {DB3}\n"
                         f"Convert it first:  python3 scripts/geometric/convert_bag.py --bag {bag}")

    # 1. unique scenes labelled from THIS bag's month -> all version file paths
    scene_files: dict[str, list[str]] = collections.defaultdict(list)
    scene_split: dict[str, str] = {}
    if prefix:
        for split in ("train", "valid", "test"):
            coco = json.load(open(DATASET / split / "_annotations.coco.json"))
            for im in coco["images"]:
                base = re.sub(r"_png\.rf\..*", "", im["file_name"])
                if base.startswith(prefix):
                    scene_files[base].append(str(DATASET / split / im["file_name"]))
                    scene_split[base] = split
    scenes = sorted(scene_files)
    print(f"[{bag}] unique '{prefix}'-labelled scenes: {len(scenes)} "
          f"({dict(collections.Counter(scene_split.values()))})", flush=True)

    # 2. bag front-camera frame ids + timestamps, coarse descriptor bank.
    #    The bank is built for EVERY bag now (D048): it serves both the prefix matcher (step 3) and
    #    the unattributed-scene gate (step 4). No-prefix bags (june/july/september) build it too —
    #    the gate is exactly the check those bags need, so there is no early-return any more.
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

    # 3. prefix-scene matching — each scene over all versions + flips (only if this bag's month has
    #    labelled scenes). UNCHANGED from the single-strand CP-0, so prefix exclusions stay
    #    byte-identical to the pre-D048 census.
    prefix_results = []
    for k, base in enumerate(scenes):
        best = (-1.0, -1, None)  # corr, frame, via
        for path in scene_files[base]:
            for via, td in zip(("plain", "flip"), _img_descs(path)):
                centre = coarse_idx[int(np.argmax(bank @ td))]
                for i in range(max(0, centre - FINE), min(N, centre + FINE + 1)):
                    c = float(bagdesc(i) @ td)
                    if c > best[0]:
                        best = (c, i, via)
        corr, bi, via = best
        prefix_results.append({"scene": base, "split": scene_split[base], "bag_frame": bi,
                               "timestamp_ns": tss[bi], "t_offset_s": round((tss[bi]-t0)/1e9, 2),
                               "corr": round(corr, 3), "matched_via": via,
                               "confidence": "high" if corr >= CORR_HI else "low"})
        if k % 25 == 0:
            print(f"  matched {k+1}/{len(scenes)} ({time.time()-t_start:.0f}s)", flush=True)

    # 4. D048 gate — attribute the 90 unattributed `color_image_*` scenes to THIS bag (reuses the
    #    coarse bank for shortlisting). present (>=200) folded into the exclusion set below;
    #    needs_review (40-200) recorded and BLOCKS CP-1; absent (<=40) recorded only.
    gate_rows = SA.gate(bank, coarse_idx, ids, cur, SA.unattributed_scenes(), fine_half=FINE)
    SA.apply_confirmations(gate_rows, SA.load_confirmations(OUT), log=print)   # residual band -> human record
    present = sorted((r for r in gate_rows if r["verdict"] == "present"), key=lambda r: r["bag_frame"])
    needs_review = sorted((r for r in gate_rows if r["verdict"] == "needs_review"),
                          key=lambda r: -r["inliers"])
    present_results = [{"scene": r["scene"], "split": "unattributed", "bag_frame": r["bag_frame"],
                        "timestamp_ns": tss[r["bag_frame"]],
                        "t_offset_s": round((tss[r["bag_frame"]]-t0)/1e9, 2),
                        "corr": None, "matched_via": "orb_d048", "confidence": "d048_present",
                        "inliers": r["inliers"]} for r in present]

    # 5. exclusion windows (frame intervals) over ALL located scenes (prefix + D048-present)
    located = prefix_results + present_results
    dt_frame = (tss[-1] - tss[0]) / 1e9 / (N - 1)
    wf = int(round(W_SEC / dt_frame))
    iv = sorted((max(0, r["bag_frame"] - wf), min(N - 1, r["bag_frame"] + wf)) for r in located)
    merged: list[list[int]] = []
    for a, b in iv:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    excluded = sum(b - a + 1 for a, b in merged)

    # 6. assemble + write. `status` is the CP-1 gate: 'needs_review' blocks manifest build (D048).
    gate_counts = {v: sum(1 for r in gate_rows if r["verdict"] == v)
                   for v in ("present", "needs_review", "absent")}
    prefix_summary = {"located": 0}
    if prefix_results:
        pspan = [min(r["t_offset_s"] for r in prefix_results),
                 max(r["t_offset_s"] for r in prefix_results)]
        prefix_summary = {
            "unique_scenes": len(prefix_results), "located": len(prefix_results), "truly_unlocated": 0,
            "high_confidence": sum(r["confidence"] == "high" for r in prefix_results),
            "low_confidence_verified": sum(r["confidence"] == "low" for r in prefix_results),
            "by_split": dict(collections.Counter(r["split"] for r in prefix_results)),
            "corr_min": min(r["corr"] for r in prefix_results),
            "corr_median": float(np.median([r["corr"] for r in prefix_results])),
            "corr_max": max(r["corr"] for r in prefix_results),
            "t_offset_span_s": pspan,
            "t_offset_span_min": [round(pspan[0]/60, 1), round(pspan[1]/60, 1)],
        }
    out = {
        "meta": {
            "checkpoint": "CP-0", "generated": datetime.datetime.now().isoformat(timespec="seconds"),
            "bag": bag, "src_bag": B["src_bag"].name, "db3": str(DB3.relative_to(GIT_ROOT)),
            "scene_prefix": prefix, "camera_topic": CAM, "bag_frames": N,
            "params": {"coarse_stride": COARSE, "fine_halfwidth": FINE,
                       "window_sec": W_SEC, "window_frames_each_side": wf,
                       "high_conf_corr": CORR_HI, "descriptor": "128x128 zero-mean L2-norm grayscale",
                       "d048": {"absent_le": SA.T_ABSENT, "present_ge": SA.T_PRESENT,
                                "shortlist_k": SA.SHORTLIST_K, "orb_nfeatures": SA.ORB_N,
                                "match_res": SA.MATCH_RES, "lowe": SA.LOWE, "ransac_px": SA.RANSAC_PX}},
            "note": ("prefix scenes: full-stream search, low-confidence (corr<0.60) matches "
                     "visually spot-checked and verified correct. D048 gate (two-stage): the 90 "
                     "unattributed scenes scored by coarse ORB+RANSAC identity; needs_review (40-200) "
                     "auto-fine-verified (full-res +/-30 frames) then finalised by d048_confirmed.json; "
                     "present (>=200) folded into exclusions, absent (<=40) recorded, any unconfirmed "
                     "residual blocks CP-1."),
        },
        "status": ("needs_review" if needs_review else "clear"),
        "summary": {
            "prefix": prefix_summary,
            "d048_gate": {"scored": len(gate_rows), **gate_counts,
                          "max_inliers": max((r["inliers"] for r in gate_rows), default=0)},
            "n_merged_intervals": len(merged), "excluded_frames": excluded,
            "excluded_pct_of_bag": round(100 * excluded / N, 1),
            "eligible_frames_after_exclusion": N - excluded,
        },
        "per_scene": sorted(located, key=lambda r: r["bag_frame"]),
        "d048_gate": {
            "present": sorted(present, key=lambda r: -r["inliers"]),
            "needs_review": needs_review,
            "absent_max_inliers": max((r["inliers"] for r in gate_rows if r["verdict"] == "absent"),
                                      default=0),
        },
        "merged_exclusion_intervals_frames": merged,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    s = out["summary"]
    print(f"\nCP-0 [{bag}]: prefix {prefix_summary['located']} located; "
          f"D048 {gate_counts['present']} present / {gate_counts['needs_review']} needs_review / "
          f"{gate_counts['absent']} absent; {s['n_merged_intervals']} intervals, "
          f"{s['excluded_frames']} frames ({s['excluded_pct_of_bag']}% of bag)")
    if needs_review:
        print(f"  ** {len(needs_review)} scene(s) in the 40-200 review band -> CP-1 BLOCKED for "
              f"{bag} until reviewed (D048). See census 'd048_gate.needs_review'.")
    print(f"saved -> {OUT.relative_to(GIT_ROOT)}")
    return out


# ================================================================================================
# CP-1 — frame-manifest builder (GEOMETRY_PIPELINE_SPEC.md §3; D033 passes; D040 whole-bag; D041)
# ================================================================================================
def frame_manifest_build(B) -> None:
    """CP-1. Writes B['manifest'] (all bag frames + flags/marker/subsample) and B['manifest_summary']
    (whole-bag counts per pass / per corridor). Reads the CP-0 census for the exclusion intervals."""
    bag, DB3, CP0 = B["bag"], B["db3"], B["census"]
    OUT, SUMMARY = B["manifest"], B["manifest_summary"]
    if not DB3.exists():
        raise SystemExit(f"ROS2 bag not found: {DB3}\n"
                         f"Convert it first:  python3 scripts/geometric/convert_bag.py --bag {bag}")
    if not CP0.exists():
        raise SystemExit(f"CP-0 census not found: {CP0}\n"
                         f"Run it first:  python3 scripts/geometric/prep.py --bag {bag}")

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
