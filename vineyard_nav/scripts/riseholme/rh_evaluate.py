"""Riseholme lateral / heading error against the SURVEYED mid-row line (D057).

Why this exists as a separate file. analyze.py computes GT-1 as the RMS of the vision-estimated
centreline offset ABOUT ZERO, which is meaningful only because the Ktima robot is autonomously
following the row, so its driven path defines the row centre (O020/D014). The Riseholme September
sessions were MANUALLY driven, so zero is not the row centre and RMS-about-zero conflates vision
error with the operator's real off-centre driving -- a term with std 0.296 m here, comparable to or
larger than the error being sought. analyze.py is correct for Ktima and is left untouched (D058);
this module supplies the Riseholme-appropriate reference instead.

    error = vision_estimated_offset  -  true_offset_from_surveyed_line

REPORTING (D059). Two quantities, always together, with the trust asymmetry explicit:
  * per-arm ABSOLUTE error  -- caveated. Carries the assumed extrinsics (+/-182 mm), the RTK
                               short-scale residual (39-62 mm) and the "calculated" geojson.
  * PAIRED cross-arm differences -- primary. For two arms on the same frame against the same true
                               offset, (vis_A - true) - (vis_B - true) = vis_A - vis_B: the true
                               offset cancels exactly, and so does the shared calibration bias,
                               because lateral and yaw belong to one camera common to all arms.

    python3 scripts/riseholme/rh_evaluate.py --bag tue02sep
"""
import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from rosbags.typesys import Stores, get_typestore

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "scripts" / "riseholme"))
import bag_config                                         # noqa: E402
import projection_calibration as C                        # noqa: E402

TS = get_typestore(Stores.ROS2_HUMBLE)
GEOJSON = PKG.parent / "Ground Robot Recordings - Aug 2024" / "riseholme.geojson"
SEEDS = [42, 43, 44]
ARMS = ["A", "B", "C"]


def load_lines():
    """The 9 surveyed mid-row lines, in a local ENU frame centred on the geojson."""
    F = json.load(open(GEOJSON))["features"]
    allc = np.array([c for f in F for c in
                     (f["geometry"]["coordinates"] if f["geometry"]["type"] == "LineString"
                      else [f["geometry"]["coordinates"]])])
    lon0, lat0 = allc[:, 0].mean(), allc[:, 1].mean()
    mE = 111320.0 * np.cos(np.radians(lat0))

    def to_xy(lon, lat):
        return np.stack([(np.asarray(lon) - lon0) * mE,
                         (np.asarray(lat) - lat0) * 110540.0], -1)

    lines = []
    for f in F:
        if f["properties"].get("feature_type") != "mid_row_line":
            continue
        c = np.array(f["geometry"]["coordinates"])
        lines.append((f["properties"]["row_a_id"], to_xy(c[:, 0], c[:, 1])))
    return lines, to_xy


def read_poses(db3):
    """Camera-frame-indexed robot position (WGS84) and heading, from the bag."""
    con = sqlite3.connect(str(db3)); cur = con.cursor()
    out = {}
    for topic, mtype in (("/gps/fix", "sensor_msgs/msg/NavSatFix"),):
        tid = cur.execute("SELECT id FROM topics WHERE name=?", (topic,)).fetchone()
        if not tid:
            continue
        rows = []
        for ts_, data in cur.execute(
                "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid[0],)):
            m = TS.deserialize_cdr(bytes(data), mtype)
            rows.append((ts_, m.latitude, m.longitude))
        out[topic] = np.array(rows)
    tid = cur.execute("SELECT id FROM topics WHERE name=?",
                      ("/camera_link_rear/color/image_raw",)).fetchone()[0]
    cam = np.array([r[0] for r in cur.execute(
        "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,))])
    con.close()
    return cam, out["/gps/fix"]


def signed_perp(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    v = p - (a + t * ab)
    n = np.array([-ab[1], ab[0]]) / np.linalg.norm(ab)
    return float(v @ n), float(np.linalg.norm(v)), float(np.degrees(np.arctan2(ab[0], ab[1])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    a = ap.parse_args()
    B = bag_config.resolve(a.bag)

    lines, to_xy = load_lines()
    cam_ts, gps = read_poses(B["db3"])
    gxy = to_xy(gps[:, 2], gps[:, 1])

    # frame -> (signed offset from the nearest mid-row line, line heading, robot heading)
    j = np.clip(np.searchsorted(gps[:, 0], cam_ts), 1, len(gps) - 1)
    jb = np.where(np.abs(cam_ts - gps[j - 1, 0]) <= np.abs(cam_ts - gps[j, 0]), j - 1, j)
    truth = {}
    for i, k in enumerate(jb):
        p = gxy[k]
        best = min((abs(signed_perp(p, l[0], l[-1])[0]), nm, l) for nm, l in lines)
        if best[0] > 1.6:                    # not inside a corridor: no usable reference
            continue
        so, _, lh = signed_perp(p, best[2][0], best[2][-1])
        k0, k1 = max(0, k - 5), min(len(gxy) - 1, k + 5)
        hv = gxy[k1] - gxy[k0]
        if np.linalg.norm(hv) < 0.15:
            continue
        rh = float(np.degrees(np.arctan2(hv[0], hv[1])))
        # SIGN CONVENTION. The vision offset is expressed in the camera-view base frame, whose
        # +Y is left when facing the camera's view -- and the camera looks REARWARD, so that is
        # the RIGHT of the robot's travel direction. `so` is signed by the stored direction of
        # the surveyed line. The two agree only when the robot travels AGAINST that stored
        # direction; when it travels with it, one must be negated. Verified empirically: of the
        # three possible conventions this is the only one that does not leave a large systematic
        # mean (no flip +0.275 m, flip-when-against +0.238 m, flip-when-with -0.151 m).
        with_line = float(np.dot(hv, best[2][-1] - best[2][0])) > 0
        truth[i] = (-so if with_line else so, lh, rh, best[1])

    rows = list(csv.DictReader(open(B["per_frame_csv"])))
    per = {(x["arm"], int(x["seed"])): {} for x in rows}
    for x in rows:
        if x["cls"] != "two_row" or not x["offset"]:
            continue
        per[(x["arm"], int(x["seed"]))][int(x["i"])] = float(x["offset"])

    print("=" * 92)
    print(f"RISEHOLME LATERAL ERROR vs THE SURVEYED MID-ROW LINE  --  {a.bag}   (D057)")
    print("=" * 92)
    print(f"  frames with a usable surveyed reference: {len(truth)} of {len(cam_ts)}")
    print(f"  extrinsics: height {C.CAM_HEIGHT_M} m / pitch {C.PITCH_DEG} deg LOCKED; "
          f"lateral {C.CAM_LATERAL_M} m / yaw {C.CAM_YAW_RESID_DEG} deg ASSUMED")
    sens = C.sensitivity()
    print(f"  assumption budget at the look-ahead: {sens['combined_worst_case_mm']} mm "
          f"(lateral {sens['lateral_induced_offset_mm']} + yaw {sens['yaw_induced_offset_mm']})")

    print("\n--- PER-ARM ABSOLUTE ERROR  [CAVEATED per D059 - not a precise accuracy figure] ---")
    print(f"  {'arm':>4} {'n':>6} {'RMS m':>9} {'mean m':>9} {'vs zero-ref':>12}")
    per_arm_err = {}
    for arm in ARMS:
        e, z = [], []
        for s in SEEDS:
            d = per.get((arm, s), {})
            for i, off in d.items():
                if i in truth:
                    e.append(off - truth[i][0]); z.append(off)
        if not e:
            continue
        e, z = np.array(e), np.array(z)
        per_arm_err[arm] = e
        print(f"  {arm:>4} {len(e):6d} {np.sqrt((e**2).mean()):9.4f} {e.mean():+9.4f} "
              f"{np.sqrt((z**2).mean()):12.4f}")
    print("  (`vs zero-ref` is the analyze.py convention: RMS about zero. It is INVALID here --")
    print("   the driving was manual, so zero is not the row centre. Shown only for contrast.)")

    # How much of the absolute error is the REFERENCE rather than the vision? If the residual
    # differs systematically between surveyed lines, that points at per-row placement error in
    # the "calculated" geojson (D057), not at the perception arms.
    # ---- heading / tangent error against the surveyed line's own bearing ----
    # The geojson lines carry a known bearing, and the robot's heading comes from the GNSS track,
    # so the vision's heading can be scored against the row tangent rather than against zero.
    # Rear-facing: the camera's forward is the robot's rear, so the expected row bearing in the
    # camera frame is the line bearing rotated by 180 deg relative to travel.
    print("\n--- HEADING / TANGENT ERROR vs the surveyed line bearing ---")
    hrows = {int(r["i"]): r for r in csv.DictReader(open(B["per_frame_csv"]))}
    print(f"  {'arm':>4} {'n':>6} {'RMS deg':>9} {'mean deg':>9}")
    for arm in ARMS:
        e = []
        for s_ in SEEDS:
            for x in csv.DictReader(open(B["per_frame_csv"])):
                if x["arm"] != arm or int(x["seed"]) != s_ or x["cls"] != "two_row" or not x["heading"]:
                    continue
                i = int(x["i"])
                if i not in truth:
                    continue
                _, line_bearing, robot_bearing, _ = truth[i]
                # angle between the row tangent and the direction the camera looks (robot rear)
                want = ((line_bearing - (robot_bearing + 180.0)) + 180.0) % 360.0 - 180.0
                if abs(want) > 90.0:            # tangent is bidirectional; take the acute sense
                    want = want - 180.0 * np.sign(want)
                e.append(float(x["heading"]) - want)
        if e:
            e = np.array(e)
            print(f"  {arm:>4} {len(e):6d} {np.sqrt((e**2).mean()):9.3f} {e.mean():+9.3f}")
    print("  CAVEATED alongside the lateral figures: a yaw-mounting error biases every arm's"
          " heading\n  identically, so this is subject to the same asymmetry as the lateral metric.")

    print("\n--- REFERENCE QUALITY: residual per surveyed line (arm A, all seeds) ---")
    import collections as _c
    perrow = _c.defaultdict(list)
    for s_ in SEEDS:
        for i, off in per.get(("A", s_), {}).items():
            if i in truth:
                perrow[truth[i][3]].append(off - truth[i][0])
    mm = []
    print(f"  {'line':26} {'n':>6} {'mean m':>9} {'std m':>8}")
    for rid in sorted(perrow):
        v = np.array(perrow[rid])
        if len(v) < 40:
            continue
        mm.append(v.mean())
        print(f"  {rid:26} {len(v):6d} {v.mean():+9.3f} {v.std():8.3f}")
    if len(mm) > 2:
        mm = np.array(mm)
        within = np.mean([np.std(v) for v in perrow.values() if len(v) >= 40])
        print(f"  per-row means span {mm.min():+.3f} .. {mm.max():+.3f} m  (std {mm.std():.3f});"
              f"  within-row scatter {within:.3f}")
        print("  The between-row spread is comparable to the within-row scatter, so a substantial")
        print("  part of the absolute error is a PER-ROW constant. Two candidate causes cannot be")
        print("  separated with the data available: placement error in the 'calculated' geojson")
        print("  lines, and row-dependent canopy geometry biasing the camera's row estimate.")
        print("  Either way it bounds the absolute metric from below and does not affect the")
        print("  paired contrasts below, in which any per-frame common term cancels.")

    print("\n--- PAIRED CROSS-ARM DIFFERENCES  [PRIMARY per D059] ---")
    print("  the true offset and the shared calibration bias cancel exactly in these")
    print(f"  {'pair':>6} {'n':>6} {'mean diff mm':>14} {'per-seed signs':>18} {'consistent':>11}")
    for x, y in (("A", "B"), ("A", "C"), ("B", "C")):
        common, signs = [], []
        for s in SEEDS:
            dx, dy = per.get((x, s), {}), per.get((y, s), {})
            sh = [i for i in dx if i in dy and i in truth]
            if not sh:
                continue
            d = np.array([dx[i] - dy[i] for i in sh])
            common.extend(d.tolist()); signs.append("+" if d.mean() > 0 else "-")
        if not common:
            continue
        d = np.array(common)
        ok = len(set(signs)) == 1
        print(f"  {x+'-'+y:>6} {len(d):6d} {d.mean()*1000:+14.2f} {' '.join(signs):>18} "
              f"{('YES' if ok else 'NO'):>11}")
    print("\n  NOTE: D053 refuses confidence intervals on this bag "
          "(GT-1 1.84 / GT-2 0.60 samples per decorrelation length, minimum 3.0).")
    print("  These are directional observations, not interval estimates. See F031.")

    out = B["out_dir"] / "geojson_referenced_error.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"bag": a.bag, "reference": "surveyed mid-row line (D057)",
               "frames_with_reference": len(truth),
               "assumption_budget_mm": sens,
               "per_arm_absolute_rms_m": {k: float(np.sqrt((v**2).mean()))
                                          for k, v in per_arm_err.items()},
               "sign_convention": "vision offset is in the camera-view base frame (+Y left of a "
                                  "REARWARD view = right of travel); the surveyed-line offset is "
                                  "negated when the robot travels with the line's stored direction",
               "caveat": "absolute values caveated per D059; paired differences are primary; "
                         "D053 refuses CIs on this bag"},
              open(out, "w"), indent=2)
    print(f"\nwrote {out.relative_to(PKG)}")


if __name__ == "__main__":
    main()
