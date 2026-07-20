"""D042 native-twist state gate — CP-P1 re-derivation + validation (PID_PIPELINE_SPEC.md §3).

Re-derives the F022 runtime state gate on NATIVE bag twist (/odometry/base_raw.twist), cross-
checked against /imu/data.angular_velocity.z, replacing F022's pose-finite-difference signal
(D042). Fits the threshold on the native signal (F022's do NOT carry over — D042 caveat) and
re-validates non-in-row rejection / in-row false-positive rates the SAME way F022 was
(scripts/geometric/mitigation_analysis.py), producing F026 alongside F022 for direct comparison.
Bag-parametrised (D042 native gate applies to April+ bags too).

  python3 scripts/control/state_gate_native.py --bag march
    -> results/geometric/{bag}/final/mitigation_evaluation/state_gate_native.json   (additive sibling
       to mitigation_analysis.json — the F022 artefact is NOT modified)

LOCKED GATE (D042 frame correction, 20 Jul 2026; F026): a SINGLE native forward-speed predicate
`v_x > V_MIN`. The original PID_PIPELINE_SPEC.md §3 mapping ("native |v_y| replaces the finite-
difference v_y") was a WORLD-frame-vs-BODY-frame error: F022's world-frame predicates (speed,
along-row v_y, heading-rate) collapse in the base_link body frame to forward v_x — the body
twist.linear.y is sideways slip (~0.05 m/s in-row), NOT along-row velocity (a literal |v_y| > 0.30
retains only ~1.5% of in-row frames). The turn predicate is DROPPED: on the native signal it adds
zero marginal non-in-row rejection and only in-row false positives (this script quantifies both, as
the evidence for dropping it). `fit_forward_floor` / `native_gate` are the locked-gate primitives,
imported by the CP-2 command generator (D043/P-5a — a state-gate rejection triggers hold-last).
"""
import sys
import json
import argparse
import collections
import bisect
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
from bag_config import resolve                       # per-bag path bundle (reused, not duplicated)

ODOM = "/odometry/base_raw"                          # nav_msgs/Odometry (body twist in base_link)
IMU = "/imu/data"                                    # sensor_msgs/Imu (gyro; angular_velocity.z)


# ----------------------------------------------------------------------------------------------
# Native-signal access + LOCKED gate primitives (reused by CP-P2)
# ----------------------------------------------------------------------------------------------
def load_native_signals(db3, frames):
    """Read native body twist + IMU yaw-rate for every manifest frame, joined by timestamp_ns.
    The bag is frame-synchronised (every topic carries one message per camera frame at an
    identical timestamp — verified), so the join is exact. Returns v_x, v_y, yaw_odom, yaw_imu
    (arrays aligned to `frames` order), all SI (m/s, rad/s)."""
    import sqlite3
    from rosbags.typesys import Stores, get_typestore
    TS = get_typestore(Stores.ROS2_HUMBLE)
    con = sqlite3.connect(str(db3)); cur = con.cursor()

    def read(topic, msgtype, extract):
        tid = cur.execute("SELECT id FROM topics WHERE name=?", (topic,)).fetchone()[0]
        return {ts_: extract(TS.deserialize_cdr(bytes(data), msgtype)) for ts_, data in
                cur.execute("SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,))}

    od = read(ODOM, "nav_msgs/msg/Odometry",
              lambda m: (m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.angular.z))
    im = read(IMU, "sensor_msgs/msg/Imu", lambda m: m.angular_velocity.z)
    ts = [f["timestamp_ns"] for f in frames]
    if any(t not in od or t not in im for t in ts):
        raise SystemExit("some frames have no co-timestamped odom/imu message (join broken)")
    vx = np.array([od[t][0] for t in ts]); vy = np.array([od[t][1] for t in ts])
    yaw_odom = np.array([od[t][2] for t in ts]); yaw_imu = np.array([im[t] for t in ts])
    con.close()
    return vx, vy, yaw_odom, yaw_imu


def fit_forward_floor(vx, elig):
    """Locked native gate threshold: forward-speed floor = in-row p1 of v_x (retain 99% of in-row,
    the F022 in-row-percentile construction applied to the native forward velocity)."""
    return float(np.percentile(vx[elig], 1))


def native_gate(vx, v_min):
    """LOCKED native state gate (D042, corrected per F026): a single forward-speed predicate.
    Returns a per-frame keep mask (True = in a row-following state). Causal — each frame uses only
    its own instantaneous twist (no centred smoothing), so it is deployable as-is (contrast F022's
    offline 15-sample centred, i.e. non-causal, finite-difference)."""
    return vx > v_min


# ----------------------------------------------------------------------------------------------
# Validation (F022-parallel — mirrors mitigation_analysis.py so the two are directly comparable)
# ----------------------------------------------------------------------------------------------
def categorise_non_in_row(man):
    """stationary / turn / transition, identical construction to mitigation_analysis.py."""
    fr = {f["i"]: f for f in man["frames"]}
    elig_idx = sorted(i for i, f in fr.items() if f["eligible"])
    elig_corr = {i: fr[i]["corridor"] for i in elig_idx}

    def category(i):
        if fr[i]["stationary"]:
            return "stationary"
        p = bisect.bisect_left(elig_idx, i)
        bc = elig_corr[elig_idx[p - 1]] if p > 0 else None
        ac = elig_corr[elig_idx[p]] if p < len(elig_idx) else None
        return ("turn" if bc == ac else "transition") if (bc is not None and ac is not None) else "transition"

    non = [f["i"] for f in man["frames"] if f["headland"] and not f["contaminated"]]
    return {i: category(i) for i in non}


def load_two_row(csv):
    """(arm, frame_i) for every two_row OUTPUT row (frame x model), as mitigation_analysis.py."""
    out = []
    for ln in Path(csv).read_text().splitlines()[1:]:
        a, s, i, cls, o, h, *_ = ln.split(",")
        if cls == "two_row" and o and h:
            out.append((a, int(i)))
    return out


def rates(rows, reject, idx_of, cat=None):
    """Per-arm (+ per-category if cat given) rejection/FP over two_row output rows."""
    per = collections.defaultdict(lambda: [0, 0]); pc = collections.defaultdict(lambda: [0, 0])
    for a, i in rows:
        r = int(reject[idx_of[i]])
        per[a][0] += 1; per[a][1] += r
        if cat is not None:
            c = cat[i]; pc[(c, a)][0] += 1; pc[(c, a)][1] += r
    pct = lambda d, k: round(100 * d[k][1] / d[k][0], 1) if d[k][0] else None
    out = {"per_arm": {a: {"n": per[a][0], "reject_%": pct(per, a)} for a in "ABC"}}
    if cat is not None:
        out["per_category"] = {c: {a: pct(pc, (c, a)) for a in "ABC"}
                               for c in ("stationary", "turn", "transition")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="march")
    bag = ap.parse_args().bag
    B_in = resolve(bag, "eligible"); B_non = resolve(bag, "non_in_row")
    man = json.load(open(B_in["manifest"]))
    frames = man["frames"]
    OUT = B_in["out_dir"].parent / "mitigation_evaluation" / "state_gate_native.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)

    vx, vy, yaw_odom, yaw_imu = load_native_signals(B_in["db3"], frames)
    elig = np.array([f["eligible"] for f in frames])
    nonrow_mask = np.array([f["headland"] and not f["contaminated"] for f in frames])
    idx_of = {f["i"]: k for k, f in enumerate(frames)}
    cat = categorise_non_in_row(man)
    in_two = load_two_row(B_in["per_frame_csv"])
    non_two = load_two_row(B_non["per_frame_csv"])

    # --- LOCKED gate: single forward-speed predicate (D042 correction, F026) --------------------
    v_min = fit_forward_floor(vx, elig)
    reject = ~native_gate(vx, v_min)
    locked = {"gate": "v_x > V_MIN", "V_MIN_ms": round(v_min, 4),
              "non_in_row_rejection": rates(non_two, reject, idx_of, cat),
              "in_row_FP": rates(in_two, reject, idx_of)}

    # --- turn-predicate decomposition: EVIDENCE that the dropped turn predicate is inactive -----
    # For each candidate yaw source, quantify what adding `& (|yaw| < in-row p99)` would do on top
    # of the forward-speed floor: extra non-in-row rejections (want >0 to justify keeping) and extra
    # in-row false positives (the cost). Both measured on two_row OUTPUT rows and on raw frames.
    def turn_addendum(yaw, label):
        hr = float(np.percentile(np.abs(yaw)[elig], 99))
        turn_only = (vx > v_min) & (np.abs(yaw) >= hr)          # rejected by turn pred, NOT by v_x
        rej2 = reject | turn_only                               # 2-predicate reject mask
        fp2 = rates(in_two, rej2, idx_of)["per_arm"]
        rj2 = rates(non_two, rej2, idx_of)["per_arm"]
        return {"source": label, "HR_THRESH_deg_s": round(float(np.degrees(hr)), 3),
                "added_nonrow_frame_rejections": int((turn_only & nonrow_mask).sum()),
                "added_inrow_frame_FP": int((turn_only & elig).sum()),
                "two_predicate_non_in_row_rejection_%": {a: rj2[a]["reject_%"] for a in "ABC"},
                "two_predicate_in_row_FP_%": {a: fp2[a]["reject_%"] for a in "ABC"}}
    turn_decomp = {
        "note": ("Adding a turn predicate `& (|yaw| < in-row p99)` on top of the forward-speed floor "
                 "adds ZERO non-in-row rejection (headland pivots already fail the v_x floor) and only "
                 "in-row false positives -> the turn predicate is dropped from the locked gate."),
        "odom_yaw": turn_addendum(yaw_odom, "/odometry/base_raw.twist.angular.z"),
        "imu_yaw": turn_addendum(yaw_imu, "/imu/data.angular_velocity.z")}

    # --- odom-vs-IMU yaw-rate cross-check (F017-style; here the sensors DISAGREE) ---------------
    x = np.array([f["x"] for f in frames]); y = np.array([f["y"] for f in frames]); t = np.array([f["t_offset_s"] for f in frames])
    moving = np.hypot(np.gradient(x, t), np.gradient(y, t)) > 0.1
    d = np.degrees(np.abs(yaw_odom - yaw_imu))
    crosscheck = {
        "corr_moving": round(float(np.corrcoef(yaw_odom[moving], yaw_imu[moving])[0, 1]), 3),
        "abs_corr_moving": round(float(np.corrcoef(np.abs(yaw_odom[moving]), np.abs(yaw_imu[moving]))[0, 1]), 3),
        "mean_abs_diff_deg_s": round(float(d[moving].mean()), 3),
        "max_abs_diff_deg_s": round(float(d[moving].max()), 2),
        "odom_inrow_p99_deg_s": round(float(np.degrees(np.percentile(np.abs(yaw_odom[elig]), 99))), 2),
        "imu_inrow_p99_deg_s": round(float(np.degrees(np.percentile(np.abs(yaw_imu[elig]), 99))), 2),
        "odom_inrow_max_deg_s": round(float(np.degrees(np.abs(yaw_odom[elig]).max())), 1),
        "imu_inrow_max_deg_s": round(float(np.degrees(np.abs(yaw_imu[elig]).max())), 1),
        "note": ("Unlike F017 (camera + LiDAR AGREE on tilt), odom twist yaw-rate and the IMU gyro "
                 "DISAGREE: signed correlation is near-zero/negative and odom carries a heavy noise "
                 "tail in-row (max >100 deg/s vs IMU ~8). The locked gate does not use yaw at all, so "
                 "it is unaffected; but /odometry/base_raw.twist.angular.z is not a reliable "
                 "standalone yaw-rate on this bag.")}

    # --- spec-3 frame-error check (literal |v_y| > 0.30 keep-predicate) -------------------------
    spec3 = {"literal_vy_gt_030_inrow_retained_pct": round(100 * float((np.abs(vy) > 0.30)[elig].mean()), 1),
             "native_vy_inrow_mean_ms": round(float(np.abs(vy)[elig].mean()), 3),
             "native_vx_inrow_mean_ms": round(float(vx[elig].mean()), 3),
             "note": ("A literal reading of the original PID_PIPELINE_SPEC.md §3 ('native |v_y| "
                      "replaces the finite-difference v_y') retains only ~1.5% of in-row frames -> "
                      "body v_y is lateral slip, not along-row velocity. §3 corrected to forward v_x.")}

    f022 = {"signal": "pose-finite-difference (offline, 15-sample centred — non-causal)",
            "gate": "speed>0.10 & |along-row v_y|>0.30 & |heading-rate|<22.1deg/s (3 world-frame predicates)",
            "non_in_row_rejection_%": {"A": 98.4, "B": 98.5, "C": 98.4},
            "in_row_FP_%": {"A": 1.2, "B": 1.2, "C": 1.2},
            "per_category_%": {"stationary": 100.0, "turn": "95.1-95.3", "transition": "95.7-96.0"}}

    report = {
        "config": {
            "bag": bag, "finding": "F026",
            "signal_source": "native /odometry/base_raw.twist (D042) + /imu/data cross-check",
            "locked_gate": "v_x > V_MIN  [single body-frame forward-speed predicate; turn predicate dropped]",
            "threshold_construction": "V_MIN = in-row p1 of v_x (F022 in-row-percentile methodology)",
            "n_in_row_frames": int(elig.sum()), "n_non_in_row_frames": int(nonrow_mask.sum()),
            "frame_correction_note": ("D042/§3 corrected 20 Jul 2026 (F026): F022's three world-frame "
                                      "predicates collapse to one native body-frame forward-speed "
                                      "predicate; body v_y is lateral slip (see spec3_check); the turn "
                                      "predicate is dropped (see turn_predicate_decomposition).")},
        "F026_locked_gate": locked,
        "turn_predicate_decomposition": turn_decomp,
        "sensor_crosscheck_odom_vs_imu": crosscheck,
        "spec3_frame_error_check": spec3,
        "F022_reference": f022,
    }
    OUT.write_text(json.dumps(report, indent=2))

    # --- console summary -----------------------------------------------------------------------
    rej = locked["non_in_row_rejection"]["per_arm"]; fp = locked["in_row_FP"]["per_arm"]
    print(f"[{bag}] LOCKED native state gate (F026): v_x > {v_min:.3f} m/s (in-row p1)  |  "
          f"in-row {int(elig.sum())}  non-in-row {int(nonrow_mask.sum())}")
    print(f"  {'':<20}{'A':>7}{'B':>7}{'C':>7}   (F022 ref: reject 98.4/98.5/98.4  FP 1.2)")
    print(f"  {'non-in-row reject %':<20}" + "".join(f"{rej[a]['reject_%']:>7}" for a in "ABC"))
    print(f"  {'in-row FP %':<20}" + "".join(f"{fp[a]['reject_%']:>7}" for a in "ABC"))
    pc = locked["non_in_row_rejection"]["per_category"]
    for c in ("stationary", "turn", "transition"):
        print(f"    {c:<16}" + "".join(f"{pc[c][a]:>7}" for a in "ABC"))
    for src in ("odom_yaw", "imu_yaw"):
        td = turn_decomp[src]
        print(f"  turn pred [{src}]: +{td['added_nonrow_frame_rejections']} non-row rejections, "
              f"+{td['added_inrow_frame_FP']} in-row FP (=> dropped)")
    print(f"  odom-vs-imu yaw: corr {crosscheck['corr_moving']}, mean|diff| {crosscheck['mean_abs_diff_deg_s']} deg/s, "
          f"odom p99 {crosscheck['odom_inrow_p99_deg_s']} vs imu {crosscheck['imu_inrow_p99_deg_s']} deg/s")
    print(f"  spec-3 check: literal |v_y|>0.30 retains {spec3['literal_vy_gt_030_inrow_retained_pct']}% of in-row (frame error)")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
