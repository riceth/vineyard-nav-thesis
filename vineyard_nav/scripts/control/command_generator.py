"""CP-P2/CP-P3 command generator (PID_PIPELINE_SPEC.md §4-§7).

Wires the locked control stack end-to-end, per (arm, seed) stream (P-1a), segmented by in-row pass:

  centreline adapter (P-1a, centreline_adapter.py)
    -> native state gate (P-5a; D042-corrected single forward-speed predicate v_x > V_MIN, F026)
    -> weighted-sum PID (P-2a) with conditional-integration anti-windup + omega_max clamp (P-6)
    -> toggleable ramp/rate limiter (P-6; output produced BOTH with and without it)
    -> hold-last on abstention OR gate rejection (D043), with the two hold reasons labelled
       separately (abstain_cls vs state_gate) and held spans flagged for the CP-P4 dual metric.

  python3 scripts/control/command_generator.py --bag march
    -> results/geometric/{bag}/final/command_evaluation/command_per_frame.csv
       results/geometric/{bag}/final/command_evaluation/command_summary.json
  (the CP-P2 command_dryrun_summary.json is the committed placeholder-gain wiring record and is
   left untouched — this run writes the P-4c locked-gain siblings.)

GAINS: P-4c FIRST-PRINCIPLES, NOT DATA-FITTED (locked 20 Jul 2026; F027 continuation). The P-4/4b
k-fold tracking objective was retired as degenerate (F027: best-linear R^2 = 0.0070, folds collapse
to ~zero gains), so the gains are DERIVED from the closed-loop plant model rather than tuned:

  unicycle + small angle, offset ~ -y and heading ~ -psi  =>  y'' + Kpsi*y' + v*Kp*y = 0
  => omega_n = sqrt(v*Kp),  zeta = Kpsi / (2*sqrt(v*Kp))

Two physical choices fix both gains: the 2%-settling distance d_s and the damping ratio zeta.
Kd = 0 (d(offset)/dt ~ -v*psi duplicates the heading term — the §5.1 double-count, made rigorous)
and Ki = 0 (an integrator would chase the F017/F016 systematic bias; CP-P2 showed wind-up to the
clamp). The F017/D038 sensor-common tilt is removed from the heading as a fixed calibration
constant before the law. Nothing here is fitted to an evaluation objective, so no circularity.
"""
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "scripts" / "control"))
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
from bag_config import resolve
from centreline_adapter import load_streams, ARMS, SEEDS
from state_gate_native import load_native_signals, fit_forward_floor, native_gate   # locked CP-P1 primitives

# --- P-4c LOCKED design point (first-principles; locked 20 Jul 2026) ----------------------------
ZETA = 1.0             # critically damped: overshoot risk (clipping a vine) outweighs correction speed
D_SETTLE_M = 20.0      # 2%-settling distance of travel. 5 m / 10 m saturate omega_max at 0.06 / 0.23 m
                       # offset — at or below the F013 RMS offset error (0.21 m), i.e. on ordinary noise;
                       # 20 m saturates only beyond F023's p99 tail (0.71 m); 30 m under-corrects within
                       # a single ~53 m row. See F027 continuation.
TILT_DEBIAS_DEG = 2.31  # F017/D038 sensor-common centreline tilt (m_centre=+0.040 -> +2.31 deg), removed
                        # from heading as a FIXED CALIBRATION CONSTANT from an independent prior finding
                        # (not a tuned parameter). Uncorrected it would consume ~30% of omega_max on a
                        # projection artefact rather than genuine row-centring error.
KD_FIXED = 0.0         # redundant with the heading term (d(offset)/dt ~ -v*psi) — §5.1
KI_FIXED = 0.0         # would integrate the F017/F016 systematic bias (CP-P2: wind-up to the clamp)
SIGN = +1.0            # corrective: offset>0 => centreline to +Y (left) => +yaw (left) toward it.
                       # Locked 20 Jul 2026 (spec §5 amendment); validated by the CP-P2 dry-run spans.
# The P-6 ramp/rate limit is DERIVED from the locked design point — see derive_ramp_rate().
NOMINAL_DT = 1.0 / 14.77


def derive_gains(v_inrow, d_s=None):
    """P-4c first-principles gains from the locked (d_s, zeta) design point and the measured in-row
    forward speed. For zeta=1 the 2% settling time is ~5.8/omega_n, so a settling DISTANCE d_s of
    travel at speed v gives omega_n = 5.8*v/d_s. Then Kp = omega_n^2/v and Kpsi = 2*zeta*omega_n,
    converted to per-DEGREE because the pipeline logs heading in degrees."""
    d_s = D_SETTLE_M if d_s is None else d_s
    omega_n = 5.8 * v_inrow / d_s
    return {"Kp": omega_n ** 2 / v_inrow,                    # rad/s per m
            "Kpsi": 2.0 * ZETA * omega_n * (np.pi / 180.0),  # rad/s per degree
            "Kd": KD_FIXED, "Ki": KI_FIXED,
            "omega_n_rad_s": omega_n, "zeta": ZETA, "d_settle_m": d_s, "v_inrow_ms": v_inrow}


def derive_ramp_rate(omega_n, omega_max, zeta=ZETA):
    """P-6 ramp/rate limit — LOCKED 20 Jul 2026 as the maximum slew the locked closed-loop design can
    legitimately demand inside its own clamped envelope; anything faster is not a control response but
    a perception jump or hold-transition step, which is what this layer exists to clip.

    Differentiating omega = Kp*offset + Kpsi*heading (Kd=Ki=0) under the same plant model used for the
    gains (d(offset)/dt = -v*psi, d(heading)/dt = -omega) gives
        omega_dot = -omega_n^2 * psi - 2*zeta*omega_n * omega
    =>  |omega_dot| <= omega_n^2 * |psi|_max + 2*zeta*omega_n * omega_max.
    Both bounds are already locked: |omega| <= omega_max (P-6 clamp), and |psi| <= omega_max/Kpsi_rad
    = omega_max/(2*zeta*omega_n) (the heading at which the heading term alone saturates the clamp).
    Substituting collapses to a closed form in already-locked constants only:
        omega_dot_max = (2*zeta + 1/(2*zeta)) * omega_n * omega_max     [= 2.5*omega_n*omega_max at zeta=1]

    CAVEAT: this rests on D038's STRAIGHT-ROW model — there is no legitimate curvature term for the
    limiter to clip. If the pipeline is ever extended to curved-row scenarios the bound must be
    RE-DERIVED, not assumed to carry over."""
    return (2.0 * zeta + 1.0 / (2.0 * zeta)) * omega_n * omega_max


CSV_HEADER = ("arm,seed,i,pass_id,cls,v_x,state_ok,source,hold_reason,abstain_cls,state_gate_reject,"
              "hold_run_len,last_valid_i,offset,heading,heading_debiased,omega_cmd,omega_cmd_ramp,"
              "omega_exec_odom,omega_exec_imu,saturated")


def _clamp(x, lo, hi):
    return hi if x > hi else lo if x < lo else x


def run_stream(stream, arm, seed, state_ok_of, sig, gains, omega_max, ramp_rate):
    """Run gate -> PID -> hold-last -> ramp over one (arm, seed) stream, segmented by pass (controller
    state resets each pass). sig[i] = (yaw_odom, yaw_imu, t). Returns a list of CSV row strings."""
    by_pass = defaultdict(list)
    for r in stream:
        by_pass[r["pass_id"]].append(r)                       # stream is i-sorted, so each pass stays i-ordered

    rows = []
    for pid in sorted(by_pass):
        integ = 0.0; last_cmd = 0.0; have_cmd = False
        last_fresh_off = last_fresh_t = None; prev_t = None; ramp_prev = 0.0
        hold_run = 0; last_valid_i = -1
        for r in by_pass[pid]:
            i = r["i"]; yo, yi, t, vxi = sig[i]
            dt = (t - prev_t) if (prev_t is not None and t > prev_t) else NOMINAL_DT
            state_ok = state_ok_of[i]                          # locked native gate (v_x > V_MIN)
            gate_reject = not state_ok
            abstain = r["abstained"]
            fresh = (not abstain) and (not gate_reject)

            if fresh:
                off, hdg_raw = r["offset"], r["heading"]
                hdg = hdg_raw - TILT_DEBIAS_DEG            # F017/D038 fixed-calibration de-bias
                deriv = ((off - last_fresh_off) / (t - last_fresh_t)
                         if (last_fresh_off is not None and t > last_fresh_t) else 0.0)
                # conditional-integration anti-windup (P-6): try the integral update; if the resulting
                # command saturates, FREEZE the integral (do not accumulate this step) and recompute.
                cand_integ = integ + off * dt
                omega_cand = SIGN * (gains["Kp"] * off + gains["Kpsi"] * hdg
                                     + gains["Kd"] * deriv + gains["Ki"] * cand_integ)
                if abs(omega_cand) > omega_max:
                    saturated = True
                    omega = SIGN * (gains["Kp"] * off + gains["Kpsi"] * hdg
                                    + gains["Kd"] * deriv + gains["Ki"] * integ)   # frozen integral
                else:
                    saturated = False
                    integ = cand_integ
                    omega = omega_cand
                omega = _clamp(omega, -omega_max, omega_max)
                last_cmd, have_cmd = omega, True
                last_fresh_off, last_fresh_t = off, t
                hold_run, last_valid_i = 0, i
                source, hold_reason, abstain_cls = "fresh", "", ""
                off_s, hdg_s, hdgc_s = f"{off:.4f}", f"{hdg_raw:.4f}", f"{hdg:.4f}"
            else:
                omega = last_cmd if have_cmd else 0.0          # hold last valid command (D043)
                saturated = have_cmd and abs(omega) >= omega_max - 1e-9
                hold_run += 1
                source = "held"
                hold_reason = "both" if (abstain and gate_reject) else ("abstain" if abstain else "state_gate")
                abstain_cls = r["cls"] if abstain else ""
                off_s = hdg_s = hdgc_s = ""

            # toggleable ramp/rate limiter (P-6): slew the command; produced alongside the un-ramped cmd
            omega_ramp = _clamp(omega, ramp_prev - ramp_rate * dt, ramp_prev + ramp_rate * dt)
            ramp_prev, prev_t = omega_ramp, t

            rows.append(
                f"{arm},{seed},{i},{pid},{r['cls']},{vxi:.3f},"
                f"{int(state_ok)},{source},{hold_reason},{abstain_cls},{int(gate_reject)},"
                f"{hold_run},{last_valid_i},{off_s},{hdg_s},{hdgc_s},{omega:.6f},{omega_ramp:.6f},"
                f"{yo:.6f},{yi:.6f},{int(saturated)}")
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bag", default="march")
    bag = ap.parse_args().bag
    B = resolve(bag, "eligible")
    streams, man = load_streams(bag)
    frames = man["frames"]

    # native signals (reused from CP-P1): v_x for the gate, executed yaw (odom + imu) for §7a reference
    vx, vy, yaw_odom, yaw_imu = load_native_signals(B["db3"], frames)
    elig = np.array([f["eligible"] for f in frames])
    v_min = fit_forward_floor(vx, elig)                        # locked gate threshold (F026)
    keep = native_gate(vx, v_min)                             # locked single-predicate gate mask
    state_ok_of = {f["i"]: bool(keep[k]) for k, f in enumerate(frames)}
    t_of = {f["i"]: f["t_offset_s"] for f in frames}
    vx_of = {f["i"]: float(vx[k]) for k, f in enumerate(frames)}
    sig = {f["i"]: (float(yaw_odom[k]), float(yaw_imu[k]), t_of[f["i"]], vx_of[f["i"]])
           for k, f in enumerate(frames)}

    # omega_max = p99 of the executed yaw-rate (P-6). F026 showed the odom yaw-rate is noise-inflated
    # (in-row p99 13.4 deg/s vs the IMU gyro's 3.5), so the IMU-based p99 is the physically sound clamp;
    # both are recorded and the choice is FLAGGED for CP-P3 review.
    omega_max_imu = float(np.percentile(np.abs(yaw_imu[elig]), 99))
    omega_max_odom = float(np.percentile(np.abs(yaw_odom[elig]), 99))
    omega_max = omega_max_imu

    OUT = B["out_dir"].parent / "command_evaluation"
    OUT.mkdir(parents=True, exist_ok=True)

    v_inrow = float(vx[elig].mean())          # measured in-row forward speed (native twist)
    GAINS = derive_gains(v_inrow)             # P-4c first-principles gains

    ramp_rate = derive_ramp_rate(GAINS["omega_n_rad_s"], omega_max)   # P-6 locked, derived

    all_rows = [CSV_HEADER]
    for arm in ARMS:
        for seed in SEEDS:
            all_rows += run_stream(streams[(arm, seed)], arm, seed, state_ok_of, sig,
                                   GAINS, omega_max, ramp_rate)
    (OUT / "command_per_frame.csv").write_text("\n".join(all_rows) + "\n")

    # --- dry-run sanity summary ----------------------------------------------------------------
    def summarise(rows):
        n = len(rows); fresh = held = ab = sg = both = sat = ramp_diff = 0
        spans = []; run = 0
        for ln in rows:
            c = ln.split(",")
            src, hr, hrl, oc, orp, satf = c[7], c[8], int(c[11]), float(c[16]), float(c[17]), int(c[20])
            if src == "fresh":
                fresh += 1
                if run: spans.append(run); run = 0
            else:
                held += 1; run = max(run, hrl)
                ab += hr in ("abstain", "both"); sg += hr in ("state_gate", "both"); both += hr == "both"
            sat += satf
            if abs(oc - orp) > 1e-6: ramp_diff += 1
        if run: spans.append(run)
        return {"n": n, "fresh": fresh, "held": held, "held_abstain": ab, "held_state_gate": sg,
                "held_both": both, "hold_spans": len(spans), "max_hold_run": max(spans) if spans else 0,
                "mean_hold_run": round(sum(spans) / len(spans), 2) if spans else 0,
                "saturated_frames": sat, "ramp_changed_frames": ramp_diff}

    per_stream = {}
    for arm in ARMS:
        for seed in SEEDS:
            r0 = [ln for ln in all_rows[1:] if ln.startswith(f"{arm},{seed},")]
            per_stream[f"{arm}_{seed}"] = summarise(r0)

    # --- design-point sensitivity: WHY d_s = 20 m (backs the F027-continuation rationale) --------
    off_rms = float(np.sqrt(np.mean([r["offset"] ** 2 for r in streams[("A", 42)] if r["offset"] is not None])))
    yaw_std = float(yaw_imu[elig].std())
    dp = {}
    for d in (5.0, 10.0, 20.0, 30.0):
        g = derive_gains(v_inrow, d)
        cmd_at_rms = g["Kp"] * off_rms                       # unclamped command at the RMS offset
        dp[f"{d:g}m"] = {
            "Kp": round(g["Kp"], 6), "Kpsi_per_deg": round(g["Kpsi"], 8),
            "saturating_offset_m": round(omega_max / g["Kp"], 4),
            "unclamped_cmd_at_rms_offset_deg_s": round(float(np.degrees(cmd_at_rms)), 3),
            "x_observed_yaw_p99": round(cmd_at_rms / omega_max, 2),
            "x_observed_yaw_std": round(cmd_at_rms / yaw_std, 2)}
    design_point = {
        "locked": "d_s=20 m, zeta=1.0",
        "in_row_offset_rms_m": round(off_rms, 4),
        "observed_in_row_yaw_std_deg_s": round(float(np.degrees(yaw_std)), 3),
        "observed_in_row_yaw_p99_deg_s": round(float(np.degrees(omega_max)), 3),
        "candidates": dp,
        "note": ("5 m / 10 m saturate at offsets at or below the in-row offset RMS (ordinary estimation "
                 "noise, F013); 20 m saturates only beyond F023's p99 tail (0.71 m); 30 m under-corrects "
                 "within a ~53 m row. A 5 m design point would also demand commands far exceeding the "
                 "yaw-rate envelope the platform actually used (see x_observed_yaw_* multipliers).")}

    # --- demanded-slew statistics (backs the P-6 ramp lock; F027-A) ----------------------------
    _by = defaultdict(list)
    for ln in all_rows[1:]:
        c = ln.split(",")
        _by[(c[0], c[1], c[3])].append((int(c[2]), float(c[16])))
    slew = []
    for vlist in _by.values():
        vlist.sort()
        for (i0, w0), (i1, w1) in zip(vlist, vlist[1:]):
            gap = t_of[i1] - t_of[i0]
            if gap > 0:
                slew.append(abs(w1 - w0) / gap)
    slew = np.array(slew)
    ramp_layer = {
        "rate_rad_s2": round(ramp_rate, 6),
        "derivation": "(2*zeta + 1/(2*zeta)) * omega_n * omega_max  [= 2.5*omega_n*omega_max at zeta=1]",
        "demanded_slew_rad_s2": {q: round(float(np.percentile(slew, q)), 5) for q in (50, 90, 95, 99)},
        "demanded_slew_max_rad_s2": round(float(slew.max()), 5),
        "clipped_pct_of_transitions": round(100 * float((slew > ramp_rate).mean()), 2),
        "interpretation": ("Median demanded slew sits BELOW the bound, so genuine transitions pass "
                           "untouched; the tail (p99 ~24x the bound) is frame-to-frame perception jitter "
                           "converted into commanded yaw-rate change - exactly what the P-6 layer exists "
                           "to clip. By construction the bound cannot throttle a legitimate response."),
        "straight_row_caveat": ("Validity rests on D038's straight-row model: no legitimate curvature term "
                                "exists to clip. Extending the pipeline to curved rows requires re-deriving "
                                "this bound rather than assuming it still holds.")}

    summary = {
        "status": "CP-P3: command stream at the P-4c LOCKED first-principles gains (d_s=20 m, zeta=1, F017 de-bias). Backs F027 continuation.",
        "config": {
            "bag": bag, "streams": "P-1a: 9 independent (arm x seed) streams, per-pass state reset",
            "gate": f"P-5a locked native gate v_x > {round(v_min,4)} m/s (D042/F026)",
            "pid": "P-2a weighted-sum on offset(m)+heading(deg)",
            "P4c_locked_gains": {k: round(v, 8) for k, v in GAINS.items()},
            "gain_derivation": ("P-4c first principles: omega_n=5.8*v/d_s (zeta=1, 2% settling over d_s of "
                                "travel); Kp=omega_n^2/v; Kpsi=2*zeta*omega_n per degree; Kd=Ki=0. NOT fitted."),
            "tilt_debias_deg": TILT_DEBIAS_DEG,
            "tilt_debias_note": ("F017/D038 sensor-common tilt removed from heading as a fixed calibration "
                                 "constant; uncorrected it would consume ~30% of omega_max on a projection artefact."),
            "sign_convention": "+1 corrective (locked, spec §5 amendment)",
            "anti_windup": "conditional integration (freeze integral on saturation, P-6)",
            "omega_max_rad_s": round(omega_max, 5),
            "omega_max_source": "IMU-gyro in-row p99 (F026: odom yaw-rate noise-inflated, unreliable)",
            "omega_max_odom_rad_s_for_reference": round(omega_max_odom, 5),
            "ramp_rate_rad_s2": round(ramp_rate, 6),
            "ramp_note": "toggleable P-6 layer (DERIVED, locked); omega_cmd (off) + omega_cmd_ramp (on)",
            "executed_yaw_reference": "omega_exec_odom (spec §7a) + omega_exec_imu (F026 alternative) both logged"},
        "design_point_sensitivity": design_point,
        "ramp_layer": ramp_layer,
        "per_stream": per_stream,
        "overall": {k: sum(per_stream[s][k] for s in per_stream)
                    for k in ("n", "fresh", "held", "held_abstain", "held_state_gate", "held_both",
                              "hold_spans", "saturated_frames", "ramp_changed_frames")}}
    (OUT / "command_summary.json").write_text(json.dumps(summary, indent=2))

    # --- console: config + summary + sample spans (abstention + gate-rejection) -----------------
    ov = summary["overall"]
    print(f"[{bag}] CP-P3 P-4c LOCKED GAINS (first-principles, not fitted)")
    print(f"  gate v_x>{v_min:.3f} m/s | omega_max {omega_max:.4f} rad/s ({np.degrees(omega_max):.2f} deg/s, IMU p99) "
          f"| ramp {ramp_rate:.5f} rad/s^2 (derived)\n  gains Kp={GAINS['Kp']:.5f} Kpsi={GAINS['Kpsi']:.6f}/deg Kd={GAINS['Kd']} Ki={GAINS['Ki']} "
          f"(omega_n={GAINS['omega_n_rad_s']:.4f} rad/s, zeta={ZETA}, d_s={D_SETTLE_M} m, v={v_inrow:.3f} m/s, tilt-debias {TILT_DEBIAS_DEG} deg)")
    print(f"  overall: {ov['n']} rows | fresh {ov['fresh']} | held {ov['held']} "
          f"(abstain {ov['held_abstain']} / state_gate {ov['held_state_gate']} / both {ov['held_both']}) "
          f"| hold-spans {ov['hold_spans']} | saturated {ov['saturated_frames']} | ramp-changed {ov['ramp_changed_frames']}")

    def show_span(reason, ctx=2, span=5):
        r0 = [ln for ln in all_rows[1:] if ln.startswith("A,42,")]
        for k, ln in enumerate(r0):
            if ln.split(",")[8] == reason:
                lo = max(0, k - ctx); hi = min(len(r0), k + span + 1)
                print(f"\n  --- sample {reason} span (A,seed42), cols: i,pass,cls,v_x,state_ok,source,hold_reason,hold_run,last_valid,offset,heading,omega,omega_ramp,sat ---")
                for ln2 in r0[lo:hi]:
                    c = ln2.split(",")
                    print("   " + " ".join([c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[11], c[12],
                                            c[13] or "-", c[15] or "-", c[16], c[17], c[20]]))
                return
        print(f"\n  (no {reason} span found in A,seed42)")

    show_span("abstain"); show_span("state_gate")
    print(f"\nwrote {OUT / 'command_per_frame.csv'}  ({ov['n']} rows)")
    print(f"wrote {OUT / 'command_summary.json'}")


if __name__ == "__main__":
    main()
