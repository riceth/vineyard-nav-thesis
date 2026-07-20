"""CP-P2 command generator — DRY RUN (plumbing/wiring check only; PID_PIPELINE_SPEC.md §4-§7).

Wires the locked control stack end-to-end, per (arm, seed) stream (P-1a), segmented by in-row pass:

  centreline adapter (P-1a, centreline_adapter.py)
    -> native state gate (P-5a; D042-corrected single forward-speed predicate v_x > V_MIN, F026)
    -> weighted-sum PID (P-2a) with conditional-integration anti-windup + omega_max clamp (P-6)
    -> toggleable ramp/rate limiter (P-6; output produced BOTH with and without it)
    -> hold-last on abstention OR gate rejection (D043), with the two hold reasons labelled
       separately (abstain_cls vs state_gate) and held spans flagged for the CP-P4 dual metric.

  python3 scripts/control/command_generator.py --bag march
    -> results/geometric/{bag}/final/command_evaluation/command_per_frame.csv
       results/geometric/{bag}/final/command_evaluation/command_dryrun_summary.json

*** DRY RUN ONLY — GAINS ARE PLACEHOLDERS, NOT TUNED. ***
The Kp/Kpsi/Kd/Ki below are provisional values chosen ONLY to exercise every code path (produce
sane-magnitude commands with occasional saturation so anti-windup/clamp engage); they are NOT a
tuned controller and the command stream is NOT a result. Gain tuning is CP-P3 via the locked
P-4/4b pass-level k-fold procedure. Do not read tracking/smoothness numbers off this artefact.
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

# --- PLACEHOLDER control parameters (CP-P2 dry run ONLY; NOT tuned — CP-P3 fits these) ----------
GAINS = {"Kp": 0.04, "Kpsi": 0.0015, "Kd": 0.01, "Ki": 0.003}  # rad/s per {m, deg, (m/s), (m*s)} — PLACEHOLDER
# (chosen only so typical |omega| sits well under omega_max with saturation on the offset tail /
#  late-pass integral wind-up, so the clamp + anti-windup are visibly exercised — NOT tuned.)
SIGN = +1.0            # corrective convention: offset>0 => centreline to +Y (left) => +yaw (left) toward it.
                       # FLAGGED for CP-P3: spec §5 wrote a leading minus under an error convention; the sign
                       # is a one-constant choice that does not affect wiring. Confirm at tuning.
RAMP_RATE = 0.30       # PLACEHOLDER rad/s^2 slew limit for the ramp/rate limiter (P-6)
NOMINAL_DT = 1.0 / 14.77

CSV_HEADER = ("arm,seed,i,pass_id,cls,v_x,state_ok,source,hold_reason,abstain_cls,state_gate_reject,"
              "hold_run_len,last_valid_i,offset,heading,omega_cmd,omega_cmd_ramp,omega_exec_odom,"
              "omega_exec_imu,saturated")


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
                off, hdg = r["offset"], r["heading"]
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
                off_s, hdg_s = f"{off:.4f}", f"{hdg:.4f}"
            else:
                omega = last_cmd if have_cmd else 0.0          # hold last valid command (D043)
                saturated = have_cmd and abs(omega) >= omega_max - 1e-9
                hold_run += 1
                source = "held"
                hold_reason = "both" if (abstain and gate_reject) else ("abstain" if abstain else "state_gate")
                abstain_cls = r["cls"] if abstain else ""
                off_s = hdg_s = ""

            # toggleable ramp/rate limiter (P-6): slew the command; produced alongside the un-ramped cmd
            omega_ramp = _clamp(omega, ramp_prev - ramp_rate * dt, ramp_prev + ramp_rate * dt)
            ramp_prev, prev_t = omega_ramp, t

            rows.append(
                f"{arm},{seed},{i},{pid},{r['cls']},{vxi:.3f},"
                f"{int(state_ok)},{source},{hold_reason},{abstain_cls},{int(gate_reject)},"
                f"{hold_run},{last_valid_i},{off_s},{hdg_s},{omega:.6f},{omega_ramp:.6f},"
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

    all_rows = [CSV_HEADER]
    for arm in ARMS:
        for seed in SEEDS:
            all_rows += run_stream(streams[(arm, seed)], arm, seed, state_ok_of, sig,
                                   GAINS, omega_max, RAMP_RATE)
    (OUT / "command_per_frame.csv").write_text("\n".join(all_rows) + "\n")

    # --- dry-run sanity summary ----------------------------------------------------------------
    def summarise(rows):
        n = len(rows); fresh = held = ab = sg = both = sat = ramp_diff = 0
        spans = []; run = 0
        for ln in rows:
            c = ln.split(",")
            src, hr, hrl, oc, orp, satf = c[7], c[8], int(c[11]), float(c[15]), float(c[16]), int(c[19])
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

    summary = {
        "status": "CP-P2 DRY RUN — plumbing/wiring check; gains are PLACEHOLDERS, not tuned (CP-P3). NOT a finding.",
        "config": {
            "bag": bag, "streams": "P-1a: 9 independent (arm x seed) streams, per-pass state reset",
            "gate": f"P-5a locked native gate v_x > {round(v_min,4)} m/s (D042/F026)",
            "pid": "P-2a weighted-sum on offset(m)+heading(deg)",
            "PLACEHOLDER_gains": GAINS, "sign_convention": "+1 corrective (FLAGGED for CP-P3)",
            "anti_windup": "conditional integration (freeze integral on saturation, P-6)",
            "omega_max_rad_s": round(omega_max, 5),
            "omega_max_source": "IMU-gyro in-row p99 (F026: odom yaw-rate noise-inflated, unreliable)",
            "omega_max_odom_rad_s_for_reference": round(omega_max_odom, 5),
            "ramp_rate_rad_s2": RAMP_RATE, "ramp_note": "toggleable P-6 layer; omega_cmd (off) + omega_cmd_ramp (on)",
            "executed_yaw_reference": "omega_exec_odom (spec §7a) + omega_exec_imu (F026 alternative) both logged"},
        "per_stream": per_stream,
        "overall": {k: sum(per_stream[s][k] for s in per_stream)
                    for k in ("n", "fresh", "held", "held_abstain", "held_state_gate", "held_both",
                              "hold_spans", "saturated_frames", "ramp_changed_frames")}}
    (OUT / "command_dryrun_summary.json").write_text(json.dumps(summary, indent=2))

    # --- console: config + summary + sample spans (abstention + gate-rejection) -----------------
    ov = summary["overall"]
    print(f"[{bag}] CP-P2 DRY RUN (PLACEHOLDER gains — not tuned)")
    print(f"  gate v_x>{v_min:.3f} m/s | omega_max {omega_max:.4f} rad/s ({np.degrees(omega_max):.2f} deg/s, IMU p99) "
          f"| ramp {RAMP_RATE} rad/s^2 | gains {GAINS}")
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
                                            c[13] or "-", c[14] or "-", c[15], c[16], c[19]]))
                return
        print(f"\n  (no {reason} span found in A,seed42)")

    show_span("abstain"); show_span("state_gate")
    print(f"\nwrote {OUT / 'command_per_frame.csv'}  ({ov['n']} rows)")
    print(f"wrote {OUT / 'command_dryrun_summary.json'}")


if __name__ == "__main__":
    main()
