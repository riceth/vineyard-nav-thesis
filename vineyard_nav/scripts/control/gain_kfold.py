"""CP-P3 step (B): pass-level k-fold gain tuning — run to DOCUMENT THE DEGENERACY of the P-4
tracking objective (PID_PIPELINE_SPEC.md P-4/4b, §7a). Produces the empirical evidence for F027.

*** THIS IS NOT A USABLE TUNING RESULT. ***
The objective `minimise RMS(omega_hat - omega_exec)` is degenerate: the executed yaw-rate of the BLT
run is not predictable from the perceived centreline (the BLT robot was under GPS/topological
navigation, not vine-row visual servoing), so the best achievable weighted-sum fit explains ~0% of
the variance and the argmin collapses toward zero gains. This script measures exactly that — best
gains per fold, their stability across folds, and the pooled out-of-fold RMS versus the
**zero-gain (command-nothing) baseline**. The strand then pivots to P-4c (fixed principled gains).

  python3 scripts/control/gain_kfold.py --bag march
    -> results/geometric/{bag}/final/command_evaluation/gain_kfold.json

STRUCTURE (D014-compliant — corrected from the initial CP-P3 scoping):
  gains are SHARED across arms, tuned ONCE per fold on the pooled training passes of all 9
  (arm x seed) streams, then scored per stream on the held-out pass. Tuning per arm would give each
  arm its own controller and destroy the controlled comparison (D014/P-2: identical controller,
  only perception differs).

  for fold k in 0..10 (held-out pass = k):
      g*_k = argmin over the gain grid of  RMS(omega_hat - omega_ref)  pooled over all streams,
             all passes != k, FRESH frames only (frames where the controller acts on geometry)
      score g*_k on pass k, per stream -> out-of-fold residuals
  pooled OOF RMS = sqrt(sum oof_sq / sum oof_n)

Controller state resets per pass, so each (stream, pass) is simulated ONCE for all gain candidates
and folds simply sum the per-pass squared-error tables. The PID law mirrors command_generator.py
exactly (SIGN=+1 corrective, conditional-integration anti-windup, omega_max clamp); the vectorised
re-implementation is verified against the CP-P2 per-frame CSV at the placeholder gains.
"""
import sys
import json
import argparse
import itertools
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "scripts" / "control"))
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
from bag_config import resolve
from centreline_adapter import load_streams, ARMS, SEEDS
from state_gate_native import load_native_signals, fit_forward_floor, native_gate

SIGN = +1.0
NOMINAL_DT = 1.0 / 14.77
PLACEHOLDER_GAINS = (0.04, 0.0015, 0.01, 0.003)      # CP-P2 values, used only for the sim-equivalence check

# --- gain grid (deterministic; includes the all-zero candidate as the command-nothing baseline) ---
GRID = {"Kp":   [0.0, 0.01, 0.02, 0.04, 0.08, 0.15, 0.30],
        "Kpsi": [0.0, 0.0005, 0.001, 0.002, 0.005],
        "Kd":   [0.0, 0.005, 0.02, 0.05],
        "Ki":   [0.0, 0.001, 0.003, 0.01]}


def build_grid():
    combos = list(itertools.product(GRID["Kp"], GRID["Kpsi"], GRID["Kd"], GRID["Ki"]))
    a = np.array(combos, float)
    G = {"Kp": a[:, 0], "Kpsi": a[:, 1], "Kd": a[:, 2], "Ki": a[:, 3]}
    zero_idx = int(np.argmin(np.abs(a).sum(axis=1)))          # the all-zero candidate
    return G, combos, zero_idx


def sim_pass(offs, hdgs, fresh, ts, refs, G, omega_max):
    """Simulate one (stream, pass) for ALL gain candidates at once. Returns {ref: sq_err (N,)} and the
    fresh-frame count. Held frames repeat the previous command and carry no geometry, so they are
    excluded from the tuning objective (the integral is frozen across them, as in command_generator)."""
    N = G["Kp"].shape[0]
    integ = np.zeros(N)
    sq = {k: np.zeros(N) for k in refs}
    cnt = 0
    last_off = last_t = prev_t = None
    for f in range(len(offs)):
        t = ts[f]
        dt = (t - prev_t) if (prev_t is not None and t > prev_t) else NOMINAL_DT
        prev_t = t
        if not fresh[f]:
            continue                                           # held: no fresh command, state frozen
        off, hdg = offs[f], hdgs[f]
        deriv = ((off - last_off) / (t - last_t)) if (last_off is not None and t > last_t) else 0.0
        base = SIGN * (G["Kp"] * off + G["Kpsi"] * hdg + G["Kd"] * deriv)
        cand = integ + off * dt
        w_cand = base + SIGN * G["Ki"] * cand
        sat = np.abs(w_cand) > omega_max
        integ = np.where(sat, integ, cand)                     # conditional-integration anti-windup
        w = np.where(sat, base + SIGN * G["Ki"] * integ, w_cand)
        w = np.clip(w, -omega_max, omega_max)
        for name, arr in refs.items():
            sq[name] += (w - arr[f]) ** 2
        cnt += 1
        last_off, last_t = off, t
    return sq, cnt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bag", default="march")
    bag = ap.parse_args().bag
    B = resolve(bag, "eligible")
    streams, man = load_streams(bag)
    frames = man["frames"]

    vx, vy, yaw_odom, yaw_imu = load_native_signals(B["db3"], frames)
    elig = np.array([f["eligible"] for f in frames])
    v_min = fit_forward_floor(vx, elig)
    keep = native_gate(vx, v_min)
    state_ok_of = {f["i"]: bool(keep[k]) for k, f in enumerate(frames)}
    t_of = {f["i"]: f["t_offset_s"] for f in frames}
    # reference signals: IMU is SIGN-INVERTED vs base_link (verified against pose-derived yaw-rate,
    # corr -0.953 on turns) -> negate. odom carried as a sensitivity reference.
    ref_imu_of = {f["i"]: -float(yaw_imu[k]) for k, f in enumerate(frames)}
    ref_odom_of = {f["i"]: float(yaw_odom[k]) for k, f in enumerate(frames)}
    omega_max = float(np.percentile(np.abs(yaw_imu[elig]), 99))      # P-6 locked clamp (|yaw|, sign-agnostic)

    G, combos, zero_idx = build_grid()
    N = len(combos)
    passes = sorted({r["pass_id"] for r in streams[("A", 42)]})
    keys = [(a, s) for a in ARMS for s in SEEDS]

    # ---- pre-slice each (stream, pass) into arrays -------------------------------------------
    sliced = {}
    for k in keys:
        by_pass = {}
        for r in streams[k]:
            by_pass.setdefault(r["pass_id"], []).append(r)
        for p, rs in by_pass.items():
            offs = np.array([r["offset"] if r["offset"] is not None else 0.0 for r in rs])
            hdgs = np.array([r["heading"] if r["heading"] is not None else 0.0 for r in rs])
            fresh = np.array([(not r["abstained"]) and state_ok_of[r["i"]] for r in rs])
            ts = np.array([t_of[r["i"]] for r in rs])
            refs = {"imu_signcorrected": np.array([ref_imu_of[r["i"]] for r in rs]),
                    "odom": np.array([ref_odom_of[r["i"]] for r in rs])}
            sliced[(k, p)] = (offs, hdgs, fresh, ts, refs)

    # ---- simulate every (stream, pass) once, for all N candidates -----------------------------
    REFS = ("imu_signcorrected", "odom")
    SQ = {rn: np.zeros((len(keys), len(passes), N)) for rn in REFS}
    CNT = np.zeros((len(keys), len(passes)))
    for ki, k in enumerate(keys):
        for pi, p in enumerate(passes):
            offs, hdgs, fresh, ts, refs = sliced[(k, p)]
            sq, cnt = sim_pass(offs, hdgs, fresh, ts, refs, G, omega_max)
            for rn in REFS:
                SQ[rn][ki, pi, :] = sq[rn]
            CNT[ki, pi] = cnt
        print(f"  simulated {k} ({int(CNT[ki].sum())} fresh frames)", flush=True)

    # ---- sim-equivalence check vs the CP-P2 per-frame CSV (placeholder gains) ------------------
    # Compare against the gains the CSV was ACTUALLY generated with (recorded in command_summary.json),
    # not a hard-coded set: command_per_frame.csv is regenerated whenever the locked gains change, so a
    # fixed reference would silently compare two different controllers.
    csv_p = B["out_dir"].parent / "command_evaluation" / "command_per_frame.csv"
    sum_p = B["out_dir"].parent / "command_evaluation" / "command_summary.json"
    tilt_debias = 0.0
    if sum_p.exists():
        _cfg = json.load(open(sum_p))["config"]
        _g = _cfg["P4c_locked_gains"]
        gains_used = (_g["Kp"], _g["Kpsi"], _g["Kd"], _g["Ki"])
        tilt_debias = float(_cfg.get("tilt_debias_deg", 0.0))
        src = "P-4c locked gains (from command_summary.json)"
    else:
        gains_used, src = PLACEHOLDER_GAINS, "CP-P2 placeholder gains (no summary found)"
    verify = {"note": f"vectorised sim vs command_generator.py CSV at {src}, stream (A,42)",
              "gains_compared": [round(float(x), 8) for x in gains_used],
              "tilt_debias_deg_applied": tilt_debias}
    if csv_p.exists():
        Gp = {n: np.array([v]) for n, v in zip(("Kp", "Kpsi", "Kd", "Ki"), gains_used)}
        got = {}
        for p in passes:
            offs, hdgs, fresh, ts, refs = sliced[(("A", 42), p)]
            # re-simulate storing per-frame omega
            integ = np.zeros(1); last_off = last_t = prev_t = None
            for f in range(len(offs)):
                t = ts[f]; dt = (t - prev_t) if (prev_t is not None and t > prev_t) else NOMINAL_DT
                prev_t = t
                if not fresh[f]:
                    continue
                off, hdg = offs[f], hdgs[f] - tilt_debias      # match command_generator's F017 de-bias
                deriv = ((off - last_off) / (t - last_t)) if (last_off is not None and t > last_t) else 0.0
                base = SIGN * (Gp["Kp"] * off + Gp["Kpsi"] * hdg + Gp["Kd"] * deriv)
                cand = integ + off * dt
                w_cand = base + SIGN * Gp["Ki"] * cand
                sat = np.abs(w_cand) > omega_max
                integ = np.where(sat, integ, cand)
                w = np.clip(np.where(sat, base + SIGN * Gp["Ki"] * integ, w_cand), -omega_max, omega_max)
                got[(p, f)] = float(w[0])
                last_off, last_t = off, t
        # map CSV fresh rows of (A,42) in pass/frame order
        rows = [ln.split(",") for ln in csv_p.read_text().splitlines()[1:] if ln.startswith("A,42,")]
        idx_in_pass = {}
        diffs = []
        for c in rows:
            p = int(c[3])
            idx_in_pass[p] = idx_in_pass.get(p, -1) + 1
            f = idx_in_pass[p]
            if c[7] == "fresh" and (p, f) in got:
                diffs.append(abs(got[(p, f)] - float(c[16])))   # col 16 = omega_cmd (heading_debiased is 15)
        verify["n_compared"] = len(diffs)
        verify["max_abs_diff_rad_s"] = float(max(diffs)) if diffs else None

    # ---- k-fold (shared gains; D014) ----------------------------------------------------------
    out = {}
    for rn in REFS:
        per_fold = []
        oof_sq_stream = np.zeros(len(keys)); oof_n_stream = np.zeros(len(keys))
        zero_oof_sq = np.zeros(len(keys))
        for pi, p in enumerate(passes):
            tr = [j for j in range(len(passes)) if j != pi]
            tr_sq = SQ[rn][:, tr, :].sum(axis=(0, 1))              # (N,) pooled over streams+train passes
            tr_n = CNT[:, tr].sum()
            rms_tr = np.sqrt(tr_sq / tr_n)
            b = int(np.argmin(rms_tr))
            te_sq = SQ[rn][:, pi, b]; te_n = CNT[:, pi]
            oof_sq_stream += te_sq; oof_n_stream += te_n
            zero_oof_sq += SQ[rn][:, pi, zero_idx]
            per_fold.append({
                "fold": pi, "held_out_pass": int(p),
                "best_gains": dict(zip(("Kp", "Kpsi", "Kd", "Ki"), [float(x) for x in combos[b]])),
                "train_rms_rad_s": round(float(rms_tr[b]), 6),
                "train_rms_zero_gain_rad_s": round(float(rms_tr[zero_idx]), 6),
                "oof_rms_rad_s": round(float(np.sqrt(te_sq.sum() / te_n.sum())), 6)})
        pooled = float(np.sqrt(oof_sq_stream.sum() / oof_n_stream.sum()))
        pooled_zero = float(np.sqrt(zero_oof_sq.sum() / oof_n_stream.sum()))
        per_arm = {}
        for a in ARMS:
            ii = [i for i, kk in enumerate(keys) if kk[0] == a]
            per_arm[a] = round(float(np.sqrt(oof_sq_stream[ii].sum() / oof_n_stream[ii].sum())), 6)
        gains_arr = np.array([[f["best_gains"][g] for g in ("Kp", "Kpsi", "Kd", "Ki")] for f in per_fold])
        stability = {g: {"values": sorted(set(gains_arr[:, j].tolist())),
                         "min": float(gains_arr[:, j].min()), "max": float(gains_arr[:, j].max()),
                         "median": float(np.median(gains_arr[:, j])),
                         "n_distinct_across_11_folds": int(len(set(gains_arr[:, j].tolist())))}
                     for j, g in enumerate(("Kp", "Kpsi", "Kd", "Ki"))}
        # grid flatness: spread of the objective over the whole grid (pooled, all passes)
        all_sq = SQ[rn].sum(axis=(0, 1)); all_rms = np.sqrt(all_sq / CNT.sum())
        out[rn] = {
            "per_fold": per_fold, "gain_stability_across_folds": stability,
            "pooled_oof_rms_rad_s": round(pooled, 6),
            "zero_gain_pooled_oof_rms_rad_s": round(pooled_zero, 6),
            "improvement_over_zero_gain_pct": round(100 * (pooled_zero - pooled) / pooled_zero, 3),
            "per_arm_oof_rms_rad_s": per_arm,
            "grid_flatness": {"best_rms": round(float(all_rms.min()), 6),
                              "worst_rms": round(float(all_rms.max()), 6),
                              "zero_gain_rms": round(float(all_rms[zero_idx]), 6),
                              "best_vs_zero_pct": round(100 * (all_rms[zero_idx] - all_rms.min()) / all_rms[zero_idx], 3)},
            "reference_std_rad_s": None}
    # reference stds (scale context)
    for rn, arr in (("imu_signcorrected", -yaw_imu), ("odom", yaw_odom)):
        out[rn]["reference_std_rad_s"] = round(float(arr[elig].std()), 6)

    # ---- ROOT CAUSE: is the reference predictable from the centreline at all? -----------------
    # Best-fit linear model [offset, heading, 1] -> reference is an UPPER BOUND on what any
    # weighted-sum law of (offset, heading) can achieve. R^2 ~ 0 => the objective is degenerate.
    pred = {}
    for rn in REFS:
        O, H, R = [], [], []
        per_arm_r2 = {}
        for a in ARMS:
            oa, ha, ra = [], [], []
            for k in [kk for kk in keys if kk[0] == a]:
                for p in passes:
                    offs, hdgs, fresh, ts, refs = sliced[(k, p)]
                    oa += offs[fresh].tolist(); ha += hdgs[fresh].tolist(); ra += refs[rn][fresh].tolist()
            oa, ha, ra = np.array(oa), np.array(ha), np.array(ra)
            X = np.column_stack([oa, ha, np.ones_like(oa)])
            bb, *_ = np.linalg.lstsq(X, ra, rcond=None); res = ra - X @ bb
            per_arm_r2[a] = {"n": len(ra),
                             "corr_offset": round(float(np.corrcoef(oa, ra)[0, 1]), 4),
                             "corr_heading": round(float(np.corrcoef(ha, ra)[0, 1]), 4),
                             "best_linear_R2": round(float(1 - np.sum(res ** 2) / np.sum((ra - ra.mean()) ** 2)), 5)}
            O += oa.tolist(); H += ha.tolist(); R += ra.tolist()
        O, H, R = np.array(O), np.array(H), np.array(R)
        X = np.column_stack([O, H, np.ones_like(O)])
        bb, *_ = np.linalg.lstsq(X, R, rcond=None); res = R - X @ bb
        pred[rn] = {"pooled": {"n": len(R),
                               "corr_offset": round(float(np.corrcoef(O, R)[0, 1]), 4),
                               "corr_heading": round(float(np.corrcoef(H, R)[0, 1]), 4),
                               "best_linear_R2": round(float(1 - np.sum(res ** 2) / np.sum((R - R.mean()) ** 2)), 5),
                               "resid_rms_rad_s": round(float(np.sqrt(np.mean(res ** 2))), 6),
                               "reference_std_rad_s": round(float(R.std()), 6)},
                    "per_arm": per_arm_r2,
                    "note": ("Upper bound on ANY weighted-sum law of (offset, heading). R^2 ~ 0 means the "
                             "perceived centreline carries essentially no information about the executed "
                             "yaw-rate, so minimising RMS(omega_hat - omega_ref) is degenerate.")}

    report = {
        "status": ("CP-P3 step (B): k-fold run to DOCUMENT DEGENERACY of the P-4 tracking objective. "
                   "NOT a usable tuning result — the strand pivots to P-4c (fixed principled gains). Evidence for F027."),
        "config": {
            "bag": bag, "n_gain_candidates": N, "grid": GRID,
            "structure": ("SHARED gains (D014): tuned once per fold on the pooled training passes of all 9 "
                          "(arm x seed) streams; scored per stream on the held-out pass. NOT per-arm."),
            "folds": f"{len(passes)} CP-1 passes (pass-level, spatial independence)",
            "objective": "minimise RMS(omega_hat - omega_ref) over FRESH frames (controller acts on geometry)",
            "references": {"primary": "imu_signcorrected = -/imu/data.angular_velocity.z (IMU z is inverted vs base_link)",
                           "sensitivity": "odom = /odometry/base_raw.twist.angular.z"},
            "gate": f"v_x > {round(v_min,4)} m/s (D042/F026)", "omega_max_rad_s": round(omega_max, 6),
            "pid": "P-2a weighted-sum, SIGN=+1, conditional-integration anti-windup, omega_max clamp (ramp layer not used for tuning)"},
        "predictability_of_reference": pred,
        "sim_equivalence_check": verify,
        "kfold": out}
    OUT = B["out_dir"].parent / "command_evaluation" / "gain_kfold.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)   # own the output dir — don't rely on command_generator having run first
    OUT.write_text(json.dumps(report, indent=2))

    # ---- console ------------------------------------------------------------------------------
    print(f"\n[{bag}] CP-P3(B) k-fold DEGENERACY documentation | {N} candidates, {len(passes)} folds, shared gains")
    if verify.get("max_abs_diff_rad_s") is not None:
        print(f"  sim-equivalence vs CP-P2 CSV: max|diff| {verify['max_abs_diff_rad_s']:.2e} rad/s over {verify['n_compared']} frames")
    for rn in REFS:
        o = out[rn]
        print(f"\n  === reference: {rn} (in-row std {o['reference_std_rad_s']:.5f} rad/s) ===")
        print(f"    pooled OOF RMS      {o['pooled_oof_rms_rad_s']:.5f} rad/s")
        print(f"    zero-gain baseline  {o['zero_gain_pooled_oof_rms_rad_s']:.5f} rad/s")
        print(f"    improvement over commanding NOTHING: {o['improvement_over_zero_gain_pct']:.3f} %")
        print(f"    grid flatness: best {o['grid_flatness']['best_rms']:.5f} / worst {o['grid_flatness']['worst_rms']:.5f} "
              f"/ zero {o['grid_flatness']['zero_gain_rms']:.5f} rad/s")
        st = o["gain_stability_across_folds"]
        print("    gain stability across 11 folds:")
        for g in ("Kp", "Kpsi", "Kd", "Ki"):
            print(f"      {g:5s} median {st[g]['median']:<8g} range [{st[g]['min']:g}, {st[g]['max']:g}] "
                  f"({st[g]['n_distinct_across_11_folds']} distinct values)")
        print(f"    per-arm OOF RMS: {o['per_arm_oof_rms_rad_s']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
