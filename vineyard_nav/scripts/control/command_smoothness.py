"""CP-P4 command smoothness — the D014 strand-3 deliverable (PID_PIPELINE_SPEC.md §7b/§7c/§7d).

Computes the command-level smoothness metrics on the locked P-4c command stream. This is the
strand's cross-arm headline: smoothness is computed on the command stream ITSELF and needs no
external reference, so it is unaffected by F027 (which invalidated the tracking objective).

  python3 scripts/control/command_smoothness.py --bag march
    -> results/geometric/{bag}/final/command_evaluation/command_smoothness.json

METRICS (§7b, locked; = PHASE_C_SPEC §232): RMS of the frame-to-frame yaw-rate change dw (jerk
proxy), command jitter (SD of dw), and saturation rate (fraction of frames at |w| = w_max).

THREE VIEWS (D043 dual metric + the locked hold-transition split):
  inclusive  — every consecutive within-pass frame pair (held frames counted)
  exclusive  — pairs where BOTH frames carry a fresh command
  hold_exit  — pairs held->fresh, i.e. the command jump on leaving a hold span
Held commands repeat the previous value, so fresh->held and held->held pairs are dw = 0 by
construction; that is what deflates the inclusive view, and why the three views are reported
separately rather than blended (the "exclusive" definition is both-fresh, NOT drop-then-difference,
which would collapse a 143-frame hold into one spurious delta).

Deltas are computed WITHIN a pass only — controller state resets per pass, so a cross-pass delta is
meaningless. dw is reported in BOTH the §7b literal form (rad/s per frame) and rad/s^2, the latter
being directly comparable to the locked P-6 ramp limit.

STATISTICS (§7c, locked): point estimates over all in-row frames; per-(arm,seed) CIs over the
Ds = 1.5 m spatially-independent subsample; per-arm across-seed and paired cross-arm CIs by
moving-block bootstrap at the block length from block_lengths.py. No p-values (D014).
CAVEAT (documented, not resolved here): that block length was derived from the offset/heading
series. Differencing whitens a series, so the true block length for dw is likely SHORTER and these
CIs are therefore conservative (wider) rather than anti-conservative.
"""
import sys
import json
import argparse
import collections
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
from bag_config import resolve
import block_lengths as BL

ARMS = ("A", "B", "C")
SEEDS = (42, 43, 44)
BOOT_B = 2000          # per-(arm,seed) simple bootstrap over the subsample
BLOCK_BOOT = 10000     # per-arm / paired moving-block bootstrap
BOOT_SEED = 42
COLS = {"arm": 0, "seed": 1, "i": 2, "pass": 3, "source": 7, "hold_run": 11,
        "omega_cmd": 16, "omega_cmd_ramp": 17, "saturated": 20}
# F007 / O009 blob map: which seeds emitted the whole-frame false mask on test scene 6799.
BLOB_SEEDS = {"A": [], "B": [42, 43], "C": [43, 44]}


def rms(a):
    a = np.asarray(a, float)
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")


def load_stream(csv_path):
    """(arm, seed) -> list of per-frame dicts ordered by (pass, frame index)."""
    st = collections.defaultdict(list)
    for ln in Path(csv_path).read_text().splitlines()[1:]:
        c = ln.split(",")
        st[(c[COLS["arm"]], int(c[COLS["seed"]]))].append({
            "i": int(c[COLS["i"]]), "pass": int(c[COLS["pass"]]),
            "fresh": c[COLS["source"]] == "fresh",
            "hold_run": int(c[COLS["hold_run"]]),
            "w": float(c[COLS["omega_cmd"]]), "wr": float(c[COLS["omega_cmd_ramp"]]),
            "sat": int(c[COLS["saturated"]])})
    for k in st:
        st[k].sort(key=lambda r: (r["pass"], r["i"]))
    return dict(st)


def deltas(rows, t_of, col):
    """Consecutive within-pass frame pairs -> (view, leading frame i, dw, dw_per_s).
    view in {exclusive, hold_exit, hold_entry, hold_hold}; 'inclusive' is the union of all."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a["pass"] != b["pass"]:
            continue                                   # controller state resets per pass
        dt = t_of[b["i"]] - t_of[a["i"]]
        if dt <= 0:
            continue
        dw = b[col] - a[col]
        if a["fresh"] and b["fresh"]:
            v = "exclusive"
        elif (not a["fresh"]) and b["fresh"]:
            v = "hold_exit"                            # the command jump on leaving a hold
        elif a["fresh"] and (not b["fresh"]):
            v = "hold_entry"                           # dw = 0 by construction
        else:
            v = "hold_hold"                            # dw = 0 by construction
        out.append((v, a["i"], dw, dw / dt))
    return out


def metric_block(dws, dws_per_s, sat_num, sat_den):
    return {"n_pairs": len(dws),
            "rms_dw_rad_s_per_frame": round(rms(dws), 8),
            "jitter_sd_dw_rad_s_per_frame": round(float(np.std(dws)), 8) if len(dws) else None,
            "rms_dw_rad_s2": round(rms(dws_per_s), 6),
            "saturation_rate_pct": round(100 * sat_num / sat_den, 3) if sat_den else None,
            "n_frames_for_saturation": sat_den}


def simple_boot_ci(vals, stat, b=BOOT_B, seed=BOOT_SEED):
    vals = np.asarray(vals, float)
    if len(vals) < 8:
        return [None, None, len(vals)]
    rng = np.random.default_rng(seed)
    n = len(vals)
    bs = [stat(vals[rng.integers(0, n, n)]) for _ in range(b)]
    return [round(float(np.percentile(bs, 2.5)), 8), round(float(np.percentile(bs, 97.5)), 8), n]


def blocks_of(rows, L):
    """rows: (pass, pos, value...). Returns list of index-blocks of length L within each pass."""
    byp = collections.defaultdict(list)
    for r in rows:
        byp[r[0]].append(r)
    out = []
    for v in byp.values():
        v.sort(key=lambda r: r[1])
        for s in range(0, len(v) - L + 1):
            out.append(v[s:s + L])
    return out


def block_ci(rows, L, valfn, stat, b=BLOCK_BOOT, seed=BOOT_SEED):
    """Moving-block bootstrap of `stat` over a (pass, pos, ...) series."""
    blks = blocks_of(rows, L)
    if len(blks) < 8:
        return [None, None]
    n = len(rows)
    nb = int(np.ceil(n / L))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(b):
        pick = rng.integers(0, len(blks), nb)
        vals = [valfn(r) for j in pick for r in blks[j]]
        out.append(stat(np.asarray(vals, float)))
    return [round(float(np.percentile(out, 2.5)), 8), round(float(np.percentile(out, 97.5)), 8)]


def paired_block_ci(rows, L, b=BLOCK_BOOT, seed=BOOT_SEED):
    """rows: (pass, pos, dw_A, dw_B). Bootstraps RMS(A) - RMS(B) with the pairing preserved."""
    blks = blocks_of(rows, L)
    if len(blks) < 8:
        return [None, None]
    nb = int(np.ceil(len(rows) / L))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(b):
        pick = rng.integers(0, len(blks), nb)
        a = [r[2] for j in pick for r in blks[j]]
        c = [r[3] for j in pick for r in blks[j]]
        out.append(rms(a) - rms(c))
    return [round(float(np.percentile(out, 2.5)), 8), round(float(np.percentile(out, 97.5)), 8)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bag", default="march")
    bag = ap.parse_args().bag
    B = resolve(bag, "eligible")
    man = json.load(open(B["manifest"]))
    CSV = B["out_dir"].parent / "command_evaluation" / "command_per_frame.csv"
    OUT = B["out_dir"].parent / "command_evaluation" / "command_smoothness.json"

    frames = man["frames"]
    t_of = {f["i"]: f["t_offset_s"] for f in frames}
    sub = set(f["i"] for f in frames if f.get("subsample_1p5m"))
    # along-track position per frame, for the moving-block bootstrap (as line_fit_eval / paired)
    pos = {}
    byp = collections.defaultdict(list)
    for f in frames:
        if f["eligible"]:
            byp[f["pass_id"]].append(f)
    for pid, fs in byp.items():
        fs.sort(key=lambda f: f["i"])
        xy = np.array([[f["x"], f["y"]] for f in fs])
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        for f, d in zip(fs, cum):
            pos[f["i"]] = (pid, float(d))

    bl = BL.pooled_block_lengths(B["per_frame_csv"], man)
    L = int(max(bl["L_GT1"], bl["L_GT2"]))        # conservative of the two locked lengths
    streams = load_stream(CSV)

    # ---- per (arm, seed) x column x view -------------------------------------------------------
    per_stream, dw_store = {}, {}
    for arm in ARMS:
        for sd in SEEDS:
            rows = streams[(arm, sd)]
            n_all = len(rows); n_fresh = sum(r["fresh"] for r in rows)
            sat_all = sum(r["sat"] for r in rows); sat_fresh = sum(r["sat"] for r in rows if r["fresh"])
            blk = {}
            for col, key in (("w", "omega_cmd"), ("wr", "omega_cmd_ramp")):
                d = deltas(rows, t_of, col)
                views = {}
                for view in ("inclusive", "exclusive", "hold_exit"):
                    sel = d if view == "inclusive" else [x for x in d if x[0] == view]
                    dws = [x[2] for x in sel]; dps = [x[3] for x in sel]
                    sn, sdn = (sat_all, n_all) if view != "exclusive" else (sat_fresh, n_fresh)
                    m = metric_block(dws, dps, sn, sdn)
                    m["ci_rms_dw_subsample"] = simple_boot_ci(
                        [x[2] for x in sel if x[1] in sub], rms)
                    views[view] = m
                    dw_store[(arm, sd, key, view)] = [(x[1], x[2]) for x in sel]
                zero = [x for x in d if x[0] in ("hold_entry", "hold_hold")]
                views["_pair_composition"] = {
                    "n_total_pairs": len(d),
                    "n_exclusive": sum(1 for x in d if x[0] == "exclusive"),
                    "n_hold_exit": sum(1 for x in d if x[0] == "hold_exit"),
                    "n_zero_by_construction": len(zero),
                    "pct_zero_by_construction": round(100 * len(zero) / max(len(d), 1), 2),
                    "note": "fresh->held and held->held repeat the previous command, so dw = 0 exactly; "
                            "they deflate the inclusive view."}
                blk[key] = views
            hr = [r["hold_run"] for r in rows if not r["fresh"]]
            spans = []
            run = 0
            for r in rows:
                if r["fresh"]:
                    if run: spans.append(run); run = 0
                else:
                    run += 1
            if run: spans.append(run)
            blk["hold_spans"] = {
                "n_spans": len(spans), "n_held_frames": len(hr),
                "mean_len": round(float(np.mean(spans)), 2) if spans else 0,
                "median_len": int(np.median(spans)) if spans else 0,
                "p95_len": int(np.percentile(spans, 95)) if spans else 0,
                "max_len": int(max(spans)) if spans else 0,
                "max_len_seconds": round(max(spans) / 14.77, 2) if spans else 0,
                "pct_frames_in_spans_over_1s": round(
                    100 * sum(s for s in spans if s > 15) / max(len(rows), 1), 2),
                "pct_frames_in_spans_over_5s": round(
                    100 * sum(s for s in spans if s > 74) / max(len(rows), 1), 2)}
            per_stream[f"{arm}_{sd}"] = blk

    # ---- per-arm: mean +/- SD across seeds (O009) ----------------------------------------------
    per_arm = {}
    for arm in ARMS:
        e = {}
        for key in ("omega_cmd", "omega_cmd_ramp"):
            e[key] = {}
            for view in ("inclusive", "exclusive", "hold_exit"):
                for met in ("rms_dw_rad_s_per_frame", "jitter_sd_dw_rad_s_per_frame",
                            "rms_dw_rad_s2", "saturation_rate_pct"):
                    v = [per_stream[f"{arm}_{s}"][key][view][met] for s in SEEDS]
                    v = [x for x in v if x is not None]
                    e[key].setdefault(view, {})[met] = [round(float(np.mean(v)), 8),
                                                        round(float(np.std(v)), 8)] if v else None
        per_arm[arm] = e

    # ---- per-arm across-seed CI + paired cross-arm CI (moving block) ---------------------------
    def across(arm, key, view):
        """Per-pair mean over the 3 seeds, on pairs present in all 3; -> (pass, pos, dw)."""
        sets = [dict(dw_store[(arm, s, key, view)]) for s in SEEDS]
        common = set(sets[0]) & set(sets[1]) & set(sets[2])
        return sorted([(pos[i][0], pos[i][1], float(np.mean([s[i] for s in sets])))
                       for i in common if i in pos])

    per_arm_ci, paired = {}, {}
    for key in ("omega_cmd", "omega_cmd_ramp"):
        for view in ("inclusive", "exclusive"):
            for arm in ARMS:
                r = across(arm, key, view)
                per_arm_ci.setdefault(key, {}).setdefault(view, {})[arm] = {
                    "n": len(r), "rms_dw": round(rms([x[2] for x in r]), 8),
                    "ci": block_ci(r, L, lambda x: x[2], rms)}
            for a, b2 in (("A", "B"), ("A", "C"), ("B", "C")):
                ra, rb = across(a, key, view), across(b2, key, view)
                ma, mb = {(p, q): v for p, q, v in ra}, {(p, q): v for p, q, v in rb}
                comm = sorted(set(ma) & set(mb))
                rows_p = [(p, q, ma[(p, q)], mb[(p, q)]) for p, q in comm]
                paired.setdefault(key, {}).setdefault(view, {})[f"{a}_minus_{b2}"] = {
                    "n": len(rows_p),
                    "delta_rms_dw": round(rms([r[2] for r in rows_p]) - rms([r[3] for r in rows_p]), 8),
                    "ci": paired_block_ci(rows_p, L)}

    # ---- blob-seed observation (descriptive only) ----------------------------------------------
    blob = {"blob_seeds": BLOB_SEEDS,
            "framing": ("DESCRIPTIVE OBSERVATION, NOT AN INFERENTIAL CLAIM. n = 3 seeds per arm, and "
                        "the link is indirect: F007's blob is a whole-frame false mask on ONE labelled "
                        "perception test scene (6799), whereas this command stream runs over the bag's "
                        "in-row frames. A difference here would be suggestive, not evidence."),
            "per_arm": {}}
    for arm in ARMS:
        bs, cs = BLOB_SEEDS[arm], [s for s in SEEDS if s not in BLOB_SEEDS[arm]]
        if not bs or not cs:
            blob["per_arm"][arm] = {"comparable": False,
                                    "reason": "all seeds in one group (no within-arm contrast)"}
            continue
        g = lambda ss: float(np.mean([per_stream[f"{arm}_{s}"]["omega_cmd"]["exclusive"]
                                      ["rms_dw_rad_s_per_frame"] for s in ss]))
        blob["per_arm"][arm] = {"comparable": True, "blob_seeds": bs, "clean_seeds": cs,
                                "rms_dw_blob": round(g(bs), 8), "rms_dw_clean": round(g(cs), 8),
                                "delta": round(g(bs) - g(cs), 8)}

    report = {
        "status": "CP-P4 — D014 strand-3 command smoothness on the locked P-4c command stream (F028).",
        "config": {
            "bag": bag, "source_csv": str(CSV.relative_to(PKG)),
            "metrics": "RMS(dw) + jitter SD(dw) + saturation rate (§7b; = PHASE_C_SPEC §232)",
            "units": "dw reported as rad/s per frame (§7b literal) and rad/s^2 (comparable to the "
                     "locked P-6 ramp limit 0.033743 rad/s^2)",
            "views": {"inclusive": "all consecutive within-pass pairs (held counted)",
                      "exclusive": "pairs where BOTH frames are fresh",
                      "hold_exit": "held->fresh pairs — the command jump on leaving a hold"},
            "delta_scope": "within-pass only (controller state resets per pass)",
            "block_length_L": L, "block_lengths_source": bl,
            "block_length_caveat": ("L is reused from block_lengths.py as §7c directs, but it was derived "
                                    "from the offset/heading series. Differencing whitens a series, so the "
                                    "true block length for dw is likely SHORTER — these CIs are therefore "
                                    "CONSERVATIVE (wider), not anti-conservative. Documented, not resolved."),
            "subsample_n": len(sub), "boot_B": BOOT_B, "block_boot_B": BLOCK_BOOT, "seed": BOOT_SEED,
            "not_recomputed": ("§7a tracking-vs-executed-yaw-rate is NOT recomputed — F027 already "
                               "reported it and demoted it to a caveated diagnostic. The §8 dead-reckoning "
                               "trajectory cross-check is SKIPPED: it compares our predicted path against a "
                               "path driven by GPS/topological navigation, so it inherits F027's invalidity "
                               "and would add no new information.")},
        "per_stream": per_stream, "per_arm_mean_sd_across_seeds": per_arm,
        "per_arm_ci_across_seed": per_arm_ci, "paired_cross_arm": paired,
        "blob_seed_observation": blob}
    OUT.write_text(json.dumps(report, indent=2))

    # ---- console ------------------------------------------------------------------------------
    print(f"[{bag}] CP-P4 command smoothness | block length L={L} | subsample {len(sub)}")
    for key in ("omega_cmd", "omega_cmd_ramp"):
        print(f"\n=== {key} ===")
        for view in ("inclusive", "exclusive", "hold_exit"):
            print(f"  -- {view} --")
            for arm in ARMS:
                m = per_arm[arm][key][view]
                r, j = m["rms_dw_rad_s_per_frame"], m["jitter_sd_dw_rad_s_per_frame"]
                s = m["saturation_rate_pct"]
                print(f"    {arm}: RMS dw {r[0]:.6f}+/-{r[1]:.6f}  jitter {j[0]:.6f}  "
                      f"rad/s^2 {m['rms_dw_rad_s2'][0]:.5f}  sat {s[0]:.2f}%")
        for view in ("inclusive", "exclusive"):
            print(f"  paired ({view}):", {k: (v["delta_rms_dw"], v["ci"])
                                          for k, v in paired[key][view].items()})
    pc = per_stream["A_42"]["omega_cmd"]["_pair_composition"]
    print(f"\n  pair composition (A,42): {pc['pct_zero_by_construction']}% of pairs are dw=0 by construction")
    hs = per_stream["A_42"]["hold_spans"]
    print(f"  hold spans (A,42): n={hs['n_spans']} mean={hs['mean_len']} max={hs['max_len']} "
          f"({hs['max_len_seconds']}s)")
    print(f"\n  blob-seed observation (descriptive):")
    for arm in ARMS:
        print(f"    {arm}: {blob['per_arm'][arm]}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
