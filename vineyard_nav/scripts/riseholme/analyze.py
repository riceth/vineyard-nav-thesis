"""Whole-bag geometric evaluation — consolidated analysis driver.

This single module runs every per-bag ANALYSIS / AGGREGATION step of the geometric pipeline. It
replaces seven leaf scripts (deleted in the same change); each script's logic now lives VERBATIM in
one function here, writing to the exact same output path via the same bag_config.resolve() bundle.

  In-row stratum  (resolve(bag, "eligible");  final/{bag}_evaluation/)
    line_fit_eval        -> line_fit_report.json      whole-bag line-fit aggregation (D040)
    paired_crossarm      -> paired_crossarm.json      paired cross-arm difference bootstrap (F013)
    config_analysis      -> config_analysis.json      downstream config sweep + argmin (F018)
    single_row_analysis  -> single_row_analysis.json  in-row abstention mechanism (F024)
    lidar_crosscheck     -> lidar_crosscheck.json     LiDAR-vs-camera heading cross-check (F017)

  Non-in-row stratum
    non_in_row_analysis  -> final/non_in_row_evaluation/non_in_row_analysis.json   (F020/F021)
    mitigation           -> final/mitigation_evaluation/mitigation_analysis.json   (F022/F023)

Usage
  python3 scripts/geometric/analyze.py --bag march                # the 5 in-row analyses
  python3 scripts/geometric/analyze.py --bag march --non-in-row   # non_in_row_analysis + mitigation
  python3 scripts/geometric/analyze.py --bag march --only line_fit_eval,lidar_crosscheck
  python3 scripts/geometric/analyze.py --bag march --non-in-row --only mitigation

--only takes a comma-separated subset of the analysis names above and selects within whichever
stratum --non-in-row picks (default: all analyses in that stratum). Bundles are wired exactly as the
originals resolved them: the in-row analyses receive resolve(bag, "eligible"); non_in_row_analysis
receives resolve(bag, "non_in_row"); mitigation receives both.

Structure notes
  * Heavy, analysis-specific dependencies are imported INSIDE the function that needs them
    (torch / ultralytics / albumentations / cv2 for single_row_analysis; sqlite3 / rosbags for
    lidar_crosscheck) so the other analyses do not pay for them and `--only <light analysis>` stays
    fast. This changes nothing numeric — every JSON is byte-for-byte identical to the old scripts.
  * Each analysis keeps its own helpers (rms / boot / block_stat_ci / ...) LOCAL. The near-duplicate
    helpers differ across the original scripts (variable names, rounding), so nothing was promoted to
    module scope: correctness over DRY.
"""
import sys
import json
import bisect
import argparse
import collections
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[2]                 # vineyard_nav/
sys.path.insert(0, str(PKG))                              # for `from scripts.perception... import ...`
sys.path.insert(0, str(PKG / "scripts" / "riseholme"))   # for the sibling pipeline modules

import projection_calibration as C          # noqa: E402  CP-2 image->ground projection
import block_lengths as BL                  # noqa: E402  Analysis-H pooled block lengths
from bag_config import resolve, frames_for_scope          # noqa: E402
exec(open(Path(__file__).resolve().parent / "row_model.py").read())   # NEAR, FARMAX, fit_side_far, centre_linefit


# ================================================================================================
# In-row stratum
# ================================================================================================
def _ci_warn(bag, bl):
    """D053: shout when a metric's block length rests on a resolution-limited decorrelation estimate.

    The CI is still written (deleting it would lose information), but it is anti-conservative — too
    narrow — because `decorr` could not be located more finely than the paired-sample spacing. Silence
    here is the failure mode this guards against: on july2023 the estimator returned L=2 with no
    indication anything was wrong, and two contrasts crossed into apparent significance as a result."""
    rel = bl.get("ci_reliability")
    if not rel:
        return
    bad = [m for m in ("GT1", "GT2") if not rel[m]["reliable"]]
    if not bad:
        return
    detail = "; ".join(f"{m}: {rel[m]['samples_per_decorr']} samples/decorr" for m in bad)
    print("\n  " + "!" * 76)
    print(f"  !! [{bag}] CI RELIABILITY WARNING (D053) — {', '.join(bad)}")
    print(f"  !! {detail}  (minimum for a trustworthy estimate: {rel['min_samples_per_decorr']})")
    print( "  !! The decorrelation length could not be resolved at this paired-sample spacing, so the")
    print( "  !! block length is under-estimated and the intervals below are ANTI-CONSERVATIVE (too")
    print( "  !! narrow). Do not report them as evidence for or against any contrast on this bag.")
    print("  " + "!" * 76 + "\n")


def line_fit_eval(B):
    """Whole-bag line-fit AGGREGATION (D040). Reads the per-frame CSV produced by line_fit_infer.py
    (12-col, all 9 models x every eligible frame) and aggregates it into line_fit_report.json."""
    MAN = json.load(open(B["manifest"]))
    B["out_dir"].mkdir(parents=True, exist_ok=True)
    IN_CSV = B["per_frame_csv"]           # 12-col, written by line_fit_infer.py
    OUT_JSON = B["line_fit_report"]

    SEEDS = [42, 43, 44]
    BOOT_B = 2000          # per-model subsample simple bootstrap
    BOOT_SEED = 42
    BLOCK_BOOT = 10000     # per-arm moving-block bootstrap

    # ---- parse the whole-bag per-frame CSV (12-col) -------------------------------------------
    D = collections.defaultdict(dict)               # (arm,seed) -> {i: (offset, heading, mc)} two_row
    rows_by_model = collections.defaultdict(list)   # (arm,seed) -> [(cls, n_base)] all frames
    for ln in IN_CSV.read_text().splitlines()[1:]:
        a, s, i, cls, off, hdg, mL, mR, mc, nb, adj, flags = ln.split(",")
        s, i, nb = int(s), int(i), int(nb)
        rows_by_model[(a, s)].append((cls, nb))
        if cls == "two_row" and off and hdg:
            D[(a, s)][i] = (float(off), float(hdg), float(mc) if mc else None)

    SUB = set(f["i"] for f in MAN["frames"] if f.get("subsample_1p5m"))   # eligible only; no split key
    N = sum(1 for f in MAN["frames"] if f["eligible"])

    def rms(a):
        a = np.asarray(a, float)
        return float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")

    def boot(vals, stat, b=BOOT_B, seed=BOOT_SEED):
        vals = np.asarray(vals, float)
        if len(vals) < 8:
            return [None, None, len(vals)]
        rng = np.random.default_rng(seed)
        n = len(vals)
        bs = [stat(vals[rng.integers(0, n, n)]) for _ in range(b)]
        return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3), n]

    # ---- per-model aggregation ----------------------------------------------------------------
    summaries = []
    for arm in "ABC":
        for seed in SEEDS:
            per = rows_by_model[(arm, seed)]
            tot = len(per)
            cov = collections.Counter(c for c, _ in per)
            d = D[(arm, seed)]
            off = [v[0] for v in d.values()]
            hdg = [v[1] for v in d.values()]
            mc = [v[2] for v in d.values() if v[2] is not None]
            # frame-index order so the fixed-seed subsample bootstrap is order-invariant
            # (independent of CSV row order; whole-bag CSV is already bag-index-ordered)
            offs_sub = [d[i][0] for i in sorted(d) if i in SUB]
            hdgs_sub = [d[i][1] for i in sorted(d) if i in SUB]
            summaries.append({
                "arm": arm, "seed": seed, "frames": tot,
                "two_row_pct": round(100 * cov["two_row"] / tot, 1),
                "single_pct": round(100 * cov["single_row"] / tot, 1),
                "none_pct": round(100 * cov["none"] / tot, 1),
                "mean_base": round(float(np.mean([nb for _, nb in per])), 1),
                "gt1_rms": round(rms(off), 3), "gt1_mean": round(float(np.mean(off)), 3),
                "gt1_ci": boot(offs_sub, rms),
                "gt2_rms": round(rms(hdg), 2), "gt2_mean": round(float(np.mean(hdg)), 2),
                "gt2_ci": boot(hdgs_sub, rms),
                "mc_mean": round(float(np.mean(mc)), 3),
                "tilt_deg": round(float(np.degrees(np.arctan(np.mean(mc)))), 2)})

    # ---- per-arm cross-seed mean+/-SD ---------------------------------------------------------
    agg = {}
    for arm in "ABC":
        rs = [s for s in summaries if s["arm"] == arm]

        def ms(k, rs=rs):
            v = np.array([r[k] for r in rs], float)
            return [round(float(v.mean()), 3), round(float(v.std()), 3)]
        agg[arm] = {k: ms(k) for k in ("two_row_pct", "gt1_rms", "gt2_rms", "mean_base", "tilt_deg")}

    # ---- per-arm across-seed moving-block bootstrap CIs (whole-bag Analysis-H block lengths) ---
    bl = BL.pooled_block_lengths(IN_CSV, MAN)
    L_GT1, L_GT2 = bl["L_GT1"], bl["L_GT2"]
    _ci_warn(B["bag"], bl)

    pos = {}
    byp = collections.defaultdict(list)
    for f in MAN["frames"]:
        if f["eligible"]:
            byp[f["pass_id"]].append(f)
    for pid, fs in byp.items():
        fs = sorted(fs, key=lambda f: f["i"])
        xy = np.array([[f["x"], f["y"]] for f in fs])
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        for f, d in zip(fs, cum):
            pos[f["i"]] = (pid, float(d))

    def block_stat_ci(rows, L, stat, b=BLOCK_BOOT, seed=42):
        """rows: (pass,pos,val); moving-block bootstrap of stat over the concatenated series."""
        bp = collections.defaultdict(list)
        for r in rows:
            bp[r[0]].append(r[2])
        blocks = []
        for v in bp.values():
            v = np.array(v)
            for st in range(0, len(v) - L + 1):
                blocks.append(v[st:st + L])
        if len(blocks) < 8:
            return [None, None]
        Nn = sum(len(v) for v in bp.values())
        nb = int(np.ceil(Nn / L))
        rng = np.random.default_rng(seed)
        out = [stat(np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), nb)])) for _ in range(b)]
        return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]

    def across(arm, k):
        """Per-frame mean over the 3 seeds (two_row in all 3), as (pass,pos,val)."""
        common = set.intersection(*[set(D[(arm, s)]) for s in SEEDS])
        return sorted([(pos[i][0], pos[i][1], np.mean([D[(arm, s)][i][k] for s in SEEDS]))
                       for i in common if i in pos])

    per_arm_ci = {}
    for arm in "ABC":
        o, h = across(arm, 0), across(arm, 1)
        per_arm_ci[arm] = {
            "n": len(o),
            "gt1_rms": round(rms([r[2] for r in o]), 4), "gt1_ci": block_stat_ci(o, L_GT1, rms),
            "gt2_rms": round(rms([r[2] for r in h]), 3), "gt2_ci": block_stat_ci(h, L_GT2, rms)}

    # ---- write report -------------------------------------------------------------------------
    report = {
        "config": {"bag": B["bag"], "split": "whole_bag", "n": N, "n_sub": len(SUB),
                   "row_model": "line-fit@2m + slope", "far_ext": True,
                   "per_model_boot_B": BOOT_B,
                   "per_model_ci": "Delta_s=1.5m subsample simple bootstrap",
                   "per_arm_block_boot_B": BLOCK_BOOT,
                   "per_arm_ci": "across-seed moving-block bootstrap (whole-bag Analysis-H block lengths)",
                   "block_lengths": bl},
        "per_model": summaries, "per_arm": agg, "per_arm_ci": per_arm_ci}
    OUT_JSON.write_text(json.dumps(report, indent=2))

    # ---- console summary ----------------------------------------------------------------------
    print(f"[{B['bag']}] whole-bag {N} frames/model; block lengths L_GT1={L_GT1} L_GT2={L_GT2}")
    print("\n==== per-arm cross-seed (mean +/- SD) ====")
    print(f"{'arm':>4}{'two-row%':>15}{'GT1 RMS':>14}{'GT2 RMS':>14}{'base':>12}{'tilt deg':>13}")
    for arm in "ABC":
        a = agg[arm]
        print(f"{arm:>4}{a['two_row_pct'][0]:>9.1f}+/-{a['two_row_pct'][1]:<4.1f}"
              f"{a['gt1_rms'][0]:>8.3f}+/-{a['gt1_rms'][1]:<5.3f}{a['gt2_rms'][0]:>7.2f}+/-{a['gt2_rms'][1]:<5.2f}"
              f"{a['mean_base'][0]:>7.1f}+/-{a['mean_base'][1]:<4.1f}{a['tilt_deg'][0]:>8.2f}+/-{a['tilt_deg'][1]:<4.2f}")
    print("\n==== per-arm across-seed RMS [moving-block 95% CI] ====")
    for arm in "ABC":
        c = per_arm_ci[arm]
        print(f"  {arm}: n={c['n']} | GT1 RMS {c['gt1_rms']} CI{c['gt1_ci']} m | GT2 RMS {c['gt2_rms']} CI{c['gt2_ci']} deg")
    print(f"\nwrote {OUT_JSON}")


def paired_crossarm(B):
    """Whole-bag paired cross-arm difference bootstrap (D040, F013) -> paired_crossarm.json."""
    BLOCK_BOOT = 10000
    SEEDS = [42, 43, 44]
    PAIRS = [("A", "B"), ("A", "C"), ("B", "C")]
    RTK = 0.038                                   # m, GT-1 floor (Polvara 2024 §5.3, March)
    G2FLOOR = {"A": 1.33, "B": 1.36, "C": 1.29}   # deg, GT-2 regression-residual floor per arm (F012)

    MAN = json.load(open(B["manifest"]))
    PF = B["per_frame_csv"]
    OUT = B["paired"]

    # path position (pass, cumulative m) per whole-bag eligible frame within its pass
    pos = {}
    byp = collections.defaultdict(list)
    for f in MAN["frames"]:
        if f["eligible"]:                        # whole-bag: eligible only, no split key
            byp[f["pass_id"]].append(f)
    for pid, fs in byp.items():
        fs = sorted(fs, key=lambda f: f["i"])
        xy = np.array([[f["x"], f["y"]] for f in fs])
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        for f, d in zip(fs, cum):
            pos[f["i"]] = (pid, float(d))

    D = collections.defaultdict(dict)   # (arm,seed) -> {i: (offset, heading)} two_row
    for ln in PF.read_text().splitlines()[1:]:
        a, s, i, cls, off, hdg, *_ = ln.split(",")
        if cls == "two_row" and off and hdg:
            D[(a, int(s))][int(i)] = (float(off), float(hdg))

    bl = BL.pooled_block_lengths(PF, MAN)                       # primary (0.1 threshold), F013 convention
    L_GT1, L_GT2 = bl["L_GT1"], bl["L_GT2"]
    _ci_warn(B["bag"], bl)
    bl_strict = BL.pooled_block_lengths(PF, MAN, thr=0.05)      # stricter GT-2 robustness (F013's 0.05 check)
    L_GT2_strict = bl_strict["L_GT2"]

    def block_stat_ci(rows, L, stat, b=BLOCK_BOOT, seed=42):
        """rows: (pass,pos,val); moving-block bootstrap of stat over the concatenated series."""
        bp = collections.defaultdict(list)
        for r in rows:
            bp[r[0]].append(r[2])
        blocks = []
        for v in bp.values():
            v = np.array(v)
            for st in range(0, len(v) - L + 1):
                blocks.append(v[st:st + L])
        if len(blocks) < 8:
            return [None, None]
        N = sum(len(v) for v in bp.values())
        nb = int(np.ceil(N / L))
        rng = np.random.default_rng(seed)
        out = [stat(np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), nb)])) for _ in range(b)]
        return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]

    def summ_pair(rows, L, floor):
        vals = [r[2] for r in rows]
        pt = float(np.mean(vals)) if vals else float("nan")
        ci = block_stat_ci(rows, L, np.mean)
        excl = ci[0] is not None and (ci[0] > 0 or ci[1] < 0)
        pct = lambda v: None if v is None else round(100 * v / floor, 1)
        return {"mean_diff": round(pt, 4), "n": len(vals), "block_L": L, "ci95": ci,
                "sign": "+" if pt > 0 else "-" if pt < 0 else "0", "ci_excludes_zero": bool(excl),
                "floor": floor, "mean_diff_pct_floor": pct(pt), "ci95_pct_floor": [pct(ci[0]), pct(ci[1])]}

    across_seed, per_seed, sign_consistency = {}, {}, {}
    for (x, y) in PAIRS:
        pk = f"{x}-{y}"
        g2f = (G2FLOOR[x] + G2FLOOR[y]) / 2

        # across-seed: per-frame mean over 3 seeds of each arm (two_row in all 3 of both arms)
        cx = set.intersection(*[set(D[(x, s)]) for s in SEEDS])
        cy = set.intersection(*[set(D[(y, s)]) for s in SEEDS])
        common = [i for i in (cx & cy) if i in pos]

        def md(i, k):
            return np.mean([D[(x, s)][i][k] for s in SEEDS]) - np.mean([D[(y, s)][i][k] for s in SEEDS])
        r1 = sorted([(pos[i][0], pos[i][1], md(i, 0)) for i in common])
        r2 = sorted([(pos[i][0], pos[i][1], md(i, 1)) for i in common])
        across_seed[pk] = {"GT1": summ_pair(r1, L_GT1, RTK), "GT2": summ_pair(r2, L_GT2, g2f),
                           "GT2_strict": summ_pair(r2, L_GT2_strict, g2f)}   # 0.05-thr robustness (F013)

        # per-seed: X_s vs Y_s (sign consistency across seeds)
        per_seed[pk] = {}
        s1, s2 = [], []
        for s in SEEDS:
            dx, dy = D[(x, s)], D[(y, s)]
            comm = [i for i in (set(dx) & set(dy)) if i in pos]
            p1 = sorted([(pos[i][0], pos[i][1], dx[i][0] - dy[i][0]) for i in comm])
            p2 = sorted([(pos[i][0], pos[i][1], dx[i][1] - dy[i][1]) for i in comm])
            g1, g2 = summ_pair(p1, L_GT1, RTK), summ_pair(p2, L_GT2, g2f)
            per_seed[pk][f"s{s}"] = {"GT1": g1, "GT2": g2}
            s1.append(g1["sign"])
            s2.append(g2["sign"])
        sign_consistency[pk] = {"GT1_signs": s1, "GT1_consistent": len(set(s1)) == 1,
                                "GT2_signs": s2, "GT2_consistent": len(set(s2)) == 1}

    report = {"config": {"bag": B["bag"], "paired_on": "both-arms two_row (whole-bag eligible)",
                         "block_boot_iters": BLOCK_BOOT, "block_lengths": bl,
                         "block_lengths_strict_gt2": bl_strict,
                         "rtk_floor_m": RTK, "gt2_floor_deg": G2FLOOR,
                         "point_estimate": "all both-two-row frames (across-seed per-frame mean)"},
              "across_seed": across_seed, "per_seed": per_seed, "sign_consistency": sign_consistency}
    OUT.write_text(json.dumps(report, indent=2))

    print(f"[{B['bag']}] whole-bag paired cross-arm; block lengths L_GT1={L_GT1} L_GT2={L_GT2}")
    print("\n=== across-seed (point mean d [moving-block 95% CI]; %=of floor) ===")
    for pk, a in across_seed.items():
        g1, g2 = a["GT1"], a["GT2"]
        print(f"{pk} (n={g1['n']}): "
              f"GT1 d={g1['mean_diff'] * 1000:+.1f} mm ({g1['mean_diff_pct_floor']:+}% RTK) "
              f"CI{[round(c * 1000, 1) for c in g1['ci95']]} mm excl0={g1['ci_excludes_zero']}")
        print(f"      GT2 d={g2['mean_diff']:+.3f} deg ({g2['mean_diff_pct_floor']:+}% floor) "
              f"CI{g2['ci95']} deg excl0={g2['ci_excludes_zero']}")
    print("\n=== sign consistency across seeds ===")
    for pk, v in sign_consistency.items():
        print(f"{pk}: GT1 {v['GT1_signs']} consistent={v['GT1_consistent']} | "
              f"GT2 {v['GT2_signs']} consistent={v['GT2_consistent']}")
    gt2_excl = [pk for pk, a in across_seed.items() if a["GT2"]["ci_excludes_zero"]]
    gt2_excl_strict = [pk for pk, a in across_seed.items() if a["GT2_strict"]["ci_excludes_zero"]]
    print(f"\nF019 check -- GT-2 pairs whose pooled CI excludes zero:")
    print(f"  primary  (L_GT2={L_GT2}, 0.1 thr): {gt2_excl if gt2_excl else 'NONE'}")
    print(f"  stricter (L_GT2={L_GT2_strict}, 0.05 thr, conservative): "
          f"{gt2_excl_strict if gt2_excl_strict else 'NONE (micro-difference does NOT persist)'}")
    print(f"wrote {OUT}")


def config_analysis(B):
    """Whole-bag Phase C downstream config analysis (D040, F018) -> config_analysis.json."""
    DETS = [B["detections"]]
    PF = B["per_frame_csv"]
    OUT = B["config"]
    SEEDS = [42, 43, 44]
    T_GRID = [1, 2, 3, 5, 8, 12]
    CONFIGS = ["agnostic", "trunk", "pole"]
    BOOT = 10000
    VIABLE_COV = 70.0   # F018 viable-regime coverage floor (%): argmin/tie-break evaluated only over
                        # cells at/above this. Sub-viable pole cells (pole_T1/T2/T3 ~1% = single-frame
                        # artefacts n_2r~1; pole_T8 ~46% = survivorship on easy frames) are NOT
                        # comparable RMS estimates and are excluded, per F018's locked definition.

    MAN = json.load(open(B["manifest"]))
    pos = {}
    byp = collections.defaultdict(list)
    for f in MAN["frames"]:
        if f["eligible"]:                       # whole-bag: eligible only, no split key
            byp[f["pass_id"]].append(f)
    for pid, fs in byp.items():
        fs = sorted(fs, key=lambda f: f["i"])
        xy = np.array([[f["x"], f["y"]] for f in fs])
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        for f, d in zip(fs, cum):
            pos[f["i"]] = (pid, float(d))

    det = collections.defaultdict(list)   # (seed,frame) -> [(cls,uc,v)]
    for path in DETS:
        for ln in Path(path).read_text().splitlines()[1:]:
            s, fr, c, u, v = ln.split(",")
            det[(int(s), int(fr))].append((int(c), float(u), float(v)))
    FRAMES = sorted(set(fr for (_, fr) in det))

    bl = BL.pooled_block_lengths(PF, MAN)
    L_GT1, L_GT2 = bl["L_GT1"], bl["L_GT2"]
    _ci_warn(B["bag"], bl)

    def select(dets, config, T):
        out = []
        for isL in (True, False):
            sd = [(c, u, v) for (c, u, v) in dets if (u < 320) == isL]
            tr = [(u, v) for (c, u, v) in sd if c == 0]
            po = [(u, v) for (c, u, v) in sd if c == 1]
            out += (tr + po) if config == "agnostic" else \
                   (tr if len(tr) >= T else tr + po) if config == "trunk" else \
                   (po if len(po) >= T else po + tr)
        return out

    def estimate(base_pts):
        L, R = [], []
        for (uc, v) in base_pts:
            g = C.project_px(uc, v, near_m=FARMAX)
            if g is not None:
                (L if uc < 320 else R).append(g)
        L = np.array(L) if L else np.empty((0, 2))
        R = np.array(R) if R else np.empty((0, 2))
        fL, fR = fit_side_far(L), fit_side_far(R)
        if fL["ok"] and fR["ok"]:
            cl = centre_linefit(L[fL["inl"]], R[fR["inl"]])
            if cl:
                return ("two_row", cl["offset"], cl["heading"], len(base_pts))
        return ("single_row" if (fL["ok"] or fR["ok"]) else "none", None, None, len(base_pts))

    def block_rms_ci(rows, L):
        bp = collections.defaultdict(list)
        for r in rows:
            bp[r[0]].append(r[2])
        ss, ln = [], []
        for v in bp.values():
            v = np.array(v)
            for st in range(0, len(v) - L + 1):
                ss.append(np.sum(v[st:st + L] ** 2))
                ln.append(L)
        if len(ss) < 8:
            return [None, None]
        ss, ln = np.array(ss), np.array(ln)
        N = sum(len(v) for v in bp.values())
        nb = int(np.ceil(N / L))
        rng = np.random.default_rng(42)
        r = [np.sqrt(ss[idx].sum() / ln[idx].sum()) for idx in (rng.integers(0, len(ss), nb) for _ in range(BOOT))]
        return [round(float(np.percentile(r, 2.5)), 4), round(float(np.percentile(r, 97.5)), 4)]

    def rms(a):
        a = np.asarray(a, float)
        return float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")

    def run_cell(selector):
        """selector(dets) -> base_pts. Returns the whole-bag across-seed cell record."""
        cov = collections.Counter()
        offs_f, hdgs_f, nb = {}, {}, []
        for fr in FRAMES:
            os_, hs = [], []
            for s in SEEDS:
                cls, o, h, n = estimate(selector(det.get((s, fr), [])))
                nb.append(n)
                if cls == "two_row":
                    os_.append(o)
                    hs.append(h)
                cov[(s, cls)] += 1
            if len(os_) == 3:
                offs_f[fr] = np.mean(os_)
                hdgs_f[fr] = np.mean(hs)
        tot = 3 * len(FRAMES)
        orow = sorted([(pos[fr][0], pos[fr][1], offs_f[fr]) for fr in offs_f if fr in pos])
        hrow = sorted([(pos[fr][0], pos[fr][1], hdgs_f[fr]) for fr in hdgs_f if fr in pos])
        return {"two_row_pct": round(100 * sum(v for (s, c), v in cov.items() if c == "two_row") / tot, 1),
                "single_pct": round(100 * sum(v for (s, c), v in cov.items() if c == "single_row") / tot, 1),
                "none_pct": round(100 * sum(v for (s, c), v in cov.items() if c == "none") / tot, 1),
                "mean_base": round(float(np.mean(nb)), 1),
                "gt1_rms": round(rms([r[2] for r in orow]), 4), "gt1_ci": block_rms_ci(orow, L_GT1),
                "gt2_rms": round(rms([r[2] for r in hrow]), 3), "gt2_ci": block_rms_ci(hrow, L_GT2),
                "n_two_row_frames": len(offs_f)}

    report = {}
    sweep_keys = []
    for config in CONFIGS:
        for T in ([1] if config == "agnostic" else T_GRID):   # agnostic is T-invariant
            key = f"{config}" + ("" if config == "agnostic" else f"_T{T}")
            report[key] = run_cell(lambda dets, config=config, T=T: select(dets, config, T))
            r = report[key]
            sweep_keys.append(key)
            print(f"{key:>14}: 2r {r['two_row_pct']}% base {r['mean_base']} | "
                  f"GT1 RMS {r['gt1_rms']} CI{r['gt1_ci']} | GT2 RMS {r['gt2_rms']} CI{r['gt2_ci']}", flush=True)

    for name, keep in [("trunk_only", 0), ("pole_only", 1)]:
        report[name] = run_cell(lambda dets, keep=keep: [(u, v) for (c, u, v) in dets if c == keep])
        r = report[name]
        print(f"{name:>14}: 2r {r['two_row_pct']}% (single {r['single_pct']}% none {r['none_pct']}%) "
              f"base {r['mean_base']} | GT1 RMS {r['gt1_rms']} CI{r['gt1_ci']} | GT2 RMS {r['gt2_rms']} CI{r['gt2_ci']} "
              f"| n_2r {r['n_two_row_frames']}", flush=True)

    # ---- argmin over the VIABLE regime (coverage >= VIABLE_COV) + pre-stated tie-break ---------
    # F018 defines the design choice over the viable regime (coverage >= 70%). Sub-viable cells are NOT
    # comparable RMS estimates: pole_T1/T2/T3 give ~1% two_row (n_2r ~ 1, single-frame artefacts), and
    # lower-coverage cells (e.g. pole_T8 ~46%) buy a lower RMS by estimating only on easy frames
    # (survivorship). Restricting the argmin/tie-break to viable cells matches F018's locked definition.
    def overlaps(a, b):
        return a[0] is not None and b[0] is not None and a[0] <= b[1] and b[0] <= a[1]

    agn = report["agnostic"]
    viable_keys = [k for k in sweep_keys if report[k]["two_row_pct"] >= VIABLE_COV]
    excluded = [{"cell": k, "two_row_pct": report[k]["two_row_pct"]}
                for k in sweep_keys if report[k]["two_row_pct"] < VIABLE_COV]
    argmin_summary = {}
    print(f"\n=== argmin over viable regime (coverage >= {VIABLE_COV}%) + tie-break (overlap agnostic -> lock agnostic) ===")
    print(f"viable cells: {viable_keys}")
    print(f"excluded (sub-viable): {[(e['cell'], e['two_row_pct']) for e in excluded]}")
    if not viable_keys:
        # Canopy bags can leave NO cell at the VIABLE_COV floor (May tops out ~63%). F018's viable-
        # regime argmin/tie-break is then undefined; record the honest result instead of crashing.
        # This does NOT change F018's 70% definition — it only handles the empty result of that
        # definition (no viable downstream config regime for this bag).
        max_two_row = max((report[k]["two_row_pct"] for k in sweep_keys), default=0.0)
        argmin_summary["viable_regime"] = {
            "coverage_floor_pct": VIABLE_COV, "any_viable": False,
            "max_two_row_pct": round(max_two_row, 1), "viable_cells": [], "excluded_cells": excluded,
            "note": (f"no config cell reaches the {VIABLE_COV:.0f}% two_row coverage floor on this bag "
                     f"(max {max_two_row:.1f}%); F018 viable-regime argmin undefined — no viable "
                     f"downstream config regime for this bag.")}
        print(f"NO VIABLE CELL: max two_row {max_two_row:.1f}% < {VIABLE_COV:.0f}% floor -> "
              f"argmin/tie-break skipped (no viable downstream regime).")
    else:
        for metric, rk, ck in [("GT1", "gt1_rms", "gt1_ci"), ("GT2", "gt2_rms", "gt2_ci")]:
            best = min(viable_keys, key=lambda k: report[k][rk])
            ov = overlaps(report[best][ck], agn[ck])
            lock = "agnostic" if ov else best
            argmin_summary[metric] = {"argmin": best, "argmin_rms": report[best][rk],
                                      "overlaps_agnostic": ov, "locked": lock}
            print(f"{metric}: argmin={best} (RMS {report[best][rk]}); CI overlaps agnostic={ov} -> LOCK {lock}")

        vci1 = [report[k]["gt1_ci"] for k in viable_keys]
        vci2 = [report[k]["gt2_ci"] for k in viable_keys]
        flat1 = all(overlaps(a, b) for a in vci1 for b in vci1)
        flat2 = all(overlaps(a, b) for a in vci2 for b in vci2)
        agnostic_locked = all(v["locked"] == "agnostic" for v in argmin_summary.values())
        argmin_summary["viable_regime"] = {
            "coverage_floor_pct": VIABLE_COV, "viable_cells": viable_keys, "excluded_cells": excluded,
            "rationale": "F018 viable-regime definition: argmin/tie-break evaluated only over cells with "
                         "two_row coverage >= 70%. Sub-viable pole cells are single-frame artefacts "
                         "(pole_T1/T2/T3, n_2r ~ 1) or survivorship on easy frames (pole_T8 ~46%), not "
                         "comparable RMS estimates."}
        argmin_summary["flat_gt1_viable_all_overlap"] = flat1
        argmin_summary["flat_gt2_viable_all_overlap"] = flat2
        argmin_summary["agnostic_locked"] = agnostic_locked
        argmin_summary["pause_flag"] = not agnostic_locked
        print(f"\nFLAT (viable)? GT-1 all viable cells CI-overlap: {flat1} | GT-2: {flat2}")
        print(f"agnostic_locked={agnostic_locked} pause_flag={not agnostic_locked}")

    out = {"config": {"bag": B["bag"], "n_frames": len(FRAMES), "seeds": SEEDS, "T_grid": T_GRID,
                      "upstream": "line-fit locked (D035-D038); conf 0.25, 15% blob guard",
                      "boot_iters": BOOT, "block_lengths": bl,
                      "note": "class-agnostic locked (F018); re-reported per bag, not re-selected"},
           "cells": report, "argmin_summary": argmin_summary}
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


def single_row_analysis(B):
    """In-row ABSTENTION analysis (F024) -> single_row_analysis.json. Re-runs the front-end on the
    single_row frames to recover why the second row side was rejected (mirrors line_fit_infer.py)."""
    import cv2
    import cuda_preload  # noqa: F401 — cuDNN cold-init guard; MUST precede torch (D049)
    import torch
    from ultralytics import YOLO
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from scripts.perception.segmentation.unet_binary.model import UNetBinary
    from scripts.perception.segmentation.unet_binary.dataset import IMAGENET_MEAN, IMAGENET_STD
    from cp3_geometry import CONF, BLOB_FRAC, FRAME_PX

    BAG = B["bag"]
    FR = B["frames_dir"]
    OUT = B["out_dir"] / "single_row_analysis.json"
    UNET_MIN_AREA = 40
    MODELS = [                                                     # identical to line_fit_infer.py
        ("A", 42, "unet", "phase_a_unet_binary_20260704_004105/checkpoints/best.pt"),
        ("A", 43, "unet", "phase_a_unet_binary_seed43_20260710_154347/checkpoints/best.pt"),
        ("A", 44, "unet", "phase_a_unet_binary_seed44_20260710_181339/checkpoints/best.pt"),
        ("B", 42, "yolo", "phase_b_yolo_binary/weights/best.pt"),
        ("B", 43, "yolo", "phase_b_yolo_binary_seed43/weights/best.pt"),
        ("B", 44, "yolo", "phase_b_yolo_binary_seed44/weights/best.pt"),
        ("C", 42, "yolo", "phase_c_yolo_multiclass/weights/best.pt"),
        ("C", 43, "yolo", "phase_c_yolo_multiclass_seed43/weights/best.pt"),
        ("C", 44, "yolo", "phase_c_yolo_multiclass_seed44/weights/best.pt"),
    ]
    _TF = A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
    dev = torch.device("cuda")

    # --- class mix + single_row frame lists straight from the committed CSV (authority) ---
    cls_ct = {(a, s): collections.Counter() for (a, s, _, _) in MODELS}
    sr_frames = {(a, s): [] for (a, s, _, _) in MODELS}
    for ln in Path(B["per_frame_csv"]).read_text().splitlines()[1:]:
        a, s, i, cls, *_ = ln.split(","); k = (a, int(s))
        if k not in cls_ct: continue
        cls_ct[k][cls] += 1
        if cls == "single_row": sr_frames[k].append(int(i))

    def yolo_base(model, img):
        r = model.predict(source=img, conf=CONF, quantize=16, device=0, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0: return []
        xy = r.boxes.xyxy.cpu().numpy()
        ar = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
        return [((x1 + x2) / 2, y2) for (x1, y1, x2, y2) in xy[ar <= BLOB_FRAC * FRAME_PX * FRAME_PX]]

    def unet_base(unet, img):
        x = _TF(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"].unsqueeze(0).to(dev)
        with torch.no_grad(): fg = (unet(x).argmax(1)[0].cpu().numpy() == 1).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(fg, 8)
        return [(st[k][0] + st[k][2] / 2., st[k][1] + st[k][3] - 1) for k in range(1, n) if st[k][4] >= UNET_MIN_AREA]

    # --- re-run the front-end on the single_row frames to recover the abstention mechanism ---
    mech = {a: {"n": 0, "not_reproduced": 0, "reason": collections.Counter(),
                "seen": collections.Counter(), "side": collections.Counter()} for a in "ABC"}
    for (arm, seed, typ, ckpt) in MODELS:
        frames = sr_frames[(arm, seed)]
        print(f"[{BAG}][{arm} s{seed}] {len(frames)} single_row frames ...", flush=True)
        if typ == "yolo":
            m = YOLO(str(PKG / "results/runs" / ckpt)); front = lambda im: yolo_base(m, im)
        else:
            m = UNetBinary(encoder_weights=None).to(dev).eval()
            m.load_state_dict(torch.load(PKG / "results/runs" / ckpt, map_location=dev, weights_only=False)["model_state_dict"])
            front = lambda im: unet_base(m, im)
        d = mech[arm]
        for fi in frames:
            img = cv2.imread(str(FR / f"{fi:05d}.jpg"))
            if img is None: continue
            L, R = [], []
            for (uc, v) in front(img):
                g = C.project_px(uc, v, near_m=FARMAX)
                if g is not None: (L if uc < 320 else R).append(g)
            L = np.array(L) if L else np.empty((0, 2)); R = np.array(R) if R else np.empty((0, 2))
            fL, fR = fit_side_far(L), fit_side_far(R)
            if fL["ok"] == fR["ok"]:                       # did not reproduce single_row (nondeterminism/edge)
                d["not_reproduced"] += 1; continue
            d["n"] += 1
            bad, P = (fR, R) if fL["ok"] else (fL, L)
            d["side"]["L_fit" if fL["ok"] else "R_fit"] += 1
            d["reason"][bad.get("reason", "?")] += 1
            nraw = len(P); nnear = int((P[:, 0] < NEAR).sum()) if nraw else 0
            d["seen"]["not_seen" if nraw == 0 else ("seen_far_only" if nnear < 2 else "seen_near")] += 1
        del m; torch.cuda.empty_cache()

    # --- aggregate: class mix per-arm mean+/-SD across seeds; mechanism per-arm over all single_row frames ---
    def pa_meansd(arm, cls):
        vals = [100 * cls_ct[(arm, s)][cls] / sum(cls_ct[(arm, s)].values()) for s in (42, 43, 44)]
        return [round(float(np.mean(vals)), 1), round(float(np.std(vals)), 2)]

    def pct(counter):
        t = sum(counter.values()) or 1
        return {k: round(100 * v / t, 1) for k, v in counter.most_common()}

    report = {
        "config": {"bag": BAG, "near_seed_window_m": NEAR, "far_max_m": FARMAX,
                   "note": "F024 in-row abstention. class_mix = per-arm mean+/-SD across seeds 42/43/44 from the committed CSV. "
                           "mechanism = failing-side rejection reason from fit_side_far, re-run on every single_row frame (all 9 models). "
                           "single_row emits NO centreline; D-G tier-2 half-spacing fallback (SPEC §10) specified but NOT implemented."},
        "class_mix_pct": {a: {c: pa_meansd(a, c) for c in ("two_row", "single_row", "none", "fitfail")} for a in "ABC"},
        "single_row_mechanism": {a: {
            "n_frames_reprocessed": mech[a]["n"], "not_reproduced": mech[a]["not_reproduced"],
            "failing_side_reason_pct": pct(mech[a]["reason"]),
            "failing_row_detected_pct": round(100 * (mech[a]["seen"]["seen_far_only"] + mech[a]["seen"]["seen_near"]) / (mech[a]["n"] or 1), 1),
            "seen_breakdown_pct": pct(mech[a]["seen"]),
            "fit_side": dict(mech[a]["side"])} for a in "ABC"},
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT}")
    for a in "ABC":
        cm = report["class_mix_pct"][a]; me = report["single_row_mechanism"][a]
        print(f"  {a}: single_row {cm['single_row'][0]}±{cm['single_row'][1]}%  "
              f"reason {me['failing_side_reason_pct']}  detected {me['failing_row_detected_pct']}%")


def lidar_crosscheck(B):
    """Whole-bag LiDAR cross-check (D040, F017) -> lidar_crosscheck.json. LiDAR row heading (Ouster
    PC2, trunk band, base_link) vs the camera line-fit heading on mid-pass anchors per corridor."""
    import sqlite3
    from rosbags.typesys import get_typestore, Stores

    ts = get_typestore(Stores.ROS2_HUMBLE)

    BAG = str(B["db3"])
    con = sqlite3.connect(BAG)
    cur = con.cursor()
    MAN = json.load(open(B["manifest"]))
    TS = {f["i"]: int(f["timestamp_ns"]) for f in MAN["frames"]}
    CORR = {f["i"]: f["corridor"] for f in MAN["frames"]}
    PF = str(B["per_frame_csv"])
    OUT = B["lidar"]
    PASS = {f["i"]: f["pass_id"] for f in MAN["frames"]}
    CORRIDORS = sorted(set(f["corridor"] for f in MAN["frames"] if f["eligible"]))   # all eligible corridors
    PER_CORR = 2                  # mid-pass anchors per corridor
    PC2_TOPIC = "/os_cloud_node/points"     # Ouster PointCloud2; resolved by NAME (topic ids are per-bag)
    PC2_TOPIC_ID = cur.execute("SELECT id FROM topics WHERE name=?", (PC2_TOPIC,)).fetchone()[0]

    # camera heading per frame = mean across 9 models; anchors = seed-42 arm-C two_row frames
    camh = collections.defaultdict(list)
    c42 = set()
    for ln in open(PF).read().splitlines()[1:]:
        a, s, i, cls, off, hdg, *_ = ln.split(",")   # 12-col whole-bag CSV; only heading used here
        if cls == "two_row" and hdg:
            camh[int(i)].append(float(hdg))
            if a == "C" and s == "42":
                c42.add(int(i))
    # Anchors: PER_CORR frames at the TRUE midpoint of a single traversal, one traversal per corridor.
    # A corridor is usually driven several times, so the corridor's frame list is a CONCATENATION of
    # passes; indexing into that concatenation (the previous `fs[len(fs)//3]`) does not give a mid-pass
    # frame and can land near a row EXIT, where both sensors degrade — LiDAR returns thin out and the
    # camera row-fit becomes unstable (cross-model heading SD roughly doubles past 90% of a pass). That
    # produced two spurious camera sign-flips per bag in both march and april. Group by pass first, take
    # the corridor's LONGEST traversal (most representative), and sample its true middle.
    anchors = []
    for cc in CORRIDORS:
        by_pass = collections.defaultdict(list)
        for i in sorted(i for i in c42 if CORR.get(i) == cc):
            by_pass[PASS[i]].append(i)
        if not by_pass:
            continue
        fs = max(by_pass.values(), key=len)                    # longest single traversal of this corridor
        mid = max(0, (len(fs) - PER_CORR) // 2)
        anchors += fs[mid: mid + PER_CORR]                     # true mid-pass frames

    ids = cur.execute("SELECT id,timestamp FROM messages WHERE topic_id=?", (PC2_TOPIC_ID,)).fetchall()
    ids_ts = np.array([t for _, t in ids])
    ids_id = np.array([i for i, _ in ids])

    def load_cloud(t):
        j = int(np.argmin(np.abs(ids_ts - t)))
        blob = cur.execute("SELECT data FROM messages WHERE id=?", (int(ids_id[j]),)).fetchone()[0]
        pc = ts.deserialize_cdr(blob, "sensor_msgs/msg/PointCloud2")
        off = {f.name: f.offset for f in pc.fields}
        buf = np.frombuffer(pc.data, dtype=np.uint8).reshape(-1, pc.point_step)
        c = lambda n: buf[:, off[n]:off[n] + 4].copy().view(np.float32).ravel()
        x, y, z = c("x"), c("y"), c("z")
        g = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & ((x != 0) | (y != 0))
        return x[g] - 0.098, y[g], z[g] + 1.0     # base_link (Table 3 identity extrinsic)

    def fit_side(P):
        """P: Nx2 (X,Y). densest-Y cluster in near field, robust line Y=mX+c. Return (m,c,n) or None."""
        if len(P) < 5:
            return None
        Y = P[:, 1]
        seed = np.median(Y[np.abs(Y - Y[np.argmax([np.sum(np.abs(Y - yy) <= 0.25) for yy in Y])]) <= 0.25])
        inl = np.abs(Y - seed) < 0.6
        Q = P[inl]
        if len(Q) < 5:
            return None
        m, c = np.polyfit(Q[:, 0], Q[:, 1], 1)
        for _ in range(2):
            r = np.abs(Q[:, 1] - (m * Q[:, 0] + c)) < 0.3
            if r.sum() < 5:
                break
            m, c = np.polyfit(Q[r, 0], Q[r, 1], 1)
        return float(m), float(c), int(len(Q))

    print(f"[{B['bag']}] {'frame':>6}{'corr':>5}{'L/R':>11}{'LiDARhdg':>10}{'CAMhdg':>9}{'diff':>8}")
    results, lid, cam = [], [], []
    for fi in anchors:
        xb, yb, zb = load_cloud(TS[fi])
        band = (zb > 0.2) & (zb < 1.2) & (xb > 1) & (xb < 8) & (np.abs(yb) < 2.6)
        P = np.column_stack([xb[band], yb[band]])
        L = P[P[:, 1] > 0.3]
        R = P[P[:, 1] < -0.3]
        fL, fR = fit_side(L), fit_side(R)
        if not (fL and fR):
            print(f"{fi:>6}{CORR[fi]:>5}   fit failed (L={len(L)},R={len(R)})")
            results.append({"frame": fi, "corridor": CORR[fi], "fit": "failed", "nL": len(L), "nR": len(R)})
            continue
        mc = (fL[0] + fR[0]) / 2
        lhdg = float(np.degrees(np.arctan(mc)))
        chdg = float(np.mean(camh[fi])) if camh[fi] else float("nan")
        print(f"{fi:>6}{CORR[fi]:>5}{f'{len(L)}/{len(R)}':>11}{lhdg:>10.2f}{chdg:>9.2f}{lhdg - chdg:>8.2f}")
        results.append({"frame": fi, "corridor": CORR[fi], "nL": len(L), "nR": len(R),
                        "mL": round(fL[0], 4), "mR": round(fR[0], 4),
                        "lidar_hdg": round(lhdg, 2), "cam_hdg": round(chdg, 2), "diff": round(lhdg - chdg, 2)})
        lid.append(lhdg)
        cam.append(chdg)

    report = {"config": {"bag": B["bag"], "anchors": anchors, "per_corridor": PER_CORR, "corridors": CORRIDORS,
                         "band": "z 0.2-1.2m, x 1-8m, |y|<2.6m",
                         "extrinsic": "Table 3 identity (x-0.098, z+1.0)",
                         "camera_heading": "mean across 9 models (tilt arm-independent)"},
              "anchors": results,
              "mean_lidar_hdg": round(float(np.mean(lid)), 2) if lid else None,
              "sd_lidar_hdg": round(float(np.std(lid)), 2) if lid else None,
              "mean_cam_hdg": round(float(np.mean(cam)), 2) if cam else None,
              "camera_minus_lidar": round(float(np.mean(cam) - np.mean(lid)), 2) if lid else None,
              "n_fitted": len(lid)}
    OUT.write_text(json.dumps(report, indent=2))

    if lid:
        print(f"\nmean LiDAR heading {np.mean(lid):+.2f} deg (SD {np.std(lid):.2f}) | "
              f"mean CAMERA heading {np.mean(cam):+.2f} deg | camera-LiDAR {np.mean(cam) - np.mean(lid):+.2f} deg")
        print("(F017: both nonzero & agreeing in sign -> row tilt is sensor-common, not a camera artefact)")
    print(f"wrote {OUT}")


# ================================================================================================
# Non-in-row stratum
# ================================================================================================
def non_in_row_analysis(B):
    """Non-in-row deployment-gap characterisation (D041 category C; F020, F021).
    Expects B = resolve(bag, "non_in_row"). Writes non_in_row_analysis.json."""
    MAN = json.load(open(B["manifest"]))
    PF = B["per_frame_csv"]
    OUT = B["non_in_row_analysis"]
    SEEDS = [42, 43, 44]
    CATS = ("stationary", "turn", "transition")

    # --- categorise each non-in-row frame: stationary / turn / transition ----------------------
    frames = {f["i"]: f for f in MAN["frames"]}
    elig_idx = sorted(i for i, f in frames.items() if f["eligible"])
    elig_corr = {i: frames[i]["corridor"] for i in elig_idx}

    def category(i):
        """stationary (headland+stationary); else turn (moving, same flanking corridor) vs
        transition (moving, different flanking corridor / bag edge)."""
        if frames[i]["stationary"]:
            return "stationary"
        pos = bisect.bisect_left(elig_idx, i)
        bc = elig_corr[elig_idx[pos - 1]] if pos > 0 else None
        ac = elig_corr[elig_idx[pos]] if pos < len(elig_idx) else None
        if bc is not None and ac is not None:
            return "turn" if bc == ac else "transition"
        return "transition"   # bag edge (no flanking pass on one side)

    cat = {i: category(i) for i in frames_for_scope(MAN, "non_in_row")}
    catcount = collections.Counter(cat.values())

    # This analysis is defined over the NON-IN-ROW stratum (cat, above, is hardcoded to it), but PF is
    # resolved from --scope. A wrong scope points PF at the in-row CSV, whose frame indices are DISJOINT
    # from cat (D041) — every row is skipped and the result is empty/garbage. Fail loudly rather than
    # silently analyse the wrong file. (Sixth single-bag-shaped footgun; D046f.) In this driver main()
    # always passes resolve(bag, "non_in_row"), so the guard is a defensive invariant.
    if B["scope"] != "non_in_row":
        raise SystemExit(
            f"non_in_row_analysis requires the non_in_row bundle (got scope '{B['scope']}'): this "
            f"analysis is defined over the non-in-row stratum, but the bundle's scope resolves the "
            f"per-frame CSV path. Re-run with:  python3 scripts/geometric/analyze.py --bag "
            f"{B['bag']} --non-in-row")

    # --- parse the non-in-row per-frame CSV ----------------------------------------------------
    cls_by = collections.defaultdict(collections.Counter)      # (arm, cat) -> cls counts (all seeds)
    two = collections.defaultdict(list)                        # (arm, cat) -> [(offset, heading)] two_row
    for ln in PF.read_text().splitlines()[1:]:
        a, s, i, cls, off, hdg, *_ = ln.split(",")
        i = int(i)
        if i not in cat:
            continue
        c = cat[i]
        cls_by[(a, c)][cls] += 1
        cls_by[(a, "ALL")][cls] += 1
        if cls == "two_row" and off and hdg:
            two[(a, c)].append((float(off), float(hdg)))
            two[(a, "ALL")].append((float(off), float(hdg)))

    def dist(counter):
        tot = sum(counter.values())
        d = {k: round(100 * counter[k] / tot, 1) for k in ("two_row", "single_row", "none", "fitfail")}
        d["n"] = tot
        return d

    def rms(v):
        v = np.asarray(v, float)
        return round(float(np.sqrt(np.mean(v ** 2))), 3) if len(v) else None

    def dpe(lst):
        if not lst:
            return {"two_row_n": 0, "driven_path_error_rms_m": None, "driven_path_heading_rms_deg": None}
        return {"two_row_n": len(lst),
                "driven_path_error_rms_m": rms([o for o, h in lst]),
                "driven_path_heading_rms_deg": rms([h for o, h in lst])}

    F020 = {"category_counts": dict(catcount),
            "per_arm_overall": {a: dist(cls_by[(a, "ALL")]) for a in "ABC"},
            "per_category": {c: {a: dist(cls_by[(a, c)]) for a in "ABC"} for c in CATS}}
    F021 = {"metric_note": ("driven_path_error = RMS lateral offset of the IPM-invalid predicted "
                            "centreline vs base_link on non-in-row two_row outputs; NOT the in-row "
                            "centreline_error_rms and not comparable to it. Conflations: (1) flat-ground "
                            "IPM invalid on headland slopes; (2) row centreline undefined on turns; "
                            "(3) turn geometry conflates with the error."),
            "per_arm_overall": {a: dpe(two[(a, "ALL")]) for a in "ABC"},
            "per_category": {c: {a: dpe(two[(a, c)]) for a in "ABC"} for c in CATS}}

    out = {"config": {"bag": B["bag"], "scope": B["scope"], "n_frames": len(cat), "seeds": SEEDS,
                      "categories": "stationary (headland+stationary) / turn (moving, same flanking "
                                    "corridor) / transition (moving, different flanking corridor or bag edge)"},
           "F020_output_distribution": F020, "F021_driven_path_error": F021}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))

    print(f"[{B['bag']}/{B['scope']}] non-in-row frames {len(cat)} | categories {dict(catcount)}")
    print("\nF020 output-class distribution (per arm, all non-in-row; % of arm's 3-seed rows):")
    for a in "ABC":
        print(f"  {a}: {F020['per_arm_overall'][a]}")
    print("\nF021 driven-path error on two_row outputs (per arm, all non-in-row):")
    for a in "ABC":
        print(f"  {a}: {F021['per_arm_overall'][a]}")
    print(f"\nwrote {OUT}")


def mitigation_analysis(B_in, B_non):
    """Two-layer rejection for non-in-row spurious two_row outputs (F022 state gate + F023 geometry
    filter). Reads both strata. Writes mitigation_evaluation/mitigation_analysis.json."""
    BAG = B_in["bag"]
    MAN = json.load(open(B_in["manifest"]))
    OUT = B_in["out_dir"].parent / "mitigation_evaluation" / "mitigation_analysis.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)

    V_MIN, VY_INROW = 0.10, 0.30
    # F023 geometry thresholds (in-row p99; see derivation in the Commit-7 gate)
    G_OFF, G_HDG, G_PAR, G_NB = 0.71, 6.7, 0.22, 12

    # --- per-frame kinematics (odometry) -------------------------------------------------------
    fr = sorted(MAN["frames"], key=lambda f: f["i"])
    x = np.array([f["x"] for f in fr]); y = np.array([f["y"] for f in fr]); t = np.array([f["t_offset_s"] for f in fr])
    speed = np.array([f["speed"] for f in fr])
    elig = np.array([f["eligible"] for f in fr])
    vy = np.convolve(np.gradient(y, t), np.ones(15) / 15, mode="same")            # along-row velocity (as CP-1)
    vx = np.convolve(np.gradient(x, t), np.ones(15) / 15, mode="same")
    # heading-rate = angular change of the velocity direction (robust cross/dot form; atan2(vy,vx) is
    # unstable at vx~0 for corridors-along-Y). Masked where speed is too low for a defined direction.
    cr = vx[:-1] * vy[1:] - vy[:-1] * vx[1:]
    dt_ = np.maximum(np.diff(t), 1e-6)
    dang = np.abs(np.arctan2(cr, vx[:-1] * vx[1:] + vy[:-1] * vy[1:]))            # angle between consecutive v
    hr_raw = np.concatenate([[0.0], dang / dt_])
    hr_raw[np.hypot(vx, vy) < 0.05] = 0.0
    hr = np.convolve(hr_raw, np.ones(15) / 15, mode="same")                        # rad/s, smoothed
    HR_THRESH = float(np.percentile(hr[elig], 99))                                 # in-row p99 (turn threshold)

    # causal state gate: keep only if moving along-row and not turning
    gate_pass = (speed > V_MIN) & (np.abs(vy) > VY_INROW) & (hr < HR_THRESH)
    state_reject = {fr[k]["i"]: (not gate_pass[k]) for k in range(len(fr))}        # per-frame (causal)

    # --- categorise non-in-row frames ----------------------------------------------------------
    frames = {f["i"]: f for f in MAN["frames"]}
    elig_idx = sorted(i for i, f in frames.items() if f["eligible"]); elig_corr = {i: frames[i]["corridor"] for i in elig_idx}
    def category(i):
        if frames[i]["stationary"]: return "stationary"
        p = bisect.bisect_left(elig_idx, i); bc = elig_corr[elig_idx[p-1]] if p > 0 else None; ac = elig_corr[elig_idx[p]] if p < len(elig_idx) else None
        return ("turn" if bc == ac else "transition") if (bc is not None and ac is not None) else "transition"
    cat = {i: category(i) for i in frames_for_scope(MAN, "non_in_row")}

    # turn-phase: contiguous runs of turn frames -> centrality in [0,1] (0=either edge, 1=deep interior);
    # lets us test whether F023's turn rejections are the transitional edge frames vs the clean deep ones.
    turn_frames = sorted(i for i, c in cat.items() if c == "turn")
    runs = []
    if turn_frames:
        s = prev = turn_frames[0]
        for i in turn_frames[1:]:
            if i != prev + 1: runs.append((s, prev)); s = i
            prev = i
        runs.append((s, prev))
    centrality = {}
    for (aa, bb) in runs:
        L = bb - aa
        for i in range(aa, bb + 1):
            pos = 0.5 if L == 0 else (i - aa) / L
            centrality[i] = 1.0 - 2.0 * abs(pos - 0.5)
    def turn_phase(i):
        c = centrality.get(i, 0.5)
        return "edge" if c < 0.34 else ("deep" if c > 0.66 else "mid")

    # --- load two_row outputs from both strata -------------------------------------------------
    def load_two(csv):
        out = []   # (arm, i, offset, heading, mL, mR, n_base)
        for ln in Path(csv).read_text().splitlines()[1:]:
            a, s, i, cls, o, h, mL, mR, mc, n, ad, fl = ln.split(",")
            if cls == "two_row" and o and h and mL and mR:
                out.append((a, int(i), abs(float(o)), abs(float(h)), abs(float(mL) - float(mR)), int(n)))
        return out
    in_two = load_two(B_in["per_frame_csv"])       # in-row two_row (valid)
    non_two = load_two(B_non["per_frame_csv"])     # non-in-row two_row (spurious, F020)

    def geom_reject(off, hdg, par, nb):
        return off > G_OFF or hdg > G_HDG or par > G_PAR or nb < G_NB

    def geom_fires(off, hdg, par, nb):   # per-threshold breakdown (which of the 4 fired)
        return {"offset": off > G_OFF, "heading": hdg > G_HDG, "parallel": par > G_PAR, "n_base": nb < G_NB}

    def summarise(rows, is_non):
        """rows: (arm, i, off, hdg, par, nb). Returns per-arm + per-category F022/F023/combined rejection."""
        res = collections.defaultdict(lambda: {"n": 0, "f022": 0, "f023": 0, "either": 0})
        for (a, i, off, hdg, par, nb) in rows:
            keys = [("ALL", a)]
            if is_non:
                keys.append((cat[i], a))
            sr = state_reject[i]; gr = geom_reject(off, hdg, par, nb)
            for k in keys:
                d = res[k]; d["n"] += 1; d["f022"] += sr; d["f023"] += gr; d["either"] += (sr or gr)
        def pct(k):
            d = res[k]; n = d["n"] or 1
            return {"n": d["n"], "f022_%": round(100*d["f022"]/n, 1), "f023_%": round(100*d["f023"]/n, 1),
                    "either_%": round(100*d["either"]/n, 1)}
        return res, pct

    non_res, non_pct = summarise(non_two, True)
    in_res, in_pct = summarise(in_two, False)

    # F022 upper bound (oracle manifest-flag state gate): 100% of non-in-row are ~eligible; 0 in-row
    report = {
        "config": {"bag": BAG, "V_MIN": V_MIN, "VY_INROW": VY_INROW, "HR_THRESH_deg_s": round(np.degrees(HR_THRESH), 3),
                   "geom_thresholds": {"offset_m": G_OFF, "heading_deg": G_HDG, "parallelism": G_PAR, "n_base_min": G_NB},
                   "note": "F022 = odometry state gate (upper bound = manifest eligible flag; causal = per-frame speed/|v_y|/heading-rate). F023 = perception geometry filter (in-row p99). Rejection% is of two_row OUTPUTS."},
        "F022_upper_bound": {"non_in_row_rejection_%": 100.0, "in_row_FP_%": 0.0,
                             "note": "manifest `eligible` = the CP-1 criteria; rejects all non-in-row, keeps all in-row by construction (oracle state)."},
        "F022_F023_causal": {
            "non_in_row": {"per_arm": {a: non_pct(("ALL", a)) for a in "ABC"},
                           "per_category": {c: {a: non_pct((c, a)) for a in "ABC"} for c in ("stationary", "turn", "transition")}},
            "in_row_FP": {a: in_pct(("ALL", a)) for a in "ABC"}},
    }
    # --- F023 mechanism decomposition (why the turn ceiling is ~50%, why the in-row FP is sub-additive) ---
    def turn_mechanism(arm):
        rows = [(i, off, hdg, par, nb) for (a, i, off, hdg, par, nb) in non_two if a == arm and cat.get(i) == "turn"]
        ph = {p: {"n": 0, "rej": 0} for p in ("edge", "mid", "deep")}
        thr = {k: 0 for k in ("offset", "heading", "parallel", "n_base")}; rejN = 0
        for (i, off, hdg, par, nb) in rows:
            f = geom_fires(off, hdg, par, nb); rej = any(f.values()); p = turn_phase(i)
            ph[p]["n"] += 1; ph[p]["rej"] += rej
            if rej:
                rejN += 1
                for k in thr: thr[k] += f[k]
        n = len(rows) or 1
        return {"turn_two_row_n": len(rows), "rejection_%": round(100 * rejN / n, 1),
                "by_phase_rejection_%": {p: round(100 * ph[p]["rej"] / (ph[p]["n"] or 1), 1) for p in ph},
                "by_phase_n": {p: ph[p]["n"] for p in ph},
                "threshold_fire_%_among_rejections": {k: round(100 * thr[k] / (rejN or 1), 1) for k in thr}}

    def in_row_overlap(arm):
        rows = [(off, hdg, par, nb) for (a, i, off, hdg, par, nb) in in_two if a == arm]
        N = len(rows) or 1; marg = {k: 0 for k in ("offset", "heading", "parallel", "n_base")}; union = 0
        for (off, hdg, par, nb) in rows:
            f = geom_fires(off, hdg, par, nb)
            for k in marg: marg[k] += f[k]
            union += any(f.values())
        m = {k: round(100 * marg[k] / N, 2) for k in marg}; sm = round(sum(m.values()), 2); u = round(100 * union / N, 2)
        return {"in_row_two_row_n": len(rows), "marginal_FP_%": m, "sum_of_marginals_%": sm,
                "union_FP_%": u, "overlap_saved_pp": round(sm - u, 2)}

    report["F023_turn_mechanism"] = {
        "per_arm": {a: turn_mechanism(a) for a in "ABC"},
        "note": "F023 turn rejections decomposed by turn phase (edge/mid/deep interior of each contiguous turn run) and by which p99 threshold fires. Heading-dominated (offset ~secondary, parallelism/n_base ~0); graded edge>deep -> transitional turn frames caught, clean deep-turn frames pass to F022."}
    report["F023_in_row_threshold_overlap"] = {
        "per_arm": {a: in_row_overlap(a) for a in "ABC"},
        "note": "In-row FP per threshold (each ~p99=1%); union is sub-additive by ~0.5pp -> the four thresholds are near-independent (weak correlation)."}

    OUT.write_text(json.dumps(report, indent=2))

    print(f"[{BAG}] HR_THRESH (in-row p99) = {np.degrees(HR_THRESH):.2f} deg/s")
    print("\n=== NON-IN-ROW two_row rejection (per arm; f022 state / f023 geom / either) ===")
    for a in "ABC":
        print(f"  {a}: {non_pct(('ALL', a))}")
    print("  per category (either%):")
    for c in ("stationary", "turn", "transition"):
        print(f"    {c:>11}: " + " ".join(f"{a}={non_pct((c,a))['either_%']}%(f022={non_pct((c,a))['f022_%']} f023={non_pct((c,a))['f023_%']})" for a in "ABC"))
    print("\n=== IN-ROW FALSE POSITIVES (per arm; two_row wrongly rejected) ===")
    for a in "ABC":
        print(f"  {a}: {in_pct(('ALL', a))}")
    print(f"\nF022 upper bound: non-in-row 100.0% rejected / in-row 0.0% FP (oracle manifest state)")
    print(f"wrote {OUT}")


# ================================================================================================
# CLI
# ================================================================================================
IN_ROW_ORDER = ["line_fit_eval", "paired_crossarm", "config_analysis", "single_row_analysis", "lidar_crosscheck"]
NON_IN_ROW_ORDER = ["non_in_row_analysis", "mitigation"]
IN_ROW_FUNCS = {"line_fit_eval": line_fit_eval, "paired_crossarm": paired_crossarm,
                "config_analysis": config_analysis, "single_row_analysis": single_row_analysis,
                "lidar_crosscheck": lidar_crosscheck}


def main():
    ap = argparse.ArgumentParser(description="Consolidated geometric-evaluation analyses (replaces 7 scripts).")
    ap.add_argument("--bag", default="march", help="bag name (default: march)")
    ap.add_argument("--non-in-row", action="store_true",
                    help="run the non-in-row stratum (non_in_row_analysis + mitigation) instead of the 5 in-row analyses")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of analysis names within the selected stratum "
                         "(in-row: %s; non-in-row: %s)" % (",".join(IN_ROW_ORDER), ",".join(NON_IN_ROW_ORDER)))
    a = ap.parse_args()

    stratum_order = NON_IN_ROW_ORDER if a.non_in_row else IN_ROW_ORDER
    if a.only:
        requested = [x.strip() for x in a.only.split(",") if x.strip()]
        unknown = [x for x in requested if x not in stratum_order]
        if unknown:
            raise SystemExit(f"--only names not in the selected stratum {stratum_order}: {unknown}")
        selected = [x for x in stratum_order if x in requested]
    else:
        selected = stratum_order

    import traceback
    if a.non_in_row:
        B_in, B_non = resolve(a.bag, "eligible"), resolve(a.bag, "non_in_row")
        runners = {"non_in_row_analysis": lambda: non_in_row_analysis(B_non),
                   "mitigation": lambda: mitigation_analysis(B_in, B_non)}
    else:
        B = resolve(a.bag, "eligible")
        B["out_dir"].mkdir(parents=True, exist_ok=True)   # so an --only subset does not depend on run order
        runners = {n: (lambda n=n: IN_ROW_FUNCS[n](B)) for n in IN_ROW_ORDER}

    # Per-analysis isolation: one failing analysis must not abort the others (restores the independence
    # the pre-merge separate scripts had). Collect failures, report a summary, exit non-zero if any failed.
    failures = []
    for name in selected:
        try:
            runners[name]()
        except Exception:
            failures.append(name)
            print(f"\n!! analyze [{a.bag}] '{name}' FAILED — continuing with the rest:\n"
                  f"{traceback.format_exc()}", file=sys.stderr, flush=True)
    ok = [n for n in selected if n not in failures]
    print(f"\nanalyze [{a.bag}] summary: {len(ok)}/{len(selected)} ok"
          + (f"; FAILED: {failures}" if failures else ""))
    if failures:
        raise SystemExit(f"analyze: {len(failures)} analysis/analyses failed: {failures}")


if __name__ == "__main__":
    main()
