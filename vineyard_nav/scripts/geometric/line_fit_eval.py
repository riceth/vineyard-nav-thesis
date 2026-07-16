"""Whole-bag line-fit AGGREGATION (D040). Reads the per-frame CSV produced by line_fit_infer.py
(12-col, all 9 models x every eligible frame) and aggregates it into the report. No inference here;
no val/test split (frames are eligible-only). Bag-agnostic multi-bag template.

  python3 line_fit_eval.py --bag march   -> results/geometric/march/final/march_evaluation/line_fit_report.json

Consumes  final/{bag}_evaluation/line_fit_per_frame.csv  (arm,seed,i,cls,offset,heading,mL,mR,mc,n_base,adj,flags)
Produces  line_fit_report.json
    per_model : coverage, mean base, GT-1 RMS/mean, GT-2 RMS/mean, mc-mean + tilt(deg); per-model
                GT-1/GT-2 CI = Delta_s=1.5m subsample simple bootstrap (B=2000, seed 42).
    per_arm   : cross-seed mean+/-SD of (two_row%, GT1 RMS, GT2 RMS, base, tilt).
    per_arm_ci: across-seed per-arm GT-1/GT-2 RMS + moving-block bootstrap 95% CI at the whole-bag
                Analysis-H block lengths (block_lengths.py).
"""
import sys
import json
import collections
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
import block_lengths as BL
from bag_config import parse_bag

B = parse_bag()
MAN = json.load(open(B["manifest"]))
B["out_dir"].mkdir(parents=True, exist_ok=True)
IN_CSV = B["per_frame_csv"]           # 12-col, written by line_fit_infer.py
OUT_JSON = B["line_fit_report"]

SEEDS = [42, 43, 44]
BOOT_B = 2000          # per-model subsample simple bootstrap
BOOT_SEED = 42
BLOCK_BOOT = 10000     # per-arm moving-block bootstrap

# ---- parse the whole-bag per-frame CSV (12-col) -----------------------------------------------
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


# ---- per-model aggregation --------------------------------------------------------------------
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

# ---- per-arm cross-seed mean+/-SD -------------------------------------------------------------
agg = {}
for arm in "ABC":
    rs = [s for s in summaries if s["arm"] == arm]

    def ms(k, rs=rs):
        v = np.array([r[k] for r in rs], float)
        return [round(float(v.mean()), 3), round(float(v.std()), 3)]
    agg[arm] = {k: ms(k) for k in ("two_row_pct", "gt1_rms", "gt2_rms", "mean_base", "tilt_deg")}

# ---- per-arm across-seed moving-block bootstrap CIs (whole-bag Analysis-H block lengths) -------
bl = BL.pooled_block_lengths(IN_CSV, MAN)
L_GT1, L_GT2 = bl["L_GT1"], bl["L_GT2"]

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

# ---- write report -----------------------------------------------------------------------------
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

# ---- console summary --------------------------------------------------------------------------
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
