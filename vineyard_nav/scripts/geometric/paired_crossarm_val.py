"""Paired cross-arm difference bootstrap on val (line-fit CP-5).

For each arm pair (A-B, A-C, B-C), on frames where BOTH arms produced a two-row estimate
(the common line-fit centreline estimator — single-row/none excluded so GT-1 offset and GT-2
heading are both defined), compute per-frame paired differences:
  ΔGT-1 = offset_X - offset_Y  (m)   ΔGT-2 = heading_X - heading_Y  (deg)

Reported (dual-mode, D-D):
  - per seed pair: X_s42 vs Y_s42, s43, s44
  - across seeds:  per-frame mean over the 3 seeds of each arm, then differenced
  - point estimate = mean Δ over ALL both-two-row frames
  - 95% bootstrap CI (10,000 iters, seed 42) over the Δs = 1.5 m spatially-independent subsample
  - sign of the difference, whether the CI excludes zero, and sign consistency across seeds

Saves results/geometric/march/paired_crossarm_val.json. Val only; nothing committed.
"""
import json, collections
import numpy as np

from pathlib import Path
PKG = Path(__file__).resolve().parents[2]
BOOT, SEED = 10000, 42
SEEDS, PAIRS = [42, 43, 44], [("A", "B"), ("A", "C"), ("B", "C")]
RTK_FLOOR = 0.038                                  # m, GT-1 floor (Polvara 2024 §5.3, March)
G2_FLOOR = {"A": 1.33, "B": 1.36, "C": 1.29}       # deg, GT-2 regression-residual floor per arm (F012)

man = json.load(open(f"{PKG}/results/geometric/march/dataset_manifest.json"))
SUB = set(f["i"] for f in man["frames"] if f["split"] == "val" and f.get("subsample_1p5m"))

D = collections.defaultdict(dict)   # (arm,seed) -> {frame_i: (offset, heading)} for two_row
for ln in open(f"{PKG}/results/geometric/march/line_fit_val_per_frame.csv").read().splitlines()[1:]:
    a, s, i, cls, off, hdg, *_ = ln.split(",")
    if cls == "two_row" and off and hdg:
        D[(a, int(s))][int(i)] = (float(off), float(hdg))


def boot_ci(diffs, b=BOOT, seed=SEED):
    diffs = np.asarray(diffs, float)
    if len(diffs) < 8:
        return [None, None]
    rng = np.random.default_rng(seed); n = len(diffs)
    means = [diffs[rng.integers(0, n, n)].mean() for _ in range(b)]
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def summarise(d_all, d_sub, floor):
    pt = float(np.mean(d_all)) if len(d_all) else float("nan")
    ci = boot_ci(d_sub)
    excl = ci[0] is not None and (ci[0] > 0 or ci[1] < 0)
    pct = lambda v: None if v is None else round(100 * v / floor, 1)
    return {"mean_diff": round(pt, 4), "n_all": int(len(d_all)), "n_sub": int(len(d_sub)),
            "ci95_sub": ci, "sign": "+" if pt > 0 else "-" if pt < 0 else "0",
            "ci_excludes_zero": bool(excl),
            "floor": floor, "mean_diff_pct_floor": pct(pt), "ci95_pct_floor": [pct(ci[0]), pct(ci[1])]}


report = {"config": {"paired_on": "both-arms two_row", "boot_iters": BOOT, "boot_seed": SEED,
                     "ci_subsample": "Δs=1.5m", "n_subsample_total": len(SUB),
                     "point_estimate": "all both-two-row frames"},
          "per_seed": {}, "across_seed": {}, "sign_consistency": {}}

for (x, y) in PAIRS:
    pk = f"{x}-{y}"; g2f = (G2_FLOOR[x] + G2_FLOOR[y]) / 2   # pair-average GT-2 floor (F012)
    report["per_seed"][pk] = {}; s1, s2 = [], []
    for s in SEEDS:
        dx, dy = D[(x, s)], D[(y, s)]; common = set(dx) & set(dy)
        d1 = np.array([dx[i][0] - dy[i][0] for i in common]); d2 = np.array([dx[i][1] - dy[i][1] for i in common])
        sub = [i for i in common if i in SUB]
        d1s = np.array([dx[i][0] - dy[i][0] for i in sub]); d2s = np.array([dx[i][1] - dy[i][1] for i in sub])
        g1, g2 = summarise(d1, d1s, RTK_FLOOR), summarise(d2, d2s, g2f)
        report["per_seed"][pk][f"s{s}"] = {"GT1": g1, "GT2": g2}; s1.append(g1["sign"]); s2.append(g2["sign"])
    report["sign_consistency"][pk] = {"GT1_signs": s1, "GT1_consistent": len(set(s1)) == 1,
                                      "GT2_signs": s2, "GT2_consistent": len(set(s2)) == 1}
    # across-seed: per-frame mean over the 3 seeds of each arm (frame must be two_row in all 3 seeds of both arms)
    cx = set.intersection(*[set(D[(x, s)]) for s in SEEDS]); cy = set.intersection(*[set(D[(y, s)]) for s in SEEDS])
    common = cx & cy
    def md(i, k): return np.mean([D[(x, s)][i][k] for s in SEEDS]) - np.mean([D[(y, s)][i][k] for s in SEEDS])
    d1 = np.array([md(i, 0) for i in common]); d2 = np.array([md(i, 1) for i in common])
    sub = [i for i in common if i in SUB]
    d1s = np.array([md(i, 0) for i in sub]); d2s = np.array([md(i, 1) for i in sub])
    report["across_seed"][pk] = {"GT1": summarise(d1, d1s, RTK_FLOOR), "GT2": summarise(d2, d2s, g2f)}

json.dump(report, open(f"{PKG}/results/geometric/march/paired_crossarm_val.json", "w"), indent=2)

print("=== paired cross-arm, across-seed (point mean Δ [95% CI on Δs=1.5m subsample]; %=of floor) ===")
for pk, a in report["across_seed"].items():
    g1, g2 = a["GT1"], a["GT2"]
    print(f"{pk}: GT1 Δ={g1['mean_diff']*1000:+.1f} mm ({g1['mean_diff_pct_floor']:+}% RTK) "
          f"CI{[round(c*1000,1) for c in g1['ci95_sub']]} mm = {g1['ci95_pct_floor']}% RTK excl0={g1['ci_excludes_zero']} "
          f"(n_all={g1['n_all']}, n_sub={g1['n_sub']})")
    print(f"      GT2 Δ={g2['mean_diff']:+.3f}° ({g2['mean_diff_pct_floor']:+}% floor) "
          f"CI{g2['ci95_sub']}° = {g2['ci95_pct_floor']}% floor excl0={g2['ci_excludes_zero']}")
print("\n=== sign consistency across seeds (per-seed pairs) ===")
for pk, v in report["sign_consistency"].items():
    print(f"{pk}: GT1 {v['GT1_signs']} consistent={v['GT1_consistent']} | GT2 {v['GT2_signs']} consistent={v['GT2_consistent']}")
print("\nwrote results/geometric/march/paired_crossarm_val.json")
