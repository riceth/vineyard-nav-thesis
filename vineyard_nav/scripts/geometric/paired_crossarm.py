"""Whole-bag paired cross-arm difference bootstrap (D040, F013). Bag-agnostic multi-bag template.

  python3 paired_crossarm.py --bag march   -> results/geometric/march/final/march_evaluation/paired_crossarm.json

For each arm pair (A-B, A-C, B-C), on frames where BOTH arms produced a two-row estimate,
per-frame paired differences dGT-1 = offset_X - offset_Y (m), dGT-2 = heading_X - heading_Y (deg):
  across_seed : per-frame mean over the 3 seeds of each arm (two-row in all 3 of both arms),
                point mean d + moving-block bootstrap 95% CI  (primary).
  per_seed    : X_s vs Y_s per seed, point mean d + block CI + sign  -> sign consistency.
Floor-relative %: GT-1 vs RTK-GNSS 3.8 cm (Polvara 2024 §5.3); GT-2 vs pair-avg regression-
residual floor (F012). ci_excludes_zero flags any pair whose CI clears zero.

Block lengths (L_GT1, L_GT2) are RE-DERIVED on the whole-bag data via block_lengths.py (Analysis-H,
2x decorrelation, conservative across pairs); GT-2 reported at both the primary (0.1) and stricter
(0.05) thresholds. Frames are eligible-only (no split); each pass is a single contiguous series, so
a moving block stays within one pass.
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

BLOCK_BOOT = 10000
SEEDS = [42, 43, 44]
PAIRS = [("A", "B"), ("A", "C"), ("B", "C")]
RTK = 0.038                                   # m, GT-1 floor (Polvara 2024 §5.3, March)
G2FLOOR = {"A": 1.33, "B": 1.36, "C": 1.29}   # deg, GT-2 regression-residual floor per arm (F012)

B = parse_bag()
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
