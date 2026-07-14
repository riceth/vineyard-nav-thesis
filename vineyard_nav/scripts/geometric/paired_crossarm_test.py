"""CP-6 test-side block-bootstrap CIs + cross-arm paired analysis (F013 methodology,
Analysis-H block lengths). Per-arm GT-1/GT-2 RMS + block CI; per-pair paired Δ + block CI +
floor-relative %. Across-seed (per-frame mean over 3 seeds, two-row in all 3)."""
import json, collections
import numpy as np
from pathlib import Path
PKG = Path(__file__).resolve().parents[2]
BOOT = 10000; L_GT1, L_GT2 = 11, 31; SEEDS = [42, 43, 44]
RTK, G2FLOOR = 0.038, {"A": 1.33, "B": 1.36, "C": 1.29}

man = json.load(open(f"{PKG}/results/geometric/march/dataset_manifest.json"))
pos = {}; byp = collections.defaultdict(list)
for f in man["frames"]:
    if f["split"] == "test" and f["eligible"]: byp[f["pass_id"]].append(f)
for pid, fs in byp.items():
    fs = sorted(fs, key=lambda f: f["i"]); xy = np.array([[f["x"], f["y"]] for f in fs])
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    for f, d in zip(fs, cum): pos[f["i"]] = (pid, float(d))
D = collections.defaultdict(dict)
for ln in open(f"{PKG}/results/geometric/march/line_fit_test_per_frame.csv").read().splitlines()[1:]:
    a, s, i, cls, off, hdg, *_ = ln.split(",")
    if cls == "two_row" and off and hdg: D[(a, int(s))][int(i)] = (float(off), float(hdg))

def block_stat_ci(rows, L, stat):   # rows: (pass,pos,val); block bootstrap of stat over concatenated series
    bp = collections.defaultdict(list)
    for r in rows: bp[r[0]].append(r[2])
    blocks = []
    for v in bp.values():
        v = np.array(v)
        for st in range(0, len(v) - L + 1): blocks.append(v[st:st+L])
    if len(blocks) < 8: return [None, None]
    N = sum(len(v) for v in bp.values()); nb = int(np.ceil(N/L)); rng = np.random.default_rng(42)
    out = [stat(np.concatenate([blocks[j] for j in rng.integers(0, len(blocks), nb)])) for _ in range(BOOT)]
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]

rms = lambda a: float(np.sqrt(np.mean(np.asarray(a, float)**2)))
def across(arm, k):   # per-frame mean over 3 seeds (two_row in all 3), (pass,pos,val)
    common = set.intersection(*[set(D[(arm, s)]) for s in SEEDS])
    return sorted([(pos[i][0], pos[i][1], np.mean([D[(arm, s)][i][k] for s in SEEDS])) for i in common if i in pos])

print("=== per-arm test (across-seed; RMS [block-bootstrap 95% CI]) ===")
arm_rows = {}
for arm in "ABC":
    o = across(arm, 0); h = across(arm, 1); arm_rows[arm] = (o, h)
    g1, g2 = rms([r[2] for r in o]), rms([r[2] for r in h])
    ci1 = block_stat_ci(o, L_GT1, rms); ci2 = block_stat_ci(h, L_GT2, rms)
    print(f"  {arm}: n={len(o)} | GT1 RMS {g1:.4f} CI{ci1} m | GT2 RMS {g2:.3f} CI{ci2} deg")

print("\n=== cross-arm paired (across-seed; mean Δ [block-bootstrap 95% CI]; % of floor) ===")
paired = {}
for (x, y) in [("A", "B"), ("A", "C"), ("B", "C")]:
    cx = set.intersection(*[set(D[(x, s)]) for s in SEEDS]); cy = set.intersection(*[set(D[(y, s)]) for s in SEEDS])
    common = [i for i in (cx & cy) if i in pos]
    def md(i, k): return np.mean([D[(x, s)][i][k] for s in SEEDS]) - np.mean([D[(y, s)][i][k] for s in SEEDS])
    r1 = sorted([(pos[i][0], pos[i][1], md(i, 0)) for i in common]); r2 = sorted([(pos[i][0], pos[i][1], md(i, 1)) for i in common])
    d1, d2 = np.mean([r[2] for r in r1]), np.mean([r[2] for r in r2])
    ci1 = block_stat_ci(r1, L_GT1, np.mean); ci2 = block_stat_ci(r2, L_GT2, np.mean)
    g2f = (G2FLOOR[x] + G2FLOOR[y]) / 2
    e1 = ci1[0] is not None and (ci1[0] > 0 or ci1[1] < 0); e2 = ci2[0] is not None and (ci2[0] > 0 or ci2[1] < 0)
    paired[f"{x}-{y}"] = {"gt1_diff": round(d1, 4), "gt1_ci": ci1, "gt1_excl0": e1,
                          "gt2_diff": round(d2, 4), "gt2_ci": ci2, "gt2_excl0": e2}
    print(f"  {x}-{y} (n={len(common)}): GT1 Δ={d1*1000:+.1f}mm ({100*d1/RTK:+.0f}% RTK) CI{[round(c*1000,1) for c in ci1]}mm excl0={e1} | "
          f"GT2 Δ={d2:+.3f}° ({100*d2/g2f:+.0f}% floor) CI{ci2}° excl0={e2}")
json.dump({"per_arm_ci": {a: {"gt1_rms": rms([r[2] for r in arm_rows[a][0]]), "gt1_ci": block_stat_ci(arm_rows[a][0], L_GT1, rms),
                              "gt2_rms": rms([r[2] for r in arm_rows[a][1]]), "gt2_ci": block_stat_ci(arm_rows[a][1], L_GT2, rms)} for a in "ABC"},
           "paired": paired}, open(f"{PKG}/results/geometric/march/paired_crossarm_test.json", "w"), indent=2)
print("\nwrote paired_crossarm_test.json")
