"""Analysis H (spatial autocorrelation of paired ΔGT-1/ΔGT-2 vs separation in metres) +
Analysis I (moving-block bootstrap). Reports decorrelation distance per pair, then compares
CI widths: current Δs=1.5m subsample vs Δs=measured-decorr subsample vs block bootstrap."""
import sys, json, collections
import numpy as np
from pathlib import Path
PKG = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
from bag_config import parse_bag
BAG = parse_bag()
B, SEED = 10000, 42
SEEDS, PAIRS = [42, 43, 44], [("A", "B"), ("A", "C"), ("B", "C")]

man = json.load(open(BAG["manifest"]))
# path position (cumulative m) per whole-bag eligible frame within its pass (D040; no split key)
byp = collections.defaultdict(list)
for f in man["frames"]:
    if f["eligible"]: byp[f["pass_id"]].append(f)
pos = {}
for pid, fs in byp.items():
    fs = sorted(fs, key=lambda f: f["i"]); xy = np.array([[f["x"], f["y"]] for f in fs])
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    for f, d in zip(fs, cum): pos[f["i"]] = (pid, float(d))

D = collections.defaultdict(dict)
for ln in open(BAG["per_frame_csv"]).read().splitlines()[1:]:
    a, s, i, cls, off, hdg, *_ = ln.split(",")
    if cls == "two_row" and off and hdg: D[(a, int(s))][int(i)] = (float(off), float(hdg))

def series(x, y):
    cx = set.intersection(*[set(D[(x, s)]) for s in SEEDS]); cy = set.intersection(*[set(D[(y, s)]) for s in SEEDS])
    rows = []
    for i in (cx & cy):
        if i not in pos: continue
        d1 = np.mean([D[(x, s)][i][0] for s in SEEDS]) - np.mean([D[(y, s)][i][0] for s in SEEDS])
        d2 = np.mean([D[(x, s)][i][1] for s in SEEDS]) - np.mean([D[(y, s)][i][1] for s in SEEDS])
        rows.append((pos[i][0], pos[i][1], d1, d2))
    return sorted(rows, key=lambda r: (r[0], r[1]))

def autocorr(rows, ci, maxd=3.0, bw=0.15):
    v = np.array([r[ci] for r in rows]); mu, var = v.mean(), v.var()
    nb = int(maxd / bw); num = np.zeros(nb); cnt = np.zeros(nb)
    bp = collections.defaultdict(list)
    for r in rows: bp[r[0]].append(r)
    for _, rs in bp.items():
        p = np.array([r[1] for r in rs]); val = np.array([r[ci] for r in rs]) - mu
        for a in range(len(rs)):
            b = a + 1
            while b < len(rs) and p[b] - p[a] < maxd:
                k = int((p[b] - p[a]) / bw)
                if k < nb: num[k] += val[a] * val[b]; cnt[k] += 1
                b += 1
    ac = np.where(cnt > 0, num / np.maximum(cnt, 1) / var, np.nan)
    return (np.arange(nb) + 0.5) * bw, ac

def decorr(centres, ac, thr):
    for c, a in zip(centres, ac):
        if not np.isnan(a) and a < thr: return round(float(c), 2)
    return None

def mean_spacing(rows):
    bp = collections.defaultdict(list)
    for r in rows: bp[r[0]].append(r[1])
    sps = [(max(v) - min(v)) / (len(v) - 1) for v in bp.values() if len(v) > 1]
    return float(np.mean(sps)) if sps else 0.05

def sub_ci(rows, ci, ds, seed=SEED):
    bp = collections.defaultdict(list); keep = []
    for r in rows: bp[r[0]].append(r)
    for _, rs in bp.items():
        last = -1e9
        for r in rs:
            if r[1] - last >= ds: keep.append(r); last = r[1]
    v = np.array([r[ci] for r in keep])
    if len(v) < 8: return [None, None, len(v)]
    rng = np.random.default_rng(seed); n = len(v)
    m = [v[rng.integers(0, n, n)].mean() for _ in range(B)]
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4), len(v)]

def block_ci(rows, ci, L, seed=SEED):
    bp = collections.defaultdict(list)
    for r in rows: bp[r[0]].append(r[ci])
    bmeans = []
    for _, v in bp.items():
        v = np.array(v)
        for st in range(0, len(v) - L + 1): bmeans.append(v[st:st + L].mean())
    if len(bmeans) < 8: return [None, None, 0]
    bmeans = np.array(bmeans); N = sum(len(v) for v in bp.values()); nb = int(np.ceil(N / L))
    rng = np.random.default_rng(seed)
    m = [bmeans[rng.integers(0, len(bmeans), nb)].mean() for _ in range(B)]
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4), nb]

for (x, y) in PAIRS:
    rows = series(x, y); sp = mean_spacing(rows)
    print(f"\n===== {x}-{y}  (n_all={len(rows)}, median entry spacing {sp*100:.1f} cm) =====")
    for ci, name, unit, scale in [(2, "GT1", "mm", 1000), (3, "GT2", "deg", 1)]:
        c, ac = autocorr(rows, ci)
        d10, d05 = decorr(c, ac, 0.1), decorr(c, ac, 0.05)
        pt = np.mean([r[ci] for r in rows]) * scale
        ac_head = [round(float(a), 2) for a in ac[:8]]
        print(f"  {name}: point Δ={pt:+.2f} {unit} | autocorr@[0.15..1.2m]={ac_head} | decorr<0.1 @ {d10} m, <0.05 @ {d05} m")
        dd = d10 if d10 else 1.5
        L = max(2, int(round(2 * dd / sp)))
        s15 = sub_ci(rows, ci, 1.5); sdd = sub_ci(rows, ci, dd); bl = block_ci(rows, ci, L)
        f = scale
        print(f"     CI Δs=1.5m (n={s15[2]}): [{s15[0]*f if s15[0] else None:+.2f},{s15[1]*f if s15[1] else None:+.2f}] {unit}")
        print(f"     CI Δs={dd}m (n={sdd[2]}): [{sdd[0]*f if sdd[0] else None:+.2f},{sdd[1]*f if sdd[1] else None:+.2f}] {unit}")
        print(f"     CI block L={L} (~{2*dd:.1f}m, nb={bl[2]}): [{bl[0]*f if bl[0] else None:+.2f},{bl[1]*f if bl[1] else None:+.2f}] {unit}")
