"""Single-class ablations for F018: trunk-only (poles excluded) + pole-only (trunks excluded).
Same offline reclustering + line-fit pipeline + block-bootstrap CIs (Analysis-H block lengths)
as config_sweep_val.py. Reports actual coverage/RMS/CI even if degenerate."""
import json, collections, sys
from pathlib import Path
import numpy as np
PKG = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PKG/"scripts"/"geometric")); sys.path.insert(0, str(PKG))
import projection_calibration as C
exec(open(Path(__file__).resolve().parent / "row_model.py").read())
from paths import CACHE_DIR
DETS = CACHE_DIR / "detections_val.csv"
SEEDS = [42, 43, 44]; BOOT = 10000; L_GT1, L_GT2 = 11, 31

man = json.load(open(PKG/"results/geometric/march/dataset_manifest.json"))
pos = {}; byp = collections.defaultdict(list)
for f in man["frames"]:
    if f["split"] == "val" and f["eligible"]: byp[f["pass_id"]].append(f)
for pid, fs in byp.items():
    fs = sorted(fs, key=lambda f: f["i"]); xy = np.array([[f["x"], f["y"]] for f in fs])
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    for f, d in zip(fs, cum): pos[f["i"]] = (pid, float(d))
det = collections.defaultdict(list)
for ln in open(DETS).read().splitlines()[1:]:
    s, fr, c, u, v = ln.split(","); det[(int(s), int(fr))].append((int(c), float(u), float(v)))
FRAMES = sorted(set(fr for (_, fr) in det))

def estimate(base_pts):
    L, R = [], []
    for (uc, v) in base_pts:
        g = C.project_px(uc, v, near_m=FARMAX)
        if g is not None: (L if uc < 320 else R).append(g)
    L = np.array(L) if L else np.empty((0, 2)); R = np.array(R) if R else np.empty((0, 2))
    fL, fR = fit_side_far(L), fit_side_far(R)
    if fL["ok"] and fR["ok"]:
        cl = centre_linefit(L[fL["inl"]], R[fR["inl"]])
        if cl: return ("two_row", cl["offset"], cl["heading"], len(base_pts))
    return ("single_row" if (fL["ok"] or fR["ok"]) else "none", None, None, len(base_pts))

def block_rms_ci(rows, L):
    bp = collections.defaultdict(list)
    for r in rows: bp[r[0]].append(r[2])
    ss, ln = [], []
    for v in bp.values():
        v = np.array(v)
        for st in range(0, len(v) - L + 1): ss.append(np.sum(v[st:st+L]**2)); ln.append(L)
    if len(ss) < 8: return [None, None]
    ss, ln = np.array(ss), np.array(ln); N = sum(len(v) for v in bp.values()); nb = int(np.ceil(N/L))
    rng = np.random.default_rng(42)
    r = [np.sqrt(ss[idx].sum()/ln[idx].sum()) for idx in (rng.integers(0, len(ss), nb) for _ in range(BOOT))]
    return [round(float(np.percentile(r, 2.5)), 4), round(float(np.percentile(r, 97.5)), 4)]

def rms(a): a = np.asarray(a, float); return float(np.sqrt(np.mean(a**2))) if len(a) else float("nan")

out = {}
for name, keep in [("trunk_only", 0), ("pole_only", 1)]:
    cov = collections.Counter(); offs_f = {}; hdgs_f = {}; nb = []
    for fr in FRAMES:
        os_, hs = [], []
        for s in SEEDS:
            bp = [(u, v) for (c, u, v) in det.get((s, fr), []) if c == keep]
            cls, o, h, n = estimate(bp); nb.append(n)
            if cls == "two_row": os_.append(o); hs.append(h)
            cov[cls] += 1
        if len(os_) == 3: offs_f[fr] = np.mean(os_); hdgs_f[fr] = np.mean(hs)
    tot = 3 * len(FRAMES)
    orow = sorted([(pos[fr][0], pos[fr][1], offs_f[fr]) for fr in offs_f if fr in pos])
    hrow = sorted([(pos[fr][0], pos[fr][1], hdgs_f[fr]) for fr in hdgs_f if fr in pos])
    out[name] = {"two_row_pct": round(100*cov["two_row"]/tot, 1), "single_pct": round(100*cov["single_row"]/tot, 1),
                 "none_pct": round(100*cov["none"]/tot, 1), "mean_base": round(float(np.mean(nb)), 1),
                 "gt1_rms": round(rms([r[2] for r in orow]), 4), "gt1_ci": block_rms_ci(orow, L_GT1),
                 "gt2_rms": round(rms([r[2] for r in hrow]), 3), "gt2_ci": block_rms_ci(hrow, L_GT2),
                 "n_two_row_frames": len(offs_f)}
    o = out[name]
    print(f"{name:>12}: 2r {o['two_row_pct']}% (single {o['single_pct']}% none {o['none_pct']}%) base {o['mean_base']} "
          f"| GT1 RMS {o['gt1_rms']} CI{o['gt1_ci']} | GT2 RMS {o['gt2_rms']} CI{o['gt2_ci']} | n_2r_frames {o['n_two_row_frames']}")
# merge with existing sweep json
sw = json.load(open(PKG/"results/geometric/march/config_sweep_val.json")); sw.update(out)
json.dump(sw, open(PKG/"results/geometric/march/config_sweep_val.json", "w"), indent=2)
# compare trunk_only vs agnostic
agn = sw["agnostic"]
def ov(a, b): return a[0] is not None and b[0] is not None and a[0] <= b[1] and b[0] <= a[1]
print(f"\ntrunk_only vs agnostic: GT1 overlap {ov(out['trunk_only']['gt1_ci'], agn['gt1_ci'])}, GT2 overlap {ov(out['trunk_only']['gt2_ci'], agn['gt2_ci'])}")
print(f"agnostic: 2r {agn['two_row_pct']}% GT1 {agn['gt1_rms']} CI{agn['gt1_ci']} GT2 {agn['gt2_rms']} CI{agn['gt2_ci']}")
print("wrote config_sweep_val.json (ablations merged)")
