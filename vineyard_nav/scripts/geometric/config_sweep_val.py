"""CP-4 Phase C downstream config sweep (D026) on val, offline on cached detections.
Configs: agnostic / trunk-primary / pole-primary; T grid {1,2,3,5,8,12}. Locked upstream
(D036/D037/D038 line-fit). Per config x T: coverage, GT-1/GT-2 RMS + block-bootstrap CI
(block = 2x Analysis-H decorrelation: GT-1 ~1.0m, GT-2 ~2.8m conservative), base points.
Cross-config CI overlap + argmin with pre-stated tie-break (overlap with agnostic -> lock agnostic)."""
import json, collections
import numpy as np
import sys
from pathlib import Path
PKG = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PKG/"scripts"/"geometric")); sys.path.insert(0, str(PKG))
import projection_calibration as C
exec(open(Path(__file__).resolve().parent / "row_model.py").read())

from paths import CACHE_DIR
DETS = CACHE_DIR / "detections_val.csv"
SEEDS = [42, 43, 44]; T_GRID = [1, 2, 3, 5, 8, 12]; CONFIGS = ["agnostic", "trunk", "pole"]
BOOT = 10000; L_GT1, L_GT2 = 11, 31   # block entries ~1.0m (GT1) / ~2.8m (GT2), 9cm spacing

man = json.load(open(PKG/"results/geometric/march/dataset_manifest.json"))
pos = {}
byp = collections.defaultdict(list)
for f in man["frames"]:
    if f["split"] == "val" and f["eligible"]: byp[f["pass_id"]].append(f)
for pid, fs in byp.items():
    fs = sorted(fs, key=lambda f: f["i"]); xy = np.array([[f["x"], f["y"]] for f in fs])
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    for f, d in zip(fs, cum): pos[f["i"]] = (pid, float(d))

det = collections.defaultdict(list)   # (seed,frame) -> [(cls,uc,v)]
for ln in open(DETS).read().splitlines()[1:]:
    s, fr, c, u, v = ln.split(","); det[(int(s), int(fr))].append((int(c), float(u), float(v)))
FRAMES = sorted(set(fr for (_, fr) in det))

def select(dets, config, T):
    out = []
    for isL in (True, False):
        sd = [(c, u, v) for (c, u, v) in dets if (u < 320) == isL]
        tr = [(u, v) for (c, u, v) in sd if c == 0]; po = [(u, v) for (c, u, v) in sd if c == 1]
        out += (tr + po) if config == "agnostic" else \
               (tr if len(tr) >= T else tr + po) if config == "trunk" else \
               (po if len(po) >= T else po + tr)
    return out

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

def block_rms_ci(rows, L):   # rows: (pass,pos,val) ordered; block bootstrap RMS
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

report = {}
for config in CONFIGS:
    for T in ([1] if config == "agnostic" else T_GRID):   # agnostic is T-invariant
        cov = collections.Counter(); offs_f = {}; hdgs_f = {}; nb = []
        for fr in FRAMES:
            os_, hs = [], []
            for s in SEEDS:
                cls, o, h, n = estimate(select(det.get((s, fr), []), config, T))
                nb.append(n)
                if cls == "two_row": os_.append(o); hs.append(h)
                cov[(s, cls)] += 1
            if len(os_) == 3: offs_f[fr] = np.mean(os_); hdgs_f[fr] = np.mean(hs)
        tot = 3 * len(FRAMES); two = cov_two = sum(v for (s, c), v in cov.items() if c == "two_row")
        orow = sorted([(pos[fr][0], pos[fr][1], offs_f[fr]) for fr in offs_f if fr in pos])
        hrow = sorted([(pos[fr][0], pos[fr][1], hdgs_f[fr]) for fr in hdgs_f if fr in pos])
        key = f"{config}" + ("" if config == "agnostic" else f"_T{T}")
        report[key] = {
            "two_row_pct": round(100*cov_two/tot, 1),
            "single_pct": round(100*sum(v for (s,c),v in cov.items() if c=="single_row")/tot, 1),
            "none_pct": round(100*sum(v for (s,c),v in cov.items() if c=="none")/tot, 1),
            "mean_base": round(float(np.mean(nb)), 1),
            "gt1_rms": round(rms([r[2] for r in orow]), 4), "gt1_ci": block_rms_ci(orow, L_GT1),
            "gt2_rms": round(rms([r[2] for r in hrow]), 3), "gt2_ci": block_rms_ci(hrow, L_GT2)}
        print(f"{key:>14}: 2r {report[key]['two_row_pct']}% base {report[key]['mean_base']} | "
              f"GT1 RMS {report[key]['gt1_rms']} CI{report[key]['gt1_ci']} | GT2 RMS {report[key]['gt2_rms']} CI{report[key]['gt2_ci']}", flush=True)
json.dump(report, open(PKG/"results/geometric/march/config_sweep_val.json", "w"), indent=2)

# overlap matrix + argmin + tie-break
def overlaps(a, b): return a[0] <= b[1] and b[0] <= a[1]
agn = report["agnostic"]
print("\n=== argmin (lowest val RMS) + tie-break (overlap agnostic -> lock agnostic) ===")
for metric, rk, ck in [("GT-1", "gt1_rms", "gt1_ci"), ("GT-2", "gt2_rms", "gt2_ci")]:
    best = min(report, key=lambda k: report[k][rk])
    ov = overlaps(report[best][ck], agn[ck])
    lock = "agnostic" if ov else best
    print(f"{metric}: argmin={best} (RMS {report[best][rk]}); CI overlaps agnostic={ov} -> LOCK {lock}")
allci1 = [report[k]["gt1_ci"] for k in report]; allci2 = [report[k]["gt2_ci"] for k in report]
flat1 = all(overlaps(a, b) for a in allci1 for b in allci1)
flat2 = all(overlaps(a, b) for a in allci2 for b in allci2)
print(f"\nFLAT? GT-1 all cells CI-overlap: {flat1} | GT-2: {flat2}")
print("wrote results/geometric/march/config_sweep_val.json")
