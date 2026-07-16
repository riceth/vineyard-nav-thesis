"""F018 test-side ablation: agnostic / trunk-only / pole-only on the 3149 test frames
(cached Phase C test detections, all 3 seeds). Locked line-fit pipeline + block-bootstrap CIs
(Analysis-H lengths). Mirrors the val F018 table on held-out data."""
import json, collections, sys
from pathlib import Path
import numpy as np
PKG = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PKG/"scripts"/"geometric")); sys.path.insert(0, str(PKG))
import projection_calibration as C
exec(open(Path(__file__).resolve().parent / "row_model.py").read())
from paths import CACHE_DIR
DETS = CACHE_DIR / "detections_test.csv"
SEEDS = [42,43,44]; BOOT = 10000; L_GT1, L_GT2 = 11, 31
man = json.load(open(PKG/"results/geometric/march/dataset_manifest.json"))
pos = {}; byp = collections.defaultdict(list)
for f in man["frames"]:
    if f["split"] == "test" and f["eligible"]: byp[f["pass_id"]].append(f)
for pid, fs in byp.items():
    fs = sorted(fs, key=lambda f: f["i"]); xy = np.array([[f["x"],f["y"]] for f in fs])
    cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))])
    for f,d in zip(fs,cum): pos[f["i"]] = (pid, float(d))
det = collections.defaultdict(list)
for ln in open(DETS).read().splitlines()[1:]:
    s,fr,c,u,v = ln.split(","); det[(int(s),int(fr))].append((int(c),float(u),float(v)))
FRAMES = sorted(set(fr for (_,fr) in det))
def sel(dets, cfg):
    if cfg == "agnostic": return [(u,v) for (c,u,v) in dets]
    k = 0 if cfg == "trunk_only" else 1
    return [(u,v) for (c,u,v) in dets if c == k]
def estimate(bp):
    L,R = [],[]
    for (uc,v) in bp:
        g = C.project_px(uc,v,near_m=FARMAX)
        if g is not None: (L if uc<320 else R).append(g)
    L = np.array(L) if L else np.empty((0,2)); R = np.array(R) if R else np.empty((0,2))
    fL,fR = fit_side_far(L), fit_side_far(R)
    if fL["ok"] and fR["ok"]:
        cl = centre_linefit(L[fL["inl"]], R[fR["inl"]])
        if cl: return ("two_row", cl["offset"], cl["heading"], len(bp))
    return ("single_row" if (fL["ok"] or fR["ok"]) else "none", None, None, len(bp))
def block_rms_ci(rows, L):
    bp = collections.defaultdict(list)
    for r in rows: bp[r[0]].append(r[2])
    ss,ln = [],[]
    for v in bp.values():
        v = np.array(v)
        for st in range(0,len(v)-L+1): ss.append(np.sum(v[st:st+L]**2)); ln.append(L)
    if len(ss) < 8: return [None,None]
    ss,ln = np.array(ss),np.array(ln); N = sum(len(v) for v in bp.values()); nb = int(np.ceil(N/L)); rng = np.random.default_rng(42)
    r = [np.sqrt(ss[idx].sum()/ln[idx].sum()) for idx in (rng.integers(0,len(ss),nb) for _ in range(BOOT))]
    return [round(float(np.percentile(r,2.5)),4), round(float(np.percentile(r,97.5)),4)]
rms = lambda a: float(np.sqrt(np.mean(np.asarray(a,float)**2))) if len(a) else float("nan")
print("=== F018 TEST-side ablation (3 configs, 3149 test frames, across-seed) ===")
out = {}
for cfg in ["agnostic","trunk_only","pole_only"]:
    cov = collections.Counter(); offs_f = {}; hdgs_f = {}; nb = []
    for fr in FRAMES:
        o_,h_ = [],[]
        for s in SEEDS:
            cls,o,h,n = estimate(sel(det.get((s,fr),[]), cfg)); nb.append(n)
            if cls == "two_row": o_.append(o); h_.append(h)
            cov[cls] += 1
        if len(o_) == 3: offs_f[fr] = np.mean(o_); hdgs_f[fr] = np.mean(h_)
    tot = 3*len(FRAMES)
    orow = sorted([(pos[fr][0],pos[fr][1],offs_f[fr]) for fr in offs_f if fr in pos])
    hrow = sorted([(pos[fr][0],pos[fr][1],hdgs_f[fr]) for fr in hdgs_f if fr in pos])
    out[cfg] = {"two_row_pct": round(100*cov["two_row"]/tot,1), "none_pct": round(100*cov["none"]/tot,1),
                "mean_base": round(float(np.mean(nb)),1), "gt1_rms": round(rms([r[2] for r in orow]),4),
                "gt1_ci": block_rms_ci(orow,L_GT1), "gt2_rms": round(rms([r[2] for r in hrow]),3),
                "gt2_ci": block_rms_ci(hrow,L_GT2), "n2r_frames": len(orow)}
    o = out[cfg]
    print(f"  {cfg:>11}: 2r {o['two_row_pct']}% (none {o['none_pct']}%) base {o['mean_base']} | GT1 RMS {o['gt1_rms']} CI{o['gt1_ci']} | GT2 RMS {o['gt2_rms']} CI{o['gt2_ci']} | n2r {o['n2r_frames']}")
json.dump(out, open(PKG/"results/geometric/march/final/test_evaluation/config_ablation_test.json","w"), indent=2)
a = out["agnostic"]; t = out["trunk_only"]
def ov(x,y): return x[0] is not None and y[0] is not None and x[0]<=y[1] and y[0]<=x[1]
print(f"\ntrunk_only vs agnostic: GT1 CI overlap {ov(t['gt1_ci'],a['gt1_ci'])}, GT2 CI overlap {ov(t['gt2_ci'],a['gt2_ci'])}")
print(f"coverage supplement (agnostic - trunk_only) = {a['two_row_pct']-t['two_row_pct']:+.1f} pp | pole_only 2r {out['pole_only']['two_row_pct']}% (degenerate check)")
print("wrote config_ablation_test.json")
