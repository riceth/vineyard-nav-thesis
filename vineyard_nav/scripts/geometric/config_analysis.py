"""Whole-bag Phase C downstream config analysis (D040, F018). Bag-agnostic multi-bag template.

  python3 config_analysis.py --bag march   -> results/geometric/march/final/march_evaluation/config_analysis.json

Reads the whole-bag detection cache (results/geometric/{bag}/cache/detections.csv, all eligible
frames, no split) and, over those frames, runs:
  1. the full downstream config sweep: class-agnostic + trunk-primary / pole-primary x T in {1,2,3,5,8,12},
     with the argmin selection logic + pre-stated tie-break (argmin CI overlaps agnostic -> lock agnostic);
  2. the single-class ablations: trunk_only, pole_only.
Locked upstream (D035/D036/D037/D038 line-fit; conf 0.25, 15% blob guard). Per cell: coverage,
GT-1/GT-2 RMS + moving-block bootstrap CI at the whole-bag Analysis-H block lengths (block_lengths.py),
mean base points.

The class-agnostic design decision is LOCKED (F018) and re-reported on each bag, not re-selected.
argmin_summary flags agnostic_locked; if a non-agnostic cell strictly wins (CI does not overlap
agnostic) pause_flag=True and the pipeline should pause for review.
"""
import sys
import json
import collections
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
sys.path.insert(0, str(PKG))
import projection_calibration as C
exec(open(Path(__file__).resolve().parent / "row_model.py").read())
import block_lengths as BL
from bag_config import parse_bag

B = parse_bag()
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


# ---- argmin over the VIABLE regime (coverage >= VIABLE_COV) + pre-stated tie-break --------------
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
