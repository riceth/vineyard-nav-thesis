"""Two-layer rejection for non-in-row spurious two_row outputs (F022 state gate + F023 geometry
filter). Bag-agnostic. Evaluates on all frames: in-row (D040, final/{bag}_evaluation/) + non-in-row
(D041 category C, final/non_in_row_evaluation/).

  python3 mitigation_analysis.py --bag march

F022 -- runtime STATE gate (odometry-based; catches the state failure the perception cannot see):
  - upper bound = the manifest `eligible` flag (oracle state): 100% non-in-row rejection, 0% in-row
    FP by construction -- the architectural potential.
  - causal per-frame = the runtime-realistic gate from odometry signals alone (no offline pass-
    length): speed > V_MIN, |v_y| > VY_INROW, |heading-rate| < HR_THRESH (HR_THRESH = in-row p99).
    Reports FP/FN including boundary frames near the thresholds.
F023 -- perception-only GEOMETRY filter (empirical in-row p99 thresholds); catches geometric
  outliers but NOT clean-geometry turns (those overlap the in-row distribution -> F022's job).
Reports per category (stationary / turn / transition), per arm, in-row FP, and the combined union.
"""
import sys, json, collections, bisect
from pathlib import Path
import numpy as np

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
from bag_config import resolve, frames_for_scope

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--bag", default="march"); BAG = ap.parse_args().bag
B_in = resolve(BAG, "eligible"); B_non = resolve(BAG, "non_in_row")
MAN = json.load(open(B_in["manifest"]))
OUT = B_in["out_dir"].parent / "mitigation_evaluation" / "mitigation_analysis.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

V_MIN, VY_INROW = 0.10, 0.30
# F023 geometry thresholds (in-row p99; see derivation in the Commit-7 gate)
G_OFF, G_HDG, G_PAR, G_NB = 0.71, 6.7, 0.22, 12

# --- per-frame kinematics (odometry) ----------------------------------------------------------
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

# --- categorise non-in-row frames -------------------------------------------------------------
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

# --- load two_row outputs from both strata ----------------------------------------------------
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
