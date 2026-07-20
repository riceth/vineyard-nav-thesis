"""F025 near-seed window sensitivity. Bag-agnostic. The near-seed window (row_model.NEAR) is a FITTING
parameter, not a detection parameter, so base points are detected ONCE per eligible frame (all 9
models) and cached; NEAR is then swept over the FIT (fit_side_far) alone -- 11x fewer inference passes
(Commit-10 Correction 1).

  python3 near_seed_sensitivity.py --bag march     -> final/{bag}_evaluation/near_seed_sensitivity.json

Per window reports (per arm, mean across seeds):
  - two_row coverage % and FULL-SET offset RMS (the Optimisation-A metric; deployed-system accuracy)
  - recovery rate (previously-abstained single_row/none -> two_row) + recovered-frame RMS
  - lost rate (two_row -> not) and existing-frame offset-shift stats (median/mean/p90/max) -- because
    widening NEAR perturbs EXISTING two_row fits too (probe: 10% shift, max ~92 cm; NOT immutable)
  - geometric-plausibility fire rate on newly-recovered two_row (F023 in-row-p99 thresholds) -- a
    bounded plausibility check, NOT a rigorous FP rate (no ground truth for rows-actually-visible)
Optimisation A (widest window with full-set RMS <= (1+tol) x baseline; tol 10%, + 5/15% sensitivity)
and Optimisation B (largest window where marginal coverage gain pp >= marginal RMS loss cm) reported.

Base-point cache (projected L/R per frame) at --cache (default repo-relative, gitignored; rebuilt if
absent). NEAR=5 slice asserts recomputed two_row offsets == committed CSV (load-bearing consistency).
"""
import sys, json, argparse, pickle, collections
from pathlib import Path
import numpy as np, cv2

PKG = Path(__file__).resolve().parents[3]                     # one_time/ adds a level: geometric/one_time/<file>
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "scripts" / "geometric"))
import torch
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scripts.perception.segmentation.unet_binary.model import UNetBinary
from scripts.perception.segmentation.unet_binary.dataset import IMAGENET_MEAN, IMAGENET_STD
import projection_calibration as C
from single_arm_dryrun import CONF, BLOB_FRAC, FRAME_PX
from bag_config import resolve, frames_for_scope

RM = {}
exec(open(Path(__file__).resolve().parent.parent / "row_model.py").read(), RM)   # ../row_model.py (in scripts/geometric/)
FARMAX = RM["FARMAX"]

ap = argparse.ArgumentParser()
ap.add_argument("--bag", default="march")
ap.add_argument("--cache", default=None)
ap.add_argument("--refresh", action="store_true")
A_ = ap.parse_args(); BAG = A_.bag
B = resolve(BAG, "eligible"); FR = B["frames_dir"]
MAN = json.load(open(B["manifest"]))
OUT = B["out_dir"] / "near_seed_sensitivity.json"
CACHE = Path(A_.cache) if A_.cache else (PKG / "results/geometric" / BAG / "near_seed_basepoint_cache.pkl")
WINDOWS = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
BASE_W = 5.0
G_OFF, G_HDG, G_PAR, G_NB = 0.71, 6.7, 0.22, 12          # F023 in-row-p99 plausibility thresholds
MODELS = [
    ("A", 42, "unet", "phase_a_unet_binary_20260704_004105/checkpoints/best.pt"),
    ("A", 43, "unet", "phase_a_unet_binary_seed43_20260710_154347/checkpoints/best.pt"),
    ("A", 44, "unet", "phase_a_unet_binary_seed44_20260710_181339/checkpoints/best.pt"),
    ("B", 42, "yolo", "phase_b_yolo_binary/weights/best.pt"),
    ("B", 43, "yolo", "phase_b_yolo_binary_seed43/weights/best.pt"),
    ("B", 44, "yolo", "phase_b_yolo_binary_seed44/weights/best.pt"),
    ("C", 42, "yolo", "phase_c_yolo_multiclass/weights/best.pt"),
    ("C", 43, "yolo", "phase_c_yolo_multiclass_seed43/weights/best.pt"),
    ("C", 44, "yolo", "phase_c_yolo_multiclass_seed44/weights/best.pt"),
]
_TF = A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
dev = torch.device("cuda")
FRAMES = frames_for_scope(MAN, "eligible")

# baseline (cls, offset, n_base) from the committed CSV -- authority for NEAR=5 + n_base (NEAR-invariant)
BASE = {}
for ln in Path(B["per_frame_csv"]).read_text().splitlines()[1:]:
    a, s, i, cls, o, h, mL, mR, mc, n, ad, fl = ln.split(",")
    BASE[(a, int(s), int(i))] = (cls, float(o) if o else None, int(n))


# ---------------- base-point cache (inference once; NEAR-independent) ----------------
def yolo_base(model, img):
    r = model.predict(source=img, conf=CONF, half=True, device=0, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0: return []
    xy = r.boxes.xyxy.cpu().numpy()
    ar = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
    return [((x1 + x2) / 2, y2) for (x1, y1, x2, y2) in xy[ar <= BLOB_FRAC * FRAME_PX * FRAME_PX]]


def unet_base(net, img):
    x = _TF(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"].unsqueeze(0).to(dev)
    with torch.no_grad(): fg = (net(x).argmax(1)[0].cpu().numpy() == 1).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    return [(st[k][0] + st[k][2] / 2., st[k][1] + st[k][3] - 1) for k in range(1, n) if st[k][4] >= 40]


def project_sides(base_pts):
    L, R = [], []
    for (uc, v) in base_pts:
        g = C.project_px(uc, v, near_m=FARMAX)
        if g is not None: (L if uc < 320 else R).append(g)
    return (np.array(L) if L else np.empty((0, 2))), (np.array(R) if R else np.empty((0, 2)))


def build_cache():
    cache = {}
    for (arm, seed, typ, ckpt) in MODELS:
        print(f"[cache {arm} s{seed}] {len(FRAMES)} frames ...", flush=True)
        if typ == "yolo":
            m = YOLO(str(PKG / "results/runs" / ckpt)); front = lambda im: yolo_base(m, im)
        else:
            m = UNetBinary(encoder_weights=None).to(dev).eval()
            m.load_state_dict(torch.load(PKG / "results/runs" / ckpt, map_location=dev, weights_only=False)["model_state_dict"])
            front = lambda im: unet_base(m, im)
        for fi in FRAMES:
            img = cv2.imread(str(FR / f"{fi:05d}.jpg"))
            if img is None: continue
            cache[(arm, seed, fi)] = project_sides(front(img))
        del m; torch.cuda.empty_cache()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(cache, open(CACHE, "wb"))        # incremental save (survives a late crash)
        print(f"  cached {arm} s{seed} (total {len(cache)} frame-models)", flush=True)
    return cache


if CACHE.exists() and not A_.refresh:
    print(f"loading base-point cache {CACHE}")
    cache = pickle.load(open(CACHE, "rb"))
else:
    cache = build_cache()


# ---------------- fit at a given NEAR (over cached projected points) ----------------
def fit_at(L, R, near):
    RM["NEAR"] = near
    fL, fR = RM["fit_side_far"](L), RM["fit_side_far"](R)
    if fL["ok"] and fR["ok"]:
        cl = RM["centre_linefit"](L[fL["inl"]], R[fR["inl"]])
        if cl is None: return ("fitfail", None, None, None)
        return ("two_row", cl["offset"], cl["heading"], abs(cl["m_L"] - cl["m_R"]))
    return ("single_row" if (fL["ok"] or fR["ok"]) else "none", None, None, None)


def rms(a):
    a = np.asarray(a, float)
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")


# ---------------- sweep ----------------
per_arm_seed = collections.defaultdict(list)
for (arm, seed, _, _) in MODELS: per_arm_seed[arm].append(seed)
mismatch = 0
sweep = {f"{w:.1f}": {} for w in WINDOWS}
for w in WINDOWS:
    for arm in "ABC":
        agg = {"two_row_n": [], "full_rms": [], "rec_rate": [], "rec_rms": [], "lost_pct": [],
               "shift": [], "plaus": []}
        for seed in per_arm_seed[arm]:
            full_off, rec_off, shifts, plaus_fire, plaus_n = [], [], [], 0, 0
            n_two = n_frames = 0; rec = lost = base_two = 0
            for fi in FRAMES:
                key = (arm, seed, fi)
                if key not in cache or key not in BASE: continue
                n_frames += 1
                L, Rr = cache[key]; base_cls, base_off, n_base = BASE[key]
                cls, off, hdg, dm = fit_at(L, Rr, w)
                if base_cls == "two_row": base_two += 1
                if cls == "two_row":
                    n_two += 1; full_off.append(off)
                    if w == BASE_W and base_cls == "two_row" and abs(off - base_off) > 1e-6: mismatch += 1
                    if base_cls == "two_row":
                        shifts.append(abs(off - base_off))
                    else:                                              # recovered from abstention
                        rec += 1; rec_off.append(off); plaus_n += 1
                        if abs(off) > G_OFF or abs(hdg) > G_HDG or dm > G_PAR or n_base < G_NB:
                            plaus_fire += 1
                elif base_cls == "two_row":
                    lost += 1                                          # previously two_row, now not
            abst = n_frames - base_two                                 # recovery denominator
            agg["two_row_n"].append(100 * n_two / max(n_frames, 1))
            agg["full_rms"].append(rms(full_off))
            agg["rec_rate"].append(100 * rec / max(abst, 1))
            agg["rec_rms"].append(rms(rec_off))
            agg["lost_pct"].append(100 * lost / max(base_two, 1))
            agg["shift"].append(np.array(shifts))
            agg["plaus"].append(100 * plaus_fire / max(plaus_n, 1))
        allshift = np.concatenate(agg["shift"]) if agg["shift"] else np.array([])
        rr = [v for v in agg["rec_rms"] if not np.isnan(v)]
        sweep[f"{w:.1f}"][arm] = {
            "two_row_pct": round(float(np.mean(agg["two_row_n"])), 1),
            "full_set_rms_m": round(float(np.mean(agg["full_rms"])), 4),
            "recovery_rate_pct": round(float(np.mean(agg["rec_rate"])), 1),
            "recovered_rms_m": round(float(np.mean(rr)), 4) if rr else None,
            "lost_pct": round(float(np.mean(agg["lost_pct"])), 2),
            "existing_shift_cm": {"median": round(float(np.median(allshift) * 100), 2) if len(allshift) else 0.0,
                                   "mean": round(float(allshift.mean() * 100), 2) if len(allshift) else 0.0,
                                   "p90": round(float(np.percentile(allshift, 90) * 100), 2) if len(allshift) else 0.0,
                                   "max": round(float(allshift.max() * 100), 2) if len(allshift) else 0.0},
            "plausibility_fire_pct": round(float(np.mean(agg["plaus"])), 1)}
    print(f"  swept NEAR={w:.1f}", flush=True)

# ---------------- optimisation ----------------
def optimum_A(arm, tol):
    base_rms = sweep[f"{BASE_W:.1f}"][arm]["full_set_rms_m"]; best = BASE_W
    for w in WINDOWS:
        if sweep[f"{w:.1f}"][arm]["full_set_rms_m"] <= base_rms * (1 + tol): best = w
    return best

def optimum_B(arm):
    best = BASE_W
    for k in range(1, len(WINDOWS)):
        w0, w1 = WINDOWS[k - 1], WINDOWS[k]
        dcov = sweep[f"{w1:.1f}"][arm]["two_row_pct"] - sweep[f"{w0:.1f}"][arm]["two_row_pct"]
        drms_cm = (sweep[f"{w1:.1f}"][arm]["full_set_rms_m"] - sweep[f"{w0:.1f}"][arm]["full_set_rms_m"]) * 100
        if dcov >= max(drms_cm, 0):        # coverage gain (pp) still outpaces RMS loss (cm)
            best = w1
        else:
            break
    return best

report = {
    "config": {"bag": BAG, "windows_m": WINDOWS, "baseline_near_m": BASE_W, "far_max_m": FARMAX,
               "opt_A_tolerance_pct": 10, "f023_plausibility_thresholds": {"offset_m": G_OFF, "heading_deg": G_HDG, "parallelism": G_PAR, "n_base_min": G_NB},
               "note": "NEAR swept over the FIT only (base points detected once). full_set_rms = deployed-system accuracy (Opt A). "
                       "existing_shift = perturbation of frames two_row at BOTH baseline and this window (widening is NOT free). "
                       "plausibility_fire = F023-threshold rate on RECOVERED two_row (bounded check, not a rigorous FP rate). "
                       "NEAR=5 slice reproduces the committed CSV (mismatches=%d)." % mismatch},
    "baseline_near5": {arm: {"two_row_pct": sweep["5.0"][arm]["two_row_pct"], "full_set_rms_m": sweep["5.0"][arm]["full_set_rms_m"]} for arm in "ABC"},
    "sweep": sweep,
    "optimisation_A": {"tolerance_pct": 10, "per_arm_optimal_window_m": {arm: optimum_A(arm, 0.10) for arm in "ABC"},
                        "sensitivity": {"5pct": {arm: optimum_A(arm, 0.05) for arm in "ABC"},
                                        "15pct": {arm: optimum_A(arm, 0.15) for arm in "ABC"}}},
    "optimisation_B": {"definition": "largest window where marginal coverage gain (pp) >= marginal full-set RMS loss (cm)",
                        "per_arm_optimal_window_m": {arm: optimum_B(arm) for arm in "ABC"}},
    "csv_consistency_mismatches": mismatch,
}
OUT.write_text(json.dumps(report, indent=2))
print(f"\nNEAR=5 CSV-consistency mismatches: {mismatch} (expect 0)")
print(f"wrote {OUT}")
for arm in "ABC":
    print(f"  {arm}: baseline RMS {sweep['5.0'][arm]['full_set_rms_m']} | Opt-A(10%) window {optimum_A(arm,0.10)} m | Opt-B window {optimum_B(arm)} m")
