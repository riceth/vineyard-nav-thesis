"""In-row ABSTENTION analysis (F024). Bag-agnostic. Characterises the pipeline's `single_row`
behaviour on the eligible (in-row) frames: the class mix (from the committed per-frame CSV) and the
MECHANISM that produces each abstention (re-run the front-end on the single_row frames and record why
the second side was rejected by `fit_side_far`, and whether that side was detected-but-unseeded).

  python3 single_row_analysis.py --bag march   -> final/{bag}_evaluation/single_row_analysis.json

Class mix is read from the committed line_fit_per_frame.csv (authority; byte-consistent with F011).
Reasons require inference (the failing-side line is not stored), so the front-end is re-run — mirroring
line_fit_infer.py EXACTLY (same CONF/BLOB_FRAC/UNET_MIN_AREA/project_px/fit_side_far). Per-arm across
all three seeds (like F011/F013). Reads-only w.r.t. committed artefacts; writes one new sibling JSON.
"""
import sys, json, argparse, collections
from pathlib import Path
import numpy as np, cv2

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "scripts" / "geometric"))
import torch
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scripts.perception.segmentation.unet_binary.model import UNetBinary
from scripts.perception.segmentation.unet_binary.dataset import IMAGENET_MEAN, IMAGENET_STD
import projection_calibration as C
from single_arm_dryrun import CONF, BLOB_FRAC, FRAME_PX
from bag_config import resolve
exec(open(Path(__file__).resolve().parent / "row_model.py").read())   # NEAR, FARMAX, fit_side_far

ap = argparse.ArgumentParser(); ap.add_argument("--bag", default="march"); BAG = ap.parse_args().bag
B = resolve(BAG, "eligible"); FR = B["frames_dir"]
OUT = B["out_dir"] / "single_row_analysis.json"
UNET_MIN_AREA = 40
MODELS = [                                                     # identical to line_fit_infer.py
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

# --- class mix + single_row frame lists straight from the committed CSV (authority) ---
cls_ct = {(a, s): collections.Counter() for (a, s, _, _) in MODELS}
sr_frames = {(a, s): [] for (a, s, _, _) in MODELS}
for ln in Path(B["per_frame_csv"]).read_text().splitlines()[1:]:
    a, s, i, cls, *_ = ln.split(","); k = (a, int(s))
    if k not in cls_ct: continue
    cls_ct[k][cls] += 1
    if cls == "single_row": sr_frames[k].append(int(i))


def yolo_base(model, img):
    r = model.predict(source=img, conf=CONF, half=True, device=0, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0: return []
    xy = r.boxes.xyxy.cpu().numpy()
    ar = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
    return [((x1 + x2) / 2, y2) for (x1, y1, x2, y2) in xy[ar <= BLOB_FRAC * FRAME_PX * FRAME_PX]]


def unet_base(unet, img):
    x = _TF(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"].unsqueeze(0).to(dev)
    with torch.no_grad(): fg = (unet(x).argmax(1)[0].cpu().numpy() == 1).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    return [(st[k][0] + st[k][2] / 2., st[k][1] + st[k][3] - 1) for k in range(1, n) if st[k][4] >= UNET_MIN_AREA]


# --- re-run the front-end on the single_row frames to recover the abstention mechanism ---
mech = {a: {"n": 0, "not_reproduced": 0, "reason": collections.Counter(),
            "seen": collections.Counter(), "side": collections.Counter()} for a in "ABC"}
for (arm, seed, typ, ckpt) in MODELS:
    frames = sr_frames[(arm, seed)]
    print(f"[{BAG}][{arm} s{seed}] {len(frames)} single_row frames ...", flush=True)
    if typ == "yolo":
        m = YOLO(str(PKG / "results/runs" / ckpt)); front = lambda im: yolo_base(m, im)
    else:
        m = UNetBinary(encoder_weights=None).to(dev).eval()
        m.load_state_dict(torch.load(PKG / "results/runs" / ckpt, map_location=dev, weights_only=False)["model_state_dict"])
        front = lambda im: unet_base(m, im)
    d = mech[arm]
    for fi in frames:
        img = cv2.imread(str(FR / f"{fi:05d}.jpg"))
        if img is None: continue
        L, R = [], []
        for (uc, v) in front(img):
            g = C.project_px(uc, v, near_m=FARMAX)
            if g is not None: (L if uc < 320 else R).append(g)
        L = np.array(L) if L else np.empty((0, 2)); R = np.array(R) if R else np.empty((0, 2))
        fL, fR = fit_side_far(L), fit_side_far(R)
        if fL["ok"] == fR["ok"]:                       # did not reproduce single_row (nondeterminism/edge)
            d["not_reproduced"] += 1; continue
        d["n"] += 1
        bad, P = (fR, R) if fL["ok"] else (fL, L)
        d["side"]["L_fit" if fL["ok"] else "R_fit"] += 1
        d["reason"][bad.get("reason", "?")] += 1
        nraw = len(P); nnear = int((P[:, 0] < NEAR).sum()) if nraw else 0
        d["seen"]["not_seen" if nraw == 0 else ("seen_far_only" if nnear < 2 else "seen_near")] += 1
    del m; torch.cuda.empty_cache()

# --- aggregate: class mix per-arm mean+/-SD across seeds; mechanism per-arm over all single_row frames ---
def pa_meansd(arm, cls):
    vals = [100 * cls_ct[(arm, s)][cls] / sum(cls_ct[(arm, s)].values()) for s in (42, 43, 44)]
    return [round(float(np.mean(vals)), 1), round(float(np.std(vals)), 2)]
def pct(counter):
    t = sum(counter.values()) or 1
    return {k: round(100 * v / t, 1) for k, v in counter.most_common()}

report = {
    "config": {"bag": BAG, "near_seed_window_m": NEAR, "far_max_m": FARMAX,
               "note": "F024 in-row abstention. class_mix = per-arm mean+/-SD across seeds 42/43/44 from the committed CSV. "
                       "mechanism = failing-side rejection reason from fit_side_far, re-run on every single_row frame (all 9 models). "
                       "single_row emits NO centreline; D-G tier-2 half-spacing fallback (SPEC §10) specified but NOT implemented."},
    "class_mix_pct": {a: {c: pa_meansd(a, c) for c in ("two_row", "single_row", "none", "fitfail")} for a in "ABC"},
    "single_row_mechanism": {a: {
        "n_frames_reprocessed": mech[a]["n"], "not_reproduced": mech[a]["not_reproduced"],
        "failing_side_reason_pct": pct(mech[a]["reason"]),
        "failing_row_detected_pct": round(100 * (mech[a]["seen"]["seen_far_only"] + mech[a]["seen"]["seen_near"]) / (mech[a]["n"] or 1), 1),
        "seen_breakdown_pct": pct(mech[a]["seen"]),
        "fit_side": dict(mech[a]["side"])} for a in "ABC"},
}
OUT.write_text(json.dumps(report, indent=2))
print(f"\nwrote {OUT}")
for a in "ABC":
    cm = report["class_mix_pct"][a]; me = report["single_row_mechanism"][a]
    print(f"  {a}: single_row {cm['single_row'][0]}±{cm['single_row'][1]}%  "
          f"reason {me['failing_side_reason_pct']}  detected {me['failing_row_detected_pct']}%")
