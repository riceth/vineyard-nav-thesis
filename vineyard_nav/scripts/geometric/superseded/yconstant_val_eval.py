"""CP-5 — 9-model geometric evaluation on the val corridors (all 4 708 eligible frames).

Runs each of the 9 trained checkpoints (Phase A U-Net binary, Phase B YOLO binary,
Phase C YOLO multiclass; seeds 42/43/44) through the CP-3-locked pipeline and reports,
per arm/seed and aggregated across seeds:

  coverage (two-row / single-row / none), base points per frame,
  GT-1 lateral offset (mean/median/SD/RMS + bootstrap CI on the Δs=1.5 m subsample),
  GT-2 heading error (mean/median/SD/RMS + bootstrap CI on the subsample),
  per-arm/seed anomalies (blob frames, systematic bias, unusual coverage).

Front-ends differ only at detection: YOLO gives bbox-bottom-centre base points (15 %-area
blob guard, D035); U-Net gives connected-component bbox-bottom-centre base points
(>= 40 px, no blob guard — the F007 blob is a YOLO mask-head pathology, U-Net is immune,
F007). Everything downstream (project -> cluster L/R -> Y-constant row -> binned centreline
-> GT-1/GT-2) is arm-agnostic and imported from single_arm_dryrun (CP-3 artefact, unmodified).

Dual-mode reporting (D-D): point estimates over ALL two-row val frames; 95 % bootstrap CI
over the Δs = 1.5 m spatially-independent subsample. Val only — no test-set evaluation.
"""
import sys
import json
from pathlib import Path

import numpy as np
import cv2

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "geometric"))
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation.unet_binary.model import UNetBinary
from segmentation.unet_binary.dataset import IMAGENET_MEAN, IMAGENET_STD

import projection_calibration as C
from single_arm_dryrun import side_valid, bin_centre, NEAR_M, BINS, CONF, BLOB_FRAC, FRAME_PX

FRAMES = REPO / "results/runs/geom_cp1_frames_640"
MANIFEST = json.load(open(REPO / "results/geometric/march/dataset_manifest.json"))
OUT_JSON = REPO / "results/geometric/march/superseded/yconstant_val_evaluation/yconstant_val_report.json"
OUT_CSV = REPO / "results/geometric/march/superseded/yconstant_val_evaluation/yconstant_val_per_frame.csv"
UNET_MIN_AREA = 40
BOOT_B = 2000
BOOT_SEED = 42

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

_UNET_TF = A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])


def yolo_base_points(model, img):
    """YOLO bbox-bottom-centre base points; 15%-area blob guard (D035)."""
    r = model.predict(source=img, conf=CONF, half=True, device=0, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return [], 0
    xyxy = r.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    keep = areas <= BLOB_FRAC * FRAME_PX * FRAME_PX
    pts = [((x1 + x2) / 2.0, y2) for (x1, y1, x2, y2) in xyxy[keep]]
    return pts, int((~keep).sum())


def unet_base_points(unet, device, img):
    """U-Net connected-component bbox-bottom-centre base points (>= UNET_MIN_AREA px)."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = _UNET_TF(image=rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = unet(x)
    fg = (logits.argmax(1)[0].cpu().numpy() == 1).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    pts = []
    for k in range(1, n):
        xx, yy, w, h, area = stats[k]
        if area < UNET_MIN_AREA:
            continue
        pts.append((xx + w / 2.0, yy + h - 1))
    return pts, 0


def estimate(base_pts):
    """Shared downstream: base points -> centreline offset/heading (CP-3-locked)."""
    L, R = [], []
    for (uc, v) in base_pts:
        g = C.project_px(uc, v, near_m=NEAR_M)
        if g is not None:
            (L if uc < 320 else R).append(g)
    L = np.array(L) if L else np.empty((0, 2))
    R = np.array(R) if R else np.empty((0, 2))
    vL, vR = side_valid(L), side_valid(R)
    out = {"nL": int(len(L)), "nR": int(len(R)), "n_near": int(len(L) + len(R)),
           "offset": None, "heading": None}
    if vL and vR:
        out["cls"] = "two_row"
        cen = [bin_centre(L, R, lo, hi) for (lo, hi) in BINS]
        out["offset"] = float(cen[0]) if cen[0] is not None else float((vL[0] + vR[0]) / 2)
        if cen[0] is not None and cen[1] is not None:
            dx = np.mean(BINS[1]) - np.mean(BINS[0])
            out["heading"] = float(np.degrees(np.arctan2(cen[1] - cen[0], dx)))
    elif vL or vR:
        out["cls"] = "single_row"
    else:
        out["cls"] = "none"
    return out


def rms(a):
    return float(np.sqrt(np.mean(np.square(a)))) if len(a) else float("nan")


def boot_ci(vals, stat, b=BOOT_B, seed=BOOT_SEED):
    vals = np.asarray(vals, float)
    if len(vals) < 8:
        return [None, None, len(vals)]
    rng = np.random.default_rng(seed)
    n = len(vals)
    bs = [stat(vals[rng.integers(0, n, n)]) for _ in range(b)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3), n]


def run_model(arm, seed, typ, ckpt, val_idx, device):
    if typ == "yolo":
        model = YOLO(str(REPO / "results/runs" / ckpt))
        front = lambda im: yolo_base_points(model, im)
    else:
        model = UNetBinary(encoder_weights=None).to(device).eval()
        ck = torch.load(REPO / "results/runs" / ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        front = lambda im: unet_base_points(model, device, im)
    per = []
    for fi in val_idx:
        img = cv2.imread(str(FRAMES / f"{fi:05d}.jpg"))
        if img is None:
            continue
        pts, blob = front(img)
        e = estimate(pts)
        e["i"] = int(fi)
        e["n_base"] = len(pts)
        e["blob"] = int(blob)
        per.append(e)
    del model
    torch.cuda.empty_cache()
    return per


def summarise(arm, seed, per, sub_ids):
    tot = len(per)
    cov = {c: sum(1 for p in per if p["cls"] == c) for c in ("two_row", "single_row", "none")}
    two = [p for p in per if p["cls"] == "two_row"]
    offs = np.array([p["offset"] for p in two])
    heads = np.array([p["heading"] for p in two if p["heading"] is not None])
    offs_sub = np.array([p["offset"] for p in two if p["i"] in sub_ids])
    heads_sub = np.array([p["heading"] for p in two if p["heading"] is not None and p["i"] in sub_ids])
    blob_frames = sum(1 for p in per if p["blob"] > 0)
    return {
        "arm": arm, "seed": seed, "frames": tot,
        "two_row_pct": round(100 * cov["two_row"] / tot, 1),
        "single_row_pct": round(100 * cov["single_row"] / tot, 1),
        "none_pct": round(100 * cov["none"] / tot, 1),
        "mean_base_points": round(float(np.mean([p["n_base"] for p in per])), 1),
        "mean_near_points": round(float(np.mean([p["n_near"] for p in per])), 1),
        "gt1_offset": {
            "mean": round(float(offs.mean()), 3), "median": round(float(np.median(offs)), 3),
            "sd": round(float(offs.std()), 3), "rms": round(rms(offs), 3),
            "rms_ci_sub": boot_ci(offs_sub, rms), "mean_ci_sub": boot_ci(offs_sub, np.mean),
        },
        "gt2_heading": {
            "mean": round(float(heads.mean()), 2), "median": round(float(np.median(heads)), 2),
            "sd": round(float(heads.std()), 2), "rms": round(rms(heads), 2),
            "rms_ci_sub": boot_ci(heads_sub, rms),
        },
        "anomalies": {
            "blob_frames": blob_frames,
            "systematic_bias_m": round(float(offs.mean()), 3),
            "n_two_row_subsample": int(len(offs_sub)),
        },
    }


def main():
    val = [f for f in MANIFEST["frames"] if f["split"] == "val"]
    val_idx = [f["i"] for f in val]
    sub_ids = set(f["i"] for f in val if f.get("subsample_1p5m"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summaries = []
    csv_rows = ["arm,seed,i,cls,offset,heading,n_base,n_near,blob"]
    for (arm, seed, typ, ckpt) in MODELS:
        print(f"[{arm} s{seed}] running {len(val_idx)} val frames ...", flush=True)
        per = run_model(arm, seed, typ, ckpt, val_idx, device)
        s = summarise(arm, seed, per, sub_ids)
        summaries.append(s)
        for p in per:
            csv_rows.append(f"{arm},{seed},{p['i']},{p['cls']},"
                            f"{'' if p['offset'] is None else round(p['offset'],4)},"
                            f"{'' if p['heading'] is None else round(p['heading'],3)},"
                            f"{p['n_base']},{p['n_near']},{p['blob']}")
        g1, g2 = s["gt1_offset"], s["gt2_heading"]
        print(f"    two-row {s['two_row_pct']}% | single {s['single_row_pct']}% | none {s['none_pct']}% "
              f"| base {s['mean_base_points']} | GT1 RMS {g1['rms']} CI{g1['rms_ci_sub'][:2]} "
              f"| GT2 RMS {g2['rms']} CI{g2['rms_ci_sub'][:2]}", flush=True)

    # cross-seed aggregate per arm
    agg = {}
    for arm in ("A", "B", "C"):
        rows = [s for s in summaries if s["arm"] == arm]
        def ms(key_fn):
            v = np.array([key_fn(s) for s in rows], float)
            return [round(float(v.mean()), 3), round(float(v.std()), 3)]
        agg[arm] = {
            "two_row_pct": ms(lambda s: s["two_row_pct"]),
            "single_row_pct": ms(lambda s: s["single_row_pct"]),
            "none_pct": ms(lambda s: s["none_pct"]),
            "mean_base_points": ms(lambda s: s["mean_base_points"]),
            "gt1_rms": ms(lambda s: s["gt1_offset"]["rms"]),
            "gt1_mean_bias": ms(lambda s: s["gt1_offset"]["mean"]),
            "gt2_rms": ms(lambda s: s["gt2_heading"]["rms"]),
        }

    report = {"config": {"split": "val", "n_frames": len(val_idx),
                         "n_subsample": len(sub_ids), "boot_B": BOOT_B, "boot_seed": BOOT_SEED,
                         "unet_min_area": UNET_MIN_AREA, "blob_frac": BLOB_FRAC, "near_m": NEAR_M},
              "per_model": summaries, "per_arm_across_seeds": agg}
    OUT_JSON.write_text(json.dumps(report, indent=2))
    OUT_CSV.write_text("\n".join(csv_rows))
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_CSV} ({len(csv_rows)-1} rows)")

    print("\n==== cross-seed aggregate (mean +/- SD over seeds) ====")
    print(f"{'arm':>4}{'two-row%':>16}{'base/frm':>14}{'GT1 RMS(m)':>16}{'GT1 bias(m)':>16}{'GT2 RMS(deg)':>16}")
    for arm in ("A", "B", "C"):
        a = agg[arm]
        print(f"{arm:>4}{a['two_row_pct'][0]:>10.1f}+/-{a['two_row_pct'][1]:<4.1f}"
              f"{a['mean_base_points'][0]:>8.1f}+/-{a['mean_base_points'][1]:<4.1f}"
              f"{a['gt1_rms'][0]:>10.3f}+/-{a['gt1_rms'][1]:<5.3f}"
              f"{a['gt1_mean_bias'][0]:>10.3f}+/-{a['gt1_mean_bias'][1]:<5.3f}"
              f"{a['gt2_rms'][0]:>10.2f}+/-{a['gt2_rms'][1]:<5.2f}")


if __name__ == "__main__":
    main()
