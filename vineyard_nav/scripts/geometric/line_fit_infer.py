"""Whole-bag line-fit INFERENCE (D040). Self-contained: runs the full front-end (Phase A UNet /
Phase B+C YOLO base points -> IPM -> far-extension row fit -> line-fit centreline) over ALL eligible
frames of a bag, for all 9 models, and writes the per-frame CSV. Bag-agnostic multi-bag template.

  python3 line_fit_infer.py --bag march      -> results/geometric/march/final/march_evaluation/line_fit_per_frame.csv

Frames are selected on `eligible` alone (whole-bag; no val/test split). The 9 model weights are
bag-independent (the same three-arm models are evaluated on every bag). Output is 12-col, full
precision:  arm,seed,i,cls,offset,heading,mL,mR,mc,n_base,adj,flags
line_fit_eval.py reads this CSV and writes line_fit_report.json (no inference/report here).
"""
import sys
import json
from pathlib import Path

import numpy as np
import cv2

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation.unet_binary.model import UNetBinary
from segmentation.unet_binary.dataset import IMAGENET_MEAN, IMAGENET_STD
import projection_calibration as C
from single_arm_dryrun import CONF, BLOB_FRAC, FRAME_PX
from bag_config import parse_bag
exec(open(Path(__file__).resolve().parent / "row_model.py").read())

B = parse_bag()
FR = B["frames_dir"]
MAN = json.load(open(B["manifest"]))
B["out_dir"].mkdir(parents=True, exist_ok=True)
OUT_CSV = B["per_frame_csv"]
UNET_MIN_AREA = 40
# 9 models (bag-independent weights; identical to the val/test-era evaluators)
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
FRAMES = [f["i"] for f in MAN["frames"] if f["eligible"]]         # whole-bag: eligible only, no split


def yolo_base(model, img):
    r = model.predict(source=img, conf=CONF, half=True, device=0, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xy = r.boxes.xyxy.cpu().numpy()
    ar = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
    return [((x1 + x2) / 2, y2) for (x1, y1, x2, y2) in xy[ar <= BLOB_FRAC * FRAME_PX * FRAME_PX]]


def unet_base(unet, dev, img):
    x = _TF(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"].unsqueeze(0).to(dev)
    with torch.no_grad():
        fg = (unet(x).argmax(1)[0].cpu().numpy() == 1).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    return [(st[k][0] + st[k][2] / 2., st[k][1] + st[k][3] - 1) for k in range(1, n) if st[k][4] >= UNET_MIN_AREA]


def estimate(base_pts):
    L, R = [], []
    for (uc, v) in base_pts:
        g = C.project_px(uc, v, near_m=FARMAX)
        if g is not None:
            (L if uc < 320 else R).append(g)
    L = np.array(L) if L else np.empty((0, 2))
    R = np.array(R) if R else np.empty((0, 2))
    fL, fR = fit_side_far(L), fit_side_far(R)
    adj = int(bool(fL.get("adjacent"))) + int(bool(fR.get("adjacent")))
    o = {"n_base": len(base_pts), "adj": adj}
    if fL["ok"] and fR["ok"]:
        cl = centre_linefit(L[fL["inl"]], R[fR["inl"]])
        if cl is None:
            o["cls"] = "fitfail"
            return o
        o.update(cls="two_row", offset=cl["offset"], heading=cl["heading"],
                 mL=cl["m_L"], mR=cl["m_R"], mc=cl["m_c"], width=cl["width"], flags=cl["flags"])
    elif fL["ok"] or fR["ok"]:
        o["cls"] = "single_row"
    else:
        o["cls"] = "none"
    return o


dev = torch.device("cuda")
csv = ["arm,seed,i,cls,offset,heading,mL,mR,mc,n_base,adj,flags"]
for (arm, seed, typ, ckpt) in MODELS:
    print(f"[{B['bag']}][{arm} s{seed}] {len(FRAMES)} frames ...", flush=True)
    if typ == "yolo":
        m = YOLO(str(PKG / "results/runs" / ckpt))
        front = lambda im: yolo_base(m, im)
    else:
        m = UNetBinary(encoder_weights=None).to(dev).eval()
        m.load_state_dict(torch.load(PKG / "results/runs" / ckpt, map_location=dev, weights_only=False)["model_state_dict"])
        front = lambda im: unet_base(m, dev, im)
    for fi in FRAMES:
        img = cv2.imread(str(FR / f"{fi:05d}.jpg"))
        if img is None:
            continue
        e = estimate(front(img))
        fl = "|".join(e.get("flags", []))
        csv.append(f"{arm},{seed},{fi},{e['cls']},{e.get('offset','')},{e.get('heading','')},"
                   f"{e.get('mL','')},{e.get('mR','')},{e.get('mc','')},{e['n_base']},{e['adj']},{fl}")
    del m
    torch.cuda.empty_cache()
OUT_CSV.write_text("\n".join(csv))
print(f"wrote {OUT_CSV} ({len(csv) - 1} rows = 9 models x {len(FRAMES)} frames)")
