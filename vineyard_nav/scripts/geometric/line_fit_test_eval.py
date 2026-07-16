"""CP-6 SINGLE-SHOT TEST evaluation (rule 5). All 9 models (A/B/C x seeds 42/43/44) on the
4 held-out test corridors (3,149 frames), class-agnostic downstream (F018 lock), locked
line-fit pipeline (D036/D037/D038). Writes per-frame CSV + per-model coverage/GT-1/GT-2 RMS.
Block-bootstrap CIs + cross-arm paired analysis run offline on the CSV afterward."""
import sys, json, collections
from pathlib import Path
import numpy as np, cv2
PKG = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG/"scripts"/"geometric"))
import torch; torch.multiprocessing.set_sharing_strategy("file_system")
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from segmentation.unet_binary.model import UNetBinary
from segmentation.unet_binary.dataset import IMAGENET_MEAN, IMAGENET_STD
import projection_calibration as C
from single_arm_dryrun import CONF, BLOB_FRAC, FRAME_PX
exec(open(Path(__file__).resolve().parent / "row_model.py").read())

FR = PKG/"results/runs/geom_cp1_frames_640"
MAN = json.load(open(PKG/"results/geometric/march/dataset_manifest.json"))
TEST = [f["i"] for f in MAN["frames"] if f["split"] == "test"]
OUT_CSV = PKG/"results/geometric/march/final/test_evaluation/line_fit_test_per_frame.csv"
OUT_JSON = PKG/"results/geometric/march/final/test_evaluation/line_fit_test_report.json"
UNET_MIN = 40
MODELS = [
    ("A",42,"unet","phase_a_unet_binary_20260704_004105/checkpoints/best.pt"),
    ("A",43,"unet","phase_a_unet_binary_seed43_20260710_154347/checkpoints/best.pt"),
    ("A",44,"unet","phase_a_unet_binary_seed44_20260710_181339/checkpoints/best.pt"),
    ("B",42,"yolo","phase_b_yolo_binary/weights/best.pt"),
    ("B",43,"yolo","phase_b_yolo_binary_seed43/weights/best.pt"),
    ("B",44,"yolo","phase_b_yolo_binary_seed44/weights/best.pt"),
    ("C",42,"yolo","phase_c_yolo_multiclass/weights/best.pt"),
    ("C",43,"yolo","phase_c_yolo_multiclass_seed43/weights/best.pt"),
    ("C",44,"yolo","phase_c_yolo_multiclass_seed44/weights/best.pt"),
]
_TF = A.Compose([A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])

def yolo_base(m, img):
    r = m.predict(source=img, conf=CONF, half=True, device=0, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0: return []
    xy = r.boxes.xyxy.cpu().numpy(); ar = (xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1])
    return [((x1+x2)/2, y2) for (x1,y1,x2,y2) in xy[ar <= BLOB_FRAC*FRAME_PX*FRAME_PX]]

def unet_base(u, dev, img):
    x = _TF(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))["image"].unsqueeze(0).to(dev)
    with torch.no_grad(): fg = (u(x).argmax(1)[0].cpu().numpy() == 1).astype(np.uint8)
    n,_,st,_ = cv2.connectedComponentsWithStats(fg, 8)
    return [(st[k][0]+st[k][2]/2., st[k][1]+st[k][3]-1) for k in range(1,n) if st[k][4] >= UNET_MIN]

def estimate(bp):
    L, R = [], []
    for (uc, v) in bp:
        g = C.project_px(uc, v, near_m=FARMAX)
        if g is not None: (L if uc < 320 else R).append(g)
    L = np.array(L) if L else np.empty((0,2)); R = np.array(R) if R else np.empty((0,2))
    fL, fR = fit_side_far(L), fit_side_far(R)
    if fL["ok"] and fR["ok"]:
        cl = centre_linefit(L[fL["inl"]], R[fR["inl"]])
        if cl: return ("two_row", cl["offset"], cl["heading"], cl["m_c"], len(bp))
    return ("single_row" if (fL["ok"] or fR["ok"]) else "none", None, None, None, len(bp))

def rms(a): a = np.asarray(a, float); return float(np.sqrt(np.mean(a**2))) if len(a) else float("nan")

dev = torch.device("cuda")
csv = ["arm,seed,i,cls,offset,heading,mc,n_base"]; summ = []
for (arm, seed, typ, ckpt) in MODELS:
    print(f"[{arm} s{seed}] {len(TEST)} test frames ...", flush=True)
    if typ == "yolo": m = YOLO(str(PKG/"results/runs"/ckpt)); front = lambda im: yolo_base(m, im)
    else:
        m = UNetBinary(encoder_weights=None).to(dev).eval()
        m.load_state_dict(torch.load(PKG/"results/runs"/ckpt, map_location=dev, weights_only=False)["model_state_dict"])
        front = lambda im: unet_base(m, dev, im)
    cov = collections.Counter(); offs = []; hdgs = []; nb = []
    for fi in TEST:
        img = cv2.imread(str(FR/f"{fi:05d}.jpg"))
        if img is None: continue
        cls, o, h, mc, n = estimate(front(img)); cov[cls] += 1; nb.append(n)
        csv.append(f"{arm},{seed},{fi},{cls},{'' if o is None else round(o,4)},{'' if h is None else round(h,3)},{'' if mc is None else round(mc,4)},{n}")
        if cls == "two_row": offs.append(o); hdgs.append(h)
    del m; torch.cuda.empty_cache()
    tot = len(TEST)
    s = {"arm": arm, "seed": seed, "two_row_pct": round(100*cov["two_row"]/tot,1),
         "single_pct": round(100*cov["single_row"]/tot,1), "none_pct": round(100*cov["none"]/tot,1),
         "mean_base": round(float(np.mean(nb)),1), "gt1_rms": round(rms(offs),4),
         "gt1_mean": round(float(np.mean(offs)),4), "gt2_rms": round(rms(hdgs),3), "gt2_mean": round(float(np.mean(hdgs)),3)}
    summ.append(s)
    print(f"    two {s['two_row_pct']}% | base {s['mean_base']} | GT1 RMS {s['gt1_rms']} (mean {s['gt1_mean']}) | GT2 RMS {s['gt2_rms']} (mean {s['gt2_mean']})", flush=True)
OUT_CSV.write_text("\n".join(csv))
agg = {}
for arm in "ABC":
    rs = [s for s in summ if s["arm"] == arm]
    ms = lambda k: [round(float(np.mean([r[k] for r in rs])),3), round(float(np.std([r[k] for r in rs])),3)]
    agg[arm] = {k: ms(k) for k in ("two_row_pct","mean_base","gt1_rms","gt1_mean","gt2_rms","gt2_mean")}
json.dump({"split":"test","n":len(TEST),"config":"class-agnostic (F018)","per_model":summ,"per_arm":agg},
          open(OUT_JSON,"w"), indent=2)
print("\n==== cross-seed (mean +/- SD) ====")
for arm in "ABC":
    a = agg[arm]
    print(f"  {arm}: two-row {a['two_row_pct'][0]}+/-{a['two_row_pct'][1]}% | GT1 RMS {a['gt1_rms'][0]}+/-{a['gt1_rms'][1]} | GT2 RMS {a['gt2_rms'][0]}+/-{a['gt2_rms'][1]} | base {a['mean_base'][0]}")
print(f"wrote {OUT_CSV.name}, {OUT_JSON.name}")
