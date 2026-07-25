"""CP-5 RE-RUN with line-fit centreline (far-extension inliers). All 9 models x 4708 val
frames. Reports coverage, GT-1 (line-fit@2m) + CI, GT-2 (slope) + CI, base pts, deltas vs the
buggy Y-const CP-5, rescued/adjacent/tilt/quality-flag stats. Val only. Nothing committed."""
import sys, json
from pathlib import Path
import numpy as np, cv2, collections
PKG = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG/"scripts"/"geometric"))
import torch; torch.multiprocessing.set_sharing_strategy("file_system")
from ultralytics import YOLO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scripts.perception.segmentation.unet_binary.model import UNetBinary
from scripts.perception.segmentation.unet_binary.dataset import IMAGENET_MEAN, IMAGENET_STD
import projection_calibration as C
from cp3_geometry import CONF, BLOB_FRAC, FRAME_PX
exec(open(Path(__file__).resolve().parent / "row_model.py").read())

FR = PKG / "results/runs/geom_cp1_frames_640"
MAN = json.load(open(PKG / "results/geometric/march/dataset_manifest.json"))
OLD_CSV = PKG / "results/geometric/march/superseded/yconstant_val_evaluation/yconstant_val_per_frame.csv"
OUT_JSON = PKG / "results/geometric/march/superseded/march_val_test_split/val_evaluation/line_fit_val_report.json"
OUT_CSV = PKG / "results/geometric/march/superseded/march_val_test_split/val_evaluation/line_fit_val_per_frame.csv"
UNET_MIN_AREA = 40; BOOT_B = 2000; BOOT_SEED = 42
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
FRAME_META = {f["i"]: (f["corridor"], f["pass_id"]) for f in MAN["frames"]}
SUB = set(f["i"] for f in MAN["frames"] if f["split"]=="val" and f.get("subsample_1p5m"))
VAL = [f["i"] for f in MAN["frames"] if f["split"]=="val"]

# old (buggy) per-frame: (arm,seed,i)->(cls,offset,heading)
OLD = {}
for ln in OLD_CSV.read_text().splitlines()[1:]:
    a,s,i,cls,off,hdg,*_ = ln.split(",")
    OLD[(a,int(s),int(i))] = (cls, float(off) if off else None, float(hdg) if hdg else None)

def yolo_base(model, img):
    r = model.predict(source=img, conf=CONF, half=True, device=0, verbose=False)[0]
    if r.boxes is None or len(r.boxes)==0: return []
    xy=r.boxes.xyxy.cpu().numpy(); ar=(xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1])
    return [((x1+x2)/2,y2) for (x1,y1,x2,y2) in xy[ar<=BLOB_FRAC*FRAME_PX*FRAME_PX]]

def unet_base(unet, dev, img):
    x=_TF(image=cv2.cvtColor(img,cv2.COLOR_BGR2RGB))["image"].unsqueeze(0).to(dev)
    with torch.no_grad(): fg=(unet(x).argmax(1)[0].cpu().numpy()==1).astype(np.uint8)
    n,_,st,_=cv2.connectedComponentsWithStats(fg,8)
    return [(st[k][0]+st[k][2]/2., st[k][1]+st[k][3]-1) for k in range(1,n) if st[k][4]>=UNET_MIN_AREA]

def estimate(base_pts):
    L,R=[],[]
    for (uc,v) in base_pts:
        g=C.project_px(uc,v,near_m=FARMAX)
        if g is not None: (L if uc<320 else R).append(g)
    L=np.array(L) if L else np.empty((0,2)); R=np.array(R) if R else np.empty((0,2))
    fL,fR=fit_side_far(L),fit_side_far(R)
    adj = int(bool(fL.get("adjacent"))) + int(bool(fR.get("adjacent")))
    o={"n_base":len(base_pts),"adj":adj}
    if fL["ok"] and fR["ok"]:
        cl=centre_linefit(L[fL["inl"]], R[fR["inl"]])
        if cl is None: o["cls"]="fitfail"; return o
        o.update(cls="two_row", offset=cl["offset"], heading=cl["heading"],
                 mL=cl["m_L"], mR=cl["m_R"], mc=cl["m_c"], width=cl["width"], flags=cl["flags"])
    elif fL["ok"] or fR["ok"]: o["cls"]="single_row"
    else: o["cls"]="none"
    return o

def rms(a): a=np.array(a,float); return float(np.sqrt(np.mean(a**2))) if len(a) else float("nan")
def boot(vals, stat, b=BOOT_B, seed=BOOT_SEED):
    vals=np.asarray(vals,float)
    if len(vals)<8: return [None,None,len(vals)]
    rng=np.random.default_rng(seed); n=len(vals)
    bs=[stat(vals[rng.integers(0,n,n)]) for _ in range(b)]
    return [round(float(np.percentile(bs,2.5)),3), round(float(np.percentile(bs,97.5)),3), n]

dev=torch.device("cuda")
summaries=[]; csv=["arm,seed,i,cls,offset,heading,mL,mR,mc,n_base,adj,flags"]
adj_by_corr=collections.Counter(); adj_by_pass=collections.Counter()
for (arm,seed,typ,ckpt) in MODELS:
    print(f"[{arm} s{seed}] ...", flush=True)
    if typ=="yolo": m=YOLO(str(PKG/"results/runs"/ckpt)); front=lambda im: yolo_base(m,im)
    else:
        m=UNetBinary(encoder_weights=None).to(dev).eval()
        m.load_state_dict(torch.load(PKG/"results/runs"/ckpt,map_location=dev,weights_only=False)["model_state_dict"])
        front=lambda im: unet_base(m,dev,im)
    per=[]
    for fi in VAL:
        img=cv2.imread(str(FR/f"{fi:05d}.jpg"))
        if img is None: continue
        e=estimate(front(img)); e["i"]=fi; per.append(e)
        fl="|".join(e.get("flags",[]))
        csv.append(f"{arm},{seed},{fi},{e['cls']},{e.get('offset','')},{e.get('heading','')},"
                   f"{e.get('mL','')},{e.get('mR','')},{e.get('mc','')},{e['n_base']},{e['adj']},{fl}")
        if e["cls"]=="two_row" and e["adj"]:
            c,p=FRAME_META[fi]; adj_by_corr[c]+=1; adj_by_pass[p]+=1
    del m; torch.cuda.empty_cache()
    # aggregate this model
    tot=len(per); cov=collections.Counter(e["cls"] for e in per)
    two=[e for e in per if e["cls"]=="two_row"]
    off=[e["offset"] for e in two]; hdg=[e["heading"] for e in two]
    offs_sub=[e["offset"] for e in two if e["i"] in SUB]; hdgs_sub=[e["heading"] for e in two if e["i"] in SUB]
    # deltas vs old
    d1=d2=0; resc=0
    for e in two:
        old=OLD.get((arm,seed,e["i"]))
        if old and old[0]=="two_row" and old[1] is not None and abs(e["offset"]-old[1])>0.1: d1+=1
        if old and old[0]=="two_row" and old[2] is not None and abs(e["heading"]-old[2])>5: d2+=1
    for e in per:
        old=OLD.get((arm,seed,e["i"]))
        if e["cls"]=="two_row" and old and old[0]!="two_row": resc+=1
    mL=[e["mL"] for e in two]; mR=[e["mR"] for e in two]; mc=[e["mc"] for e in two]
    steep=sum(1 for e in two if "steep_slope" in e.get("flags",[]))
    mism=sum(1 for e in two if "slope_mismatch" in e.get("flags",[]))
    fail=cov.get("fitfail",0)
    summaries.append({"arm":arm,"seed":seed,"frames":tot,
        "two_row_pct":round(100*cov['two_row']/tot,1),"single_pct":round(100*cov['single_row']/tot,1),
        "none_pct":round(100*cov.get('none',0)/tot,1),
        "mean_base":round(float(np.mean([e['n_base'] for e in per])),1),
        "gt1_rms":round(rms(off),3),"gt1_mean":round(float(np.mean(off)),3),"gt1_ci":boot(offs_sub,rms),
        "gt2_rms":round(rms(hdg),2),"gt2_mean":round(float(np.mean(hdg)),2),"gt2_ci":boot(hdgs_sub,rms),
        "delta_gt1_gt0p1":d1,"delta_gt2_gt5":d2,"rescued":resc,
        "adj_frames":sum(1 for e in two if e["adj"]),
        "mL_mean":round(float(np.mean(mL)),3),"mR_mean":round(float(np.mean(mR)),3),
        "mc_mean":round(float(np.mean(mc)),3),"tilt_deg":round(float(np.degrees(np.arctan(np.mean(mc)))),2),
        "steep":steep,"mismatch":mism,"fitfail":fail})
    s=summaries[-1]
    print(f"  two {s['two_row_pct']}% | GT1 {s['gt1_rms']} CI{s['gt1_ci'][:2]} | GT2 {s['gt2_rms']} CI{s['gt2_ci'][:2]} "
          f"| base {s['mean_base']} | tilt {s['tilt_deg']} | resc {s['rescued']} | d1>{s['delta_gt1_gt0p1']} d2>{s['delta_gt2_gt5']} "
          f"| steep {s['steep']} mism {s['mismatch']} fail {s['fitfail']}", flush=True)

agg={}
for arm in ("A","B","C"):
    rs=[s for s in summaries if s["arm"]==arm]
    def ms(k): v=np.array([r[k] for r in rs],float); return [round(float(v.mean()),3),round(float(v.std()),3)]
    agg[arm]={k:ms(k) for k in ("two_row_pct","gt1_rms","gt2_rms","mean_base","tilt_deg")}
report={"config":{"split":"val","n":len(VAL),"n_sub":len(SUB),"row_model":"line-fit@2m + slope",
        "far_ext":True,"boot_B":BOOT_B},"per_model":summaries,"per_arm":agg,
        "adjacent_by_corridor":dict(adj_by_corr),"adjacent_by_pass":dict(adj_by_pass)}
OUT_JSON.write_text(json.dumps(report,indent=2)); OUT_CSV.write_text("\n".join(csv))
print("\n==== cross-seed (mean +/- SD) ====")
print(f"{'arm':>4}{'two-row%':>15}{'GT1 RMS':>14}{'GT2 RMS':>14}{'base':>12}{'tilt deg':>13}")
for arm in ("A","B","C"):
    a=agg[arm]
    print(f"{arm:>4}{a['two_row_pct'][0]:>9.1f}+/-{a['two_row_pct'][1]:<4.1f}{a['gt1_rms'][0]:>8.3f}+/-{a['gt1_rms'][1]:<5.3f}"
          f"{a['gt2_rms'][0]:>7.2f}+/-{a['gt2_rms'][1]:<5.2f}{a['mean_base'][0]:>7.1f}+/-{a['mean_base'][1]:<4.1f}{a['tilt_deg'][0]:>8.2f}+/-{a['tilt_deg'][1]:<4.2f}")
print(f"\nadjacent by corridor: {dict(adj_by_corr)}\nadjacent by pass: {dict(sorted(adj_by_pass.items()))}")
print(f"wrote {OUT_JSON.name}, {OUT_CSV.name}")
