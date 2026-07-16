"""Line-fit validation: 10 frames (8 CP-3 dry-run + CP-2 4107/4223). Plot Y-const (dashed)
vs line-fit (solid) per side + both centrelines, with offset@2m/heading/flags. Then scan all
val (Phase C s42) for slope-quality flags + tilt distribution."""
import sys, json
from pathlib import Path
import numpy as np, cv2
PKG = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG/"scripts"/"geometric"))
import torch; torch.multiprocessing.set_sharing_strategy("file_system")
from ultralytics import YOLO
import projection_calibration as C
from single_arm_dryrun import CONF, BLOB_FRAC, FRAME_PX
exec(open(Path(__file__).resolve().parents[1] / "row_model.py").read())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SC = str(PKG / "results/geometric/march/diagnostics/figures/rowfit_validation")
FR = PKG / "results/runs/geom_cp1_frames_640"; MAN = json.load(open(PKG/"results/geometric/march/dataset_manifest.json"))
model = YOLO(str(PKG / "results/runs/phase_c_yolo_multiclass/weights/best.pt"))

def project(img):
    r = model.predict(source=img, conf=CONF, half=True, device=0, verbose=False)[0]
    L, R = [], []
    if r.boxes is not None and len(r.boxes):
        xy=r.boxes.xyxy.cpu().numpy(); ar=(xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1])
        for (x1,y1,x2,y2) in xy[ar<=BLOB_FRAC*FRAME_PX*FRAME_PX]:
            g = C.project_px((x1+x2)/2, y2, near_m=FARMAX)
            if g is not None: (L if (x1+x2)/2<320 else R).append(g)
    return (np.array(L) if L else np.empty((0,2))), (np.array(R) if R else np.empty((0,2)))

# ---- 10 validation plots ----
for fi in [3991,3992,3993,3994,3995,3996,3997,3998,4107,4223]:
    img = cv2.imread(str(FR / f"{fi:05d}.jpg")); L, R = project(img)
    fL, fR = fit_side_far(L), fit_side_far(R)
    if not (fL["ok"] and fR["ok"]): print(f"frame {fi}: not two-row ({fL.get('reason')},{fR.get('reason')})"); continue
    Li, Ri = L[fL["inl"]], R[fR["inl"]]; cl = centre_linefit(Li, Ri)
    mL, cL_, mR, cR_ = cl["lines"]; xs = np.array([0, 9.0])
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].axis("off"); ax[0].set_title(f"frame {fi}")
    for P, f, p, col in ((L, fL, (mL,cL_), "b"), (R, fR, (mR,cR_), "r")):
        inl_n, inl_f = f["inl_near"], f["inl_far"]; rej = ~(inl_n|inl_f)
        ax[1].scatter(-P[inl_n,1], P[inl_n,0], c=col, s=32, marker="o")
        ax[1].scatter(-P[inl_f,1], P[inl_f,0], c=col, s=46, marker="^")
        ax[1].scatter(-P[rej,1], P[rej,0], facecolors="none", edgecolors=col, s=26)
        ax[1].plot([-f["y"], -f["y"]], [0,9], col+"--", lw=1.4)               # Y-const
        ax[1].plot(-(p[0]*xs+p[1]), xs, col+"-", lw=2.2)                       # line-fit
    yc = (np.median(Li[:,1])+np.median(Ri[:,1]))/2
    ax[1].plot([-yc,-yc],[0,9], "g--", lw=1.4, label=f"Y-const centre {yc:+.2f}")
    clc = ((mL*xs+cL_)+(mR*xs+cR_))/2
    ax[1].plot(-clc, xs, "g-", lw=2.4, label=f"line-fit off@2m={cl['offset']:+.2f} hdg={cl['heading']:+.1f}")
    ax[1].axvline(0, color="0.6", lw=.5); ax[1].set_xlim(-4,4); ax[1].set_ylim(0,9)
    ax[1].set_xlabel("-Y (right +, m)"); ax[1].set_ylabel("X fwd (m)"); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8, loc="upper left")
    ax[1].set_title(f"dashed=Y-const solid=line-fit  m_L={mL:+.3f} m_R={mR:+.3f} {cl['flags']}", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{SC}/linefit_final/lfval_f{fi}.png", dpi=100); plt.close(fig)
print("saved lfval_f*.png")

# ---- val quality scan ----
val = [f["i"] for f in MAN["frames"] if f["split"]=="val"]
steep=mismatch=fitfail=two=0; mLs=[]; mRs=[]; mCs=[]
for fi in val:
    L, R = project(cv2.imread(str(FR / f"{fi:05d}.jpg")))
    fL, fR = fit_side_far(L), fit_side_far(R)
    if not (fL["ok"] and fR["ok"]): continue
    two+=1; cl = centre_linefit(L[fL["inl"]], R[fR["inl"]])
    if cl is None: fitfail+=1; continue
    mLs.append(cl["m_L"]); mRs.append(cl["m_R"]); mCs.append(cl["m_c"])
    if "steep_slope" in cl["flags"]: steep+=1
    if "slope_mismatch" in cl["flags"]: mismatch+=1
mLs,mRs,mCs=map(np.array,(mLs,mRs,mCs))
print(f"=== val quality scan (Phase C s42, {two} two-row frames) ===")
print(f"steep slope (|m|>0.3): {steep} ({100*steep/two:.1f}%) | slope mismatch (|mL-mR|>0.2): {mismatch} ({100*mismatch/two:.1f}%) | linefit fail: {fitfail}")
print(f"tilt m_L: mean {mLs.mean():+.3f} SD {mLs.std():.3f} | m_R: mean {mRs.mean():+.3f} SD {mRs.std():.3f} | m_centre: mean {mCs.mean():+.3f} SD {mCs.std():.3f}")
print(f"m_centre -> heading: mean {np.degrees(np.arctan(mCs.mean())):+.2f} deg")
