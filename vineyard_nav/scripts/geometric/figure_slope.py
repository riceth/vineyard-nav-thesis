"""Sample plots: Y-constant fit (dashed vertical) vs line-fit Y=mX+c (solid sloped) per side,
with both centrelines, over the far-ext inliers. 8 frames incl. Edosa's 3 (4107/4223/3998)."""
import sys, json
from pathlib import Path
import numpy as np, cv2
PKG = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG/"scripts"/"geometric"))
import torch; torch.multiprocessing.set_sharing_strategy("file_system")
from ultralytics import YOLO
import projection_calibration as C
from single_arm_dryrun import CONF, BLOB_FRAC, FRAME_PX
exec(open(Path(__file__).resolve().parent / "row_model.py").read())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
SC = str(PKG / "results/geometric/march/diagnostics/figures/rowfit_validation")
FR = PKG / "results/runs/geom_cp1_frames_640"
model = YOLO(str(PKG / "results/runs/phase_c_yolo_multiclass/weights/best.pt"))

def base(img):
    r = model.predict(source=img, conf=CONF, half=True, device=0, verbose=False)[0]
    if r.boxes is None or len(r.boxes)==0: return []
    xy=r.boxes.xyxy.cpu().numpy(); ar=(xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1])
    return [((x1+x2)/2,y2) for (x1,y1,x2,y2) in xy[ar<=BLOB_FRAC*FRAME_PX*FRAME_PX]]

for fi in [4107, 4223, 3998, 3991, 3992, 3994, 3996, 3997]:
    img = cv2.imread(str(FR / f"{fi:05d}.jpg"))
    L, R = [], []
    for (uc,v) in base(img):
        g = C.project_px(uc, v, near_m=FARMAX)
        if g is not None: (L if uc<320 else R).append(g)
    L=np.array(L) if L else np.empty((0,2)); R=np.array(R) if R else np.empty((0,2))
    fL, fR = fit_side_far(L), fit_side_far(R)
    if not (fL["ok"] and fR["ok"]): continue
    Li, Ri = L[fL["inl"]], R[fR["inl"]]
    pL = np.polyfit(Li[:,0], Li[:,1], 1); pR = np.polyfit(Ri[:,0], Ri[:,1], 1)
    xs = np.array([0, 9.0])
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].axis("off"); ax[0].set_title(f"frame {fi}")
    for P, f, p, col in ((L, fL, pL, "b"), (R, fR, pR, "r")):
        inl_n, inl_f = f["inl_near"], f["inl_far"]; rej = ~(inl_n | inl_f)
        ax[1].scatter(-P[inl_n,1], P[inl_n,0], c=col, s=32, marker="o")
        ax[1].scatter(-P[inl_f,1], P[inl_f,0], c=col, s=46, marker="^")
        ax[1].scatter(-P[rej,1], P[rej,0], facecolors="none", edgecolors=col, s=28)
        ax[1].plot([-f["y"], -f["y"]], [0, 9], col+"--", lw=1.6)          # Y-const (vertical)
        ax[1].plot(-(p[0]*xs+p[1]), xs, col+"-", lw=2.2)                  # line-fit (sloped)
    # centrelines
    yc = (np.median(Li[:,1]) + np.median(Ri[:,1]))/2
    ax[1].plot([-yc, -yc], [0, 9], "g--", lw=1.6, label=f"Y-const centre off={yc:+.2f}")
    cl = ((pL[0]*xs+pL[1]) + (pR[0]*xs+pR[1]))/2
    off2 = ((pL[0]*2+pL[1]) + (pR[0]*2+pR[1]))/2
    hdg = np.degrees(np.arctan2(-((pL[0]+pR[0])/2), 1))
    ax[1].plot(-cl, xs, "g-", lw=2.4, label=f"line-fit centre off@2m={off2:+.2f} hdg={hdg:+.1f}")
    ax[1].axvline(0, color="0.6", lw=.5); ax[1].set_xlim(-4, 4); ax[1].set_ylim(0, 9)
    ax[1].set_xlabel("-Y (right +, m)"); ax[1].set_ylabel("X fwd (m)"); ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8, loc="upper left")
    ax[1].set_title(f"dashed=Y-const  solid=line-fit   m_L={pL[0]:+.3f} m_R={pR[0]:+.3f}", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{SC}/linefit/linefit_f{fi}.png", dpi=100); plt.close(fig)
print("saved linefit_f*.png")
