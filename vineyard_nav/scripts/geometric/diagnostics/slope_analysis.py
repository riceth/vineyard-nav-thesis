"""Convergence investigation: for every far-extension two-row val frame (Phase C s42), fit
Y = mX + c per side to the inliers and record slopes. Determine whether the apparent
inward convergence is systematic (A) or noise (B). Sign convention (ROS Y, +Y=left):
  left row Y>0 converging inward => m_L < 0 ; right row Y<0 converging inward => m_R > 0.
  width slope (m_L - m_R) < 0  => corridor narrows with range (convergence)."""
import sys, json
from pathlib import Path
import numpy as np, cv2
PKG = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG/"scripts"/"geometric"))
import torch; torch.multiprocessing.set_sharing_strategy("file_system")
from ultralytics import YOLO
import projection_calibration as C
from cp3_geometry import bin_centre, CONF, BLOB_FRAC, FRAME_PX
exec(open(Path(__file__).resolve().parents[1] / "row_model.py").read())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

FR = PKG / "results/runs/geom_cp1_frames_640"; MAN = json.load(open(PKG/"results/geometric/march/dataset_manifest.json"))
val = [f["i"] for f in MAN["frames"] if f["split"] == "val"]
model = YOLO(str(PKG / "results/runs/phase_c_yolo_multiclass/weights/best.pt"))
SC = str(PKG / "results/geometric/march/diagnostics/figures/rowfit_validation")

mL, mR, nL, nR, yL, yR = [], [], [], [], [], []
off_const, off_line, hdg_line = [], [], []
for fi in val:
    img = cv2.imread(str(FR / f"{fi:05d}.jpg"))
    r = model.predict(source=img, conf=CONF, quantize=16, device=0, verbose=False)[0]
    L, R = [], []
    if r.boxes is not None and len(r.boxes):
        xy = r.boxes.xyxy.cpu().numpy(); ar=(xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1])
        for (x1,y1,x2,y2) in xy[ar <= BLOB_FRAC*FRAME_PX*FRAME_PX]:
            g = C.project_px((x1+x2)/2, y2, near_m=FARMAX)
            if g is not None: (L if (x1+x2)/2<320 else R).append(g)
    L = np.array(L) if L else np.empty((0,2)); R = np.array(R) if R else np.empty((0,2))
    fL, fR = fit_side_far(L), fit_side_far(R)
    if not (fL["ok"] and fR["ok"]): continue
    Li, Ri = L[fL["inl"]], R[fR["inl"]]
    if len(Li) < 3 or len(Ri) < 3: continue
    pL = np.polyfit(Li[:,0], Li[:,1], 1); pR = np.polyfit(Ri[:,0], Ri[:,1], 1)
    mL.append(pL[0]); mR.append(pR[0]); nL.append(len(Li)); nR.append(len(Ri))
    yL.append(np.median(Li[:,1])); yR.append(np.median(Ri[:,1]))
    off_const.append((np.median(Li[:,1]) + np.median(Ri[:,1]))/2)                  # Y-const centreline offset
    off_line.append(((pL[0]*2+pL[1]) + (pR[0]*2+pR[1]))/2)                          # line-fit centreline @ X=2m
    hdg_line.append(np.degrees(np.arctan2(-((pL[0]+pR[0])/2), 1)))                  # centreline slope -> heading (deg), +Y=left
mL, mR, nL, nR = map(np.array, (mL, mR, nL, nR))
oc, ol, hl = np.array(off_const), np.array(off_line), np.array(hdg_line)
N = len(mL)
def s(a): return f"mean {a.mean():+.4f} median {np.median(a):+.4f} SD {a.std():.4f} [p10 {np.percentile(a,10):+.3f}, p90 {np.percentile(a,90):+.3f}]"
print(f"=== slope analysis, {N} far-ext two-row val frames (Phase C s42) ===")
print(f"m_L (left, <0 = converge inward):  {s(mL)}")
print(f"m_R (right, >0 = converge inward): {s(mR)}")
print(f"width slope m_L - m_R (<0 = corridor narrows with range): {s(mL-mR)}")
print(f"fraction of frames with m_L<0 AND m_R>0 (both inward): {100*np.mean((mL<0)&(mR>0)):.1f}%")
print(f"fraction converging (m_L-m_R<0): {100*np.mean(mL-mR<0):.1f}%")
print(f"corr(m_L, m_R): {np.corrcoef(mL,mR)[0,1]:+.3f}   (mirror convergence -> negative)")
print(f"corr(|m_L|, nL): {np.corrcoef(np.abs(mL),nL)[0,1]:+.3f}   corr(|m_R|, nR): {np.corrcoef(np.abs(mR),nR)[0,1]:+.3f}   (density vs slope mag)")
print(f"mean convergence over X=2->7m: left {5*mL.mean():+.3f} m, right {5*mR.mean():+.3f} m")
print(f"--- centreline: Y-const vs line-fit ---")
print(f"offset  Y-const RMS {np.sqrt(np.mean(oc**2)):.3f}  line-fit@2m RMS {np.sqrt(np.mean(ol**2)):.3f}  mean|diff| {np.mean(np.abs(oc-ol)):.4f} max {np.max(np.abs(oc-ol)):.3f}")
print(f"line-fit centreline heading: {s(hl)}  RMS {np.sqrt(np.mean(hl**2)):.2f} deg")
# histogram
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].hist(mL, bins=60, alpha=.6, color="b", label=f"m_L (mean {mL.mean():+.3f})")
ax[0].hist(mR, bins=60, alpha=.6, color="r", label=f"m_R (mean {mR.mean():+.3f})")
ax[0].axvline(0, color="k", lw=.8); ax[0].set_xlabel("row slope m = dY/dX (m/m)"); ax[0].set_xlim(-0.6,0.6); ax[0].legend(); ax[0].set_title("per-side row slopes")
ax[1].hist(mL-mR, bins=60, color="purple", alpha=.7); ax[1].axvline(0, color="k", lw=.8)
ax[1].axvline((mL-mR).mean(), color="orange", lw=2, label=f"mean {np.mean(mL-mR):+.3f}")
ax[1].set_xlabel("width slope m_L - m_R (<0 = converge)"); ax[1].set_xlim(-1,1); ax[1].legend(); ax[1].set_title("corridor width slope")
fig.tight_layout(); fig.savefig(f"{SC}/linefit/slope_hist.png", dpi=110); plt.close(fig)
print("saved slope_hist.png")
np.savez(f"{SC}/slopes.npz", mL=mL, mR=mR, nL=nL, nR=nR)
