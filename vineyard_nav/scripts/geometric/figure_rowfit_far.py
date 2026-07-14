"""Validate far-field extension (fit_side_far) vs near-5m hybrid. Project to 10m; plot
[image | OLD near-5m | NEW far-extended] with X to 8m. Near inliers = filled circle,
far inliers = filled triangle, rejected = hollow, adjacent corridor = dotted line + flag."""
import sys, json
from pathlib import Path
import numpy as np, cv2
PKG = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG/"scripts"/"geometric"))
import torch; torch.multiprocessing.set_sharing_strategy("file_system")
from ultralytics import YOLO
import projection_calibration as C
from single_arm_dryrun import bin_centre, CONF, BLOB_FRAC, FRAME_PX
exec(open(Path(__file__).resolve().parent / "row_model.py").read())
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SC = str(PKG / "results/geometric/march/diagnostics/figures/rowfit_validation")
FR = PKG / "results/runs/geom_cp1_frames_640"
model = YOLO(str(PKG / "results/runs/phase_c_yolo_multiclass/weights/best.pt"))

def project_all(pts):
    L, R = [], []
    for (uc, v) in pts:
        g = C.project_px(uc, v, near_m=FARMAX)     # project out to 10m
        if g is not None: (L if uc < 320 else R).append(g)
    return (np.array(L) if L else np.empty((0,2))), (np.array(R) if R else np.empty((0,2)))

def yolo_base(model, img):
    r = model.predict(source=img, conf=CONF, half=True, device=0, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0: return []
    xy = r.boxes.xyxy.cpu().numpy(); ar = (xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1])
    return [((x1+x2)/2, y2) for (x1,y1,x2,y2) in xy[ar <= BLOB_FRAC*FRAME_PX*FRAME_PX]]

def draw_new(ax, L, R, fL, fR, off, hdg):
    for P, f, col in ((L, fL, "b"), (R, fR, "r")):
        if len(P) == 0: continue
        if f.get("ok"):
            inl_n, inl_f = f["inl_near"], f["inl_far"]; rej = ~(inl_n | inl_f)
            ax.scatter(-P[inl_n,1], P[inl_n,0], c=col, s=34, marker="o")            # near inlier
            ax.scatter(-P[inl_f,1], P[inl_f,0], c=col, s=48, marker="^")            # far inlier
            ax.scatter(-P[rej,1], P[rej,0], facecolors="none", edgecolors=col, s=30)  # rejected
            ax.plot([-f["y"], -f["y"]], [0, FARMAX], col+"-", lw=2)
            if f.get("adjacent"):
                ay = f["adjacent"]["y_median"]
                ax.axvline(-ay, color=col, ls=":", lw=1, alpha=.6)
                ax.text(-ay, 8.3, f"adj n={f['adjacent']['n']}", color=col, fontsize=7, ha="center")
        else:
            ax.scatter(-P[:,1], P[:,0], facecolors="none", edgecolors=col, s=26)
    if off is not None:
        cn = bin_centre(L[fL["inl"]], R[fR["inl"]], 1, 3); cf = bin_centre(L[fL["inl"]], R[fR["inl"]], 5, 9)
        if cn is not None and cf is not None:
            ax.plot([-cn, -cf], [2, 7], "g-", lw=2.2); ax.scatter([-cn,-cf],[2,7],c="g",s=55,marker="s")
    ax.axvline(0, color="0.6", lw=.5); ax.set_xlim(-4.5, 4.5); ax.set_ylim(0, 8.6)
    ax.set_xlabel("-Y (right +, m)"); ax.set_ylabel("X fwd (m)"); ax.grid(alpha=.3)

def draw_old(ax, L, R, fL, fR, off, hdg):
    for P, f, col in ((L, fL, "b"), (R, fR, "r")):
        Pn = P[P[:,0] < NEAR]
        if len(Pn): ax.scatter(-Pn[:,1], Pn[:,0], c=col, s=28)
        if f.get("ok"): ax.plot([-f["y"], -f["y"]], [0, NEAR], col+"--", lw=1.7)
    ax.axvline(0, color="0.6", lw=.5); ax.set_xlim(-4.5, 4.5); ax.set_ylim(0, 8.6)
    ax.set_xlabel("-Y (right +, m)"); ax.set_ylabel("X fwd (m)"); ax.grid(alpha=.3)

frames = [4107, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 4223]
print(f"{'frame':>6}{'OLD cls':>10}{'OLDoff':>8}{'NEW cls':>10}{'NEWoff':>8}{'NEWhdg':>8}"
      f"{'Lnear':>6}{'Lfar':>6}{'Rnear':>6}{'Rfar':>6}{'adj':>5}")
for fi in frames:
    img = cv2.imread(str(FR / f"{fi:05d}.jpg"))
    L, R = project_all(yolo_base(model, img))
    # OLD near-5m
    oL, oR = fit_side_hybrid(L), fit_side_hybrid(R)
    old_ok = oL["ok"] and oR["ok"]; old_cls = "two_row" if old_ok else ("single_row" if (oL["ok"] or oR["ok"]) else "none")
    old_off = bin_centre(L[L[:,0]<NEAR][oL["inl"]], R[R[:,0]<NEAR][oR["inl"]], 1, 3) if old_ok else None
    # NEW far-extended
    fL, fR = fit_side_far(L), fit_side_far(R)
    new_ok = fL["ok"] and fR["ok"]; new_cls = "two_row" if new_ok else ("single_row" if (fL["ok"] or fR["ok"]) else "none")
    new_off = new_hdg = None
    if new_ok:
        new_off, new_hdg = centre_far(L[fL["inl"]], R[fR["inl"]], bin_centre)
    adj = "".join([s for s,f in (("L",fL),("R",fR)) if f.get("ok") and f.get("adjacent")])
    print(f"{fi:>6}{old_cls:>10}{(f'{old_off:+.2f}' if old_off is not None else ' -- '):>8}"
          f"{new_cls:>10}{(f'{new_off:+.2f}' if new_off is not None else ' -- '):>8}"
          f"{(f'{new_hdg:+.1f}' if new_hdg is not None else ' -- '):>8}"
          f"{fL.get('n_near',0):>6}{fL.get('n_far',0):>6}{fR.get('n_near',0):>6}{fR.get('n_far',0):>6}{adj:>5}")

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].axis("off"); ax[0].set_title(f"frame {fi}")
    draw_old(ax[1], L, R, oL, oR, old_off, None); ax[1].set_title(f"OLD near-5m  {old_cls} off={old_off if old_off is None else round(old_off,2)}", fontsize=9)
    draw_new(ax[2], L, R, fL, fR, new_off, new_hdg)
    ax[2].set_title(f"NEW far-ext (o=near ^=far)  {new_cls} off={new_off if new_off is None else round(new_off,2)} hdg={new_hdg if new_hdg is None else round(new_hdg,1)}", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{SC}/far_ext/rowfar_f{fi}.png", dpi=100); plt.close(fig)
print("saved rowfar_f*.png")
