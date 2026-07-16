"""Validate the hybrid cluster+RANSAC row fit vs the old global-median fit.
For each validation frame (8 CP-3 dry-run frames + CP-2 marked frames 4107/4223/3991),
run Phase C s42 -> base points -> project (near-5m) -> cluster L/R, then compute BOTH the
OLD fit (global-median side_valid) and the NEW hybrid fit, and render a 3-panel:
[ image | OLD bird's-eye | NEW bird's-eye ] so the fitted row can be eyeballed against the
tight vertical cluster (Edosa's black lines)."""
import sys, json
from pathlib import Path
import numpy as np, cv2
PKG = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG/"scripts"/"geometric"))
import torch; torch.multiprocessing.set_sharing_strategy("file_system")
from ultralytics import YOLO
import projection_calibration as C
from single_arm_dryrun import side_valid, bin_centre, NEAR_M, BINS, CONF, BLOB_FRAC, FRAME_PX
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SC = str(PKG / "results/geometric/march/diagnostics/figures/rowfit_validation")
FR = PKG / "results/runs/geom_cp1_frames_640"
EXPECT_WIDTH = 2.45

# ---- NEW hybrid row fit ----------------------------------------------------
WIN_HALF = 0.25      # densest window half-width (0.5m total)
RTOL = 0.25          # RANSAC inlier tolerance (m)
RSPAN, RSTEP = 0.30, 0.05
MIN_INL = 3
MAX_ABS_Y = 3.0
MIN_X_SPAN = 1.0
ADJ_GAP = 0.7        # a secondary cluster must be >0.7m beyond the row band

def fit_side_hybrid(P):
    """P: Nx2 (X,Y) near-field points on one side. Returns dict incl inlier mask + adjacent log."""
    if len(P) < MIN_INL:
        return {"ok": False, "reason": "too_few_points"}
    Y = P[:, 1]
    # Step 1 — densest 0.5m window seed
    counts = np.array([np.sum(np.abs(Y - y) <= WIN_HALF) for y in Y])
    seed = float(np.median(Y[np.abs(Y - Y[counts.argmax()]) <= WIN_HALF]))
    # Step 2 — RANSAC refine: candidate Y in seed +/- 0.3 (0.05 steps), max inliers
    cands = seed + np.arange(-RSPAN, RSPAN + 1e-9, RSTEP)
    best_c, best_inl = None, None
    for c in cands:
        inl = np.abs(Y - c) < RTOL
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl, best_c = inl, c
    if best_inl.sum() < MIN_INL:
        return {"ok": False, "reason": "too_few_inliers"}
    rowY = float(np.median(Y[best_inl]))
    Xin = P[best_inl, 0]
    x_span = float(Xin.max() - Xin.min())
    # Step 3 — sanity
    if abs(rowY) > MAX_ABS_Y:
        return {"ok": False, "reason": "abs_y_too_large", "y": rowY}
    if x_span < MIN_X_SPAN or len(np.unique(np.round(Xin, 1))) < 3:
        return {"ok": False, "reason": "horizontal_cluster", "y": rowY, "x_span": x_span}
    # adjacent-corridor: same-side dots beyond the row band by > ADJ_GAP
    same = P[np.sign(Y) == np.sign(rowY)] if rowY != 0 else P
    outer = same[np.abs(same[:, 1]) > abs(rowY) + RTOL + ADJ_GAP]
    adjacent = {"y_median": float(np.median(outer[:, 1])), "n": int(len(outer))} if len(outer) >= 3 else None
    return {"ok": True, "y": rowY, "inl": best_inl, "n_inl": int(best_inl.sum()),
            "x_span": x_span, "adjacent": adjacent}

def project_side(pts):
    L, R = [], []
    for (uc, v) in pts:
        g = C.project_px(uc, v, near_m=NEAR_M)
        if g is not None:
            (L if uc < 320 else R).append(g)
    return (np.array(L) if L else np.empty((0, 2))), (np.array(R) if R else np.empty((0, 2)))

def yolo_base(model, img):
    r = model.predict(source=img, conf=CONF, half=True, device=0, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0: return []
    xy = r.boxes.xyxy.cpu().numpy(); ar = (xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1])
    return [((x1+x2)/2, y2) for (x1,y1,x2,y2) in xy[ar <= BLOB_FRAC*FRAME_PX*FRAME_PX]]

def centre_from(Li, Ri):
    cen = [bin_centre(Li, Ri, lo, hi) for (lo, hi) in BINS]
    off = cen[0] if cen[0] is not None else None
    hdg = (float(np.degrees(np.arctan2(cen[1]-cen[0], np.mean(BINS[1])-np.mean(BINS[0]))))
           if (cen[0] is not None and cen[1] is not None) else None)
    return cen, off, hdg

def draw(ax, L, R, side_fits, cen, off, hdg, title, new):
    xb = [np.mean(b) for b in BINS]
    for P, fit, col in ((L, side_fits[0], "b"), (R, side_fits[1], "r")):
        if len(P) == 0: continue
        if new and fit.get("ok"):
            inl = fit["inl"]
            ax.scatter(-P[inl,1], P[inl,0], c=col, s=30)                       # inliers filled
            ax.scatter(-P[~inl,1], P[~inl,0], facecolors="none", edgecolors=col, s=30)  # rejected hollow
            ax.plot([-fit["y"], -fit["y"]], [0, 5], col+"-", lw=2)             # NEW row (vertical)
            if fit.get("adjacent"):
                ay = fit["adjacent"]["y_median"]
                ax.axvline(-ay, color=col, ls=":", lw=1, alpha=.6)
                ax.text(-ay, 5.2, f"adj n={fit['adjacent']['n']}", color=col, fontsize=7, ha="center")
        else:
            ax.scatter(-P[:,1], P[:,0], c=col, s=26)
            v = side_valid(P)
            if v: ax.plot([-v[0], -v[0]], [0, 5], col+"--", lw=1.6)            # OLD median row
    if off is not None:
        ax.plot([-cen[0], -cen[1]], xb, "g-", lw=2.2)
        ax.scatter([-cen[0], -cen[1]], xb, c="g", s=60, marker="s")
    ax.axvline(0, color="0.6", lw=.5); ax.set_xlim(-4, 4); ax.set_ylim(0, 6)
    ax.set_xlabel("-Y (right +, m)"); ax.set_ylabel("X fwd (m)"); ax.grid(alpha=.3)
    t = title + (f"  off@2m={off:+.2f} hdg={hdg:+.1f}" if off is not None else "  (no two-row)")
    ax.set_title(t, fontsize=9)

model = YOLO(str(PKG / "results/runs/phase_c_yolo_multiclass/weights/best.pt"))
frames = [3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 4107, 4223]
rows = []
for fi in frames:
    img = cv2.imread(str(FR / f"{fi:05d}.jpg"))
    L, R = project_side(yolo_base(model, img))
    # OLD
    ovL, ovR = side_valid(L), side_valid(R)
    old_cls = "two_row" if (ovL and ovR) else ("single_row" if (ovL or ovR) else "none")
    old_cen = old_off = old_hdg = None
    if ovL and ovR: old_cen, old_off, old_hdg = centre_from(L, R)
    # NEW
    fL, fR = fit_side_hybrid(L), fit_side_hybrid(R)
    new_cls = "two_row" if (fL["ok"] and fR["ok"]) else ("single_row" if (fL["ok"] or fR["ok"]) else "none")
    new_cen = new_off = new_hdg = None; width = None; sym_flag = False
    if fL["ok"] and fR["ok"]:
        Li, Ri = L[fL["inl"]], R[fR["inl"]]
        new_cen, new_off, new_hdg = centre_from(Li, Ri)
        width = fL["y"] - fR["y"]; sym_flag = abs(width - EXPECT_WIDTH) > 1.0
    adj = [s for s, f in (("L", fL), ("R", fR)) if f.get("ok") and f.get("adjacent")]
    rows.append((fi, old_cls, old_off, old_hdg, new_cls, new_off, new_hdg, width, sym_flag, adj))

    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].axis("off"); ax[0].set_title(f"frame {fi}")
    draw(ax[1], L, R, (ovL if isinstance(ovL, dict) else {}, {}), old_cen, old_off, old_hdg, "OLD global-median", new=False)
    draw(ax[2], L, R, (fL, fR), new_cen, new_off, new_hdg,
         "NEW hybrid" + (f"  W={width:.2f}{' FLAG' if sym_flag else ''}" if width else ""), new=True)
    fig.tight_layout(); fig.savefig(f"{SC}/rowfit_f{fi}.png", dpi=100); plt.close(fig)

print(f"{'frame':>6}{'OLD cls':>10}{'OLDoff':>8}{'NEW cls':>10}{'NEWoff':>8}{'d_off':>8}{'width':>7}{'sym':>5}{'adj':>6}")
for (fi, oc, oo, oh, nc, no, nh, w, sf, adj) in rows:
    doff = (f"{no-oo:+.2f}" if (oo is not None and no is not None) else "  -- ")
    print(f"{fi:>6}{oc:>10}{(f'{oo:+.2f}' if oo is not None else ' -- '):>8}{nc:>10}"
          f"{(f'{no:+.2f}' if no is not None else ' -- '):>8}{doff:>8}"
          f"{(f'{w:.2f}' if w else ' -- '):>7}{('Y' if sf else ''):>5}{(''.join(adj) if adj else ''):>6}")
print("saved rowfit_f*.png")
