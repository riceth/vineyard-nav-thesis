import numpy as np
NEAR=5.0; FARMAX=10.0; WIN_HALF=0.25; RTOL=0.25; RSPAN=0.30; RSTEP=0.05
FAR_TOL=0.5; MIN_TOTAL=3; MAX_ABS_Y=3.0; MIN_X_SPAN=1.0; ADJ_GAP=0.7

# ---- v1: near-field-only hybrid (previous fix), kept for OLD-vs-NEW comparison ----
def fit_side_hybrid(P):
    P = P[P[:,0] < NEAR]
    if len(P) < MIN_TOTAL: return {"ok": False, "reason": "too_few_points"}
    Y = P[:,1]
    counts = np.array([np.sum(np.abs(Y - y) <= WIN_HALF) for y in Y])
    seed = float(np.median(Y[np.abs(Y - Y[counts.argmax()]) <= WIN_HALF]))
    best_inl = None
    for c in seed + np.arange(-RSPAN, RSPAN + 1e-9, RSTEP):
        inl = np.abs(Y - c) < RTOL
        if best_inl is None or inl.sum() > best_inl.sum(): best_inl = inl
    if best_inl.sum() < MIN_TOTAL: return {"ok": False, "reason": "too_few_inliers"}
    rowY = float(np.median(Y[best_inl])); Xin = P[best_inl,0]; x_span = float(Xin.max()-Xin.min())
    if abs(rowY) > MAX_ABS_Y: return {"ok": False, "reason": "abs_y_too_large", "y": rowY}
    if x_span < MIN_X_SPAN or len(np.unique(np.round(Xin,1))) < 3:
        return {"ok": False, "reason": "horizontal_cluster", "y": rowY}
    return {"ok": True, "y": rowY, "inl": best_inl, "n_inl": int(best_inl.sum())}

# ---- v2: near-seed + cluster-consistent far-field extension (current refinement) ----
def fit_side_far(P):
    """P: Nx2 (X,Y) projected points out to FARMAX. Seed on X<5m densest cluster, RANSAC
    refine, then extend to 5-10m dots within FAR_TOL of the row Y; refit median of all inliers.
    Returns boolean masks (on P) for near/far inliers + adjacent-corridor log."""
    X, Y = P[:,0], P[:,1]
    near_mask = X < NEAR; far_mask = (X >= NEAR) & (X <= FARMAX)
    if near_mask.sum() < 2: return {"ok": False, "reason": "too_few_near_seed", "n_near_raw": int(near_mask.sum())}
    Yn = Y[near_mask]
    counts = np.array([np.sum(np.abs(Yn - y) <= WIN_HALF) for y in Yn])
    seed = float(np.median(Yn[np.abs(Yn - Yn[counts.argmax()]) <= WIN_HALF]))
    best_c, best_n = seed, -1
    for c in seed + np.arange(-RSPAN, RSPAN + 1e-9, RSTEP):
        n = int(np.sum(np.abs(Yn - c) < RTOL))
        if n > best_n: best_n, best_c = n, c
    nearY = float(np.median(Yn[np.abs(Yn - best_c) < RTOL]))
    inl_near = near_mask & (np.abs(Y - best_c) < RTOL)          # near-field RANSAC inliers
    inl_far = far_mask & (np.abs(Y - nearY) <= FAR_TOL)         # far-field extension (cluster-consistent)
    inl = inl_near | inl_far
    if inl.sum() < MIN_TOTAL: return {"ok": False, "reason": "too_few_total", "y": nearY, "n_inl": int(inl.sum())}
    rowY = float(np.median(Y[inl])); xs = X[inl]; x_span = float(xs.max() - xs.min())
    if abs(rowY) > MAX_ABS_Y: return {"ok": False, "reason": "abs_y_too_large", "y": rowY}
    if x_span < MIN_X_SPAN or len(np.unique(np.round(xs,1))) < 3:
        return {"ok": False, "reason": "horizontal_cluster", "y": rowY, "x_span": x_span}
    same_out = (np.sign(Y) == np.sign(rowY)) & (np.abs(Y) > abs(rowY) + RTOL + ADJ_GAP) & (~inl)
    adjacent = {"y_median": float(np.median(Y[same_out])), "n": int(same_out.sum())} if same_out.sum() >= 3 else None
    return {"ok": True, "y": rowY, "inl": inl, "inl_near": inl_near, "inl_far": inl_far,
            "n_near": int(inl_near.sum()), "n_far": int(inl_far.sum()), "x_span": x_span, "adjacent": adjacent}

def centre_linefit(Li, Ri, look=2.0):
    """Line-fit centreline (approved row model). Fit Y=mX+c per side (least squares) to the
    far-extension inliers; centreline = midline. GT-1 = centreline Y at X=look (2m); GT-2 =
    centreline slope (deg, +Y=left); width = mean Y_L - mean Y_R (rows parallel). Flags
    steep slopes (|m|>0.3, possible residual contamination) and L/R slope mismatch (|dm|>0.2)."""
    if len(Li) < 2 or len(Ri) < 2: return None
    mL, cL = np.polyfit(Li[:,0], Li[:,1], 1)
    mR, cR = np.polyfit(Ri[:,0], Ri[:,1], 1)
    yL, yR = mL*look + cL, mR*look + cR
    m_c = (mL + mR) / 2.0
    flags = []
    if abs(mL) > 0.3 or abs(mR) > 0.3: flags.append("steep_slope")
    if abs(mL - mR) > 0.2: flags.append("slope_mismatch")
    return {"offset": float((yL + yR)/2), "heading": float(np.degrees(np.arctan2(m_c, 1.0))),
            "m_L": float(mL), "m_R": float(mR), "m_c": float(m_c),
            "width": float(np.mean(Li[:,1]) - np.mean(Ri[:,1])), "flags": flags,
            "lines": (float(mL), float(cL), float(mR), float(cR))}

def centre_far(Li, Ri, bin_centre, BINS_HEAD=((1.0,3.0),(5.0,9.0)), BINS_FALL=((1.0,3.0),(3.0,5.0))):
    """Offset at look-ahead 2m (1-3m bin) + heading over a long baseline (2m->7m via far ext)."""
    def mid(lo,hi):
        return bin_centre(Li, Ri, lo, hi)
    off = mid(1.0,3.0)
    if off is None and len(Li) and len(Ri): off = float((np.median(Li[:,1]) + np.median(Ri[:,1])) / 2)
    cn, cf, xn, xf = mid(*BINS_HEAD[0]), mid(*BINS_HEAD[1]), np.mean(BINS_HEAD[0]), np.mean(BINS_HEAD[1])
    if cn is None or cf is None:
        cn, cf, xn, xf = mid(*BINS_FALL[0]), mid(*BINS_FALL[1]), np.mean(BINS_FALL[0]), np.mean(BINS_FALL[1])
    hdg = float(np.degrees(np.arctan2(cf - cn, xf - xn))) if (cn is not None and cf is not None) else None
    return off, hdg
