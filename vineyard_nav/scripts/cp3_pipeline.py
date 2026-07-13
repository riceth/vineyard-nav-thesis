"""CP-3 locked geometry pipeline + single-arm dry-run reproducer.

Full image->centreline pipeline with the CP-3-locked parameters (see
GEOMETRY_PIPELINE_SPEC.md sections 4-6 and DECISIONS.md D-F/D034/D035):

  1. Detections at conf 0.25 (D030), any of the 9 trained checkpoints.
  2. Mask-area guard: drop a detection whose bbox exceeds 15% of the frame.
     On the bare-vine March bag the largest legitimate detection (a close
     trellis pole) is ~10.5% of the frame, so a 15% cap rejects only a gross
     whole-frame blob (the F007 canopy pathology) without dropping real poles.
     The per-frame outlier defence is the Y-constant row fit's median +/- tol
     inlier test, which rejects any single off-row spurious base point.
  3. Project bbox-bottom-centre base points to the ground plane
     (cp2_projection, CP-2), restricted to the near field X < 5 m where the
     projection fan is smallest.
  4. Cluster left/right by image column (u < 320 / u >= 320).
  5. Y-constant row model per side: a side is valid if >= 3 projected points
     lie within 0.5 m of that side's median Y. Real vineyard rows sit at
     constant Y in the robot frame, so a constant (not sloped) model estimates
     the correct quantity and is robust to the projection fan.
  6. two-row centreline = midpoint of the left/right medians, binned by range:
       GT-1 lateral offset  = centreline lateral position at the 1-3 m bin
                              (look-ahead ~2 m), signed (+Y = left).
       GT-2 heading         = angle of the centreline direction between the
                              1-3 m and 3-5 m bins. The projection fan is
                              symmetric (left row fans +Y, right row fans -Y
                              with range), so it CANCELS in the centreline
                              midpoint; the residual centreline slope recovers
                              the true robot-to-row angle fan-free. Per-row
                              slopes are fan-corrupted and are NOT used.
  7. single-row fallback: centre = row_median +/- half-spacing, reported at
     both 1.2 m (trajectory-anchored, primary) and 0.96 m (projection-
     consistent, sensitivity) per D-G.

Running this file as a script reproduces the CP-3 single-arm dry run:
Phase C seed 42 over the val frames, printing coverage / GT-1 / GT-2 and
saving a few bird's-eye visualisations.
"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG / "scripts"))
import cp2_projection as C  # noqa: E402  (CP-2 image->ground projection)

# ---- CP-3 locked parameters -------------------------------------------------
CONF = 0.25              # D030 detection confidence
NEAR_M = 5.0             # near-field cutoff (fan smallest here)
INL = 3                  # min inlier points for a valid row side
TOL = 0.5                # Y-constant inlier band (m)
BLOB_FRAC = 0.15         # drop bbox > 15% of frame (gross blob guard; keeps real poles)
BINS = [(1.0, 3.0), (3.0, 5.0)]   # centreline range bins -> look-ahead ~2 m and ~4 m
LOOKAHEAD_BIN = 0        # BINS index used for the GT-1 offset
HALF = {"traj": 1.2, "proj": 0.96}   # D-G single-row half-spacing (primary / sensitivity)
FRAME_PX = 640


def side_valid(P):
    """Y-constant validity: return (median Y of inliers, n_inliers) or None."""
    if len(P) < INL:
        return None
    Y = P[:, 1]
    med = float(np.median(Y))
    inl = np.abs(Y - med) < TOL
    if inl.sum() < INL:
        return None
    return float(np.median(Y[inl])), int(inl.sum())


def bin_centre(L, R, lo, hi):
    """Centreline lateral position (midpoint of side medians) within [lo, hi) m."""
    Lb = L[(L[:, 0] >= lo) & (L[:, 0] < hi)]
    Rb = R[(R[:, 0] >= lo) & (R[:, 0] < hi)]
    if len(Lb) and len(Rb):
        return float((np.median(Lb[:, 1]) + np.median(Rb[:, 1])) / 2)
    return None


def process_frame(img, model):
    """Run one frame through the locked pipeline. Returns a result dict with
    'cls' in {two_row, single_row, none} plus offset/heading when available."""
    r = model.predict(source=img, conf=CONF, half=True, device=0, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return {"cls": "none", "blob_dropped": 0, "nL": 0, "nR": 0}
    xyxy = r.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    keep = areas <= BLOB_FRAC * FRAME_PX * FRAME_PX
    blob_dropped = int((~keep).sum())
    L, R = [], []
    for (x1, y1, x2, y2) in xyxy[keep]:
        uc = (x1 + x2) / 2
        g = C.project_px(uc, y2, near_m=NEAR_M)
        if g is not None:
            (L if uc < 320 else R).append(g)
    L = np.array(L) if L else np.empty((0, 2))
    R = np.array(R) if R else np.empty((0, 2))
    vL, vR = side_valid(L), side_valid(R)
    out = {"blob_dropped": blob_dropped, "nL": int(len(L)), "nR": int(len(R))}
    if vL and vR:
        out["cls"] = "two_row"
        centres = [bin_centre(L, R, lo, hi) for (lo, hi) in BINS]
        c_look = centres[LOOKAHEAD_BIN]
        out["offset_m"] = c_look if c_look is not None else (vL[0] + vR[0]) / 2
        if centres[0] is not None and centres[1] is not None:
            dx = np.mean(BINS[1]) - np.mean(BINS[0])
            out["heading_deg"] = float(np.degrees(np.arctan2(centres[1] - centres[0], dx)))
        else:
            out["heading_deg"] = None
        out["_L"], out["_R"], out["_centres"] = L, R, centres
    elif vL or vR:
        out["cls"] = "single_row"
        row, sign = (vL[0], -1.0) if vL else (vR[0], +1.0)   # centre = row + sign*half
        out["offset_m_traj"] = row + sign * HALF["traj"]
        out["offset_m_proj"] = row + sign * HALF["proj"]
    else:
        out["cls"] = "none"
    return out


def run_arm(model, frame_indices, frames_dir):
    """Run the pipeline over frame_indices; return list of per-frame results."""
    results = []
    for fi in frame_indices:
        img = cv2.imread(str(Path(frames_dir) / f"{fi:05d}.jpg"))
        if img is None:
            continue
        res = process_frame(img, model)
        res["i"] = int(fi)
        results.append(res)
    return results


def _summarise(results):
    from collections import Counter
    cov = Counter(r["cls"] for r in results)
    tot = len(results)
    offs = np.array([r["offset_m"] for r in results if r["cls"] == "two_row"])
    heads = np.array([r["heading_deg"] for r in results
                      if r["cls"] == "two_row" and r.get("heading_deg") is not None])
    blob_frames = sum(1 for r in results if r.get("blob_dropped"))
    return {
        "frames": tot,
        "two_row": cov["two_row"], "two_row_pct": round(100 * cov["two_row"] / tot, 1),
        "single_row": cov["single_row"], "single_row_pct": round(100 * cov["single_row"] / tot, 1),
        "none": cov["none"], "none_pct": round(100 * cov["none"] / tot, 1),
        "offset_mean": round(float(offs.mean()), 3), "offset_sd": round(float(offs.std()), 3),
        "offset_abs_median": round(float(np.median(np.abs(offs))), 3),
        "heading_mean": round(float(heads.mean()), 2), "heading_sd": round(float(heads.std()), 2),
        "heading_abs_median": round(float(np.median(np.abs(heads))), 2),
        "blob_frames": blob_frames,
    }


if __name__ == "__main__":
    import torch
    torch.multiprocessing.set_sharing_strategy("file_system")
    from ultralytics import YOLO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames_dir = PKG / "results/runs/geom_cp1_frames_640"
    manifest = json.load(open(PKG / "results/geometric/march/cp1_manifest.json"))
    val = [f["i"] for f in manifest["frames"] if f["split"] == "val"]
    outdir = PKG / "results/geometric/march/cp3_samples"
    outdir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(PKG / "results/runs/phase_c_yolo_multiclass/weights/best.pt"))
    results = run_arm(model, val, frames_dir)
    s = _summarise(results)
    print("==================== CP-3 dry run (Phase C seed 42, val) ====================")
    print(f"frames {s['frames']} | two-row {s['two_row']} ({s['two_row_pct']}%) | "
          f"single {s['single_row']} ({s['single_row_pct']}%) | none {s['none']} ({s['none_pct']}%)")
    print(f"GT-1 offset@2m : mean {s['offset_mean']:+.3f}  SD {s['offset_sd']:.3f}  "
          f"|median| {s['offset_abs_median']:.3f} m")
    print(f"GT-2 heading   : mean {s['heading_mean']:+.2f}  SD {s['heading_sd']:.2f}  "
          f"|median| {s['heading_abs_median']:.2f} deg (fan-free centreline)")
    print(f"blob guard (>15% frame): {s['blob_frames']} frames affected")

    # sample visualisations
    viz = 0
    for r in results:
        if r["cls"] != "two_row" or r["nL"] < 5 or r["nR"] < 5 or r["_centres"][0] is None:
            continue
        if viz >= 4:
            break
        fi = r["i"]
        img = cv2.imread(str(frames_dir / f"{fi:05d}.jpg"))
        L, R, cen = r["_L"], r["_R"], r["_centres"]
        xb = [np.mean(b) for b in BINS]
        fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].axis("off"); ax[0].set_title(f"frame {fi}")
        ax[1].scatter(-L[:, 1], L[:, 0], c="b", s=18, label="left row")
        ax[1].scatter(-R[:, 1], R[:, 0], c="r", s=18, label="right row")
        ax[1].plot([-cen[0], -cen[1]], xb, "g-", lw=2,
                   label=f"centreline (offset@2m={r['offset_m']:+.2f} m, hdg={r['heading_deg']:+.1f} deg)")
        ax[1].scatter([-cen[0], -cen[1]], xb, c="g", s=70, marker="s")
        ax[1].axvline(0, color="0.6", lw=.5)
        ax[1].set_xlim(-3, 3); ax[1].set_ylim(0, 5)
        ax[1].set_xlabel("-Y  (right +, m)"); ax[1].set_ylabel("X forward (m)")
        ax[1].legend(fontsize=8, loc="lower left"); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(outdir / f"cp3_dryrun_f{fi}.png", dpi=110); plt.close(fig)
        viz += 1

    for r in results:
        r.pop("_L", None); r.pop("_R", None); r.pop("_centres", None)
    json.dump({"params": {"conf": CONF, "near_m": NEAR_M, "inl": INL, "tol": TOL,
                          "blob_frac": BLOB_FRAC, "bins": BINS, "half": HALF},
               "summary": s},
              open(PKG / "results/geometric/march/cp3_dryrun_report.json", "w"), indent=2)
    print(f"saved {viz} visualisations -> {outdir}")
