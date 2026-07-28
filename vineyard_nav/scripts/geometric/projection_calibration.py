#!/usr/bin/env python3
"""CP-2 image->world projection module (GEOMETRY_PIPELINE_SPEC.md §6, §9 CP-2).

Builds the image->ground inverse-perspective mapping (IPM) from the bag camera intrinsics
and the Polvara et al. 2024 Table 3 extrinsics (base_link -> Zed2 Front), assuming a flat
Z=0 ground plane in base_link. Detections run on the 640x640 (stretch-preprocessed) frame;
their pixel coords are back-mapped to the native 1920x1080 before projection.

`project_px(u640, v640) -> (X_forward, Y_left) metres in base_link, or None` is the reusable
entry point (imported by CP-3+). Run as a script, `main()` regenerates the calibration report
+ a few bird's-eye overlays.

Known limitation (see docstring of validate() and spec §6): projection-measured corridor width
is ~22% narrower than the trajectory-derived 2.45 m spacing — symmetric, so it does NOT bias the
two-row centreline metric; it only shifts the D-G single-row fallback prior.

Run:  python3 vineyard_nav/scripts/geometric/projection_calibration.py
"""
from __future__ import annotations
import json, datetime
from pathlib import Path
import numpy as np

# ---- intrinsics (bag /front/zed_node/rgb/camera_info, 1920x1080) ----
W0, H0 = 1920, 1080
K = np.array([[1057.0, 0.0, 952.2], [0.0, 1057.0, 553.6], [0.0, 0.0, 1.0]])
KINV = np.linalg.inv(K)
# ---- extrinsics base_link <- Zed2 Front (Polvara et al. 2024, Table 3) ----
T_BASE_CAM = np.array([0.345, 0.060, 0.763])                       # translation (m)
_Q = np.array([0.0, 0.017, 0.0, 1.0]); _Q /= np.linalg.norm(_Q); _qx, _qy, _qz, _qw = _Q
_R_BB = np.array([                                                 # base <- ZedFront(body) from quat
    [1-2*(_qy*_qy+_qz*_qz), 2*(_qx*_qy-_qz*_qw), 2*(_qx*_qz+_qy*_qw)],
    [2*(_qx*_qy+_qz*_qw), 1-2*(_qx*_qx+_qz*_qz), 2*(_qy*_qz-_qx*_qw)],
    [2*(_qx*_qz-_qy*_qw), 2*(_qy*_qz+_qx*_qw), 1-2*(_qx*_qx+_qy*_qy)]])
_R_BODY_OPT = np.array([[0, 0, 1.0], [-1, 0, 0], [0, -1, 0]])       # body <- optical (z-fwd,x-right,y-down)
R_BASE_OPT = _R_BB @ _R_BODY_OPT
CAM_POS = T_BASE_CAM                                                # camera origin in base_link
PITCH_DEG = float(np.degrees(2 * np.arcsin(_qy)))                   # ~1.95 deg down
NEAR_M = 8.0                                                        # near-field cutoff (IPM accurate)


def project_px(u640: float, v640: float, near_m: float = NEAR_M):
    """640x640 (stretched) pixel -> ground (X forward, Y left) metres in base_link, or None."""
    d = R_BASE_OPT @ (KINV @ np.array([u640 * W0 / 640.0, v640 * H0 / 640.0, 1.0]))
    if d[2] >= -1e-6:
        return None
    g = CAM_POS + (-CAM_POS[2] / d[2]) * d
    return np.array([g[0], g[1]]) if (0.0 < g[0] < near_m) else None


def project_ground(X: float, Y: float, Z: float = 0.0):
    """Inverse of `project_px`: base_link point (X fwd, Y left, Z up) metres -> 640x640 (stretched)
    pixel (u, v), or None if behind the image plane. Analytic inverse of the SAME K / R_BASE_OPT /
    CAM_POS forward model — `project_px` is untouched. Used ONLY to draw fitted rows / centreline back
    onto the raw image in the report figures (figures.py); not part of the metric pipeline."""
    ray_opt = R_BASE_OPT.T @ (np.array([X, Y, Z]) - CAM_POS)    # base_link vector -> optical frame
    if ray_opt[2] <= 1e-6:
        return None                                             # behind the image plane
    pix = K @ (ray_opt / ray_opt[2])                            # native 1920x1080 homogeneous pixel
    return (float(pix[0] * 640.0 / W0), float(pix[1] * 640.0 / H0))


# ---------- calibration validation (script entry) ----------
_LIMITATION = (
    "Projection-measured corridor width (median 1.91 m, IQR [1.59, 2.45]) is ~22% narrower than "
    "the trajectory-derived 2.45 m spacing. This symmetric narrowing does NOT bias the primary "
    "two-row centreline metric (midpoint is preserved); it affects only width-dependent measures "
    "(D-G single-row fallback prior). Likely cause: bbox-bottom projects to the visible inner edge "
    "of trunk/pole rather than true ground contact, plus possible sub-cm pitch/height offset from "
    "Table 3 nominal. Refinement (true-ground-contact detection) is future work."
)


def _validate():
    import cv2
    import cuda_preload  # noqa: F401 — cuDNN cold-init guard; MUST precede torch (D049)
    from ultralytics import YOLO
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    PKG = Path(__file__).resolve().parents[2]; GIT = Path(__file__).resolve().parents[3]
    FR = PKG / "results/runs/geom_cp1_frames_640"
    man = json.load(open(PKG / "results/geometric/march/dataset_manifest.json"))
    out_json = PKG / "results/geometric/march/projection_calibration_report.json"
    samples = PKG / "results/geometric/march/projection_calibration_samples"; samples.mkdir(parents=True, exist_ok=True)

    sanity = {name: (lambda g: None if g is None else [round(float(g[0]), 2), round(float(g[1]), 2)])(project_px(u, v))
              for name, (u, v) in {"image_centre": (320, 320), "bottom_centre": (320, 600),
                                   "bottom_left": (120, 620), "bottom_right": (520, 620)}.items()}

    model = YOLO(str(PKG / "results/runs/phase_c_yolo_multiclass/weights/best.pt"))
    val = [f["i"] for f in man["frames"] if f["split"] == "val"]
    samp = [val[i] for i in range(0, len(val), max(1, len(val) // 80))]
    widths, plotted = [], 0
    for fi in samp:
        img = cv2.imread(str(FR / f"{fi:05d}.jpg"))
        r = model.predict(source=img, conf=0.25, quantize=16, device=0, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        L, R = [], []
        for (x1, y1, x2, y2) in r.boxes.xyxy.cpu().numpy():
            g = project_px((x1 + x2) / 2, y2)
            if g is not None:
                (L if (x1 + x2) / 2 < 320 else R).append(g)
        L, R = np.array(L), np.array(R)
        if len(L) < 6 or len(R) < 6:
            continue
        mL, cL = np.polyfit(L[:, 0], L[:, 1], 1); mR, cR = np.polyfit(R[:, 0], R[:, 1], 1)
        if abs((cL + cR) / 2) < 0.4 and abs(mL) < 0.4 and abs(mR) < 0.4 and (cL - cR) > 0:
            widths.append(float(cL - cR))
            if plotted < 3:
                fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
                ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].set_title(f"frame {fi}"); ax[0].axis("off")
                ax[1].scatter(-L[:, 1], L[:, 0], c="b", label=f"left Y={cL:+.2f}")
                ax[1].scatter(-R[:, 1], R[:, 0], c="r", label=f"right Y={cR:+.2f}")
                xs = np.linspace(0, NEAR_M, 10); ax[1].plot(-(mL*xs+cL), xs, "b--"); ax[1].plot(-(mR*xs+cR), xs, "r--")
                ax[1].axvline(0, color="k", lw=.5); ax[1].set_xlabel("-Y (right +, m)"); ax[1].set_ylabel("X fwd (m)")
                ax[1].set_title(f"width {cL-cR:.2f} m (expect ~2.45)"); ax[1].legend(); ax[1].grid(alpha=.3); ax[1].axis("equal")
                fig.tight_layout(); fig.savefig(str(samples / f"projection_birdseye_f{fi}.png"), dpi=110); plt.close(fig); plotted += 1
    w = np.array(widths)
    report = {
        "meta": {"checkpoint": "CP-2", "generated": datetime.datetime.now().isoformat(timespec="seconds"),
                 "intrinsics_1920x1080": {"fx": 1057.0, "fy": 1057.0, "cx": 952.2, "cy": 553.6},
                 "extrinsics_base_to_cam": {"xyz_m": T_BASE_CAM.tolist(), "quat_xyzw": [0.0, 0.017, 0.0, 1.0],
                                            "pitch_deg_down": round(PITCH_DEG, 2), "source": "Polvara et al. 2024 Table 3"},
                 "ground_plane": "Z=0 in base_link (base_link at ground level)"},
        "sanity_projections_XY_m": sanity,
        "row_width_m": {"median": round(float(np.median(w)), 2), "mean": round(float(w.mean()), 2),
                        "iqr": [round(float(np.percentile(w, 25)), 2), round(float(np.percentile(w, 75)), 2)],
                        "n_frames": len(w), "trajectory_spacing_m": 2.45},
        "d_g_half_spacing_prior_m": {"primary_trajectory_anchored": 1.2,
                                     "sensitivity_projection_consistent": round(float(np.median(w)) / 2, 2)},
        "known_limitation": _LIMITATION,
    }
    out_json.write_text(json.dumps(report, indent=2))
    print(f"pitch {PITCH_DEG:.2f} deg down | sanity {sanity}")
    print(f"row width: median {np.median(w):.2f} m, IQR [{np.percentile(w,25):.2f},{np.percentile(w,75):.2f}], n={len(w)}")
    print(f"D-G half-spacing prior: 1.2 m (trajectory) / {np.median(w)/2:.2f} m (projection)")
    print(f"saved {out_json.relative_to(GIT)} + {plotted} overlays -> {samples.relative_to(GIT)}")


if __name__ == "__main__":
    _validate()
