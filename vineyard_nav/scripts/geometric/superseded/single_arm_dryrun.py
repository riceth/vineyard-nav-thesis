"""Retired CP-3 single-arm dry-run reproducer.

Kept for provenance of the committed artifact only (results/geometric/march/single_arm_dryrun_report.json
+ single_arm_dryrun_samples/), superseded by the real March/April whole-bag runs. The CP-3 locked
geometry constants and functions this used to define now live in the shared library
scripts/geometric/cp3_geometry.py; this script only imports them and reproduces the original
Phase C seed-42 val dry run.

Run:  python3 scripts/geometric/superseded/single_arm_dryrun.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import cv2

PKG = Path(__file__).resolve().parents[3]                       # vineyard_nav (one level deeper: superseded/)
sys.path.insert(0, str(PKG / "scripts" / "geometric"))         # so cp3_geometry resolves
from cp3_geometry import CONF, NEAR_M, INL, TOL, BLOB_FRAC, BINS, LOOKAHEAD_BIN, HALF, FRAME_PX, side_valid, bin_centre, process_frame, run_arm, _summarise  # noqa: E402,F401


if __name__ == "__main__":
    import torch
    torch.multiprocessing.set_sharing_strategy("file_system")
    from ultralytics import YOLO
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames_dir = PKG / "results/runs/geom_cp1_frames_640"
    manifest = json.load(open(PKG / "results/geometric/march/dataset_manifest.json"))
    val = [f["i"] for f in manifest["frames"] if f["split"] == "val"]
    outdir = PKG / "results/geometric/march/single_arm_dryrun_samples"
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
        fig.tight_layout(); fig.savefig(outdir / f"single_arm_dryrun_f{fi}.png", dpi=110); plt.close(fig)
        viz += 1

    for r in results:
        r.pop("_L", None); r.pop("_R", None); r.pop("_centres", None)
    json.dump({"params": {"conf": CONF, "near_m": NEAR_M, "inl": INL, "tol": TOL,
                          "blob_frac": BLOB_FRAC, "bins": BINS, "half": HALF},
               "summary": s},
              open(PKG / "results/geometric/march/single_arm_dryrun_report.json", "w"), indent=2)
    print(f"saved {viz} visualisations -> {outdir}")
