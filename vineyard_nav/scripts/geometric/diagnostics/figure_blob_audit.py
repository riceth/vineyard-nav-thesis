"""Render the F007 blob-scale detections a bag's geometric stream produced (D035 guard diagnostics).

`extract_detections.py` writes `cache/blob_audit.json` for every bag (the F007 audit sidecar): each
detection whose bbox exceeds the D035 15%-of-frame guard is logged with its seed, frame, area and box.
This script re-renders those frames so the audit's counts can be *seen* — a blob-scale detection is
either the F007 whole-canopy false positive (the pathology the guard exists for) or a genuine close-up
structure (which the guard would be wrongly dropping), and only the image distinguishes them.

  python3 scripts/geometric/diagnostics/figure_blob_audit.py --bag june
      -> results/geometric/june/diagnostics/blob_audit/blob_s42_f07460.png

Bag-agnostic; a no-op with a printed summary when the bag logged zero blobs (march/april/may). The
guard-dropped detection is drawn in red with its mask overlaid; every kept detection is drawn thin
(trunk orange / pole yellow) so it is visible whether the blob REPLACED the true detections or was
purely additive noise on top of them.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path

import numpy as np
import cv2

PKG = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
import cuda_preload  # noqa: E402,F401 — cuDNN cold-init guard; MUST precede torch (D049)
from ultralytics import YOLO  # noqa: E402
from cp3_geometry import CONF, BLOB_FRAC, FRAME_PX  # noqa: E402
from bag_config import resolve  # noqa: E402

SEED_RUN = {42: "phase_c_yolo_multiclass", 43: "phase_c_yolo_multiclass_seed43",
            44: "phase_c_yolo_multiclass_seed44"}
COL_BLOB, COL_TRUNK, COL_POLE = (0, 0, 255), (255, 160, 0), (0, 210, 255)


def render(bag: str) -> int:
    B = resolve(bag, "eligible")
    audit_path = B["cache_dir"] / "blob_audit.json"
    if not audit_path.exists():
        print(f"[{bag}] no blob_audit.json — run extract_detections.py --bag {bag} first")
        return 1
    audit = json.load(open(audit_path))
    blobs = audit.get("blobs", [])
    s = audit.get("summary", {})
    print(f"[{bag}] blob audit: {s.get('total_blob_dropped', 0)} blob-scale detection(s), "
          f"max {s.get('max_area_frac_overall', 0):.1%} of frame")
    if not blobs:
        print(f"[{bag}] nothing to render (no detection exceeded the {BLOB_FRAC:.0%} guard)")
        return 0

    out_dir = B["out_dir"].parent.parent / "diagnostics" / "blob_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    guard_px = BLOB_FRAC * FRAME_PX * FRAME_PX
    models: dict[int, YOLO] = {}

    for b in blobs:
        seed, fi = int(b["seed"]), int(b["frame"])
        img = cv2.imread(str(B["frames_dir"] / f"{fi:05d}.jpg"))
        if img is None:
            print(f"  seed {seed} frame {fi}: frame image missing — skipped")
            continue
        if seed not in models:
            models[seed] = YOLO(str(PKG / "results/runs" / SEED_RUN[seed] / "weights/best.pt"))
        r = models[seed].predict(source=img, conf=CONF, quantize=16, device=0, verbose=False)[0]

        vis, overlay = img.copy(), img.copy()
        xy = r.boxes.xyxy.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int)
        cf = r.boxes.conf.cpu().numpy()
        ar = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
        masks = None if r.masks is None else r.masks.data.cpu().numpy()
        n_kept = int((ar <= guard_px).sum())

        for i, (box, c, a, cn) in enumerate(zip(xy, cl, ar, cf)):
            x1, y1, x2, y2 = [int(v) for v in box]
            if a > guard_px:
                if masks is not None and i < len(masks):
                    overlay[cv2.resize(masks[i], (FRAME_PX, FRAME_PX)) > 0.5] = COL_BLOB
                cv2.rectangle(vis, (x1, y1), (x2, y2), COL_BLOB, 3)
                cv2.putText(vis, f"{r.names[c]} {a / FRAME_PX ** 2:.1%} conf {cn:.2f} - DROPPED by 15% guard",
                            (x1 + 3, max(15, y1 + 17)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_BLOB, 2)
                print(f"  seed {seed} frame {fi}: {r.names[c]} {a / FRAME_PX ** 2:.2%} conf {cn:.2f} "
                      f"bbox=[{x1},{y1},{x2},{y2}] | {n_kept} kept detections remain")
            else:
                cv2.rectangle(vis, (x1, y1), (x2, y2), COL_TRUNK if c == 0 else COL_POLE, 1)

        vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
        cv2.putText(vis, f"{bag} frame {fi} - seed {seed} (Phase C multiclass)", (6, FRAME_PX - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        out = out_dir / f"blob_s{seed}_f{fi:05d}.png"
        cv2.imwrite(str(out), vis)
        print(f"    wrote {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="june", help="bag name (default: june)")
    sys.exit(render(ap.parse_args().bag))
