"""Extract Phase C per-detection (class, base-point) over ALL eligible frames of a bag for the
config analysis (D026, D040 whole-bag). Locked upstream: conf 0.25, 15% blob guard (D035). Caches
seed,frame,class,uc,v to CSV for offline reclustering. Bag-agnostic multi-bag template.

  python3 extract_detections.py --bag march   -> results/geometric/march/cache/detections.csv

3 seeds x all eligible frames (no val/test split). Single output detections.csv read by
config_analysis.py. This is the per-bag extraction stage of the seasonal (multi-bag) pipeline.
"""
import sys, json
from pathlib import Path
import numpy as np, cv2
PKG = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(PKG / "scripts" / "geometric"))
import torch; torch.multiprocessing.set_sharing_strategy("file_system")
from ultralytics import YOLO
from cp3_geometry import CONF, BLOB_FRAC, FRAME_PX
from bag_config import parse_bag
B = parse_bag()
FR = B["frames_dir"]
MAN = json.load(open(B["manifest"]))
frames = [f["i"] for f in MAN["frames"] if f["eligible"]]         # whole-bag: eligible only, no split
B["cache_dir"].mkdir(parents=True, exist_ok=True)
OUT = B["detections"]
PATHS = {42: "phase_c_yolo_multiclass", 43: "phase_c_yolo_multiclass_seed43", 44: "phase_c_yolo_multiclass_seed44"}
rows = ["seed,frame,cls,uc,v"]
# --- F007 blob audit (additive, zero-cost: reuses the guard's own area computation; the kept-
#     detection path below is unchanged, so detections.csv stays byte-identical). ---
GUARD_PX = BLOB_FRAC * FRAME_PX * FRAME_PX      # D035 15% guard: bbox area above this = blob (dropped)
NEAR_PX = 0.10 * FRAME_PX * FRAME_PX            # 10-15% near-blob band (passes guard; audited for context)
audit = {"config": {"blob_frac": BLOB_FRAC, "frame_px": FRAME_PX, "guard_thresh_px": GUARD_PX,
                    "near_lo_px": NEAR_PX, "conf": CONF, "arm": "C (Phase C multiclass)",
                    "note": "F007 canopy-blob audit. Detections whose bbox area exceeds the D035 15% guard "
                            "(blob-scale, dropped from the stream) + the 10-15% near-blob band (kept). Emitted "
                            "at the guard's own site over the same inference as base-point extraction -> free; "
                            "runs on every bag."},
         "per_seed": {}, "blobs": []}
for seed, sub in PATHS.items():
    model = YOLO(str(PKG / "results/runs" / sub / "weights/best.pt"))
    if seed == 42: print("class names:", model.names)            # confirm 0=trunk,1=pole
    ndet = nblob = nnear = 0; blobfr = set(); nearfr = set(); maxfrac = 0.0
    for fi in frames:
        img = cv2.imread(str(FR / f"{fi:05d}.jpg"))
        r = model.predict(source=img, conf=CONF, half=True, device=0, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0: continue
        xy = r.boxes.xyxy.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int)
        ar = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
        keep = ar <= GUARD_PX                                     # (== old BLOB_FRAC * FRAME_PX * FRAME_PX)
        for (x1, y1, x2, y2), c in zip(xy[keep], cl[keep]):
            rows.append(f"{seed},{fi},{c},{(x1 + x2) / 2:.1f},{y2:.1f}")
        # blob audit over the same predictions (no extra inference)
        ndet += len(ar); maxfrac = max(maxfrac, float(ar.max()) / (FRAME_PX * FRAME_PX))
        for idx in np.where(~keep)[0]:                           # blob-scale -> dropped by the guard
            nblob += 1; blobfr.add(int(fi))
            audit["blobs"].append({"seed": seed, "frame": int(fi), "area_px": round(float(ar[idx]), 1),
                                   "area_frac": round(float(ar[idx]) / (FRAME_PX * FRAME_PX), 4),
                                   "cls": int(cl[idx]), "bbox": [round(float(x), 1) for x in xy[idx]]})
        near = (ar > NEAR_PX) & keep                             # 10-15% -> kept, audited for context
        nnear += int(near.sum())
        if near.any(): nearfr.add(int(fi))
    audit["per_seed"][seed] = {"n_det": ndet, "n_blob_dropped": nblob, "n_frames_with_blob": len(blobfr),
                               "n_near_blob_10_15pct": nnear, "n_frames_near_blob": len(nearfr),
                               "max_area_frac": round(maxfrac, 4)}
    print(f"[{B['bag']}] seed {seed} done, {len(rows) - 1} dets cumulative "
          f"(blob>15%={nblob}, near10-15%={nnear}, max={maxfrac:.1%})", flush=True)
    del model; torch.cuda.empty_cache()
open(OUT, "w").write("\n".join(rows))
print(f"wrote {OUT} ({len(rows) - 1} detections)")
audit["summary"] = {"bag": B["bag"],
                    "total_blob_dropped": sum(s["n_blob_dropped"] for s in audit["per_seed"].values()),
                    "max_area_frac_overall": max((s["max_area_frac"] for s in audit["per_seed"].values()), default=0.0)}
BLOB_OUT = B["cache_dir"] / "blob_audit.json"
json.dump(audit, open(BLOB_OUT, "w"), indent=2)
print(f"wrote {BLOB_OUT} (F007 audit: {audit['summary']['total_blob_dropped']} blob-scale dets, "
      f"max {audit['summary']['max_area_frac_overall']:.1%} of frame)")
