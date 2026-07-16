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
from single_arm_dryrun import CONF, BLOB_FRAC, FRAME_PX
from bag_config import parse_bag
B = parse_bag()
FR = B["frames_dir"]
MAN = json.load(open(B["manifest"]))
frames = [f["i"] for f in MAN["frames"] if f["eligible"]]         # whole-bag: eligible only, no split
B["cache_dir"].mkdir(parents=True, exist_ok=True)
OUT = B["detections"]
PATHS = {42: "phase_c_yolo_multiclass", 43: "phase_c_yolo_multiclass_seed43", 44: "phase_c_yolo_multiclass_seed44"}
rows = ["seed,frame,cls,uc,v"]
for seed, sub in PATHS.items():
    model = YOLO(str(PKG / "results/runs" / sub / "weights/best.pt"))
    if seed == 42: print("class names:", model.names)            # confirm 0=trunk,1=pole
    for fi in frames:
        img = cv2.imread(str(FR / f"{fi:05d}.jpg"))
        r = model.predict(source=img, conf=CONF, half=True, device=0, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0: continue
        xy = r.boxes.xyxy.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int)
        ar = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
        keep = ar <= BLOB_FRAC * FRAME_PX * FRAME_PX
        for (x1, y1, x2, y2), c in zip(xy[keep], cl[keep]):
            rows.append(f"{seed},{fi},{c},{(x1 + x2) / 2:.1f},{y2:.1f}")
    print(f"[{B['bag']}] seed {seed} done, {len(rows) - 1} dets cumulative", flush=True)
    del model; torch.cuda.empty_cache()
open(OUT, "w").write("\n".join(rows))
print(f"wrote {OUT} ({len(rows) - 1} detections)")
