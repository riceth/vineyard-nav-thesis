"""Extract Phase C per-detection (class, base-point) over TEST frames for the F018 test-side
ablation. Locked upstream (conf 0.25, 15% blob guard). 3 seeds x 3149 test frames."""
import sys, json
from pathlib import Path
import numpy as np, cv2
PKG = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(PKG/"scripts"/"geometric"))
import torch; torch.multiprocessing.set_sharing_strategy("file_system")
from ultralytics import YOLO
from cp3_geometry import CONF, BLOB_FRAC, FRAME_PX
FR = PKG/"results/runs/geom_cp1_frames_640"
MAN = json.load(open(PKG/"results/geometric/march/dataset_manifest.json"))
test = [f["i"] for f in MAN["frames"] if f["split"] == "test"]
from paths import CACHE_DIR
OUT = CACHE_DIR / "detections_test.csv"
PATHS = {42:"phase_c_yolo_multiclass", 43:"phase_c_yolo_multiclass_seed43", 44:"phase_c_yolo_multiclass_seed44"}
rows = ["seed,frame,cls,uc,v"]
for seed, sub in PATHS.items():
    model = YOLO(str(PKG/"results/runs"/sub/"weights/best.pt"))
    for fi in test:
        img = cv2.imread(str(FR/f"{fi:05d}.jpg"))
        r = model.predict(source=img, conf=CONF, half=True, device=0, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0: continue
        xy = r.boxes.xyxy.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int)
        ar = (xy[:,2]-xy[:,0])*(xy[:,3]-xy[:,1]); keep = ar <= BLOB_FRAC*FRAME_PX*FRAME_PX
        for (x1,y1,x2,y2), c in zip(xy[keep], cl[keep]):
            rows.append(f"{seed},{fi},{c},{(x1+x2)/2:.1f},{y2:.1f}")
    print(f"seed {seed} done, {len(rows)-1} dets cumulative", flush=True)
    del model; torch.cuda.empty_cache()
open(OUT,"w").write("\n".join(rows))
print(f"wrote {OUT} ({len(rows)-1} detections)")
