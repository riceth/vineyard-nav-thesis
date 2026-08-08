"""Screen every extracted Riseholme frame for identifiable people before any figure is published.

Same standard and same COCO backbone (yolo11n-seg, class 0 = person) applied to the June Ktima
figure set. A person was already observed in the Tue-02-Sep imagery during reconnaissance, so this
is a hard prerequisite, not a precaution: figures.py must refuse to use any frame this flags.

Writes results/riseholme/<bag>/diagnostics/privacy_screen.json with a per-frame verdict and a
publishable allow-list. Frames are flagged on ANY person detection above the confidence floor --
deliberately conservative, since a false positive costs one candidate frame and a false negative
publishes someone's face.

  python3 scripts/riseholme/diagnostics/privacy_screen.py --bag tue02sep
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PKG = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PKG / "scripts" / "riseholme"))
import cuda_preload  # noqa: F401,E402
from ultralytics import YOLO  # noqa: E402
import bag_config  # noqa: E402

PERSON_CLS = 0
CONF = 0.20          # deliberately below the pipeline's 0.25: err toward flagging
AREA_NOTE = 0.001    # fraction of frame below which a hit is likely distant/unidentifiable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--conf", type=float, default=CONF)
    a = ap.parse_args()
    B = bag_config.resolve(a.bag)
    frames = sorted(B["frames_dir"].glob("*.jpg"))
    if not frames:
        raise SystemExit(f"no extracted frames in {B['frames_dir']}")

    model = YOLO(str(PKG / "yolo11n-seg.pt"))
    flagged, details = [], []
    for n, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        r = model.predict(source=img, conf=a.conf, device=0, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        cls = r.boxes.cls.cpu().numpy().astype(int)
        keep = cls == PERSON_CLS
        if not keep.any():
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()[keep]
        conf = r.boxes.conf.cpu().numpy()[keep]
        areas = ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])) / (img.shape[0] * img.shape[1])
        idx = int(fp.stem)
        flagged.append(idx)
        details.append({"frame": idx, "n_person": int(keep.sum()),
                        "max_conf": round(float(conf.max()), 3),
                        "max_area_frac": round(float(areas.max()), 5),
                        "likely_distant": bool(areas.max() < AREA_NOTE)})
        if n % 500 == 0:
            print(f"  screened {n}/{len(frames)}, {len(flagged)} flagged so far", flush=True)

    allow = [int(f.stem) for f in frames if int(f.stem) not in set(flagged)]
    out = B["out_dir"].parent.parent / "diagnostics" / "privacy_screen.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"bag": a.bag, "backbone": "yolo11n-seg (COCO)", "person_class": PERSON_CLS,
               "conf_floor": a.conf, "frames_screened": len(frames),
               "flagged_count": len(flagged), "flagged": sorted(flagged),
               "publishable_count": len(allow),
               "note": "figures.py MUST intersect its candidates with `publishable`. Flagging is "
                       "deliberately conservative: a false positive costs one candidate frame, a "
                       "false negative publishes an identifiable person."},
              open(out, "w"), indent=2)

    print(f"\nscreened {len(frames)} frames at conf>={a.conf}")
    print(f"  FLAGGED (contain a person): {len(flagged)}  ({100*len(flagged)/len(frames):.1f}%)")
    print(f"  publishable                : {len(allow)}")
    if details:
        d = sorted(details, key=lambda x: -x["max_area_frac"])
        print(f"  largest person detections (frame, conf, % of frame):")
        for x in d[:8]:
            print(f"    {x['frame']:6d}  conf {x['max_conf']:.2f}  {100*x['max_area_frac']:.2f}%"
                  f"{'  [distant]' if x['likely_distant'] else ''}")
    print(f"\nwrote {out.relative_to(PKG)}")


if __name__ == "__main__":
    main()
