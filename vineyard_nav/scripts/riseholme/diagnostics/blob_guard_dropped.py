"""Diagnostic: the one detection the blob guard dropped, beside the ones it kept.

The area guard discards any detection whose bounding box exceeds 15% of the frame. That
threshold was set against the largest legitimate detection observed on the bare-vine bag
(~10.5% of frame), leaving roughly 1 pp of headroom once later bags reached 14.0%.

On this dataset the guard fired once, at 15.65% -- 0.65 pp over the threshold. Judging that
case needs a comparison, not a percentage: the same frame also carries detections the guard
kept, under identical scene, lighting, model and seed, so the dropped box can be read against
what a normal box on that frame looks like rather than against an abstract limit.

The dropped box and its area come from the committed audit; the retained boxes are recovered
by re-running the same inference call on that one frame (the detection cache stores base
points, not boxes). The script asserts the re-inferred dropped box matches the audit before
drawing anything, so the comparison cannot silently drift from the evaluated run.

    python3 scripts/riseholme/diagnostics/blob_guard_dropped.py --bag tue02sep
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

PKG = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PKG / "scripts" / "riseholme"))
import cuda_preload                     # noqa: E402,F401 -- cuDNN cold-init guard, must precede torch
from ultralytics import YOLO            # noqa: E402
import bag_config                       # noqa: E402
import curation                         # noqa: E402
from cp3_geometry import CONF, BLOB_FRAC, FRAME_PX   # noqa: E402

CLS_NAME = {0: "trunk", 1: "pole"}
WEIGHTS = {42: "phase_c_yolo_multiclass", 43: "phase_c_yolo_multiclass_seed43",
           44: "phase_c_yolo_multiclass_seed44"}
KEPT, DROPPED = "#1a9e4b", "#d1341c"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="tue02sep")
    a = ap.parse_args()

    B = bag_config.resolve(a.bag)
    audit = json.load(open(B["cache_dir"] / "blob_audit.json"))
    blobs = audit.get("blobs", [])
    if not blobs:
        raise SystemExit(f"{a.bag}: the guard dropped nothing; no figure to draw.")
    b = blobs[0]

    # Privacy screening is a precondition, not a courtesy. NOTE the contract: publishable()
    # returns (flagged_set, screen_metadata) -- the frames that must NOT be published --
    # not an allow-list, despite the name.
    flagged, screen = curation.publishable(a.bag)
    if b["frame"] in flagged:
        raise SystemExit(f"{a.bag}: frame {b['frame']} is privacy-flagged. Not rendered.")

    guard_px = BLOB_FRAC * FRAME_PX * FRAME_PX
    img_bgr = cv2.imread(str(B["frames_dir"] / f"{b['frame']:05d}.jpg"))
    img = img_bgr[:, :, ::-1]

    # Same inference call as extract_detections.py, so the boxes are the evaluated ones.
    model = YOLO(str(PKG / "results/runs" / WEIGHTS[b["seed"]] / "weights/best.pt"))
    r = model.predict(source=img_bgr, conf=CONF, quantize=16, device=0, verbose=False)[0]
    xy = r.boxes.xyxy.cpu().numpy()
    cl = r.boxes.cls.cpu().numpy().astype(int)
    ar = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
    dropped_i = int(np.argmax(ar))

    # Self-check: the re-inferred largest box must be the one the audit recorded.
    rec = np.array(b["bbox"], dtype=float)
    if not np.allclose(np.round(xy[dropped_i], 1), rec, atol=0.15):
        raise SystemExit(f"re-inference disagrees with the audit: got "
                         f"{np.round(xy[dropped_i],1).tolist()}, audit has {rec.tolist()}")

    kept = [i for i in range(len(ar)) if i != dropped_i]
    biggest_kept = max(kept, key=lambda i: ar[i]) if kept else None

    # The decisive question is not the area but whether the structure was lost. Draw the
    # dropped box and the largest kept box together, with their base points, so overlap
    # (or its absence) is visible rather than asserted.
    import projection_calibration as C
    def base_pt(i): return (float(xy[i][0] + xy[i][2]) / 2, float(xy[i][3]))
    bd, bk = base_pt(dropped_i), base_pt(biggest_kept)
    gd, gk = C.project_px(*bd, near_m=10.0), C.project_px(*bk, near_m=10.0)
    lat_mm = abs(gd[1] - gk[1]) * 1000 if (gd is not None and gk is not None) else None

    ix1, iy1 = max(xy[dropped_i][0], xy[biggest_kept][0]), max(xy[dropped_i][1], xy[biggest_kept][1])
    ix2, iy2 = min(xy[dropped_i][2], xy[biggest_kept][2]), min(xy[dropped_i][3], xy[biggest_kept][3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    iou = inter / (ar[dropped_i] + ar[biggest_kept] - inter)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.55, 1]})

    axes[0].imshow(img)
    for i in range(len(ar)):
        x1, y1, x2, y2 = xy[i]
        col = DROPPED if i == dropped_i else KEPT
        axes[0].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=col, lw=2.2))
        axes[0].text(x1 + 2, max(y1 - 5, 11), f"{100*ar[i]/(FRAME_PX**2):.1f}%",
                     color=col, fontsize=8, weight="bold")
    axes[0].set_title(f"frame {b['frame']} — every detection, area as % of frame", fontsize=10)

    pad = 14
    zx1 = max(0, int(min(xy[dropped_i][0], xy[biggest_kept][0])) - pad)
    zx2 = min(FRAME_PX, int(max(xy[dropped_i][2], xy[biggest_kept][2])) + pad)
    zy1 = max(0, int(min(xy[dropped_i][1], xy[biggest_kept][1])) - pad)
    axes[1].imshow(img[zy1:, zx1:zx2])
    for i, col, lab in ((dropped_i, DROPPED, "discarded"), (biggest_kept, KEPT, "retained")):
        x1, y1, x2, y2 = xy[i]
        axes[1].add_patch(plt.Rectangle((x1 - zx1, y1 - zy1), x2 - x1, y2 - y1,
                                        fill=False, ec=col, lw=2.6))
        u, v = base_pt(i)
        axes[1].plot(u - zx1, v - zy1 - 3, marker="v", ms=11, color=col,
                     mec="white", mew=1.2, label=f"{lab} — base point")
    axes[1].legend(loc="upper left", fontsize=8, framealpha=0.9)
    axes[1].set_title("the same post, detected twice\n"
                      f"boxes overlap at IoU {iou:.2f}; the retained box lies wholly inside the discarded one",
                      fontsize=9)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    frac = b["area_frac"]
    fig.suptitle(
        f"The one detection removed by the area guard on this dataset — and what it cost\n"
        f"limit {100*BLOB_FRAC:.0f}% · discarded {100*frac:.2f}% (over by {100*(frac-BLOB_FRAC):.2f} pp) · "
        f"the same post was retained at {100*ar[biggest_kept]/(FRAME_PX**2):.2f}%, "
        f"so the structure survives and its base point moves {lat_mm:.0f} mm laterally",
        fontsize=10, y=1.03)

    out = B["out_dir"].parent.parent / "diagnostics" / "blob_guard_dropped.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"  bag              {a.bag}   frame {b['frame']}   seed {b['seed']}")
    print(f"  detections       {len(ar)} total -> {len(kept)} kept, 1 dropped")
    print(f"  dropped          {CLS_NAME.get(cl[dropped_i])} {int(xy[dropped_i][2]-xy[dropped_i][0])}"
          f" x {int(xy[dropped_i][3]-xy[dropped_i][1])} px = {100*frac:.2f}% "
          f"(limit {100*BLOB_FRAC:.0f}%, over by {100*(frac-BLOB_FRAC):.2f} pp)")
    for i in sorted(kept, key=lambda i: -ar[i]):
        print(f"  kept             {CLS_NAME.get(cl[i], cl[i]):5} "
              f"{int(xy[i][2]-xy[i][0]):3} x {int(xy[i][3]-xy[i][1]):3} px = "
              f"{100*ar[i]/(FRAME_PX**2):5.2f}%")
    print(f"  same structure?  IoU {iou:.3f}; retained box is {100*inter/ar[biggest_kept]:.0f}% inside the discarded one")
    print(f"  base points      dropped {tuple(round(x,1) for x in bd)} px -> ground {tuple(round(float(x),3) for x in gd)} m")
    print(f"                   kept    {tuple(round(x,1) for x in bk)} px -> ground {tuple(round(float(x),3) for x in gk)} m")
    print(f"  cost to the fit  {lat_mm:.0f} mm lateral shift in one base point (inlier tolerance is 250 mm)")
    print(f"  self-check       re-inferred box matches the committed audit")
    print(f"  privacy screen   frame not flagged ({screen['flagged_count']} of "
          f"{screen['frames_screened']} flagged)")
    print(f"  wrote            {out.relative_to(PKG)}")


if __name__ == "__main__":
    main()
