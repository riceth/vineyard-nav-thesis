"""Diagnostic: does the surveyed mid-row line actually correspond to the real row in the image?

Deliberately separates two questions that the banded figure conflates:
  (1) does the reference line land on the row a human can see?   <- THIS script: one clean line
  (2) how far could it legitimately shift given the calibration? <- the banded figure

A single hairline answers (1) unambiguously. Internal verification only; not a dissertation figure.

Also reports, across the whole dataset, how often the surveyed line is actually within the camera's
visible ground window -- so the figure cannot flatter itself by only showing frames where it is.

  python3 scripts/riseholme/diagnostics/gt_line_sanity.py --bag tue02sep
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

PKG = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PKG / "scripts" / "riseholme"))
import bag_config                       # noqa: E402
import projection_calibration as C      # noqa: E402
import curation                         # noqa: E402

LOOKAHEAD_M = 2.0


def visible_window():
    """The ground range this camera can actually see, from the image bottom to the horizon cutoff."""
    near = None
    for v in range(639, 300, -1):
        g = C.project_px(320, v, near_m=50.0)
        if g is not None:
            near = float(g[0]); break
    return near, C.NEAR_M


def line_in_view(line, rxy, hdg, xlo, xhi, extrapolate=True):
    """Points of the (optionally extended) surveyed line that fall inside the visible window."""
    a, b = line[0], line[-1]
    ts = np.linspace(-1.5, 2.5, 400) if extrapolate else np.linspace(0, 1, 200)
    pts, native = [], []
    for t in ts:
        p = a + t * (b - a)
        X, Y = curation.enu_to_cvb(p, rxy, hdg)
        if xlo <= X <= xhi and abs(Y) < 4.0:
            px = C.project_ground(X, Y)
            if px:
                pts.append(px); native.append(0.0 <= t <= 1.0)
    return np.array(pts), np.array(native, dtype=bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--frames", default="1963,3911,3834")
    a = ap.parse_args()
    B = bag_config.resolve(a.bag)
    want = [int(x) for x in a.frames.split(",")]
    xlo, xhi = visible_window()
    print(f"camera visible ground window: {xlo:.2f} m (image bottom) .. {xhi:.2f} m (IPM cutoff)")

    pose, lines = curation.robot_pose_enu(a.bag)
    rows = {int(r["i"]): r for r in csv.DictReader(open(B["per_frame_csv"]))
            if r["arm"] == "C" and int(r["seed"]) == 42}

    # ---- dataset-wide: how often is the surveyed line actually in view? ----
    flagged, _ = curation.publishable(a.bag)
    tot = seg = ext = 0
    for i, (rxy, hdg) in pose.items():
        if i in flagged or i not in rows or rows[i]["cls"] != "two_row":
            continue
        tot += 1
        best = min(((min(np.linalg.norm(rxy - l[0]), np.linalg.norm(rxy - l[-1])), nm, l)
                    for nm, l in lines), key=lambda t: t[0])[2]
        p_seg, _ = line_in_view(best, rxy, hdg, xlo, xhi, extrapolate=False)
        p_ext, _ = line_in_view(best, rxy, hdg, xlo, xhi, extrapolate=True)
        seg += len(p_seg) > 3
        ext += len(p_ext) > 3
    print(f"across {tot} publishable two-row frames with a pose:")
    print(f"  surveyed SEGMENT in view          : {seg} ({100*seg/max(tot,1):.1f}%)")
    print(f"  line in view when EXTRAPOLATED    : {ext} ({100*ext/max(tot,1):.1f}%)")
    print(f"  -> the {100*(ext-seg)/max(tot,1):.1f}% difference is frames where the robot has driven")
    print(f"     past a surveyed row end, so the finite segment stops short of the camera's view.")

    fig, axes = plt.subplots(1, len(want), figsize=(5.0 * len(want), 4.6))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, want):
        img = cv2.imread(str(B["frames_dir"] / f"{i:05d}.jpg"))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        note = []
        if i in pose and i in rows:
            rxy, hdg = pose[i]
            best = min(((min(np.linalg.norm(rxy - l[0]), np.linalg.norm(rxy - l[-1])), nm, l)
                        for nm, l in lines), key=lambda t: t[0])
            pts, native = line_in_view(best[2], rxy, hdg, xlo, xhi)
            if len(pts) > 3:
                if native.any():
                    ax.plot(pts[native][:, 0], pts[native][:, 1], "-", color="#10b981", lw=2.4,
                            label="surveyed line (within segment)")
                if (~native).any():
                    ax.plot(pts[~native][:, 0], pts[~native][:, 1], "--", color="#10b981", lw=1.8,
                            alpha=0.85, label="extrapolated past row end")
            else:
                note.append("line not in view")
            r = rows[i]
            if r["cls"] == "two_row" and r["offset"]:
                off = float(r["offset"]); head = float(r["heading"] or 0.0)
                vis = []
                for X in np.linspace(xlo, min(xhi, 6.0), 30):
                    px = C.project_ground(X, off + (X - LOOKAHEAD_M) * np.tan(np.radians(head)))
                    if px:
                        vis.append(px)
                if vis:
                    v = np.array(vis)
                    ax.plot(v[:, 0], v[:, 1], "-", color="#f59e0b", lw=2.4,
                            label=f"vision centreline ({off:+.3f} m)")
            else:
                note.append(f"cls={r['cls']}")
        else:
            note.append("no pose")
        ax.set_xlim(0, 640); ax.set_ylim(640, 0); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"frame {i}" + ("  ·  " + ", ".join(note) if note else ""), fontsize=9)
    axes[0].legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.suptitle("SANITY CHECK (no uncertainty band): does the surveyed line fall on the real row?\n"
                 f"drawn only over the camera's visible ground window {xlo:.2f}–{min(xhi,6.0):.1f} m; "
                 "dashed = extrapolated past a surveyed row end", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    out = B["out_dir"].parent.parent / "diagnostics" / "gt_line_sanity.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\nwrote {out.relative_to(PKG)}")


if __name__ == "__main__":
    main()
