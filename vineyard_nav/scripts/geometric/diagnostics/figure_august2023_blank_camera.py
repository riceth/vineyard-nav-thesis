"""Evidence figure for D054: the august2023 session's RGB camera recorded no imagery.

Renders three frames sampled across the august2023 session against three from july2023 at
comparable elapsed times, straight from each bag's .db3 — no pipeline stage in between, so the
figure shows what was recorded rather than what the pipeline made of it. Per-frame pixel mean and
standard deviation are annotated: a std of 0 is a mathematically uniform image.

  python3 scripts/geometric/diagnostics/figure_august2023_blank_camera.py
"""
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rosbags.typesys import Stores, get_typestore

PKG = Path(__file__).resolve().parents[3]
GIT = PKG.parent
sys.path.insert(0, str(PKG / "scripts" / "geometric"))

TS = get_typestore(Stores.ROS2_HUMBLE)
CAM = "/front/zed_node/rgb/image_rect_color/compressed"
OUT = PKG / "results/geometric/august2023/diagnostics/august2023_blank_camera.png"
# Near-start / mid / near-end of each session. july2023's are shifted off the round fractions
# because people appear in frame around mid-session (a vineyard worker at ~0.50); this figure is
# publishable, so the comparison frames are chosen from stretches with no person present.
FRACS = {"august2023": [0.05, 0.50, 0.95], "july2023": [0.05, 0.72, 0.95]}


def grab(db3, fracs):
    """Return [(elapsed_s, image_bgr, mean, std, jpeg_bytes)] at the given fractions of the session."""
    con = sqlite3.connect(str(db3))
    cur = con.cursor()
    tid = cur.execute("SELECT id FROM topics WHERE name=?", (CAM,)).fetchone()[0]
    rows = list(cur.execute(
        "SELECT id, timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,)))
    t0 = rows[0][1]
    out = []
    for fr in fracs:
        mid, ts = rows[int(fr * (len(rows) - 1))]
        data = cur.execute("SELECT data FROM messages WHERE id=?", (mid,)).fetchone()[0]
        m = TS.deserialize_cdr(bytes(data), "sensor_msgs/msg/CompressedImage")
        im = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if im.ndim == 3 and im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        out.append(((ts - t0) / 1e9, im, float(im.mean()), float(im.std()), len(m.data)))
    con.close()
    return out


def main():
    bags = [("august2023", "august2023  (2023-08-01)", GIT / "kg_august2023_ros2/kg_august2023_ros2.db3"),
            ("july2023", "july2023  (2023-07-25)", GIT / "kg_july2023_ros2/kg_july2023_ros2.db3")]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.4))
    for r, (bag, label, db3) in enumerate(bags):
        for c, (t, im, mu, sd, nb) in enumerate(grab(db3, FRACS[bag])):
            ax = axes[r][c]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            # a pure-white frame is invisible against the figure ground; outline every panel
            for s in ax.spines.values():
                s.set_edgecolor("0.45")
                s.set_linewidth(1.2)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"t = {t:6.1f} s", fontsize=10, pad=4)
            ax.set_xlabel(f"mean {mu:.1f}   std {sd:.1f}   {nb/1024:.0f} kB",
                          fontsize=9, labelpad=3,
                          color="#b00020" if sd == 0 else "#20603a")
        axes[r][0].text(-0.09, 0.5, label, transform=axes[r][0].transAxes, rotation=90,
                        va="center", ha="center", fontsize=11, fontweight="bold")
    fig.suptitle("august2023 recorded no imagery — every frame is a uniform white image\n"
                 "$/front/zed\\_node/rgb/image\\_rect\\_color/compressed$, decoded directly from each bag",
                 fontsize=12)
    fig.tight_layout(rect=(0.015, 0, 1, 0.99))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"wrote {OUT.relative_to(GIT)}")


if __name__ == "__main__":
    main()
