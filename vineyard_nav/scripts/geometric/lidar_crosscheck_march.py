"""March pooled LiDAR cross-check (D040, F017). Consolidates lidar_crosscheck_val.py +
lidar_crosscheck_test.py.

LiDAR row heading (Ouster PC2, trunk-height band, transformed to base_link via the Table 3
identity extrinsic) vs the camera line-fit heading (pooled per-frame CSV, mean across all 9
models -- tilt is arm-independent) on anchor frames spanning the pooled corridors: PER_CORR
mid-pass anchors per corridor 0-4, drawn from the Phase C seed-42 two_row frames. Confirms F017
(camera yaw / row tilt is sensor-common, not a camera artefact) on the whole March bag.

Writes results/geometric/march/final/march_evaluation/lidar_crosscheck_march.json.
"""
import sqlite3
import json
import collections
from pathlib import Path

import numpy as np
from rosbags.typesys import get_typestore, Stores

ts = get_typestore(Stores.ROS2_HUMBLE)

PKG = Path(__file__).resolve().parents[2]
BAG = "/workspaces/dissertation/kg_march_23_ros2/kg_march_23_ros2.db3"
con = sqlite3.connect(BAG)
cur = con.cursor()
MAN = json.load(open(f"{PKG}/results/geometric/march/dataset_manifest.json"))
TS = {f["i"]: int(f["timestamp_ns"]) for f in MAN["frames"]}
CORR = {f["i"]: f["corridor"] for f in MAN["frames"]}
PF = f"{PKG}/results/geometric/march/final/march_evaluation/line_fit_march_per_frame.csv"
OUT = PKG / "results/geometric/march/final/march_evaluation/lidar_crosscheck_march.json"
CORRIDORS = [0, 1, 2, 3, 4]   # all 5 pooled corridors
PER_CORR = 2                  # mid-pass anchors per corridor

# camera heading per frame = mean across 9 models; anchors = seed-42 arm-C two_row frames
camh = collections.defaultdict(list)
c42 = set()
for ln in open(PF).read().splitlines()[1:]:
    a, s, i, cls, off, hdg, mc, nb = ln.split(",")
    if cls == "two_row" and hdg:
        camh[int(i)].append(float(hdg))
        if a == "C" and s == "42":
            c42.add(int(i))
anchors = []
for cc in CORRIDORS:
    fs = sorted(i for i in c42 if CORR.get(i) == cc)
    if not fs:
        continue
    anchors += fs[len(fs) // 3: len(fs) // 3 + PER_CORR]   # mid-pass frames per corridor

ids = cur.execute("SELECT id,timestamp FROM messages WHERE topic_id=28").fetchall()
ids_ts = np.array([t for _, t in ids])
ids_id = np.array([i for i, _ in ids])


def load_cloud(t):
    j = int(np.argmin(np.abs(ids_ts - t)))
    blob = cur.execute("SELECT data FROM messages WHERE id=?", (int(ids_id[j]),)).fetchone()[0]
    pc = ts.deserialize_cdr(blob, "sensor_msgs/msg/PointCloud2")
    off = {f.name: f.offset for f in pc.fields}
    buf = np.frombuffer(pc.data, dtype=np.uint8).reshape(-1, pc.point_step)
    c = lambda n: buf[:, off[n]:off[n] + 4].copy().view(np.float32).ravel()
    x, y, z = c("x"), c("y"), c("z")
    g = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & ((x != 0) | (y != 0))
    return x[g] - 0.098, y[g], z[g] + 1.0     # base_link (Table 3 identity extrinsic)


def fit_side(P):
    """P: Nx2 (X,Y). densest-Y cluster in near field, robust line Y=mX+c. Return (m,c,n) or None."""
    if len(P) < 5:
        return None
    Y = P[:, 1]
    seed = np.median(Y[np.abs(Y - Y[np.argmax([np.sum(np.abs(Y - yy) <= 0.25) for yy in Y])]) <= 0.25])
    inl = np.abs(Y - seed) < 0.6
    Q = P[inl]
    if len(Q) < 5:
        return None
    m, c = np.polyfit(Q[:, 0], Q[:, 1], 1)
    for _ in range(2):
        r = np.abs(Q[:, 1] - (m * Q[:, 0] + c)) < 0.3
        if r.sum() < 5:
            break
        m, c = np.polyfit(Q[r, 0], Q[r, 1], 1)
    return float(m), float(c), int(len(Q))


print(f"{'frame':>6}{'corr':>5}{'L/R':>11}{'LiDARhdg':>10}{'CAMhdg':>9}{'diff':>8}")
results, lid, cam = [], [], []
for fi in anchors:
    xb, yb, zb = load_cloud(TS[fi])
    band = (zb > 0.2) & (zb < 1.2) & (xb > 1) & (xb < 8) & (np.abs(yb) < 2.6)
    P = np.column_stack([xb[band], yb[band]])
    L = P[P[:, 1] > 0.3]
    R = P[P[:, 1] < -0.3]
    fL, fR = fit_side(L), fit_side(R)
    if not (fL and fR):
        print(f"{fi:>6}{CORR[fi]:>5}   fit failed (L={len(L)},R={len(R)})")
        results.append({"frame": fi, "corridor": CORR[fi], "fit": "failed", "nL": len(L), "nR": len(R)})
        continue
    mc = (fL[0] + fR[0]) / 2
    lhdg = float(np.degrees(np.arctan(mc)))
    chdg = float(np.mean(camh[fi])) if camh[fi] else float("nan")
    print(f"{fi:>6}{CORR[fi]:>5}{f'{len(L)}/{len(R)}':>11}{lhdg:>10.2f}{chdg:>9.2f}{lhdg - chdg:>8.2f}")
    results.append({"frame": fi, "corridor": CORR[fi], "nL": len(L), "nR": len(R),
                    "mL": round(fL[0], 4), "mR": round(fR[0], 4),
                    "lidar_hdg": round(lhdg, 2), "cam_hdg": round(chdg, 2), "diff": round(lhdg - chdg, 2)})
    lid.append(lhdg)
    cam.append(chdg)

report = {"config": {"anchors": anchors, "per_corridor": PER_CORR, "corridors": CORRIDORS,
                     "band": "z 0.2-1.2m, x 1-8m, |y|<2.6m",
                     "extrinsic": "Table 3 identity (x-0.098, z+1.0)",
                     "camera_heading": "mean across 9 models (tilt arm-independent)"},
          "anchors": results,
          "mean_lidar_hdg": round(float(np.mean(lid)), 2) if lid else None,
          "sd_lidar_hdg": round(float(np.std(lid)), 2) if lid else None,
          "mean_cam_hdg": round(float(np.mean(cam)), 2) if cam else None,
          "camera_minus_lidar": round(float(np.mean(cam) - np.mean(lid)), 2) if lid else None,
          "n_fitted": len(lid)}
OUT.write_text(json.dumps(report, indent=2))

if lid:
    print(f"\nmean LiDAR heading {np.mean(lid):+.2f} deg (SD {np.std(lid):.2f}) | "
          f"mean CAMERA heading {np.mean(cam):+.2f} deg | camera-LiDAR {np.mean(cam) - np.mean(lid):+.2f} deg")
    print("(F017: both nonzero & agreeing in sign -> row tilt is sensor-common, not a camera artefact)")
print(f"wrote {OUT.name}")
