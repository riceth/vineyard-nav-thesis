"""LiDAR cross-check of the ~2.3 deg camera yaw (F015). For 6 anchor frames, fit vine rows
from the Ouster point cloud (trunk-height band, transformed to base_link via Table 3 identity
extrinsic) and compare the LiDAR centreline heading to the camera line-fit heading.
If LiDAR ~ 0 deg while camera ~ +2.3 deg -> camera yaw independently confirmed."""
import sqlite3, json, collections
import numpy as np
from rosbags.typesys import get_typestore, Stores
ts = get_typestore(Stores.ROS2_HUMBLE)
from pathlib import Path
PKG = Path(__file__).resolve().parents[3]
con = sqlite3.connect("/workspaces/dissertation/kg_march_23_ros2/kg_march_23_ros2.db3"); cur = con.cursor()
man = json.load(open(f"{PKG}/results/geometric/march/dataset_manifest.json"))
TS = {f["i"]: int(f["timestamp_ns"]) for f in man["frames"]}
ANCHORS = [3998, 4223, 4107, 3991, 3994, 3996]

# camera heading per frame = mean across all 9 models (tilt is arm-independent)
camh = collections.defaultdict(list)
for ln in open(f"{PKG}/results/geometric/march/superseded/march_val_test_split/val_evaluation/line_fit_val_per_frame.csv").read().splitlines()[1:]:
    a, s, i, cls, off, hdg, *_ = ln.split(",")
    if cls == "two_row" and hdg: camh[int(i)].append(float(hdg))

# cheap id+timestamp index for os_cloud
ids = cur.execute("SELECT id,timestamp FROM messages WHERE topic_id=28").fetchall()
ids_ts = np.array([t for _, t in ids]); ids_id = np.array([i for i, _ in ids])

def load_cloud(t):
    j = int(np.argmin(np.abs(ids_ts - t)))
    blob = cur.execute("SELECT data FROM messages WHERE id=?", (int(ids_id[j]),)).fetchone()[0]
    pc = ts.deserialize_cdr(blob, "sensor_msgs/msg/PointCloud2")
    off = {f.name: f.offset for f in pc.fields}
    buf = np.frombuffer(pc.data, dtype=np.uint8).reshape(-1, pc.point_step)
    c = lambda n: buf[:, off[n]:off[n]+4].copy().view(np.float32).ravel()
    x, y, z = c("x"), c("y"), c("z")
    g = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & ((x != 0) | (y != 0))
    return x[g] - 0.098, y[g], z[g] + 1.0     # base_link

def fit_side(P):
    """P: Nx2 (X,Y). densest-Y cluster in near field, robust line Y=mX+c. Return (m,c,n) or None."""
    if len(P) < 5: return None
    Y = P[:, 1]
    # densest 0.5m Y-window seed, keep within +/-0.6m
    seed = np.median(Y[np.abs(Y - Y[np.argmax([np.sum(np.abs(Y - yy) <= 0.25) for yy in Y])]) <= 0.25])
    inl = np.abs(Y - seed) < 0.6; Q = P[inl]
    if len(Q) < 5: return None
    m, c = np.polyfit(Q[:, 0], Q[:, 1], 1)
    for _ in range(2):
        r = np.abs(Q[:, 1] - (m * Q[:, 0] + c)) < 0.3
        if r.sum() < 5: break
        m, c = np.polyfit(Q[r, 0], Q[r, 1], 1)
    return float(m), float(c), int(len(Q))

print(f"{'frame':>6}{'Lidar pts L/R':>14}{'L_m':>8}{'R_m':>8}{'LiDARhdg':>10}{'CAMhdg':>9}{'diff':>8}")
lid_h, cam_h = [], []
for fi in ANCHORS:
    xb, yb, zb = load_cloud(TS[fi])
    band = (zb > 0.2) & (zb < 1.2) & (xb > 1) & (xb < 8) & (np.abs(yb) < 2.6)
    P = np.column_stack([xb[band], yb[band]])
    L = P[P[:, 1] > 0.3]; R = P[P[:, 1] < -0.3]
    fL, fR = fit_side(L), fit_side(R)
    if not (fL and fR):
        print(f"{fi:>6}  fit failed (L={len(L)},R={len(R)})"); continue
    mc = (fL[0] + fR[0]) / 2
    lhdg = np.degrees(np.arctan(mc))
    chdg = float(np.mean(camh[fi])) if camh[fi] else float("nan")
    print(f"{fi:>6}{f'{len(L)}/{len(R)}':>14}{fL[0]:>8.3f}{fR[0]:>8.3f}{lhdg:>10.2f}{chdg:>9.2f}{lhdg-chdg:>8.2f}")
    lid_h.append(lhdg); cam_h.append(chdg)
print(f"\nmean LiDAR heading {np.mean(lid_h):+.2f} deg (SD {np.std(lid_h):.2f}) | mean CAMERA heading {np.mean(cam_h):+.2f} deg")
print(f"camera - LiDAR = {np.mean(cam_h)-np.mean(lid_h):+.2f} deg  (if ~+2.3 with LiDAR~0 -> camera yaw confirmed)")
