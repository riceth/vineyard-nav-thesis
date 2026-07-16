"""F017 test-side: LiDAR row heading vs camera heading on 6 test anchor frames (2 per test
corridor 2/3/4). Confirms tilt sensor-commonality on held-out data. Ouster PC2, no model inference."""
import sqlite3, json, collections
import numpy as np
from rosbags.typesys import get_typestore, Stores
ts = get_typestore(Stores.ROS2_HUMBLE)
from pathlib import Path
PKG = Path(__file__).resolve().parents[2]
con = sqlite3.connect("/workspaces/dissertation/kg_march_23_ros2/kg_march_23_ros2.db3"); cur = con.cursor()
man = json.load(open(f"{PKG}/results/geometric/march/dataset_manifest.json"))
TS = {f["i"]: int(f["timestamp_ns"]) for f in man["frames"]}; CORR = {f["i"]: f["corridor"] for f in man["frames"]}
# camera heading per frame = mean across 9 models (cp6); pick anchors = two_row C42, 2 per test corridor
camh = collections.defaultdict(list); c42 = set()
for ln in open(f"{PKG}/results/geometric/march/final/test_evaluation/line_fit_test_per_frame.csv").read().splitlines()[1:]:
    a,s,i,cls,off,hdg,mc,nb = ln.split(",")
    if cls == "two_row" and hdg:
        camh[int(i)].append(float(hdg))
        if a == "C" and s == "42": c42.add(int(i))
anchors = []
for cc in (2, 3, 4):
    fs = sorted(i for i in c42 if CORR.get(i) == cc)
    anchors += fs[len(fs)//3: len(fs)//3+2]   # 2 mid-pass frames per corridor
ids = cur.execute("SELECT id,timestamp FROM messages WHERE topic_id=28").fetchall()
ids_ts = np.array([t for _,t in ids]); ids_id = np.array([i for i,_ in ids])
def load(t):
    j = int(np.argmin(np.abs(ids_ts-t))); blob = cur.execute("SELECT data FROM messages WHERE id=?", (int(ids_id[j]),)).fetchone()[0]
    pc = ts.deserialize_cdr(blob, "sensor_msgs/msg/PointCloud2"); off = {f.name: f.offset for f in pc.fields}
    buf = np.frombuffer(pc.data, dtype=np.uint8).reshape(-1, pc.point_step)
    c = lambda n: buf[:, off[n]:off[n]+4].copy().view(np.float32).ravel()
    x,y,z = c("x"),c("y"),c("z"); g = np.isfinite(x)&np.isfinite(y)&np.isfinite(z)&((x!=0)|(y!=0))
    return x[g]-0.098, y[g], z[g]+1.0
def fit_side(P):
    if len(P) < 5: return None
    Y = P[:,1]; seed = np.median(Y[np.abs(Y-Y[np.argmax([np.sum(np.abs(Y-yy)<=0.25) for yy in Y])])<=0.25])
    inl = np.abs(Y-seed) < 0.6; Q = P[inl]
    if len(Q) < 5: return None
    m,c = np.polyfit(Q[:,0], Q[:,1], 1)
    for _ in range(2):
        r = np.abs(Q[:,1]-(m*Q[:,0]+c)) < 0.3
        if r.sum() < 5: break
        m,c = np.polyfit(Q[r,0], Q[r,1], 1)
    return float(m)
print(f"{'frame':>6}{'corr':>5}{'L/R':>9}{'LiDARhdg':>10}{'CAMhdg':>9}{'diff':>8}")
lid,cam = [],[]
for fi in anchors:
    xb,yb,zb = load(TS[fi]); band = (zb>0.2)&(zb<1.2)&(xb>1)&(xb<8)&(np.abs(yb)<2.6)
    P = np.column_stack([xb[band],yb[band]]); L = P[P[:,1]>0.3]; R = P[P[:,1]<-0.3]
    mL,mR = fit_side(L), fit_side(R)
    if mL is None or mR is None: print(f"{fi:>6}{CORR[fi]:>5}  fit fail (L{len(L)}/R{len(R)})"); continue
    lh = np.degrees(np.arctan((mL+mR)/2)); ch = float(np.mean(camh[fi]))
    print(f"{fi:>6}{CORR[fi]:>5}{f'{len(L)}/{len(R)}':>9}{lh:>10.2f}{ch:>9.2f}{lh-ch:>8.2f}")
    lid.append(lh); cam.append(ch)
print(f"\nmean LiDAR {np.mean(lid):+.2f} deg (SD {np.std(lid):.2f}) | mean CAMERA {np.mean(cam):+.2f} deg | camera-LiDAR {np.mean(cam)-np.mean(lid):+.2f}")
print("(val F017: LiDAR +3.84, camera +3.25 -> sensor-common; test confirms if both nonzero & agree in sign)")
