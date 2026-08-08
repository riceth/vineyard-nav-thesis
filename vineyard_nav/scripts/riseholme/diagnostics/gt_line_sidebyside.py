"""Side-by-side, zoomed: hairline sanity check (top) vs uncertainty band (bottom), same frames.

Separates two questions the combined figure conflates:
  (1) does the surveyed line fall on the row a human can see?  -- answered by the single line
  (2) how far could that line legitimately shift?              -- answered by the band

The band shows the assumed-extrinsics term ONLY (D056). The reference additionally carries a
~130 mm per-row systematic (F031) which is deliberately NOT merged into the band: combining them
would hide which error source is which, and keeping them separable is the point of D059.
Internal verification; not a dissertation figure.
"""
import sys, csv
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG = Path("/workspaces/dissertation/vineyard_nav")
sys.path.insert(0, str(PKG / "scripts" / "riseholme"))
import bag_config, projection_calibration as C, curation

BAND_M, LOOK = 0.182, 2.0
FRAMES = (1963, 3911, 3834)
B = bag_config.resolve("tue02sep")

def visible():
    for v in range(639, 300, -1):
        g = C.project_px(320, v, near_m=50.0)
        if g is not None: return float(g[0]), min(C.NEAR_M, 6.0)
xlo, xhi = visible()
pose, lines = curation.robot_pose_enu("tue02sep")
rows = {int(r["i"]): r for r in csv.DictReader(open(B["per_frame_csv"]))
        if r["arm"]=="C" and int(r["seed"])==42}

def geom(i):
    rxy, hdg = pose[i]
    line = min(((min(np.linalg.norm(rxy-l[0]), np.linalg.norm(rxy-l[-1])), nm, l)
                for nm, l in lines), key=lambda t: t[0])[2]
    cen, lo, hi, nat = [], [], [], []
    for t in np.linspace(-1.5, 2.5, 500):
        p = line[0] + t*(line[-1]-line[0])
        X, Y = curation.enu_to_cvb(p, rxy, hdg)
        if not (xlo <= X <= xhi and abs(Y) < 4.0): continue
        c = C.project_ground(X, Y); a_ = C.project_ground(X, Y-BAND_M); b_ = C.project_ground(X, Y+BAND_M)
        if c and a_ and b_:
            cen.append(c); lo.append(a_); hi.append(b_); nat.append(0.0 <= t <= 1.0)
    r = rows[i]; vis = []
    if r["cls"]=="two_row" and r["offset"]:
        off, hd = float(r["offset"]), float(r["heading"] or 0.0)
        vis = [v for v in (C.project_ground(X, off+(X-LOOK)*np.tan(np.radians(hd)))
                           for X in np.linspace(xlo, xhi, 40)) if v]
    return (np.array(cen), np.array(lo), np.array(hi), np.array(nat, bool), np.array(vis),
            float(r["offset"]) if r["offset"] else None)

fig, axes = plt.subplots(2, len(FRAMES), figsize=(4.6*len(FRAMES), 8.4))
for col, i in enumerate(FRAMES):
    img = cv2.cvtColor(cv2.imread(str(B["frames_dir"]/f"{i:05d}.jpg")), cv2.COLOR_BGR2RGB)
    cen, lo, hi, nat, vis, off = geom(i)
    allpx = np.vstack([p for p in (cen, vis) if len(p)])
    cx, cy = allpx[:,0].mean(), allpx[:,1].mean()
    half = 165
    x0, x1 = int(max(0, cx-half)), int(min(640, cx+half))
    y0, y1 = int(max(0, cy-half)), int(min(640, cy+half))
    for row in (0, 1):
        ax = axes[row, col]
        ax.imshow(img)
        if row == 0:
            if nat.any():
                ax.plot(cen[nat][:,0], cen[nat][:,1], "-", color="#059669", lw=2.6,
                        label="surveyed line")
            if (~nat).any():
                ax.plot(cen[~nat][:,0], cen[~nat][:,1], "--", color="#059669", lw=2.2,
                        label="extrapolated past row end")
        else:
            poly = np.vstack([lo, hi[::-1]])
            ax.fill(poly[:,0], poly[:,1], color="#10b981", alpha=0.40, lw=0,
                    label="surveyed line ±182 mm")
            for e in (lo, hi):
                if nat.any(): ax.plot(e[nat][:,0], e[nat][:,1], color="#059669", lw=1.0)
                if (~nat).any(): ax.plot(e[~nat][:,0], e[~nat][:,1], "--", color="#059669", lw=1.0)
        if len(vis):
            ax.plot(vis[:,0], vis[:,1], "-", color="#f59e0b", lw=2.6,
                    label=f"vision centreline ({off:+.3f} m)" if row==0 else None)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(["(1) SANITY: single line\ndoes it fall on the real row?",
                           "(2) UNCERTAINTY: ±182 mm band\nhow far could it legitimately shift?"][row],
                          fontsize=10)
        if row == 0:
            ax.set_title(f"frame {i}", fontsize=11)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
fig.suptitle("Riseholme reference check — zoomed to the drawn region (2.48–6.0 m ground window)\n"
             "top: is the surveyed line on the real row?   bottom: how far could it shift under the "
             "unmeasured camera-mounting parameters alone?\n"
             "the band excludes the reference's own ~130 mm per-row systematic, so a centreline "
             "outside it is not an unexplained error", fontsize=11)
fig.tight_layout(rect=(0,0,1,0.94))
out = PKG/"results/riseholme/tue02sep/diagnostics/gt_line_sidebyside.png"
fig.savefig(out, dpi=150); print("wrote", out)
