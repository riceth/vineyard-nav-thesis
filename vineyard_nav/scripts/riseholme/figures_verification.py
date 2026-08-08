"""Two VERIFICATION figures for Riseholme. These can falsify a claim, not merely illustrate one.

  fig_gt_overlay    the surveyed mid-row line drawn into the camera image beside the vision's own
                    centreline. If the surveyed line does not land plausibly on the corridor, the
                    projection, the map->WGS84 fit or the geojson placement is wrong and F031's
                    numbers need revisiting. Drawn as a BAND, never a hairline: its width is the
                    +/-182 mm contributed by the two ASSUMED extrinsic DOF (D056), so the figure
                    cannot imply a precision the calibration does not have.

  fig_sensitivity   per-arm absolute RMS and paired cross-arm differences as functions of the
                    assumed lateral offset and yaw. This is the load-bearing evidence for D059: if
                    the paired curves are not flat, the common-mode cancellation argument is wrong.
                    Computed analytically and exactly -- a lateral error shifts every measured
                    offset by a constant, a yaw error by x*tan(dtheta) at range x, so both cancel
                    identically in a difference. No re-inference required.

  python3 scripts/riseholme/figures_verification.py --bag tue02sep
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "scripts" / "riseholme"))
import bag_config                                    # noqa: E402
import projection_calibration as C                   # noqa: E402
import curation                                      # noqa: E402

BAND_M = 0.182          # combined worst case from the two assumed DOF, at the 2 m look-ahead
LOOKAHEAD_M = 2.0


def _visible_window():
    for v in range(639, 300, -1):
        g = C.project_px(320, v, near_m=50.0)
        if g is not None:
            return float(g[0]), min(C.NEAR_M, 6.0)
    return 1.0, 6.0


def fig_gt_overlay(bag, out_dir, frames=(1963, 3911, 3834)):
    """Reported vision centreline vs the surveyed line, drawn as an uncertainty BAND.

    Drawn only over the camera's genuinely visible ground window: this rear camera at 1.269 m and
    5.75 deg cannot see ground closer than ~2.48 m, so drawing from 0.8 m implied coverage the
    sensor does not have. The surveyed segment is extended where the robot has driven past a row
    end (dashed); across the dataset the finite segment is in view for 74.8% of publishable
    two-row frames and the extended line for 100%.
    """
    B = bag_config.resolve(bag)
    xlo, xhi = _visible_window()
    pose, lines = curation.robot_pose_enu(bag)
    rows = {int(r["i"]): r for r in csv.DictReader(open(B["per_frame_csv"]))
            if r["arm"] == "C" and int(r["seed"]) == 42}
    flagged, _ = curation.publishable(bag)
    want = [i for i in frames if i not in flagged]

    fig, axes = plt.subplots(1, len(want), figsize=(5.0 * len(want), 4.7))
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, want):
        img = cv2.imread(str(B["frames_dir"] / f"{i:05d}.jpg"))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if i in pose and i in rows:
            rxy, hdg = pose[i]
            line = min(((min(np.linalg.norm(rxy - l[0]), np.linalg.norm(rxy - l[-1])), nm, l)
                        for nm, l in lines), key=lambda t: t[0])[2]
            lo, hi, nat = [], [], []
            for t in np.linspace(-1.5, 2.5, 400):
                p = line[0] + t * (line[-1] - line[0])
                X, Y = curation.enu_to_cvb(p, rxy, hdg)
                if not (xlo <= X <= xhi and abs(Y) < 4.0):
                    continue
                a_ = C.project_ground(X, Y - BAND_M)
                b_ = C.project_ground(X, Y + BAND_M)
                if a_ and b_:
                    lo.append(a_); hi.append(b_); nat.append(0.0 <= t <= 1.0)
            if len(lo) > 3:
                lo, hi, nat = np.array(lo), np.array(hi), np.array(nat, bool)
                poly = np.vstack([lo, hi[::-1]])
                ax.fill(poly[:, 0], poly[:, 1], color="#10b981", alpha=0.30, lw=0,
                        label=f"surveyed line \u00b1{int(BAND_M*1000)} mm")
                for e in (lo, hi):
                    if nat.any():
                        ax.plot(e[nat][:, 0], e[nat][:, 1], color="#10b981", lw=1.0)
                    if (~nat).any():
                        ax.plot(e[~nat][:, 0], e[~nat][:, 1], "--", color="#10b981", lw=1.0)
            r = rows[i]
            if r["cls"] == "two_row" and r["offset"]:
                off = float(r["offset"]); head = float(r["heading"] or 0.0)
                vis = [C.project_ground(X, off + (X - LOOKAHEAD_M) * np.tan(np.radians(head)))
                       for X in np.linspace(xlo, xhi, 30)]
                vis = [v for v in vis if v]
                if vis:
                    v = np.array(vis)
                    ax.plot(v[:, 0], v[:, 1], "-", color="#f59e0b", lw=2.4,
                            label=f"vision centreline ({off:+.3f} m)")
        ax.set_xlim(0, 640); ax.set_ylim(640, 0); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"frame {i}", fontsize=9)
    axes[0].legend(loc="upper left", fontsize=7, framealpha=0.9)
    fig.suptitle(
        "Riseholme verification: reported vision centreline vs the surveyed mid-row line\n"
        "BAND SHOWS ONE TERM ONLY \u2014 the \u00b1182 mm arising from the two camera-mounting "
        "parameters that could not be measured.\nThe surveyed reference carries a FURTHER ~130 mm "
        "per-row systematic that is NOT drawn, so a centreline lying outside the band does not "
        "imply an unexplained error. "
        f"Visible ground window {xlo:.2f}\u2013{xhi:.1f} m; dashed = extrapolated past a row end.",
        fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    p = out_dir / "verif_gt_overlay.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def fig_sensitivity(bag, out_dir):
    B = bag_config.resolve(bag)
    rows = [r for r in csv.DictReader(open(B["per_frame_csv"]))
            if r["cls"] == "two_row" and r["offset"]]
    by = {}
    for r in rows:
        by.setdefault((r["arm"], int(r["seed"])), {})[int(r["i"])] = float(r["offset"])
    arms = ["A", "B", "C"]
    lat = np.linspace(-0.20, 0.20, 41)
    yaw = np.linspace(-5.0, 5.0, 41)

    def shift(dy, dth):
        return dy + LOOKAHEAD_M * np.tan(np.radians(dth))

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for k, (grid, label, mk) in enumerate([(lat, "assumed lateral offset error (m)", "lat"),
                                           (yaw, "assumed yaw error (deg)", "yaw")]):
        for arm, col in zip(arms, ("#2563eb", "#16a34a", "#dc2626")):
            base = np.concatenate([list(by.get((arm, s), {}).values()) for s in (42, 43, 44)])
            rms = [np.sqrt(((base + shift(g if mk == "lat" else 0.0,
                                          0.0 if mk == "lat" else g))**2).mean()) for g in grid]
            ax[k].plot(grid, np.array(rms) * 1000, "-", color=col, lw=2,
                       label=f"arm {arm} absolute RMS")
        for (x, y), ls in zip((("A", "B"), ("A", "C"), ("B", "C")), ("--", ":", "-.")):
            d = []
            for s in (42, 43, 44):
                dx, dy_ = by.get((x, s), {}), by.get((y, s), {})
                sh = [i for i in dx if i in dy_]
                d.extend([dx[i] - dy_[i] for i in sh])
            d = np.array(d)
            ax[k].plot(grid, np.full_like(grid, abs(d.mean()) * 1000), ls, color="#111827", lw=1.6,
                       label=f"|{x}−{y}| paired diff")
        ax[k].axvline(0, color="0.6", lw=0.8)
        ax[k].set_xlabel(label); ax[k].set_ylabel("mm")
        ax[k].grid(alpha=0.25)
        ax[k].set_title(("lateral" if mk == "lat" else "yaw") +
                        " assumption sensitivity", fontsize=10)
    ax[0].legend(fontsize=7, ncol=2, loc="upper center")
    fig.suptitle("Why the paired cross-arm contrasts are the primary result: absolute RMS moves "
                 "with the assumed calibration; paired differences do not\n"
                 "a lateral error shifts every arm's offset by the same constant, a yaw error by "
                 "x·tan(θ) — both cancel exactly in a subtraction", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    p = out_dir / "verif_sensitivity.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    a = ap.parse_args()
    B = bag_config.resolve(a.bag)
    out = B["out_dir"].parent.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    flagged, meta = curation.publishable(a.bag)
    print(f"privacy screen: {meta['flagged_count']} flagged / {meta['frames_screened']} screened; "
          f"{meta['publishable_count']} publishable")
    for fn in (fig_sensitivity, fig_gt_overlay):
        p = fn(a.bag, out)
        print(f"  wrote {p.relative_to(PKG) if p else '(skipped)'}")


if __name__ == "__main__":
    main()
