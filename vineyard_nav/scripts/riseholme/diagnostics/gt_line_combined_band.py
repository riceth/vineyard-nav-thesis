"""Combined-uncertainty overlay: is a vision-reference disagreement REAL, or already explained?

The single-term figure (figures_verification.py) shows only the assumed-extrinsics band and exists
to keep error sources separable for D059. This one asks the sharper question: could the observed
disagreement be accounted for ENTIRELY by known, documented, non-vision error?

  inner band  +/-182 mm   assumed extrinsics only            (D056)
  outer band  +/-312 mm   assumed extrinsics + per-row ref   (D056 + F031)

COMBINATION METHOD: DIRECT SUM (182 + 130 = 312 mm), not root-sum-square (224 mm).

Justification. RSS is the right combination for independent RANDOM errors, where the two are
unlikely to reach their extremes together. Neither term here is random for a given frame:
  * the calibration term is ONE fixed unknown -- a single mounting, mis-assumed by a definite
    amount with a definite sign, identical in every frame of the dataset;
  * the per-row term is a systematic constant WITHIN a row (its 0.128 m is the spread of per-row
    means, i.e. the between-row component after the common calibration bias has already cancelled).
Two systematic biases of unknown sign can align, so their worst case is the sum.

The direction of the choice also matters. The test is "does the disagreement exceed everything we
already know about?" A conservative (wider) band yields fewer positives, but each surviving one is
a strong claim of genuine vision error. Using RSS would risk labelling real vision error as
"explained", which is the more damaging mistake here.

  python3 scripts/riseholme/diagnostics/gt_line_combined_band.py --bag tue02sep
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

CAL_M = 0.182          # D056: assumed lateral (70 mm) + yaw at 2 m look-ahead (112 mm)
REF_M = 0.130          # F031: between-row spread of the reference residual
SUM_M = CAL_M + REF_M  # 0.312 -- see the module docstring for why sum and not RSS
RSS_M = float(np.hypot(CAL_M, REF_M))
LOOK = 2.0
FRAMES = (1963, 3911, 3834)


def visible():
    for v in range(639, 300, -1):
        g = C.project_px(320, v, near_m=50.0)
        if g is not None:
            return float(g[0]), min(C.NEAR_M, 6.0)
    return 1.0, 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="tue02sep")
    a = ap.parse_args()
    B = bag_config.resolve(a.bag)
    xlo, xhi = visible()
    pose, lines = curation.robot_pose_enu(a.bag)
    rows = {int(r["i"]): r for r in csv.DictReader(open(B["per_frame_csv"]))
            if r["arm"] == "C" and int(r["seed"]) == 42}
    flagged, _ = curation.publishable(a.bag)

    print(f"combination: SUM {SUM_M*1000:.0f} mm  (RSS would be {RSS_M*1000:.0f} mm -- not used, "
          f"see docstring: both terms are systematic, not random)")

    # ---- decisive statistic over the whole dataset ----
    inside_cal = inside_sum = tot = 0
    for i, (rxy, hdg) in pose.items():
        if i in flagged or i not in rows or rows[i]["cls"] != "two_row" or not rows[i]["offset"]:
            continue
        line = min(((min(np.linalg.norm(rxy - l[0]), np.linalg.norm(rxy - l[-1])), nm, l)
                    for nm, l in lines), key=lambda t: t[0])[2]
        ab = line[-1] - line[0]
        n = np.array([-ab[1], ab[0]]) / np.linalg.norm(ab)
        so = float((rxy - line[0]) @ n)
        with_line = float(np.dot(np.array([np.sin(np.radians(hdg)), np.cos(np.radians(hdg))]), ab)) > 0
        true_off = -so if with_line else so
        err = abs(float(rows[i]["offset"]) - true_off)
        tot += 1
        inside_cal += err <= CAL_M
        inside_sum += err <= SUM_M
    print(f"over {tot} publishable two-row frames (arm C, seed 42):")
    print(f"  |disagreement| within calibration band alone (+/-{CAL_M*1000:.0f} mm): "
          f"{inside_cal} ({100*inside_cal/max(tot,1):.1f}%)")
    print(f"  |disagreement| within COMBINED band          (+/-{SUM_M*1000:.0f} mm): "
          f"{inside_sum} ({100*inside_sum/max(tot,1):.1f}%)")
    print(f"  -> {100*(tot-inside_sum)/max(tot,1):.1f}% of frames disagree by MORE than every known "
          f"non-vision source combined; those are the only frames that evidence real vision error.")

    fig, axes = plt.subplots(1, len(FRAMES), figsize=(5.6 * len(FRAMES), 5.8))
    for ax, i in zip(np.atleast_1d(axes), FRAMES):
        img = cv2.cvtColor(cv2.imread(str(B["frames_dir"] / f"{i:05d}.jpg")), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        rxy, hdg = pose[i]
        line = min(((min(np.linalg.norm(rxy - l[0]), np.linalg.norm(rxy - l[-1])), nm, l)
                    for nm, l in lines), key=lambda t: t[0])[2]
        band = {CAL_M: [[], []], SUM_M: [[], []]}
        cen = []
        for t in np.linspace(-1.5, 2.5, 500):
            p = line[0] + t * (line[-1] - line[0])
            X, Y = curation.enu_to_cvb(p, rxy, hdg)
            if not (xlo <= X <= xhi and abs(Y) < 4.0):
                continue
            c = C.project_ground(X, Y)
            if c:
                cen.append(c)
            for w in (CAL_M, SUM_M):
                lo_, hi_ = C.project_ground(X, Y - w), C.project_ground(X, Y + w)
                if lo_ and hi_:
                    band[w][0].append(lo_); band[w][1].append(hi_)
        for w, col, alp, lab in ((SUM_M, "#f97316", 0.26,
                                  f"combined ±{SUM_M*1000:.0f} mm (mounting + reference)"),
                                 (CAL_M, "#10b981", 0.42,
                                  f"camera mounting only ±{CAL_M*1000:.0f} mm")):
            lo_, hi_ = np.array(band[w][0]), np.array(band[w][1])
            if len(lo_) > 3:
                ax.fill(np.vstack([lo_, hi_[::-1]])[:, 0], np.vstack([lo_, hi_[::-1]])[:, 1],
                        color=col, alpha=alp, lw=0, label=lab)
        if cen:
            cen = np.array(cen)
            ax.plot(cen[:, 0], cen[:, 1], "-", color="#065f46", lw=1.6, label="surveyed line")
        r = rows[i]
        off, hd = float(r["offset"]), float(r["heading"] or 0.0)
        vis = np.array([v for v in (C.project_ground(X, off + (X - LOOK) * np.tan(np.radians(hd)))
                                    for X in np.linspace(xlo, xhi, 40)) if v])
        ax.plot(vis[:, 0], vis[:, 1], "-", color="#1d4ed8", lw=3.0,
                label=f"vision centreline ({off:+.3f} m)")
        allpx = np.vstack([cen, vis]); cx, cy = allpx[:, 0].mean(), allpx[:, 1].mean(); h = 185
        ax.set_xlim(max(0, cx - h), min(640, cx + h)); ax.set_ylim(min(640, cy + h), max(0, cy - h))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"frame {i}", fontsize=12)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    fig.suptitle(
        "Is the vision–reference disagreement REAL, or already explained by known error?\n"
        f"outer ±{SUM_M*1000:.0f} mm = unmeasured camera-mounting parameters + the surveyed "
        f"reference's own per-row systematic, combined by DIRECT SUM (both are fixed biases, not "
        f"random noise)\n"
        "a centreline inside the outer band is fully accounted for by non-vision sources; only "
        "outside it is evidence of genuine vision error", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = B["out_dir"].parent.parent / "diagnostics" / "gt_line_combined_band.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out.relative_to(PKG)}")


if __name__ == "__main__":
    main()
