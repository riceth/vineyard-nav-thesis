"""Riseholme report figures: inputs, per-arm detections, row fits, success/failure, tables.

Complements figures_verification.py (the two falsification figures) and the diagnostics overlays.
Every frame is drawn from curation.select(), which intersects candidates with the privacy
allow-list, so a flagged frame cannot reach a figure. Base points and row fits come from the
pipeline's OWN estimate() and base extractors via curation.load_arm, never a re-implementation,
so a figure cannot show geometry the metrics did not compute.

No decision or finding identifiers appear in rendered text: a reader of the report has no access
to those files. They remain in docstrings and comments, which are not rendered.

  python3 scripts/riseholme/figures.py --bag tue02sep
"""
import argparse
import collections
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
import bag_config                       # noqa: E402
import projection_calibration as C      # noqa: E402
import curation                         # noqa: E402

ARMS = [("A", "U-Net binary"), ("B", "YOLO binary"), ("C", "YOLO multiclass")]
COL = {"A": "#2563eb", "B": "#16a34a", "C": "#dc2626"}
LOOK = 2.0


def _vis_window():
    for v in range(639, 300, -1):
        g = C.project_px(320, v, near_m=50.0)
        if g is not None:
            return float(g[0]), min(C.NEAR_M, 6.0)
    return 1.0, 6.0


def _load_rows(B, arm="C", seed=42):
    return {int(r["i"]): r for r in csv.DictReader(open(B["per_frame_csv"]))
            if r["arm"] == arm and int(r["seed"]) == seed}


# ---------------------------------------------------------------- 1. representative inputs
def fig_inputs(bag, out, n=6):
    B = bag_config.resolve(bag)
    man = json.load(open(B["manifest"]))
    bycor = collections.defaultdict(list)
    flagged, _ = curation.publishable(bag)
    for f in man["frames"]:
        if f["eligible"] and f["i"] not in flagged:
            bycor[f["corridor"]].append(f["i"])
    picks = [v[len(v) // 2] for k, v in sorted(bycor.items())][:n]
    fig, axes = plt.subplots(2, (len(picks) + 1) // 2, figsize=(3.4 * ((len(picks) + 1) // 2), 7.0))
    for ax, i in zip(axes.ravel(), picks):
        ax.imshow(cv2.cvtColor(cv2.imread(str(B["frames_dir"] / f"{i:05d}.jpg")), cv2.COLOR_BGR2RGB))
        ax.set_title(f"frame {i}", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    fig.suptitle("Riseholme input frames, one per corridor traversed\n"
                 "rear-facing RealSense D435I, 1280×720 resized to 640×640 as in training",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = out / "rh_fig1_inputs.png"; fig.savefig(p, dpi=140); plt.close(fig); return p


# ------------------------------------------------- 2 & 3. per-arm detections, base points, fits
def fig_arms_and_fits(bag, out, n=3):
    B = bag_config.resolve(bag)
    rows = _load_rows(B, "C")
    cand = [i for i in curation.select(bag, n=40, cls="two_row", arm="C") if i in rows][:n]
    xlo, xhi = _vis_window()
    est = curation.estimate_fn(bag)
    fig, axes = plt.subplots(len(ARMS), len(cand), figsize=(4.4 * len(cand), 4.1 * len(ARMS)))
    axes = np.atleast_2d(axes)
    for r, (arm, name) in enumerate(ARMS):
        front, kind, _ = curation.load_arm(arm, 42, bag)
        for c, i in enumerate(cand):
            ax = axes[r, c]
            img = cv2.imread(str(B["frames_dir"] / f"{i:05d}.jpg"))
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            base = front(img)
            e = est(base)
            for (uc, v) in base:
                ax.plot(uc, v, "o", ms=5, mfc="none", mec=COL[arm], mew=1.6)
            if e.get("cls") == "two_row":
                off, hd = e["offset"], e["heading"]
                pts = [C.project_ground(X, off + (X - LOOK) * np.tan(np.radians(hd)))
                       for X in np.linspace(xlo, xhi, 30)]
                pts = np.array([q for q in pts if q])
                if len(pts):
                    ax.plot(pts[:, 0], pts[:, 1], "-", color="#f59e0b", lw=2.4)
                tag = f"two-row · offset {off:+.3f} m · heading {hd:+.2f}°"
            else:
                tag = f"{e.get('cls','none')} · {len(base)} base points"
            ax.set_xlim(0, 640); ax.set_ylim(640, 0); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(tag, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{arm} — {name}", fontsize=10, color=COL[arm])
        del front
    for c, i in enumerate(cand):
        axes[0, c].text(0.5, 1.22, f"frame {i}", transform=axes[0, c].transAxes,
                        ha="center", fontsize=11)
    fig.suptitle("Detections, base points and fitted centreline for all three arms on the same "
                 "frames\ncircles are the base points each arm supplies; orange is the fitted "
                 "mid-row centreline", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = out / "rh_fig2_arms_basepoints_fits.png"; fig.savefig(p, dpi=140); plt.close(fig); return p


# ---------------------------------------------------------- 5. success and failure examples
def fig_success_failure(bag, out, n=3):
    B = bag_config.resolve(bag)
    xlo, xhi = _vis_window()
    est = curation.estimate_fn(bag)
    front, _, _ = curation.load_arm("C", 42, bag)
    groups = [("two_row", "SUCCESS — two rows fitted"),
              ("single_row", "PARTIAL — one row only, centreline inferred"),
              ("none", "FAILURE — no row fitted, pipeline abstains")]
    fig, axes = plt.subplots(len(groups), n, figsize=(4.4 * n, 4.1 * len(groups)))
    axes = np.atleast_2d(axes)
    for r, (cls, label) in enumerate(groups):
        picks = curation.select(bag, n=n, cls=cls, arm="C")
        for c in range(n):
            ax = axes[r, c]
            if c >= len(picks):
                ax.axis("off"); continue
            i = picks[c]
            img = cv2.imread(str(B["frames_dir"] / f"{i:05d}.jpg"))
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            base = front(img); e = est(base)
            for (uc, v) in base:
                ax.plot(uc, v, "o", ms=4, mfc="none", mec="#dc2626", mew=1.4)
            if e.get("cls") == "two_row":
                pts = [C.project_ground(X, e["offset"] + (X - LOOK) * np.tan(np.radians(e["heading"])))
                       for X in np.linspace(xlo, xhi, 30)]
                pts = np.array([q for q in pts if q])
                if len(pts):
                    ax.plot(pts[:, 0], pts[:, 1], "-", color="#f59e0b", lw=2.4)
            ax.set_xlim(0, 640); ax.set_ylim(640, 0); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"frame {i} · {len(base)} base points", fontsize=8)
            if c == 0:
                ax.set_ylabel(label, fontsize=9)
    fig.suptitle("Successful, partial and failed frames (multiclass arm)\n"
                 "the pipeline abstains rather than guessing when too few base points survive",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = out / "rh_fig3_success_failure.png"; fig.savefig(p, dpi=140); plt.close(fig); return p


# ------------------------------------------------------------------- 6. result tables + plots
def fig_tables(bag, out):
    B = bag_config.resolve(bag)
    rep = json.load(open(B["line_fit_report"]))
    paired = json.load(open(B["paired"]))
    allrows = list(csv.DictReader(open(B["per_frame_csv"])))
    cov, rej = {}, {}
    for arm, _ in ARMS:
        a = [x for x in allrows if x["arm"] == arm]
        cc = collections.Counter(x["cls"] for x in a); n = len(a)
        cov[arm] = 100 * cc["two_row"] / n
        rej[arm] = {"single": 100 * cc["single_row"] / n, "none": 100 * cc["none"] / n,
                    "slope_mismatch": 100 * sum("slope_mismatch" in x["flags"] for x in a) / n}

    fig = plt.figure(figsize=(15.5, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1])

    ax = fig.add_subplot(gs[0, :]); ax.axis("off")
    hdr = ["arm", "coverage\n(two-row %)", "single-row %", "no-fit %", "slope-mismatch %",
           "n frames", "lateral RMS (m)", "heading RMS (deg)"]
    body = []
    for arm, name in ARMS:
        ci = rep["per_arm_ci"][arm]
        body.append([f"{arm} — {name}", f"{cov[arm]:.1f}", f"{rej[arm]['single']:.1f}",
                     f"{rej[arm]['none']:.1f}", f"{rej[arm]['slope_mismatch']:.1f}",
                     f"{ci['n']}", f"{ci['gt1_rms']:.3f}", f"{ci['gt2_rms']:.2f}"])
    t = ax.table(cellText=body, colLabels=hdr, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.9)
    for k in range(len(hdr)):
        t[0, k].set_facecolor("#e5e7eb")
    ax.set_title("Per-arm results. Coverage and rejection rates are exact; the two RMS columns are "
                 "CAVEATED — they carry unmeasured camera-mounting terms and a reference offset, "
                 "and are not precise accuracy figures.", fontsize=10, pad=14)

    ax = fig.add_subplot(gs[1, 0])
    x = np.arange(3); w = 0.26
    ax.bar(x - w, [cov[a] for a, _ in ARMS], w, label="two-row (usable)", color="#16a34a")
    ax.bar(x, [rej[a]["single"] for a, _ in ARMS], w, label="single-row", color="#f59e0b")
    ax.bar(x + w, [rej[a]["none"] for a, _ in ARMS], w, label="no fit", color="#dc2626")
    ax.set_xticks(x); ax.set_xticklabels([a for a, _ in ARMS]); ax.set_ylabel("% of frames")
    ax.set_title("Coverage and failure rate", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")

    ax = fig.add_subplot(gs[1, 1])
    for k, (arm, _) in enumerate(ARMS):
        ci = rep["per_arm_ci"][arm]
        ax.errorbar(k, ci["gt1_rms"] * 1000,
                    yerr=[[(ci["gt1_rms"] - ci["gt1_ci"][0]) * 1000],
                          [(ci["gt1_ci"][1] - ci["gt1_rms"]) * 1000]],
                    fmt="o", ms=9, color=COL[arm], capsize=5)
    ax.axhspan(0, 182, color="#94a3b8", alpha=.30,
               label="unmeasured mounting terms (±182 mm)")
    ax.set_xticks(range(3)); ax.set_xticklabels([a for a, _ in ARMS])
    ax.set_ylabel("lateral RMS (mm)")
    ax.set_title("Absolute accuracy sits within the\nunmeasured calibration budget", fontsize=10)
    ax.legend(fontsize=7); ax.grid(alpha=.25, axis="y")

    ax = fig.add_subplot(gs[1, 2])
    acr = paired["across_seed"]; sc = paired["sign_consistency"]
    names = ["A-B", "A-C", "B-C"]
    for k, nm in enumerate(names):
        d = acr[nm]["GT1"]
        ok = sc[nm]["GT1_consistent"]
        ax.errorbar(k, d["mean_diff"] * 1000,
                    yerr=[[abs(d["mean_diff"] - d["ci95"][0]) * 1000],
                          [abs(d["ci95"][1] - d["mean_diff"]) * 1000]],
                    fmt="o" if ok else "x", ms=10, capsize=5,
                    color="#111827" if ok else "#9ca3af")
    ax.axhline(0, color="0.5", lw=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(
        [f"{n}\n{'consistent' if sc[n]['GT1_consistent'] else 'sign flips'}" for n in names],
        fontsize=8)
    ax.set_ylabel("paired difference (mm)")
    ax.set_title("Paired cross-arm differences\nintervals shown are unreliable at this sample density",
                 fontsize=10)
    ax.grid(alpha=.25, axis="y")

    fig.suptitle("Riseholme results summary", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out / "rh_fig4_results.png"; fig.savefig(p, dpi=140); plt.close(fig); return p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bag", required=True)
    a = ap.parse_args()
    B = bag_config.resolve(a.bag)
    out = B["out_dir"].parent.parent / "figures"; out.mkdir(parents=True, exist_ok=True)
    _, meta = curation.publishable(a.bag)
    print(f"privacy: {meta['flagged_count']} flagged / {meta['frames_screened']}; "
          f"{meta['publishable_count']} publishable")
    for fn in (fig_inputs, fig_tables, fig_success_failure, fig_arms_and_fits):
        try:
            print(f"  wrote {fn(a.bag, out).relative_to(PKG)}")
        except Exception as e:
            print(f"  FAILED {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
