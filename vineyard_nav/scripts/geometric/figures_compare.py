"""Cross-bag comparison figures for the multi-bag seasonal evaluation (D040/D046).

  python3 figures_compare.py                      # default --bags march april
  python3 figures_compare.py --bags march april may

Each figure reads the per-bag COMMITTED JSON artefacts and plots one series/group per bag, so the
same code extends to whatever bags actually completed — no per-N special-casing. These figures belong
to no single bag, so they live in results/geometric/comparison/figures/ (a cross-bag sibling of the
per-bag final/ dirs). Data-driven only (no model inference), so this module stays light — it does NOT
import figures.py (which pulls in torch/ultralytics). Style constants are mirrored from figures.py.

Four figures, all naturally list-of-N (grouped by bag):
  cmp_forest            F013 paired cross-arm GT-1 + GT-2 differences (indistinguishability across bags)
  cmp_tilt              F017 camera-vs-LiDAR row-tilt agreement per bag (sensor-common)
  cmp_nonrow_distribution  F020 spurious two_row rate per category per arm
  cmp_mitigation_closure   F022 state-gate rejection per category (arm-invariant -> one value/bag)
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
from bag_config import resolve

# ---- style mirrored from figures.py (kept in sync by hand; these are stable) ----
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "savefig.dpi": 300, "figure.dpi": 150, "savefig.bbox": "tight"})
ARM = {"A": "#4477aa", "B": "#ee6677", "C": "#228833"}
CATS = ("stationary", "turn", "transition")
RTK_MM = 38.0            # 3.8 cm RTK-GNSS localisation floor (Polvara 2024 §5.3), the GT-1 yardstick
OUT = PKG / "results" / "geometric" / "comparison" / "figures"


def _eval_dir(bag):
    return resolve(bag, "eligible")["out_dir"]


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p); plt.close(fig)
    print(f"  {name} -> {p.relative_to(PKG)}")
    return p


# ============================ cmp_forest (F013) ============================
def cmp_forest(bags):
    """Paired cross-arm GT-1 (mm) and GT-2 (deg) differences with 95% CIs, one row per (bag,pair).
    The headline: every GT-1 CI crosses zero on every bag; the GT-2 panel shows march's sub-noise-floor
    offset (CIs excluding zero) not reproducing on april (all cross zero)."""
    pairs = ["A-B", "A-C", "B-C"]
    data = {b: json.load(open(_eval_dir(b) / "paired_crossarm.json"))["across_seed"] for b in bags}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 1.4 + 0.5 * len(bags) * len(pairs)))

    def panel(ax, key, scale, unit, floor_label):
        rows, labels, colors = [], [], []
        y = 0
        yticks, yticklabels = [], []
        for b in bags:
            for p in pairs:
                d = data[b][p][key]
                mid = d["mean_diff"] * scale
                lo, hi = d["ci95"][0] * scale, d["ci95"][1] * scale
                excl = not (lo <= 0 <= hi)
                ax.plot([lo, hi], [y, y], "-", lw=2.2, color=("#c0392b" if excl else "#555"))
                ax.plot(mid, y, "o", ms=5, color=("#c0392b" if excl else "#555"))
                yticks.append(y); yticklabels.append(f"{b[:3]}  {p}")
                y += 1
            y += 0.6  # gap between bags
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_yticks(yticks); ax.set_yticklabels(yticklabels)
        ax.invert_yaxis(); ax.set_xlabel(f"paired difference ({unit})")
        ax.grid(axis="x", alpha=0.3)
        return floor_label

    panel(axL, "GT1", 1000.0, "mm", None)
    axL.axvspan(-RTK_MM, RTK_MM, color="#2166ac", alpha=0.06)
    axL.set_title("GT-1 lateral offset — indistinguishable on every bag\n(shaded = ±RTK floor 38 mm; dashed = 0)")
    panel(axR, "GT2", 1.0, "deg", None)
    axR.set_title("GT-2 heading — march's all-pair offset\ndoes not reproduce on any later bag (red = CI excludes 0)")
    fig.suptitle(f"Cross-arm paired differences · bags: {', '.join(bags)} · red = 95% CI excludes zero",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    return _save(fig, "cmp_forest.png")


# ============================ cmp_tilt (F017) ============================
def cmp_tilt(bags):
    """Camera vs LiDAR row-tilt on the mid-pass anchors per bag (post-D047). Shows both sensors nonzero,
    same sign, and agreeing — the sensor-common result — across bags at their (scene-dependent) magnitudes."""
    fig, ax = plt.subplots(figsize=(1.8 + 1.6 * len(bags), 4.2))
    x = np.arange(len(bags)); w = 0.32
    cam, lid, lsd = [], [], []
    for b in bags:
        d = json.load(open(_eval_dir(b) / "lidar_crosscheck.json"))
        cam.append(d["mean_cam_hdg"]); lid.append(d["mean_lidar_hdg"]); lsd.append(d["sd_lidar_hdg"])
    ax.bar(x - w / 2, cam, w, label="camera (line-fit)", color="#4477aa")
    ax.bar(x + w / 2, lid, w, yerr=lsd, capsize=4, label="LiDAR (trunk-band fit)", color="#d95f02")
    for xi, (c, l) in enumerate(zip(cam, lid)):
        ax.text(xi - w / 2, c + 0.05, f"{c:+.2f}", ha="center", va="bottom", fontsize=7)
        ax.text(xi + w / 2, l + 0.05, f"{l:+.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(bags)
    ax.set_ylabel("row heading vs base_link (deg)")
    ax.set_title("Camera vs LiDAR row tilt (mid-pass anchors)\nboth sensors nonzero, same sign, agreeing on every bag")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "cmp_tilt.png")


# ============================ cmp_nonrow_distribution (F020) ============================
def cmp_nonrow_distribution(bags):
    """Spurious two_row rate per non-in-row category, per arm, per bag. NOTE: this is the busiest
    comparison figure — it carries three dimensions (bag x category x arm). It is clean at 2-3 bags;
    beyond that it would need faceting (one small-multiple per bag). Not engineered for that here —
    revisit if many bags land. The arm dimension is essential (F020's stationary arm-divergence)."""
    data = {b: json.load(open(resolve(b, "non_in_row")["out_dir"] / "non_in_row_analysis.json"))
            ["F020_output_distribution"]["per_category"] for b in bags}
    fig, axes = plt.subplots(1, len(CATS), figsize=(3.2 * len(CATS), 4), sharey=True)
    x = np.arange(len(bags)); w = 0.26
    for ax, cat in zip(axes, CATS):
        for k, arm in enumerate("ABC"):
            vals = [data[b][cat][arm]["two_row"] for b in bags]
            ax.bar(x + (k - 1) * w, vals, w, label=f"arm {arm}", color=ARM[arm])
        ax.set_title(cat); ax.set_xticks(x); ax.set_xticklabels(bags)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("spurious two_row (% of category)")
    axes[-1].legend()
    fig.suptitle(f"Spurious two_row per non-in-row category · bags: {', '.join(bags)}\n"
                 "turn is the highest-rate category on every bag; every category falls under canopy", y=1.04, fontsize=10)
    fig.tight_layout()
    return _save(fig, "cmp_nonrow_distribution.png")


# ============================ cmp_mitigation_closure (F022) ============================
def cmp_mitigation_closure(bags):
    """State-gate rejection per non-in-row category, per bag. F022 is arm-invariant per category, so
    one value per (bag, category) (arm A shown; arms coincide). Shows the transition-category collapse
    (~96% march -> ~70% april) that drives the headline overall-closure drop."""
    data = {b: json.load(open(resolve(b, "eligible")["out_dir"].parent / "mitigation_evaluation"
            / "mitigation_analysis.json"))["F022_F023_causal"]["non_in_row"]["per_category"] for b in bags}
    fig, ax = plt.subplots(figsize=(2.2 + 1.7 * len(bags), 4.2))
    x = np.arange(len(CATS)); w = 0.8 / len(bags)
    palette = ["#2166ac", "#d95f02", "#1a9e4b", "#8844aa", "#88419d", "#a6761d"]
    for j, b in enumerate(bags):
        vals = [data[b][cat]["A"]["f022_%"] for cat in CATS]
        ax.bar(x + (j - (len(bags) - 1) / 2) * w, vals, w, label=b, color=palette[j % len(palette)])
        for xi, v in enumerate(vals):
            ax.text(x[xi] + (j - (len(bags) - 1) / 2) * w, v + 0.6, f"{v:.0f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(CATS); ax.set_ylim(0, 108)
    ax.set_ylabel("state-gate rejection (% of spurious two_row)")
    ax.set_title("State-gate closure per category (arm-invariant)\nstationary holds at 100% on every bag; transition is weakest and collapses on june")
    ax.legend(title="bag"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save(fig, "cmp_mitigation_closure.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", default=["march", "april"])
    ap.add_argument("--only", default=None, help="one of: forest tilt nonrow mitigation")
    a = ap.parse_args()
    print(f"[figures_compare] bags = {a.bags}")
    figs = {"forest": cmp_forest, "tilt": cmp_tilt,
            "nonrow": cmp_nonrow_distribution, "mitigation": cmp_mitigation_closure}
    for name, fn in figs.items():
        if a.only and a.only != name:
            continue
        fn(a.bags)
    print("done.")


if __name__ == "__main__":
    main()
