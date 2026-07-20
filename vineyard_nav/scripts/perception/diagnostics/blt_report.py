#!/usr/bin/env python3
"""Sections 2-5: histograms, viability summary, seasonal breakdown, sample overlays.

Aggregate / viability / seasonal computed on UNIQUE frames only (is_duplicate==False).
"""
import os, json, re
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cv2

from blt_analysis import (load_all, CLASSES, CLASS_NAMES, TRUNK_ID, POLE_ID,
                          instance_mask, OUT)

HIST_DIR = os.path.join(OUT, "aggregate_histograms")
SAMP_DIR = os.path.join(OUT, "sample_visualisations")
for d in (HIST_DIR, SAMP_DIR):
    os.makedirs(d, exist_ok=True)

W = 640  # all images 640x640
NBINS = 64

# Distinct BGR colours per class for overlays
CLASS_COLOR = {
    "building": (180, 180, 180),
    "pipe":     (255, 200, 0),    # cyan-ish (BGR)
    "pole":     (0, 140, 255),    # orange
    "robot":    (200, 0, 200),    # magenta
    "trunk":    (0, 0, 255),      # red
    "vehicle":  (0, 255, 255),    # yellow
}


def load_unique_with_anns():
    """Records (with anns) for unique frames only, in CSV order."""
    df = pd.read_pickle(os.path.join(OUT, "_all_stats.pkl"))
    records = load_all()
    by_fn = {r["file_name"]: r for r in records}
    uniq = df[~df.is_duplicate].copy()
    recs = [by_fn[fn] for fn in uniq.file_name]
    return uniq, recs


# ---------------------------------------------------------------- section 2
def aggregate_histograms(recs):
    """Pixel-coordinate histograms accumulated incrementally + per-side counts."""
    trunk_x = np.zeros(NBINS); pole_x = np.zeros(NBINS); trunk_y = np.zeros(NBINS)
    edges = np.linspace(0, W, NBINS + 1)
    for rec in recs:
        w, h = rec["width"], rec["height"]
        tmask = np.zeros((h, w), np.uint8); pmask = np.zeros((h, w), np.uint8)
        for a in rec["anns"]:
            if a["category_id"] == TRUNK_ID:
                np.maximum(tmask, instance_mask(a, w, h), out=tmask)
            elif a["category_id"] == POLE_ID:
                np.maximum(pmask, instance_mask(a, w, h), out=pmask)
        tys, txs = np.where(tmask > 0)
        pys, pxs = np.where(pmask > 0)
        if len(txs):
            trunk_x += np.histogram(txs, bins=edges)[0]
            trunk_y += np.histogram(tys, bins=edges)[0]
        if len(pxs):
            pole_x += np.histogram(pxs, bins=edges)[0]
    centers = (edges[:-1] + edges[1:]) / 2

    def bar(data, title, xlabel, fname, vline=True):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(centers, data, width=W / NBINS, color="#3b6", edgecolor="none")
        if vline:
            ax.axvline(W / 2, color="k", ls="--", lw=1, label="image centre (x=320)")
            ax.legend()
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("pixel count")
        ax.set_xlim(0, W)
        fig.tight_layout(); fig.savefig(os.path.join(HIST_DIR, fname), dpi=90)
        plt.close(fig)

    bar(trunk_x, "Trunk pixel x-distribution (unique frames)", "x (px)", "trunk_pixel_x.png")
    bar(pole_x, "Pole pixel x-distribution (unique frames)", "x (px)", "pole_pixel_x.png")
    bar(trunk_y, "Trunk pixel y-distribution (unique frames)", "y (px, 0=top)", "trunk_pixel_y.png")
    return trunk_x, pole_x, trunk_y


def side_count_histograms(uniq):
    """Distribution of per-side instance counts across unique frames."""
    specs = [
        ("trunk_left_instances", "Trunks per LEFT side", "trunks_per_left.png"),
        ("trunk_right_instances", "Trunks per RIGHT side", "trunks_per_right.png"),
        ("pole_left_instances", "Poles per LEFT side", "poles_per_left.png"),
        ("pole_right_instances", "Poles per RIGHT side", "poles_per_right.png"),
    ]
    for col, title, fname in specs:
        vals = uniq[col].values
        mx = int(vals.max())
        fig, ax = plt.subplots(figsize=(7, 4))
        bins = np.arange(0, mx + 2) - 0.5
        ax.hist(vals, bins=bins, color="#48c", edgecolor="white")
        ax.axvline(2.5, color="r", ls="--", lw=1.2, label="RANSAC threshold (>=3)")
        ax.set_title(f"{title} (n={len(vals)} frames)")
        ax.set_xlabel("instances on side"); ax.set_ylabel("# frames")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(HIST_DIR, fname), dpi=90); plt.close(fig)


# ---------------------------------------------------------------- section 3
def viability_summary(uniq):
    n = len(uniq)
    L = lambda c: uniq[c].values

    def pct(mask):
        return 100.0 * mask.sum() / n

    tl, tr = L("trunk_left_instances"), L("trunk_right_instances")
    pl, pr = L("pole_left_instances"), L("pole_right_instances")
    cl, cr = tl + pl, tr + pr  # combined trunk+pole per side

    trunk_both = (tl >= 3) & (tr >= 3)
    pole_both = (pl >= 3) & (pr >= 3)
    comb_both = (cl >= 3) & (cr >= 3)

    # For trunk-fail frames, do poles supplement?
    trunk_fail = ~trunk_both
    nfail = trunk_fail.sum()
    rescued = (trunk_fail & comb_both).sum()  # trunk-fail but combined passes both sides
    # among trunk-fail frames, pole sparsity
    pole_inst_fail = (pl + pr)[trunk_fail]

    lines = []
    A = lines.append
    A("SemanticBLT per-side RANSAC viability summary")
    A("=" * 60)
    A(f"Basis: {n} UNIQUE frames (train 6x augmentation removed; valid+test as-is).")
    A("Threshold: >=3 structural instances on a side ~ minimum for stable RANSAC line fit.")
    A("'Side' = instance centroid x relative to image centre (x=320 of 640).")
    A("")
    A("TRUNKS")
    A(f"  >=3 trunks on LEFT side : {pct(tl>=3):5.1f}%  ({(tl>=3).sum()}/{n})")
    A(f"  >=3 trunks on RIGHT side: {pct(tr>=3):5.1f}%  ({(tr>=3).sum()}/{n})")
    A(f"  >=3 on BOTH sides       : {pct(trunk_both):5.1f}%  ({trunk_both.sum()}/{n})")
    A(f"  mean trunks/frame: {uniq.trunk_instances.mean():.1f}   frames with 0 trunks: {(uniq.trunk_instances==0).sum()}")
    A("")
    A("POLES")
    A(f"  >=3 poles on LEFT side  : {pct(pl>=3):5.1f}%  ({(pl>=3).sum()}/{n})")
    A(f"  >=3 poles on RIGHT side : {pct(pr>=3):5.1f}%  ({(pr>=3).sum()}/{n})")
    A(f"  >=3 on BOTH sides       : {pct(pole_both):5.1f}%  ({pole_both.sum()}/{n})")
    A(f"  mean poles/frame: {uniq.pole_instances.mean():.1f}   frames with 0 poles: {(uniq.pole_instances==0).sum()}")
    A("")
    A("TRUNKS + POLES COMBINED (per side)")
    A(f"  >=3 combined on LEFT    : {pct(cl>=3):5.1f}%  ({(cl>=3).sum()}/{n})")
    A(f"  >=3 combined on RIGHT   : {pct(cr>=3):5.1f}%  ({(cr>=3).sum()}/{n})")
    A(f"  >=3 on BOTH sides       : {pct(comb_both):5.1f}%  ({comb_both.sum()}/{n})")
    A("")
    A("DO POLES SUPPLEMENT TRUNK-SPARSE FRAMES?")
    A(f"  Frames failing trunk-both-sides test: {nfail}/{n} ({100.0*nfail/n:.1f}%)")
    A(f"  Of those, combined trunk+pole passes both sides: {rescued} ({100.0*rescued/max(nfail,1):.1f}% of failures rescued)")
    A(f"  In trunk-fail frames, median poles present: {np.median(pole_inst_fail):.0f}, mean: {pole_inst_fail.mean():.1f}")
    A(f"  Trunk-fail frames that ALSO have <3 total poles: {((pl+pr<3)&trunk_fail).sum()} "
      f"({100.0*((pl+pr<3)&trunk_fail).sum()/max(nfail,1):.1f}% of failures)")
    A("")
    A("INTERPRETATION")
    if pct(trunk_both) >= 60:
        A("  Trunks alone clear the per-side threshold on a majority of frames.")
    else:
        A("  Trunks alone are insufficient on a large share of frames per side.")
    if pct(comb_both) - pct(trunk_both) >= 8:
        A("  Adding poles meaningfully increases per-side viability -> combined fit recommended.")
    else:
        A("  Poles add only marginal per-side coverage over trunks alone.")
    txt = "\n".join(lines)
    open(os.path.join(OUT, "viability_summary.txt"), "w").write(txt + "\n")
    print(txt)
    return dict(trunk_both=pct(trunk_both), comb_both=pct(comb_both),
               pole_both=pct(pole_both), n=n, rescued=rescued, nfail=int(nfail))


# ---------------------------------------------------------------- overlay render
def render_overlay(rec, title=None):
    img = cv2.imread(rec["path"])
    if img is None:
        img = np.zeros((rec["height"], rec["width"], 3), np.uint8)
    h, w = img.shape[:2]
    overlay = img.copy()
    present = set()
    for a in rec["anns"]:
        cid = a["category_id"]
        if cid not in CLASSES:
            continue
        name = CLASSES[cid]; present.add(name)
        col = CLASS_COLOR[name]
        for seg in a["segmentation"]:
            if len(seg) >= 6:
                pts = np.array(seg, np.float64).reshape(-1, 2).astype(np.int32)
                cv2.fillPoly(overlay, [pts], col)
                cv2.polylines(img, [pts], True, col, 1)
    out = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    # draw centre line
    cv2.line(out, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    return out, present


def save_overlay_png(rec, fname, title):
    out, present = render_overlay(rec)
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    handles = [Patch(facecolor=np.array(CLASS_COLOR[n][::-1]) / 255, label=n)
               for n in CLASS_NAMES if n in present]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7)
    fig.tight_layout()
    path = os.path.join(SAMP_DIR if fname.startswith("sample") else OUT, fname)
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    # keep under 500KB
    if os.path.getsize(path) > 500_000:
        fig2 = None
    return path


# ---------------------------------------------------------------- section 4
def seasonal_summary(uniq, recs):
    by_fn = {r["file_name"]: r for r in recs}
    months_order = ["march", "april", "may", "unknown"]
    md = ["# Seasonal Breakdown (unique frames)\n",
          "Note: only march/april/may are encoded in filenames; `unknown` = unprefixed "
          "`color_image_NNNN` files (likely the June batch, dataset is named *2024-june*, "
          "but month is not verifiable). No June–September prefixes exist in the data.\n"]
    rep_dir = os.path.join(OUT, "seasonal_representatives")
    os.makedirs(rep_dir, exist_ok=True)

    md.append("| month | n frames | mean trunks | mean poles | mean pipe | mean building | trunk both-sides >=3 |")
    md.append("|---|---|---|---|---|---|---|")
    rows = []
    for mo in months_order:
        sub = uniq[uniq.month == mo]
        if not len(sub):
            continue
        both = ((sub.trunk_left_instances >= 3) & (sub.trunk_right_instances >= 3)).mean() * 100
        md.append(f"| {mo} | {len(sub)} | {sub.trunk_instances.mean():.1f} | "
                  f"{sub.pole_instances.mean():.1f} | {sub.pipe_instances.mean():.1f} | "
                  f"{sub.building_instances.mean():.1f} | {both:.0f}% |")
        rows.append((mo, len(sub), both))

    md.append("\n## Per-class mean instance counts by month\n")
    md.append("| month | " + " | ".join(CLASS_NAMES) + " |")
    md.append("|---|" + "---|" * len(CLASS_NAMES))
    for mo in months_order:
        sub = uniq[uniq.month == mo]
        if not len(sub):
            continue
        cells = " | ".join(f"{sub[f'{c}_instances'].mean():.1f}" for c in CLASS_NAMES)
        md.append(f"| {mo} | {cells} |")

    # representative image per month: pick frame closest to that month's median trunk count
    md.append("\n## Representative frames\n")
    for mo in months_order:
        sub = uniq[uniq.month == mo]
        if not len(sub):
            continue
        med = sub.trunk_instances.median()
        idx = (sub.trunk_instances - med).abs().idxmin()
        fn = sub.loc[idx, "file_name"]
        rec = by_fn[fn]
        fname = f"seasonal_representatives/rep_{mo}.png"
        out, present = render_overlay(rec)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{mo}: {int(sub.loc[idx,'trunk_instances'])} trunks, "
                     f"{int(sub.loc[idx,'pole_instances'])} poles", fontsize=10)
        ax.axis("off")
        handles = [Patch(facecolor=np.array(CLASS_COLOR[n][::-1]) / 255, label=n)
                   for n in CLASS_NAMES if n in present]
        ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, fname), dpi=80, bbox_inches="tight")
        plt.close(fig)
        md.append(f"### {mo}\n![{mo}]({fname})\n")

    open(os.path.join(OUT, "seasonal_summary.md"), "w").write("\n".join(md) + "\n")
    print("Wrote seasonal_summary.md")
    return rows


# ---------------------------------------------------------------- section 5
def sample_visualisations(uniq, recs):
    by_fn = {r["file_name"]: r for r in recs}
    u = uniq.copy()
    u["trunk_min_side"] = u[["trunk_left_instances", "trunk_right_instances"]].min(axis=1)
    chosen = []  # (file_name, out_fname, title)
    used = set()

    def pick(cond, fname, title, sort_col=None, asc=True):
        sub = u[cond & ~u.file_name.isin(used)]
        if not len(sub):
            return
        if sort_col:
            sub = sub.sort_values(sort_col, ascending=asc)
        fn = sub.iloc[0].file_name
        used.add(fn); chosen.append((fn, fname, title))

    # Clear/dense frames, spanning months
    pick((u.month == "march") & (u.trunk_min_side >= 4), "sample_march_dense.png",
         "March — dense, trunks both sides", "trunk_instances", False)
    pick((u.month == "may") & (u.trunk_min_side >= 3), "sample_may_dense.png",
         "May — multiple trunks per side", "trunk_instances", False)
    pick((u.month == "unknown") & (u.trunk_min_side >= 4), "sample_june_dense.png",
         "Unknown/June — dense canopy view", "trunk_instances", False)
    pick((u.month == "april"), "sample_april.png",
         "April — bare-vine frame", "trunk_instances", False)
    # Sparse / row-end frames
    pick((u.month == "march") & (u.trunk_instances <= 2) & (u.trunk_instances >= 1),
         "sample_march_sparse.png", "March — sparse trunks (row end?)", "trunk_instances", True)
    pick((u.trunk_instances == 0) & (u.pole_instances >= 1),
         "sample_no_trunk_poles_only.png", "No trunks — poles only", "pole_instances", False)
    pick((u.trunk_instances == 0) & (u.pole_instances == 0) & (u.building_instances > 0),
         "sample_structureless.png", "No trunks/poles — background only", None)
    # Occluded / low element count but present
    pick((u.trunk_min_side == 0) & (u.trunk_instances >= 3),
         "sample_one_sided.png", "One-sided trunks (all on one side)", "trunk_instances", False)
    # Robot frame prominent
    pick((u.robot_area_px > u.robot_area_px.quantile(0.9)),
         "sample_robot_prominent.png", "Robot frame prominent in view", "robot_area_px", False)
    # Vehicle present
    pick((u.vehicle_instances >= 1), "sample_vehicle.png",
         "Vehicle present in row", "vehicle_instances", False)
    # Building heavy
    pick((u.building_area_px > u.building_area_px.quantile(0.95)),
         "sample_building.png", "Heavy background building/structure", "building_area_px", False)
    # Pipe prominent
    pick((u.pipe_instances >= u.pipe_instances.quantile(0.95)),
         "sample_pipe.png", "Irrigation pipe prominent", "pipe_instances", False)
    # Most dense overall
    pick(u.trunk_instances >= 1, "sample_max_density.png",
         "Highest trunk density frame", "trunk_instances", False)
    # A balanced 'easy' centred frame
    pick((u.trunk_left_instances >= 3) & (u.trunk_right_instances >= 3) &
         (abs(u.trunk_left_instances - u.trunk_right_instances) <= 1),
         "sample_balanced_easy.png", "Balanced easy frame — even trunks both sides",
         "trunk_instances", False)

    for fn, out_fname, title in chosen:
        save_overlay_png(by_fn[fn], out_fname, title)
    print(f"Wrote {len(chosen)} sample visualisations")
    return chosen


def main():
    uniq, recs = load_unique_with_anns()
    print(f"Unique frames: {len(uniq)}")
    print("Section 2: aggregate histograms...")
    aggregate_histograms(recs)
    side_count_histograms(uniq)
    print("Section 3: viability summary...")
    viability_summary(uniq)
    print("Section 4: seasonal summary...")
    seasonal_summary(uniq, recs)
    print("Section 5: sample visualisations...")
    sample_visualisations(uniq, recs)
    print("All report sections done.")


if __name__ == "__main__":
    main()
