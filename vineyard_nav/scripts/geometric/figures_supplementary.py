"""Supplementary cross-bag figures for the progress update / Notion pages.

Three figures the locked per-bag set (figures.py) does not cover. They are written to the shared
comparison directory, so the locked per-bag sets (15 figures, 13 on june) are left untouched.

  python3 scripts/geometric/figures_supplementary.py --bags march april may june

  1. cmp_model_outputs_<bag>.png   what each ARM ACTUALLY OUTPUTS on one scene: arm A's per-pixel
                                   mask, arm B's instance masks, arm C's class-labelled instances.
                                   figures.py fig2 compares the resulting centrelines; nothing showed
                                   the raw outputs. Rendered on that bag's curated `arm_invariance`
                                   frame (verified two_row on all three arms, privacy-screened).
  2. cmp_season_contrast.png       a bare-vine frame beside a canopy frame, same arm, base points
                                   drawn + counted -- makes the canopy base-point starvation visible.
  3. cmp_coverage_trend.png        the seasonal headline, data-only from committed JSONs: two-row
                                   coverage against the 70% viability floor, the pole contribution to
                                   coverage (season-invariant), and base points per frame (collapses).

Figures 1-2 need the model checkpoints (they re-run inference on a handful of frames); figure 3 is
data-only and runs from the committed JSONs alone.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG)); sys.path.insert(0, str(PKG / "scripts" / "geometric"))
import cuda_preload  # noqa: E402,F401 — cuDNN cold-init guard; MUST precede torch (D049)
import figures as FG  # noqa: E402  (model cache, styling, curated FRAMES registry)
from bag_config import resolve  # noqa: E402

SEED = 42
SEASON = {"march": "bare-vine", "april": "bare-vine", "may": "canopy", "june": "canopy",
          "july": "canopy", "september": "canopy",
          "july2023": "canopy", "august2023": "canopy"}   # verified from the extracted frames at A3
C_BARE, C_CANOPY = "#4477aa", "#228833"
OUT = PKG / "results/geometric/comparison/figures"


def _out(name):
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / name


def _frame(bag, key="arm_invariance"):
    B = resolve(bag, "eligible")
    fi = FG.FRAMES[bag][key]
    img = cv2.imread(str(B["frames_dir"] / f"{fi:05d}.jpg"))
    if img is None:
        raise SystemExit(f"figures_supplementary: missing frame {fi} for bag {bag}")
    return fi, img


def _blend(img, mask, colour, alpha=0.45):
    """Overlay a boolean mask on a BGR image in `colour` (matplotlib hex)."""
    bgr = tuple(int(colour[i:i + 2], 16) for i in (5, 3, 1))    # hex RGB -> BGR
    out = img.copy()
    out[mask] = ((1 - alpha) * img[mask] + alpha * np.array(bgr)).astype(np.uint8)
    return out


def fig_model_outputs(bag):
    """Arm A per-pixel mask | arm B instance masks | arm C class-labelled instances, one scene."""
    fi, img = _frame(bag)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 3, figsize=(11.4, 4.3))

    # --- arm A: per-pixel binary mask (argmax over the U-Net class map) ---
    _, net = FG._model("A", SEED)
    x = FG._TF(image=rgb)["image"].unsqueeze(0).to(FG._DEV)
    import torch
    with torch.no_grad():
        fg = (net(x).argmax(1)[0].cpu().numpy() == 1)
    a_img = _blend(img, fg, FG.COL["binary"])
    # how the mask becomes structures downstream: connected components >= UNET_MIN_AREA (pipeline step 2)
    ncomp, _, stats, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), 8)
    nkeep = int(sum(1 for k in range(1, ncomp) if stats[k][4] >= FG.UNET_MIN_AREA))
    ax[0].imshow(cv2.cvtColor(a_img, cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"A — U-Net binary\nper-pixel mask, no instance concept\n"
                    f"{int(fg.sum()):,} px → {nkeep} components (≥{FG.UNET_MIN_AREA} px)", fontsize=8.5)

    # --- arms B and C: per-instance masks ---
    # distinct hues so the INSTANCE separation is visible; deliberately avoids the trunk-blue /
    # pole-yellow of panel C, where colour is semantic rather than arbitrary
    palette = ["#00d0d0", "#e05fd8", "#ff8c42", "#4cd964", "#a06cff", "#ff5c7a", "#00a3a3", "#c2185b"]
    for k, arm in enumerate(("B", "C"), start=1):
        _, m = FG._model(arm, SEED)
        r = m.predict(source=img, conf=FG.CONF, quantize=16, device=0, verbose=False)[0]
        vis = img.copy()
        n = 0
        if r.masks is not None and r.boxes is not None:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            xy = r.boxes.xyxy.cpu().numpy()
            area = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1])
            keep = area <= FG.BLOB_FRAC * FG.FRAME_PX * FG.FRAME_PX      # same 15% guard as the pipeline
            for j, mk in enumerate(r.masks.data.cpu().numpy()):
                if not keep[j]:
                    continue
                mm = cv2.resize(mk, (FG.FRAME_PX, FG.FRAME_PX)) > 0.5
                col = FG.cls_col(int(cls[j])) if arm == "C" else palette[n % len(palette)]
                vis = _blend(vis, mm, col)
                n += 1
        ax[k].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        if arm == "B":
            ax[k].set_title(f"B — YOLOv11-seg binary\nper-instance masks, one foreground class\n"
                            f"{n} instances (arbitrary colour each)", fontsize=8.5)
        else:
            nt = int(((cls == 0) & keep).sum()); npl = int(((cls == 1) & keep).sum())
            ax[k].set_title(f"C — YOLOv11-seg multiclass\nper-instance masks, class-labelled\n"
                            f"{n} instances: {nt} trunk (blue) / {npl} pole (yellow)", fontsize=8.5)
    for a in ax:
        a.axis("off")
    fig.suptitle(f"What each arm outputs — {bag} ({SEASON.get(bag, '?')}), frame {fi}", y=1.01)
    fig.text(0.5, -0.02, "All three outputs are reduced to the same primitive downstream: one base point per structure "
                         "(bottom-centre), projected to the ground plane. The row fit is class-agnostic.",
             ha="center", va="top", fontsize=8, color="0.25")
    fig.tight_layout()
    p = _out(f"cmp_model_outputs_{bag}.png"); fig.savefig(p); plt.close(fig)
    return p


def _representative_frame(bag, coco, arm="C", seed=SEED, shortlist=60):
    """An in-row two_row frame whose base-point count is closest to THIS BAG'S MEDIAN, and which
    contains no identifiable people. The curated `anatomy` frames are best-case by construction, so
    using them here would understate the seasonal difference — the median frame is the honest pick."""
    B = resolve(bag, "eligible")
    rows = []
    for ln in Path(B["per_frame_csv"]).read_text().splitlines()[1:]:
        a, s, i, cls, *_rest = ln.split(",")
        if a == arm and int(s) == seed and cls == "two_row":
            rows.append((int(i), int(_rest[5])))          # ... mL,mR,mc,n_base,adj,flags
    med = float(np.median([n for _, n in rows]))
    for i, n in sorted(rows, key=lambda r: (abs(r[1] - med), r[0]))[:shortlist]:
        img = cv2.imread(str(B["frames_dir"] / f"{i:05d}.jpg"))
        if img is None:
            continue
        r = coco.predict(source=img, conf=0.25, verbose=False)[0]
        cl = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
        if not any(int(k) == 0 for k in cl):              # COCO class 0 = person
            return i, img, med
    raise SystemExit(f"figures_supplementary: no person-free representative frame for {bag}")


def fig_season_contrast(bare="march", canopy="june", arm="C"):
    """One bare-vine frame beside one canopy frame, same arm, base points drawn and counted.
    Frames are each bag's MEDIAN-detection frame (see _representative_frame), not a curated best case."""
    from ultralytics import YOLO
    coco = YOLO(str(PKG / "yolo11n-seg.pt"))
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for k, bag in enumerate((bare, canopy)):
        fi, img, med = _representative_frame(bag, coco, arm)
        base = FG.frontend(arm, SEED, img)
        o = FG.fit_frame(base)
        vis = img.copy()
        for (uc, v, c) in base:
            cv2.circle(vis, (int(uc), int(v)), 5, tuple(int(FG.cls_col(c)[i:i + 2], 16) for i in (5, 3, 1)), -1)
            cv2.circle(vis, (int(uc), int(v)), 5, (20, 20, 20), 1)
        ax[k].imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)); ax[k].axis("off")
        near = int(sum(1 for p in np.vstack([o["L"], o["R"]]) if p[0] <= FG.NEAR)) if len(base) else 0
        ax[k].set_title(f"{bag} — {SEASON.get(bag,'?')}, frame {fi} (median-detection frame)\n"
                        f"{len(base)} base points ({near} within the 5 m near-seed window) · bag median {med:.0f}",
                        fontsize=9, color=(C_BARE if SEASON.get(bag) == "bare-vine" else C_CANOPY))
    fig.suptitle(f"Two seasons, each at its typical frame — detections available to the row fit (arm {arm})", y=1.02)
    fig.text(0.5, -0.02, "Canopy roughly halves the detections available to the fit (bag medians 32 -> 14). The rows stay visible; what collapses is the\n"
                         "number of usable ground-contact points. Abstention then follows per SIDE — a frame is declined when either side alone lacks two near-field seeds.",
             ha="center", va="top", fontsize=8, color="0.25")
    fig.tight_layout()
    p = _out("cmp_season_contrast.png"); fig.savefig(p); plt.close(fig)
    return p


def fig_coverage_trend(bags):
    """Data-only: coverage vs the 70% floor, the pole contribution, and base points per frame."""
    cov_a, cov_t, base_a, base_p = [], [], [], []
    for b in bags:
        c = json.load(open(resolve(b, "eligible")["out_dir"] / "config_analysis.json"))["cells"]
        cov_a.append(c["agnostic"]["two_row_pct"]); cov_t.append(c["trunk_only"]["two_row_pct"])
        base_a.append(c["agnostic"]["mean_base"]); base_p.append(c["pole_only"]["mean_base"])
    x = np.arange(len(bags))
    cols = [C_BARE if SEASON.get(b) == "bare-vine" else C_CANOPY for b in bags]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    # -- panel 1: coverage + the pole contribution, against the 70% floor --
    ax[0].bar(x - 0.19, cov_t, 0.38, color=cols, alpha=0.45, edgecolor="0.3", lw=0.6, label="trunk-only")
    ax[0].bar(x + 0.19, cov_a, 0.38, color=cols, edgecolor="0.3", lw=0.6, label="class-agnostic (trunk + pole)")
    ax[0].axhline(70, ls="--", lw=1.2, color="#d1341c")
    ax[0].text(-0.42, 71.2, "70% viability floor", color="#d1341c", fontsize=8, ha="left")
    for i, (t, a) in enumerate(zip(cov_t, cov_a)):
        ax[0].annotate(f"+{a - t:.1f} pp", (i, a + 1.5), ha="center", fontsize=8, color="0.25")
    ax[0].set_xticks(x); ax[0].set_xticklabels([f"{b}\n{SEASON.get(b,'')}" for b in bags])
    ax[0].set_ylabel("two-row coverage (%)"); ax[0].set_ylim(0, 95)
    ax[0].set_title("Coverage falls below the deployability floor —\nbut the pole contribution does not change", fontsize=9)
    ax[0].legend(fontsize=8, loc="upper right", framealpha=0.95); ax[0].grid(alpha=0.3, axis="y")

    # -- panel 2: base points per frame --
    ax[1].plot(x, base_a, "-o", color="0.25", label="all detections (class-agnostic)")
    ax[1].plot(x, base_p, "-s", color=FG.COL["pole"], mec="0.3", label="pole detections only")
    for i, v in enumerate(base_a):
        ax[1].annotate(f"{v:.1f}", (i, v + 1.1), ha="center", fontsize=8, color="0.25")
    for i, v in enumerate(base_p):
        ax[1].annotate(f"{v:.1f}", (i, v - 2.2), ha="center", fontsize=8, color="0.45")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{b}\n{SEASON.get(b,'')}" for b in bags])
    ax[1].set_ylabel("mean base points per frame"); ax[1].set_ylim(0, max(base_a) * 1.25)
    ax[1].set_title("The mechanism: base-point availability collapses\n(pole visibility falls hardest)", fontsize=9)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    fig.suptitle("What the canopy season costs — four bags, two seasons", y=1.02)
    fig.text(0.5, -0.02, "Class structure contributes the same coverage gain in both seasons (+12.4 to +13.7 pp); what canopy "
                         "removes is the detections themselves. Source: each bag's committed config_analysis.json.",
             ha="center", va="top", fontsize=8, color="0.25")
    fig.tight_layout()
    p = _out("cmp_coverage_trend.png"); fig.savefig(p); plt.close(fig)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", default=["march", "april", "may", "june"])
    ap.add_argument("--only", default=None, choices=["outputs", "contrast", "trend"])
    a = ap.parse_args()
    print(f"[figures_supplementary] bags = {a.bags}")
    if a.only in (None, "outputs"):
        for b in a.bags:
            print(f"  {fig_model_outputs(b).relative_to(PKG)}")
    if a.only in (None, "contrast"):
        bare = next((b for b in a.bags if SEASON.get(b) == "bare-vine"), None)
        can = next((b for b in reversed(a.bags) if SEASON.get(b) == "canopy"), None)   # deepest canopy
        if bare and can:
            print(f"  {fig_season_contrast(bare, can).relative_to(PKG)}")
        else:
            print("  (skipped contrast: need one bare-vine and one canopy bag)")
    if a.only in (None, "trend"):
        print(f"  {fig_coverage_trend(a.bags).relative_to(PKG)}")
    print("done.")
