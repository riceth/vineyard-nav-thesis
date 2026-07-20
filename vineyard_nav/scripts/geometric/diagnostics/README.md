# Geometric diagnostics — one-time, NOT needed to reproduce the pipeline

**Skip this folder if you are reproducing the work.** These scripts produced the row-model
validation figures and the supporting analyses behind D036–D038; none of them is a pipeline
stage. The reproduction path is `scripts/geometric/README.md`.

Run from `vineyard_nav/` as `python3 scripts/geometric/diagnostics/<name>.py`.

## Row-model validation figures

Rendered the committed sample-frame overlays under
`results/geometric/march/diagnostics/figures/rowfit_validation/`, which are the visual evidence
that the fit tracks the detected dots. Run in the order the row model evolved:

1. `figure_rowfit_hybrid.py` — D036 hybrid clustering + RANSAC (frames 4223 / 3991 / 4107).
2. `figure_rowfit_far.py` — D037 far-field extension (`far_ext/`).
3. `figure_line_fit.py` — D038 line-fit centreline (`linefit/`, `linefit_final/`).

## Slope / tilt analysis

- `slope_analysis.py` — per-side slope distribution over the two-row frames; the measurement
  behind D038's systematic ~2.3° common tilt (and F017).
- `figure_slope.py` — renders the slope histogram (`linefit/slope_hist.png`).

## Bootstrap block-length exploration

- `autocorrelation_block_analysis.py` — Analysis-I decorrelation-distance study comparing CI
  widths (Δs = 1.5 m subsample vs measured-decorrelation subsample vs moving-block bootstrap).
  The **reusable** outcome of this exploration lives in `scripts/geometric/block_lengths.py`,
  which the pipeline imports; this script is the one-time investigation behind it.
