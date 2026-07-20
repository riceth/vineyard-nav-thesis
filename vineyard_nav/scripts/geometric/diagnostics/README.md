# Geometric diagnostics — one-time, NOT needed to reproduce the pipeline

**Skip this folder if you are reproducing the work.** These scripts produced the row-model
validation figures and the analyses that justified each refinement of the row fit; none of
them is a pipeline stage. The reproduction path is `scripts/geometric/README.md`.

Run from `vineyard_nav/` as `python3 scripts/geometric/diagnostics/<name>.py`.

## Row-model validation figures

Rendered the committed sample-frame overlays under
`results/geometric/march/diagnostics/figures/rowfit_validation/`, which are the visual evidence
that the fit tracks the detected dots. Run in the order the row model evolved:

1. `figure_rowfit_hybrid.py` — seeding on the densest near-field cluster then refining by RANSAC,
   which stopped the fit landing in the gap between two rows (frames 4223 / 3991 / 4107).
2. `figure_rowfit_far.py` — extending the fit to far-field detections that agree with the near-field
   row, recovering rows whose evidence lies mostly beyond 5 m (`far_ext/`).
3. `figure_line_fit.py` — fitting a sloped line per row instead of a constant offset, so the
   centreline can tilt with the rows (`linefit/`, `linefit_final/`).

## Slope / tilt analysis

- `slope_analysis.py` — distribution of per-side row slopes across the two-row frames; the
  measurement showing the rows lean a consistent ~2.3° rather than varying randomly, which is what
  a projection/mounting offset looks like.
- `figure_slope.py` — renders that distribution (`linefit/slope_hist.png`).

## Bootstrap block-length exploration

- `autocorrelation_block_analysis.py` — measures the distance over which consecutive frames stay
  correlated, and compares the resulting CI widths three ways (Δs = 1.5 m subsample vs
  measured-decorrelation subsample vs moving-block bootstrap). The **reusable** outcome is
  `scripts/geometric/block_lengths.py`, which the pipeline imports; this script is the one-time
  investigation behind it.
