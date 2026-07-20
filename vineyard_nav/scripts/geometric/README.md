# Geometric strand — scripts

Row-model pipeline for the in-row centreline evaluation, from bag frame
extraction through to the committed centreline results. The flat
`.py` files here are the **reproduction pipeline**; `diagnostics/` and
`superseded/` are separated by role. Run any script as
`python3 vineyard_nav/scripts/geometric/<path>/<name>.py`.

## Pipeline (flat, this directory) — reproduces the committed results
The pooled analysis scripts are **bag-agnostic**: each takes `--bag <name>` and
resolves every per-bag path via `bag_config.py`, so the same logic runs on
march / april / … unchanged (e.g. `line_fit_infer.py --bag march`). Frames are
selected on `eligible` alone — every usable in-row frame is evaluated together
rather than being split into val/test, which keeps the sample large enough for
tight confidence intervals.

- `contamination_census.py`, `extract_frames.py`, `frame_manifest_build.py` — contamination census, frame extraction, whole-bag manifest build. (`frame_manifest_build.py` — renamed from `dataset_split.py` — marks every usable frame `split="eligible"` and picks a Δs = 1.5 m spaced subsample for the bootstrap. It deliberately does **no** val/test partitioning: the whole bag is pooled into a single evaluation for statistical power.)
- `projection_calibration.py` — image→ground IPM projection (shared `project_px`, imported by drivers).
- `single_arm_dryrun.py` — single-arm dry-run **and** the shared parameter/function module (`CONF`, `BLOB_FRAC`, `FRAME_PX`, `side_valid`, `bin_centre`) imported by every driver.
- `row_model.py` — fits a straight line to each vine row and takes the midline between them as the centreline: seeds on the densest near-field cluster of detections, refines with RANSAC, then extends the fit to far-field detections consistent with that row. `exec`'d by the drivers + figure scripts.
- `bag_config.py` — per-bag path resolution (`--bag`); single source of manifest / frames / db3 / cache / output paths for the whole-bag scripts. Add a bag by adding one `BAGS` entry.
- `block_lengths.py` — how many consecutive frames count as a single independent sample (adjacent frames are near-duplicates, so CIs would be over-tight otherwise); re-derived per bag and shared by the CI estimators — line-fit / paired / config — so they cannot drift apart.
- `extract_detections.py` — Phase-C detection cache over all eligible frames → `cache/detections.csv`.
- `line_fit_infer.py` — whole-bag line-fit inference (9 models × all eligible frames) → `line_fit_per_frame.csv` (12-col, full precision).
- `line_fit_eval.py` — aggregates the per-frame CSV into the headline accuracy numbers: lateral offset and heading error per arm, with bootstrap confidence intervals → `line_fit_report.json`.
- `paired_crossarm.py` — compares the arms on the *same* frames and bootstraps the per-frame difference, so error common to all arms cancels out of the comparison.
- `config_analysis.py` — sweeps the downstream class/threshold configurations and runs single-class ablations to show which detections actually carry the row fit (reads the detection cache).
- `lidar_crosscheck.py` — checks the camera-derived row heading against LiDAR, i.e. whether an independent sensor sees the same tilt.
- `paths.py` — legacy `CACHE_DIR` constant, now used only by the superseded val/test scripts.

## `diagnostics/` — investigation / figure regeneration (not needed for the numbers)
- `figure_rowfit_hybrid.py`, `figure_rowfit_far.py`, `figure_line_fit.py`, `figure_slope.py` — regenerate the committed `results/geometric/march/diagnostics/figures/rowfit_validation/` PNGs.
- `slope_analysis.py` — histogram of the fitted row slopes, used to test whether the rows share a systematic lean rather than scattering randomly; it found the ~2.3° common tilt that motivated the line-fit centreline.
- `autocorrelation_block_analysis.py` — measures how far apart two frames must be before they stop being correlated, then cross-checks (`--bag`) the block lengths `block_lengths.py` derives from it.

## `superseded/` — retired, not for reproduction
- `yconstant_val_eval.py` — the pre-refinement evaluator, which modelled each row as a constant lateral position across the near 5 m instead of fitting a sloped line.
- `line_fit_{val,test}_eval.py`, `paired_crossarm_{val,test}.py`, `config_sweep_val.py`, `config_ablation_{val,test}.py`, `lidar_crosscheck_{val,test}.py`, `extract_detections_{val,test}.py` — the val/test-split evaluators (11 scripts). They were retired when the split was dropped and every in-row frame pooled into one evaluation, which the bag-agnostic scripts above now do in a single pass. Paths re-pointed to `results/geometric/march/superseded/march_val_test_split/`; kept as audit trail, not on the reproduction path.

## Module / output split — `single_arm_dryrun`
The **module** `single_arm_dryrun.py` stays in the pipeline because the current
drivers import shared constants/functions from it. Its **output** — the
dry-run report `results/geometric/march/single_arm_dryrun_report.json` (+ samples)
— is the *superseded* near-5 m Y-constant row model. Module and output are split
by role. (Result-side placement of that output is deferred to the pooling task and
is not changed here.)
