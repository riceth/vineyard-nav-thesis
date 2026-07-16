# Geometric strand — scripts

Row-model pipeline for the in-row centreline evaluation (CP-0…CP-6). The flat
`.py` files here are the **reproduction pipeline**; `diagnostics/` and
`superseded/` are separated by role. Run any script as
`python3 vineyard_nav/scripts/geometric/<path>/<name>.py`.

## Pipeline (flat, this directory) — reproduces the committed results
- `contamination_census.py`, `extract_frames.py`, `dataset_split.py` — CP-0/CP-1 census, extraction, val/test split.
- `projection_calibration.py` — CP-2 image→ground IPM (shared `project_px`, imported by drivers).
- `single_arm_dryrun.py` — CP-3 dry-run **and** the shared parameter/function module (`CONF`, `BLOB_FRAC`, `FRAME_PX`, `side_valid`, `bin_centre`) imported by every driver.
- `row_model.py` — D036–D038 row model (`exec`'d by the drivers + figure scripts).
- `paths.py` — shared `CACHE_DIR` for the detection cache.
- `extract_detections_{val,test}.py` — Phase-C detection cache.
- `config_sweep_val.py`, `config_ablation_{val,test}.py` — F018 sweep + single-class ablations.
- `line_fit_{val,test}_eval.py` — CP-5/CP-6 evaluation → `line_fit_*` reports.
- `paired_crossarm_{val,test}.py` — F013/F019 paired bootstrap.
- `lidar_crosscheck_{val,test}.py` — F017 LiDAR cross-check.

## `diagnostics/` — investigation / figure regeneration (not needed for the numbers)
- `figure_rowfit_hybrid.py`, `figure_rowfit_far.py`, `figure_line_fit.py`, `figure_slope.py` — regenerate the committed `results/geometric/march/diagnostics/figures/rowfit_validation/` PNGs.
- `slope_analysis.py` — slope histogram + D038 motivation.
- `autocorrelation_block_analysis.py` — Analyses H/I; derives the moving-block lengths hardcoded in `paired_crossarm_*.py`.

## `superseded/` — retired, not for reproduction
- `yconstant_val_eval.py` — the pre-refinement near-5 m Y-constant (D035) val evaluator.

## Module / output split — `single_arm_dryrun`
The **module** `single_arm_dryrun.py` stays in the pipeline because the current
drivers import shared constants/functions from it. Its **output** — the CP-3
dry-run report `results/geometric/march/single_arm_dryrun_report.json` (+ samples)
— is the *superseded* near-5 m Y-constant row model. Module and output are split
by role. (Result-side placement of that output is deferred to the pooling task and
is not changed here.)
