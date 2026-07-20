# Geometric strand — scripts

Row-model pipeline for the in-row centreline evaluation, from bag frame
extraction through to the committed centreline results. The flat
`.py` files here are the **reproduction pipeline**; `diagnostics/` and
`superseded/` are separated by role. Run any script as
`python3 vineyard_nav/scripts/geometric/<path>/<name>.py`.

## Pipeline (flat, this directory) — reproduces the committed results
The pooled analysis scripts are **bag-agnostic** (D040 whole-bag): each takes
`--bag <name>` and resolves every per-bag path via `bag_config.py`, so the same
logic runs on march / april / … unchanged (e.g. `line_fit_infer.py --bag march`).
Frames are selected on `eligible` alone — there is no val/test split in the
pipeline logic.

- `contamination_census.py`, `extract_frames.py`, `frame_manifest_build.py` — contamination census, frame extraction, whole-bag manifest build. (`frame_manifest_build.py` — renamed from `dataset_split.py` in Commit 5 — writes a single canonical `split="eligible"` marker and a whole-bag Δs=1.5 m subsample; no val/test partitioning. D040/D041.)
- `projection_calibration.py` — image→ground IPM projection (shared `project_px`, imported by drivers).
- `single_arm_dryrun.py` — single-arm dry-run **and** the shared parameter/function module (`CONF`, `BLOB_FRAC`, `FRAME_PX`, `side_valid`, `bin_centre`) imported by every driver.
- `row_model.py` — D036–D038 row model (`exec`'d by the drivers + figure scripts).
- `bag_config.py` — per-bag path resolution (`--bag`); single source of manifest / frames / db3 / cache / output paths for the whole-bag scripts. Add a bag by adding one `BAGS` entry.
- `block_lengths.py` — Analysis-H moving-block lengths, re-derived per bag; shared by the CI estimators (line-fit / paired / config) so they cannot drift.
- `extract_detections.py` — Phase-C detection cache over all eligible frames → `cache/detections.csv`.
- `line_fit_infer.py` — whole-bag line-fit inference (9 models × all eligible frames) → `line_fit_per_frame.csv` (12-col, full precision).
- `line_fit_eval.py` — aggregation of the per-frame CSV → `line_fit_report.json` (F010–F013).
- `paired_crossarm.py` — F013/F019 paired cross-arm difference bootstrap.
- `config_analysis.py` — F018 config sweep + single-class ablations (reads the detection cache).
- `lidar_crosscheck.py` — F017 LiDAR vs camera row-heading cross-check.
- `paths.py` — legacy `CACHE_DIR` constant, now used only by the superseded val/test scripts.

## `diagnostics/` — investigation / figure regeneration (not needed for the numbers)
- `figure_rowfit_hybrid.py`, `figure_rowfit_far.py`, `figure_line_fit.py`, `figure_slope.py` — regenerate the committed `results/geometric/march/diagnostics/figures/rowfit_validation/` PNGs.
- `slope_analysis.py` — slope histogram + D038 motivation.
- `autocorrelation_block_analysis.py` — Analyses H/I; human cross-check (`--bag`) of the moving-block lengths derived by `block_lengths.py`.

## `superseded/` — retired, not for reproduction
- `yconstant_val_eval.py` — the pre-refinement near-5 m Y-constant (D035) val evaluator.
- `line_fit_{val,test}_eval.py`, `paired_crossarm_{val,test}.py`, `config_sweep_val.py`, `config_ablation_{val,test}.py`, `lidar_crosscheck_{val,test}.py`, `extract_detections_{val,test}.py` — the val/test-split evaluators (11 scripts), superseded by the whole-bag pooled scripts above (**D040**). Paths re-pointed to `results/geometric/march/superseded/march_val_test_split/`; kept as audit trail, not on the reproduction path.

## Module / output split — `single_arm_dryrun`
The **module** `single_arm_dryrun.py` stays in the pipeline because the current
drivers import shared constants/functions from it. Its **output** — the
dry-run report `results/geometric/march/single_arm_dryrun_report.json` (+ samples)
— is the *superseded* near-5 m Y-constant row model. Module and output are split
by role. (Result-side placement of that output is deferred to the pooling task and
is not changed here.)
