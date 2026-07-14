# Geometric strand — March bag (`kg_march_23`) results

Artefacts from the geometric-strand evaluation on the March bag: image → base
points → CP-2 IPM projection → row fit → centreline → GT-1 lateral offset / GT-2
heading, run across all three arms (A/B/C × seeds 42/43/44).

Two directories separate **reported results** (`final/`) from **diagnostic
evidence** (`diagnostics/`) and **retired artefacts** (`superseded/`). The
checkpoint artefacts CP-0…CP-3 remain at top level as locked historical state.

## Top level — committed checkpoint artefacts (do not move)

| Artefact | Checkpoint | Commit | Role |
|---|---|---|---|
| `contamination_census_exclusions.json` | CP-0 | 4885320 | Contamination census / frame-exclusion list |
| `dataset_manifest.json`, `val_test_split_summary.json`, `dataset_split_samples/` | CP-1 | 8365ed9 | Processed dataset + val/test split |
| `projection_calibration_report.json`, `projection_calibration_samples/` | CP-2 | 32de7c8 | Image→world projection calibration (D034) |
| `single_arm_dryrun_report.json`, `single_arm_dryrun_samples/` | CP-3 | 798d7d4 | Single-arm dry run — **superseded row model**, kept as locked historical state (see `superseded/README.md`) |

## `final/` — reported results

**`val_evaluation/`** — validation (passes 2,4,5,6,7,8,10; 4,708 eligible frames):
- `line_fit_val_report.json` — CP-5 9-model val evaluation, locked line-fit pipeline (D036–D038). Findings F010–F013.
- `line_fit_val_per_frame.csv` — per-frame GT-1/GT-2/coverage, 9 models.
- `paired_crossarm_val.json` — paired cross-arm difference bootstrap, val (F013).
- `config_sweep_val.json` — class-config sweep + single-class ablations, val (F018).
- `lidar_crosscheck_val.json` — val-side LiDAR vs camera row-heading cross-check (F017).

**`test_evaluation/`** — single-shot held-out test (passes 0,1,3,9; 3,149 eligible frames; evaluated once, rule 5):
- `line_fit_test_report.json` — CP-6 9-model test evaluation (F019).
- `line_fit_test_per_frame.csv` — per-frame test.
- `paired_crossarm_test.json` — paired cross-arm difference bootstrap, test (F019).
- `config_ablation_test.json` — test-side config ablation (F018 confirmation).
- `lidar_crosscheck_test.json` — test-side LiDAR cross-check (F017 confirmation).

**`figures/`** — placeholder (`.gitkeep`) for report figures generated after the multi-bag seasonal runs.

## `diagnostics/` — diagnostic evidence

Row-fit visual-validation figure tree (D036/D037/D038 development). See
`diagnostics/README.md`. The Analyses A–I numerical results live in
`docs/FINDINGS.md` and the session logs, not as standalone JSON.

## `superseded/` — retired artefacts

Buggy Y-constant CP-5 v1 run, retained as audit trail. See
`superseded/README.md`. (The superseded CP-3 *row model* artefacts stay at top
level — committed, locked historical state — and are not duplicated here.)
