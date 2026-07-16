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
| `dataset_manifest.json`, `manifest_summary.json` | CP-1 | 8365ed9 | Processed whole-bag dataset — `split="eligible"` marker + Δs=1.5 m subsample (D040/D041; the val/test-split manifest was superseded in Commit 5). Overlay samples moved to `superseded/dataset_split_samples/`. |
| `projection_calibration_report.json` | CP-2 | 32de7c8 | Image→world projection calibration (D034) |
| `single_arm_dryrun_report.json`, `single_arm_dryrun_samples/` | CP-3 | 798d7d4 | Single-arm dry run — **superseded row model**, kept as locked historical state (see `superseded/README.md`) |

## `final/` — reported results

**`march_evaluation/`** — whole-bag evaluation (**D040**): all 7,857 eligible frames
(passes 0–10, all 5 corridors; no val/test split). Filenames are bag-agnostic
(Option 1 — the `march/…/march_evaluation/` path carries the bag):
- `line_fit_per_frame.csv` — per-frame GT-1/GT-2/coverage + per-side slopes (12-col: `arm,seed,i,cls,offset,heading,mL,mR,mc,n_base,adj,flags`), 9 models × 7,857 frames, full precision.
- `line_fit_report.json` — per-model / per-arm aggregation + moving-block CIs. Findings F010–F013.
- `paired_crossarm.json` — paired cross-arm difference bootstrap (F013/F019).
- `config_analysis.json` — Phase-C config sweep + single-class ablations (F018).
- `lidar_crosscheck.json` — LiDAR vs camera row-heading cross-check (F017).

The pre-pooling val/test-split artefacts are retained under
`superseded/march_val_test_split/` (D040).

**`figures/`** — placeholder (`.gitkeep`) for report figures generated after the multi-bag seasonal runs.

## `diagnostics/` — diagnostic evidence

Row-fit visual-validation figure tree (D036/D037/D038 development). See
`diagnostics/README.md`. The Analyses A–I numerical results live in
`docs/FINDINGS.md` and the session logs, not as standalone JSON.

## `superseded/` — retired artefacts

- `yconstant_val_evaluation/` — buggy Y-constant CP-5 v1 run (D035).
- `march_val_test_split/{val,test}_evaluation/` — the pre-pooling val/test-split
  evaluation artefacts (5 files each), superseded by `final/march_evaluation/`
  under **D040** (whole-bag pooling). Retained verbatim as audit trail.

See `superseded/README.md`. (The superseded CP-3 *row model* artefacts stay at top
level — committed, locked historical state — and are not duplicated here.)
