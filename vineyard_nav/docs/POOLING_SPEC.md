# POOLING_SPEC.md — March val+test pooling

**Purpose.** Execution contract for pooling the March (`kg_march_23`) geometric
strand's validation and held-out test evaluations into a single whole-bag
evaluation. Establishes the per-month template for multi-bag. Rationale and the
superseded framing are in **DECISIONS.md D040**.

**One-line rationale.** The within-bag val/test split served config-lock
leakage-control (F018 selected on val, locked before CP-6); with the config
locked it has served its purpose. Seasonal generalisation is claimed at the
**multi-bag** level (whole-bag evaluation per month, no per-bag splits), so
per-bag pooling maximises statistical power and standardises methodology across
bags. March accordingly loses its within-bag held-out check (accepted; D040).

---

## A. Target structure

```
results/geometric/march/
├── final/
│   └── march_evaluation/                      # pooled (replaces val_evaluation/ + test_evaluation/)
│       ├── line_fit_march_report.json
│       ├── line_fit_march_per_frame.csv
│       ├── paired_crossarm_march.json
│       ├── config_analysis_march.json         # sweep + ablations, consolidated
│       └── lidar_crosscheck_march.json
└── superseded/
    ├── yconstant_val_evaluation/              # existing, unchanged
    └── march_val_test_split/                  # moved from final/ (Commit 2b)
        ├── val_evaluation/                     # 5 files, verbatim
        └── test_evaluation/                    # 5 files, verbatim
```

File-merge map (pooled ← val + test):
| Pooled | Merges |
|---|---|
| `line_fit_march_report.json` + `line_fit_march_per_frame.csv` | line_fit val + test |
| `paired_crossarm_march.json` | paired val + test |
| `config_analysis_march.json` | `config_sweep_val.json` (sweep + val ablations) + `config_ablation_test.json` |
| `lidar_crosscheck_march.json` | lidar val + test |

Per-month template (multi-bag): `results/geometric/<month>/final/<month>_evaluation/`
+ `results/geometric/cross_month/final/` for cross-bag synthesis.

---

## B. Analyses (re-run on pooled frames)

Pooled eligible frames = **val 4,708 + test 3,149 = 7,857** (all in-row eligible
frames across passes 0–10; spans all 5 corridors). Paired analyses use the pooled
**both-two-row** subset (val ~3,600 + test ~2,200; exact count from the re-run).

1. **Line-fit evaluation** — all 9 models × 7,857 frames, no split filter →
   `line_fit_march_report.json` + `line_fit_march_per_frame.csv` (per-arm GT-1 RMS,
   GT-2 tilt, coverage, base-points, decomposition inputs).
2. **Paired cross-arm bootstrap** — pooled both-two-row frames →
   `paired_crossarm_march.json`. **Re-derive Analysis-H decorrelation distances +
   moving-block lengths on the pooled data** (the L_GT1=11 / L_GT2=31 were
   val-derived; pooled cross-corridor autocorrelation may differ).
3. **Config sweep + single-class ablations** — pooled frames →
   `config_analysis_march.json`. Design decision (class-agnostic) is **already
   locked, not re-selected** — re-reported on pooled data for consistency only.
4. **LiDAR cross-check** — anchor frames drawn from the pooled corridors →
   `lidar_crosscheck_march.json` (F017 sensor-common tilt on pooled anchors).

The findings' numerical content is produced here; findings *text* is written in
Commit 3, after these numbers exist.

---

## C. Scripts (new pooled scripts; val/test scripts → superseded)

**New (pipeline, `scripts/geometric/`):**
- `line_fit_march_eval.py` ← consolidates `line_fit_val_eval.py` + `line_fit_test_eval.py` (evaluate all eligible frames)
- `paired_crossarm_march.py` ← consolidates `paired_crossarm_val.py` + `paired_crossarm_test.py`
- `config_analysis_march.py` ← consolidates `config_sweep_val.py` + `config_ablation_val.py` + `config_ablation_test.py`
- `lidar_crosscheck_march.py` ← consolidates `lidar_crosscheck_val.py` + `lidar_crosscheck_test.py`
- `diagnostics/autocorrelation_block_analysis.py` — re-pointed to `final/march_evaluation/line_fit_march_per_frame.csv` (stays in place; it is the block-length derivation for #2)

**Moved to `scripts/geometric/superseded/` (Commit 2b, audit trail):** the 8
val/test scripts above. Consistent with the project's kept-but-marked pattern
(yconstant, F015).

Pooled scripts read `dataset_manifest.json` (all eligible frames, no split key) +
the detection cache (`CACHE_DIR`), write into `final/march_evaluation/`. Path
substitution + the split-filter removal are the only logic differences from the
val/test originals (documented per script).

---

## D. Findings plan (Commit 3 — needs Commit-2a numbers)

- **F013** → pooled March cross-arm result. Drop "held-out" / "confirmed on test"
  framing. Report pooled GT-1 (primary) and GT-2 (secondary) paired CIs. **If the
  pooled GT-2 B-vs-others difference persists (CI excl 0), it is reported honestly**
  — F013-pooled is not assumed clean-invariant a priori.
- **F019** → **SUPERSEDED banner** (kept as historical trail, like F015). Its
  purpose (test-side confirmation) is absorbed into pooled F013; D040 supersedes
  the *interpretation that F019 was load-bearing*, not F019-the-record.
- **F010, F011, F012, F014, F016, F017, F018** → merge each `Test-side
  confirmation (CP-6)` block into the main measured content; drop the val-vs-test
  distinction; report pooled numbers; note the merge inline.
- **Writeup-wording blocks** (all findings) → revise: Fully-defensible re-worded
  to "pooled March measurement"; Citation maps → `final/march_evaluation/…`;
  NOT-defensible drops val/test-specific lines.

---

## E. STATUS + template (Commit 4)
Rewrite current-state to the unified pooled March strand; close the pooling open
item; add the multi-bag-whole-bag-evaluation note; record the per-month template
(A) for the seasonal bags.

---

## F. Untouched by pooling (verify at end)
Perception (all), CP-0/1/2/3 top-level artefacts + their citations, the retained
dev material (`single_arm_dryrun_samples/`, results `superseded/yconstant_val_evaluation/`,
`diagnostics/figures/rowfit_validation/`), utilities. A final grep confirms no
cross-reference to a moved val/test path is left dangling.

---

## G. Commit sequence (5)

1. **Plan** — `POOLING_SPEC.md` + `DECISIONS.md` D040. Docs-only.
2a. **Execute pooling analyses** — create the 4 pooled scripts + re-point
   autocorrelation → run → produce `final/march_evaluation/` artefacts. Val/test
   scripts + artefacts still present (verifiable in isolation).
2b. **Supersede val/test** — move the 8 val/test scripts → `scripts/geometric/superseded/`;
   move val/test artefacts → `results/geometric/march/superseded/march_val_test_split/{val,test}_evaluation/`;
   update READMEs.
3. **Findings** — F013 pooled, F019 superseded, 7 test-side blocks merged, writeup
   wording revised, citation maps → `march_evaluation/`.
4. **STATUS + template** — unified March state, per-month template, open items.

### Verification checkpoint (between 2a and 2b — must pass before superseding)
- The 4 pooled scripts run clean end-to-end.
- Pooled outputs exist + are readable.
- Pooled frame count = 7,857 (= val 4,708 + test 3,149, minus any documented exclusions).
- Pooled paired count consistent with val + test both-two-row pool.
- Pooled F013 numbers plausible vs the val + test individual numbers (weighted-average sanity check).
- **Report whether F019's B-vs-others GT-2 micro-difference persists in the pooled paired analysis.**

If anything is unexpected, **pause and investigate before 2b** (do not move
artefacts to superseded until the pooled results are validated).
