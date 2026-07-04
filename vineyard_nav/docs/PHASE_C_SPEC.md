# PHASE_C_SPEC.md

**Scope:** Implementation contract for Phase C — YOLOv11-seg multiclass + class-aware downstream sweep.
**Audience:** Claude Code (and any future implementer). Read fully before writing code.
**Status:** Locked.
**Role in three-arm design:** The contribution. Comparison against Phase B (same architecture, different class structure) isolates the class-structure effect and enables downstream attribution via the 3-config sweep.

---

## 1. Goal

Fine-tune YOLOv11-seg from COCO-pretrained weights on SemanticBLT's trunk+pole 2-class labels. Sweep downstream geometry configurations (A/B/C × 6 T values) on validation to select a locked pipeline configuration. Evaluate test **once** at the locked configuration. Report sensitivity analysis and attribution.

Phase C is the arm where the class-aware advantage is tested and, if present, attributed to its source.

## 2. Folder structure (within `/workspaces/dissertation/vineyard_nav/`)

```
segmentation/yolo_multiclass/
├── __init__.py
├── data_prep.py           # COCO → YOLO multiclass labels (trunk=0, pole=1)
├── train.py               # ultralytics YOLO training entry point
├── evaluate.py            # Test-set evaluation entry point
├── inference.py           # Single-image / split inference utility
└── visualize.py           # Detection overlay rendering (class-coloured)

geometry/
├── clustering.py          # Per-side clustering (used by all arms)
├── ransac.py              # RANSAC line fitting
├── centreline.py          # Centreline as bisector of left/right lines
└── configs.py             # Config A/B/C dispatch for Phase C

evaluation/
├── downstream_sweep.py    # 3 configs × 6 T values on val
├── phase_c_test.py        # test evaluation at locked (config*, T*)
└── attribution.py         # 3-way comparison analysis

configs/
├── phase_c_yolo_multiclass_data.yaml
├── phase_c_yolo_multiclass_train.yaml
└── phase_c_downstream_sweep.yaml   # sweep parameters

data/yolo_multiclass/
├── images/
│   ├── train/             # symlinks or copies from data/semanticblt
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/

results/runs/
├── phase_c_yolo_multiclass_<timestamp>/    # YOLO training run
│   ├── weights/best.pt, last.pt
│   ├── results.csv
│   ├── config_snapshot.yaml
│   ├── git_commit.txt
│   └── args.yaml
└── phase_c_downstream_sweep_<timestamp>/   # downstream sweep run
    ├── sweep_results.csv                    # one row per (config, T) evaluated on val
    ├── selected_config.json                 # locked (config*, T*)
    ├── sensitivity_plots/
    └── test_metrics.json                    # final test at locked config
```

## 3. Implementation order (mandatory)

1. **`data_prep.py`** — COCO → YOLO conversion, trunk=0, pole=1
2. **Spot check** — YOLO labels visually verified on 3–5 samples per canopy state
3. **Data config YAML**
4. **`train.py`** — 2-epoch smoke run first
5. **YOLO training** — full run to completion; `best.pt` locked
6. **Downstream geometry modules** (`clustering.py`, `ransac.py`, `centreline.py`, `configs.py`)
7. **`downstream_sweep.py`** — sweep on val; `(config*, T*)` locked
8. **`phase_c_test.py`** — test evaluated once at locked config
9. **`attribution.py`** — 3-way comparison analysis
10. Test metrics + attribution paragraph recorded in DECISIONS.md

## 4. Data preparation (`data_prep.py`)

### 4.1 Class mapping

- COCO `category_id = 5` (trunk) → YOLO class 0
- COCO `category_id = 3` (pole) → YOLO class 1
- All other annotations (pipe, building, robot, vehicle) → dropped

Rationale in D025.

### 4.2 YOLO segmentation label format

Same as Phase B (Section 4.2 of PHASE_B_SPEC.md), but with two classes now. Each line: `<class_id> <x1_norm> <y1_norm> ...` where `class_id ∈ {0, 1}`.

### 4.3 Split-manifest-driven copy

Same 70/20/10 resplit manifest as Phase A and B (D024). Files placed into `data/yolo_multiclass/images/{train,val,test}/`.

### 4.4 Canopy-state tags

Same `canopy_state_map.json` structure as Phase B.

### 4.5 Acceptance criteria

- Trunk annotations produce lines with `class_id = 0`; pole annotations produce lines with `class_id = 1`
- No annotation lines for pipe, building, robot, vehicle
- Split assignments identical to Phase B (same resplit manifest)
- Spot-check confirms class colours match ground truth on 3–5 samples per canopy state

## 5. Data config YAML (`phase_c_yolo_multiclass_data.yaml`)

```yaml
path: /workspaces/dissertation/vineyard_nav/data/yolo_multiclass
train: images/train
val: images/val
test: images/test

nc: 2
names:
  0: trunk
  1: pole
```

## 6. Training (`train.py`)

Structurally identical to Phase B (Section 6 of PHASE_B_SPEC.md), with only the data config and `nc` changing. Same hyperparameters, same augmentations, same model size (`yolo11n-seg.pt`).

The identical training regime across Phases B and C is what makes the B ↔ C comparison a controlled experiment isolating class structure.

Training config saved to `configs/phase_c_yolo_multiclass_train.yaml`.

## 7. Downstream geometry modules

### 7.1 Per-side clustering (`clustering.py`)

Two input paths converge on the same output:

- **From YOLO detections:** each instance has a centroid (from its polygon mask). Take all centroids of relevant class(es) per config. Split into left/right sides by image column (x-coordinate ≶ image_width / 2, or by another rule if refinement needed).
- **From U-Net binary mask (Phase A):** foreground pixels clustered via connected components; centroid of each component is treated as an instance centroid. Same left/right split rule applies.

Output: `{"left": [(x, y), ...], "right": [(x, y), ...]}` for use by RANSAC.

### 7.2 RANSAC line fitting (`ransac.py`)

Standard OpenCV or scikit-learn RANSAC with a linear model fitting `y = mx + c` (or parametric line if near-vertical). Returns `(m, c)` per side plus inlier count.

Robust to outliers via RANSAC. Threshold and max_trials as hyperparameters — locked to defensible defaults; documented if tuned.

### 7.3 Centreline (`centreline.py`)

Given left-side line `(m_L, c_L)` and right-side line `(m_R, c_R)`, compute centreline as the bisector — parametric line whose direction is the average of left and right unit direction vectors, and whose position is the average of left and right x-positions at a fixed reference y (e.g. y = image_height / 2).

Output: `(m_center, c_center)` plus a confidence signal (e.g., product of left and right inlier counts).

### 7.4 Config dispatch (`configs.py`)

Three functions with identical signature:

```python
def apply_config(detections, config, T=None):
    """
    detections: list of (class_id, centroid_x, centroid_y)
    config: "A" (trunk primary), "B" (pole primary), "C" (class-agnostic)
    T: int or None (only used for A and B)
    returns: {"left": [...], "right": [...]} — points ready for RANSAC
    """
```

Behaviour:
- **Config A:** per side, if trunk count ≥ T, use trunk detections only. Else use trunk + pole combined.
- **Config B:** per side, if pole count ≥ T, use pole detections only. Else use trunk + pole combined.
- **Config C:** always use trunk + pole combined. `T` ignored.

## 8. Downstream sweep (`downstream_sweep.py`)

### 8.1 Sweep grid

- Configs: `["A", "B", "C"]`
- T values: `[1, 2, 3, 5, 8, 12]` for A and B; N/A for C
- Total sweep cells: 6 + 6 + 1 = 13

### 8.2 Procedure

For each val image:
1. Run Phase C YOLO inference — get detections `[(class_id, x, y), ...]`
2. For each (config, T) cell in the sweep grid:
   a. Apply config to get `{"left": [...], "right": [...]}`
   b. Fit RANSAC per side → left line, right line
   c. Compute centreline
   d. Compute frame-level geometric metric (RMS lateral error to teleoperator trajectory reference)
3. Aggregate per (config, T) cell across all val images: mean RMS lateral error, canopy-stratified

Output: `sweep_results.csv` with columns `config, T, canopy_state, n_frames, mean_rms_lateral_error, ci_low, ci_high`.

### 8.3 Selection rule

`(config*, T*) = argmin over (config, T) of mean RMS lateral error on val (overall, both canopy bins pooled)`

Locked in `selected_config.json`:

```json
{
  "config": "A",
  "T": 3,
  "val_metric": 0.XXX,
  "selection_datetime": "..."
}
```

### 8.4 Sensitivity plots

Two plots saved to `sensitivity_plots/`:
1. Mean RMS lateral error vs T, one line per config (A, B), horizontal reference for C — overall
2. Same plot but stratified by canopy state (bare-vine, canopy)

### 8.5 Acceptance criteria

- All 13 cells evaluated on val
- Locked config recorded
- Sensitivity plots produced
- Discussion in DECISIONS.md addresses: does the multiclass advantage (if any) depend sensitively on T, or hold across the range?

## 9. Test evaluation (`phase_c_test.py`)

### 9.1 Behaviour

- Loads Phase C YOLO `best.pt`
- Loads `selected_config.json` (locked `(config*, T*)`)
- Runs test-set inference and downstream at locked config only
- Computes:
  - Perception metrics: mAP@50, precision, recall (overall + canopy-stratified)
  - Geometric metrics: RMS lateral error, heading error, success rate (overall + canopy-stratified, with bootstrap CIs)
  - Command-level metrics: PID smoothness (RMS yaw-rate diff, jitter, saturation rate)
- Writes `test_metrics.json`

### 9.2 Guardrails

- Test evaluated **once** at locked `(config*, T*)`
- No iteration on test — surprising metrics are discussed, not re-selected

## 10. Attribution analysis (`attribution.py`)

### 10.1 Purpose

Given test metrics from Phases A, B, and C (at locked config for C), produce the 3-way comparison narrative:

1. **A ↔ B:** architecture effect. Does modernising architecture change binary performance? Report ΔRMS lateral error with bootstrap CI and effect size.
2. **B ↔ C:** class-structure effect. Does class-aware pipeline improve over class-agnostic YOLO? Report ΔRMS lateral error with bootstrap CI and effect size.
3. **B ↔ C-configC (from sweep):** does the class-aware downstream logic itself matter, or does the multiclass model just detect better? Report ΔRMS lateral error between YOLO-binary and multiclass-detected-but-class-agnostic-processed.

Attribution paragraph written to `attribution_summary.md` and appended to DECISIONS.md.

### 10.2 Statistical framework

- Bootstrap over per-frame metric differences (10,000 bootstrap samples)
- 95% CIs reported
- Cohen's d as the effect size for standardised comparison
- Canopy-stratified reporting

## 11. Reproducibility

- Seeds set (torch, numpy, random) for YOLO training and downstream sweep
- Git commit hash in every run directory
- Sweep config snapshot preserved
- `selected_config.json` records the locked configuration exactly

## 12. Optional supplementary (only if time)

**All-6-classes multiclass experiment (O002 in DECISIONS.md):** train an additional YOLOv11-seg with all 6 SemanticBLT classes (trunk, pole, pipe, building, robot, vehicle). Compare downstream metrics against the 2-class primary Phase C. Report as one paragraph + one figure in Results. Does the richer supervision transfer back to trunk/pole detection quality that matters for centreline fitting?

## 13. Definition of "Phase C complete"

- [ ] YOLOv11-seg multiclass trained; `best.pt` locked
- [ ] Downstream sweep completed on val; sensitivity plots produced; `(config*, T*)` locked in `selected_config.json`
- [ ] Test evaluated **once** at locked config; `test_metrics.json` written
- [ ] Attribution analysis produced; `attribution_summary.md` written
- [ ] DECISIONS.md updated with final locked config, test metrics, attribution paragraph
- [ ] STATUS.md phase tracker updated

When all six checked, dissertation writing phase can consume results.
