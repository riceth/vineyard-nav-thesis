# PHASE_B_SPEC.md

**Scope:** Implementation contract for Phase B — YOLOv11-seg binary baseline.
**Audience:** Claude Code (and any future implementer). Read fully before writing code.
**Status:** Locked.
**Role in three-arm design:** Modernised binary baseline. Comparison against Phase C **isolates the class-structure effect** — same backbone, hyperparameters, data and augmentation, differing only in label granularity. Comparison against Phase A is a **baseline-versus-modernised-pipeline contrast, not a controlled architecture comparison**: the two arms differ in at least thirteen respects (see the correction at D006/D021), so no architecture-attributable claim may rest on it.

---

## 1. Goal

Fine-tune YOLOv11-seg from COCO-pretrained weights on SemanticBLT's binary-collapsed labels (trunk + pole → single class). Produce a locked best model, three-strand evaluation metrics, and canopy-stratified reports.

Phase B does **not** involve multiclass training or class-aware downstream logic.

## 2. Folder structure (within `/workspaces/dissertation/vineyard_nav/`)

```
segmentation/yolo_binary/
├── __init__.py
├── data_prep.py           # COCO → YOLO binary labels conversion
├── train.py               # ultralytics YOLO training entry point
├── evaluate.py            # Test-set evaluation entry point
├── inference.py           # Single-image / split inference utility
└── visualize.py           # Detection overlay rendering

configs/
├── phase_b_yolo_binary_data.yaml    # YOLO data configuration
└── phase_b_yolo_binary_train.yaml   # Training hyperparameters

data/yolo_binary/
├── images/
│   ├── train/             # symlinks or copies from data/semanticblt
│   ├── val/
│   └── test/
└── labels/
    ├── train/             # .txt files, one per image, YOLO format
    ├── val/
    └── test/

results/runs/
└── phase_b_yolo_binary_<timestamp>/
    ├── weights/
    │   ├── best.pt        # Best val mAP
    │   └── last.pt        # Last epoch
    ├── results.csv        # Per-epoch dump (ultralytics default)
    ├── config_snapshot.yaml
    ├── git_commit.txt
    ├── args.yaml          # ultralytics training args
    └── predictions/       # Test-set visualisations
```

## 3. Implementation order (mandatory)

1. **`data_prep.py`** — COCO → YOLO conversion, binary class collapse
2. **Spot check** — verify YOLO labels visually on 3–5 samples per canopy state
3. **Data config YAML** — `data.yaml` for ultralytics
4. **`train.py`** — 2-epoch smoke run first
5. **`evaluate.py`, `visualize.py`** — run on best weights
6. Test metrics recorded in DECISIONS.md

## 4. Data preparation (`data_prep.py`)

### 4.1 Two paths, pick one (see O005 in DECISIONS.md)

**Path A: Roboflow re-export** — download SemanticBLT again from Roboflow in YOLOv11 segmentation format. Then re-apply 70/20/10 stratified resplit manifest to the exported labels.

**Path B: In-place COCO → YOLO conversion** — write a script that reads the existing COCO JSON, converts polygons to YOLO segmentation format, and writes `.txt` label files.

Path B is preferred for reproducibility (single source of truth for labels) and control. Spec below assumes Path B.

### 4.2 YOLO segmentation label format

One `.txt` file per image, one line per instance:

```
<class_id> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ... <xn_norm> <yn_norm>
```

- All coordinates normalised to `[0, 1]` by dividing by image width/height
- At least 3 points per polygon
- **Binary class collapse:** for Phase B, every foreground annotation gets `class_id = 0`. Annotations with COCO `category_id ∈ {3, 5}` (pole, trunk) become foreground; other classes and unannotated regions become background (absence of any label).

### 4.3 Split-manifest-driven copy

Files placed into `data/yolo_binary/images/{train,val,test}/` according to the 70/20/10 resplit manifest (D024). Labels placed correspondingly into `data/yolo_binary/labels/{train,val,test}/`.

### 4.4 Canopy-state tags

Because ultralytics doesn't natively handle stratified evaluation, canopy-state labels are stored separately in `data/yolo_binary/canopy_state_map.json`:

```json
{"march_0007.jpg": "bare_vine", "may_0012.jpg": "canopy", ...}
```

Loaded by `evaluate.py` for post-hoc stratification.

### 4.5 Acceptance criteria

- Every foreground annotation becomes a valid YOLO polygon line
- All non-trunk/pole annotations are dropped (no lines produced for them)
- Normalisation correct: no coordinates outside [0, 1]
- Split assignments match resplit manifest exactly
- Canopy-state map has entries for all images across all splits
- Spot-check visualisation confirms labels correctly overlaid on images

## 5. Data config YAML (`phase_b_yolo_binary_data.yaml`)

```yaml
# ultralytics data configuration
path: /workspaces/dissertation/vineyard_nav/data/yolo_binary
train: images/train
val: images/val
test: images/test

nc: 1              # single foreground class
names:
  0: crop          # trunk + pole collapsed
```

## 6. Training (`train.py`)

### 6.1 Approach

Use ultralytics `YOLO` class. Load pretrained `yolo11n-seg.pt` (nano) as starting point. Fine-tune on Phase B data.

Rationale for `nano`: fastest training, smallest checkpoint, sufficient capacity for a 2-class-if-binary / 3-class-if-multiclass problem on 725 training images. If validation metrics indicate under-capacity, escalate to `yolo11s-seg.pt` (small) — flag before doing so.

### 6.2 Hyperparameters (LOCKED, in `phase_b_yolo_binary_train.yaml`)

```yaml
experiment_name: phase_b_yolo_binary
seed: 42

model: yolo11n-seg.pt

data: configs/phase_b_yolo_binary_data.yaml

train:
  epochs: 100
  imgsz: 640
  batch: 16                # ultralytics default; adjust if OOM
  device: 0
  workers: 4
  patience: 30             # ultralytics early stopping patience
  save: true
  save_period: 10
  project: results/runs
  name: phase_b_yolo_binary
  exist_ok: false
  pretrained: true
  optimizer: SGD           # ultralytics default
  lr0: 0.01                # ultralytics default
  lrf: 0.01                # final lr = lr0 * lrf
  momentum: 0.937
  weight_decay: 0.0005
  warmup_epochs: 3
  amp: true

augmentation:
  # ultralytics defaults with vertical flip disabled and rotation-agnostic settings
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 10.0            # small rotations to match Phase A
  translate: 0.1
  scale: 0.5
  shear: 0.0
  flipud: 0.0              # NO vertical flip (rows have orientation)
  fliplr: 0.5              # horizontal flip
  mosaic: 1.0              # ultralytics default
  mixup: 0.0
  copy_paste: 0.0

reproducibility:
  deterministic: true
```

### 6.3 Training entry point

```python
from ultralytics import YOLO
import yaml, subprocess, shutil, os

# Load config
with open("configs/phase_b_yolo_binary_train.yaml") as f:
    cfg = yaml.safe_load(f)

# Seed + reproducibility
import torch, numpy, random
torch.manual_seed(cfg["seed"])
numpy.random.seed(cfg["seed"])
random.seed(cfg["seed"])

# Load model
model = YOLO(cfg["model"])

# Train
results = model.train(**cfg["train"], **cfg["augmentation"])

# Record git commit
commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
run_dir = f"{cfg['train']['project']}/{cfg['train']['name']}"
with open(f"{run_dir}/git_commit.txt", "w") as f:
    f.write(commit)
shutil.copy("configs/phase_b_yolo_binary_train.yaml", f"{run_dir}/config_snapshot.yaml")
```

### 6.4 Acceptance criteria

- 2-epoch smoke run on subsample completes without OOM
- Full run produces `weights/best.pt`, `results.csv`, config snapshot, git commit hash
- ultralytics `args.yaml` present in run directory (auto-produced)
- Two runs with same config and seed produce closely matching metrics (ultralytics is not fully deterministic; small variation acceptable, document if present)

## 7. Evaluation module (`evaluate.py`)

### 7.1 Behaviour

- Loads `weights/best.pt` from specified run directory
- Runs inference on test split (once, at end of Phase B)
- Uses ultralytics `model.val()` on the test split via a temporary data.yaml with test as val
- Additionally, computes canopy-stratified metrics using `canopy_state_map.json`
- Saves prediction visualisations for every test image
- Writes `test_metrics.json`:

```json
{
  "overall":   {"map50": ..., "map50_95": ..., "precision": ..., "recall": ...},
  "bare_vine": { ... },
  "canopy":    { ... },
  "n_frames":  {"overall": ..., "bare_vine": ..., "canopy": ...}
}
```

### 7.2 Test-set guardrails

- Test evaluated **once**, after `best.pt` locked
- No iteration on test results

## 8. Reproducibility

- Seeds set (torch, numpy, random)
- ultralytics `deterministic=True` (not fully deterministic — document any variation)
- Git commit hash written into run directory
- Config snapshot preserved
- ultralytics `args.yaml` auto-preserved

## 9. Out of scope for Phase B

- Multiclass training (Phase C)
- Class-aware downstream logic (Phase C)
- Downstream geometry stage — separate spec, comes after all three arms trained
- Offline PID — separate spec
- Geometric or command-level evaluation — separate spec

## 10. Definition of "Phase B complete"

- [ ] YOLOv11-seg binary trained; `best.pt` locked
- [ ] Test evaluated once; `test_metrics.json` written
- [ ] `results.csv` (per-epoch dump) preserved
- [ ] Prediction visualisations saved for all test frames
- [ ] DECISIONS.md updated with final hyperparameters and test metrics summary
- [ ] STATUS.md phase tracker updated

When all six checked, Phase C can begin.
