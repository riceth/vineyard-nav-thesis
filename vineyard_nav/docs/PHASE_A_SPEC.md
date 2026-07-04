# PHASE_A_SPEC.md

**Scope:** Implementation contract for Phase A — U-Net binary baseline (SMP + ImageNet pretrained encoder).
**Audience:** Claude Code (and any future implementer). Read fully before writing code.
**Status:** Locked.
**Role in three-arm design:** Official baseline representing de Silva 2024's binary-mask paradigm.

---

## 1. Goal

Fine-tune an SMP U-Net with an ImageNet-pretrained ResNet-34 encoder on SemanticBLT's binary-collapsed labels (trunk + pole → foreground). Produce a locked best checkpoint, three-strand evaluation metrics, and canopy-stratified reports.

Phase A does **not** involve multiclass training, YOLO, or class-aware downstream logic.

## 2. Folder structure (within `/workspaces/dissertation/vineyard_nav/`)

```
segmentation/unet_binary/
├── __init__.py
├── dataset.py             # SemanticBLT binary dataset class
├── model.py               # SMP U-Net wrapper
├── losses.py              # BCE + Dice combined loss
├── metrics.py             # mIoU, per-class IoU, precision, recall, F1
├── train.py               # Training loop entry point
├── evaluate.py            # Test-set evaluation entry point
├── inference.py           # Single-image / split inference utility
└── visualize.py           # Mask overlay rendering

configs/
└── phase_a_unet_binary.yaml    # All Phase A hyperparameters

results/runs/
└── phase_a_unet_binary_<timestamp>/
    ├── checkpoints/
    │   ├── best.pt        # Best val mIoU
    │   └── final.pt       # Last epoch
    ├── tensorboard/       # Event files
    ├── metrics.csv        # Per-epoch dump
    ├── config_snapshot.yaml
    ├── git_commit.txt
    └── predictions/       # Test-set visualisations
```

## 3. Implementation order (mandatory)

1. **`dataset.py`** — including visualisation method
2. Visual spot-check on 3–5 samples per canopy state. Stop. Confirm before continuing.
3. **`model.py`** — SMP wrapper + smoke test (random input → expected output shape, parameter count)
4. **`losses.py`, `metrics.py`** — with unit tests on synthetic input
5. **`train.py`** — 2-epoch smoke run on subset first
6. **`evaluate.py`, `visualize.py`** — run on best checkpoint
7. Test metrics recorded in DECISIONS.md

## 4. Dataset module (`dataset.py`)

### 4.1 Class signature

```python
class SemanticBLTBinaryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: str,                    # /workspaces/dissertation/vineyard_nav/data/semanticblt
        split: str,                   # "train" | "valid" | "test" (from resplit manifest)
        split_manifest: str,          # path to 70/20/10 resplit manifest
        transform: Optional[A.Compose] = None,
    ): ...
    def __len__(self) -> int: ...
    def __getitem__(self, idx: int) -> dict:
        # returns {
        #   "image": Tensor[3, H, W] float32 normalised,
        #   "mask":  Tensor[H, W]    int64 in {0, 1},
        #   "canopy_state": str,   # "bare_vine" | "canopy"
        #   "image_id": int,
        #   "filename": str,
        # }
```

### 4.2 Label-collapsing rule (LOCKED)

For each pixel, compute the binary mask from COCO polygon annotations:
- Foreground (`mask == 1`) = pixels belonging to any annotation with `category_id ∈ {3, 5}` (pole, trunk)
- Background (`mask == 0`) = everything else, including pipe (cat 2), building (cat 1), robot (cat 4), vehicle (cat 6), and unannotated pixels

Use `pycocotools` to rasterise polygons. Where multiple foreground annotations overlap, foreground wins.

### 4.3 Canopy-state parsing (LOCKED)

Parse from filename prefix:
- `march_*` or `april_*` → `"bare_vine"`
- `may_*` or `color_image_*` → `"canopy"`

Anything else → raise `ValueError`. No silent default.

### 4.4 Augmentations (train split only)

```python
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
```

Val/test get only normalisation + `ToTensorV2`. No vertical flip, no cropping.

### 4.5 Visualisation

`visualize.py` renders foreground mask as semi-transparent red overlay on RGB image. Saves to `results/runs/<...>/dataset_spotcheck/`. Manual verification gate before model code.

### 4.6 Acceptance criteria

- Returns correct shapes and types as specified
- Canopy-state counts match resplit manifest expectations
- Manual spot-check confirms foreground covers trunk + pole regions, excludes pipes
- No augmentation applied to val/test
- Deterministic under fixed seeds

## 5. Model module (`model.py`)

### 5.1 Architecture (LOCKED)

Use `segmentation_models_pytorch`:

```python
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=2,             # 2-channel softmax output (see D007)
    activation=None,       # raw logits
)
```

Encoder: ResNet-34 pretrained on ImageNet. Decoder: SMP U-Net default. Output: `[B, 2, H, W]` raw logits.

### 5.2 Forward signature

```python
def forward(self, x: Tensor) -> Tensor:
    # x: [B, 3, H, W]
    # returns: [B, 2, H, W] — raw logits
```

Spatial dimensions preserved. For 640×640 input, output is 640×640.

### 5.3 Smoke test

`__main__` block:
- Instantiates model
- Runs random `(2, 3, 640, 640)` through forward
- Asserts output shape `(2, 2, 640, 640)`
- Prints total parameter count and trainable parameter count

## 6. Loss module (`losses.py`)

Combined loss = `0.5 * CE + 0.5 * Dice`.

- CE: `nn.CrossEntropyLoss` over 2-channel logits, no class weighting
- Dice: Generalised soft Dice across both classes, computed on softmax probabilities

Unit tests on synthetic input: perfect prediction → loss ≈ 0; uniform prediction → loss > 0.

## 7. Metrics module (`metrics.py`)

Per-batch confusion-matrix accumulation, aggregated over full split, reported at epoch end.

- Per-class IoU (background, foreground)
- Mean IoU (mIoU)
- Per-class precision, recall, F1

Unit test on synthetic predictions with known IoU.

## 8. Training module (`train.py`)

### 8.1 Hyperparameters (LOCKED, in `phase_a_unet_binary.yaml`)

```yaml
experiment_name: phase_a_unet_binary
seed: 42

data:
  root: /workspaces/dissertation/vineyard_nav/data/semanticblt
  split_manifest: /workspaces/dissertation/vineyard_nav/data/splits/resplit_70_20_10.json
  image_size: 640
  num_workers: 4

model:
  arch: unet
  encoder: resnet34
  encoder_weights: imagenet
  num_classes: 2

loss:
  ce_weight: 0.5
  dice_weight: 0.5

optim:
  optimizer: adam
  lr: 1.0e-4
  weight_decay: 1.0e-5

schedule:
  type: cosine_annealing
  t_max: 60
  eta_min: 1.0e-6

train:
  epochs: 60                     # reduced from 100 because pretrained converges faster
  batch_size: 8
  grad_accumulation_steps: 1     # bump to 2 (batch_size to 4) if OOM
  amp: true
  early_stopping:
    metric: val_miou
    mode: max
    patience: 10

logging:
  tensorboard: true
  csv: true
  log_predictions_every_n_epochs: 5
  prediction_subset_size: 8

reproducibility:
  deterministic: true
  benchmark: false
```

### 8.2 Training loop responsibilities

- Seed PyTorch, NumPy, Python `random` before any data loading
- Instantiate `GradScaler` **once** before training starts; persist across all epochs; include `scaler.state_dict()` in checkpoints
- Per-epoch: train pass → val pass → log → checkpoint if best
- Mixed precision: `autocast` for forward + loss, `scaler.scale(loss).backward()`, `scaler.step(optimizer)`, `scaler.update()`
- Gradient accumulation: if `grad_accumulation_steps > 1`, scale loss by `1 / steps`, `optimizer.step()` every N micro-batches
- Checkpoint schema: `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `epoch`, `val_miou`, `config`, `git_commit`
- Per-epoch CSV row: `epoch, train_loss, val_loss, val_miou, val_iou_foreground, val_iou_background, val_precision, val_recall, val_f1, lr`
- TensorBoard: scalars (losses, metrics, lr), images (sample predictions every 5 epochs)

### 8.3 Acceptance criteria

- 2-epoch smoke run on subsample (50 train + 10 val) completes without OOM
- Full run produces `best.pt`, `metrics.csv`, TensorBoard logs, config snapshot, git commit hash
- Training resumable from any checkpoint (model + optimizer + scheduler + scaler + RNG state)
- Two runs with same config and seed produce identical metrics (within float tolerance)

## 9. Evaluation module (`evaluate.py`)

### 9.1 Behaviour

- Loads `best.pt` from a specified run directory
- Runs inference on test split (once, at end of Phase A)
- Computes perception metrics overall and stratified by canopy state
- Saves prediction visualisations for every test image
- Writes `test_metrics.json`:

```json
{
  "overall":   {"miou": ..., "iou_foreground": ..., "iou_background": ...,
                "precision_foreground": ..., "recall_foreground": ..., "f1_foreground": ...},
  "bare_vine": { ... },
  "canopy":    { ... },
  "n_frames":  {"overall": ..., "bare_vine": ..., "canopy": ...}
}
```

### 9.2 Test-set guardrails

- `evaluate.py` runs on test split **once**, after training complete and `best.pt` locked
- No iteration on test results — surprising metrics are discussed, not re-hyperparameter-tuned then re-evaluated
- Validation set is for all iteration

## 10. Reproducibility (LOCKED)

- Seeds set: torch, numpy, random, cuda
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `requirements.txt` versions pinned
- Git commit hash in every checkpoint and every run's `git_commit.txt`

## 11. Out of scope for Phase A (do not implement)

- YOLO training (Phases B and C)
- Multiclass anything (Phase C)
- Class-aware downstream logic (Phase C)
- Downstream geometry stage — comes after all three arms trained; separate spec
- Offline PID — separate spec
- Geometric or command-level evaluation — separate spec

## 12. Definition of "Phase A complete"

- [ ] U-Net binary trained; `best.pt` locked
- [ ] Test evaluated once; `test_metrics.json` written
- [ ] Per-epoch `metrics.csv` and TensorBoard logs preserved
- [ ] Prediction visualisations saved for all test frames
- [ ] DECISIONS.md updated with final hyperparameters and test metrics summary (O003)
- [ ] STATUS.md phase tracker updated

When all six checked, Phase B can begin.
