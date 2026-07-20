# Perception strand — scripts

Everything needed to reproduce the three-arm perception comparison (Phase A U-Net binary,
Phase B YOLO binary, Phase C YOLO multiclass). The flat `.py` files here are the
**reproduction pipeline**; `diagnostics/` holds one-time troubleshooting you can skip.
`segmentation/` is the importable model package.

**Run every command from `vineyard_nav/`.** Training and evaluation use `python -m` module
invocation — the package path moved (`segmentation.…` → `scripts.perception.segmentation.…`).

## Reproduce, in order

**1. Scene-level split** — splits by *scene*, not by image, so augmented copies of the same scene
cannot land on both sides of the split and inflate the scores
→ writes `data/splits/resplit_70_20_10.json`
```bash
python3 scripts/perception/resplit_dataset.py
```

**2. COCO → YOLO labels** — consumes step 1's split manifest
```bash
python3 scripts/perception/coco_to_yolo.py --mode binary
python3 scripts/perception/coco_to_yolo.py --mode multiclass
```

**3. Train each arm** — smoke run first (2 epochs), then the full run. Repeat per seed with
the `_seed43` / `_seed44` configs.
```bash
python3 -m scripts.perception.segmentation.unet_binary.train --smoke
python3 -m scripts.perception.segmentation.unet_binary.train --config configs/phase_a_unet_binary.yaml
python3 -m scripts.perception.segmentation.yolo_binary.train     --config configs/phase_b_yolo_binary_train.yaml
python3 -m scripts.perception.segmentation.yolo_multiclass.train --config configs/phase_c_yolo_multiclass_train.yaml
```
Each run writes to `results/runs/<experiment>_<timestamp>/` (gitignored).

**4. Evaluate** — `--run-dir` is required. The **test split is evaluated once per arm** (working
rule 5); use `--split val` while iterating.
```bash
python3 -m scripts.perception.segmentation.unet_binary.evaluate --run-dir results/runs/<run> --split test
python3 -m scripts.perception.segmentation.yolo_binary.evaluate --run-dir results/runs/<run> --split test
```
Also emits `test_per_frame_metrics.csv`, which step 6 consumes.

**5. Operating point** — selects `conf* = 0.25` for the YOLO arms
```bash
python3 scripts/perception/phase_b_conf_sweep.py
```

**6. Bootstrap confidence intervals** — resamples the test scenes to put an uncertainty range on
each score, so small differences between arms are not over-read. Consumes the per-frame CSV
from step 4.
```bash
python3 scripts/perception/bootstrap.py \
  --per-frame-csv results/runs/<run>/test_per_frame_metrics.csv \
  --output        results/runs/<run>/test_bootstrap_ci.json
```

## Also reusable (cited by the write-up, run as needed)

- `median_conf_sweep.py` — repeats the step-5 sweep using the median rather than the mean per-frame
  score, so the operating point is not driven by a few catastrophic frames.
- `blob_overlap_6799.py --runs <run> <run> …` — on one canopy test scene, several YOLO runs emit a
  single huge false mask covering most of the frame. This regenerates the overlap comparison showing
  those failures land in near-identical places across seeds and arms, i.e. it is a reproducible
  architecture/scene interaction rather than a random bad run.

## `segmentation/` (package, not scripts)

Model / dataset / loss / metric modules plus the `train` · `evaluate` · `visualize` entry points
for all three arms. Import as `from scripts.perception.segmentation.<arm>.<module> import …`;
invoke as `python -m scripts.perception.segmentation.<arm>.<entry>`.

## Downstream

The geometric strand consumes the trained checkpoints (see `scripts/geometric/README.md`); the
control strand consumes the geometric centreline (see `scripts/control/README.md`).
