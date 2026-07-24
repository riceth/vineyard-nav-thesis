# Perception strand — scripts

Everything needed to reproduce the three-arm perception comparison: Phase A
(U-Net binary), Phase B (YOLO binary), Phase C (YOLO multiclass). The flat `.py`
files here are the reproduction pipeline; `diagnostics/` holds one-time
troubleshooting you can skip; `segmentation/` is the importable model package.

> **⚠️ Not re-verified in the current multi-bag session.** Perception training was
> completed in earlier phases of the project (the resulting test metrics are
> recorded in `docs/DECISIONS.md`, O003, and `docs/FINDINGS.md`, F001–F009). It
> was **not** re-run during the March/April multi-bag work, so — unlike the
> geometric and control strand READMEs — the commands below are **not** annotated
> with freshly observed runtimes or "you should see" output. They are the
> documented procedure that produced the committed checkpoints; treat runtimes as
> unmeasured here. Every path and config named below has been confirmed to exist
> in the current repository, but the end-to-end run has not been repeated.

---

## Before you start

**Be in the right directory** — every command runs from `vineyard_nav/`:

```bash
cd vineyard_nav
pwd     # must end in /vineyard_nav
```

**What this strand needs:**
- The **SemanticBLT dataset** at the repo root (`SemanticBLT.v1-2024-june.coco-segmentation/`);
  see the top-level `README.md` for where to obtain it.
- A **CUDA GPU** (the project used an RTX 5050, 8 GB; PyTorch 2.11 + CUDA 12.8).
- No ROS bag is needed — perception trains and evaluates on the labelled dataset,
  not on the bags. The bags are consumed only by the *downstream* geometric strand.

**Invocation note:** the flat scripts run as `python3 scripts/perception/<name>.py`;
the model package runs as `python -m scripts.perception.segmentation.<arm>.<entry>`
(the package path is `scripts.perception.segmentation.…` after the D045 reorg).

---

## Reproduce, in order

Each step names the file it needs from the previous step.

**1. Scene-level split** — splits by *scene*, not by image, so augmented copies
of one scene cannot land on both sides and inflate scores.
```bash
python3 scripts/perception/resplit_dataset.py
```
- **Produces:** `data/splits/resplit_70_20_10.json` (this manifest **is**
  committed, so steps that only consume it can run without re-splitting).

**2. COCO → YOLO labels** — converts the dataset annotations for the YOLO arms.
```bash
python3 scripts/perception/coco_to_yolo.py --mode binary
python3 scripts/perception/coco_to_yolo.py --mode multiclass
```
- **Needs:** the split manifest from step 1.

**3. Train each arm — all three seeds.** The study uses seeds 42/43/44 (nine
checkpoints total: 3 arms × 3 seeds; multi-seed rationale in O009). Smoke-run
first (2 epochs) to catch config errors cheaply, then the full runs:
```bash
# smoke check (Phase A shown; do the same per arm if desired)
python3 -m scripts.perception.segmentation.unet_binary.train --smoke

# seed 42 (the base configs)
python3 -m scripts.perception.segmentation.unet_binary.train     --config configs/phase_a_unet_binary.yaml
python3 -m scripts.perception.segmentation.yolo_binary.train     --config configs/phase_b_yolo_binary_train.yaml
python3 -m scripts.perception.segmentation.yolo_multiclass.train --config configs/phase_c_yolo_multiclass_train.yaml

# seeds 43 and 44 (the *_seed43 / *_seed44 config variants)
python3 -m scripts.perception.segmentation.unet_binary.train     --config configs/phase_a_unet_binary_seed43.yaml
python3 -m scripts.perception.segmentation.unet_binary.train     --config configs/phase_a_unet_binary_seed44.yaml
python3 -m scripts.perception.segmentation.yolo_binary.train     --config configs/phase_b_yolo_binary_seed43_train.yaml
python3 -m scripts.perception.segmentation.yolo_binary.train     --config configs/phase_b_yolo_binary_seed44_train.yaml
python3 -m scripts.perception.segmentation.yolo_multiclass.train --config configs/phase_c_yolo_multiclass_seed43_train.yaml
python3 -m scripts.perception.segmentation.yolo_multiclass.train --config configs/phase_c_yolo_multiclass_seed44_train.yaml
```
- **Needs:** the dataset (all arms) and the YOLO labels from step 2 (B and C).
- **Produces:** one `results/runs/<experiment>_<timestamp>/` per run (gitignored),
  each recording the git commit it was trained from. **These nine directories are
  the model weights** the geometric strand later consumes.

**4. Evaluate** — `--run-dir` is required. The **test split is evaluated once per
arm** (working rule 5); use `--split val` while iterating.
```bash
python3 -m scripts.perception.segmentation.unet_binary.evaluate --run-dir results/runs/<run> --split test
python3 -m scripts.perception.segmentation.yolo_binary.evaluate --run-dir results/runs/<run> --split test
python3 -m scripts.perception.segmentation.yolo_multiclass.evaluate --run-dir results/runs/<run> --split test
```
- **Needs:** a trained `<run>` directory from step 3.
- **Produces:** the run's test metrics + `test_per_frame_metrics.csv` (consumed by
  step 6).

**5. Operating point** — selects `conf* = 0.25` for the YOLO arms.
```bash
python3 scripts/perception/phase_b_conf_sweep.py
```

**6. Bootstrap confidence intervals** — resamples the test scenes to put an
uncertainty range on each score, so small arm differences are not over-read.
```bash
python3 scripts/perception/bootstrap.py \
  --per-frame-csv results/runs/<run>/test_per_frame_metrics.csv \
  --output        results/runs/<run>/test_bootstrap_ci.json
```
- **Needs:** the per-frame CSV from step 4.

---

## Also reusable (cited by the write-up, run as needed)

- `median_conf_sweep.py` — repeats the step-5 sweep using the median rather than
  the mean per-frame score, so the operating point is not driven by a few
  catastrophic frames.
- `blob_overlap_6799.py --runs <run> <run> …` — on one canopy test scene several
  YOLO runs emit a single huge false mask covering most of the frame; this
  regenerates the overlap comparison showing those failures land in near-identical
  places across seeds and arms (a reproducible architecture/scene interaction, not
  a random bad run). This is finding F007.

## `segmentation/` (package, not scripts)

Model / dataset / loss / metric modules plus the `train` · `evaluate` ·
`visualize` entry points for all three arms. Import as
`from scripts.perception.segmentation.<arm>.<module> import …`; invoke as
`python -m scripts.perception.segmentation.<arm>.<entry>`.

## Downstream

The geometric strand consumes the trained checkpoints (`scripts/geometric/README.md`);
the control strand consumes the geometric centreline (`scripts/control/README.md`).
The nine checkpoints are gitignored — a reader who only wants the committed
*results* does not need to retrain; see the top-level `README.md` routing table.
