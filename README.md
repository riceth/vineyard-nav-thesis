# Vineyard Navigation — MSc Dissertation

**Multiclass Instance Segmentation for In-Row Vineyard Navigation: A Controlled
Comparison Against the Binary-Mask Baseline.**

**Author:** Edosa Ebohon (30436293) · MSc Robotics and Artificial Intelligence,
University of Lincoln · CMP9140 Research Project · A2 submission 26 August 2026.

📄 **[Read the dissertation (PDF)](MSc-Dissertation.pdf)** — work in progress, updated as it is written.

This repository compares three perception formulations for estimating the row
centreline a vineyard robot would steer down, under a controlled experiment that
isolates one variable at a time:

| Arm | Model | Class structure | Role |
|---|---|---|---|
| A | U-Net (ImageNet-pretrained ResNet-34) | Binary (trunk+pole → foreground) | Baseline (de Silva 2024 paradigm) |
| B | YOLOv11-seg | Binary | Modernised binary baseline |
| C | YOLOv11-seg | Multiclass (trunk, pole distinct) | The contribution |

**B-vs-C is the controlled comparison** — same backbone, hyperparameters, data and
augmentation, differing only in label granularity. A-vs-B is *not* an architecture
comparison: the two arms differ in at least thirteen respects, so it is reported as a
baseline-versus-modernised-pipeline contrast and no architecture claim rests on it.
Each arm is evaluated at three levels — perception, geometric (centreline), and
command (steering) — across multiple seasonal recordings.

**Headline result so far:** across two seasons (March + April), the three arms
are statistically indistinguishable at the navigation output — both the
geometric centreline error and the steering-command smoothness — so the
binary-vs-multiclass distinction does not reach the robot's behaviour on this
data. See `vineyard_nav/docs/FINDINGS.md`.

---

## ⚠️ Where to run everything — read this first

**Every command in this project runs from the `vineyard_nav/` directory**, which
sits inside the repository you just cloned. Before running anything:

```bash
cd vineyard_nav
pwd     # must end in /vineyard_nav
```

If `pwd` does not end in `/vineyard_nav`, commands will fail with path errors.
This is the single most common mistake. Every code block below assumes you have
already done this.

---

## What do you want to do? (routing)

| I want to… | What I need | Go to |
|---|---|---|
| **Understand what was done and found** | nothing — just this repo | `vineyard_nav/docs/STATUS.md`, then `FINDINGS.md`; browse the figures in `vineyard_nav/results/geometric/<bag>/final/figures/` |
| **Verify a committed result without re-running any model** | nothing — the per-frame data ships in the repo | Re-run an *analysis* on the committed CSV, e.g. `python3 scripts/geometric/analyze.py --bag march --only line_fit_eval` — it recomputes the report JSON from data already in the repo |
| **Re-run the geometric (centreline) pipeline on a bag** | a ROS bag **+** the 9 model weights **+** ~150 GB free disk **+** a CUDA GPU | `scripts/geometric/README.md` (full walkthrough, April as the worked example) |
| **Re-run the control (PID / command) strand** | that bag's geometric outputs (run its **non-in-row** branch first) **+** the bag `.db3` — **no weights, no GPU** | `scripts/control/README.md` |
| **Re-train the perception models** | the SemanticBLT dataset + a CUDA GPU | `scripts/perception/README.md` |

**The key distinction:** verifying the committed *analysis* needs nothing but
this repository — the per-frame CSVs are committed. Only re-running the *models*
(inference or training) needs the large files below, which are not in the repo.

---

## How the strands fit together

The three strands run in a fixed order, each consuming the previous one's output:

```
perception ──▶    geometric ──▶       control
 train the         run the models      turn the centreline
 models            on a bag →          into steering commands →
                   centreline          smoothness comparison
```

You cannot run a strand until the previous one's outputs exist. Perception's
outputs are the nine checkpoints — **not** in the repo (gitignored; train or
obtain them), so re-running the geometric strand needs them first. The committed
geometric/control *results* let you verify without re-running anything (routing
table above).

**How often each runs:**

| Strand | Run it… | Because |
|---|---|---|
| Perception | **once for the whole project** | the nine checkpoints are reused unchanged on every bag — it is *not* re-run per season |
| Geometric | **once per bag** (`--bag march`, `--bag april`, …) | each bag is a different recording |
| Control | **once per bag** | it runs on that bag's geometric output |

**Core pipeline vs. supporting analysis:** within each strand, some scripts are
the core pipeline that produces the committed headline artefacts (centreline CSV,
accuracy report, command stream, smoothness comparison); others are supporting
analyses of specific behaviours (abstention, near-seed sensitivity, the config
sweep, the mitigation gate, the `one_time/` studies). Each strand README marks
which is which — to reproduce just the headline results you need the core
pipeline, not every analysis script.

---

## Run a seasonal bag start to finish

Perception is already trained (the 9 checkpoints are bag-independent), so a new
bag is **geometric then control**. All commands run from `vineyard_nav/`; replace
`may` with any registered bag. The two strand READMEs hold the per-step detail,
runtimes, and "you should see" checks — this is the chaining overview.

1. **Convert** the ROS1 bag to ROS2 — `scripts/geometric/convert_bag.py --bag may`
   (needs `../kg_may_06.bag`; ~1× its size in free disk).
2. **Prepare** — `scripts/geometric/prep.py --bag may` runs CP-0 (prefix-scene
   census + the D048 gate over the 90 unattributed scenes) then CP-1 (frame
   manifest). If CP-0 flags `needs_review`, it stops before the manifest — open
   the census `d048_gate.needs_review`, confirm each scene present/absent, re-run.
3. **Geometric strand** — follow `scripts/geometric/README.md` from Stage A3 with
   `--bag may` (extract → detection cache [+ the automatic F007 blob-guard audit]
   → 9-model inference → `analyze.py` → figures), including the `--scope non_in_row`
   branch (needed by the control strand's F026 validation).
4. **Control strand** — follow `scripts/control/README.md` with `--bag may`
   (no weights, no GPU; reads the geometric outputs + the bag `.db3`).
5. **Cross-bag figures** — `figures_compare.py --bags march april may` to fold the
   new bag into the seasonal comparison.
6. **Confirm the bag is done** — `scripts/geometric/check_bag_complete.py --bag may`.
   It passes only when every artefact exists **and** `docs/STATUS.md` carries a
   consolidated "Confirmed on may" summary; `control.py` runs it automatically at the
   end of step 4. **A bag is not done until this passes.**

Stages 3–4 reuse the same 9 checkpoints as march/april — **no retraining**.

---

## What ships in the repo vs. what you must obtain separately

| Component | In the repo? | How to obtain |
|---|---|---|
| Pipeline + analysis code (`scripts/`) | ✅ | — |
| Documentation (`docs/`) | ✅ | — |
| Committed results — per-frame CSVs, JSON artefacts, figures | ✅ | — |
| ROS1 bags (`kg_<month>_*.bag`, ~100 GB each) | ❌ gitignored | Obtain from the L-CAS BLT recordings; place at the repo root |
| SemanticBLT segmentation dataset | ❌ | Roboflow: https://universe.roboflow.com/gaia-hse8w/semanticblt — place at `SemanticBLT.v1-2024-june.coco-segmentation/` in the repo root (perception training only) |
| 9 model checkpoints (`results/runs/phase_*`) | ❌ gitignored | Train them (`scripts/perception/README.md`), or obtain from the author |
| Extracted frames, detection cache, ROS2 `.db3` | ❌ gitignored | Regenerated by the pipeline (`convert_bag.py`, `extract_frames.py`, …) |

The dataset directory name contains "june" because that is the Roboflow *export*
date, not an acquisition month — see the note on bag coverage below.

---

## Which bags are evaluated (why six are registered but two are done)

The pipeline is bag-parametrised: `--bag <month>` selects the recording. Six
bags are registered in `scripts/geometric/bag_config.py`, but only two have been
evaluated so far:

| Bag | Status |
|---|---|
| `march` | ✅ evaluated (the primary bag; all findings anchored here) |
| `april` | ✅ evaluated (second season; confirms the March findings) |
| `may` | ⏭️ next — unblocked, not yet run |
| `june`, `july`, `september` | ⏳ armed — run the CP-0 D048 gate at their turn; proceed unless it flags `needs_review` |

**O019 — resolved and wired (D048).** 90 of the 230 labelled SemanticBLT scenes
carry no month in their filename (39% of the dataset, of unknown origin — summer-
foliage "canopy" scenes, most plausibly from the summer bags). The risk: if any
belong to an evaluated bag, that bag is under-excluded and contaminated. This is
now handled automatically at **CP-0** for every bag: the D048 gate
(`scripts/geometric/scene_attribution.py`) scores those 90 scenes against the bag
by ORB+RANSAC identity and either excludes them (≥200 inliers), flags them for
review (40–200, which blocks the bag's evaluation until confirmed), or clears them
(≤40). March and April measured all 90 as absent. June/July/September are no
longer hard-blocked — they simply run the gate at their CP-0 and proceed unless it
raises `needs_review`. Full rationale: `docs/DECISIONS.md` (D046, D048) and
`docs/STATUS.md` (O019).

---

## Environment

An L-CAS ROS2 Humble devcontainer is provided (`.devcontainer/`). Open the repo
in VS Code and "Reopen in Container", or replicate the container manually.

- Python 3.10, PyTorch 2.11 + CUDA 12.8. Package pins: `vineyard_nav/requirements.txt`.
- GPU used for this work: RTX 5050 Laptop, 8 GB, Blackwell sm_120 (requires the
  CUDA-12.8 PyTorch build; older builds lack sm_120 support).
- No ROS installation is required — bags are read with the pure-Python `rosbags`
  package.

**Long-run caveat (observed, not theoretical):** the full geometric + control
pipeline for one bag is several hours and GPU-bound, and during this project the
dev-container connection dropped several times, killing foreground jobs. Long
steps should be launched detached (`setsid nohup … &`) so a dropped connection
does not lose them; two scripts (`near_seed_sensitivity.py`, the detection cache)
were made resumable for exactly this reason. The per-strand READMEs mark which
steps are long.

---

## Repository map

```
dissertation/                         ← you clone this (git root)
├── README.md                         ← this file
├── .devcontainer/                    ← the ROS2 Humble container
├── SemanticBLT…/  (obtain separately, gitignored)   ← perception dataset
├── kg_<month>_*.bag  (obtain, gitignored)           ← raw ROS1 recordings
├── kg_<month>_ros2/  (generated, gitignored)        ← ROS2 conversion (.db3)
└── vineyard_nav/                     ← THE PROJECT — run all commands from here
    ├── docs/                         ← all specifications and the decision/finding logs
    ├── requirements.txt
    ├── configs/                      ← YAML training configs
    ├── data/splits/                  ← committed dataset split manifests
    ├── scripts/
    │   ├── perception/               ← train/evaluate the 3 arms  (strand 1)
    │   ├── geometric/                ← centreline pipeline         (strand 2)
    │   │   ├── one_time/             ←   one-off analyses (not pipeline steps)
    │   │   ├── diagnostics/          ←   supporting/dev scripts
    │   │   └── superseded/           ←   retained historical scripts
    │   └── control/                  ← command-level PID strand    (strand 3)
    └── results/
        └── geometric/
            ├── march/  april/        ← per-bag: final/ cache/ diagnostics/
            └── comparison/figures/   ← cross-bag comparison figures
```

(Note: an empty `vineyard_nav/runs/` directory may appear — it is an unused
stray, not part of the layout; experiment outputs live under `results/runs/`.)

---

## Documentation index (`vineyard_nav/docs/`)

**Start here:**
- `STATUS.md` — current progress, what's done, what's next, open items.
- `PROJECT_PLAN.md` — full scope and the three-arm design.

**The evidence trail (these feed the dissertation directly):**
- `DECISIONS.md` — every locked decision with rationale (D001–D047).
- `FINDINGS.md` — every empirical finding (F001–F028), each with its evidence.

**Implementation contracts (what each stage must do):**
- `PHASE_A_SPEC.md` / `PHASE_B_SPEC.md` / `PHASE_C_SPEC.md` — the three perception arms.
- `GEOMETRY_PIPELINE_SPEC.md` — the centreline pipeline (CP-0 → CP-6).
- `POOLING_SPEC.md` — whole-bag pooling methodology.
- `PID_PIPELINE_SPEC.md` — the command-level control strand.
- `CONTROL_DESIGN_INTENT.md` — design intent behind the controller.
- `FIGURE_SPEC.md` — the report-figure contract.

`vineyard_nav/CLAUDE.md` holds working rules used during development.

---

## Per-strand entry points

Each strand README is self-contained and followable from a clean checkout:

- **`scripts/perception/README.md`** — train and evaluate the three arms.
  *Needs the SemanticBLT dataset.* (Not re-run during the most recent
  multi-bag session; commands are from the strand's own documentation.)
- **`scripts/geometric/README.md`** — the centreline pipeline end-to-end, with
  April as a fully worked example (every command and observed runtime).
  *Needs a ROS bag + the model weights.*
- **`scripts/control/README.md`** — the command-level PID strand, run on the
  geometric strand's outputs.

---

## Honest notes

- **Reference trajectory is autonomous, not teleoperated.** The BLT recordings
  were collected under the platform's own **GPS/topological autonomous
  navigation** (Polvara et al. 2024, §3.3.3), not manual teleoperation. The
  geometric and command metrics compare our vision-derived centreline against
  that *driven path*. This matters: the platform did not steer from vision, which
  is why the recorded steering cannot be used to *tune* a vision controller
  (finding F027). Some older documents still say "teleoperator"; that wording is
  superseded and scheduled for correction (STATUS.md follow-up item).
- **The pipeline's "bag-agnostic" claim became true only after April.** Several
  code paths were implicitly correct only while one bag existed and were silently
  wrong for a second; they were found and fixed during the April run (DECISIONS
  D046f). Treat any path not yet exercised by a second bag as unproven.
- **What is verified in this repo:** the March and April geometric + non-in-row
  + control results, and the figures, were produced and checked during the most
  recent session. Perception *training* was not re-run then — those results date
  from earlier phases and are recorded in `docs/DECISIONS.md` (O003).

---

*Reproducibility: every trained checkpoint records the git commit it came from;
split manifests are in `vineyard_nav/data/splits/`. Seed 42 throughout.*
