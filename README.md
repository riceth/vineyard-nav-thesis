# Vineyard Navigation — MSc Dissertation

Multiclass Instance Segmentation for In-Row Vineyard Navigation: A Controlled Comparison Against the Binary-Mask Baseline.

**Author:** Edosa Ebohon (30436293)
**Programme:** MSc Robotics and Artificial Intelligence, University of Lincoln
**Module:** CMP9140 Research Project
**Submission:** A2 (dissertation), 26 August 2026

## Research question

Does a class-aware segmentation formulation (over trunk and pole classes distinctly), paired with class-aware per-side line fitting, produce more accurate centreline estimates for vineyard in-row navigation than the binary-mask formulation prevailing in vineyard literature?

## Design overview

Three-arm controlled comparison:

| Arm | Model | Class structure | Role |
|---|---|---|---|
| A | U-Net (SMP + ImageNet-pretrained ResNet-34) | Binary (trunk+pole -> foreground) | Official baseline, de Silva 2024 paradigm |
| B | YOLOv11-seg (COCO pretrained) | Binary | Modernised binary baseline |
| C | YOLOv11-seg (COCO pretrained) | Multiclass (trunk, pole distinct) | Contribution |

Two isolated comparisons: A vs B (architecture effect); B vs C (class-structure effect). Full design in `vineyard_nav/docs/PROJECT_PLAN.md`.

## Repository layout
## Documentation

Start in `vineyard_nav/docs/`:

- `STATUS.md` — current progress and immediate next task
- `PROJECT_PLAN.md` — full scope and three-arm design
- `DECISIONS.md` — locked decisions with rationale (feeds A2 Methodology chapter)
- `PHASE_A_SPEC.md` / `PHASE_B_SPEC.md` / `PHASE_C_SPEC.md` — per-phase implementation contracts
- `MILESTONES.md` — git commit trigger definitions

And `vineyard_nav/CLAUDE.md` for Claude Code working rules.

## External artefacts (not tracked in this repo)

Required at runtime; obtained separately due to size.

**SemanticBLT dataset** — expected at `/workspaces/dissertation/SemanticBLT.v1-2024-june.coco-segmentation/`

Public dataset from GAIA/L-CAS via Roboflow: https://universe.roboflow.com/gaia-hse8w/semanticblt

1035 images across 230 unique scenes, 6-class instance segmentation, 640x640, spans March-June (multi-month canopy variation).

**ROS bag** (`kg_march_23.bag`) — expected at `/workspaces/dissertation/kg_march_23.bag`

~19 minutes of teleoperated Thorvald traversal at a Lincoln vineyard, ZED stereo + LiDAR + odometry. Provided by the L-CAS lab. Consumed only by downstream evaluation (`vineyard_nav/evaluation/`); not required for training Phases A, B, or C.

## Environment

L-CAS ROS2 Humble devcontainer. See `.devcontainer/`. Python 3.10, PyTorch 2.11+cu128. Package pins in `vineyard_nav/requirements.txt`.

## Reproducibility

Every trained checkpoint records the git commit it was trained from (`git_commit.txt` in each run directory). Split manifests are tracked in `vineyard_nav/data/splits/`. To reproduce a phase's results: check out the relevant commit, place the external artefacts at the paths above, and re-run the training config in the run directory.
