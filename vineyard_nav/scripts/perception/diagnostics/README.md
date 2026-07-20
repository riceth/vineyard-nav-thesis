# Perception diagnostics — one-time, NOT needed to reproduce the pipeline

**Skip this folder if you are reproducing the work.** Nothing here is part of the perception
pipeline; these are one-off troubleshooting and exploratory scripts kept for the audit trail.
The reproduction path is `scripts/perception/README.md`.

Run from `vineyard_nav/` as `python3 scripts/perception/diagnostics/<name>.py`.

## Scene-6799 forensics — the whole-frame false mask

Investigated why YOLO foreground IoU collapses on one canopy test scene. Diagnostic only —
locked checkpoints, no retraining, no committed test metric.

- `diagnostic_6799.py` — forensic breakdown of the failure.
- `diagnostic_6799_viz.py` — high-res figures of the Phase B failure.
- `phase_c_6799_viz.py` — the same check for Phase C.

*(The reusable cross-seed overlap analysis that the write-up actually cites was promoted out of
this folder — it is now `scripts/perception/blob_overlap_6799.py`.)*

## Arm comparison

- `diagnostic_a_vs_b.py` — per-image 4-panel strips comparing Phase A (U-Net) and Phase B (YOLO)
  foreground predictions on test scenes.

## Label QA gates (run once, already passed)

- `phase_c_labels_spotcheck.py` — class-coloured overlay of Phase C multiclass labels
  (trunk red / pole blue), PHASE_C_SPEC §4.5 gate.

## Dataset exploration (pre-dates the pipeline)

- `blt_analysis.py` — per-image structural-element statistics over the SemanticBLT export.
- `blt_report.py` — histograms, viability summary, seasonal breakdown, sample overlays.
- `output/` — their rendered outputs (untracked).

## Ad-hoc

- `inspect_bag.py` — throwaway snippet for listing ROS bag topics. Superseded in practice by
  `scripts/geometric/bag_config.py` and the frame-manifest builder.
