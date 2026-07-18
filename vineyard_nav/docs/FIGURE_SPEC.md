# FIGURE_SPEC.md — March strand report figures (O012, Commit 9) — **as built**

The locked figure contract for the report figures illustrating the **complete March geometric-strand
narrative**: in-row headline (F013/F017/F018) + in-row abstention (F024) → non-in-row deployment gap
(F020/F021) → mitigation demonstration (F022/F023). **15 figures.** Regenerated deterministically by
`scripts/geometric/figures.py`; bag-parametrised so April+ reuse the module without edit.

```
python3 scripts/geometric/figures.py --bag march            # regenerate all 15
python3 scripts/geometric/figures.py --bag march --only 4b  # one figure by id
```

---

## 1. Scope and discipline

- **Illustrate, do not re-derive.** Every figure visualises an already-committed finding; caption
  numbers trace to the committed artefacts (`line_fit_report.json`, `paired_crossarm.json`,
  `non_in_row_analysis.json`, `single_row_analysis.json`, `mitigation_analysis.json`).
- **CSV-consistency assertion (LOAD-BEARING).** Every per-frame figure recomputes `(cls, offset,
  heading)` via the mirrored `estimate()` (`fit_frame`) and **asserts equality with the committed
  per-frame CSV row** before it plots (`assert_csv`). A figure can never disagree with its metric; a
  mismatch aborts. **All 15 figures pass** (12 per-frame assertions + 3 inference-free summaries).
- **RMS naming discipline (LOCKED).** In-row captions reference `centreline_error_rms`; non-in-row
  captions reference `driven_path_error`; **never conflated in a side-by-side "RMS vs RMS"**.
  `single_row` figures carry neither (abstention — excluded from both, F024).
- **Representative, not cherry-picked.** Frames chosen by explicit criteria (`scratchpad/
  figure_frame_select.py`, `single_row_select.py`).
- **No git.** Edosa runs all git.

---

## 2. Locked styling

`matplotlib.rcParams`: `font.family=serif` (DejaVu Serif, bundled — no external font dep),
`font.size=9`, `savefig.dpi=300`, `savefig.bbox=tight`. Text uses only DejaVu-safe glyphs (no `⋯ ★ ∪
≥`; `° − Δ` are fine). Colour conventions (LOCKED):

| Element | Colour |
|---|---|
| Trunk base points (Phase C class 0) | blue `#2b6cff` |
| Pole base points (Phase C class 1) | yellow `#ffd21e` |
| Binary base points (Phase A/B) | cyan `#00d0d0` |
| Fitted rows | red `#d1341c` |
| Centreline | green `#1a9e4b` |
| Driven-path (in-row ref / non-in-row odometry) | red dotted `#d1341c` |
| Fit inlier / outlier (bird's-eye) | filled / hollow marker |
| Mitigation accept / reject | green / red banner + reason |
| Arm A / B / C (summary plots) | `#4477aa` / `#ee6677` / `#228833` |

**Combined view** (per-frame): 2 panels, `figsize=(10, 4.6)` — left = raw 640×640 frame + overlays
(base points class-coloured; fitted rows red; centreline green; driven-path red dotted; drawn back onto
the image via `project_ground`, D1); right = bird's-eye (base_link X-forward vs −Y; filled = fit
inlier, hollow = outlier; 2 m look-ahead green ★). Descriptive caption is a **`fig.suptitle`** (full
width, no per-axis title collision); axis titles are short ("raw frame + detections" / "bird's-eye").
**3-up** figures `figsize=(12, 4.2–4.4)`; **summary** figures `figsize=(7–9, 3.4–4)`.

---

## 3. Module structure (`scripts/geometric/figures.py`, NEW)

- **Self-contained inference front-end** mirrors `line_fit_infer.py` exactly (`frontend` = YOLO/UNet
  base points + Phase-C class; `fit_frame` = byte-identical `estimate()` logic + geometry kept for
  drawing). The verified headline script is **not imported or refactored** (Commit 2b provenance).
  Base-point pixel coords + class are not stored in the CSV, so inference is re-run for the ~12
  selected frames only; the assertion guarantees no drift.
- **`project_ground(X,Y)`** (D1) — analytic inverse of `project_px` added additively to
  `projection_calibration.py` (`project_px` untouched; round-trip < 1e-13 px). Draws rows/centreline
  onto the raw image.
- Public API: `plot_in_row_frame(bag, frame, arm, *, anatomy, near_seed, tilt)`,
  `plot_non_in_row_frame(bag, frame, category, arm, *, driven)`,
  `plot_mitigation_frame(bag, frame, category, layer, arm)` (`layer ∈ {f022, f023, turn_blind}`);
  3-up wrappers `plot_arm_invariance`, `plot_mitigation_3up`; summaries `fig_forest`, `fig_tilt_sensor`,
  `fig_dist_bars`, `fig_complementarity`; CLI `build(bag, only)` over the locked `FIGURES` manifest.
- **Output:** `results/geometric/{bag}/final/figures/{in_row,non_in_row,mitigation}/` — PNGs committed
  (report deliverable; ~24 MB for the 15, dominated by the 3-up raw-image panels at 300 dpi). Frames
  dir itself stays gitignored.

---

## 4. Frame-reuse as a design principle

Frames recur **deliberately** across strands — this is narrative continuity, not redundancy: the same
frame carrying characterisation → mitigation is the strand's strongest visual device.

- **Frame 6** (stationary): Fig 5 (hallucination) → Fig 9 (F022 rejects, speed<0.10).
- **Frame 10111** (turn): Fig 6 (hallucination) → Fig 9 (F022 rejects, |v_y|<0.30).
- **Frame 11264** (transition): Fig 7 (hallucination) → Fig 8 (driven_path_error) → Fig 9 (F022 rejects).
- **Frame 10247** (in-row): Fig 1 (arm A anatomy) → Fig 4 (arm C class colours) — cross-arm consistency.

The viewer sees a failure, then sees the same failure rejected. Fig 12 aggregates all such frames.

---

## 5. The locked figure set (15, as built)

**In-row (6) — F013 / F017 / F018 / F024:**

| # | File | Shows | Frame(s) |
|---|---|---|---|
| 1 | `in_row/fig1_anatomy_10247.png` | Methodology anatomy: base_link fwd (red dotted), 2 m look-ahead (★), offset/heading, IPM bird's-eye | 10247, arm A |
| 2 | `in_row/fig2_arm_invariance_7397.png` | F013: 3-up A/B/C, near-identical centrelines (offset +0.157/+0.157/+0.156) | 7397 |
| 2b | `in_row/fig2b_forest_paired.png` | F013: paired cross-arm bootstrap forest — GT-1 all CIs include 0; GT-2 sub-noise-floor | `paired_crossarm.json` |
| 3 | `in_row/fig3_tilt_sensor_common.png` | F017: camera vs LiDAR heading across 10 anchors × 5 corridors — both means positive (cam +1.86°, LiDAR +2.57°, diff −0.71°), sensor-common tilt | `lidar_crosscheck.json` |
| 4 | `in_row/fig4_mechanism_10247_C.png` | F018: Phase-C class colours (blue trunks load-bearing near-field, yellow poles), class-agnostic fit | 10247, arm C |
| 4b | `in_row/fig4b_abstention_13820.png` | **F024** + **F025**: `single_row` — no centreline emitted (left side has only 1 detection *within* the 5 m near-seed window, D037 requires ≥2 to seed; right side has 6, fits); caption also cites the F025 sensitivity result (5 m near-optimal) | 13820, arm A |

**Non-in-row (5) — F020 / F021:**

| # | File | Shows | Frame(s) |
|---|---|---|---|
| 5 | `non_in_row/fig5_stationary_6.png` | F020: hallucinated two_row, stationary | 6 |
| 5b | `non_in_row/fig5b_output_distribution.png` | F020: spurious two_row rate by category×arm (turn spikes 76–80%) | `non_in_row_analysis.json` |
| 6 | `non_in_row/fig6_turn_10111.png` | F020: hallucinated centreline mid-turn | 10111 |
| 7 | `non_in_row/fig7_transition_11264.png` | F020: off-nominal hallucinated geometry (heading +13°) | 11264 |
| 8 | `non_in_row/fig8_driven_path_11264.png` | F021: hallucinated centreline (green) vs actual driven path (red dotted, odometry) — `driven_path_error` | 11264 |

**Mitigation (4) — F022 / F023:**

| # | File | Shows | Frame(s) |
|---|---|---|---|
| 9 | `mitigation/fig9_f022_3up.png` | F022: state gate rejects per category (speed / \|v_y\| / heading-rate) | 6, 10111, 11264 |
| 10 | `mitigation/fig10_f023_3up.png` | F023: geometry filter catches; firing in-row-p99 threshold labelled | 423, 12801, 653 |
| 11 | `mitigation/fig11_turn_blind_14987.png` | **F023 turn-blindness**: deep turn, F022 rejects (\|v_y\| collapse), F023 accepts (geometry within p99) | 14987 |
| 12 | `mitigation/fig12_complementarity.png` | F022 ∪ F023 by category: F022-only / both / F023-only (tiny) / neither | `mitigation_analysis.json` |

**Fig 4b caption (F024 + F025).** The PNG suptitle carries a short per-side count line + a two-line F025
pointer (kept short so the tight bbox does not widen the canvas); the **full report caption** (LaTeX
`\caption{}`) reads: *"cls == single_row: the pipeline abstains rather than extrapolating — the right
side has 6 detections within the 5 m near-seed window and is fitted (red), while the left side has only
1 within 5 m (9 beyond) and `fit_side_far` requires ≥ 2 to seed, so it is rejected `too_few_near_seed`
and no centreline is emitted (F024) — a count criterion, not "all points beyond 5 m". F025 (near-seed
sensitivity) confirms 5 m is empirically near-optimal: widening to 6 m recovers ~28% of abstained frames
at ~4% RMS cost, but wider windows degrade the full-set metric via adjacent-row corruption (max
existing-fit shift ~1.85 m at 6.5 m); production widening requires the D036/F014 adjacency-rejection
guard first."* Wired via the `caption_extra` parameter of `plot_in_row_frame` (Fig 4b only; Figs 1/3/4
unchanged).

**Fig 3 (F017) — summary, not single-frame (C3).** F017 is an aggregate, *sensor-common* claim; no
single frame can show arm-invariance or camera-vs-LiDAR agreement (a single frame such as 3998 arm A
even has m_L=−0.029, so "both positive, near-equal" is false per-frame — it is a **pooled** property).
Fig 3 is therefore a summary: camera (line-fit centreline, mean of 9 models) vs LiDAR (independent
row-plane fit) heading across the 10 pooled anchors (2 per corridor, all 5 corridors). Both means are
positive (cam +1.86°, LiDAR +2.57°, diff −0.71°) and the sensors **track the corridor-dependent tilt**;
that an independent sensor sees the same tilt shows it is a real robot-to-row geometry offset
(sensor-common), not a camera-projection artefact — superseding the camera-yaw hypothesis (F015).
Honest: in the flattest corridor (1) the camera heading dips slightly negative (−0.72°/−0.79°) while
LiDAR stays low-positive (+0.9°); both agree it is the flattest corridor. **Summary-integrity
assertion** (the summary analog of the CSV assertion): the plotted per-anchor means equal the committed
`lidar_crosscheck.json` mean fields.

---

## 6. Dependencies
matplotlib 3.10.9, numpy 2.2.6, opencv 5.0.0, torch + ultralytics + the 9 committed weights;
`bag_config`, `projection_calibration` (+ `project_ground`), `row_model`, `single_arm_dryrun`. No new
third-party packages.

## 7. Gate decisions — resolved
- **D1** `project_ground` inverse — **done** (additive; `project_px` untouched; round-trip < 1e-13 px).
- **D2** self-contained front-end + CSV-consistency assertion — **done** (headline script untouched; all
  assertions pass).
- **D3** output `final/figures/{in_row,non_in_row,mitigation}/`, PNGs committed — **done** (~24 MB).
- **D4** turn-blindness uses the `|v_y|` mechanism — **done** (Fig 11).
- **D5** figure set — **15** (Fig 4b retargeted to abstention per the F024 finding; Fig 2b/5b added as
  the quantitative F013/F020 summaries; Fig 3 built as the C3 camera-vs-LiDAR sensor-common summary —
  F017 is an aggregate claim, so a single frame under-illustrates it and 3998 arm A even fails the
  per-frame "both positive" caption).
