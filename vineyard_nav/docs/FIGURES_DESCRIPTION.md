# FIGURES_DESCRIPTION.md — figure catalogue for the A2 write-up

**Purpose.** A single index of every figure the pipeline has produced, where it lives, what it shows,
and a report-ready caption. Written so a chapter draft can pull a figure without opening the repo.

**Status:** current as of 9 August 2026. Regenerate this file if `figures*.py` change.

---

## 0. How to use this file

Each entry gives:

- **Path** — relative to `vineyard_nav/`. Clickable in the IDE.
- **Shows** — the claim the figure carries.
- **Caption (report-ready)** — drop-in `\caption{}` text, **content language only**. Numbers in these
  captions are *March* unless stated; for another bag read the value from the **Traces to** artefact.
- **Traces to** — the committed JSON the numbers come from, plus the working-doc finding/decision id.

> **Caption discipline (LOCKED).** Rendered PNG suptitles carry **no `D0xx`/`F0xx` identifiers** —
> a marker has no access to `DECISIONS.md` or `FINDINGS.md`. The **Traces to** column is for your own
> cross-referencing while drafting; it must not appear in the submitted caption. Enforced by an AST
> check over the figure scripts.

**Shared colour legend** (state once in the report, near the first figure; do not repeat per caption):

| Element | Colour |
|---|---|
| Trunk base points (multiclass class 0) | blue `#2b6cff` |
| Pole base points (multiclass class 1) | yellow `#ffd21e` |
| Binary base points (U-Net / YOLO-binary) | cyan `#00d0d0` |
| Fitted row lines | red `#d1341c` |
| Predicted centreline | green `#1a9e4b` |
| Driven path (odometry) | red dotted `#d1341c` |
| Fit inlier / outlier (bird's-eye) | filled / hollow marker |
| Mitigation accept / reject | green / red banner + reason |
| Arm A / B / C in summary plots | `#4477aa` / `#ee6677` / `#228833` |

**Arm naming in captions.** Use *U-Net binary* (A), *YOLO binary* (B), *YOLO multiclass* (C) — not the
bare letters — on first use in each chapter.

---

## 1. Ktima single-bag figure set — 15 per bag

Produced by `scripts/geometric/figures.py --bag <bag>`. Four bags: `march`, `april`, `may`, `june`.
Filenames embed the frame index, so they differ per bag — the **Availability matrix** in §5 gives the
exact filename for each bag.

Path root: `results/geometric/<bag>/final/figures/`

### 1a. In-row (6)

| # | File (march) | Shows | Traces to |
|---|---|---|---|
| 1 | `in_row/fig1_anatomy_10247.png` | Method anatomy: driven trajectory vs predicted centreline, 2 m look-ahead marker, offset/heading readout, IPM bird's-eye | method figure — no finding |
| 2 | `in_row/fig2_arm_invariance_7397.png` | 3-up A/B/C on one frame; near-identical centrelines | `line_fit_report.json` · F013 |
| 2b | `in_row/fig2b_forest_paired.png` | Paired cross-arm bootstrap forest; GT-1 CIs all include zero | `paired_crossarm.json` · F013 |
| 3 | `in_row/fig3_tilt_sensor_common.png` | Camera vs LiDAR heading, 10 anchors × 5 corridors — sensor-common tilt | `lidar_crosscheck.json` · F017 |
| 4 | `in_row/fig4_mechanism_10247_C.png` | Multiclass class colours; trunks load-bearing in the near field; fit is class-agnostic | `line_fit_report.json` · F018 |
| 4b | `in_row/fig4b_abstention_13820.png` | `single_row` abstention — one side has too few near detections to seed a fit | `single_row_analysis.json` · F024, F025 |

**Captions.**

- **Fig 1** — *Anatomy of one in-row frame. The predicted centreline (green) is fitted from the two
  detected row lines (red); the driven path from odometry (red dotted) is the reference. Lateral offset
  is read at a 2 m look-ahead (★). Right panel: the same frame in the inverse-perspective bird's-eye
  view in which the fit is performed.*

- **Fig 2** — *The same frame through all three arms. The recovered centrelines are near-identical
  (lateral offset +0.157 / +0.157 / +0.156 m for U-Net binary, YOLO binary and YOLO multiclass), so the
  geometry that reaches the controller is indistinguishable between them.*
  **Bag-specific:** replace the three offsets from that bag's `line_fit_report.json`.

- **Fig 2b** — *Paired cross-arm differences with moving-block bootstrap confidence intervals, computed
  on the frames where both arms emit a centreline. Every lateral-offset interval spans zero; the heading
  differences sit below the sensor noise floor. Pairing removes the scene, so what remains is attributable
  to the arm.*

- **Fig 3** — *Camera-derived heading against an independent LiDAR estimate at ten anchor positions in each
  of five corridors. Both means are positive (camera +1.86°, LiDAR +2.57°, difference −0.71°), so the
  apparent tilt is common to both sensors rather than an artefact of the vision pipeline.*
  **Bag-specific:** three numbers from `lidar_crosscheck.json`.

- **Fig 4** — *Multiclass output on the anatomy frame. Trunks (blue) supply almost all of the near-field
  base points that seed the fit; poles (yellow) contribute mainly at range. The line fit itself is
  class-agnostic, which is why the extra class labels do not move the recovered geometry.*

- **Fig 4b** — *An abstention. The pipeline declines to emit a centreline rather than extrapolate: the
  right side has six detections inside the 5 m near-seed window and is fitted (red), while the left side
  has only one within 5 m (nine lie beyond) and seeding requires at least two, so that side is rejected
  and no centreline is produced. The criterion is a count inside the window, not an absence of detections.
  Widening the window to 6 m recovers roughly 28% of abstained frames at about 4% RMS cost, but wider
  windows corrupt the fit with adjacent-row points (largest observed shift ~1.85 m at 6.5 m).*

### 1b. Non-in-row (5) — the deployment gap

| # | File (march) | Shows | Traces to |
|---|---|---|---|
| 5 | `non_in_row/fig5_stationary_6.png` | Spurious two-row fit while the robot is stationary | F020 |
| 5b | `non_in_row/fig5b_output_distribution.png` | Spurious two-row rate by category × arm | `non_in_row_analysis.json` · F020 |
| 6 | `non_in_row/fig6_turn_10111.png` | Spurious centreline during a headland manoeuvre | F020 |
| 7 | `non_in_row/fig7_transition_11264.png` | Off-nominal spurious geometry during a corridor transition | F020 |
| 8 | `non_in_row/fig8_driven_path_11264.png` | Spurious centreline against the actual driven path | F021 |

**Captions.**

- **Fig 5** — *The pipeline emits a confident two-row centreline while the robot is stationary outside a
  corridor. Nothing in the per-frame output distinguishes this from a valid in-row fit.*
- **Fig 5b** — *Rate of spurious two-row output per non-in-row category and per arm. Turning frames are
  the worst case (76–80% on this bag). The failure is a property of the geometry stage, not of any one
  segmentation arm.*
- **Fig 6** — *A spurious centreline produced mid-turn in the headland. The fit is geometrically
  well-formed and carries no indication that the robot is not in a row.*
- **Fig 7** — *A corridor transition. The recovered heading is +13°, far outside the in-row distribution,
  yet the frame is still emitted as a valid two-row fit.*
- **Fig 8** — *The same transition frame with the actual driven path overlaid (red dotted). The predicted
  centreline diverges from where the robot in fact went; following it would have steered the platform
  into the row.*

### 1c. Mitigation (4)

| # | File (march) | Shows | Traces to |
|---|---|---|---|
| 9 | `mitigation/fig9_f022_3up.png` | Motion-state gate rejecting one frame per category | `mitigation_analysis.json` · F022 |
| 10 | `mitigation/fig10_f023_3up.png` | Geometry filter catches, with the in-row p99 threshold marked | `mitigation_analysis.json` · F023 |
| 11 | `mitigation/fig11_turn_blind_14987.png` | Deep turn: state gate rejects, geometry filter accepts | F023 |
| 12 | `mitigation/fig12_complementarity.png` | Coverage of the two filters by category | `mitigation_analysis.json` · F022, F023 |

**Captions.**

- **Fig 9** — *The motion-state gate applied to one frame from each non-in-row category. Each panel shows
  the triggering quantity (forward speed, lateral velocity magnitude, or heading rate) and the resulting
  reject banner.*
- **Fig 10** — *The geometry filter on three frames it catches. The threshold shown on each panel is the
  99th percentile of the corresponding in-row distribution, so a rejection means the frame is
  geometrically unlike any valid in-row fit.*
- **Fig 11** — *The two filters disagree. In a deep turn the lateral-velocity term collapses, so the
  motion-state gate rejects, while the recovered geometry stays inside the in-row envelope and the
  geometry filter accepts. The filters are complementary rather than redundant.*
- **Fig 12** — *Which filter catches which frames, by category: state-gate only, both, geometry-only
  (small), and neither. The residual "neither" fraction is the part of the deployment gap that neither
  cheap mitigation closes.*

> **June withholds two figures.** All 40 of June's turning two-row frames contain identifiable people, so
> `fig6_turn` and `fig11_turn_blind` are **not rendered for June** — privacy screening is a hard
> prerequisite and the exclusion is declared in `FIGURE_EXCEPTIONS`. June's turn **statistics** are
> unaffected and remain in the tables; only the imagery is withheld. Say this in the text if you show
> June's set, so the gap does not read as a missing result.

---

## 2. Cross-bag comparison set — 4 figures

Produced by `scripts/geometric/figures_compare.py`. Pooled across the four Ktima bags.
Path root: `results/geometric/comparison/figures/`

| File | Shows | Caption (report-ready) | Traces to |
|---|---|---|---|
| `cmp_forest.png` | Paired cross-arm differences, all bags on one forest | *Paired cross-arm differences across all four bags. Red marks an interval excluding zero. The pairing is within-frame, so scene difficulty cancels and the residual is attributable to the arm.* | `paired_crossarm.json` · F013 |
| `cmp_tilt.png` | Camera-vs-LiDAR tilt agreement across bags | *Camera and LiDAR heading estimates agree in sign and approximate magnitude on every bag, confirming the tilt is sensor-common rather than a vision artefact.* | `lidar_crosscheck.json` · F017 |
| `cmp_nonrow_distribution.png` | Spurious two-row rate per category, all bags | *Spurious two-row output per non-in-row category across the four bags. The deployment gap reproduces on every bag; its composition shifts with the headland stratum sampled.* | `non_in_row_analysis.json` · F020 |
| `cmp_mitigation_closure.png` | How much of the gap each filter closes, per bag | *Fraction of spurious non-in-row output rejected by the motion-state gate, per bag. Closure falls from ~98% on March to 62–65% on June, because June's headland stratum is dominated by straight-line transitions the gate cannot see.* | `mitigation_analysis.json` · F022 |

---

## 3. Supplementary set — 6 figures

Produced by `scripts/geometric/figures_supplementary.py`. Written to the **same directory** as §2,
so the two sets are distinguished by generating script, not by path.

Path root: `results/geometric/comparison/figures/`

| File | Shows | Caption |
|---|---|---|
| `cmp_coverage_trend.png` | Two-row coverage across four bags, two seasons | *Two-row coverage for each arm across the four bags. Coverage drops sharply from the bare-vine season to the canopy season, and the ordering of the arms is preserved throughout.* |
| `cmp_season_contrast.png` | One bare-vine and one canopy frame side by side | *Two seasons at their typical frame, showing the detections available to the row fit. Under canopy, foliage occludes the trunk bases the fit depends on.* |
| `cmp_model_outputs_march.png` | Raw output of each arm on one March frame | *What each arm actually outputs on the same frame: a binary mask (U-Net), binary instances (YOLO binary), and class-labelled instances (YOLO multiclass). The three representations differ; the geometry derived from them does not.* |
| `cmp_model_outputs_april.png` | as above, April | as above, substituting the bag and season |
| `cmp_model_outputs_may.png` | as above, May | as above |
| `cmp_model_outputs_june.png` | as above, June | as above |

Useful as the **first** figure in the Methodology chapter — it makes the three-arm design concrete before
any metric appears.

---

## 4. Riseholme supplementary strand — 10 figures

Isolated tree; produced by `scripts/riseholme/figures.py` and `figures_verification.py`.
Path root: `results/riseholme/tue02sep/`

### 4a. Illustration set (4)

| File | Shows | Caption |
|---|---|---|
| `figures/rh_fig1_inputs.png` | Six input frames, one per corridor | *Riseholme input frames, one per corridor traversed. The site differs from the Greek vineyard in row spacing, structure and camera mounting, which is what makes it an out-of-distribution test rather than a second sample of the same conditions.* |
| `figures/rh_fig2_arms_basepoints_fits.png` | Detections, base points and fit for all three arms on one frame | *The same Riseholme frame through all three arms, showing detections, extracted base points and the fitted centreline. The pipeline is byte-identical to the one used at the primary site; only the camera calibration differs.* |
| `figures/rh_fig3_success_failure.png` | Success, partial and failure frames | *Successful, partial and failed frames from the multiclass arm. Failures are dominated by too few near-field detections to seed a fit — the same abstention mechanism as at the primary site, triggered more often.* |
| `figures/rh_fig4_results.png` | Results tables rendered as a figure | *Riseholme results summary. Absolute errors are caveated by the calibration uncertainty; the paired cross-arm differences are not, because the calibration is common to all three arms and cancels in the difference.* |

### 4b. Verification set (2) — these carry the calibration argument

| File | Shows | Caption |
|---|---|---|
| `figures/verif_gt_overlay.png` | Surveyed reference line projected into three frames, with the calibration-uncertainty band | *The surveyed mid-row reference projected into three frames using the recovered extrinsics. The shaded band is the **calibration uncertainty alone** — the envelope swept by the assumed lateral offset and residual yaw — not a confidence interval on the estimate.* |
| `figures/verif_sensitivity.png` | How absolute RMS moves under plausible calibration error | *Why the paired cross-arm contrasts are the primary Riseholme result: absolute RMS error moves substantially under plausible calibration error, while the paired differences are unaffected because the calibration term is common to all three arms.* |

### 4c. Riseholme diagnostics (4) — appendix or defence, not the main chapters

| File | Shows |
|---|---|
| `diagnostics/gt_line_sanity.png` | Plain overlay, no uncertainty band: does the surveyed line land on the real row? |
| `diagnostics/gt_line_plain_zoom.png` | The same, zoomed to the drawn 2.48–6.0 m ground window |
| `diagnostics/gt_line_sidebyside.png` | Zoomed side-by-side comparison across frames |
| `diagnostics/gt_line_combined_band.png` | Combined-uncertainty band (calibration + reference), conservative direct sum |

Use `gt_line_combined_band.png` **only** alongside a sentence justifying the direct sum over root-sum-square:
the terms are fixed systematic biases on a single deployment, not independent random errors, so RSS would
understate the envelope.

---

## 5. Availability matrix — exact filenames per bag

Frame indices differ per bag. `—` means the figure is not rendered for that bag.

| # | march | april | may | june |
|---|---|---|---|---|
| 1 | `fig1_anatomy_10247` | `fig1_anatomy_19491` | `fig1_anatomy_4657` | `fig1_anatomy_2140` |
| 2 | `fig2_arm_invariance_7397` | `fig2_arm_invariance_13467` | `fig2_arm_invariance_4644` | `fig2_arm_invariance_2095` |
| 2b | `fig2b_forest_paired` | same | same | same |
| 3 | `fig3_tilt_sensor_common` | same | same | same |
| 4 | `fig4_mechanism_10247_C` | `fig4_mechanism_19491_C` | `fig4_mechanism_4657_C` | `fig4_mechanism_2140_C` |
| 4b | `fig4b_abstention_13820` | `fig4b_abstention_17967` | `fig4b_abstention_3006` | `fig4b_abstention_2026` |
| 5 | `fig5_stationary_6` | `fig5_stationary_546` | `fig5_stationary_15053` | `fig5_stationary_4063` |
| 5b | `fig5b_output_distribution` | same | same | same |
| 6 | `fig6_turn_10111` | `fig6_turn_15707` | `fig6_turn_15165` | **—** (privacy) |
| 7 | `fig7_transition_11264` | `fig7_transition_624` | `fig7_transition_5584` | `fig7_transition_4052` |
| 8 | `fig8_driven_path_11264` | `fig8_driven_path_624` | `fig8_driven_path_5584` | `fig8_driven_path_4052` |
| 9 | `fig9_f022_3up` | same | same | same |
| 10 | `fig10_f023_3up` | same | same | same |
| 11 | `fig11_turn_blind_14987` | `fig11_turn_blind_15705` | `fig11_turn_blind_15577` | **—** (privacy) |
| 12 | `fig12_complementarity` | same | same | same |

**Totals:** 15 + 15 + 15 + 13 = **58** Ktima single-bag · **6** comparison · **4** supplementary ·
**10** Riseholme = **78 report-eligible figures**.

---

## 6. Not for the report

Present in `results/` but diagnostic-only — generated to validate the pipeline, not to illustrate a claim.
Listed so you do not go hunting for them later, and do not mistake them for deliverables.

| Location | Count | What it is |
|---|---|---|
| `results/geometric/march/diagnostics/figures/rowfit_validation/` (+ `linefit/`, `linefit_final/`, `far_ext/`) | 39 | Row-fit development sweeps — superseded by the locked fit |
| `results/geometric/june/diagnostics/blob_audit/` | 1 | Blob-fraction threshold audit |
| `results/geometric/march/single_arm_dryrun_samples/` | 4 | Pre-three-arm dry-run samples |
| `results/riseholme/tue02sep/diagnostics/` | 4 | See §4c — appendix-eligible, not main-chapter |

Any of these can be promoted to an appendix if a viva question needs them; none belongs in the main text.

---

## 7. Suggested allocation by chapter

Do not use all 78. A first pass that carries the whole argument:

| Chapter | Figures |
|---|---|
| Methodology | `cmp_model_outputs_march` (three arms made concrete) · **Fig 1** (anatomy) |
| Results — in-row | **Fig 2**, **Fig 2b**, `cmp_forest` |
| Results — mechanism | **Fig 4**, **Fig 3** |
| Results — abstention | **Fig 4b** |
| Results — season | `cmp_coverage_trend`, `cmp_season_contrast` |
| Results — deployment gap | **Fig 5b**, **Fig 8** |
| Results — mitigation | **Fig 12**, `cmp_mitigation_closure` |
| Results — generalisation | `rh_fig1_inputs`, `rh_fig4_results`, `verif_sensitivity` |
| Appendix | remaining per-bag sets, Riseholme diagnostics |

That is **17 in the main text**, which is proportionate for the word count.

---

## 8. Regeneration

```bash
python3 scripts/geometric/figures.py --bag march            # all 15 for one bag
python3 scripts/geometric/figures.py --bag march --only 4b  # one figure by id
python3 scripts/geometric/figures_compare.py                # cross-bag set
python3 scripts/geometric/figures_supplementary.py          # model-output set
python3 scripts/riseholme/figures.py                        # Riseholme illustration
python3 scripts/riseholme/figures_verification.py           # Riseholme verification
```

All are deterministic (seed 42) and assert CSV-consistency: every per-frame figure recomputes the
class, offset and heading it draws and fails if they disagree with the committed artefact. A figure that
renders is a figure whose numbers match the results tables.
