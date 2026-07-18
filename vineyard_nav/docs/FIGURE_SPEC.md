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
- **Content-language captions (LOCKED, Commit 12).** The **rendered PNG captions carry NO finding /
  decision / open-item identifiers** (no `F###`, `D###`, `O###`, `GT-#`) — a report figure must stand
  alone for a marker who never sees the working docs. Captions cite the mechanism or measurement
  directly (e.g. "sensor-common tilt", "state gate", "geometry filter", "lateral offset"). The
  finding cross-references live **here in the spec** (§5 *Shows* column = working-doc navigation; §5a =
  the verbatim content-language PNG captions). Verification: `grep -E "F0|D0|O0|GT-"` over
  `figures.py` returns only committed-JSON access keys (`F020_output_distribution`, `F022_F023_causal`,
  `GT1`/`GT2` selectors) — **zero in any displayed string**.
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

The **Shows** column is the **working-doc reference** (F-labels + cross-refs, for internal navigation
only — these do **not** appear on the figures). The **verbatim content-language PNG captions** (what a
marker actually sees) are recorded in **§5a**.

**In-row (6) — F013 / F017 / F018 / F024:**

| # | File | Shows | Frame(s) |
|---|---|---|---|
| 1 | `in_row/fig1_anatomy_10247.png` | Methodology anatomy: driven trajectory (red dotted, odometry) vs predicted centreline (green), 2 m look-ahead (★), offset/heading, IPM bird's-eye | 10247, arm A |
| 2 | `in_row/fig2_arm_invariance_7397.png` | F013: 3-up A/B/C, near-identical centrelines (offset +0.157/+0.157/+0.156) | 7397 |
| 2b | `in_row/fig2b_forest_paired.png` | F013: paired cross-arm bootstrap forest — GT-1 all CIs include 0; GT-2 sub-noise-floor | `paired_crossarm.json` |
| 3 | `in_row/fig3_tilt_sensor_common.png` | F017: camera vs LiDAR heading across 10 anchors × 5 corridors — both means positive (cam +1.86°, LiDAR +2.57°, diff −0.71°), sensor-common tilt | `lidar_crosscheck.json` |
| 4 | `in_row/fig4_mechanism_10247_C.png` | F018: Phase-C class colours (blue trunks load-bearing near-field, yellow poles), class-agnostic fit | 10247, arm C |
| 4b | `in_row/fig4b_abstention_13820.png` | **F024** + **F025**: `single_row` — no centreline emitted (left side has only 1 detection *within* the 5 m near-seed window, D037 requires ≥2 to seed; right side has 6, fits); caption also cites the F025 sensitivity result (5 m near-optimal) | 13820, arm A |

**Non-in-row (5) — F020 / F021:**

| # | File | Shows | Frame(s) |
|---|---|---|---|
| 5 | `non_in_row/fig5_stationary_6.png` | F020: spurious two_row, robot stationary | 6 |
| 5b | `non_in_row/fig5b_output_distribution.png` | F020: spurious two_row rate by category×arm (turn spikes 76–80%) | `non_in_row_analysis.json` |
| 6 | `non_in_row/fig6_turn_10111.png` | F020: spurious centreline, robot in headland manoeuvre | 10111 |
| 7 | `non_in_row/fig7_transition_11264.png` | F020: off-nominal spurious geometry (heading +13°), robot in corridor transition | 11264 |
| 8 | `non_in_row/fig8_driven_path_11264.png` | F021: spurious centreline (green) vs actual driven path (red dotted, odometry) — `driven_path_error` | 11264 |

**Mitigation (4) — F022 / F023:**

| # | File | Shows | Frame(s) |
|---|---|---|---|
| 9 | `mitigation/fig9_f022_3up.png` | F022: state gate rejects per category (speed / \|v_y\| / heading-rate) | 6, 10111, 11264 |
| 10 | `mitigation/fig10_f023_3up.png` | F023: geometry filter catches; firing in-row-p99 threshold labelled | 423, 12801, 653 |
| 11 | `mitigation/fig11_turn_blind_14987.png` | **F023 turn-blindness**: deep turn, F022 rejects (\|v_y\| collapse), F023 accepts (geometry within p99) | 14987 |
| 12 | `mitigation/fig12_complementarity.png` | F022 ∪ F023 by category: F022-only / both / F023-only (tiny) / neither | `mitigation_analysis.json` |

**Fig 4b caption (working-doc ref: F024 + F025).** The PNG suptitle carries a short per-side count line
+ a two-line near-optimal-window pointer, **content-language only** (no F/D labels — see §5a for the
verbatim text); the **full report caption** (LaTeX `\caption{}`, which *may* carry F-labels as the
working-doc report reference) reads: *"cls == single_row: the pipeline abstains rather than extrapolating — the right
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

## 5a. Rendered PNG captions (content-language, verbatim — Commit 12)

What actually appears on each figure (no F/D/O/GT identifiers). Per-frame numeric fields (offset,
heading, kinematics) vary by frame; the fixed caption text is shown.

| # | Rendered caption (verbatim) |
|---|---|
| 1 | `frame 10247 · arm A · two_row · offset=… m, heading=…°  (centreline_error_rms convention)` + line 2 `red dotted = driven trajectory (odometry) · green = predicted centreline · lateral gap = this frame's offset (single frame, not the pooled RMS)` (driven path on both panels; 2 m look-ahead green star) |
| 2 | suptitle `Arm-invariance · frame 7397 (near-identical centrelines; lateral offset indistinguishable across arms)` + line 2 `red dotted = driven trajectory (odometry); gap = per-frame offset (single frame, not the pooled RMS)`; per-arm `arm A/B/C · offset=… hdg=…` (driven path + 2 m star per panel) |
| 2b | suptitle `Paired cross-arm bootstrap (moving-block, whole-bag) · blue = CI includes 0`; panels `Lateral offset — all CIs include 0` / `Heading — sub-noise-floor difference`; axes `cross-arm Δ lateral offset (m)` / `cross-arm Δ heading (°)` |
| 3 | `Sensor-common tilt — camera vs LiDAR heading, 10 anchors × 5 corridors (cam +1.86°, LiDAR +2.57°, diff −0.71°)`; legend `camera (line-fit centreline, mean of 9 models)` / `LiDAR (independent row-plane fit)` |
| 4 | `frame 10247 · arm C · two_row · offset=… (centreline_error_rms convention)` + line 2 `Phase-C classes: trunks (blue) load-bearing in the near field; the row fit is class-agnostic` + line 3 `red dotted = driven trajectory (odometry) · green = predicted centreline · lateral gap = this frame's offset (single frame, not the pooled RMS)` |
| 4b | `frame 13820 · arm A · single_row — no centreline emitted (in-row abstention)` + `Left side: 1 detection within the 5 m near-seed window (fit needs >=2 to seed) · right side: 6, fits` + `the 5 m near-seed window is empirically near-optimal — widening to 6 m recovers ~28% of abstentions / at ~4% RMS cost; wider degrades the full-set metric via adjacent-row corruption (adjacency guard needed)` + `red dotted = driven trajectory (odometry) — robot driving in-row; the pipeline still abstained here`; bird's-eye annotation `5 m near-seed window` |
| 5 | `frame 6 · stationary · arm A · spurious two_row output · offset=… heading=…  (driven_path_error)` + line 2 `robot stationary  ·  v=… m/s, \|v_y\|=…` |
| 5b | `Non-in-row output distribution — spurious two_row rate by category` |
| 6 | `frame 10111 · turn · arm A · spurious two_row output · offset=… heading=…  (driven_path_error)` + line 2 `robot in headland manoeuvre  ·  v=… m/s, \|v_y\|=…` |
| 7 | `frame 11264 · transition · arm A · spurious two_row output · offset=… heading=…  (driven_path_error)` + line 2 `robot in corridor transition  ·  v=… m/s, \|v_y\|=…` |
| 8 | as Fig 7 + line 3 `green = spurious centreline · red dotted = actual driven path (odometry) — their divergence is the driven_path_error` (driven path on both panels) |
| 9 | suptitle (**blue accent**) `State gate — reject per category (odometry: speed / \|v_y\| / heading-rate)`; panels `frame N · category: REJECT (reason)`; **footer** `State gate catches 98.4% of spurious non-in-row outputs at 1.2% in-row false-positive · arm-invariant · uses odometry only (speed, \|v_y\|, heading-rate) - no perception input.` |
| 10 | suptitle (**orange accent**) `Geometry filter — off-nominal catches (firing in-row-p99 threshold labelled)`; panels `frame N · category: REJECT (threshold)`; **footer** `Geometry filter catches ~40% via off-nominal geometry at ~3% in-row false-positive · perception-based, odometry-free (deployment fallback) · cannot resolve clean-geometry turns.` |
| 11 | banner (**purple**) `state gate REJECT (\|v_y\|<0.30)   \|   geometry filter ACCEPT (geometry within in-row p99: \|off\|=… \|hdg\|=…)`; panels `frame 14987 · turn · arm A` / `bird's-eye · two_row`; **footer** `Fundamental limit: this frame's geometry (\|offset\|=… m, \|heading\|=…°) is INSIDE the in-row p99 envelope (0.71 m / 6.7°) -> the geometry filter accepts it; but the along-row velocity has collapsed (\|v_y\|<0.30) -> the state gate rejects it. Perception alone cannot resolve a clean-geometry turn without a state input.` |
| 12 | title `State gate + geometry filter (union) complementarity by category (mean over arms)`; **per-segment % labels** on each bar (state-gate-only 63/46/55, both 37/49/40, neither 0/3/4); legend `state gate only (blue) / both (purple) / geometry filter only (orange) / neither (grey)`; **footer** `State gate does the primary work (state-gate-only 46-63% across categories); the geometry filter marginally extends coverage (union 96-100%); the ~0-4% residual is architectural - needs learned state classification / sensor fusion.` |

## 5b. Overlay enhancements (Commit 12b)

- **Driven-path reference on in-row figures (1, 2, 4, 4b).** The odometry driven trajectory (future
  base_link poses via `_driven_path`, in-row-capped at any corridor-end turn) is drawn **red dotted on
  both the raw-image and bird's-eye panels** (previously only the bird's-eye, and only for non-in-row).
  It gives the physical reference for the in-row error: the metric is the centreline's lateral offset
  at the 2 m look-ahead against the driven-path reference (FINDINGS *Evaluation scope*; the driven path
  ≈ base_link-forward for in-row driving). Fig 4b shows it even though the pipeline abstained (the robot
  was driving in-row). **Caveat encoded in every caption:** the single-frame lateral gap is *this
  frame's* offset, **not** the pooled ~0.19 m RMS.
- **Full-length fits (all per-frame figures: 1, 2, 4, 4b, 5, 6, 7, 8, 9, 10, 11).** Fitted rows +
  centreline on the raw-image panel are drawn over `IMG_X0..IMG_X1 = 0.5..9 m` (near field → horizon)
  to match the bird's-eye extent, instead of the inlier X-span. Drawn-range fix (`draw_combined`,
  `plot_arm_invariance`, `plot_mitigation_3up`); `project_ground` was already range-unlimited. Uniform
  across every mitigation figure (9/10/11) too.
- **`HALLUCINATED` → `spurious two_row output`** on Figs 5/6/7/8, with per-category robot-state context
  (stationary / headland manoeuvre / corridor transition) — the pipeline detects real rows; the failure
  is context (state), not fabrication.
- **Mitigation narrative annotations (Figs 9/10/11/12).** Mechanism-colour accents on the
  suptitle/banner — **state gate = blue (`#2166ac`, odometry), geometry filter = orange (`#d95f02`,
  perception), turn-blindness/mixed = purple (`#8844aa`)** — plus a one-line summary-statistic footer
  per figure, so a cold reader gets the through-line without the working docs: state gate does the
  primary work (98.4% @ 1.2% FP, arm-invariant, odometry-only); geometry filter is the perception-only,
  odometry-free ~40% @ ~3% fallback that cannot catch clean-geometry turns; turn-blindness is the
  fundamental limit (clean-geometry turn is inside the in-row p99 so geometry accepts, but |v_y| has
  collapsed so state rejects). Fig 12 gains per-segment percentage labels + mechanism-matched bar
  colours + the primary-work / marginal-extension / architectural-residual summary. All content-language
  (numbers from `mitigation_analysis.json`; no F/D/O identifiers).

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
