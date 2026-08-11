# PROJECT_PLAN.md

**Project:** Multiclass Instance Segmentation for In-Row Vineyard Navigation: A Controlled Comparison Against the Binary-Mask Baseline
**Student:** Edosa Ebohon (30436293), MSc Robotics and Artificial Intelligence, University of Lincoln
**Module:** CMP9140 Research Project
**Timeline:** 13 June 2026 → 26 August 2026 (A2 submission)
**Status:** Design locked. **Implementation complete** — five bags evaluated end-to-end (march, april, may, june, july2023) across three arms x three seeds; the Riseholme generalisation strand is complete and boxed pending the calibration reply. Write-up in progress.

---

## 1. Research question

Does a class-aware segmentation formulation (over trunk and pole classes distinctly) — paired with class-aware per-side line fitting — produce more accurate centreline estimates for vineyard in-row navigation than the binary-mask formulation prevailing in vineyard literature?

## 2. Contribution

The methodological contribution is the **controlled comparison itself**: a three-arm experimental design that varies two axes — pipeline generation (U-Net baseline vs YOLO-seg) and class-structure (binary vs multiclass with class-aware downstream). No published work has done this comparison for vineyard centreline detection. Whichever direction the results land, the comparison informs practice.  *(Correction, 8 Aug 2026: only the class-structure axis is controlled. Arms A and B differ in at least thirteen respects, so A ↔ B is a baseline-versus-modernised-pipeline contrast, not an architecture-isolating one — see the correction at D006/D021. No architecture-attributable claim rests on it.)*

## 3. Three-arm design

Three model arms, all feeding the same downstream pipeline and evaluated by the same three-strand framework:

| Arm | Phase | Model | Class structure | Comparison role |
|---|---|---|---|---|
| 1 | A | U-Net (SMP + ImageNet-pretrained ResNet-34 encoder) | Binary: trunk+pole → foreground | Official baseline; represents de Silva 2024 paradigm |
| 2 | B | YOLOv11-seg (COCO pretrained) | Binary: trunk+pole → 1 class | Modernised binary baseline |
| 3 | C | YOLOv11-seg (COCO pretrained) | Multiclass: trunk, pole distinct | The contribution |

Two isolated comparisons emerge:

- **A ↔ B:** **pipeline-generation** effect at fixed binary labelling. Answers: does the modernised binary pipeline as a whole change binary-mask performance? **Not an architecture effect** — see the correction below.
- **B ↔ C:** class-structure effect at fixed YOLO architecture. Answers: does making class identity available (and using it downstream) improve centreline detection?

> **Correction (9 August 2026, additive — the two comparisons above are unchanged; the A ↔ B *label* and the isolability claim that followed are corrected).** A and B differ in at least **thirteen** respects (D006/D021): architecture, pre-training corpus, segmentation paradigm, optimiser, learning rate (100×), weight decay, schedule, epochs, batch size, loss, early-stopping patience, augmentation policy, output representation. **A ↔ B is therefore a baseline-versus-modernised-pipeline contrast and isolates nothing.** Only **B ↔ C** is controlled — same backbone, same hyperparameters, same data, same augmentation, differing solely in label granularity. The design remains defensible against confounded-comparison criticism **because the class-structure question rests on B ↔ C alone**, not because both axes are isolable. The superseded sentence read: *"The two axes are independent, uncorrelated with each other, and each isolable — the methodological property that makes the design defensible against confounded-comparison criticism."*

## 4. Pipeline architecture

Three stages, mirrored in the folder structure (`segmentation/`, `geometry/`, `control/`) and in the three-strand evaluation (`evaluation/`):

1. **Perception** — segmentation model produces either a per-pixel foreground mask (Phase A) or per-instance detections with polygon masks (Phases B and C).
2. **Geometry** — points (foreground pixel clusters or instance centroids) split into left/right sides → RANSAC line fitting per side → centreline as bisector.
3. **Control** — offline PID controller consumes lateral and heading errors from the centreline estimate → produces angular-velocity (yaw-rate) commands.

Downstream geometry accepts two input paths (pixel mask, instance detections) converging on the same clustering + RANSAC step. Phase C adds class-aware fallback logic to the geometry stage (see Section 6).

## 5. Three-strand evaluation

All strands stratified by canopy state (bare-vine: march/april prefixes; canopy: may/color_image prefixes):

| Strand | Metrics | Statistical treatment |
|---|---|---|
| Perception | Per-arm intrinsic: mIoU for Phase A; mAP@50, precision, recall for Phases B and C | Reported per-arm; NOT cross-arm-compared (different formulations) |
| Geometric | Centreline vs the autonomous driven-path trajectory *(BLT ran autonomous GPS/topological navigation, Polvara 2024 — not teleoperated; DECISIONS D014 amendment / D-F)*: RMS lateral error, heading error, frame-level success rate | Cross-arm comparable. Bootstrap CIs over per-frame differences. Effect sizes reported. |
| Command-level | PID command smoothness: cross-arm RMS frame-to-frame yaw-rate change, command jitter, saturation rate *(no vision-derived command reference — the platform drove autonomously; comparison is across arms; DECISIONS D014 amendment)* | Cross-arm comparable. Bootstrap CIs, effect sizes. |

**Perception metrics are not cross-arm-compared** because Phase A produces per-pixel segmentation and Phases B/C produce per-instance detections — mIoU and mAP@50 measure fundamentally different things. Perception is reported per-arm as internal training validation.

**Geometric and command-level metrics ARE cross-arm-compared** because all three arms feed identical downstream stages. This is the level at which the multi-arm comparison happens and where the research question is answered.

P-values excluded (small test set makes them unreliable and over-promising). Bootstrap CIs and effect sizes carry the statistical inference.

## 6. Phase C downstream sweep

Phase C's advantage over Phase B lies not just in richer detections but in how the geometry stage uses them. Three configurations tested to attribute the effect:

- **Config A** — trunk primary, pole fallback: use trunk detections for each side; if trunk count on a side falls below threshold T, fall back to trunk + pole combined for that side
- **Config B** — pole primary, trunk fallback: symmetric to Config A with roles reversed
- **Config C** — class-agnostic: trunk + pole detections treated as one pool (no class-aware logic)

**Selection rule:**
- T sweep on validation set only: T ∈ {1, 2, 3, 5, 8, 12} instance counts
- Config C has no T parameter (evaluated once)
- Best (config*, T*) selected on val by RMS lateral error to the driven-path trajectory *(BLT autonomous, Polvara 2024; D-F)*
- Test set evaluated **once** at locked (config*, T*)

**Attribution story enabled by the three configs:**
- If B ≈ C in metrics → training on distinct classes doesn't improve detection quality itself
- If C ≈ A/B in metrics → downstream class-aware logic doesn't matter; multiclass model just detects better
- If C < A/B in metrics → class-aware fallback logic is where the multiclass advantage originates

This 3-way comparison earns distinction-level attribution under CRG LO6 ("robust interpretation of research findings").

## 7. Phase completion criteria

### Phase A complete
- [ ] U-Net binary trained; best checkpoint locked (val mIoU recorded)
- [ ] Test set evaluated **once**
- [ ] Perception + geometric + command metrics produced, canopy-stratified
- [ ] Metrics summary appended to DECISIONS.md

### Phase B complete
- [ ] YOLOv11-seg binary trained; best weights locked
- [ ] Test set evaluated **once**
- [ ] Perception + geometric + command metrics produced, canopy-stratified
- [ ] Metrics summary appended to DECISIONS.md

### Phase C complete
- [ ] YOLOv11-seg multiclass trained; best weights locked
- [ ] Downstream sweep on val: 3 configs × 6 T values; (config*, T*) recorded
- [ ] Test set evaluated **once** at (config*, T*)
- [ ] Sensitivity analysis figure produced (metric vs T for each config)
- [ ] Three-arm pairwise comparisons produced with bootstrap CIs and effect sizes
- [ ] Metrics summary + attribution paragraph appended to DECISIONS.md

## 8. Working rules (LOCKED from A1; carry throughout A2)

1. Every claim defensible
2. No redundancy — every word earns its place
3. Citations confirmed before use (no hallucination)
4. Critical evaluation evident, not just description
5. Limitations of existing work explicit
6. Flow: motivation → formulation → experiment → discussion
7. Operational recommendation present (evidenced design recommendation)
8. One sentence per cited paper in lit survey, with specific platform + task
9. Results discussion ties to literature
10. Data confirmed; ask when uncertain; no hallucination
11. **No directional framing of comparison results before Results chapter** (added 28 Jun 2026)

**Workflow:** Claude drafts → Edosa reviews → explicit approval → changes applied. Edosa also makes manual edits independently and flags them.

## 9. Honest known limitations (must surface in A2 dissertation)

Not blockers — surfaced because full data exploration and design refinement is what happened. Acknowledging openly rather than hiding is what distinction-level methodology looks like.

1. **"Poles remain visible" framing retired.** A1 asserted poles remain visible while trunks are occluded. Measured retention: trunks 35%, poles 24%. Both degrade; both retain enough signal for class-aware combination. The contribution argument survives — it never depended on relative pole/trunk robustness.
2. **Reference trajectory is the platform's autonomous driven path, not algorithmic ground truth.** The platform navigated by GPS/topological waypoints (Polvara 2024), not vision (DECISIONS D014 amendment; F016/F027); geometric metrics characterise agreement with that recorded driven path, not deviation from a surveyed true centreline.
3. **Test set is 10% of resplit** (~100 frames; ~50 per canopy bin). Improved from 23 frames under Roboflow default. Bootstrap CIs meaningful; p-values still excluded.
4. **ROS bag is fixed recording.** PID is offline characterisation, not closed-loop. Ziegler-Nichols excluded (requires closed-loop oscillation).
5. **SemanticBLT folds foliage into background.** Cannot use foliage as a navigation cue.
6. **A1 committed to all-6-class multiclass training.** Refined to trunk+pole only, because only these classes feed the downstream RANSAC and the tighter class structure yields a cleaner controlled experiment.
7. **A1 committed to scratch U-Net.** Refined to SMP + ImageNet pretrained encoder.
8. **A1 committed to U-Net for both binary and multiclass.** Refined to three-arm design after supervisor feedback on architecture modernity.

## 10. Refinements from A1 proposal (for A2 Methodology chapter)

The A2 Methodology chapter must contain a "Refinements from proposal" subsection. Each refinement is a case where engagement with data or literature revealed a better approach:

| A1 said | A2 does | Reason |
|---|---|---|
| Multiclass semantic segmentation (U-Net, scratch) | Three-arm design: U-Net binary + YOLO binary + YOLO multiclass | Supervisor feedback on architecture modernity; three-arm design isolates class-structure at fixed architecture (B↔C); A↔B is a pipeline-generation contrast, not architecture-isolating — see D006/D021 correction |
| U-Net trained from scratch | U-Net with ImageNet-pretrained encoder | Reduces training risk; enables direct architectural comparison with YOLO's COCO pretraining; scratch justification (educational depth) less critical when U-Net is one of three arms |
| Multiclass training over all 6 annotated classes | Multiclass training on trunk + pole only | Only trunk and pole feed downstream; tighter controlled experiment; all-6 kept as optional supplementary |
| Roboflow default split (95/5/2, 966/46/23) | 70/20/10 stratified resplit with augmentation-leakage guard | Supervisor feedback; 23 test frames insufficient for statistical inference |
| Poles remain visible while trunks are occluded | Both classes degrade across canopy states; class-aware combination extracts complementary information | Empirical data contradicted the A1 framing (pole retention 24% vs trunk 35%) |

## 11. Related documents

- `STATUS.md` — current progress tracker and handover document
- `DECISIONS.md` — running decisions log with rationale (feeds A2 Methodology directly)
- `PHASE_A_SPEC.md` — U-Net binary implementation contract
- `PHASE_B_SPEC.md` — YOLO binary implementation contract
- `PHASE_C_SPEC.md` — YOLO multiclass implementation contract
- `Masters_Dissertation_Proposal.pdf` — A1 proposal (submitted; source of truth for original research question)
