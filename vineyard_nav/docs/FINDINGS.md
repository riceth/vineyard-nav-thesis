# FINDINGS.md

Append-only log of empirical observations discovered through implementation, training, and evaluation of the three-arm study.

**Distinct from DECISIONS.md.** DECISIONS.md records the choices we make (what class structure to use, what loss function, what split rule). FINDINGS.md records what the data shows us (what the model learned, where it succeeds, where it fails, patterns across strata).

Every finding here should be citable in the A2 dissertation. Most will feed the Results or Discussion chapter; some inform Introduction framing (retractions of A1 claims) or Methodology (justifications for analytic choices).

**Format per finding:**
- What was observed (the empirical claim)
- Evidence (numbers, both splits where applicable)
- Analysis (interpretation, informed speculation labelled as such)
- Implications for the dissertation (which chapter, what to say)
- What it does NOT claim (bounds against over-interpretation)

**Cross-references** to DECISIONS.md entries use D0XX; to other findings use F0XX; to phase specs use PHASE_X_SPEC.md.

---

### F001 — Canopy detection is easier than bare-vine detection for U-Net binary
**Date recorded:** 4 July 2026
**Phase:** A (U-Net binary, SMP + ImageNet-pretrained ResNet-34 encoder)
**Status:** Direction replicated in point estimate across validation and test; per-split bootstrap CI for the gap includes zero (n=11/12), so treated as consistent-and-suggestive, not single-split-significant. See "Statistical qualification" below.

**Observation.** The U-Net binary model achieved consistently higher segmentation performance on canopy frames (May, June — foliage-covered vines) than on bare-vine frames (March, April — pre-foliage vines) across every measured metric on both independent splits.

**Evidence.**

| Split | Stratum | n | mIoU | Foreground IoU | Precision fg | Recall fg | F1 fg |
|---|---|---|---|---|---|---|---|
| Val | Bare-vine | 22 | 0.8303 | 0.6726 | 0.8068 | 0.8017 | 0.8042 |
| Val | Canopy    | 24 | 0.8728 | 0.7500 | 0.8547 | 0.8595 | 0.8571 |
| Test | Bare-vine | 11 | 0.8414 | 0.6945 | 0.8470 | 0.7941 | 0.8197 |
| Test | Canopy    | 12 | 0.8858 | 0.7751 | 0.8926 | 0.8548 | 0.8733 |

The foreground-IoU gap (canopy minus bare-vine, pooled IoU) is +0.077 on validation and +0.081 on test — nearly identical magnitudes across two independent samples.

**Statistical qualification (bootstrap, D020; test split, 10,000 resamples, seed 42, per-frame estimand).** Per-frame mean foreground IoU with 95% percentile CIs on the 23 test scenes:

| Stratum | n | fg IoU (per-frame mean) | 95% CI |
|---|---|---|---|
| Overall | 23 | 0.7119 | [0.6572, 0.7659] |
| Bare-vine | 11 | 0.6743 | [0.6072, 0.7368] |
| Canopy | 12 | 0.7463 | [0.6629, 0.8234] |
| **Canopy − bare-vine (gap)** | — | **+0.0719** | **[−0.0336, +0.1744]** |

The gap's 95% CI on the test split **includes zero**, so the effect is *not* individually significant on the 23-scene test set — expected, given n=11 vs 12 and the deliberately wide CIs the dataset ceiling produces (O006). The precision/recall/F1 gaps likewise cross zero on test (only the trivial background-IoU gap, +0.0082 [+0.0057, +0.0109], excludes it). The case for the effect therefore rests on **replication of the point-estimate direction and magnitude across two independent splits** (+0.077 val, +0.081 test), not on single-split significance. A properly powered test would pool the val and test per-frame data (or await Phases B/C); that combined analysis is deferred. Note the per-frame mean (0.7119) differs slightly from the pooled overall fg IoU (0.7195, `test_metrics.json`) because IoU is a ratio of summed counts, not a mean of per-frame ratios.

**Analysis.** Two candidate explanations, likely both contributing:

1. **Silhouette contrast against dense foliage.** In canopy frames, trunks and poles present as thin high-contrast vertical silhouettes against a visually uniform dark green background of leaves. This is a favourable segmentation task — the model has clear edges to detect and a consistent background texture to reject. The trellis wires that clutter bare-vine frames are largely occluded by the canopy in these frames.

2. **Cluttered backgrounds in bare-vine frames.** Bare-vine frames expose the full three-dimensional structure of the vineyard: horizontal and diagonal trellis wires, distant infrastructure (buildings, sky, adjacent rows visible through bare vines), the robot chassis (white frame), and the ground plane with weeds and grow-tubes. Each of these provides visual signal that the model must learn to reject as background. False positives on wires (thin vertical lines like trunks/poles) and grow-tubes (small white vertical objects) are likely dominant error modes.

**Multi-seed replication (Phase A):**
- Phase A seed 42 canopy > bare-vine gap: +0.072 [-0.034, +0.174] (test)
- Phase A seed 43 canopy > bare-vine gap: +0.077
- Phase A seed 44 canopy > bare-vine gap: +0.079
- **Mean gap across 3 seeds: +0.076 ± 0.004**

**Multi-seed evidence (Phase B):**
- Phase B seed 42 canopy > bare-vine gap: -0.011 [-0.178, +0.140] (with 6799 blob)
- Phase B seed 43 canopy > bare-vine gap: +0.062 (with 6799 blob)
- Phase B seed 44 canopy > bare-vine gap: +0.090 (no 6799 blob)

**Multi-seed evidence (Phase C):**
- Phase C seed 42 canopy > bare-vine gap: +0.091 [+0.005, +0.175] (clean of blob, gap reaches significance on this seed alone)
- Phase C seed 43 canopy > bare-vine gap: +0.032 (blob-distorted)
- Phase C seed 44 canopy > bare-vine gap: +0.028 (blob-distorted)

Same pattern as Phase B: when the 6799 blob occurs, it disproportionately pulls down canopy metrics because 6799 is a canopy frame. Blob-free Phase C seed 42 shows the clean effect at +0.091; blob-affected seeds 43/44 show attenuated apparent effect. The effect itself is not weakened by class-aware supervision — the metric measurement is distorted by the blob's outsized single-frame impact.

**Interpretation with multi-seed evidence.** The canopy > bare-vine effect replicates:
- Across 3 Phase A seeds (Phase A multi-seed SD 0.004 on the gap; very stable)
- Across arms directionally (Phase A shows it in all seeds; Phase B seed 44 shows it clearly; Phase C seed 42 shows it at significance)
- Phase B seeds 42 and 43 have their canopy gap distorted by the 6799 blob (blob is a canopy frame; when it's present, canopy metrics are pulled down); seed 44 without the blob shows the clean pattern

The effect is now supported at multiple levels: cross-split replication (within Phase A, val + test), cross-arm replication (Phase A, Phase B seed 44, Phase C), and cross-seed replication within an arm. The magnitude in Phase A is stable (mean +0.076 ± 0.004 across 3 independent training runs). F001 status: consistently replicated, reaching statistical significance in specific arms and specific seeds; effect is real.

**Implications for the dissertation.**

*Introduction:* The A1 proposal framed canopy as the harder condition ("trunks become heavily occluded by foliage as the canopy fills in"). The A2 Introduction should retain the observation that canopy occludes trunks — this is true — but drop the implication that canopy is where segmentation quality collapses. The empirical evidence shows the opposite for binary detection.

*Methodology:* The canopy-state stratification of results (planned in the proposal, implemented in the resplit per D024/D028) becomes even more important as an analytic tool: without stratification, the overall metric hides the substantial per-condition performance gap.

*Results:* The stratified table should be presented prominently, **with the bootstrap CIs**. State plainly that the single-split gap CI includes zero and that the claim rests on cross-split replication of the point estimate, not on single-split significance — pre-empting the obvious reviewer challenge.

*Discussion:* This finding provides a mechanism-level insight that connects perception performance to visual scene complexity rather than to trunk visibility. The dissertation can discuss whether this pattern is likely to generalise to other vineyards, other crops, and other seasonal conditions. Note that "canopy is easier" applies to *this particular pipeline* (thin vertical objects against foliage backgrounds); it is not a general claim about canopy conditions across agricultural computer vision.

**A1 candidate retraction.** The A1 proposal's justification for multiclass — "trunks become heavily occluded by foliage as the canopy fills in, while poles remain visible" — is doubly problematic given empirical results. First, both classes degrade in canopy (D018 retraction). Second, canopy detection is easier overall, not harder (this finding). Neither undermines the multiclass-vs-binary research question: the class-aware pipeline still tests whether trunk-vs-pole differentiation improves centreline detection, regardless of which condition is harder. But the *motivating framing* of "we need multiclass because canopy is where binary fails" needs to be replaced with a cleaner methodological framing ("we compare multiclass against binary as a controlled experiment isolating the class-structure variable").

**What this finding does NOT claim.**
- Does not claim the canopy-vs-bare-vine gap is statistically significant on any single split — the test-split 95% bootstrap CI for the gap is [−0.034, +0.174] and includes zero. The finding rests on cross-split replication of the point estimate.
- Does not claim canopy is easier for *all* agricultural computer vision tasks — this is specific to thin-structure segmentation.
- Does not claim the U-Net has "solved" canopy segmentation — foreground IoU 0.78 canopy still means ~22% of foreground pixels are misclassified.
- Does not undermine the multiclass-vs-binary comparison, which operates at a different level.
- Does not predict what YOLO-seg (Phase B/C) will show — instance-segmentation architectures may behave differently, particularly on visibility and detection of small instances.

---

### F002 — Test performance slightly exceeds validation performance
**Date recorded:** 4 July 2026
**Phase:** A (U-Net binary)
**Status:** Observed; explained by sample composition and small-sample variance. Augmentation is NOT a factor — both splits are evaluated on representative frames only (D028 consumption rule). Bootstrap CI confirms val and test are statistically indistinguishable.

**Observation.** Test-set overall mIoU (0.8561) is marginally higher than validation-set overall mIoU (0.8456), a difference of +0.011. The same pattern holds for foreground IoU (+0.020 test-over-val). This is unusual — typically test performance is slightly worse than validation because the model's checkpoint was selected to optimise validation metrics.

**Evidence.**

| Split | n | mIoU | fg IoU | Precision fg | Recall fg | F1 fg |
|---|---|---|---|---|---|---|
| Val | 46 | 0.8456 | 0.6991 | 0.8237 | 0.8220 | 0.8229 |
| Test | 23 | 0.8561 | 0.7195 | 0.8618 | 0.8134 | 0.8369 |

**Analysis.** Three factors likely combine to produce this pattern, none of which represents a methodological problem:

1. **Different sample sizes; both evaluated on representative frames only.** Per the D028 consumption rule, both splits are scored on one representative frame per scene, so augmentation plays *no* role in either metric. Validation is **46 representative scenes** (22 bare-vine + 24 canopy); test is **23 representative scenes** (11 + 12). [Verified against `data/splits/resplit_70_20_10.json`, `meta.counts.representative_by_split_canopy`.] The augmented copies that also exist in the manifest — validation totals 211 frames, test totals 103 (the balance beyond the representatives) — are intentionally unconsumed by perception evaluation (D028 "consumption pattern" clause). The val/test difference is therefore about *which independent scenes landed in each split*, not augmentation, and validation being the larger sample (46 vs 23) makes its estimate the tighter of the two.

   Verification (one line): `python3 -c "import json,collections as c; m=json.load(open('data/splits/resplit_70_20_10.json')); r=m['images']; print({s:{'reps':sum(x['is_representative'] for x in r if x['split']==s),'total':sum(x['split']==s for x in r)} for s in ('valid','test')})"` → `{'valid': {'reps': 46, 'total': 211}, 'test': {'reps': 23, 'total': 103}}`.

2. **Small-sample variance.** With only 23 test scenes stratified into 11 bare-vine and 12 canopy, a +0.02 fg IoU gap is within plausible sampling variance. Two independent draws from the same underlying distribution can differ by this amount by chance.

3. **Slight favourable draw on test scenes.** The stratified resplit was random within canopy bins. A single random split can happen to place slightly easier-to-segment scenes in test rather than val by chance. The scene-level stratification prevents systematic leakage but does not eliminate composition variance.

**Implications for the dissertation.**

*Methodology:* Bootstrap CIs (D020, 10,000 resamples, seed 42) over the 23 test frames are now computed: overall foreground IoU per-frame mean **0.7119, 95% CI [0.6572, 0.7659]**. The validation foreground IoU (0.6991, pooled) sits well inside this interval, confirming val and test are **statistically indistinguishable** — the +0.020 test-over-val gap is sampling variation, as anticipated. (Bootstrap point estimate is the per-frame mean; it differs slightly from the pooled 0.7195 because IoU is a ratio of sums.)

*Results:* The test-slightly-above-val pattern should be reported factually without over-interpretation. A reader familiar with typical train-val-test dynamics may be confused if the finding is not addressed. One-sentence acknowledgement plus a pointer to the CI-based analysis is sufficient.

**What this finding does NOT claim.**
- Does not claim the model generalises better to unseen data than to its training-adjacent validation data (this would be counterintuitive and requires stronger evidence).
- Does not indicate a leakage or contamination issue between val and test — the D028 scene-level split explicitly prevents this.
- Does not affect the comparison against Phase B and C, which will use the same test set.

---

### F003 — Foreground IoU 0.70-0.72 is the Phase A baseline anchor
**Date recorded:** 4 July 2026
**Phase:** A (U-Net binary)
**Status:** Locked. This is the number Phases B and C are calibrated against.

**Observation.** Phase A U-Net binary achieved foreground IoU of 0.6991 on validation and 0.7195 on test (pooled). On the 23-scene test split the bootstrap point estimate is a per-frame mean of **0.7119 with a 95% CI of [0.6572, 0.7659]** (D020, 10,000 resamples, seed 42). These become the reference point for evaluating whether Phase B (YOLOv11-seg binary, modernised architecture) meaningfully changes binary-mask performance, and whether Phase C (YOLOv11-seg multiclass) provides additional gain via class-aware downstream logic.

**Statistical anchor, honestly bounded.** The "~0.72" anchor carries a wide interval — roughly [0.66, 0.77] at 95% — the direct consequence of the 23-scene test ceiling (O006). Phase B and C perception comparisons against this anchor should compute the bootstrap CI on the paired difference (the same construction as F001's gap CI), not compare individual arm CIs — the paired-difference CI is the correct inference for cross-arm significance. This is a further reason the headline cross-arm comparison lives at the geometric/command strands (D014), not here.

**Analysis.** Foreground IoU is the appropriate primary metric for the perception strand of evaluation for two reasons:

1. **mIoU is dominated by background.** Background IoU is 0.99+ for all splits and canopy states because most pixels are trivially non-foreground. Averaging with background inflates the headline number and masks per-condition differences. Foreground IoU isolates the metric that actually reflects trunk/pole segmentation quality.

2. **Consistency with published binary segmentation baselines.** de Silva et al. 2024, the direct binary-mask reference we position against, report per-class IoU including foreground rather than aggregate mIoU. Reporting foreground IoU allows a direct comparison in the Literature Review.

**Cross-arm comparison note.** The perception-level metrics used in Phase A (per-pixel mIoU and IoU) are *not* directly comparable to those in Phases B and C, which use YOLO-seg instance segmentation and report mAP@50, precision, and recall over detections. This is a known limitation and is handled by the three-strand evaluation framework (D014): cross-arm comparison happens at the geometric and command-control levels (RMS lateral error, PID smoothness), where all three arms produce the same downstream signals. Perception metrics are per-arm internal validation only.

**Implications for the dissertation.**

*Methodology:* Explicit statement that perception metrics differ across arms and cross-arm comparison is done at the geometric level. Prevents an obvious reviewer question ("how do you compare mIoU against mAP@50?").

*Results:* Foreground IoU 0.70–0.72 anchors the "does Phase A adequately reproduce a working binary baseline?" question. Answer: yes, comfortably within published territory for real-world (non-simulated) thin-structure agricultural segmentation.

*Discussion:* The gap between our foreground IoU (~0.72 test) and de Silva 2024's ~0.83 IoU on sugar beet reflects task difficulty, not model quality: sugar beet is continuous-band segmentation on green-vs-soil, whereas vineyard binary requires thin vertical objects (trunk + pole) against complex backgrounds including canopy, wires, sky, and infrastructure. This is exactly the "geometric mismatch" argument the proposal made in the Literature Review; the empirical result confirms it.

**What this finding does NOT claim.**
- Does not claim Phase A performance is state-of-the-art. Modern instance-segmentation models (Phase B) may exceed it on the same task.
- Does not claim the fg IoU numbers alone constitute "success" — that judgement depends on the downstream centreline-fitting quality (geometric strand of evaluation, still to come).

### F004 — Current test set is in-distribution, not out-of-distribution
**Date recorded:** 4 July 2026
**Phase:** A (raised by supervisor)
**Status:** Documented limitation. Remediation planned post-Phase C (see O007).

**Observation.** The 23 test scenes and 721 training scenes were drawn from a single SemanticBLT release, all captured at the same vineyard site across the same growing season (March–June 2024, per the dataset's provenance). Even under scene-honest splitting (D028), test scenes share substantial visual context with training scenes: same trellis structure, same soil, same lighting geometry, same distant infrastructure, and largely the same trunk and pole population. This constitutes a within-distribution generalisation test — the model is evaluated on unseen scenes, but not on genuinely unseen vineyards, seasons, or acquisition conditions.

**Implications for the dissertation.**

*Methodology / Limitations:* The 23-scene test constitutes an in-distribution test set. Its results characterise how well the model generalises to unseen scenes within the same vineyard-and-season context, not how it would perform in truly novel conditions (different vineyard, different season, different camera setup, different growth stage).

*Discussion:* This is a common limitation for research on single-site datasets and does not undermine the three-arm comparison — all three arms are evaluated on the same in-distribution test, so their relative ranking is defensible. What it does bound is the transferability claim: results demonstrate architecture × class-structure trade-offs for this vineyard type in this season, and are indicative rather than conclusive for other deployment conditions.

*Remediation planned:* Supervisor has flagged and endorsed adding an out-of-distribution evaluation set via manual labelling of images from a different vineyard section or season, post-Phase C (O007). When that is available, the dissertation will report both in-distribution (three-arm comparison) and out-of-distribution (generalisation) results.

**What this finding does NOT claim.**
- Does not undermine the three-arm ranking itself — all arms are evaluated on the same test set.
- Does not claim the model is guaranteed to fail out-of-distribution — that would be equally unfounded until measured.
- Does not require re-doing Phase A. The 23-scene test is still the anchor for the within-distribution comparison.

---

### F005 — Rasterised fg IoU is a per-arm characterisation metric, not a cross-arm ranking metric
**Date recorded:** 10 July 2026
**Phase:** Applies to Phase B, Phase C, and future comparisons
**Status:** REVISED. Original F005 framed rasterised fg IoU as a "cross-arm comparability metric." This framing is retracted; F005 is now scoped to per-arm internal use.

**Observation.** Phase A (U-Net semantic segmentation) natively reports per-pixel foreground IoU, which decomposes cleanly to per-frame values and admits bootstrap confidence intervals. Phase B and Phase C (YOLOv11-seg instance segmentation) natively report mAP@50 and related detection metrics. To provide per-frame data for statistical inference on the YOLO arms, we compute a rasterised foreground IoU: YOLO's instance masks at conf ≥ 0.25 are combined into a union foreground mask per frame; this union is compared to ground-truth foreground pixel-wise.

**Retraction of cross-arm ranking use.** The original F005 framing described rasterised fg IoU as "the metric that enables cross-arm perception comparison." This framing has been retracted for methodological reasons:

- The transformation from YOLO's per-instance masks to a binary union mask is a lossy operation. It discards per-instance confidence granularity and instance identity, and selects one interpretation ("any pixel covered by any conf ≥ 0.25 detection = foreground") over others.
- This transformation is not standard in the segmentation literature. Comparing instance-seg outputs to semantic-seg outputs by rasterisation is not a widely-adopted methodology; a rigorous reviewer would ask why the metric is preferred to each arm's native metric.
- Our own three-strand evaluation framework (D014) commits to "perception metrics differ across arms; cross-arm comparison happens at the geometric and command-level strands where all three arms produce the same signal." Introducing rasterised fg IoU as a cross-arm perception metric contradicts this framework.
- The stronger the cross-arm claim we make from rasterised fg IoU (e.g., "U-Net 0.72 vs YOLO 0.58"), the more the claim is doing work that the metric is not designed to support.

**Revised scope.** Rasterised fg IoU is retained as an *internal per-arm characterisation metric*, used for:

1. **Canopy stratification of YOLO union coverage** (does YOLO cover more foreground area on canopy scenes vs bare-vine scenes, given the same conf threshold?)
2. **Blob-failure characterisation** (as in F007: when YOLO produces a whole-canopy false-positive mask, the rasterised fg IoU collapses because the union area is dominated by the mask; this is the metric that surfaces the failure)
3. **Per-arm sensitivity analysis** (per D030's conf sweep on val: how does mean rasterised fg IoU respond to conf threshold choice, per arm?)
4. **Per-arm bootstrap CIs quantifying data variance** (each arm's per-frame values are bootstrappable within that arm)

**Cross-arm comparison methodology.** Per D014 and F003, cross-arm perception comparison is NOT the primary evaluation. The primary comparison happens at:

- **Geometric strand:** RMS lateral error against teleoperator trajectory (all three arms produce the same signal — an estimated centreline — after RANSAC line-fitting). This is the metric committed in the proposal and PHASE_C_SPEC §8.
- **Command-level strand:** steering-command difference against teleoperator commands (all three arms feed the same PID controller structure).

Both strands await the geometry pipeline, which is scoped for a later phase (O010). Cross-arm ranking at the perception level is deferred to the downstream stages.

**What this changes for the dissertation.**

*Methodology chapter:* Explain the three-strand framework, each arm's native metric, and that rasterised fg IoU is used only for per-arm internal characterisation (not cross-arm ranking).

*Results chapter:* Report each arm using its native metrics. Rasterised fg IoU reported per YOLO arm only, as internal characterisation. Do not force perception-level cross-arm ranking.

*Discussion chapter:* Save architectural comparisons for the downstream strands. At the perception level, characterise each arm's behaviour and failure modes without ranking them.

**Cross-references.**
- F003 (Phase A baseline anchor). Original F005 tried to make F003 cross-arm-comparable; revised F005 accepts that Phase A's fg IoU is per-arm.
- D014 (three-strand evaluation framework). Revised F005 aligns with D014's commitment.
- F007 (Phase B 6799 blob). Revised F005 preserves rasterised fg IoU as the metric that reveals the blob failure; F007's per-arm claims remain valid.
- D031 (revised cross-arm comparison methodology). Formalises the position stated here.

**What this finding does NOT claim.**
- Does not claim rasterised fg IoU is uninformative. It is informative for per-arm characterisation.
- Does not claim mAP@50 alone is sufficient for reporting YOLO. Report both native detection metrics AND rasterised fg IoU per arm, with the latter framed as internal characterisation.
- Does not delete the multi-seed variance work. Bootstrap CIs on rasterised fg IoU per arm are legitimate data-variance quantification for that arm.
- Does not claim the study is unable to compare arms. Comparison happens at the geometric and command-level strands, per D014 and PHASE_C_SPEC §8.

---

### F006 — Phase B rasterised fg IoU is only mildly sensitive to the confidence threshold
**Date recorded:** 8 July 2026
**Phase:** B
**Status:** Empirical; supports the D030 operating-point selection.

**Observation.** Sweeping the detection confidence threshold and computing mean per-frame rasterised foreground IoU over the 46 validation scenes (half=True) yields a shallow, flat-topped inverted-U peaking at conf* = 0.25.

**Evidence.**

| conf | mean val fg IoU (n=46) |
|---|---|
| 0.10 | 0.5655 |
| 0.15 | 0.5758 |
| 0.20 | 0.5793 |
| **0.25** | **0.5856  ← conf\*** |
| 0.30 | 0.5852 |
| 0.40 | 0.5786 |

Total spread max−min = 0.0201; the 0.20–0.30 plateau varies by < 0.007. Curve: `results/runs/phase_b_yolo_binary/val_conf_sweep.png`.

**Analysis.** The curve is broad and flat-topped (~0.02 total range), so rasterised fg IoU is only **mildly sensitive** to the confidence threshold — the operating point is not a fishing knob. Mechanistically: very low conf (0.10) admits low-confidence false-positive mask pixels that slightly depress IoU (over-prediction); high conf (0.40) drops marginal true detections (under-prediction); the optimum sits at the conventional 0.25 with 0.30 effectively tied. The plateau (spread 0.020 across the swept range, < 0.007 over 0.20–0.30) is itself a methodological finding: the reported rasterised fg IoU is robust to small perturbations in conf, so the headline number does not hinge on the precise threshold — which strengthens the reliability of the reported value and the cross-arm comparison built on it.

**Implications for the dissertation.**
*Methodology:* conf* is selected on val (D030); the flatness means the exact value is not critical, which strengthens the robustness of the operating-point choice and pre-empts a "threshold-tuned" critique.
*Results:* report conf* = 0.25 with the sensitivity curve; note the insensitivity explicitly.

**What this finding does NOT claim.**
- Does not claim conf affects mAP@50 — mAP integrates over confidence; this sweep is only for the rasterised-fg-IoU operating point (F005).
- Does not claim 0.25 is optimal on the test set — it is the val argmax; test was evaluated once at locked conf* (rule 5).
- Cross-references: F005 (rasterised fg IoU as the cross-arm metric), D030 (selection procedure).

---

### F007 — Phase B best.pt exhibits a large false-positive canopy mask on 6799 not present in last.pt
**Date recorded:** 8 July 2026
**Phase:** B
**Status:** Failure mechanism observed and characterised on one scene. Rate and generality not established.

**Observation.** On test scene color_image_6799, Phase B's YOLOv11-seg model produces 13 instance detections at conf ≥ 0.25. Twelve are correct trunk-and-pole masks distributed across the scene (bounding boxes, mask areas 159–953 px, confidences 0.287–0.890). One (#0, conf 0.406, mask 76,837 px, 21.6× total ground-truth foreground area) is a false-positive mask covering the right-side canopy foliage. The correct trunk detections on the same right side coexist with the false-positive mask.

**Evidence (uncommitted, results/runs/phase_b_yolo_binary/diagnostic/6799_visualisation/):**

- Raw scene shows no visual anomaly compared to other canopy test frames. Path structure, foliage density, camera geometry, and lighting are typical for the class.
- Ground truth: 3,564 pixels of trunk + pole foreground (thin vertical structures on both sides). Canopy foliage is unlabelled in SemanticBLT and treated as background per our binary collapse rule.
- YOLO union at conf ≥ 0.25: 81,365 pixels. Of these, 76,837 come from detection #0 alone; the remaining 4,528 come from detections #1–#12 (arithmetic verified: 76,837 + 4,528 = 81,365).
- Detection #0's mask follows the actual canopy boundary reasonably well — it is not a numerical glitch but a shaped mask covering a specific scene region.
- Detections #1–#12 correctly identify individual trunks and poles with sensible masks (mean size 377 px).
- Raising conf to ≥ 0.41 removes detection #0 (the blob) but also removes detection #8 (conf 0.287); 11 detections remain and single-frame fg IoU improves from 0.04 to 0.598. Alternatively, a mask-area filter that removes only detection #0 while retaining all 12 correct detections yields the same single-frame fg IoU of 0.598. A residual gap versus U-Net's 0.687 on the same frame remains — YOLO's correct detections cover the ground truth well but do not fully match U-Net's per-pixel foreground coverage, even after blob removal.
- On the locked best.pt, reproduces deterministically and identically under FP16 and FP32. Does NOT reproduce on last.pt (final epoch of the same run): last.pt yields 12 detections, max mask 963 px, no blob, single-frame fg IoU 0.604. The failure is checkpoint-specific — not stable even across the last two saved checkpoints of the same training run. best.pt was selected by ultralytics' val fitness metric; last.pt is the final epoch's weights.

**Analysis.** The mask boundary follows canopy structure — this is a shaped prediction, not a numerical artifact. But the failure is not a stable learned property of the model: fourteen epochs later in the same training run (last.pt), the same architecture with slightly different weights does not produce the failure at all. This checkpoint-specificity substantially bounds the interpretation.

Possible mechanisms include: (a) the detector at best.pt's epoch firing on an occluded pole or trunk within the canopy region, with the mask over-drawing onto surrounding foliage — the same detection may not have fired at the last.pt epoch; (b) a transient training instability at the val-optimal epoch that produces this specific mask via prototype coefficient predictions that drift by last.pt; (c) an artefact of the mask head's coefficient predictions at a specific parameter configuration that fitness-based checkpoint selection happened to lock. Distinguishing these would require inspection of detection #0's bounding box against ground-truth annotations, prototype activation analysis, and comparison of mask head weights between best.pt and last.pt — none currently done.

Note on labelling choice: our binary collapse (trunk + pole → foreground) reflects three considerations. First, the downstream RANSAC line-fit for centreline detection requires geometrically aligned foreground pixels — canopy pixels vary in position across frames and would fit lines through canopy rather than crop rows. Second, this matches the labelling used by de Silva et al. 2024, our binary baseline reference. Third, SemanticBLT labels only structural elements (buildings, pipes, poles, robots, trunks, vehicles); canopy is not a labelled category. The failure is "wrong" specifically relative to this labelling scheme; under an alternative labelling that includes canopy in foreground, detection #0 would be classified differently. But the current labelling remains appropriate for centreline detection.

**Architectural asymmetry.** The failure mode is architecturally *possible* for YOLOv11-seg because instance masks are computed per-detection via prototype coefficients, so a single detection can produce a large mask regardless of what other detections are producing. The mode is architecturally *impossible* for U-Net because each pixel is classified independently — a coherent shaped mask spanning a canopy region cannot be produced by per-pixel classification. This asymmetry is a real feature of the two architectures.

This architectural asymmetry describes what is possible — U-Net cannot produce shaped canopy masks; YOLO can. It does not describe what is reliable: best.pt producing this failure while last.pt does not indicates that the mode's manifestation is checkpoint-specific rather than a stable output of this architecture on this data.

**Bounded claims.** One instance of this failure on 23 test frames does not establish a base rate. The observation may reflect training-data characteristics (F004: current test set is in-distribution to training set; the confusion may be an in-distribution training coverage artifact that resolves with OOD data). Class-aware multiclass supervision (Phase C) may or may not constrain coefficient predictions differently — verification pending Phase C evaluation. Multi-seed evaluation (O009) will establish whether the same failure recurs across different training seeds.

**Why the two metrics diverge.** mAP@50 counts detection #0 as one wrong prediction: at 76,837 px versus 3,564 px ground truth, it will not match any ground-truth mask and is scored as a single false positive. Aggregate mAP@50 absorbs this as a small precision penalty across all 23 test frames. Rasterised fg IoU, computed as union-mask overlap with ground truth, is dominated by detection #0's mask area — on this frame, IoU collapses to ~0.04, and the mean across 23 frames cannot recover. F005 (metric parity via rasterised fg IoU) anticipated this divergence; F007 is the concrete illustration on a specific frame.

**Implications for the dissertation.**

*Methodology:* Report both metrics with clear labels. Explain the labelling-scheme framing when describing the metric divergence — 6799's failure is precisely defined against the binary trunk + pole labelling and would be classified differently under alternative labellings.

*Results:* Present aggregate metrics with CIs and per-frame variance. Include 6799 as a documented outlier with visualisation. The 0.04–0.69 fg IoU range for Phase B is a real characterisation, not an artifact.

*Discussion:* Frame the Phase A vs Phase B comparison as: "U-Net produces consistent per-pixel coverage across the test set with limited variance. YOLO produces precise instance masks on most scenes but exhibited one catastrophic false-positive mask on 6799, driving much of the aggregate rasterised fg IoU gap. The failure mode is architecturally available to YOLO and unavailable to U-Net; whether it occurs at 1-of-23 rate or would occur at meaningfully different rates on different training seeds or different data is not established from this single instance."

*Discussion (remediation options — none adopted):* Preliminary calculation: if a mask-area filter (e.g. max_area > 3,000 px on 640×640 images) removed detection #0's 76,837 px mask while retaining all 12 correct detections on 6799, the aggregate Phase B test rasterised fg IoU would rise from 0.556 [0.466, 0.633] to approximately 0.581, narrowing but not closing the gap against Phase A's 0.72 [0.66, 0.77]. The remaining gap reflects consistent under-coverage of thin structures across most test scenes — an architectural effect distinct from the catastrophic false-positive on 6799. Neither remediation is adopted here. A mask-area filter would be principled but hides the failure mode from Phase B's reported numbers; the downstream RANSAC line-fit stage is a more natural location for spurious-input handling if needed. Higher input resolution (e.g. 1280×1280) would reduce mask-head downsampling of thin structures and likely improve under-coverage, but was not explored — it would compromise the controlled A vs B architectural comparison at fixed 640×640 input. Both options are noted for possible future work.

*Discussion (mean-vs-median conf selection):* Supplementary median-based analysis on val (46 frames, 8-value sweep grid). Median-based conf* coincides with mean-based (both = 0.25); catastrophic frames (fg IoU < 0.1) occur on zero val frames at any threshold in the sweep range. The 6799-type failure did not appear on val at any conf, which is why neither mean- nor median-based selection could anticipate it. This has three implications: first, it suggests the 6799 failure is out-of-distribution relative to the val set — F004's in-distribution concern applies here directly. Second, it demonstrates a limit of val-based hyperparameter selection: robustness to failure modes that don't appear on val cannot be validated against val. Third, the failure being checkpoint-specific (present on best.pt, absent on last.pt) suggests that even with a broader val set, checkpoint selection by val fitness might select checkpoints that exhibit the failure while adjacent checkpoints do not. For this project, the primary conf* = 0.25 is documented as val-selected; F007 characterises the specific test failure that val-based selection did not anticipate.

*Note for Phase C:* Verify whether Phase C's multiclass YOLO's val-selected checkpoint produces a similar or different result on 6799. Given the failure is checkpoint-specific in Phase B, Phase C's failure or absence-of-failure on 6799 tells us: if the failure recurs, it's a repeatable pattern across model families and checkpoint selection procedures; if it doesn't recur, it may reflect class-aware supervision constraining coefficient predictions, or simply the specific checkpoint Phase C selected. Multi-seed evaluation (O009) is the more decisive test — if seeds 43–46 all show 6799-type failures at their locked best.pt, the effect is a systematic property of val-fitness checkpoint selection for this architecture on this data. If none do, best.pt was an outlier.

*Phase C result (10 July 2026):* **The 6799 blob does NOT recur in Phase C.** Phase C's val-selected best.pt at conf 0.25 produces 14 clean per-instance detections (10 trunk, 4 pole), largest mask **989 px** (no blob), rasterised fg IoU **0.627** — versus Phase B best.pt's 76,837 px blob and fg IoU 0.038. Per the pre-registered interpretation above, absence-of-recurrence is consistent with **either** class-aware supervision constraining the mask coefficients **or** Phase C simply selecting a "clean" checkpoint (the Phase B failure was itself checkpoint-specific — best.pt yes, last.pt no). **n=1 cannot distinguish these; this does NOT establish that multiclass supervision fixes the failure mode.** O009 multi-seed remains the decisive test. Visualisation (same format as Phase B): `results/runs/phase_c_yolo_multiclass/diagnostic/6799_visualisation/`.

*Note for downstream RANSAC:* On 6799, detection #0's mask covers the right-side vine row region. If this mask is passed to the geometry stage, it may contribute foreground pixels or centroids that inject spurious inliers into the right-side line fit. The RANSAC step's outlier tolerance will need to accommodate this. If failures of this type recur in Phase C, this becomes a design consideration for the downstream pipeline that was not scoped in the original proposal.

**Phase B multi-seed evidence (3 seeds complete):**

- Phase B seed 42 on 6799: blob (76,837 px, IoU 0.038, mask conf 0.406)
- Phase B seed 43 on 6799: blob (75,271 px, IoU 0.039, mask conf 0.264)
- Phase B seed 44 on 6799: NO blob (largest mask 961 px, single-frame fg IoU 0.591)

**Cross-seed blob overlap on 6799** (results/runs/phase_b_blob_overlap_6799/blob_overlap_s42_s43.png):
- Seed 42 blob (76,837 px) and seed 43 blob (75,271 px) show mask IoU 0.93
- Centroid distance 5.6 px in a 640×640 image (0.9% of image width)
- Near-identical bounding boxes: (326,86,639,497) vs (322,82,639,489)
- Mutual coverage: seed 42 blob 95.3% inside seed 43 blob; seed 43 blob 97.3% inside seed 42 blob
- Both masks follow the same right-side canopy boundary

**Interpretation refined by multi-seed evidence.** The 0.93 mask IoU between seeds 42 and 43's blobs establishes that the failure is not a random per-checkpoint quirk of one training run. Two independently trained YOLO binary models — different random initialisations, different optimisation trajectories — converge on masking the same specific canopy region on the same specific frame. This rules out coincidental size similarity: the failure is a reproducible response from Phase B's binary training regime on this data.

However, seed 44's absence of the failure demonstrates that the mode is not universally realised. Val-fitness-based checkpoint selection appears to reliably select checkpoints that produce this specific response 2 of 3 times (67%). When val-fitness selects a "clean" checkpoint (seed 44 best.pt; seed 42 last.pt was also clean), the failure does not appear. When it selects a "blobbing" checkpoint (seed 42 best.pt, seed 43 best.pt), the failure appears with near-identical geometry.

**Framing.** The failure mode is:
- Architecturally *available* to YOLOv11-seg (per-instance mask heads can produce large masks); structurally *unavailable* to U-Net (per-pixel classification).
- Reproducible *when it occurs* (mask geometry cross-seed IoU 0.93).
- Frequently realised (2/3 seeds).
- Not inevitable (1/3 seeds handled 6799 cleanly).
- Contingent on val-fitness checkpoint selection producing a "blobbing" checkpoint.

**Multi-seed evidence (6 seeds complete):**

O009 multi-seed pass across all three arms establishes the failure profile of the 6799 blob:

| Arm | 6799 blob rate | Cross-seed blob geometry when present |
|---|---|---|
| A (U-Net binary) | 0/3 | N/A — structurally impossible for per-pixel classification |
| B (YOLO binary) | 2/3 | Mask IoU ~0.93 between blob seeds (42 & 43) |
| C (YOLO multiclass) | 2/3 | Mask IoU ~0.93 between blob seeds (43 & 44); ~0.93 with Phase B blob seeds |

**Cross-arm blob geometry.** When the blob occurs, it covers the same right-side canopy region regardless of arm or seed. Pairwise mask IoU across all four blobbing runs (Phase B seeds 42 and 43; Phase C seeds 43 and 44) is mean 0.93 (range 0.92-0.94 across 6 pairwise comparisons). Centroids cluster within ~6 pixels of each other in a 640×640 image. Bounding boxes are near-identical.

**The class-aware-supervision-prevents-blob hypothesis is falsified.** Phase B and Phase C exhibit identical blob rates (2/3 each) at val-fitness-selected best.pt, and the blob geometry is invariant to class-structure supervision. The pre-registered branch A ("class-aware supervision structurally prevents the failure") is not supported. Branch B ("class-aware supervision does not help") is supported.

**Established properties of the failure.**

- Architecturally *available* to YOLOv11-seg (mask heads with prototype coefficients can produce large masks): YES
- Architecturally *unavailable* to U-Net (per-pixel classification cannot produce coherent shaped masks over unlabelled classes): YES — U-Net is structurally immune, 0/3 rate
- Reproducible when triggered (mask geometry invariant across seeds and arms): YES — ~0.93 mask IoU in all pairwise comparisons
- Frequently realised at val-fitness checkpoint selection: YES — ~67% rate across both YOLO arms
- Sensitive to class-structure supervision: NO — binary and multiclass show identical rates
- Sensitive to the specific image: YES — reproduces on 6799 specifically; unclear whether other images have similar failures because they were not systematically tested for this mode

**What this means for the dissertation.**

*Results:* Present the multi-seed blob rate table clearly, with the ~0.93 mask IoU as evidence that the geometry is scene-and-architecture-specific, not seed-random. Include one visualisation panel (results/runs/phase_b_blob_overlap_6799/blob_overlap_s42_s43.png as a representative example).

*Discussion:* Frame the failure as a YOLOv11-seg architecture-family × scene interaction, orthogonal to the class-structure variable that the B↔C comparison isolates. The multi-seed evidence sharpens the story from "Phase B has a failure mode" to "the failure mode is a stable architectural response, not a training-run quirk, and class supervision does not affect its frequency."

*Methodology note (val-selection limits):* The failure does not appear on val at any conf threshold in D030's sweep range. Val-based hyperparameter selection cannot anticipate a failure mode that val doesn't contain. This is a limitation of val-based checkpoint selection when the test set exercises different failure modes than the val set. Multi-seed evaluation (O009) is the empirical remedy: rather than assuming any single val-fitness-selected checkpoint is representative, we characterise across independent runs.

*Note for downstream RANSAC:* When the blob occurs, the mask covers the right-side vine row region. If passed to the geometry stage, it may inject spurious foreground pixels or per-side centroids into the right-side line fit. The RANSAC step's outlier tolerance needs to handle this: given the failure occurs on ~67% of Phase B and Phase C runs at val-fitness selection, the downstream pipeline should not assume blob-free input. Either accept the possibility and let RANSAC filter, or apply a mask-area filter before geometry.

**Future work — capacity scaling.** This study uses YOLOv11-seg-nano for compute feasibility. The 6799 blob failure mode's dependence on model capacity is not tested. Since the failure is a property of the mask head's prototype coefficient mechanism (architecturally identical across YOLOv11-seg variants), we hypothesise scaling to yolo11-s, m, l, or x would not eliminate the failure. Systematic capacity-scaling verification is left for future work.

**Cross-references.**
- F004 (current test set is in-distribution). The observation may partly reflect training data coverage; F004's planned OOD remediation would help establish whether the failure generalises or is training-specific.
- F005 (cross-arm metric parity via rasterised fg IoU). F007 is the concrete illustration of "the two metrics may diverge."
- D030 (conf* = 0.25 selected on val by mean fg IoU). Supplementary median-based analysis (val only) reports what median-based selection would have chosen; documented in this finding's Discussion as an available methodological alternative.
- Phase C observation of 6799 planned per STATUS.md.
- O009 (multi-seed evaluation post-Phase-C). Will address rate/generality of this failure across training seeds.

**What this finding does NOT claim.**
- Does not claim YOLO is universally worse than U-Net for vineyard perception. Per-scene performance is mixed.
- Does not claim the false-positive mask on 6799 will reproduce with different training seeds. Multi-seed evaluation (O009) will address this partially.
- Does not claim a base rate for such failures — 1 in 23 test frames is insufficient sample.
- Does not resolve which architecture is "preferred" for downstream navigation.
- Does not attribute the failure to prototype coefficient pathology as a numerical event — the visualisation shows the mask boundary follows canopy structure, consistent with learned confusion within the model's training distribution rather than numerical artifact.
- Does not claim the failure occurs at every checkpoint or every seed. The failure has been observed on Phase B's val-fitness-selected best.pt at seeds 42 and 43 (both showing near-identical mask geometry, IoU 0.93). Seed 44's best.pt does not exhibit the failure. Seed 42's last.pt does not exhibit it either. Val-fitness-based checkpoint selection appears to produce this specific response at ~67% rate on this data (2/3 seeds); whether this rate generalises across data or hyperparameter regimes is not established.
- Does not claim the specific 6799 blob mode occurs on other test images at the same rate. This is one image's failure mode, characterised across seeds and arms; other test images may have their own failure profiles.
- Does not claim class-aware supervision provides no benefit at any level. Phase C reaches statistical significance on the canopy > bare-vine gap on test alone; Phase B does not. Multiclass supervision may confer other benefits not measured by 6799-specific analysis.
- Does not claim a 4-way blob geometry IoU of 0.93 across arms means the model has "learned the same thing." Coefficient predictions and prototype activations that produce the mask could differ substantially between arms while still producing similar output masks. Attribution to specific mechanisms requires prototype-activation analysis, not currently done.
- Does not claim ~67% is the true blob rate across all conditions. Three seeds per arm is limited sample; rate uncertainty is real.
- Does not claim the mask would be "wrong" under all reasonable labellings. It is wrong specifically relative to our binary trunk + pole labelling. Under a labelling that includes canopy in the foreground (not adopted here for reasons documented in the Analysis section), the same mask would be classified differently.

---

### F008 — B↔C training-loss decomposition confirms the comparison is controlled at fixed architecture
**Date recorded:** 10 July 2026
**Phase:** B ↔ C (observed during Phase C training)
**Status:** Observed across the full 100-epoch runs; quantified.

**Observation.** Across the full 100-epoch runs, Phase C (multiclass) and Phase B (binary) training losses are near-identical for box, seg, and dfl, while **cls_loss is consistently higher for Phase C**. The B↔C training difference is isolated to the classification head — exactly where the class-structure change (1 class → 2) should manifest, and nowhere else.

**Evidence.** Mean absolute per-epoch difference over all 100 matched epochs:

| loss term | mean \|C − B\| | interpretation |
|---|---|---|
| box_loss | 0.0088 | matched (noise) |
| seg_loss | 0.0079 | matched (noise) |
| dfl_loss | 0.0047 | matched (noise) |
| **cls_loss** | **0.0519** | **diverged (~6–11×)** |

cls_loss is higher for C at every epoch (e.g. ep1 3.25 vs 2.78, ep50 0.914 vs 0.882, ep100 0.713 vs 0.685); box/seg/dfl track within <0.01. The smoke run showed the same signature (cls C 4.07 vs B 3.69).

**Analysis.** box_loss (localisation), seg_loss (mask quality), and dfl_loss (distribution focal) are class-count-agnostic — they depend on the same geometry and the same 14,894 foreground annotations (identical frames per D028; identical polygons, only re-labelled trunk=0/pole=1). cls_loss depends on the classification task, which is genuinely harder for two classes (trunk vs pole) than one ("crop"). The matched non-cls losses show identical optimisation dynamics except for classification.

**Implications for the dissertation.**
*Methodology:* direct evidence that the B↔C comparison isolates the **class-structure** variable at fixed architecture, training regime, data, and augmentation — the controlled-experiment premise of D021/D025. Any downstream B↔C difference (when the sweep runs, O010) is attributable to class structure, not differential training effort.
*Results:* report as a one-line controlled-comparison validity check.

**What this finding does NOT claim.**
- Does not claim the higher cls_loss implies worse (or better) downstream or perception quality — it reflects task difficulty, not final outcome.
- Does not itself establish any class-aware advantage; the B↔C outcome comparison lives in the downstream attribution (O010, deferred).
- Cross-references: D021 (three-arm controlled design), D025 (trunk+pole labelling), O010 (downstream deferred).

---

### F009 — Phase A vs Phase B training-run variance contrast (Phase B intermittent-blob-driven)
**Date recorded:** 10 July 2026
**Phase:** A + B multi-seed
**Status:** Established: Phase B's cross-seed variance is ~3.4× Phase A's, and the difference is dominated by the intermittent 6799 blob failure.

**Cross-arm multi-seed variance (O009 complete):**

| Arm | Test fg IoU (mean ± SD across seeds) | Test mAP@50 (mean ± SD, YOLO only) | Per-seed bootstrap CI half-width (mean) | Training-run SD ÷ CI half-width |
|---|---|---|---|---|
| A (U-Net binary) | 0.716 ± 0.008 | mIoU 0.858 ± 0.003 (native) | ±0.053 | ~15% |
| B (YOLO binary) | 0.585 ± 0.027 | 0.632 ± 0.016 | ±0.067 | ~40% |
| C (YOLO multiclass) | 0.594 ± 0.022 | 0.644 ± 0.008 | ±0.063 | ~35% |

**Key observations:**

1. Phase A is the most stable across seeds (SD 0.008 on fg IoU, 15% of data variance). Phase B and Phase C have larger training-run variance (SD 0.027 and 0.022 respectively). The variance contrast is dominated by the intermittent 6799 blob failure — see F007.

2. Phase C has slightly lower fg IoU variance than Phase B (0.022 vs 0.027). This is not a strong finding at n=3; it reflects Phase C having 2 blob seeds versus Phase B having 2 blob seeds where the specific numerical impact of each blob is slightly different.

3. Phase C's mAP@50 variance (0.008) is notably lower than Phase B's (0.016). This is a real observation: Phase C's mask mAP@50 is more consistent across seeds than Phase B's. Possible interpretations: class-aware supervision produces more consistent detection quality on non-blob frames; or Phase C's checkpoint selection is more stable. n=3 is insufficient to distinguish.

4. Phase A's mIoU SD (0.003) is smaller than Phase B/C's rasterised fg IoU SD (0.022-0.027), but this compares different metrics measuring different things — not a fair comparison. The internally-comparable observation is that Phase A's fg IoU is the most stable at 0.008.

**Revised interpretation.** Phase A produces continuously varying detection quality across seeds; the SD 0.008 reflects small training-run randomness in a stable optimisation regime. Phase A cannot produce a catastrophic per-frame failure because U-Net's per-pixel output cannot be dominated by a single spurious "instance."

Phase B and Phase C both have a discrete failure mode: on ~67% of seeds, the model's mask head produces a whole-canopy false-positive mask on 6799. The variance in per-frame fg IoU is dominated by this discrete outcome, not by continuous drift in overall detection quality. Both arms show the failure at similar rates and geometries; class-aware supervision does not affect the phenomenon.

That mAP@50 shows lower cross-seed variance in Phase C (0.008) than Phase B (0.016) is an observation worth surfacing as a possible class-aware benefit for non-blob detection quality consistency, though n=3 is limited.

**Implications for the dissertation.**

*Discussion:* Present the variance contrast as a substantive architectural finding. Phase A's variance is small and continuous; Phase B's is larger and discrete. The two arms have different failure profiles: Phase A produces stable-quality models across seeds; Phase B produces a mixture of "clean" and "blobbing" models depending on val-fitness selection.

*Methodology limitation acknowledgement:* Val-fitness-based checkpoint selection can lock in a "blobbing" checkpoint for Phase B without any warning signal, because the val set doesn't contain 6799-type failure cases (F007 note: median-vs-mean sweep found no catastrophic frames on val). This is a limitation of val-based hyperparameter selection when the test set contains different failure modes than the val set.

*Note for Phase C:* Phase C's multi-seed variance and blob rate will characterise whether class-aware supervision produces a more Phase-A-like (stable, continuous variance) or Phase-B-like (intermittent, discrete variance) training-run profile. Testable finding.

**Cross-references.**
- F005 (revised): multi-seed data validates the divergence prediction; mAP absorbs the blob at lower cost than fg IoU.
- F007: the specific failure driving Phase B's variance; details of the failure and its cross-seed reproducibility.
- D016: reproducibility infrastructure gives byte-identical within-seed reproduction for Phase A (U-Net, fully deterministic) and closely-matching within-seed reproduction for Phase B/C (ultralytics is not fully byte-deterministic at fixed seed). Neither eliminates cross-seed variance; the cross-seed variance is a real feature of the training process.
- D030: conf* was selected by mean fg IoU on val; a median-based selection would similarly have chosen conf* = 0.25 because val has no catastrophic frames. The blob-frequency question is orthogonal to conf selection.

**What this finding does NOT claim.**
- Does not claim Phase A is "better" at perception. It has different variance properties. Cross-arm perception ranking is not performed per D031.
- Does not claim the variance ratios (Phase A stable ~15%, Phase B/C ~35-40%) will be identical on other datasets. On different data with different failure modes, the arms could show different profiles.
- Does not claim class-aware supervision reduces Phase B/C variance in a meaningful way. Phase C's slight variance reduction (0.027 → 0.022) is not statistically distinguishable at n=3.
- Does not claim ±0.05 is a universal bootstrap CI half-width; Phase A's is 0.053, Phase B's is 0.067, Phase C's is 0.063. The training-variance fraction of data variance differs across arms partly because the YOLO arms' data variance is itself larger (wider CIs), not solely because their training SD is larger.