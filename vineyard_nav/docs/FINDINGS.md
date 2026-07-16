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
**Status:** Observed; explained by sample composition and small-sample variance. Augmentation is NOT a factor — both splits are evaluated on representative frames only (D028 consumption rule). Bootstrap CI confirms val and test are statistically indistinguishable. Originally a seed-42 finding; O009 multi-seed validates it across seeds 43 and 44 (see Multi-seed validation below).

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

**Multi-seed validation (O009, seeds 43–44).** The test-slightly-above-val pattern is a seed-42 observation; O009 replicates it across all three Phase A seeds. Test − val overall mIoU = +0.011 / +0.015 / +0.010 (seeds 42 / 43 / 44); in every seed the pooled validation foreground IoU (0.699 / 0.701 / 0.702) falls inside the test per-frame 95% bootstrap CI ([0.657, 0.766] / [0.676, 0.775] / [0.657, 0.765]). The original conclusion — val and test are statistically indistinguishable and the small positive gap is sampling variance — holds across seeds 43 and 44.

**What this finding does NOT claim.**
- Does not claim the model generalises better to unseen data than to its training-adjacent validation data (this would be counterintuitive and requires stronger evidence).
- Does not indicate a leakage or contamination issue between val and test — the D028 scene-level split explicitly prevents this.
- Does not affect the comparison against Phase B and C, which will use the same test set.

---

### F003 — Foreground IoU 0.70-0.72 is the Phase A baseline anchor
**Date recorded:** 4 July 2026
**Phase:** A (U-Net binary)
**Status:** Locked. This is the number Phases B and C are calibrated against. Originally a seed-42 anchor; O009 multi-seed validates its stability across seeds 43 and 44 (see Multi-seed validation below).

**Observation.** Phase A U-Net binary achieved foreground IoU of 0.6991 on validation and 0.7195 on test (pooled). On the 23-scene test split the bootstrap point estimate is a per-frame mean of **0.7119 with a 95% CI of [0.6572, 0.7659]** (D020, 10,000 resamples, seed 42). These become the reference point for evaluating whether Phase B (YOLOv11-seg binary, modernised architecture) meaningfully changes binary-mask performance, and whether Phase C (YOLOv11-seg multiclass) provides additional gain via class-aware downstream logic.

**Statistical anchor, honestly bounded.** The "~0.72" anchor carries a wide interval — roughly [0.66, 0.77] at 95% — the direct consequence of the 23-scene test ceiling (O006). Phase B and C perception comparisons against this anchor should compute the bootstrap CI on the paired difference (the same construction as F001's gap CI), not compare individual arm CIs — the paired-difference CI is the correct inference for cross-arm significance. This is a further reason the headline cross-arm comparison lives at the geometric/command strands (D014), not here.

**Multi-seed validation (O009, seeds 43–44).** The 0.70–0.72 anchor is a seed-42 point estimate; O009 confirms its stability across seeds — test per-frame foreground IoU 0.712 / 0.725 / 0.712 (seeds 42 / 43 / 44), mean **0.716 ± 0.008** (training-run SD ≈ 1% of the metric). The anchor and the Phase B/C calibration reference stand unchanged.

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

- **Geometric strand:** RMS lateral error against the driven-path reference (the robot's driven trajectory; BLT is autonomous deployment, Polvara 2024 §3.3.3) — all three arms produce the same signal (an estimated centreline) after RANSAC line-fitting. This is the metric committed in the proposal and PHASE_C_SPEC §8.
- **Command-level strand:** steering-command difference against the driven commands (all three arms feed the same PID controller structure).

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
- Detections #1–#12 correctly identify individual trunks and poles with sensible masks (individual mask areas average ~406 px; their union-additional contribution beyond the blob is 4,528 px, i.e. 76,837 + 4,528 = 81,365).
- Raising conf to ≥ 0.41 removes detection #0 (the blob) but also removes detection #8 (conf 0.287); 11 detections remain and single-frame fg IoU improves from 0.04 to 0.598. Alternatively, a mask-area filter that removes only detection #0 while retaining all 12 correct detections yields the same single-frame fg IoU of 0.598. A residual gap versus U-Net's 0.697 on the same frame remains — YOLO's correct detections cover the ground truth well but do not fully match U-Net's per-pixel foreground coverage, even after blob removal.
- On the locked best.pt, reproduces deterministically and identically under FP16 and FP32. Does NOT reproduce on last.pt (final epoch of the same run): last.pt yields 12 detections, max mask 963 px, no blob, single-frame fg IoU 0.604. The failure is checkpoint-specific — not stable even across the last two saved checkpoints of the same training run. best.pt was selected by ultralytics' val fitness metric; last.pt is the final epoch's weights.

**Analysis.** The mask boundary follows canopy structure — this is a shaped prediction, not a numerical artifact. But the failure is not a stable learned property of the model: fourteen epochs later in the same training run (last.pt), the same architecture with slightly different weights does not produce the failure at all. This checkpoint-specificity substantially bounds the interpretation.

Possible mechanisms include: (a) the detector at best.pt's epoch firing on an occluded pole or trunk within the canopy region, with the mask over-drawing onto surrounding foliage — the same detection may not have fired at the last.pt epoch; (b) a transient training instability at the val-optimal epoch that produces this specific mask via prototype coefficient predictions that drift by last.pt; (c) an artefact of the mask head's coefficient predictions at a specific parameter configuration that fitness-based checkpoint selection happened to lock. Distinguishing these would require inspection of detection #0's bounding box against ground-truth annotations, prototype activation analysis, and comparison of mask head weights between best.pt and last.pt — none currently done.

Note on labelling choice: our binary collapse (trunk + pole → foreground) reflects three considerations. First, the downstream RANSAC line-fit for centreline detection requires geometrically aligned foreground pixels — canopy pixels vary in position across frames and would fit lines through canopy rather than crop rows. Second, this matches the labelling used by de Silva et al. 2024, our binary baseline reference. Third, SemanticBLT labels only structural elements (buildings, pipes, poles, robots, trunks, vehicles); canopy is not a labelled category. The failure is "wrong" specifically relative to this labelling scheme; under an alternative labelling that includes canopy in foreground, detection #0 would be classified differently. But the current labelling remains appropriate for centreline detection.

**Architectural asymmetry.** The failure mode is architecturally *possible* for YOLOv11-seg because instance masks are computed per-detection via prototype coefficients, so a single detection can produce a large mask regardless of what other detections are producing. The mode is architecturally *impossible* for U-Net because each pixel is classified independently — a coherent shaped mask spanning a canopy region cannot be produced by per-pixel classification. This asymmetry is a real feature of the two architectures.

This architectural asymmetry describes what is possible — U-Net cannot produce shaped canopy masks; YOLO can. It does not describe what is reliable: best.pt producing this failure while last.pt does not indicates that the mode's manifestation is checkpoint-specific rather than a stable output of this architecture on this data.

**Bounded claims.** One instance of this failure on 23 test frames does not establish a base rate. The observation may reflect training-data characteristics (F004: current test set is in-distribution to training set; the confusion may be an in-distribution training coverage artifact that resolves with OOD data). Class-aware multiclass supervision (Phase C) may or may not constrain coefficient predictions differently — verification pending Phase C evaluation. Multi-seed evaluation (O009) will establish whether the same failure recurs across different training seeds. *(This sentence is a frozen pre-Phase-C prediction, commit 2a69c95, 8 July 2026.)* **→ Resolved (O009 complete):** established — the failure recurs at 2/3 in each YOLO arm; see "Multi-seed evidence (6 seeds complete)" below.

**Why the two metrics diverge.** mAP@50 counts detection #0 as one wrong prediction: at 76,837 px versus 3,564 px ground truth, it will not match any ground-truth mask and is scored as a single false positive. Aggregate mAP@50 absorbs this as a small precision penalty across all 23 test frames. Rasterised fg IoU, computed as union-mask overlap with ground truth, is dominated by detection #0's mask area — on this frame, IoU collapses to ~0.04, and the mean across 23 frames cannot recover. F005 (metric parity via rasterised fg IoU) anticipated this divergence; F007 is the concrete illustration on a specific frame.

**Implications for the dissertation.**

*Methodology:* Report both metrics with clear labels. Explain the labelling-scheme framing when describing the metric divergence — 6799's failure is precisely defined against the binary trunk + pole labelling and would be classified differently under alternative labellings.

*Results:* Present aggregate metrics with CIs and per-frame variance. Include 6799 as a documented outlier with visualisation. The 0.04–0.69 fg IoU range for Phase B is a real characterisation, not an artifact.

*Discussion:* Frame the Phase A vs Phase B comparison as: "U-Net produces consistent per-pixel coverage across the test set with limited variance. YOLO produces precise instance masks on most scenes but exhibited one catastrophic false-positive mask on 6799, driving much of the aggregate rasterised fg IoU gap. The failure mode is architecturally available to YOLO and unavailable to U-Net; whether it occurs at 1-of-23 rate or would occur at meaningfully different rates on different training seeds or different data is not established from this single instance."

*Discussion (remediation options — none adopted):* Preliminary calculation: if a mask-area filter (e.g. max_area > 3,000 px on 640×640 images) removed detection #0's 76,837 px mask while retaining all 12 correct detections on 6799, the aggregate Phase B test rasterised fg IoU would rise from 0.556 [0.466, 0.633] to approximately 0.581, narrowing but not closing the gap against Phase A's 0.72 [0.66, 0.77]. The remaining gap reflects consistent under-coverage of thin structures across most test scenes — an architectural effect distinct from the catastrophic false-positive on 6799. Neither remediation is adopted here. A mask-area filter would be principled but hides the failure mode from Phase B's reported numbers; the downstream RANSAC line-fit stage is a more natural location for spurious-input handling if needed. Higher input resolution (e.g. 1280×1280) would reduce mask-head downsampling of thin structures and likely improve under-coverage, but was not explored — it would compromise the controlled A vs B architectural comparison at fixed 640×640 input. Both options are noted for possible future work.

*Discussion (mean-vs-median conf selection):* Supplementary median-based analysis on val (46 frames, 8-value sweep grid). Median-based conf* coincides with mean-based (both = 0.25); catastrophic frames (fg IoU < 0.1) occur on zero val frames at any threshold in the sweep range. The 6799-type failure did not appear on val at any conf, which is why neither mean- nor median-based selection could anticipate it. This has three implications: first, it suggests the 6799 failure is out-of-distribution relative to the val set — F004's in-distribution concern applies here directly. Second, it demonstrates a limit of val-based hyperparameter selection: robustness to failure modes that don't appear on val cannot be validated against val. Third, the failure being checkpoint-specific (present on best.pt, absent on last.pt) suggests that even with a broader val set, checkpoint selection by val fitness might select checkpoints that exhibit the failure while adjacent checkpoints do not. For this project, the primary conf* = 0.25 is documented as val-selected; F007 characterises the specific test failure that val-based selection did not anticipate.

*Note for Phase C — pre-registered before any Phase C run (commit 2a69c95, 8 July 2026; frozen prediction, kept verbatim, not edited post-result):* Verify whether Phase C's multiclass YOLO's val-selected checkpoint produces a similar or different result on 6799. Given the failure is checkpoint-specific in Phase B, Phase C's failure or absence-of-failure on 6799 tells us: if the failure recurs, it's a repeatable pattern across model families and checkpoint selection procedures; if it doesn't recur, it may reflect class-aware supervision constraining coefficient predictions, or simply the specific checkpoint Phase C selected. Multi-seed evaluation (O009) is the more decisive test — if seeds 43–46 all show 6799-type failures at their locked best.pt, the effect is a systematic property of val-fitness checkpoint selection for this architecture on this data. If none do, best.pt was an outlier.

**→ Resolved (O009 complete):** the failure recurs at 2/3 in *both* YOLO arms (binary and multiclass) with near-identical geometry — a systematic property of val-fitness checkpoint selection for this architecture on this data, not a Phase-B-only outlier, and not prevented by class-aware supervision. See "Multi-seed evidence (6 seeds complete)" below.

*Phase C result (10 July 2026):* **The 6799 blob does NOT recur in Phase C.** Phase C's val-selected best.pt at conf 0.25 produces 14 clean per-instance detections (10 trunk, 4 pole), largest mask **989 px** (no blob), rasterised fg IoU **0.627** — versus Phase B best.pt's 76,837 px blob and fg IoU 0.038. Per the pre-registered interpretation above, absence-of-recurrence is consistent with **either** class-aware supervision constraining the mask coefficients **or** Phase C simply selecting a "clean" checkpoint (the Phase B failure was itself checkpoint-specific — best.pt yes, last.pt no). **n=1 cannot distinguish these; this does NOT establish that multiclass supervision fixes the failure mode.** O009 multi-seed remains the decisive test. **→ Resolved (O009 complete):** across 3 Phase C seeds the blob recurs 2/3 — seed 42's clean 6799 was a checkpoint outcome (like Phase B seed 44), not a supervision effect; class-aware supervision does not prevent the failure. See "Multi-seed evidence (6 seeds complete)" below. Visualisation (same format as Phase B): `results/runs/phase_c_yolo_multiclass/diagnostic/6799_visualisation/`.

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

*Note for Phase C — pre-registered before Phase C multi-seed (commit 4044395, 10 July 2026; frozen prediction, kept verbatim, not edited post-result):* Phase C's multi-seed variance and blob rate will characterise whether class-aware supervision produces a more Phase-A-like (stable, continuous variance) or Phase-B-like (intermittent, discrete variance) training-run profile. Testable finding. **→ Resolved (O009 complete):** Phase C lands firmly Phase-B-like — intermittent, discrete, blob-driven variance (fg IoU SD 0.022; 6799 blob 2/3); class-aware supervision does not shift it toward Phase A's stability. See the variance table above.

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

---

## Geometric strand — in-row centreline evaluation (kg_march_23 bag, CP-5 val)

*F010–F015 report the geometric strand: image → base points → ground projection → line-fit centreline → GT-1 lateral offset (at 2 m look-ahead) / GT-2 heading (centreline slope), evaluated on the 4,708 val frames of kg_march_23 across all nine models (A/B/C × seeds 42/43/44) with the locked D036–D038 line-fit pipeline. These are **val** results; the discriminating cross-arm comparison (paired bootstrap) and the held-out test (CP-6) are pending. Each entry follows: Finding · Evidence · Implication · Cross-arm treatment.*

## Evaluation scope (Methodology framing)

This work evaluates the pipeline on **7,857 in-row frames** per model — **47%** of the 16,656 total March-bag frames (D040 whole-bag pooling; D041 frame accounting). Non-in-row segments (headland manoeuvres, corridor transitions, stationary intervals) — **5,841 frames (35%)** — are characterised **separately** as deployment-gap behaviour with explicit metric caveats (a *driven-path error*, not the in-row centreline RMS; see the non-in-row characterisation and **D041**). Contaminated frames — **2,958 (18%)**, those overlapping the segmentation training set within ±1.0 s of a CP-0 exclusion interval — are excluded to prevent perception leakage. Headland exclusion is a **methodological-validity constraint**, not merely scoping: the inverse projective mapping assumes flat ground, which does not hold on headland slopes (`GEOMETRY_PIPELINE_SPEC.md` §7). Frame accounting closes exactly: 7,857 + 5,841 + 2,958 = 16,656 (D041, mutually exclusive and exhaustive under contamination-first ordering).

---

### F010 — Systematic ~2.3° heading tilt is consistent across all nine models (projection, not perception)

**Finding.** Every model's GT-2 (line-fit centreline slope) carries a systematic positive tilt with cross-arm SD ≤ 0.07°, and the per-side slope structure (m_L, m_R both positive and near-equal) is identical across arms. The magnitude is scene-dependent (val ~2.28°, held-out test ~1.9°, pooled whole-bag ~2.10–2.18°) but its **arm-consistency is invariant** across all three — strong empirical support that the tilt is a projection effect, not a per-arm perception property.

**Evidence.** Per-arm tilt was measured on **val** (`superseded/march_val_test_split/val_evaluation/line_fit_val_report.json`): A 2.25 ± 0.03°, B 2.30 ± 0.07°, C 2.28 ± 0.07°; **confirmed on the held-out test** (A +1.86° / B +1.99° / C +1.94°, cross-arm SD 0.054°); and **re-confirmed on the pooled whole bag** (`final/march_evaluation/line_fit_report.json`): A 2.10° / B 2.18° / C 2.15° (cross-arm SD 0.04°). The per-side slopes were characterised on val only in the val/test-era artefacts (the test-side per-frame CSV stored no m_L/m_R); the **12-column per-frame schema introduced in Commit 2b (D040) records m_L/m_R for all frames**, enabling whole-bag per-side characterisation for the first time — pooled per-arm m_L/m_R (19,049–19,234 two-row frames per arm) A +0.036/+0.038, B +0.036/+0.040, C +0.033/+0.042, positive and near-equal on every model across the whole bag. Frame 3998 shows the slant directly (right row m_R = +0.10). Root cause: F017 (sensor-common; the F015 camera-yaw form refuted).

**Implication.** Absolute GT-1 at the 2 m look-ahead includes a residual ~8 cm offset contribution from this tilt (tan 2.3° × 2 m). It does not affect the *ranking* of arms.

**Cross-arm treatment.** Neutral — identical across arms, cancels in paired cross-arm differences. Absolute GT-1/GT-2 include it; stated wherever reported.

**Cross-references.** F017 (sensor-common root cause; supersedes the F015 camera-yaw form); F015 (historical rule-out investigation); D038 (line-fit centreline); D034 / §6 known limitation 2; D040/D041 (whole-bag pooling).

**Writeup wording (A2):**

**Fully defensible.** Across all nine models (three arms × three seeds), the line-fit centreline heading (GT-2) carries a systematic positive tilt with cross-arm standard deviation ≤ 0.07°, and the per-side slopes are positive and near-equal on every model (whole-bag pooled m_L ≈ +0.035, m_R ≈ +0.040). Because the tilt is identical across arms and seeds it is a property of the image-to-ground projection, not of any perception arm; at the 2 m look-ahead a ~2.1–2.3° tilt contributes a residual ~7–8 cm to the absolute lateral offset (tan θ × 2 m). Being common to all arms it cancels in paired cross-arm comparison, and is retained in absolute GT-1/GT-2 wherever those are reported. The magnitude is scene-dependent — validation ~2.28°, held-out test ~1.9°, pooled whole-bag ~2.10–2.18° — but the arm-consistency is invariant across all three (the whole-bag per-side m_L/m_R characterisation is enabled by the Commit 2b 12-column schema, D040).

**Candidate explanations.** Root-cause mechanism is treated in F017 (sensor-common base_link-to-row offset); F010 asserts arm-consistency only.

**NOT defensible.**
- ✗ attribute the tilt to any specific mechanism here (camera yaw / base_link error) — that is F017, and the camera-yaw form was refuted.
- ✗ claim the ~8 cm contribution affects arm *ranking* (it cancels).
- ✗ claim the magnitude is bag-invariant (val ~2.3°, test ~1.9°, pooled ~2.1° — scene-dependent; the *arm-consistency* is what is invariant).

**Citation map.** Ours: `final/march_evaluation/line_fit_report.json` (pooled per-arm tilt + whole-bag m_L/m_R, 12-col schema); historical val/test in `superseded/march_val_test_split/{val,test}_evaluation/line_fit_{val,test}_report.json`. No paper support needed (measured pipeline property). Mechanism → F017.

### F011 — Far-field extension (D037) rescues ~20 pp of two-row coverage with zero loss, arm-independently

**Finding.** Two-row coverage rises from ~64% (superseded near-5 m Y-constant, D035) to **83–84%** (line-fit + far-extension) across all nine models, rescuing **871–975 frames per model with zero frames lost**. The rescued-frame count is similar across arms → not arm-specific.

**Evidence.** Two-row coverage measured on **val**: A 83.8 ± 0.5%, B 84.1 ± 0.9%, C 83.2 ± 0.1% (rescued old≠two-row → new two-row 871–975/model, lost = 0); **confirmed on the held-out test** ~77% (scene-harder — the test-only corridor 4 is the sparsest, F019 — not a regression); **pooled whole-bag** (`final/march_evaluation/line_fit_report.json`): A 81.3 ± 0.5%, B 81.6 ± 0.9%, C 80.8 ± 0.1% — arm-independent (spread ≤ 0.8 pp) across all three. D037 admits same-row dots at 5–10 m within ±0.5 m of the near-field row Y.

**Implication.** The coverage gain adds *harder* (sparse-near) frames to the two-row set, so the aggregate GT-1 RMS does not fall — per-frame accuracy is characterised by the bias / residual-SD **decomposition in F012, not raw RMS**; the coverage-impact metric is frames-rescued, not an accuracy change. More frames now yield a two-row centreline for the sweep and CIs.

**Cross-arm treatment.** Neutral — same extension and gate for all arms; rescued frames are common across arms, so paired differences are unaffected.

**Cross-references.** D037 (far-field extension); F012 (per-frame accuracy decomposition, Analysis A); D-G (coverage reporting).

**Writeup wording (A2):**

**Fully defensible.** The far-field inlier extension (D037) raises two-row coverage from ~64 % (the superseded near-5 m Y-constant model, D035) to 83–84 % across all nine models (A 83.8 ± 0.5 %, B 84.1 ± 0.9 %, C 83.2 ± 0.1 %), rescuing 871–975 previously-unusable frames per model with zero frames lost; the rescued counts are similar across arms, so the gain is a pipeline property, not arm-specific. Because the rescued frames are the harder, sparse-near-field cases, aggregate GT-1 RMS does not fall — the coverage effect is reported as frames-rescued, and per-frame accuracy is characterised by the bias/residual-SD decomposition (F012) rather than aggregate RMS. Coverage is ~83–84 % on val, ~77 % on the held-out test (scene-explained — the test-only corridor 4 is the sparsest-detection corridor, F019 — not a pipeline regression), and **80.8–81.6 % pooled whole-bag** (arm-independent), between the two scene regimes.

**Candidate explanations.** None — the mechanism is the explicit ±0.5 m Y-consistency gate (D037), a design choice, not a hypothesis.

**NOT defensible.**
- ✗ claim the coverage gain improves per-frame accuracy (it adds harder frames; RMS does not fall).
- ✗ claim 83–84 % generalises to all bags (test ~77 %).
- ✗ claim 0-frames-lost is a general property (measured on this bag under the specific D037 gate; other bags/gates untested).

**Citation map.** Ours: `final/march_evaluation/line_fit_report.json` (pooled coverage per arm); historical val rescued/lost counts in `superseded/march_val_test_split/val_evaluation/line_fit_val_report.json`. D037 (design). No paper support.

### F014 — Adjacent-corridor detection is scene-geometry-driven, not arm-driven

**Finding.** Adjacent-corridor detections (secondary same-side clusters at higher |Y|, logged and rejected by D036) occur at a consistent rate across arms — 3,262–3,698 per model on the 4,708-frame val set, and 86.1–88.5 % of two-row frames on the pooled whole bag (spread 2.4 pp) — and concentrate by scene: corridor 3 (the dominant, 4×-traversed corridor) dominates, with a stable per-pass distribution.

**Evidence.** On **val** the adjacent-by-corridor structure (summed over 9 models) is corridor 3 (13.3 k) > 1 (10.5 k) > 0 (7.9 k); by pass 2 (7.5 k) > 10 (5.8 k) > 6 (4.8 k); per-model adjacent-frame counts 3,262–3,698 (spread < 15 %). Adjacent counts were **val-only in the val/test-era artefacts** (the test-side per-frame CSV stored no adj flag); the **12-column schema introduced in Commit 2b (D040) records adj for all frames**, enabling whole-bag characterisation for the first time — the fraction of two-row frames with an adjacent-corridor cluster logged is A 86.1 % / B 87.7 % / C 88.5 % (spread 2.4 pp across arms), a consistent, arm-independent rate on the whole bag.

**Implication.** Neighbouring rows becoming visible at the far-extension range (5–10 m) is a property of the vineyard geometry and pass trajectory, not of the perception arm. The row fit correctly rejects them on all arms.

**Cross-arm treatment.** Neutral — a data property; the rejection logic is arm-agnostic.

**Cross-references.** D036 (adjacent-corridor logging); D037 (far-field range); D033 (corridor-3 pass split).

**Writeup wording (A2):**

**Fully defensible.** Adjacent-corridor detections — secondary same-side clusters at higher |Y|, logged and rejected by the row fit (D036) — occur at a consistent per-model rate (3,262–3,698 per model on the 4,708-frame validation set, spread < 15 %; and 86.1–88.5 % of two-row frames on the pooled whole bag, spread 2.4 pp across arms — the whole-bag characterisation enabled by the Commit 2b 12-column schema, D040) and concentrate by scene rather than by arm: corridor 3 dominates (13.3 k summed over nine models, vs 10.5 k for corridor 1 and 7.9 k for corridor 0), with a stable per-pass distribution. Neighbouring rows becoming visible at the 5–10 m far-extension range is therefore a property of the vineyard geometry and pass trajectory, and the rejection logic operates identically across all arms.

**Candidate explanations.** None — a measured data property.

**NOT defensible.**
- ✗ claim the rejection is perfect / zero-error (the finding shows a consistent rate and correct rejection on sampled frames, not a proven error rate).
- ✗ attribute the corridor-3 concentration to anything beyond scene geometry / pass trajectory.

**Citation map.** Ours: `final/march_evaluation/line_fit_per_frame.csv` (pooled adj-flag rate per arm, 12-col schema); historical val adjacent-by-corridor/by-pass counts in `superseded/march_val_test_split/val_evaluation/line_fit_val_report.json`. D036 (rejection), D033 (corridor-3 split). No paper support.

### F015 — Front-camera yaw offset (~2.2–2.3°) not captured in the published extrinsics

> **STATUS: SUPERSEDED (kept, not deleted).** The camera-yaw attribution proposed in F015 was **refuted by independent LiDAR cross-check (F017)**. Both the Zed2 camera and the Ouster LiDAR (identity extrinsic, cannot share camera-mount yaw) measure the same ~2.3–3.8° tilt of the vine rows relative to base_link. This localises the tilt to a **sensor-common base_link-vs-row geometric offset**, not a camera-specific yaw.
> The investigation in F015 (four alternative rule-outs, positive regression test) remains valid as characterisation of the camera-only evidence; **the four rule-outs still exclude those causes** (distortion, projection convention, cropping, driving-direction) as contributors to the observed heading offset. What is refuted is the specific attribution to camera-mount yaw. Refer to **F017** for the current interpretation.

**Finding.** A systematic heading offset of ~+2.28° (cross-arm SD ≤ 0.07°) is present in all nine models. A four-way rule-out investigation plus a positive localisation test attribute it to an **unmodelled front-camera yaw** absent from Polvara et al. (2024) Table 3 (which specifies q_z = 0, zero yaw). Attribution is **strongly supported by camera-only evidence; independent LiDAR cross-check pending.**

**Evidence.**
- *Ruled out:* (1) radial distortion — camera_info `D = [0,0,0,0,0]`, `image_rect_color` stream rectified; (2) projection convention — principal ray projects to Y = +0.060 m constant at all ranges, symmetric points sum to 2× the lateral offset (pure translation, no rotational bias); (3) image cropping — full-frame `cv2.resize` stretch, principal point preserved; (4) robot driving angle — crab (yaw − path tangent) = −0.03° ± 1.64°, angle-to-corridor = −0.15° ± 0.99° over 4,708 frames (robot drives aligned with the rows).
- *Positive test:* regressing camera GT-2 against the robot's true angle-to-corridor gives slope −0.75, r −0.50, **intercept +2.20°** — with the robot perfectly aligned to the row the camera still reads +2.20°, localising the residual to the camera.

**Per-pass structure (Analysis C).** The mean heading offset of +2.28° is composed of at least two components: (1) a systematic contribution consistent with unmodelled camera-mount yaw, supported by the alternative rule-outs and positive regression test above; and (2) a per-pass component of **~0.9° half-range, up to ~1.1° in the most extreme pass (pass 8)** beyond the systematic contribution. The source of the per-pass component is not identified within this work; candidate causes include but are not limited to ground non-planarity coupling to apparent yaw through the ground-plane projection assumption, per-pass differences in detection distributions, and per-pass variation in adjacent-corridor rejection patterns. Disentangling these is out of scope; the 2.28° pooled mean is reported with the per-pass wander explicitly quantified rather than attributed.

**Implication.** The ~2.3° tilt (F010) and its ~8 cm GT-1 contribution are a sensor-extrinsic artefact, not a navigation/perception result. Documented as a Methodology limitation with a removal path (extrinsic re-calibration adding yaw, or the LiDAR GT-3).

**Cross-arm treatment.** Neutral — a shared projection effect; cancels in paired cross-arm differences.

**Cross-references.** F010 (the cross-arm tilt it explains); D038, D034 / §6 known limitation 2; D-B (Table 3 extrinsics); §12 references (Polvara et al. 2024).

**Writeup wording (A2):**

**Fully defensible.** An initial investigation of the systematic tilt (F010) considered an unmodelled front-camera yaw, since Polvara et al. (2024) Table 3 specifies zero yaw for the front camera (q_z = 0). Four alternative causes were excluded on camera-only evidence and remain excluded: radial distortion (`camera_info D = [0,0,0,0,0]` on the rectified `image_rect_color` stream), projection convention (principal ray projecting to a constant Y = +0.060 m at all ranges, no rotational bias), image cropping (full-frame resize preserving the principal point), and driving angle (measured crab −0.03° ± 1.64°, angle-to-corridor −0.15° ± 0.99° over 4,708 frames). Regressing GT-2 on angle-to-corridor leaves an intercept of +2.20° — a heading offset persisting when the robot is aligned to the row. The specific attribution of that residual to a *camera-mount* yaw was then tested against an independent sensor and not supported: the LiDAR, which cannot share a camera-mount yaw, measures the same tilt (F017). The current interpretation is a sensor-common base_link-to-row offset (F017); the camera-yaw form is not retained.

**Candidate explanations.** Closed at F015 — current candidates live in F017.

**NOT defensible.**
- ✗ state the tilt is caused by a camera-mount yaw (tested against LiDAR, not supported; F017).
- ✗ claim Polvara's Table 3 is erroneous or omits a real yaw (it specifies zero yaw; we have no evidence the physical camera has one — the tilt is sensor-common).
- ✗ present the four rule-outs or the +2.20° intercept as *establishing* camera yaw (they exclude alternatives; the residual is now understood as sensor-common).
- ✗ foreground the refutation as a headline (it is a documented course-correction; the operative finding is F017).

**Citation map.** Ours: `final/val_evaluation/line_fit_val_per_frame.csv` (intercept +2.20°, crab/angle-to-corridor) *(now under `superseded/march_val_test_split/val_evaluation/`)*; bag `camera_info` D. Paper: Polvara Table 3 (camera q_z = 0). Superseded-by: F017 (`final/val_evaluation/lidar_crosscheck_val.json` *(now under `superseded/march_val_test_split/val_evaluation/`)*).

### F012 — Geometric noise floor is characterised by decomposition (arm-invariant)

**Finding.** The geometric noise floor is reported by decomposition, not raw RMS. GT-2 = a systematic tilt (mean +2.28° val / ~2.1° whole-bag; F010/F017) plus a residual. Regressing GT-2 on the robot's true angle-to-corridor gives slope −0.74 to −0.79 (r ≈ −0.50) on val, −0.88 to −0.94 on the held-out test, and **−0.75 to −0.79 (r −0.45 to −0.47) on the pooled whole bag** — the value expected (−1, attenuated by measurement noise) if the perception tracks the true row direction — leaving a **regression-residual RMS of A 1.37° / B 1.33° / C 1.32° pooled** (val A 1.33 / B 1.36 / C 1.29; test A 1.15 / B 1.03 / C 1.05), arm-invariant across all three (pooled spread 0.05°).

**Evidence.** Analysis A (RMS decomposition, all 9 models sane, RMS² − mean² ≥ 0): GT-1 = 0.14 m bias ⊕ 0.16 m residual SD; GT-2 = 2.28° tilt ⊕ 1.5° residual SD. Analysis B (regression, GT-2 ~ robot angle-to-corridor), measured at three stages and consistent: **val** slope −0.74…−0.79, intercept 2.07–2.28°, r −0.45…−0.53, residual RMS A 1.33 / B 1.36 / C 1.29° (SD ≤ 0.08, overlapping); **confirmed on the held-out test** slope −0.88…−0.94 (r −0.50…−0.55), residual A 1.145 / B 1.028 / C 1.052° (the A/B/C ordering flips vs val → noise, not a real arm difference; test floor slightly tighter ~1.0–1.15°); **pooled whole-bag** (angle-to-corridor re-derived from `/robot_pose` yaw over all 7,857 eligible frames, atc mean 0.0° SD 0.94°) slope A −0.79 / B −0.75 / C −0.77, r −0.45…−0.47, regression-residual RMS A 1.37 / B 1.33 / C 1.32° (arm-invariant, spread 0.05°). The raw global residual SDs (0.16 m, 1.5°) are upper bounds — they absorb per-pass bias wander (Analysis C), whose GT-1 direction-structure component is characterised in **F016**.

**Implication.** The regression-residual (removes the tilt via the intercept and the robot's heading via the slope) is the cleanest available noise-floor estimate; it still bundles perception + row-fit + ground non-planarity, not disentangled within this work. No external benchmark exists (Analysis G): Polvara et al. (2024) report only the RTK-GNSS localization floor (2–3 cm, a sensor accuracy) and SLAM APE/RPE trajectory RMSE (a localization-algorithm metric), neither comparable to an image-derived perception lateral/heading error, and no heading-uncertainty bound. This characterisation is therefore a contribution of this work.

**Cross-arm treatment.** Arm-invariant (regression-residual 1.29–1.36°, overlapping SDs; Analyses B/F) — no hidden arm difference lurks under the raw RMS.

**Cross-references.** F010/F017 (tilt component); F016 (the GT-1 direction structure within the per-pass wander — not restated here); F013 (indistinguishability); Analyses A/B/C/G; D038 (line-fit metric); D040/D041 (whole-bag pooling).

**Writeup wording (A2):**

**Fully defensible.** The geometric noise floor is characterised by decomposition rather than raw RMS. GT-2 decomposes into a systematic tilt (F010/F017) and a residual; regressing GT-2 on the robot's measured angle-to-corridor gives slope −0.74 to −0.79 (r ≈ −0.50) on val and −0.75 to −0.79 (r −0.45 to −0.47) on the pooled whole bag — consistent with perception tracking the true row direction (expected slope −1, attenuated by noise) — leaving a regression-residual RMS of A 1.37 / B 1.33 / C 1.32° per arm pooled (val A 1.33 / B 1.36 / C 1.29°), arm-invariant. The GT-1 decomposition is a 0.14 m bias term with a 0.16 m residual SD. This regression-residual is the cleanest available noise-floor estimate; it still bundles perception, row-fit and ground non-planarity, which are not separated within this work. No comparable external benchmark exists: Polvara et al. (2024) report an RTK-GNSS localisation floor (2–3 cm; §5.3) and SLAM trajectory APE/RPE, neither an image-derived perception lateral/heading error, and no heading-uncertainty bound. Characterising a perception-level geometric noise floor for this platform is therefore a contribution of the present work. It is confirmed on the held-out test set (slope −0.88 to −0.94; residual RMS A 1.145 / B 1.028 / C 1.052°, arm-invariant) and on the pooled whole bag.

**Candidate explanations.** The 1.29–1.36° regression-residual bundles perception noise, row-fit variance, and non-planar ground effects; the relative contribution of each is untested and out of scope to disentangle.

**NOT defensible.**
- ✗ present the regression-residual as a pure *perception* error (it bundles row-fit + ground non-planarity).
- ✗ claim Polvara's RTK 2–3 cm or SLAM APE/RPE is a comparable perception benchmark, or imply the paper reports one.
- ✗ claim the val/test A/B/C residual ordering reflects a real arm difference (it flips val→test; 0.12° spread = noise).

**Citation map.** Ours: `final/march_evaluation/line_fit_per_frame.csv` (pooled Analyses A/B; angle-to-corridor re-derived from `/robot_pose` yaw); historical val/test in `superseded/march_val_test_split/{val,test}_evaluation/line_fit_{val,test}_per_frame.csv`. Paper: Polvara §5.3 (RTK floor, SLAM APE/RPE) — cited only to establish absence of a comparable perception benchmark.

### F013 — Cross-arm indistinguishability (GT-1) val→test→pooled; a sub-noise-floor GT-2 offset that surfaces on pooling

**Finding.** The three arms are **indistinguishable on the primary GT-1 lateral-offset metric**, and carry a **sub-noise-floor systematic GT-2 offset that becomes resolvable only on the pooled whole-bag data**. On paired **val** frames, no cross-arm pair (A-B, A-C, B-C) shows a GT-1 or GT-2 paired difference whose 95% bootstrap CI excludes zero (moving-block bootstrap, block ≈ 2× measured Analysis-H decorrelation, all ~3,600 both-two-row frames): GT-1 A-B [−1.9, +8.8], A-C [−1.9, +6.2], B-C [−3.7, +1.7] mm — all within roughly a fifth of the 3.8 cm RTK-GNSS floor (Polvara et al. 2024, §5.3); GT-2 all pairs within [−7%, +2%] of the F012 ~1.3° noise floor; all CIs include zero. This was **confirmed on the held-out test** for GT-1 (all paired CIs include zero, ≤ 2.8 mm) with a bounded partial GT-2 divergence at test n (F019). On the **pooled whole-bag evaluation (7,857 frames, ~5,800 both-two-row per pair)**:
- **GT-1 stays indistinguishable** — A-B +1.0 [−3.2, +5.0], A-C +0.6 [−2.4, +3.6], B-C −0.9 [−3.1, +1.6] mm; all ≤ 1 mm and within ±2.7% of the RTK floor, CIs include zero.
- **GT-2 makes explicit what val bounded and test hinted:** a sub-noise-floor systematic offset — A-B −0.075° [−0.114, −0.027], A-C −0.057° [−0.093, −0.009], B-C +0.015° [+0.001, +0.033] (CIs exclude zero at the primary block), ordering A < C < B, largest gap 0.075° = **5.6% of the ~1.3° regression-residual noise floor (F012)**. It is characterised honestly, not dismissed: (i) sub-noise-floor; (ii) **sign-inconsistent across seeds for A-B and B-C — only A-C is sign-consistent** (all three seeds negative); (iii) **not robust to a conservative block** — at the stricter 0.05-threshold block (L=38, ~3.2 m) A-B and A-C still exclude zero but **B-C dissolves** ([−0.001, +0.033]). The navigation-relevant conclusion — equivalent centreline geometry at the primary GT-1 metric — holds.

Cross-checked with the **Δs = decorrelation-distance subsample bootstrap** (n = 329–751 per pair) — same conclusion, CIs 20–40% wider but overlapping with the block-bootstrap CIs. For GT-2, a **stricter block length** (2× the 0.05-threshold decorrelation, ~2.2–2.9 m) was also computed: A-B [−0.090, +0.015]°, A-C [−0.081, +0.027]°, B-C [−0.014, +0.023]° — **even at the strictest autocorrelation threshold, all CIs still include zero.**

The three arms are therefore indistinguishable on GT-1 at the tight bounds allowed by measured autocorrelation (on val, on held-out test, and on the pooled whole bag), not merely at the conservative bounds implied by pre-specified spatial-independence thresholds; the sub-noise-floor GT-2 offset above is the single qualification and does not affect the GT-1 conclusion.

**Evidence.** Val: `superseded/march_val_test_split/val_evaluation/paired_crossarm_val.json` + Analyses H/I; decorrelation distances per-pair per-metric (0.22–0.67 m at the 0.1 threshold; 0.22–1.43 m at the 0.05 threshold); on val, sign inconsistency preserved (5 of 6 pairs flip), and the one sign-consistent case (A-C GT-2, all negative) was negligible (0.02–0.06°) with an aggregate CI including zero. **Pooled:** `final/march_evaluation/paired_crossarm.json` (block lengths re-derived on the pooled data, Analysis H: L_GT1=9, L_GT2=20 primary / 38 conservative). On the pooled data the A-C GT-2 sign-consistency persists (all three seeds negative) and its CI now **excludes** zero — the same sub-noise-floor offset that val's wider CIs could not resolve; A-B and B-C remain sign-inconsistent across seeds. Per-arm block-bootstrap CIs all overlap (A GT-1 0.197 [0.188, 0.204] / GT-2 2.50 [2.37, 2.60]; B 0.193 [0.183, 0.201] / 2.53 [2.40, 2.64]; C 0.194 [0.185, 0.201] / 2.52 [2.39, 2.64]). (The earlier preliminary framing — analytical MDD ≈ 3.5 mm at n ≈ 11k — is subsumed by the paired-bootstrap analysis.)

**Methodology note (subsample bootstrap).** The per-model Δs = 1.5 m subsample bootstrap CI is now computed on a **frame-index-sorted** subsample, making the fixed-seed estimator order-invariant (the val/test-era ordering depended on CSV row order). This shifted the granular per-model CI bounds by ≤ 0.027° versus the val/test-era artefacts — below the reported precision, and not a headline claim (the headline paired and per-arm CIs use the moving-block bootstrap). A reproducibility improvement, not a numerical change.

**Implication / status.** **Confirmed on the pooled whole-bag evaluation.** GT-1 indistinguishability holds at both statistical (CIs include zero) and practical (≤ 1 mm, ≪ RTK floor) levels across val, held-out test, and pooled. On GT-2, pooling resolves a sub-noise-floor systematic offset (A < C < B, ≤ 5.6% of the noise floor, sign-unstable for two of three pairs, B-C dissolving under the conservative block) — reported honestly and below navigation relevance. The held-out-test confirmation (F019) is absorbed into this pooled finding under D040.

**Cross-arm treatment.** This finding *is* the cross-arm treatment: on val, the arms cannot be distinguished by centreline-estimation quality at tight autocorrelation-corrected bounds. **F013's cross-arm claim is unaffected by the F015→F017 revision:** a base_link-common tilt cancels in paired differences equally with a camera-specific tilt.

**Cross-references.** F012 (noise floor); F016, Analysis C (common per-pass and direction structure that cancels in paired differences); F017 (sensor-common tilt, cancels equally); Analyses D/E/H/I; §5.3 RTK floor; D-D spatial-independence subsampling; CP-6 test-set final confirmation.

> **Note (whole-bag).** The GT-2 sub-noise-floor offset that F019 first detected at test n is now resolved directly on the pooled data (above): A-B and A-C exclude zero robustly (including under the conservative 0.05-threshold block), B-C only at the primary block. It sits far below the noise floor and is sign-unstable across seeds for two of the three pairs; the operative navigation-relevant conclusion (arms deliver equivalent geometry at GT-1) holds. F019 is retained as the historical CP-6 trail (SUPERSEDED, D040).

**Writeup wording (A2):**

**Fully defensible.** On the validation set, no cross-arm pair (A–B, A–C, B–C) shows a mean paired GT-1 or GT-2 difference whose 95 % CI excludes zero. Using a moving-block bootstrap with block length twice the per-pair measured decorrelation distance (Analysis H) over all ~3,600 both-two-row frames, the GT-1 paired differences are A–B [−1.9, +8.8] mm, A–C [−1.9, +6.2] mm, B–C [−3.7, +1.7] mm — each within roughly a fifth of the RTK-GNSS floor (3.8 cm; Polvara et al. 2024, §5.3) — and the GT-2 differences fall within [−7 %, +2 %] of the ~1.3° regression-residual noise floor (F012); all intervals include zero. This conclusion is stable across three bootstrap estimators (moving block, decorrelation-distance subsample, and stricter 0.05-threshold block). The arms are therefore statistically and practically indistinguishable on validation at the tight bounds permitted by the measured autocorrelation. On the held-out test set the primary GT-1 metric confirms this (all paired CIs include zero, ≤ 2.8 mm; F019); the secondary GT-2 metric shows a bounded partial divergence (two of three pairs exclude zero by 0.035–0.122°), sub-noise-floor and reported in F019. On the **pooled whole-bag evaluation (7,857 frames)**, GT-1 remains indistinguishable (paired differences ≤ 1 mm, all CIs include zero) and the secondary GT-2 offset is resolved directly: a sub-noise-floor systematic ordering A < C < B (largest 0.075°, 5.6 % of the ~1.3° noise floor) whose CIs exclude zero at the primary block but which is **sign-inconsistent across seeds for two of the three pairs** (only A–C consistent) and **dissolves for B–C under a stricter 0.05-threshold block**. This is reported as a bounded characterisation — below navigation relevance and not attributable to an arm-level mechanism at this magnitude (comparable to the scene-dependent tilt component, F017) — not as an arm ranking. The navigation-relevant conclusion — equivalent centreline geometry at the primary GT-1 metric — holds on val, held-out test, and the pooled whole bag.

**Candidate explanations.** None — a statistical result, not a mechanism.

**NOT defensible.**
- ✗ state the arms are "identical" or "proven equal" (indistinguishable at measured bounds = failure to reject, not proof of equality).
- ✗ claim GT-2 indistinguishability unconditionally (pooled: a sub-noise-floor offset persists — A–B and A–C exclude zero robustly).
- ✗ call B "worse"/"better" on GT-2 (sub-noise-floor, sign-inconsistent for two of three pairs, B–C dissolves under the conservative block).
- ✗ **hide or omit the GT-2 offset** (it surfaces on pooling and is reported honestly).
- ✗ call the RTK floor a perception benchmark (it is a localisation-sensor accuracy, used as a yardstick).
- ✗ report p-values (CIs + effect sizes only).

**Citation map.** Ours: `final/march_evaluation/paired_crossarm.json` (pooled paired CIs, primary + strict 0.05-threshold GT-2 blocks) + `final/march_evaluation/line_fit_report.json` (per-arm CIs); Analyses H/I (decorrelation/block lengths, reproducible via `scripts/geometric/diagnostics/autocorrelation_block_analysis.py`); historical val/test in `superseded/march_val_test_split/{val,test}_evaluation/paired_crossarm_{val,test}.json`. Paper: Polvara §5.3 (3.8 cm yardstick).

### F016 — GT-1 bias is direction-dependent in the driven-path reference

**Finding.** The per-pass GT-1 bias (pooled across all nine models) depends on driving direction. Across all **11 passes** of the whole bag, the **six negative-heading passes {0,2,4,6,8,9}** have GT-1 bias 0.195–0.224 m (mean ~0.21 m) and the **five positive-heading passes {1,3,5,7,10}** have 0.021–0.076 m (mean ~0.05 m) — **a ~0.15 m difference associated with driving direction** (the biases are both positive; see candidate explanations for the underlying structure). All nine models exhibit the identical per-pass structure, so the effect is a property of the ground-truth reference (the robot's **driven `/robot_pose` trajectory** relative to corridor geometry — BLT is autonomous deployment, Polvara 2024 §3.3.3), not of any perception arm.

**Evidence.** Per-pass pooled GT-1 bias (all 9 models): neg-heading pass 0/2/4/6/8/9 = 0.198/0.203/0.195/0.206/0.224/0.214 m; pos-heading pass 1/3/5/7/10 = 0.059/0.021/0.038/0.076/0.055 m — overall pooled bias ~0.135 m. This was first measured on the 7 val passes (2,4,6,8 / 5,7,10) and confirmed on the 4 test passes (0,9 / 1,3), consistent val→test→pooled. At pass level (n = 11) the bias has no significant correlation with corridor (r = −0.02), recording time (r = −0.37), or speed (r = −0.20); canopy state is constant (single bare-vine March session, no variation).

**Implication.** The pooled GT-1 bias (~0.14 m) averages two direction-dependent regimes and is a direction-structured quantity, not a single constant — a limitation of the driven-path trajectory as an *absolute* lateral reference, not a pipeline error.

**Cross-arm treatment.** Neutral — all nine models share the identical direction structure (same frames), so it cancels exactly in paired cross-arm differences; F013's claim is unaffected.

**Candidate explanations (not asserted; out of scope to disentangle).** The measured biases are both positive (0.21 vs 0.05 m) — a magnitude difference, not a sign flip. Named candidates: (i) **driven-path systematic offset** — the robot's autonomous centring may have consistently favoured one physical row of the corridor, producing a driving-direction-linked bias structure (the most operationally likely mechanism, and consistent with the both-positive, different-magnitude pattern); (ii) a direction-independent constant (e.g. a camera lateral/yaw offset contribution) superimposed on a sign-flipping driven-path physical offset, which would also reproduce a both-positive pattern; (iii) world-frame corridor asymmetry (e.g. non-parallel row geometry).

**Cross-references.** F012 (per-pass wander at the pipeline level; F016 is its ground-truth-reference component); F013 (cancels in paired differences); D-F (driven-path-centred GT-1 assumption + limitation); F010/F017 (the GT-2 tilt, a separate effect); D040/D041 (whole-bag pooling).

**Writeup wording (A2):**

**Fully defensible.** The per-pass GT-1 bias depends on the driving direction. Pooled across all nine models and all 11 passes, the six negative-heading passes {0,2,4,6,8,9} have bias 0.195–0.224 m (mean ~0.21 m) and the five positive-heading passes {1,3,5,7,10} have 0.021–0.076 m (mean ~0.05 m) — a ~0.15 m magnitude difference (both regimes positive). All nine models show the identical per-pass structure on the same frames, so the effect is a property of the ground-truth reference (the robot's driven `/robot_pose` trajectory relative to corridor geometry — BLT is autonomous deployment, Polvara 2024 §3.3.3), not of any perception arm; it cancels exactly in paired comparison. At the pass level (n = 11) the bias has no significant correlation with corridor (r = −0.02), recording time (r = −0.37) or speed (r = −0.20), and canopy state is constant across the single March session. The pooled ~0.14 m GT-1 bias is thus a direction-structured quantity, not a single constant — a characterised limitation of the driven-path trajectory as an *absolute* lateral reference. Prior work on this platform does not report a direction-dependence of the reference; this characterisation is a contribution of the present work. It was first measured on the 7 val passes and confirmed on the 4 held-out test passes (negative-heading 0.206 m vs positive-heading 0.040 m), consistent val→test→pooled.

**Candidate explanations.** (i) Driven-path systematic offset — the robot's autonomous centring consistently favouring one physical side (most operationally likely; consistent with the both-positive, different-magnitude pattern). (ii) A direction-independent constant (e.g. a camera lateral/yaw contribution) superimposed on a sign-flipping driven-path offset. (iii) World-frame corridor asymmetry (non-parallel rows). None tested against the others.

**NOT defensible.**
- ✗ assert the driven-path offset as *the* cause (three candidates fit).
- ✗ call the ~0.15 m difference a pipeline/perception error (property of the reference; cancels in paired).
- ✗ describe it as a sign flip (both regimes positive — magnitude difference).
- ✗ claim the specific magnitude generalises beyond this platform's driven trajectories.

**Citation map.** Ours: `final/march_evaluation/line_fit_per_frame.csv` (pooled per-pass bias, pass-level correlations); historical val/test in `superseded/march_val_test_split/{val,test}_evaluation/line_fit_{val,test}_per_frame.csv`. D-F (driven-path-centred GT-1). No paper support (contribution).

### F017 — Sensor-common ~2.3–3.8° base_link-to-row tilt (mechanism open)

**Finding.** The ~2.28° mean heading offset observed in the camera pipeline (F010) is **also observed by the Ouster OS1-16 LiDAR at ~+3.84° (SD 0.17°)** on the six anchor frames tested. The LiDAR has an **identity extrinsic** in Table 3 (base_link → LiDAR quaternion = (0,0,0,1)), so it cannot exhibit the camera-mount yaw hypothesised in F015. That both sensors independently measure a nonzero tilt of the same sign **refutes the camera-yaw attribution** and localises the tilt to a geometric relationship between the robot's body frame (base_link) and the vine rows.

**Evidence.** LiDAR row heading vs camera row heading on the 6 val anchor frames (`superseded/march_val_test_split/val_evaluation/lidar_crosscheck_val.json`; the whole-bag cross-check is below):

| frame | LiDAR | camera | diff |
|---|---|---|---|
| 3998 | +3.84° | +3.63° | +0.21° |
| 4223 | +3.76° | +2.67° | +1.10° |
| 4107 | +4.20° | +2.61° | +1.59° |
| 3991 | +3.72° | +3.21° | +0.51° |
| 3994 | +3.79° | +3.46° | +0.33° |
| 3996 | +3.70° | +3.94° | −0.24° |
| **mean** | **+3.84° (SD 0.17)** | **+3.25°** | |

Anchor frames were selected toward high-tilt scenes for visualisation clarity, giving values above the ~2.28° val-mean; the core conclusion (LiDAR ≠ 0° and agrees with the camera in direction) is robust to this selection. LiDAR row heading = centreline slope from a robust per-side line fit on the trunk-height band (0.2 < Z < 1.2 m) transformed to base_link.

**Candidate explanations (not asserted; out of scope to disentangle).** A **body-frame (base_link) vs odometry-driving-direction offset** would produce this pattern: `/robot_pose` yaw tracks odometry (measured crab angle ≈ 0°; F015 evidence), while sensors mounted to the physical body observe the world tilted relative to odometry. **World-frame corridor asymmetry** (non-parallel row geometry) is an alternative. Resolving base_link-physical vs odometry-frame definitively requires TF-tree analysis beyond the current time-box.

**Caveats.** (1) The LiDAR extrinsic is nominal from Table 3; a small LiDAR-frame yaw cannot be fully excluded but cannot plausibly produce ~+3.8° from an expected ~0°. (2) Six-frame sample; extension to full val or per-pass structure requires further work.

**Cross-arm impact.** Unchanged and reinforced. A base_link-common tilt is shared by all three camera arms and cancels exactly in paired cross-arm differences; F013's practical-indistinguishability claim on val is unaffected.

**Cross-references.** F010 (cross-arm tilt consistency observation, unchanged); F015 (superseded, kept as the camera-only investigation trail); F013 (cross-arm claim unaffected); D-B (Table 3 extrinsics); Polvara et al. 2024 §5.3; D040/D041 (whole-bag pooling).

**Whole-bag cross-check (val→test→pooled).** The sensor-common tilt was measured across three anchor sets, all consistent in the operative claim (LiDAR nonzero, same sign as the camera): **val** (6 anchors) LiDAR +3.84° (SD 0.17°) / camera +3.25°; **confirmed on the held-out test** (6 anchors, 2 per test corridor) LiDAR +3.04° (SD 0.36) / camera +2.74° (Δ −0.31°); and the **pooled whole-bag cross-check** (10 anchors, 2 per corridor across all five corridors; `final/march_evaluation/lidar_crosscheck.json`) LiDAR **+2.57° (SD 0.96)** / camera **+1.86°**, camera−LiDAR **−0.71°**. The per-anchor magnitude is scene-dependent (the val anchors were high-tilt-selected; the pooled set spans all corridors including the sparse corridor 4, lowering the mean), but LiDAR and camera agree in sign and nonzero-ness on every set — the sensor-commonality that refutes the camera-yaw attribution is invariant val→test→pooled.

**Writeup wording (A2):**

**Fully defensible.** The systematic heading tilt observed in the camera pipeline (mean +2.28°, F010) is also observed by the vehicle's Ouster OS1-16 LiDAR. On six anchor frames, a robust per-side line fit to the trunk-height LiDAR returns (0.2 < Z < 1.2 m), transformed into base_link, yields a mean row heading of +3.84° (SD 0.17°), agreeing with the camera in sign on every frame (camera mean +3.25° on the same frames). Because the LiDAR has an identity base_link→sensor extrinsic in Polvara et al. (2024) Table 3 (quaternion (0,0,0,1)), it cannot carry the camera-mount yaw an earlier hypothesis proposed (F015); that two independent sensors measure a nonzero tilt of the same sign localises the tilt to the geometric relationship between the robot's body frame and the vine rows, rather than to a camera-specific artefact. It is confirmed on the held-out test set (LiDAR +3.04° / camera +2.74°, within 0.31°) and on the pooled whole-bag cross-check across all five corridors (LiDAR +2.57° / camera +1.86°, camera−LiDAR −0.71°). Characterising the tilt as sensor-common — thereby excluding a camera-specific cause — is a contribution of the present work; the paper provides the extrinsics but does not investigate the tilt. (Anchor frames were selected toward high-tilt scenes for visual clarity, giving magnitudes above the ~2.28° pooled mean; the same-sign, sensor-common conclusion is robust to that selection.)

**Candidate explanations.** (i) A base_link (physical body) vs odometry-driving-direction offset. This hypothesis is consistent with the observed pattern (`/robot_pose` yaw at ~0° while sensors observe nonzero tilt); it has not been tested against alternatives. (ii) World-frame corridor asymmetry (non-parallel rows). Resolving base_link-physical vs odometry-frame definitively needs TF-tree analysis beyond the current time-box; not identified within this work.

**NOT defensible.**
- ✗ claim the base_link extrinsic or the odometry frame is "wrong" (mechanism unresolved; we show commonality, not a frame error).
- ✗ claim the camera has a yaw (refuted; sensor-common).
- ✗ claim the paper investigates/reports the tilt (Table 3 = extrinsics only).
- ✗ present +3.84° as the tilt magnitude (anchors are high-tilt-selected; pooled camera mean ~2.28° val / ~1.9° test — the robust claim is nonzero, same-sign, sensor-common).
- ✗ fully exclude a small LiDAR-frame yaw (nominal extrinsic; cannot plausibly give ~+3.8° from ~0°, but the six-frame sample is a stated limit).

**Citation map.** Ours: `final/march_evaluation/lidar_crosscheck.json` (pooled 10-anchor cross-check, all five corridors); historical val/test 6-anchor tables in `superseded/march_val_test_split/{val,test}_evaluation/lidar_crosscheck_{val,test}.json`. Paper: Polvara Table 3 (LiDAR identity extrinsic + camera extrinsics), §5.3. Supersedes F015.

### F018 — Phase C downstream config sweep + single-class ablations: trunks load-bearing; poles supplement coverage, not quality

**Finding.** Across the Phase C downstream config sweep (trunk-primary / pole-primary × T∈{1,2,3,5,8,12}, class-agnostic) plus single-class ablations (trunk-only, pole-only): (1) in the **viable regime (coverage ≥70%)**, class structure does **not** distinguish centreline quality — GT-1 RMS 0.193–0.204 m and GT-2 RMS 2.63–2.72° are mutually CI-overlapping (block bootstrap, Analysis-H block lengths); (2) **trunk-only ≈ class-agnostic on quality** (GT-1/GT-2 CIs overlap) at 71.0% vs 82.9% coverage; (3) **pole-only degenerates** — 1.1% two-row, 85.5% no-estimate; (4) adding poles to trunks (agnostic) raises coverage +12 pp (71.0→82.9%) at zero quality change.

The sweep is **Phase-C-multiclass-specific by design (D026)**: only the multiclass arm carries the trunk/pole class structure the sweep varies. Phase A (U-Net binary) and Phase B (YOLO binary) collapse foreground to a single class and have no class-priority / T parameter to sweep, so the class-config analysis is meaningful only on arm C — this is a deliberate design asymmetry, not a coverage gap in the comparison.

**Evidence.** Cross-config table, measured on **val** (`superseded/march_val_test_split/val_evaluation/config_sweep_val.json`):

| config | coverage | base pts | GT-1 RMS · CI | GT-2 RMS · CI |
|---|---|---|---|---|
| class-agnostic | 82.9% | 31.6 | 0.200 · [0.189, 0.210] | 2.65 · [2.42, 2.83] |
| trunk-only | 71.0% | 18.3 | 0.204 · [0.192, 0.216] | 2.72 · [2.48, 2.95] |
| pole-only | 1.1% (85.5% none) | 13.3 | (degenerate) | (degenerate) |
| trunk-primary (T grid) | 71–82% | 18–27 | 0.198–0.204 (all overlap) | 2.65–2.72 (all overlap) |
| pole-primary (T grid) | 1→80% | — | degenerate ≤T5; T12 ≈ agnostic | — |

trunk-only GT-1 0.204 [0.192, 0.216] overlaps agnostic 0.200 [0.189, 0.210] (GT-2 likewise). **Pole-only produced n = 1 two-row frame at 1.1% coverage; the reported 0.118 m / 1.26° values are single-frame artefacts, not RMS estimates, and are reported for transparency rather than as a comparable measurement.** Pole-primary collapses at T≤5 (poles too sparse); viable only at T12 (trunk-fallback-dominated). Argmin among viable cells = pole_T12, whose CIs overlap agnostic → pre-stated tie-break locks class-agnostic.

**Pooled whole-bag re-report (`final/march_evaluation/config_analysis.json`; the locked class-agnostic design is re-reported, NOT re-selected — D040).** The mechanism reproduces exactly: class-agnostic 80.6% coverage, GT-1 RMS 0.194 [0.185, 0.201], GT-2 RMS 2.52 [2.39, 2.64]; in the viable regime (coverage ≥ 70%: agnostic, trunk_T3/T5/T8/T12, pole_T12) all cells' CIs mutually overlap on both metrics; the viable argmin is pole_T12 (GT-1 0.188 / GT-2 2.50), whose CIs overlap agnostic → the pre-stated tie-break locks **class-agnostic** (`agnostic_locked=True, pause_flag=False`). Sub-viable pole cells reproduce their degeneracy (pole_T1/T2/T3 ~1% two-row, single-frame artefacts). The mechanism is confirmed val→held-out-test→pooled: on the 3,149 held-out test frames trunk-only GT-1/GT-2 CI-overlap agnostic (0.187 vs 0.183 m; 2.38 vs 2.30°), poles supplement coverage +13.3 pp (agnostic 77.3% vs trunk-only 64.0%), and pole-only degenerates (0.9% two-row).

**Interpretation.** In the viable regime (coverage ≥70%), class structure affects base-point availability but does not distinguish centreline quality; low-coverage pole-only cells degenerate before this can be tested at matched base-point counts. The pole collapse is not merely a coverage limitation — it is the **mechanistic explanation** for the flat quality result: if poles carried independent, complementary structural information, pole-primary at high T (or pole-only) should have produced distinguishable centreline quality; instead, pole-only base points are too sparse to fit a stable row (1–5% coverage at T1–T5), and the only viable pole-primary cell (T12) is one where fallback to trunks dominates. Multiclass information is therefore **not silently helpful** — it exposes that **trunks are the load-bearing feature and poles cannot substitute at this bag's density**. **Class-aware information has two operational contributions to the pipeline: a *quality* contribution (none within measurement bounds — trunk-only and agnostic centreline RMS are CI-indistinguishable) and a *coverage* contribution (+12 pp when poles supplement trunks — 71.0% trunk-only vs 82.9% agnostic).** The ablations resolve the hypothesis: trunk-only matches agnostic on quality (Hypothesis A on quality); poles contribute measurably to coverage but not quality, and cannot substitute alone.

**Design decision.** Class-agnostic locked for CP-6: **it captures the coverage benefit at zero measurable quality cost — the positive reason for locking it beyond the argmin tie-break** — with the highest coverage (82.9%), centreline RMS indistinguishable from every viable config, and the simplest logic (no class-priority or T threshold to justify). The pole-primary/pole-only collapse confirms poles cannot substitute for trunks as the primary row cue.

**Caveats.** The config-invariance of centreline quality is measured at F013's bounds (val, held-out test, and pooled whole bag). A denser-canopy bag (different trunk/pole visibility) is a multi-bag question — poles may carry more structural information at higher visibility.

**Cross-references.** F013 (arm-level indistinguishability; F018 is the config-level analogue); D026 (sweep design, Phase-C-specific); D030/D032 (Phase C conf/config); Analysis H (block lengths); D040/D041 (whole-bag pooling; config re-reported not re-selected).

**Writeup wording (A2):**

**Fully defensible.** A downstream configuration sweep of the Phase C multiclass pipeline (trunk-primary and pole-primary priority × T ∈ {1,2,3,5,8,12}, plus class-agnostic) with single-class ablations (trunk-only, pole-only) shows that within the viable regime (coverage ≥ 70 %) class structure does not distinguish centreline quality: GT-1 RMS 0.193–0.204 m and GT-2 RMS 2.63–2.72° are mutually CI-overlapping (block bootstrap, Analysis-H block lengths). Trunk-only matches class-agnostic on quality (GT-1 0.204 [0.192, 0.216] m vs 0.200 [0.189, 0.210] m; GT-2 likewise) at 71.0 % vs 82.9 % coverage, while pole-only degenerates (1.1 % two-row, 85.5 % no estimate). Adding poles to trunks therefore contributes +12 pp coverage (71.0 → 82.9 %) at zero measurable quality change. The pole degeneration is the mechanistic explanation for the flat quality result: had poles carried independent complementary structure, pole-primary at high T or pole-only would have produced distinguishable quality; instead pole base points are too sparse to fit a stable row (1–5 % coverage at T1–T5) and the only viable pole-primary cell (T12) is trunk-fallback-dominated. Trunks are the load-bearing feature and poles cannot substitute at this bag's density; class-aware information contributes to coverage but not to centreline quality within measurement bounds. Class-agnostic was locked on this basis. The mechanism replicates on the held-out test set (trunk-only vs agnostic GT-1 0.187 vs 0.183 m, GT-2 2.38 vs 2.30° CI-overlapping; poles +13.3 pp; pole-only degenerate) and re-reports on the pooled whole bag (class-agnostic 80.6 % coverage; in the viable regime all cells' CIs mutually overlap, the viable argmin pole_T12 overlaps agnostic → the pre-stated tie-break locks class-agnostic — re-reported, not re-selected, D040). The sweep is Phase-C-multiclass-specific by design (D026): the binary arms A and B collapse foreground to one class and have no class-priority or threshold to sweep, so this analysis applies to arm C only. A downstream analysis of which class carries the geometric signal is not present in prior work on this platform; it is a contribution of the present work.

**Candidate explanations.** Poles may carry more structural information on a denser-canopy bag (higher pole visibility) — a multi-bag question, untested; the pole conclusion is stated at this bag's density. Alternatively, the sparse-pole density at this bag may itself be scene-specific — a bag with denser pole/trellis structure (e.g. a different vine training system) might reveal pole-borne signal not present here. Multi-bag evaluation is required to characterise this.

**NOT defensible.**
- ✗ claim poles carry no information in general (bounded to this bag's density; higher-visibility bags untested).
- ✗ report the pole-only 0.118 m / 1.26° as an RMS/quality figure (single-frame artefact, n = 1).
- ✗ call multiclass "useless" or "harmful" (quality-neutral within bounds, coverage-positive +12 pp — "not silently helpful," not detrimental).
- ✗ claim a quality difference below CI resolution (trunk-only and agnostic CI-indistinguishable).

**Citation map.** Ours: `final/march_evaluation/config_analysis.json` (pooled sweep + ablations + viable-regime argmin/tie-break); historical val/test in `superseded/march_val_test_split/{val,test}_evaluation/config_sweep_val.json` and `config_ablation_test.json`. D026 (sweep design, Phase-C-specific), D030/D032 (Phase C config), Analysis H (block lengths). No paper support (contribution).

### F019 — CP-6 held-out test: GT-1 indistinguishable (confirms F013); GT-2 a negligible-but-detectable B-vs-others micro-difference

> **STATUS: SUPERSEDED (16 July 2026, D040) — kept as historical trail.** The March val/test split was pooled into a single whole-bag evaluation (D040); F019's role — test-side confirmation of F013 — is **absorbed into the pooled F013**, which reports the GT-1 indistinguishability and the persisting sub-noise-floor GT-2 offset on all 7,857 frames directly. The CP-6 held-out result below is retained **unchanged** as the historical derivation (like F015). What is superseded is the *interpretation* that a separate within-bag held-out test was load-bearing — seasonal generalisation is now claimed at the multi-bag level (D040) — **not** the record. See F013 (pooled), D040/D041.

**Finding.** On the 4 held-out test corridors (3,149 frames, class-agnostic locked config, all 9 models), the three arms are **indistinguishable on GT-1 lateral offset** — paired differences ≤2.8 mm (≤7% of the RTK floor), all block-bootstrap CIs include zero — confirming F013 on held-out data. On **GT-2 heading**, 2 of 3 pairs' CIs **exclude zero** (A-B −0.122°, B-C +0.035°; A-C includes zero): B's heading sits 0.035–0.122° above A's and C's — statistically detectable at test n but **practically negligible** (3–9% of the ~1.3° noise floor; a small fraction of the ~2.3–3.8° systematic tilt, F017).

**Evidence.** Per-arm test (across-seed, block-bootstrap CI): A GT-1 RMS 0.181 [0.167, 0.194] / GT-2 2.28 [2.11, 2.47]; B 0.183 [0.165, 0.200] / 2.33 [2.16, 2.51]; C 0.183 [0.167, 0.194] / 2.30 [2.14, 2.50] — all per-arm CIs overlap. Paired (across-seed): A-B GT-1 −2.8 mm [−11.1,+3.9] (incl 0), GT-2 −0.122° [−0.199,−0.031] (**excl 0**); A-C GT-1 −2.1 mm (incl 0), GT-2 −0.084° [−0.167,+0.002] (incl 0); B-C GT-1 −0.8 mm (incl 0), GT-2 +0.035° [+0.010,+0.057] (**excl 0**). `final/test_evaluation/line_fit_test_report.json`, `final/test_evaluation/paired_crossarm_test.json` *(both now under `superseded/march_val_test_split/test_evaluation/`)*. Coverage ~77% (all arms) vs val's ~83%; systematic tilt ~1.9° on test vs ~2.28° on val (F017's scene-dependent component).

**Interpretation.** The primary metric (GT-1) confirms F013 on held-out data — the arms are indistinguishable, sub-RTK-floor. On the complementary GT-2, the B arm shows a tiny, statistically-detectable-at-test-n but practically-negligible higher heading. **The B-vs-others GT-2 difference (0.035–0.122°) is smaller than the val-to-test change in the systematic tilt itself (2.28° → ~1.9°, a shift of ~0.4°). A per-arm heading difference of that magnitude cannot be reliably distinguished from residual per-arm sensitivity to the same corridor-level scene variation that produced the tilt shift. The paired significance is reported as what the test data shows, without asserting an arm-level mechanism that could not be distinguished from residual scene-sensitivity at this magnitude.** The operative conclusion (arms deliver equivalent navigation-relevant centreline geometry) holds.

**Caveats.** Single-shot test (not iterated). The GT-2 micro-difference is at the edge of metric resolution; its mechanism is not investigated (out of scope). **The val-to-test coverage drop (83→77%) is scene-explained, not pipeline:** the shared corridor 3 is consistent (val 90.2% / test 88.5%); the test set adds corridor 4 (test-only), the sparsest-detection corridor (mean base 21.3 vs 26–32 elsewhere; 64.0% two-row, 10.4% none), which drags the test average down — scene-harder, not a pipeline regression.

**Cross-references.** F013 (val — GT-1 confirmed, GT-2 partial divergence); F012 (noise floor); F017 (systematic tilt dominates GT-2; scene-dependent); F018 (locked config); Analysis H (block lengths); D033 (corridor-3 val/test split).

**Writeup wording (A2):**

**Fully defensible.** On the single held-out test evaluation (3,149 frames across four test corridors, all nine models at the locked class-agnostic configuration), the three arms are indistinguishable on the primary GT-1 lateral-offset metric: all per-arm CIs overlap (A 0.181 [0.167, 0.194] m, B 0.183 [0.165, 0.200] m, C 0.183 [0.167, 0.194] m) and every paired difference is ≤ 2.8 mm with a CI including zero (≤ 7 % of the 3.8 cm RTK floor), confirming F013 on held-out data. On the secondary GT-2 heading metric, two of three pairs' CIs exclude zero (A–B −0.122° [−0.199, −0.031], B–C +0.035° [+0.010, +0.057]; A–C includes zero): the B arm sits 0.035–0.122° above A and C. This is statistically detectable at the test sample size but practically negligible — 3–9 % of the ~1.3° regression-residual noise floor (F012) and a small fraction of the ~2.3° systematic tilt (F017); it is also smaller than the val-to-test change in the tilt itself (2.28° → ~1.9°, a ~0.4° shift), so it cannot be reliably separated from residual per-arm sensitivity to the corridor-level scene variation that produced that shift, and is reported as what the test data show without asserting an arm-level mechanism. The val-to-test coverage change (83 → 77 %) is scene-explained: shared corridor 3 is consistent (val 90.2 % / test 88.5 %), while the test-only corridor 4 is the sparsest-detection corridor (mean base points 21.3 vs 26–32 elsewhere; 64.0 % two-row), lowering the test average. The navigation-relevant conclusion — equivalent centreline geometry at the primary GT-1 metric — holds on held-out data.

**Candidate explanations.** The GT-2 B-vs-others micro-difference could reflect a real tiny B-arm heading tendency or residual scene-sensitivity; the two cannot be separated at this magnitude (not investigated, out of scope). The A–C GT-2 pair (which includes zero) suggests A and C behave similarly on heading; the B-vs-others structure could reflect a subtle B-specific mask-geometry difference. Not investigated, out of scope.

**NOT defensible.**
- ✗ call B "worse"/"better" (GT-2 difference sub-floor and not separable from scene-sensitivity; primary GT-1 indistinguishable).
- ✗ claim the test was iterated/re-run (single-shot, rule 5).
- ✗ attribute the coverage drop to a pipeline regression (scene-explained, corridor 4).
- ✗ extend the GT-2 significance to a general arm ranking (edge of metric resolution, mechanism-unattributed).

**Citation map.** Ours: `final/test_evaluation/line_fit_test_report.json` (per-arm CIs, coverage, per-corridor); `final/test_evaluation/paired_crossarm_test.json` (paired CIs) *(both now under `superseded/march_val_test_split/test_evaluation/`)*. Paper: Polvara §5.3 (RTK yardstick). Confirms F013; tilt context F017; floor F012.