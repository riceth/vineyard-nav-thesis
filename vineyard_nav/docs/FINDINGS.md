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

**Statistical anchor, honestly bounded.** The "~0.72" anchor carries a wide interval — roughly [0.66, 0.77] at 95% — the direct consequence of the 23-scene test ceiling (O006). Phase B/C comparisons against this anchor must be read against that width: only differences comfortably outside it are interpretable as real at the perception level, which is a further reason the headline cross-arm comparison lives at the geometric/command strands (D014), not here.

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
