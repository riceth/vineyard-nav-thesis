# DECISIONS.md

Running log of locked, provisional, and open decisions. Append-only — when a provisional or previously locked decision is refined, a new entry is added with `SUPERSEDES` reference. Old entries stay in the log for provenance. This document feeds the Methodology chapter of A2.

Statuses:
- **LOCKED** — final, no further discussion unless new evidence arrives
- **PROVISIONAL** — direction agreed, parameters pending empirical data
- **OPEN** — flagged, not yet resolved
- **SUPERSEDED** — decision was replaced by a later entry; kept for historical record

---

## D001 — Implementation order: baseline first
**Date:** 28 June 2026
**Status:** LOCKED
**Decision:** Baseline arm(s) implemented and evaluated before the contribution arm. Full pipeline — segmentation, geometry, control, evaluation — implemented and metrics produced on baselines before novel work.
**Rationale:** Supervisor's explicit instruction. De-risks the downstream pipeline (RANSAC, PID) before adding multiclass complexity on top.
**Note:** Original scope was Phase A binary → Phase B multiclass; refined to A → B → C after three-arm design locked (D021).

---

## D002 — U-Net implementation: scratch — **SUPERSEDED by D022 (2 Jul 2026)**
**Date:** 28 June 2026
**Original status:** LOCKED
**Original decision:** Scratch-implemented 4-level Ronneberger U-Net, no pretraining.
**Why superseded:** Post-supervisor-feedback three-arm redesign made U-Net binary one of three arms rather than the primary model. The "scratch for educational depth" argument became less critical, while the training risk (966 images, 31M params from scratch) remained. Refined to SMP + ImageNet pretrained encoder.

---

## D003 — No pretraining — **SUPERSEDED by D022 (2 Jul 2026)**
**Date:** 28 June 2026
**Original status:** LOCKED
**Original decision:** Phase A and Phase B U-Net trained from random initialisation on SemanticBLT.
**Why superseded:** See D002. Pretraining now used (D022).

---

## D004 — Mixed precision training (AMP) enabled from day one
**Date:** 28 June 2026
**Status:** LOCKED
**Decision:** `torch.cuda.amp` autocast + `GradScaler` enabled for all U-Net training. `GradScaler` instantiated once at training start, persisted across epochs, state saved in checkpoints. (YOLO handles precision internally via ultralytics.)
**Rationale:** 8 GB VRAM at native 640×640 with a 31M-param U-Net is tight. AMP roughly halves activation memory and is standard practice for memory-constrained segmentation training — numerical risk minimal and well-trodden.

---

## D005 — Training resolution: native 640×640, no downsampling
**Date:** 28 June 2026
**Status:** LOCKED
**Decision:** Train at native 640×640 resolution. Use AMP, gradient accumulation, or batch-size reduction if VRAM is tight. Do not downsample.
**Rationale:** Trunks and poles are thin vertical features. Downsampling kills thin-feature recall first, which is exactly the navigation-relevant signal.

---

## D006 — Label collapsing for binary arms (Phases A and B)
**Date:** 28 June 2026
**Status:** LOCKED
**Decision:** Binary foreground = pixels/instances covered by COCO annotations with `category_id ∈ {3, 5}` (pole, trunk). Background = everything else: pipe (cat 2), building (cat 1), robot (cat 4), vehicle (cat 6), and unannotated pixels.
**Rationale:** Both binary arms (Phase A U-Net binary, Phase B YOLO binary) apply the same collapsing rule so the A ↔ B comparison isolates architecture only. Pipe is excluded from foreground by design, not oversight (documented in A2 Methodology).

---

## D007 — U-Net output head: 2-channel softmax
**Date:** 28 June 2026
**Status:** LOCKED
**Decision:** Phase A U-Net outputs 2-channel logits, softmax + CE loss. NOT 1-channel sigmoid + BCE.
**Rationale:** No meaningful difference in training behaviour for binary; uniform code path.

---

## D008 — Scratch-training fallback rule — **SUPERSEDED by D022 (2 Jul 2026)**
**Date:** 28 June 2026
**Original status:** LOCKED
**Original decision:** Pre-committed mIoU thresholds (0.45 / 0.30 / 0.40) for fallback to SMP+ImageNet under scratch training.
**Why superseded:** Scratch U-Net replaced with SMP+ImageNet from day one (D022). Fallback rule no longer needed.

---

## D009 — Phase A loss function
**Date:** 28 June 2026
**Status:** LOCKED
**Decision:** `0.5 * CrossEntropy + 0.5 * Dice`, equal weighting. The Dice term is **equal-weight soft Dice across both classes** (mean of per-class soft Dice on softmax probabilities), NOT class-frequency-weighted. No class weighting on CE for binary.
**Rationale:** CE handles per-pixel classification. Dice handles foreground-background imbalance. Equal weighting is conventional starting point; can be revisited if early training shows pathology.
**Clarification (3 July 2026, not a supersede):** "Generalised" in PHASE_A_SPEC §6 means the multiclass generalisation of soft Dice with equal per-class weighting. The Sudre et al. (2017) Generalised Dice Loss (inverse-squared-volume class weighting) was considered and rejected as an over-correction for a 2-class problem where the foreground Dice term already addresses imbalance. Implemented in `segmentation/unet_binary/losses.py`.

---

## D010 — Multiclass loss — **SUPERSEDED by D023 (2 Jul 2026)**
**Date:** 28 June 2026
**Original status:** PROVISIONAL
**Original decision:** CrossEntropy + GeneralisedDice with capped class weights for multiclass U-Net.
**Why superseded:** Multiclass arm moved from U-Net to YOLOv11-seg (D021). Ultralytics handles loss internally (D023).

---

## D011 — Optimiser, schedule, training duration (Phase A U-Net only)
**Date:** 28 June 2026
**Status:** LOCKED (Phase A only)
**Decision:**
- Adam, lr=1e-4, weight_decay=1e-5
- Cosine annealing, T_max=100, eta_min=1e-6
- Max 100 epochs, early stopping on val mIoU (patience 15)
- Effective batch size 8 (true batch=8 if VRAM allows; else true batch=4 with grad_accumulation=2)
- Fixed seed = 42

**Note:** Phase B and C use YOLO's default optimiser and schedule via ultralytics (D023).

---

## D012 — Data splits — **SUPERSEDED by D024 (2 Jul 2026)**
**Date:** 28 June 2026
**Original status:** LOCKED
**Original decision:** Use Roboflow's existing splits (966/46/23).
**Why superseded:** Supervisor feedback — 23 test frames insufficient. Resplit to 70/20/10 stratified (D024).

---

## D013 — Augmentations (training only)
**Date:** 28 June 2026
**Status:** LOCKED (Phase A explicit; Phases B/C via ultralytics defaults, tuned for parity)
**Decision:** For Phase A: albumentations `HorizontalFlip(p=0.5)`, `RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)`, `Rotate(limit=10, p=0.5)`, ImageNet normalisation, `ToTensorV2`. No vertical flip. No aggressive cropping. Val/test get only normalisation.
For Phases B and C: ultralytics defaults, with augmentation intensity aligned to Phase A where controllable (HSV bounds, rotation, flip probabilities).
**Rationale:** Rows have orientation → no vertical flip. Full-frame geometry needed downstream → no cropping. Cross-arm comparability requires reasonably equivalent augmentation regimes.

---

## D014 — Three-strand evaluation framework
**Date:** 28 June 2026
**Status:** LOCKED
**Decision:** Three independent evaluation strands, all stratified by canopy state:
1. Perception — mIoU (Phase A) or mAP@50/precision/recall (Phases B/C) — per-arm only, not cross-arm-compared
2. Geometric — centreline error vs teleoperator trajectory — cross-arm comparable
3. Command-level — PID command smoothness — cross-arm comparable

Statistical treatment: bootstrap CIs over per-frame metric differences for pairwise comparisons. Effect sizes alongside point estimates. No p-values.
**Rationale:** Cross-arm comparability lives at the geometry and control levels where all arms feed identical pipelines. Perception metrics are internal to each arm.

---

## D015 — Logging: TensorBoard + CSV
**Date:** 28 June 2026
**Status:** LOCKED
**Decision:** Both logging channels active for every training run.
- TensorBoard for live training visualisation
- CSV per-epoch dump for post-hoc plotting; header: `epoch, train_loss, val_loss, val_metric_1, ..., lr` (metric columns vary by phase)
- CSV appended each epoch, not rewritten (survives crashes)
**Rationale:** TensorBoard answers "is training healthy now?"; CSV answers "give me a publication-ready plot in three lines of matplotlib."

---

## D016 — Reproducibility setup
**Date:** 28 June 2026
**Status:** LOCKED
**Decision:**
- Seeds set: `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`, `torch.cuda.manual_seed_all(42)`, `ultralytics.set_seed(42)` for YOLO
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- All dependency versions pinned in `requirements.txt`
- Git commit hash written into every checkpoint and every run directory's `git_commit.txt`
- Config snapshot saved per run
**Rationale:** Any reported metric must be traceable to a config, a commit, a checkpoint, and a CSV row.
**Clarification (3 July 2026, not a supersede) — concrete measures for bitwise reproducibility, found while validating the Phase A smoke run:** `cudnn.deterministic=True` alone was insufficient; two same-seed runs diverged. Full determinism (verified: two smoke runs → byte-identical `metrics.csv`) required all of:
1. `torch.use_deterministic_algorithms(True, warn_only=True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8` **set before the first CUDA/cuBLAS call** (setting it after `torch.cuda.*` is too late — cuBLAS GEMM stays nondeterministic).
2. Cross-entropy computed via one-hot + `log_softmax` (deterministic) instead of `nn.CrossEntropyLoss`, whose CUDA `nll_loss2d` kernel has no deterministic implementation in PyTorch 2.11. Mathematically identical (unit-tested); see D009.
3. Seeding albumentations via `A.Compose(seed=...)` — albumentations 2.x does **not** honour `random.seed()`/`np.random.seed()` globals, so augmentations otherwise vary run-to-run.
4. `num_workers=0` (also forced by the 64 MB `/dev/shm`; removes worker-ordering as a variable).
These are documented for the A2 Methodology reproducibility subsection.

---

## D017 — Class-aware downstream (Phase B) — **SUPERSEDED by D026 (2 Jul 2026)**
**Date:** 28 June 2026
**Original status:** PROVISIONAL
**Original decision:** Trunk-only RANSAC by default; fall back to combined trunk+pole when trunk pixel count < T. T ∈ {50, 100, 200, 400, 800, 1600} pixel counts.
**Why superseded:** Multiclass moved from Phase B to Phase C after three-arm redesign (D021). Sweep expanded to include Config A (trunk primary), Config B (pole primary), Config C (class-agnostic), and re-anchored to instance counts (D026).

---

## D018 — "Poles remain visible" framing retired
**Date:** 28 June 2026
**Status:** LOCKED (dissertation framing)
**Decision:** A2 Methodology and Discussion must retire the A1 proposal's "poles remain visible" framing. Replace with: "both trunks and poles degrade in visibility across canopy state, but both retain enough signal for class-aware combination to extract complementary information."
**Rationale:** Empirical data: pole retention ~24% vs trunk retention ~35% across canopy state. The A1 framing was overstated. Contribution argument survives — never depended on relative pole/trunk robustness.

---

## D019 — Folder structure
**Date:** 28 June 2026
**Status:** LOCKED
**Decision:** Code lives in `/workspaces/dissertation/vineyard_nav/`. Subfolders mirror the three-stage pipeline: `segmentation/`, `geometry/`, `control/`, `evaluation/`. Plus `data/` (symlink), `configs/`, `results/runs/<experiment>/`, `notebooks/`, `docs/`.
**Rationale:** Architecture mirrors the three-strand evaluation. Per-experiment run directories prevent cross-contamination.

---

## D020 — Statistical framework
**Date:** 28 June 2026
**Status:** LOCKED
**Decision:** Bootstrap confidence intervals over per-frame metric differences for pairwise comparisons. Effect sizes reported alongside point estimates. P-values intentionally excluded.
**Rationale:** Small test set makes p-values unreliable and over-promising. Bootstrap CIs and effect sizes are robust to small samples.

---

## D021 — Three-arm design
**Date:** 2 July 2026
**Status:** LOCKED
**Decision:** Three model arms replacing the two-arm A1 design:
- Arm 1 (Phase A): U-Net binary (SMP + ImageNet pretrained ResNet-34 encoder)
- Arm 2 (Phase B): YOLOv11-seg binary
- Arm 3 (Phase C): YOLOv11-seg multiclass (trunk + pole)

Isolated comparisons: A ↔ B (architecture effect, binary fixed); B ↔ C (class-structure effect, YOLO fixed).
**Rationale:** Supervisor feedback on architecture modernity. Preserves de Silva 2024 as reproduced baseline (Phase A) while adding modernised binary baseline (Phase B) and modernised multiclass contribution (Phase C). Two independent axes cleanly isolated — no confounding between architecture and class-structure effects. Also collapses what was previously a separate Phase C robustness check into the primary design.
**Consequences:**
- D002, D003, D008 superseded (no scratch U-Net)
- D010 superseded (no multiclass U-Net)
- D012 superseded (needs resplit — see D024)
- D017 superseded (Phase B → Phase C — see D026)
- No separate "Phase C robustness check" required

---

## D022 — U-Net binary: SMP + ImageNet pretrained encoder
**Date:** 2 July 2026
**Status:** LOCKED
**SUPERSEDES:** D002, D003, D008
**Decision:** Phase A U-Net implemented via `segmentation_models_pytorch` library. Encoder: ResNet-34 pretrained on ImageNet. Decoder: SMP default U-Net decoder. Fine-tuned on SemanticBLT.
**Rationale:** With U-Net as one of three arms rather than primary model, the "scratch for educational depth" justification is weaker while training risk on 966 images remains high. Pretrained encoder converges faster, more robustly, and provides a fair architectural comparison to YOLO's COCO-pretrained backbone.
**A2 documentation:** Methodology chapter documents refinement from A1's scratch commitment and provides rationale as a controlled-comparison move (fair architectural comparison to pretrained YOLO baseline).

---

## D023 — YOLOv11-seg for Phases B and C
**Date:** 2 July 2026
**Status:** LOCKED
**SUPERSEDES:** D010
**Decision:** YOLOv11-seg via ultralytics library. COCO pretrained. Fine-tuned on SemanticBLT-derived YOLO-format labels. Default optimiser (SGD + momentum) and schedule via ultralytics.
**Rationale:** YOLOv11 is the current version, well-supported by ultralytics, and matches supervisor's suggestion of a modern instance-segmentation architecture. Same architecture used for Phases B and C so the class-structure comparison is not confounded with architecture.
**Data preparation:** COCO polygon annotations converted to YOLO segmentation format (normalised polygon coordinates, one line per instance, `class x1 y1 x2 y2 ... xn yn`). Two path options: Roboflow re-export in YOLOv11 format, or in-place conversion via ultralytics `convert_coco()` utility.

---

## D024 — 70/20/10 stratified resplit with augmentation-leakage guard — **SUPERSEDED by D028 (3 Jul 2026)**
**Date:** 2 July 2026
**Status:** SUPERSEDED by D028 (3 Jul 2026)
**SUPERSEDES:** D012
**Why superseded:** Written before the Roboflow export was inspected. It assumed the 70/20/10 target could be expressed in *image* counts (725/207/103) while keeping augmentation groups intact. Inspection (3 Jul 2026) showed the export contains only **230 unique scenes**; Roboflow augmented only the 161 original-train scenes (6× each), while the 69 val/test scenes are single clean frames. Meeting the 207/103 image targets therefore *requires* placing 6× augmented copies of a handful of scenes into val/test, so "~100 test frames" would collapse to ~18 independent scenes — violating the independence assumption of the bootstrap CIs the resplit exists to enable (D020). Superseded by a scene-level resplit (D028). Original body retained below for provenance.
**Decision (original):** Resplit SemanticBLT 70/20/10 stratified by canopy state. Approximate final counts (from 1035 total): 725 train / 207 val / 103 test, with ~50 test frames per canopy bin.

**Split procedure:**
1. Identify unique base images (strip augmentation suffixes)
2. Split base images 70/20/10 stratified by canopy state (bare-vine: march/april; canopy: may/color_image)
3. Assign all augmentations of a base image to the same split as the base
4. Verify no leakage: no augmented duplicates of a base image in different splits
5. Save split assignments to a manifest file for reproducibility

**Rationale:** Supervisor feedback — Roboflow default (95/5/2, 23 test frames) is academically thin. 70/20/10 resplit brings test to ~100 frames (~50 per canopy bin), making bootstrap CIs meaningful. Canopy-state stratification preserves the per-bin analysis. Augmentation-leakage guard preserves experimental integrity.

---

## D025 — YOLO multiclass: trunk + pole only
**Date:** 2 July 2026
**Status:** LOCKED
**Decision:** Phase C YOLOv11-seg multiclass trained on 2 classes: trunk (class 0) and pole (class 1). Other annotated classes (pipe, building, robot, vehicle) dropped from training.
**Rationale:** Only trunk and pole feed downstream RANSAC. Training on all 6 classes introduces confounding: YOLO binary vs YOLO multiclass would differ in both class count AND supervision signal from unrelated classes. Trunk + pole only isolates purely "does distinguishing trunk from pole improve the pipeline?"
**Supplementary experiment:** All-6-classes multiclass kept as optional Phase C.2 if time permits. Would test whether richer supervision transfers back to trunk/pole detection quality.
**A2 documentation:** Methodology chapter documents refinement from A1's all-6-classes commitment.

---

## D026 — Phase C downstream sweep: 3 configs × 6 T values
**Date:** 2 July 2026
**Status:** LOCKED
**SUPERSEDES:** D017
**Decision:** Phase C downstream geometry tested in three configurations:
- **Config A:** trunk primary, pole fallback below threshold T
- **Config B:** pole primary, trunk fallback below threshold T
- **Config C:** class-agnostic (trunk + pole treated as one pool, no fallback logic)

**Sweep:**
- T ∈ {1, 2, 3, 5, 8, 12} instance counts (per side, per frame)
- Config C has no T parameter
- Total val evaluations: 6 (Config A) + 6 (Config B) + 1 (Config C) = 13
- Selection criterion: minimise RMS lateral error to teleoperator trajectory on val
- Test evaluated **once** at locked (config*, T*)

**Sensitivity reporting:** Full sweep curves plotted in Results (metric vs T for A and B; C as horizontal reference).

**Rationale:**
- Empirically testing which class should be primary rather than assuming (A1 assumed "trunks primary" without evidence)
- Config C isolates whether class-aware downstream logic itself matters, versus multiclass model just detecting better — enables attribution of any Phase C advantage
- T range anchored to mean per-side instance counts (~half of mean-per-frame; mean trunks/frame drops from ~16 bare-vine to ~6 canopy)
- Val-set selection + sensitivity reporting inoculates against "T as fishing knob" adversarial reading

**Attribution story:**
- B ≈ C → training on distinct classes doesn't itself improve detection quality
- C ≈ A/B → downstream class-aware logic doesn't matter; multiclass model just detects better
- C < A/B → class-aware fallback logic is where multiclass advantage originates

---

## D027 — No directional framing before Results chapter
**Date:** 2 July 2026
**Status:** LOCKED (working rule)
**Decision:** Introduction, Literature Review, Methodology, and Implementation chapters must state the research question neutrally and describe the comparison as "whichever direction the result lands" — never assert or imply that multiclass will outperform binary. Discussion chapter explains whichever way the results landed.
**Rationale:** Pre-stating a winner before experiments run undermines the methodology and gives a marker/reviewer grounds for questioning objectivity. The contribution is the *quality of the comparison*, not the direction of the result. All three possible outcomes (multiclass wins / binary wins / no difference) are publishable conclusions.

---

## D028 — Scene-level 70/20/10 resplit (evaluation on independent frames)
**Date:** 3 July 2026
**Status:** LOCKED
**SUPERSEDES:** D024
**Decision:** Resplit SemanticBLT at the **scene level** (unique base image = one physical scene), 70/20/10 stratified by canopy state, seed 42.

- **230 unique scenes → 161 train / 46 valid / 23 test scenes.** Per canopy stratum: bare-vine (110) → 77/22/11; canopy (120) → 84/24/12.
- **Augmentation-leakage guard:** every Roboflow augmentation of a scene inherits that scene's split. No scene's frames appear in more than one split.
- **Representative frame:** each scene has exactly one frame tagged `is_representative`. Selection is the lexicographically-first frame of the scene — a fixed, deterministic choice, because the true pre-augmentation original is not identifiable from the Roboflow export (all versions carry `.rf.<hash>` names).
- **Consumption rule:** training uses **all** frames in the train split (Roboflow augmentations plus on-the-fly albumentations/ultralytics augmentation per D013). Validation and test perception metrics are computed on **representative frames only** (one per scene), so reported bootstrap CIs (D020) are over genuinely independent scenes.
- **Manifest:** `data/splits/resplit_70_20_10.json`. Every image row carries `filename`, `orig_split`, `split`, `canopy_state`, `scene_id`, `is_representative`. Header block records seed, ratios, canopy rule, representative rule, source root, and both augmentation-inclusive and representative counts.

**Rationale:** The resplit's purpose (D024, D020) is meaningful bootstrap CIs. Bootstrap assumes independent evaluation units; six augmentations of one scene are not independent. Scene-level splitting with representative-only evaluation is the only construction on this data that makes the CIs honest. The alternative (image-count split per D024) buys a larger nominal test count purely by counting near-duplicate augmented frames as if independent — indefensible under marker questioning (working rule 2).

**Honest test size — acknowledged limitation:** the representative test set is **23 scenes (11 bare-vine + 12 canopy)**, close to the Roboflow default (23) the supervisor flagged as thin. This is a genuine ceiling of the dataset (only 230 unique scenes exist), not a design artifact. It must be raised with the supervisor and acknowledged in the A2 Discussion/Limitations. See O006.

**A2 documentation:** Methodology documents the resplit as scene-honest and explains representative-frame evaluation. Limitations documents the 23-scene test ceiling and why augmentation cannot substitute for independent scenes.

**Consequences and consumption pattern (added 4 July 2026):**
- Perception metrics (Phase A now; Phases B/C when they land) are computed on **representative frames only — one per scene**: 46 validation, 23 test (22/24 and 11/12 by canopy). Verified against `data/splits/resplit_70_20_10.json`, field `meta.counts.representative_by_split_canopy`.
- The augmented copies of validation/test scenes remain physically in the manifest (validation 211 total frames, test 103 total — the balance beyond the representatives; verified via the manifest `images` rows) but are **intentionally unconsumed** by perception evaluation and never reported as an evaluation count.
- **The earlier "dual-reporting / headline-on-103" idea is retired.** There is no 103-frame test figure. The test perception result *is* the 23-representative-scene result, and its uncertainty is expressed with bootstrap CIs (D020) over those 23 (and 11/12 per stratum). Augmentation-inflated counts are never presented as the evaluation n.
- Training still consumes all augmented frames in the *train* split; this clause governs *evaluation* only.
- No claim is made about how the source frames were acquired (e.g. video vs single-frame capture) — the manifest does not record acquisition method, so none is asserted.

---

## D029 — Test evaluation precision matches training precision
**Date:** 8 July 2026
**Status:** LOCKED

For arms trained under AMP mixed precision (Phase B, Phase C), test-time inference and evaluation are performed with `half=True` (FP16) to match training. This ensures evaluation code reproduces training-time val metrics exactly.

Rationale: mAP is a threshold-based metric sensitive to small numeric differences at prediction boundaries. Running training under FP16 and evaluation under FP32 introduced a systematic gap of ~0.024 in mask mAP@50 (0.6053 FP32 vs 0.6291 FP16, matching training's 0.6292). Same model, same weights — different precision, different metric. Matching precision resolves the discrepancy without changing the model.

Phase A (U-Net) is unaffected — training and evaluation both used the same AMP settings via the training script.

Cross-references: O003 (test metrics), F002 (Phase A reproducibility discipline established the precedent).

---

## D030 — Phase B conf-threshold selected on validation
**Date:** 8 July 2026
**Status:** LOCKED

The rasterised per-frame fg IoU used for Phase B's cross-arm perception comparison (F005) is measured at conf*, selected on the 46-scene validation set by argmax mean fg IoU across conf ∈ {0.10, 0.15, 0.20, 0.25, 0.30, 0.40}. Test evaluated once at locked conf*. mAP metrics (headline detection quality) are not affected by conf choice at this stage; they were reported at ultralytics defaults.

Rationale: aligns Phase B methodology with Phase C's val-based T-sweep. Provides a principled operating point rather than reliance on the ultralytics default (0.25). Preserves rule 5: threshold selected on val, test evaluated once at locked value.

Sweep result (val, n=46, half=True; `scripts/phase_b_conf_sweep.py`): mean fg IoU by conf = {0.10: 0.5655, 0.15: 0.5758, 0.20: 0.5793, 0.25: 0.5856, 0.30: 0.5852, 0.40: 0.5786}. **conf\* = 0.25** (argmax; 0.30 within 0.0004). Curve + data: `results/runs/phase_b_yolo_binary/val_conf_sweep.{png,json}`. Sensitivity discussed in F006.

**Outcome:** conf\* = 0.25 coincides with the ultralytics default used for the already-committed test result (O003), so that result **stands unchanged** as the locked Phase B test evaluation — no supersede, no test re-run (rule 5 preserved). Had conf\* differed, test would have been re-evaluated once at conf\* and the conf=0.25 files retained as `*_conf025_preliminary.json`; that branch was not taken. The coincidence is recorded for provenance: the operating point was validated post-hoc as optimal on val, not merely inherited from a default.

**Supplementary median-based analysis (8 July 2026, not a supersede):** `scripts/median_conf_sweep.py` computed both mean and median per-frame fg IoU across the 46 val frames at each grid conf, plus catastrophic-frame count (fg IoU < 0.1). **Median-based conf\* = 0.25, identical to the mean-based conf\*** — the two selection criteria coincide, so no mean-vs-median tradeoff arises. Catastrophic frames = 0 at every conf on val (the 6799-type failure appears on no val frame). Primary mean-based conf\* = 0.25 is unchanged; result discussed in F007. Data: `results/runs/phase_b_yolo_binary/val_conf_sweep_median.{json,png}`.

---

## D031 — Cross-arm perception comparison methodology: native metrics per arm, ranking deferred to geometric strand
**Date:** 10 July 2026
**Status:** LOCKED

Cross-arm perception-level comparison uses each arm's native metric:
- U-Net (semantic segmentation): mIoU + per-class IoU + precision/recall/F1 for foreground class
- YOLO (instance segmentation): mAP@50 + mAP@50-95 + per-class mAP + precision/recall

Direct arm-to-arm perception ranking is NOT performed at this stage. Rasterised foreground IoU (previously used as a cross-arm comparison metric in F005) is retained per YOLO arm only as an internal characterisation metric — useful for canopy stratification, blob-failure detection, and per-arm bootstrap CIs — but is not used to rank arms against each other.

Primary cross-arm comparison happens at:
- Geometric strand: RMS lateral error against teleoperator trajectory (all three arms produce a centreline estimate via RANSAC line-fitting after their per-arm perception outputs)
- Command-level strand: steering-command difference against teleoperator commands (all three arms feed the same PID controller structure)

Both strands await the geometry pipeline, which is scoped for O010 (post-multi-seed phase). Cross-arm ranking at the perception level is DEFERRED to the downstream stages.

**Rationale for retirement of rasterised fg IoU as cross-arm ranking metric:**

1. Not standard in the segmentation literature. Comparing instance-seg outputs to semantic-seg outputs by rasterisation is not a widely-adopted methodology; a reviewer would question the choice.

2. Rasterisation is a lossy transformation. Converting per-instance masks with per-instance confidence into a binary union mask discards granularity and selects one interpretation over others.

3. Contradicts our own D014 framework. D014 committed to "perception metrics differ across arms; cross-arm comparison happens at the geometric strand." The rasterised fg IoU cross-arm metric was introduced later as an implicit contradiction to D014 without acknowledging it.

4. F005's original framing has been retracted. F005 (revised) now describes rasterised fg IoU as a per-arm characterisation metric only.

**Cross-references.**
- D014: three-strand evaluation framework — this D031 aligns with D014.
- F005 (revised): describes the per-arm role of rasterised fg IoU.
- F003: Phase A baseline anchor at fg IoU 0.72; anchor is per-arm, not cross-arm.
- F007: uses rasterised fg IoU per arm as blob-failure characterisation; no cross-arm claim.
- O010: downstream sweep deferred to geometric-pipeline phase; will implement the cross-arm comparison via RMS lateral error.

**What this decision does NOT claim.**
- Does not delete the perception metrics from reporting. Report each arm's native metric fully with CIs.
- Does not claim perception is uninteresting; it characterises each arm's behaviour, and per-arm findings (F001, F007, F009) remain valid.
- Does not force removing comparisons that were legitimate. Where per-arm sensitivity analyses (D030 conf sweep, canopy stratification) benefit from rasterised fg IoU as an internal metric, they are retained.
- Does not commit to the geometric strand producing a "winner" — that's the empirical question the downstream strand will answer.

---

## D032 — YOLO variant choice (yolo11n-seg) rationale
**Date:** 11 July 2026
**Status:** LOCKED (retrospective justification)

YOLOv11-seg-nano (yolo11n-seg, 2.8M parameters) was selected as the study's YOLO baseline for computational feasibility on our hardware. All three arms use the same variant on the same scene-honest split, making the B↔C class-structure comparison internally consistent and not confounded by variant choice.

Compute constraint: RTX 5050 8GB VRAM. Larger variants (yolo11s-seg, yolo11m-seg, yolo11l-seg) would likely produce higher aggregate mAP scores but were not tested. yolo11m-seg at 640×640 typically requires 10-12GB training VRAM at batch size 16, potentially not fitting our hardware without batch-size reduction that would affect training dynamics.

Since the three-arm study isolates architecture (A↔B) and class-structure (B↔C) effects at fixed variant, the study's internal validity is not affected by choice of variant. External comparison to arbitrary published YOLO numbers is not the study's objective; the controlled comparisons are.

**Cross-references:**
- F007: architectural failure mode analysis; variant choice does not affect the mode's architectural availability.
- F009: multi-seed variance profile; per-arm characterisation, no external ranking.
- D031: cross-arm perception ranking deferred to geometric strand.

---

## D033 — Geometric-strand val/test split: pass-level (supersedes corridor-level)
**Date:** 12 July 2026
**Status:** LOCKED
**Refines:** GEOMETRY_PIPELINE_SPEC.md D-D (§3, §10); O010; CP-1 (March bag).

The geometric-strand evaluation on `kg_march_23.bag` splits eligible frames val/test at the **pass level** — the split unit is one **pass** (individual corridor traversal), NOT the corridor.

- 7,857 eligible frames form **11 pass-traversals across 5 corridors**, distributed **7 val / 4 test (60/40 by frames)**.
- **Val:** 7 passes (p2, p4, p5, p6, p7, p8, p10) = **4,708 frames**, corridors 0, 1, 3.
- **Test:** 4 passes (p0, p1, p3, p9) = **3,149 frames**, corridors 2, 3, 4.
- Δs = 1.5 m independence subsample: val 179, test 106.

**Rationale (supersedes corridor-level).** The earlier corridor-level split put **corridor 3 (42 % of eligible frames, traversed 4×) entirely in val**, risking the downstream sweep (config*/T*, D026) overfitting to corridor-3 conditions the test set lacks. Pass-level splitting **deliberately splits corridor 3 across val (1,890 frames) and test (1,412 frames)** so both splits contain it. **No single corridor exceeds 45 % of its containing split** (val max 40 %, test max 45 %).

**Structural limitation.** Corridors 2 and 4 have **only one pass each**, so they cannot appear in both splits — val-only corridors are 0 and 1, test-only are 2 and 4. This val/test corridor asymmetry is inherent to the single-bag (March) scope; **additional bags (April onward, the 6-bag scope) will progressively address it** as more traversals of each corridor become available.

**Cross-references:** GEOMETRY_PIPELINE_SPEC.md D-D / §3; D026 (downstream sweep); D028 (scene-honest split discipline, analogue); O010 (geometry pipeline).

---

## D034 — Geometric-strand image→world projection (CP-2) + D-G two-value half-spacing prior
**Date:** 13 July 2026
**Status:** LOCKED
**Refines:** GEOMETRY_PIPELINE_SPEC.md D-B, D-G, §5, §6; O010; CP-2.

The CP-2 image→ground projection (IPM) is built from the **Polvara et al. 2024 Table 3 extrinsics** (base_link → Zed2 Front) + bag intrinsics + a flat Z = 0 ground plane, validated on 22 well-detected val frames (parallel rows; correct centreline). Module: `scripts/geometric/projection_calibration.py`.

**Known limitation (accepted, not blocking).** Projection-measured corridor width is **median 1.91 m (IQR [1.59, 2.45])** — ~22 % narrower than the trajectory-derived **2.45 m** corridor spacing. The narrowing is **symmetric**, so it does **not** bias the primary two-row centreline metric (midpoint preserved); it shifts only width-dependent measures (the D-G single-row fallback prior). Likely cause: bbox-bottom projects to the visible inner edge of the trunk/pole rather than true ground contact, plus possible sub-cm pitch/height offset from Table 3 nominal. Refinement (true-ground-contact detection) is future work.

> **Update (D036–D038, 13 July 2026).** The ~22 % narrowing measured here was largely **adjacent-corridor + far-field-fan contamination of the row fit** (the CP-2 validation used a near-8 m line fit that mixed in far/neighbouring-row dots), not a pure projection error. The revised row model (hybrid clustering D036 + far-field extension D037 + line-fit D038) gives **width median ≈ 2.5–2.6 m** (near the true 2.45 m), substantially resolving this limitation. A residual sub-cm pitch/height offset and the ~2.3° yaw tilt (D038) remain. The extrinsics themselves (D-B) are unchanged.

**D-G half-spacing prior — two-value reporting.** The single-row fallback prior is reported at **two values side by side**: **1.2 m** (primary, trajectory-anchored — true vineyard geometry) and **0.96 m** (sensitivity, projection-consistent — half the measured 1.91 m width), making the projection-vs-truth sensitivity transparent.

**Cross-references:** GEOMETRY_PIPELINE_SPEC.md D-B (§6 extrinsics), D-G (§10), §5, §6; D033 (pass-level split); O010.

---

## D035 — Geometric-strand locked pipeline + GT-2 heading redefinition (CP-3)
**Date:** 13 July 2026
**Status:** **SUPERSEDED (13 July 2026)** by **D036** (hybrid clustering + RANSAC), **D037** (far-field extension), **D038** (line-fit centreline). The CP-3 artefacts (`scripts/geometric/single_arm_dryrun.py`, `single_arm_dryrun_report.json`, `single_arm_dryrun_samples/`) **remain committed as a locked historical state** (commit 32de7c8) — this entry is retained, not deleted, to document the row model CP-5 evolved past.
**Superseded because:** (1) the **near-field 5 m cutoff excluded valid same-row detections** — on frame 4107, 6 of the 8 left-row dots lie at X > 5 m, leaving a fragile 2-dot fit (→ D037); (2) the **Y-constant model missed a systematic ~2.3° common tilt** — frame 3998's right row visibly slants (m_R = +0.10), and the slope distribution over 3 910 frames is m_centre = +0.040 ± 0.026 (→ D038); (3) the **global-median fit landed in the gap between the true-row cluster and adjacent-corridor detections** — frame 4223's old fit sat at Y = +1.44 vs the true row at +0.6 (→ D036). **Retained forward:** the 15 % blob-area guard (below) and the CP-2 projection (D034) carry into D036–D038 unchanged.
**Refines (historical):** GEOMETRY_PIPELINE_SPEC.md §4 (steps 5–6), §5 (GT-2), §9 (CP-3), D-F; D034.

The CP-3 single-arm dry run (Phase C seed 42, all 4 708 val frames) locked the (now-superseded) row model and metric construction. Module: `scripts/geometric/single_arm_dryrun.py`; report `results/geometric/march/single_arm_dryrun_report.json`.

**Row model — near-field Y-constant.** Base points are restricted to the **near field X < 5 m** and each side is fit as a **Y-constant row** (median Y; valid if ≥ 3 points within 0.5 m of the median). **Rationale:** the CP-2 projection fan (D034) makes projected points fan outward with range, which *destroys per-row line fits* — a naïve RANSAC line-fit pipeline gave only **10 % two-row coverage**. Vine rows are at constant Y in the robot frame, so a constant model estimates the correct quantity (row lateral position) and is fan-robust, recovering **64.0 % two-row coverage** (single-row 26.7 %, none 9.2 %) on 4 708 val frames — ≈ 3 015 usable frames, ample for the sweep and bootstrap CIs.

**GT-2 heading — centreline-derived, fan-free (redefinition of D-F GT-2).** The per-row slope is fan-corrupted (spurious ±45°). But the fan is **symmetric** (left row +Y, right row −Y with range), so it **cancels in the centreline midpoint**. GT-2 is therefore redefined as the direction of the **centreline** between its 2 m and 4 m look-ahead bins, compared against the robot's actual heading from `/robot_pose` orientation (base_link frame). CP-3 gives an **unbiased** heading (mean −0.32°, |median| 3.21°, SD 7.73°), confirming the fan-free construction carries genuine signal. The robot's own row-misalignment is a common floor across all three arms (as with GT-1's centring floor), cancelling in the cross-arm ranking; the slip-corrected form (subtract path-tangent-minus-heading ≈ 0) is a documented refinement.

**GT-1 lateral offset.** Centreline lateral position at the 1–3 m (≈ 2 m) look-ahead bin. CP-3: mean +0.164 m, SD 0.144 m, |median| 0.167 m — consistent with near-centred teleoperation plus a small per-pass bias (reported separately, GT-1).

**Blob guard — lenient (CP-3 finding).** The F007 canopy-blob pathology does **not** manifest on the bare-vine March bag: the largest detections (~10.5 % of frame) are **real close-up trellis poles**, verified visually. An aggressive area cap would reject real poles, so the guard is set at **15 % of frame** — it drops 0 real detections in March val while still rejecting a gross whole-frame blob. The per-frame outlier defence is the row fit's median ± 0.5 m inlier test. (The blob guard remains relevant for the leafy April/June bags, future work.)

**Cross-references:** GEOMETRY_PIPELINE_SPEC.md §4–6, §9, D-F; D034 (projection fan); D033 (pass-level split); F007 (blob pathology, canopy scenes).

---

## D036 — Hybrid clustering + RANSAC row-fitting
**Date:** 13 July 2026
**Status:** LOCKED (supersedes the D035 global-median row fit)
**Refines:** GEOMETRY_PIPELINE_SPEC.md §4 (step 5); D035.

**Decision.** Fit each side's row by **densest-cluster seed → RANSAC refinement**, not a global median: (a) slide a **0.5 m Y-window** over the near-field (X < 5 m) points, take the densest window's median as the seed; (b) **RANSAC-refine** — search candidate row Y over **seed ± 0.3 m in 0.05 m steps**, pick the Y with the most inliers within **± 0.25 m**, refit as the median of those inliers; (c) **sanity checks** — reject if |Y| > 3 m, < 3 inliers, or X-span < 1 m (a horizontal blob, not a row); (d) **adjacent-corridor logging** — a secondary same-side cluster beyond the row band (|Y| > row |Y| + ~0.95 m) is recorded and rejected. The **15 % bbox-area blob guard** (from D035) is retained.

**Rationale.** The D035 global-median fit fails when a side contains **two clusters** — the true row and an adjacent corridor: the median of a bimodal set falls in the empty gap between the modes. The densest-window seed locks onto the true (nearest, most-detected) row; RANSAC tightens it; the adjacent cluster is logged and excluded.

**Alternatives considered.** (i) Global-median per side (D035) — lands in the inter-cluster gap on bimodal frames. (ii) Plain RANSAC line fit over all near-field points — fan- and adjacent-corrupted (only ~10 % two-row coverage at CP-3). Both rejected.

**Sample frames (visual verification, `results/geometric/march/diagnostics/figures/rowfit_validation/`).** 4223 — old median at Y = +1.44 sits between the true row (+0.6) and the adjacent corridor (+2.4…+4.7); 3991 — adjacent-left dots at Y ≈ −3.1 correctly rejected; 4107 — adjacent-right dots at Y ≈ +4 rejected.

**Cross-arm comparability.** Applied identically to all 9 models; a cleaner, arm-agnostic inlier selection. Preserves fairness — the same rejection logic runs for A/B/C.

---

## D037 — Far-field inlier extension
**Date:** 13 July 2026
**Status:** LOCKED (extends D036)
**Refines:** GEOMETRY_PIPELINE_SPEC.md §4 (step 5), §6 (projection range); D035 (near-field cutoff).

**Decision.** Project detections out to **X ≤ 10 m** (was 5 m). After the near-field (X < 5 m) hybrid fit (D036) establishes the row Y, **include far-field dots (5 m ≤ X ≤ 10 m) as inliers when within ± 0.5 m of the established row Y**; refit the row on all (near + far) inliers. Far dots outside ± 0.5 m are rejected as fan-noise or adjacent corridor.

**Rationale.** The near-5 m cutoff (D035) **discarded valid same-row detections** beyond 5 m, leaving fragile fits and abstentions. Real same-row trunks/poles at range project to nearly the **same Y** (within ± 0.5 m); only the adjacent corridor sits off-Y — so a Y-consistency gate safely re-admits the far same-row dots while still rejecting neighbours.

**Alternatives considered.** (i) Near-5 m only (D035) — 61.2 % two-row coverage, fragile near-field fits. (ii) Naïve near-10 m with no ± 0.5 m gate — re-admits adjacent-corridor contamination (the original bug). Rejected.

**Impact / evidence.** Two-row coverage **61.2 % → 83.1 %** (+1 030 frames rescued, **0 lost**) on Phase C s42 val. Frame 4107: left row went from 2 near-field dots (single_row) → **8 inliers** (two_row). Adjacent corridors are now visible and logged in **81 % of frames** — all rejected. GT-1 aggregate RMS rises ~29 mm, but only because the rescued (harder, sparse) frames are added — per-frame offset on already-two-row frames is unchanged (mean |Δ| = 6 mm).

**Sample frames.** 4107 (rescued single→two-row), 4223 (row extended along the full column, adjacent flagged "adj n=5"). `diagnostics/figures/rowfit_validation/far_ext/`.

**Cross-arm comparability.** Same extension and same ± 0.5 m gate for all arms; rescued frames are common across arms → paired differences unaffected. The coverage gain (and its added variance) is shared, so fairness holds.

---

## D038 — Line-fit centreline (per-side line regression)
**Date:** 13 July 2026
**Status:** LOCKED (supersedes the D035 Y-constant / bin-centre construction)
**Refines:** GEOMETRY_PIPELINE_SPEC.md §4 (step 6), §5 (GT-1, GT-2); D035; D-E, D-F.

**Decision.** Fit **Y = mX + c per side (least squares)** on the far-extension inliers (D037). Centreline = **midline** of the two fitted lines. **GT-1 lateral offset = centreline Y at X = 2 m** (the Pure-Pursuit look-ahead, D-E). **GT-2 heading = centreline slope in degrees** (arctan of the centreline dY/dX). **Width = mean Y_L − mean Y_R** (rows parallel, for the D-G prior). **Quality flags:** steep slope |m| > 0.3, L/R slope mismatch |m_L − m_R| > 0.2, fit failure.

**Rationale.** A slope analysis over 3 910 two-row frames found a **systematic common tilt**: m_L = +0.036 ± 0.043, m_R = +0.045 ± 0.042, **m_centre = +0.040 ± 0.026 → ~2.31° centreline heading**. The **corridor width is parallel** (width-slope m_L − m_R ≈ −0.009, symmetric around 0 → the "convergence" is noise), so the tilt is a **common lean**, most consistent with a small **unmodelled camera yaw (~2°)** — Table 3 (D-B) encodes pitch (q_y = 0.017) with q_z = 0 (zero yaw). The Y-constant model (D035) **washes this tilt out** and, by averaging Y over the whole range, produces a **~0.20 m range-bias** in GT-1 (Y-const offset RMS 0.332 vs **line-fit @ 2 m 0.226**). The far-extension adjacent-rejection (D037) cleans the inliers enough that the per-row slope is now reliable (**only 0.3 %** of frames flag |m| > 0.3) — resolving the original D035 "fan-corrupted slope" concern that had motivated the bin-centre GT-2.

**Alternatives considered.** (i) Y-constant median (D035) — range-biased by the tilt (RMS 0.332). (ii) Fitted-row all-inlier midpoint ("Option 1") — same all-range bias. (iii) Near-bin [1,3) midpoint — sparse (median 1–2 inliers/side; 18.8 % of frames get no heading). Line-fit uniquely uses **all inliers via the line** *and* evaluates at the **2 m look-ahead**, and yields a physically-grounded slope-heading. Chosen.

**Sample frames.** 3998 — right row m_R = +0.103 visibly slants, line-fit tracks it while Y-const stays vertical (offset +0.20 → +0.01); 4223, 4107 similar; near-vertical rows (m_L ≈ +0.02) keep near-zero slope (no over-fit). Histogram `slope_hist.png`; plots `diagnostics/figures/rowfit_validation/linefit_final/`.

**Cross-arm comparability & limitation.** The ~2.3° tilt is a **projection (likely yaw-extrinsic) effect common to all 9 models**, so **paired cross-arm differences cancel it**. Absolute GT-1 numbers include this systematic component but remain comparable across arms and are meaningful for absolute performance characterisation (stated wherever the number is reported). Documented limitation in Methodology; a future extrinsic re-calibration (add yaw) or LiDAR GT (GT-3) would remove it.

---

## D039 — U-Net binary geometry front-end (base-point extraction)
**Date:** 13 July 2026
**Status:** LOCKED
**Refines:** GEOMETRY_PIPELINE_SPEC.md §4 (steps 1–2).

**Decision.** The only arm-specific pipeline stage is base-point extraction. **YOLO arms (B, C):** per-instance **bbox-bottom-centre** on the 640² frame (detections back-mapped to 1920×1080 for projection), 15 % blob guard. **U-Net arm (A):** the foreground mask is **connected-component-labelled (8-connectivity)**, and each component ≥ **40 px** contributes its **bbox-bottom-centre**. Everything downstream (projection D-B/D037, row-fit D036/D037, centreline D038) is **arm-agnostic**.

**Rationale.** U-Net binary produces a foreground mask with **no instance separation**, so connected components stand in for YOLO instances. Validated on val frames: base points land on trunk/vine bases and feed cleanly through projection + row-fit.

**Alternatives considered.** Per-column lowest-foreground-pixel sampling (denser, not per-instance) — deferred; it would inflate U-Net point density relative to YOLO and confound the A-vs-YOLO structural comparison.

**Sample frames.** `phaseA_f3991`, `phaseA_f3993` (earlier validation) — CC base points on both rows.

**Cross-arm comparability.** This is the **single** arm-specific stage; keeping the downstream identical means the A-vs-YOLO contrast isolates the perception difference. The **~27 base points/frame (U-Net) vs ~31–33 (YOLO)** density difference is an **intentional, expected characteristic of binary-vs-instance perception** — the U-Net foreground fragments into connected components rather than resolving individual instances — **not a pipeline artefact or a disadvantage to the binary arm**. The downstream is **robust to this variation by design**: the row fit operates on the *distribution* of base points, not on an instance count — the **D036 densest-window + RANSAC** clustering needs only a dense-enough cluster to seed, and the **D038 line-fit** needs only ≥ 3 inliers per side, so both arms reach comparable two-row coverage despite the density gap (the median/least-squares aggregation absorbs the extra or fewer points). Reported transparently at CP-5 so the methodology write-up can attribute any residual A-vs-YOLO geometric difference to *perception*, not to base-point counting. (U-Net also shows ~2 pp more "none" frames — the same structural property.)

---

## Open items

### O001 — Threshold T range (Phase C)
Currently {1, 2, 3, 5, 8, 12} instance counts. May need re-anchoring after seeing YOLO multiclass detection densities on val. Locked in D026; range revisitable if densities differ substantially from expectation.

### O002 — All-6-classes supplementary experiment
Only if Phases A, B, C complete on time. Would test whether richer supervision transfers to trunk/pole detection quality on Phase C.

### O003 — Phase A + B + C test metrics
Appended to this file as each phase completes. Empirical basis for A2 Results.

**Phase A — U-Net binary (SMP + ImageNet ResNet-34). Test evaluated once, 4 July 2026.**
- Run: `results/runs/phase_a_unet_binary_20260704_004105/`; checkpoint `best.pt` @ epoch 42; git `5b4f1c05`; seed 42.
- Selection: best val mIoU 0.8456 @ epoch 42 (early-stopped @ 52, patience 10).
- Test split: 23 representative scenes (11 bare-vine + 12 canopy), one frame per scene (D028).

| Stratum | n | mIoU | IoU fg | Precision fg | Recall fg | F1 fg |
|---|---|---|---|---|---|---|
| Overall | 23 | 0.8561 | 0.7195 | 0.8618 | 0.8134 | 0.8369 |
| Bare-vine | 11 | 0.8414 | 0.6945 | 0.8470 | 0.7941 | 0.8197 |
| Canopy | 12 | 0.8858 | 0.7751 | 0.8926 | 0.8548 | 0.8733 |

- Test overall mIoU (0.8561) sits slightly above validation (0.8456) — no negative generalisation surprise.
- Reported factually, no cross-arm framing (D027). Point estimates only; per-frame bootstrap CIs over the 23 scenes are a follow-up (D020/O006) — evaluate.py must first export per-frame metrics. Not blocking Phase A closure.
- Test evaluated exactly once; not to be re-run (rule 5).

**Phase A multi-seed evaluation** (D016 verified reproducibility + O009 methodology):

| Seed | Test mIoU | Test fg IoU per-frame [95% CI] | Canopy > bare-vine gap |
|---|---|---|---|
| 42 | 0.856 | 0.712 [0.657, 0.766] | +0.072 |
| 43 | 0.861 | 0.725 [0.676, 0.775] | +0.077 |
| 44 | 0.857 | 0.712 [0.657, 0.765] | +0.079 |
| **Mean ± SD** | 0.858 ± 0.003 | 0.716 ± 0.008 | +0.076 ± 0.004 |

Per-seed 95% bootstrap CIs on test per-frame fg IoU average ±0.053 half-width (data variance). Training-run SD 0.008 is ~15% of data variance CI half-width, indicating training variance is small relative to data variance.

Canopy > bare-vine gap replicates directionally across all 3 seeds (all positive, mean +0.076 ± 0.004). See F001 for cross-arm and cross-split replication.

**Phase B — YOLOv11-seg binary (yolo11n-seg, COCO-pretrained). Test evaluated once, 8 July 2026.**
- Run: `results/runs/phase_b_yolo_binary/`; checkpoint `best.pt` @ epoch 86; git `7884bca`; seed 42.
- Data: `data/yolo_binary/` from `scripts/coco_to_yolo.py` (O005); D028 routing (train 721 / val 46 / test 23 representative).
- Training: 100 epochs, 45.2 min, peak VRAM 4.23/8 GB. Val mask mAP@50 0.629 reproduced exactly by `evaluate.py` (half=True, AMP-consistent — see methods note).
- Perception metric is mAP (D014); computed under FP16/AMP to match training-time validation (D004) and Phase A's AMP eval regime.

Phase B YOLOv11-seg binary (single-seed baseline: seed 42):
- Test mask mAP@50: 0.616
- Test box mAP@50: 0.722
- Test mask mAP@50-95: 0.289

Per-class (single class = crop):
- Overall: 0.616 mask, 0.722 box

Per canopy state:
- Bare-vine (n=11): 0.625 mask, 0.689 box
- Canopy (n=12): 0.619 mask, 0.829 box

Internal characterisation via rasterised fg IoU (per-arm; F005 revised):
- Overall: 0.556 [0.466, 0.633]
- Bare-vine: 0.562 [0.507, 0.619]
- Canopy: 0.551 [0.390, 0.687]
- Canopy > bare-vine gap: -0.011 [-0.178, +0.140]

conf* = 0.25 (val-selected per D030; sweep methodology also validated by median analysis; no catastrophic val frames at any threshold in sweep range).

**Phase B multi-seed evaluation** (D016 verified reproducibility + O009 methodology):

| Seed | Test mask mAP@50 | Test rasterised fg IoU [95% CI] | 6799 result |
|---|---|---|---|
| 42 | 0.616 | 0.556 [0.466, 0.633] | Blob (76,837 px, IoU 0.038, conf 0.406) |
| 43 | 0.648 | 0.589 [0.513, 0.655] | Blob (75,271 px, IoU 0.039, conf 0.264) |
| 44 | 0.633 | 0.609 [0.561, 0.656] | NO blob (largest mask 961 px, IoU 0.591) |
| **Mean ± SD** | 0.632 ± 0.016 | 0.585 ± 0.027 | Blob rate 2/3 |

Cross-seed blob overlap on 6799 (results/runs/phase_b_blob_overlap_6799/blob_overlap_s42_s43.png): seed 42 vs seed 43 mask IoU 0.93, centroid distance 5.6 px in a 640×640 image; near-identical geometry establishes the failure is not coincidental. See F007 for full analysis.

Training-run SD on mAP@50 (0.016) is smaller than SD on rasterised fg IoU (0.027) — the intermittent blob dominates fg IoU variance while affecting mAP@50 more mildly. See F009 for the metric-divergence analysis.
- Each seed's best.pt test-evaluated once at conf 0.25; not to be re-run (rule 5).

**Phase C — YOLOv11-seg multiclass (yolo11n-seg, trunk=0 / pole=1). Test evaluated once, 10 July 2026.**
- Run: `results/runs/phase_c_yolo_multiclass/`; checkpoint `best.pt` @ epoch 94; git `2a69c95`; seed 42.
- Data: `data/yolo_multiclass/` from `scripts/coco_to_yolo.py --mode multiclass` (O005 / D025); D028 routing (train 721 / val 46 / test 23 representative). Training regime **identical to Phase B** (100 epochs, patience 30, batch 16, workers 0, imgsz 640, SGD schedule, augmentation) — only `nc` and the data path differ (verified: B↔C non-cls training losses match to <0.01, F008).
- Training: 100 epochs (no early stop), 49.3 min, peak VRAM 4.25/8 GB. Val mask mAP@50 0.613 reproduced by `evaluate.py` (0.6126, half=True, D029).
- **Downstream sweep + test-at-locked-config: DEFERRED (O010).** Phase C closes at **perception only**.

**(a) Detection quality (mAP@50)** — `half=True` (D029); overall = trunk+pole class-mean.

| Stratum | n | mask mAP@50 | mask mAP@50-95 | box mAP@50 | trunk mask@50 | pole mask@50 |
|---|---|---|---|---|---|---|
| Overall | 23 | 0.6378 | 0.3100 | 0.7268 | 0.6778 | 0.5978 |
| Bare-vine | 11 | 0.6355 | 0.2971 | 0.6914 | 0.6708 | 0.6001 |
| Canopy | 12 | 0.6572 | 0.3582 | 0.8359 | 0.7101 | 0.6043 |

Per-class trunk > pole (mask 0.678 vs 0.598 overall) — noted, **not established as significant** (no per-class bootstrap CI at n=23; O009 multi-seed to confirm).

**(b) Pixel coverage of foreground union (rasterised fg IoU)** — class-agnostic trunk+pole union at conf 0.25; per-frame, bootstrap 95% CIs (D020); cross-arm-comparable (F005). NOT the same quantity as mAP@50 above.

| Stratum | n | fg IoU [95% CI] |
|---|---|---|
| Overall | 23 | 0.6185 [0.5724, 0.6662] |
| Bare-vine | 11 | 0.5708 [0.5207, 0.6244] |
| Canopy | 12 | 0.6623 [0.5946, 0.7275] |

- Canopy − bare-vine gap: +0.0914 [+0.0050, +0.1753] (excludes zero).
- **6799 (F007 informant): NO blob** — 14 detections (10 trunk, 4 pole), largest mask 989 px, fg IoU 0.627 (vs Phase B best.pt 0.038). Bounded interpretation in F007 (class-aware supervision vs clean-checkpoint; n=1, O009 decisive).
- Cross-arm (factual, no directional claim — D027; interpretation deferred to attribution): overall fg IoU 0.619 (C) vs 0.556 (B); much of the raw gap is the single 6799 frame (B 0.038 vs C 0.627); residual difference deferred.
- Artifacts: `test_metrics.json`, `test_per_frame_metrics.csv`, `test_bootstrap_ci.json`, `predictions_test/` (23 panels), `diagnostic/6799_visualisation/`.
- Test evaluated exactly once at conf 0.25; not to be re-run (rule 5).

**Phase C multi-seed evaluation** (D016 verified reproducibility + O009 methodology):

| Seed | Test mask mAP@50 | Test rasterised fg IoU [95% CI] | 6799 result |
|---|---|---|---|
| 42 | 0.638 | 0.619 [0.572, 0.666] | NO blob (989 px, IoU 0.627) |
| 43 | 0.653 | 0.586 [0.510, 0.654] | Blob (75,256 px, IoU 0.042, 16 dets) |
| 44 | 0.642 | 0.577 [0.502, 0.642] | Blob (76,035 px, IoU 0.038) |
| **Mean ± SD** | 0.644 ± 0.008 | 0.594 ± 0.022 | Blob rate 2/3 |

Per-class mask mAP@50 (mean across seeds):
- Trunk (class 0): ~0.69 ± 0.02
- Pole (class 1): ~0.60 ± 0.03
(Individual per-seed values in test_metrics.json files)

Cross-arm blob overlap on 6799 (results/runs/phase_c_blob_overlap_6799/): Phase C seed 43 blob shows mask IoU ~0.93 with Phase B seed 42 blob and Phase B seed 43 blob. All four blobbing runs (Phase B seeds 42, 43; Phase C seeds 43, 44) produce blob masks in the same right-side canopy region with near-identical geometry.

**Regeneration recipe** (the overlap PNGs live under `results/runs/` and are gitignored like every other run artefact; they are reproducible on demand, not merely held locally): `python scripts/blob_overlap_6799.py` — default runs are the four blobbing runs (Phase B seeds 42, 43; Phase C seeds 43, 44); predicts 6799 with each run's locked `weights/best.pt` at conf 0.25 (half=True, D029), takes the largest-area mask per run, and writes `overlap_<a>_<b>.png` + `overlap_summary.json`. Provenance: analysis script committed with this multi-seed pass; Phase B seed configs committed at 4044395, Phase C seed configs at d44cccf. Measured 4-way result: largest-mask areas 75,256–76,837 px; pairwise mask IoU mean 0.929 (range 0.924–0.937 across 6 pairs); centroids within ~6 px.

Blob rate across arms: Phase B 2/3, Phase C 2/3. Class-aware supervision does not affect the failure rate. See F007 for full analysis.

Training-run SD on rasterised fg IoU (0.022) is slightly lower than Phase B (0.027) but higher than Phase A (0.008). The intermittent blob failure dominates variance in both YOLO arms.

### O004 — Literature review extension
Supervisor flagged A1's 6 references as thin. Must reach ~12–15 for A2. Extension planned during dissertation writing phase.

### O005 — COCO→YOLO conversion: in-place script
**Date:** 4 July 2026
**Status:** LOCKED. Path B chosen (in-place COCO→YOLO conversion).

Decision: convert COCO polygon annotations to YOLO segmentation format via an in-repo script (`vineyard_nav/scripts/coco_to_yolo.py`), parameterised by class-collapse rule. Do NOT re-download from Roboflow in YOLO format.

Rationale:
- Preserves single source of truth (COCO JSON is master, YOLO labels derived)
- Applies D028 scene-honest split manifest directly (no override needed)
- Reusable for Phase C multiclass (same script, different collapse rule)
- Auditable, unit-testable, deterministic

Script uses ultralytics `convert_coco()` utility for polygon-to-YOLO conversion; adds split-manifest routing and class-collapse logic on top.

**Implementation note (4 July 2026):** `convert_coco()` writes `class = category_id − 1` (verified: pole cat 3 → 2, trunk cat 5 → 4) and normalised segments. The script runs it once over the three source COCO JSONs, then keeps only foreground classes (COCO cat ∈ {3,5}), rewrites them to the collapsed id, and routes frames per the D028 manifest. Frame routing honours the D028 consumption rule: **train = all 721 frames; val = 46 representative; test = 23 representative** (augmented copies of val/test scenes are not placed). Manifest split `valid` maps to the ultralytics `val/` directory.

### O010 — Phase C downstream sweep deferred to geometry-pipeline phase
**Date:** 10 July 2026
**Status:** DEFERRED. Not blocking Phase C closure or multi-seed pass.

The downstream sweep (3 configs × 6 T values on val, selection by RMS lateral error) requires a geometry pipeline that is scoped for a later phase of the dissertation (RANSAC + centreline + trajectory extraction from the ROS bag). Rather than build a proxy sweep now or a rushed geometry pipeline mid-Phase-C, the sweep is deferred to when the geometry pipeline is built as scoped.

The Phase C best.pt is locked and deterministic. Sweep can be run at any later point in the project without model retraining.

Preserves scope and stays faithful to the A1 proposal and PHASE_C_SPEC §8 commitment to RMS lateral error as the selection metric.

### O009 — Multi-seed evaluation planned post-Phase C
**Date:** 8 July 2026
**Status:** LOCKED. Committed for post-Phase-C robustness check.

After Phase C completes with seed 42, re-run all three arms with seeds 43, 44, 45, 46 for a 5-seed robustness check. Report per-arm mean and SD across seeds alongside per-run bootstrap CIs.

Rationale: single-seed studies characterise data variance (via bootstrap CIs) but not training-run variance. Multi-seed averaging is standard practice in ML methodology papers and strengthens conclusions from the three-arm comparison.

Cost: ~25-30 GPU-hours total (5 seeds × 3 arms × ~1-2 hours per run). Accepted for dissertation quality.

Constraints:
- Data locked (same D028 manifest, same 23-scene test set)
- Hyperparameters locked (same YAML configs)
- Each specific model evaluated exactly once on the 23-scene test — rule 5 applied per seed. That means 5 test evaluations per arm across the multi-seed pass, but each is of a distinct trained model, not of the same model repeated. The test data is the same; the models being evaluated are different.
- Bootstrap CIs computed per seed; means and SDs computed across seeds

Reporting format: "fg IoU 0.72 ± 0.03 (mean across 5 seeds)" alongside per-seed CIs.

Timeline: after Phase C closes, before A2 Results write-up.

**O009 status: COMPLETE.**

Multi-seed evaluation performed across all three arms with seeds 42, 43, 44 as directed by supervisor. Cross-arm summary:

| Arm | Test fg IoU (mean ± SD) | Test mAP@50 (mean ± SD, YOLO only) | 6799 blob rate |
|---|---|---|---|
| A (U-Net binary) | 0.716 ± 0.008 | mIoU 0.858 ± 0.003 (native) | 0/3 (structurally immune) |
| B (YOLO binary) | 0.585 ± 0.027 | 0.632 ± 0.016 | 2/3 |
| C (YOLO multiclass) | 0.594 ± 0.022 | 0.644 ± 0.008 | 2/3 |

**Findings.**

1. Phase A U-Net is the most stable across training seeds (fg IoU SD 0.008); structurally cannot produce the 6799 blob failure that both YOLO arms exhibit.

2. Phase B and Phase C exhibit identical 6799 blob rates (2/3 each) at val-fitness-selected best.pt. Blob geometry is invariant across seeds and arms (~0.93 pairwise mask IoU). Class-aware supervision does not affect the failure.

3. Phase C's mAP@50 variance (0.008) is notably lower than Phase B's (0.016). This may indicate class-aware supervision produces more consistent detection quality on non-blob frames, though n=3 is limited.

4. Canopy > bare-vine effect replicates directionally across all arms and all seeds (gap magnitudes in the +0.07-0.09 range for Phase A and Phase C; distorted in Phase B when blob is present).

**O009 closure implications for the dissertation.**

*Methodology chapter:* Multi-seed evaluation methodology (3 seeds per arm, 9 total training runs) documented as O009 methodology.

*Results chapter:* Multi-seed tables per arm; blob rate as a cross-arm reproducible failure characterisation.

*Discussion chapter:* Frame the 6799 blob as a YOLOv11-seg architecture-family × scene pathology, class-structure-invariant. Phase A's structural immunity to this failure mode is a real architectural advantage of per-pixel semantic segmentation over instance segmentation for thin-structure vineyard tasks. The failure rate (~67% at val-fitness selection) has real implications for downstream pipeline design (RANSAC robustness needed).

**Timeline:** O009 complete before A2 Results write-up. OOD annotation experiment (O007) planned as final work per Riccardo's guidance.

### O008 — opencv version drift from ultralytics install — RESOLVED
**Date:** 4 July 2026
**Status:** RESOLVED.
Installing `ultralytics` pulled `opencv-python==5.0.0.93` (non-headless) alongside the pinned `opencv-python-headless==4.13.0.92`; both ship a `cv2` module, so cv2 resolved to the newer non-headless build (a GUI-dependent package in a headless container, and two conflicting installs). Fix: `pip uninstall -y opencv-python`, then `pip install --upgrade opencv-python-headless` (→ 5.0.0.93). Verified afterwards: cv2 5.0.0 imports, `imread`/`imwrite`/`fillPoly` work, and ultralytics/YOLO/smp/pycocotools/albumentations/torch all import (torch unchanged, 2.11.0+cu128). `requirements.txt` pin updated 4.13.0.92 → 5.0.0.93; cv2 is now single-sourced and headless.

### O006 — Test set is 23 independent scenes (dataset ceiling)
The scene-honest resplit (D028) yields only 23 representative test scenes (11 bare-vine + 12 canopy), because the export contains just 230 unique scenes total. Augmentation cannot manufacture independent evaluation frames. Raise with supervisor: accept 23 scenes with strong bootstrap-CI caveats, or source additional raw frames. Must be acknowledged in A2 Limitations regardless.

### O007 — Out-of-distribution evaluation set (supervisor-requested)
**Date raised:** 4 July 2026 (supervisor feedback via Teams)
**Status:** Open. Not blocking Phase B/C. To be scoped in supervisor meeting.

Supervisor observation: test scenes are visually similar to train scenes (same vineyard, same season, same acquisition run). Even with scene-honest splitting, this is an in-distribution test, not a genuine held-out generalisation test.

Proposed remediation: manually label images from a different part of the vineyard or a different season to form an OOD evaluation set. This would sit alongside the existing 23-scene in-distribution test, not replace it. Reported separately in A2 Results.

Supervisor position: proceed with current pipeline setup (Phases B and C, downstream evaluation), then revisit dataset expansion once pipeline machinery is complete.

Timing: post-Phase C pipeline completion, before final A2 Results write-up. Meeting required to scope: how many images, which season/site, annotation tool and protocol, class scheme (same 6 as SemanticBLT? just trunk + pole?).