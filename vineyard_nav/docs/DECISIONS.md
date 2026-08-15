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
**Rationale:** Both binary arms (Phase A U-Net binary, Phase B YOLO binary) apply the same collapsing rule so the A ↔ B comparison isolates architecture only.

> **Correction (8 August 2026, additive — the text above is unchanged, and the decision it records stands).** The phrase describing **A ↔ B as isolating architecture is inaccurate and must not be reproduced in the dissertation.** Arms A and B differ in at least **thirteen** respects, verified against `configs/phase_a_unet_binary.yaml` and `configs/phase_b_yolo_binary_train.yaml`: architecture (U-Net/ResNet-34 vs YOLOv11-seg), pre-training corpus (ImageNet vs COCO), segmentation paradigm (semantic per-pixel vs instance), optimiser (Adam vs SGD), learning rate (1e-4 vs 1e-2, a factor of 100), weight decay (1e-5 vs 5e-4), schedule (cosine annealing vs linear), epochs (60 vs 100), batch size (8 vs 16), loss (0.5·CE + 0.5·Dice vs the YOLO box/seg/cls/dfl composite), early-stopping patience (10 vs 30), augmentation policy (minimal vs mosaic/HSV/affine), and output representation (dense argmax map vs instance masks with boxes). A ↔ B is therefore a **baseline-versus-modernised-pipeline comparison**, not a controlled architecture contrast, and no architecture-attributable claim may rest on it. **Only B ↔ C is controlled** — same backbone, same hyperparameters, same data, same augmentation, differing solely in label granularity — which is why the class-structure question is answerable and the architecture question is not. Pipe is excluded from foreground by design, not oversight (documented in A2 Methodology).

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
2. Geometric — centreline error vs teleoperator trajectory *(driven-path in current terminology; BLT autonomous, Polvara 2024)* — cross-arm comparable
3. Command-level — PID command smoothness — cross-arm comparable

Statistical treatment: bootstrap CIs over per-frame metric differences for pairwise comparisons. Effect sizes alongside point estimates. No p-values.
**Rationale:** Cross-arm comparability lives at the geometry and control levels where all arms feed identical pipelines. Perception metrics are internal to each arm.

> **Amendment (19 July 2026, additive — the original D014 text above is unchanged).** D014's **"teleoperator commands" / "teleoperator trajectory"** language is **imprecise** and should be read with this correction going forward. The March bag carries `/current_node` and `/closest_node` (`std_msgs/String`, the topological-navigation stack), indicating the BLT run was **very likely under existing autonomous navigation, not hand-teleoperation**. The command-level strand's evaluation reference (`/odometry/base_raw.twist.angular.z`) should therefore be described as **"executed yaw-rate from the BLT autonomy run,"** not "teleoperator commands." (The geometric-strand phrase "teleoperator trajectory" is already footnoted *"driven-path in current terminology; BLT autonomous, Polvara 2024"* at strands 1–2 above and at D031/D-F; this amendment extends the same correction to strand 3.) See **D042** (native-twist signal source) and **PID_PIPELINE_SPEC.md**.

> **Confirmed at primary source (9 August 2026, additive — no change of position; the hedge is removed).** The amendment above inferred autonomy from the presence of topological-navigation topics and called it *"very likely."* Polvara et al. (2024) states it directly. §3.2: *"we use the Topological Navigation Toolkit to enable autonomous navigation along all the corridors."* §1: *"driving autonomously along the crop rows."* **O020 is therefore established by the dataset paper, not inferred from topic names** — the platform ran autonomous topological navigation, and "teleoperator" is wrong as a matter of record rather than of probability. The convention *name* remains deliberately retained where it appears in `GEOMETRY_PIPELINE_SPEC.md` to match committed artefact keys.

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

> **Correction (9 August 2026, additive — the text above is unchanged and the retirement decision stands; the *grouping* the numbers describe was mislabelled, and the numbers themselves are replaced).** The entry states neither its grouping nor its deduplication basis. Both are now fixed, from `scripts/perception/diagnostics/output/per_image_stats.csv`.
>
> **The numbers reproduce, but they are not canopy-state numbers.** 24% / 35% is the **month-extreme** pair — mean instances per image in `unknown` against `march` (pole 24.1%, trunk 35.4%, all 1,035 rows). It is not a bare-vine → canopy contrast, which is what the entry claims. `unknown` (405 rows) is an *attribution bucket*, not a season.
>
> **Canonical replacement — pole 31.8%, trunk 37.3%**, grouped **bare-vine (march + april) → canopy (may + unknown)** and computed on the **230 deduplicated unique scenes** (`is_duplicate` excluded; 805 of the 1,035 rows are augmented variants of those same 230 scenes). Deduplication is not optional here: six augmentations of one photograph are one observation, and including them inflates n more than four-fold while adding no information — it would also weight scenes unequally, since augmentation counts are not uniform. For completeness, the same canopy grouping **with** duplicates gives pole 29.6% / trunk 36.4%, and the month-extreme pair deduplicated gives pole 25.4% / trunk 36.1%; all four variants are reported here so the choice is auditable.
>
> **What the retraction rests on.** Not the gap between the classes — under the correct grouping that gap *narrows* (31.8 vs 37.3, a 5.5 pp spread, against 24.1 vs 35.4), making the two classes **more** similar, not less. The retraction rests on **pole retention being 31.8% in absolute terms**: poles lose roughly **68% of their instances** under canopy, so they do **not** "remain visible" on a density measure. That is the whole argument, and it does not depend on any trunk comparison.
>
> **Two measures disagree, and both must be named.** On an image-level **presence** measure — the fraction of frames containing at least one instance — **poles are in 99.2% of canopy frames while trunks fall to 77.5%** (bare-vine: poles 100%, trunks 99.1%; 230 unique scenes). A1's wording is therefore **defensible on presence and wrong on density**: something pole-like is nearly always somewhere in the image, but the *number* of usable pole instances collapses by two-thirds.
>
> | measure | bare-vine → canopy, poles | bare-vine → canopy, trunks |
> |---|---|---|
> | **density** (mean instances/image) — *used here* | 12.06 → 3.84 (**31.8%**) | 15.96 → 5.96 (**37.3%**) |
> | presence (% frames with ≥ 1) | 100 → **99.2%** | 99.1 → **77.5%** |
>
> **Why this work uses density.** The row fit consumes **base points**, and a side seeds only when at least two fall inside the near-seed window (D037). One pole in frame contributes one base point and cannot seed a side; the fit is driven by *how many* instances are available, not by whether any exist. Presence is therefore the wrong measure for this pipeline even though it is the measure on which A1's wording survives. State the measure explicitly wherever either number appears — the two support opposite readings of the same data.
>
> Cite **31.8% / 37.3% density, bare-vine → canopy, 230 unique scenes** in A2, with the presence figures alongside as the honest counterpoint.
>
> **Independent sensor support for the direction (9 August 2026, additive).** The retraction's direction is corroborated by a different sensor entirely. Polvara et al. (2024) §5.1: *"Moving from late winter across spring and into summer, poles disappear from the map because the plants are now covered by leaves."* That is a **LiDAR mapping** observation, independent of the image annotations these percentages are counted from, and it runs the same way: pole availability falls as the canopy closes. Two measurement modalities, one direction — which is stronger support for retiring "poles remain visible" than the annotation counts alone. It corroborates the **direction only**; the 31.8% magnitude remains an annotation-density figure.

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

> **Correction (8 August 2026, additive — the text above is unchanged, and the decision it records stands).** The phrase describing **A ↔ B as isolating architecture is inaccurate and must not be reproduced in the dissertation.** Arms A and B differ in at least **thirteen** respects, verified against `configs/phase_a_unet_binary.yaml` and `configs/phase_b_yolo_binary_train.yaml`: architecture (U-Net/ResNet-34 vs YOLOv11-seg), pre-training corpus (ImageNet vs COCO), segmentation paradigm (semantic per-pixel vs instance), optimiser (Adam vs SGD), learning rate (1e-4 vs 1e-2, a factor of 100), weight decay (1e-5 vs 5e-4), schedule (cosine annealing vs linear), epochs (60 vs 100), batch size (8 vs 16), loss (0.5·CE + 0.5·Dice vs the YOLO box/seg/cls/dfl composite), early-stopping patience (10 vs 30), augmentation policy (minimal vs mosaic/HSV/affine), and output representation (dense argmax map vs instance masks with boxes). A ↔ B is therefore a **baseline-versus-modernised-pipeline comparison**, not a controlled architecture contrast, and no architecture-attributable claim may rest on it. **Only B ↔ C is controlled** — same backbone, same hyperparameters, same data, same augmentation, differing solely in label granularity — which is why the class-structure question is answerable and the architecture question is not.
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

> **Class identity sourced (9 August 2026, additive — no change to the decision; `pipe` is still dropped).** The repo's only gloss on `pipe` is in a regenerable diagnostic output (`scripts/perception/diagnostics/output/quality_observations.md`), which reads *"`pipe` (irrigation/trellis wire)"* — naming two different objects and committing to neither. **Polvara et al. (2024) Figure 9 caption commits:** *"Only the vertical poles and the horizontal water pipe are present in these two time snapshots."* The class is **irrigation water pipe**, not trellis wire. This does not affect D025 — `pipe` is dropped from training either way — but the Methodology chapter should name it correctly, and the ambiguous gloss should not be quoted.

> **Dataset attribution resolved (11 August 2026, additive — closes the CC BY 4.0 open item).** The Roboflow export credits only *"a Roboflow user"*, with no named author, which left the licence's attribution requirement unsatisfiable and was recorded as an open item. **The canonical citation is de Silva et al. (2025), "Keypoint Semantic Integration for Improved Feature Matching in Outdoor Agricultural Environments", arXiv:2503.08843**, who claim it as their contribution (iii): *"the Semantic Bacchus Long Term (SemanticBLT) dataset with panoptic segmentation annotations in vineyards"*. Their stated composition — **six classes (buildings, pipes, poles, robots, trunks, vehicles), 1,035 images, March to September, single vineyard** — matches the export exactly: 966 + 46 + 23 = 1,035 images across the three splits, the same six categories, and the March–September 2022 campaign window. **Cite de Silva et al. (2025); the anonymous Roboflow credit is not the attribution.** Note this is the same research group as the de Silva (2024) binary-mask baseline that arm A reproduces, and it means the dataset's own authors describe the annotations as *panoptic*, while this study consumes only the trunk/pole subset.

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
- Selection criterion: minimise RMS lateral error to teleoperator trajectory *(driven-path in current terminology; BLT autonomous, Polvara 2024)* on val
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

Sweep result (val, n=46, half=True; `scripts/perception/diagnostics/phase_b_conf_sweep.py`): mean fg IoU by conf = {0.10: 0.5655, 0.15: 0.5758, 0.20: 0.5793, 0.25: 0.5856, 0.30: 0.5852, 0.40: 0.5786}. **conf\* = 0.25** (argmax; 0.30 within 0.0004). Curve + data: `results/runs/phase_b_yolo_binary/val_conf_sweep.{png,json}`. Sensitivity discussed in F006.

**Outcome:** conf\* = 0.25 coincides with the ultralytics default used for the already-committed test result (O003), so that result **stands unchanged** as the locked Phase B test evaluation — no supersede, no test re-run (rule 5 preserved). Had conf\* differed, test would have been re-evaluated once at conf\* and the conf=0.25 files retained as `*_conf025_preliminary.json`; that branch was not taken. The coincidence is recorded for provenance: the operating point was validated post-hoc as optimal on val, not merely inherited from a default.

**Supplementary median-based analysis (8 July 2026, not a supersede):** `scripts/perception/diagnostics/median_conf_sweep.py` computed both mean and median per-frame fg IoU across the 46 val frames at each grid conf, plus catastrophic-frame count (fg IoU < 0.1). **Median-based conf\* = 0.25, identical to the mean-based conf\*** — the two selection criteria coincide, so no mean-vs-median tradeoff arises. Catastrophic frames = 0 at every conf on val (the 6799-type failure appears on no val frame). Primary mean-based conf\* = 0.25 is unchanged; result discussed in F007. Data: `results/runs/phase_b_yolo_binary/val_conf_sweep_median.{json,png}`.

---

## D031 — Cross-arm perception comparison methodology: native metrics per arm, ranking deferred to geometric strand
**Date:** 10 July 2026
**Status:** LOCKED

Cross-arm perception-level comparison uses each arm's native metric:
- U-Net (semantic segmentation): mIoU + per-class IoU + precision/recall/F1 for foreground class
- YOLO (instance segmentation): mAP@50 + mAP@50-95 + per-class mAP + precision/recall

Direct arm-to-arm perception ranking is NOT performed at this stage. Rasterised foreground IoU (previously used as a cross-arm comparison metric in F005) is retained per YOLO arm only as an internal characterisation metric — useful for canopy stratification, blob-failure detection, and per-arm bootstrap CIs — but is not used to rank arms against each other.

Primary cross-arm comparison happens at:
- Geometric strand: RMS lateral error against teleoperator trajectory *(driven-path in current terminology; BLT autonomous, Polvara 2024)* (all three arms produce a centreline estimate via RANSAC line-fitting after their per-arm perception outputs)
- Command-level strand: steering-command difference against teleoperator commands *(driven/autonomous commands in current terminology; see above)* (all three arms feed the same PID controller structure)

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

Since the three-arm study contrasts pipelines (A↔B) and isolates class-structure (B↔C) at fixed variant *(A↔B is not a controlled architecture comparison — see the correction at D006/D021)*, the study's internal validity is not affected by choice of variant. External comparison to arbitrary published YOLO numbers is not the study's objective; the controlled comparisons are.

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

**GT-1 lateral offset.** Centreline lateral position at the 1–3 m (≈ 2 m) look-ahead bin. CP-3: mean +0.164 m, SD 0.144 m, |median| 0.167 m — consistent with near-centred teleoperation *(driven-path; BLT autonomous, Polvara 2024)* plus a small per-pass bias (reported separately, GT-1).

**Blob guard — lenient (CP-3 finding).** The F007 canopy-blob pathology does **not** manifest on the bare-vine March bag: the largest detections (~10.5 % of frame) are **real close-up trellis poles**, verified visually. An aggressive area cap would reject real poles, so the guard is set at **15 % of frame** — it drops 0 real detections in March val while still rejecting a gross whole-frame blob. The per-frame outlier defence is the row fit's median ± 0.5 m inlier test. (The blob guard remains relevant for the leafy April/June bags, future work.)

> **Canopy blob check done (27 Jul 2026 — F007 geometric-stream audit).** The "future work" flagged above (does the blob pathology manifest on the leafy bags?) has been run: every eligible frame × 3 Phase-C seeds across march/april/may (~887k detections) yields **0** detections above the 15 % guard on any bag, canopy may included; the largest legitimate detections reach 14.0 % (april and may alike; march 10.5 %). The guard is never triggered on the data tested, and the audit is now emitted per bag automatically by `extract_detections.py` (`cache/blob_audit.json`). Full result + the bag-independent thin-margin caveat: F007 (27 Jul amendment).

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

## D040 — Pool March val+test into a single whole-bag evaluation
**Date:** 16 July 2026
**Status:** LOCKED
**Supersedes:** the held-out-test framing for March (CLAUDE.md rule 5 as applied *within-bag*; the val/test purpose of D033). **Refines:** FINDINGS F013/F019; POOLING_SPEC.md.

**Decision.** Pool the March (`kg_march_23`) validation (passes 2/4/5/6/7/8/10, 4,708 eligible frames) and held-out test (passes 0/1/3/9, 3,149 eligible frames) into a **single whole-bag evaluation** of all **7,857** eligible frames. Every geometric analysis (line-fit evaluation, paired cross-arm bootstrap, config sweep + single-class ablations, LiDAR cross-check) is re-run on the pooled set. The class-agnostic downstream config (F018) is **already locked and is NOT re-selected** — the sweep is re-reported on pooled data for consistency only.

**Rationale.** The within-bag val/test split served two procedural purposes: methodological consistency with the perception phases, and leakage-control for the config lock (F018 was selected on val and locked *before* the single-shot test, CP-6). With the config now locked, the split has served its purpose. The **seasonal-generalisation claim is made at the multi-bag level** (whole-bag evaluation per month, no per-bag splits) — the within-March split does not itself test seasonal generalisation. Pooling therefore (i) maximises statistical power for the March baseline, (ii) standardises methodology across all bags (each bag evaluated whole), and (iii) simplifies the write-up to "pooled per-bag result, generalisation confirmed cross-bag."

**Consequence (accepted).** March no longer has an independent within-bag held-out check; the seasonal-generalisation claim rests entirely on the multi-bag comparison. This trade — maximum power + cross-bag methodological consistency, minus the within-March held-out confirmation — is accepted as defensible given the multi-bag design provides the genuine generalisation test. Rule 5 (single-shot test) applies at the multi-bag level for the geometric strand.

**Findings impact.** F013 becomes the pooled March cross-arm finding (drops the "held-out / confirmed on test" framing). F019 (the CP-6 held-out-test finding) is retained as a historical trail with a SUPERSEDED banner; its purpose (test-side confirmation) is absorbed into the pooled F013. **This decision supersedes the *interpretation that F019 was load-bearing*, not the F019 record itself.** The per-finding "Test-side confirmation (CP-6)" blocks (F010/F011/F012/F014/F016/F017/F018) are merged into each finding's main measured content.

**Alternatives considered.** (i) Keep the val/test split and report both — rejected: adds write-up complexity ("val + held-out test + multi-bag") without testing the generalisation claim, and costs statistical power. (ii) Re-select the config on pooled data — rejected: the config is locked (F018); re-selecting on pooled data (which includes the former test set) would be exactly the leakage the split was designed to prevent. The sweep is re-reported, not re-decided.

**Scope.** Restructures `results/geometric/march/final/` (val_evaluation + test_evaluation → `march_evaluation/`; the val/test artefacts move to `superseded/march_val_test_split/`). New pooled scripts; val/test scripts move to `scripts/geometric/superseded/`. Perception, CP-0/1/2/3 artefacts, and the retained diagnostic/superseded material are untouched. Execution contract: `POOLING_SPEC.md`.

---

## D041 — Frame accounting for the March bag evaluation
**Date:** 16 July 2026
**Status:** LOCKED
**Refines:** CP-0 contamination criteria; CP-1 eligibility criteria (`GEOMETRY_PIPELINE_SPEC.md` §3, §7, D-D — locked 11 Jul 2026); D040 (whole-bag pooling scope); `POOLING_SPEC.md`. **Documents:** the evaluation scope for methodological transparency (A2 Methodology).

**Decision.** The **16,656** `kg_march_23` camera frames are categorised into exactly **three mutually-exclusive, exhaustive** buckets (contamination-first ordering; verified against `dataset_manifest.json` — all pairwise intersections empty, zero frames uncovered):

| # | Category | Count | % | Definition (manifest flags) | Treatment |
|---|---|---|---|---|---|
| A | **In-row eligible** | **7,857** | 47 % | `inrow ∧ ¬contaminated` | primary pooled evaluation (D040; F010–F018 pooled) |
| B | **Contaminated** (CP-0 leakage) | **2,958** | 18 % | `contaminated` (taken first) | excluded from all evaluation |
| C | **Non-in-row** (clean) | **5,841** | 35 % | `¬contaminated ∧ headland` | deployment-gap characterisation (Commit 6, F020+) |
| | **Total** | **16,656** | 100 % | | |

C sub-splits into **3,946 row-end stops** (`headland ∧ stationary`) + **1,895 turns / corridor transitions** (`headland ∧ moving`).

**Accounting property.** 7,857 + 5,841 + 2,958 = 16,656 — every frame in the bag has a documented treatment; there is no silent exclusion. The partition holds *by construction*: `inrow ⟹ ¬stationary` (in-row requires along-row speed |v_y| > 0.30 m/s, exceeding the v_min = 0.10 m/s stationary threshold — verified `inrow ∧ stationary = 0`), so the eligible set reduces to `inrow ∧ ¬contaminated` and stationary removal adds no exclusions beyond headland.

**Rationale.**
1. Mutually exclusive and exhaustive — the Methodology chapter can state exactly how all 16,656 frames are treated; no silent exclusion of 53 % of the bag.
2. **Contamination-first ordering** ensures leakage-affected frames (which may themselves be in-row or headland) do not confound either strand. Contamination is a perception-leakage exclusion (frames overlapping the SemanticBLT segmentation training set within ±1.0 s of a CP-0 exclusion interval); it reflects memorisation, not deployment reality, so it is documented but not re-evaluated.
3. **In-row and non-in-row are evaluated separately** because they answer distinct questions with distinct metric interpretations: in-row RMS (A) measures pipeline-design performance and is comparable across arms and bags; the non-in-row metric (C) is a *driven-path error* characterising deployment-gap behaviour with explicit caveats — the flat-ground IPM projection is invalid on headland slopes, the row centreline is undefined on turns, and turn geometry conflates with the error. They are **not** conflated into one "RMS".

**Consequence (Methodology chapter).** The evaluation scope is stated explicitly: *"This work evaluates the pipeline on 7,857 in-row frames (47 % of the 16,656 total). A further 5,841 frames (35 %) are characterised as non-in-row deployment-gap behaviour with explicit metric caveats. 2,958 frames (18 %) are excluded due to perception training-set overlap."* This preempts the "what about the rest of the bag?" question with a concrete accounting.

**Documentation trail.** CP-0 contamination criteria (`GEOMETRY_PIPELINE_SPEC.md` §2, D-C); CP-1 eligibility criteria (`GEOMETRY_PIPELINE_SPEC.md` §3, §7, D-D — locked 11 Jul 2026); pooling scope (D040); operational cite-ready counts (`GEOMETRY_PIPELINE_SPEC.md` §3); non-in-row characterisation (Commit 6, F020+ as they land).

**Cross-references.** D033 (CP-1 pass-level split of the eligible set); D040 (whole-bag pooling); CP-0 contamination census; `GEOMETRY_PIPELINE_SPEC.md` §3 (operational reference).

---

## D042 — PID state-gate signal source: native bag twist (supersedes F022's pose-finite-difference *for the control strand*)
**Date:** 19 July 2026
**Status:** LOCKED (signal-source decision). **Gate thresholds and rejection/FP rates are NOT yet validated** on the native signal — see the carry-over caveat below.
**Refines:** F022 (state gate, `mitigation_analysis.py`); D014 command-level strand (+ its 19 Jul 2026 amendment). **Feeds:** `PID_PIPELINE_SPEC.md` §1, §3. **Does not alter** the March geometric strand — F022's numbers stand as the *geometric-strand* mitigation result.

**Decision.** For the PID/control strand the state gate reads the robot's **native measured twist from the bag**, rather than re-deriving velocity from `/robot_pose` position replay (F022's approach in `mitigation_analysis.py`):
- **Primary signals** — `/odometry/base_raw` (`nav_msgs/Odometry`), field `twist.twist`: `linear.x` (v_x, forward), `linear.y` (v_y, lateral), `angular.z` (yaw-rate). Frame `base_link`; frame-synced 1:1 with each camera frame (timestamps byte-identical to the RGB topic — verified).
- **Cross-check** — `/imu/data.angular_velocity.z` (independent measured yaw-rate), read alongside and used to sanity-check the odometry yaw-rate (agreement reported; disagreement flagged), in the spirit of F017's sensor-common cross-check.

**Rationale.** A real onboard controller reads the measured body twist directly from the base/odometry EKF; it does not have access to an offline, whole-trajectory, 15-sample **centred** (i.e. **non-causal** — it peeks at future frames) finite-difference of a GPS-fused global position. Native twist is therefore **more representative of deployed behaviour**, is causal, and is the signal F022's pose-difference gate was only a proxy for.

**Carry-over caveat (must be honoured before the native gate is used as validated).** F022's validated numbers — **98.4 % non-in-row rejection, 1.2 % in-row false-positive**, at HR_THRESH = in-row p99 = **22.1 deg/s** — were derived on the **pose-finite-difference** signal and **do NOT transfer automatically** to native twist (different noise, bias, latency, scaling). Therefore, before use:
1. Re-fit the equivalent gate thresholds (`v_min`, `|v_y|` floor, heading-rate ceiling) against the native signal, using F022's in-row-p99 methodology (or an explicitly documented variant).
2. Re-derive and re-validate the **non-in-row rejection rate** and the **in-row false-positive rate** on the native signal.
3. Report the result as a **new finding (F026, or the next available number — D044)**, presented **alongside the original F022** so the two signal sources are directly comparable. Until that finding lands, the native-twist gate is **not** treated as validated.

**Cross-references.** F022 (superseded *for the control strand only*; stands for the geometric strand); F017 (sensor-common cross-check precedent); D043 (the other control-strand runtime layer); D044 (findings numbering); `PID_PIPELINE_SPEC.md` §1, §3.

> **Amendment (20 July 2026, additive — the D042 decision above stands; this corrects the gate's *predicate*, per the CP-P1 result F026).** The native gate was scoped as F022's three predicates re-fitted (speed, along-row `|v_y|`, heading-rate). CP-P1 showed this rests on a **world-frame-vs-body-frame error**: `/odometry/base_raw.twist.linear.y` is **body-lateral slip** (~0.05 m/s in-row), **not** the world-frame along-row velocity F022's `v_y` measured — a literal `|v_y| > 0.30` keep-predicate retains only **1.5%** of in-row frames. In the base_link body frame F022's "moving" and "moving-along-row" predicates **both collapse to forward `v_x`**, and the turn predicate is **inactive** (on native signals it adds **zero** marginal non-in-row rejection and only in-row false positives — so keeping it "for F022 parity" is not a real justification). **The locked native gate is therefore a single forward-speed predicate: `v_x > V_MIN`** (V_MIN = in-row p1 of `v_x` = **0.30 m/s**), turn predicate **dropped**. **Validated performance (F026): 97.5–97.6 % non-in-row rejection at 0.9 % in-row FP**, arm-invariant — reproducing F022's 98.4 % / 1.2 % on the deployable causal signal (the ~0.8 pp rejection shortfall is the transition category, a genuine body-frame limitation, not tuning). The odom yaw-rate is additionally an unreliable standalone signal (disagrees with the IMU gyro — F026), but the locked gate does not use it. See **F026** and `PID_PIPELINE_SPEC.md` §3.

---

## D043 — F024 abstention handling: hold-last-command + dual-metric evaluation
**Date:** 19 July 2026
**Status:** LOCKED
**Refines:** F024 (in-row abstention, `single_row_analysis.py`). **Feeds:** `PID_PIPELINE_SPEC.md` §6 (hold logic + span flagging), §7 (dual-metric definition).

**Decision (controller behaviour).** When the pipeline emits **no centreline** for a frame (`cls != two_row` — i.e. `single_row` or `none`, the F024 abstention set, **12.8–13.9 %** of in-row frames), the controller **holds its last valid commanded yaw-rate** (the most recent command produced from a `two_row` frame). "Hold-last" is the **actual runtime behaviour to implement**, not merely an evaluation convenience. *(Forward-linear-velocity handling during a hold, and any hold-duration safety cap, are open sub-questions for the spec — not decided here.)*

**Decision (evaluation).** Command-level tracking and smoothness metrics are computed **twice** over the same in-row frame stream:
1. **Inclusive** — over **all** in-row frames, including held-command frames.
2. **Exclusive** — over **only** the frames where the pipeline produced a fresh centreline (held frames removed).

Reporting both makes the **effect of held commands on the metric visible** rather than silently absorbed (a held command can flatter a smoothness metric and distort a tracking metric). Both are documented as **two views of the same finding**, not competing findings — the difference between them *is* the quantified command-level cost of abstention.

**Rationale.** F024 established that abstention is evidence-based conservatism (not failure) and is arm-consistent (≤1.1 pp spread). Hold-last is the minimal safe response that neither fabricates a centreline nor stops dead mid-row. Inclusive-vs-exclusive reporting keeps the command-level comparison honest and cross-arm-fair — all arms abstain at similar rates, so the dual metric stays comparable across arms.

**Cross-references.** F024 (abstention characterisation); D042 (the other runtime layer); D044 (findings numbering); `PID_PIPELINE_SPEC.md` §6, §7.

---

## D044 — PID/control-strand findings numbering
**Date:** 19 July 2026
**Status:** LOCKED

**Decision.** PID/control-strand findings **continue the main `F0xx` series in `FINDINGS.md`** (next available number; the first is expected to be **F026**, under D042's re-validation requirement). There is **no separate `PID_FINDINGS.md` file**. This mirrors how the geometric-strand findings (F010–F025) were tracked: one findings file, one monotonic series, with the four-part writeup discipline.

**Rationale.** A single findings series keeps cross-strand references simple (e.g. a control finding citing F013's centreline RMS or F022's gate) and keeps the marker-facing narrative in one place. Separate files fragment the audit trail with no benefit.

**Cross-references.** `FINDINGS.md` (F010–F025 geometric strand); D042 (expects F026); D043; `PID_PIPELINE_SPEC.md`.

---

## D045 — Repository reorganisation (Phase 1: `scripts/`) + path and invocation maps
**Date:** 20 July 2026
**Status:** LOCKED
**Purpose.** `scripts/` was reorganised so that reusable pipeline code and one-time diagnostics are separable by folder, and so that all perception code (including the segmentation package) sits under `scripts/perception/`. **Historical path citations elsewhere in this file, in `FINDINGS.md`, `STATUS.md` and the phase specs are left exactly as written** (additive-preservation); this entry is the single authoritative map that makes them resolvable. A reproducer needs **both** maps below — one to *locate* files, one to *run* them.

**(a) Path map (old → new).**

| Old path | New path |
|---|---|
| `segmentation/**` (whole package) | `scripts/perception/segmentation/**` |
| `scripts/perception/pipeline/coco_to_yolo.py` | `scripts/perception/coco_to_yolo.py` |
| `scripts/perception/pipeline/resplit_dataset.py` | `scripts/perception/resplit_dataset.py` |
| `scripts/perception/diagnostics/blob_overlap_6799.py` | `scripts/perception/blob_overlap_6799.py` |
| `scripts/perception/diagnostics/phase_b_conf_sweep.py` | `scripts/perception/phase_b_conf_sweep.py` |
| `scripts/perception/diagnostics/median_conf_sweep.py` | `scripts/perception/median_conf_sweep.py` |
| `evaluation/bootstrap.py` | `scripts/perception/bootstrap.py` |
| `scripts/utilities/inspect_bag.py` | `scripts/perception/diagnostics/inspect_bag.py` |
| `analysis/blt_analysis.py`, `analysis/blt_report.py`, `analysis/output/` | `scripts/perception/diagnostics/` |
| `geometry/`, `control/` (empty `__init__.py` only), `notebooks/`, `evaluation/` | **deleted** (dead stubs, zero importers) |

`scripts/geometric/` and `scripts/control/` are unchanged. New empty `scripts/__init__.py` and `scripts/perception/__init__.py` make the tree importable for `-m` invocation.

**(b) Invocation map (old → new).** Still run from `vineyard_nav/`:

| Old command | New command |
|---|---|
| `python -m segmentation.unet_binary.train` | `python -m scripts.perception.segmentation.unet_binary.train` |
| `python -m segmentation.yolo_binary.train` | `python -m scripts.perception.segmentation.yolo_binary.train` |
| `python -m segmentation.yolo_multiclass.train` | `python -m scripts.perception.segmentation.yolo_multiclass.train` |
| (same pattern for each arm's `.evaluate` / `.visualize`) | `python -m scripts.perception.segmentation.<arm>.<entry>` |

**(c) Import map.** `from segmentation.<arm>.<mod> import …` → `from scripts.perception.segmentation.<arm>.<mod> import …` (24 sites / 15 files). The **canonical dotted path is the only supported form**: a `sys.path`-based alias was rejected because the same module reachable under two top-level names can be instantiated twice in one process (distinct module objects, duplicated state). Cross-subpackage imports *inside* the package are relative (`from ..yolo_binary.visualize import …`).

**Rationale.** (1) Segmentation is perception work; keeping it a sibling of `scripts/` obscured that. (2) `pipeline/` was a redundant nesting level once `diagnostics/` carries the reusable-vs-one-time distinction. (3) Two scripts sitting in `diagnostics/` are in fact reproduction-critical — `blob_overlap_6799.py` is DECISIONS' own *"Regeneration recipe"* for the F007/O009 artefacts, and `phase_b_conf_sweep.py` selected the locked operating point conf\* = 0.25 (D030); `median_conf_sweep.py` is cited in D030's supplementary note, so under the "every claim backed by a committed script" rule it must stay runnable, not archived.

**Verification performed.** All three arms' `python -m …train --help` resolve; canonical imports load; `compileall` clean over `scripts/`; and re-running `state_gate_native.py`, `line_fit_eval.py`, `command_generator.py` reproduced their committed artefacts **byte-identically** (md5).

**Scope note.** This is Phase 1 (`scripts/` only). The `results/` tree is **unchanged** and its restructure is deferred to Phase 2, which must additionally resolve: bag-level nesting (`results/<strand>/<bag>/…`, on which `bag_config.resolve()` depends), the mixed content of the gitignored `results/runs/`, the spec-locked control artefact paths (`PID_PIPELINE_SPEC.md` §3/§6 — which would need their own additive amendments, not silent moves), and `superseded/` + `cache/` as first-class categories alongside `final/` + `diagnostics/`.

**Cross-references.** D020 (`bootstrap.py`), D025/O005 (`coco_to_yolo.py`), D028 (`resplit_dataset.py`), D030 (`phase_b_conf_sweep.py`, `median_conf_sweep.py`), O003/O009 + F007 (`blob_overlap_6799.py`), PHASE_A/B/C_SPEC (directory layouts, unedited — resolve via map (a)).

---

## D046 — Multi-bag generalisation of the pre-CP-2 stages; contamination-attribution limits
**Date:** 20 July 2026
**Status:** LOCKED

**Context.** The geometric pipeline from CP-2 onward was already bag-agnostic (D040). The stages *before* it were not: CP-0 and CP-1 carried hardcoded March paths and a March-only assertion, and the ROS1→ROS2 conversion that produces every bag's `.db3` was undocumented — it existed on disk for March with no script or instruction recording how it was made. Extending the evaluation to a second bag forced both gaps closed, and in doing so surfaced a contamination question that had been assumed rather than verified.

**(a) Pre-CP-2 stages are now bag-parametrised.**

| Stage | Change |
|---|---|
| **CP-(−1)** ROS1→ROS2 | **New** `scripts/geometric/convert_bag.py` — a thin, pinned wrapper around `rosbags-convert` (no ROS install needed), with a disk-headroom check and skip-if-converted. Closes the undocumented gap. |
| **CP-0** census | `contamination_census.py` takes `--bag`; scene selection uses `BAGS[bag]["scene_prefix"]`. A bag with no prefix, or with zero prefix-matched scenes, writes a **schema-valid empty census** and returns before building the descriptor bank, so CP-1 can consume it unconditionally. |
| **CP-1** manifest | `frame_manifest_build.py` takes `--bag`. The March-only `assert len(passes) == 11` becomes an *optional* guard driven by `BAGS[bag]["expected_passes"]`; `None` reports the detected count instead of asserting, which is the correct behaviour for any bag not yet characterised. |
| **CP-1 QA** | `extract_frames.py` writes overlays to a per-bag `qa_samples` path. March keeps its historical `superseded/dataset_split_samples/` location so committed artefacts still resolve; new bags use `results/geometric/{bag}/diagnostics/frame_samples/`. |

`bag_config.BAGS` gains `src_bag` / `ros2_dir` / `db3` (raw + converted bag paths, resolved from the repo root), `scene_prefix`, `expected_passes` and `qa_samples`; `resolve()` returns them alongside the existing bundle. All six bags are registered. Adding a bag remains a one-entry edit.

**Verification.** March reproduces **byte-identically** after the refactor: CP-1 re-run twice against the real bag (md5-identical manifest), and the CP-0 empty-scenes guard was exercised against the real March bag by forcing `scene_prefix=None` — it fired correctly, exited 0, and March's true census was then regenerated byte-identically.

**Pass-composition note — what `expected_passes` actually asserts.** It is a **reproduction guard** (it catches a silently changed bag file or a changed threshold constant), **not a count of physical traversals**. CP-1 defines a pass as a maximal run of smoothed `|v_y| > VY_INROW` (0.30 m/s) with `|Δy| > PASS_MIN_Y` (10 m), so two things inflate the count relative to physical reality: a brief dip under the velocity threshold mid-row splits one traversal in two (both halves independently clear the 10 m bar), and a partial traversal that still clears 10 m counts as one.

Both bags exhibit this, identically — the rule is not behaving differently on April:

| Bag | Locked count | Composition |
|---|---|---|
| march | 11 | includes one split pair (p8/p9, separated by **0.5 s**) |
| april | 12 | **10 full passes + 1 partial + 1 split pair** |

April's partial is **p4**: 266 frames / 18.0 s against a 52–66 s norm for full passes, followed by a **397 s idle gap** (the bag is 51% stationary, vs March's 29% — the robot stopped for ~6.6 min mid-session). April's split pair is **p7/p8**, divided by a **2-frame / 0.14 s** dip of smoothed `|v_y|` below 0.30 m/s at frames 16237–16238 while the robot was driving straight down corridor 0 at 0.43 m/s with `x` unchanged; merged, p7+p8 = 61.4 s / 824 frames, squarely inside the full-pass envelope.

Both counts are therefore locked **as-built, not idealised** — asserting a "corrected" 11 for April would encode a judgement rather than the pipeline's behaviour, and would fail on re-run. April's eligible set is nonetheless the stronger pooling unit of the two: 8,889 eligible frames vs March's 7,857, with better corridor balance (max corridor share 33.0% vs 42.0%).

**(b) Dataset month coverage — the precise claim.**
Of the 230 unique SemanticBLT scenes: **march 100, unattributed `color_image_*` 90, may 30, april 10, june 0, july 0, september 0.** An earlier working description of the dataset as "spanning March–June" is **unsupported** — there are no june-named scenes. COCO `date_captured` is the Roboflow *export* timestamp (identical across every image) and carries no acquisition information, so **the filename prefix is the only month signal available, and 39% of the dataset carries no month attribution at all.**

The defensible statement is therefore: *"the dataset contains no scenes whose filename identifies them as July or September"* — **not** "no labelled scenes were collected in those months". The difference is material, because 90 scenes could belong to any session.

**(c) Matching all 230 scenes against every bag was considered and REJECTED on evidence.**
The natural fix — stop trusting the prefix, match every scene against every bag, exclude on correlation — was tested before adoption via a controlled probe with known positives (month-prefixed scenes vs their own bag) and known negatives (vs a foreign bag): `scripts/geometric/one_time/unattributed_scene_probe.py` → `results/geometric/scene_attribution_probe.json`.

*Raw peak correlation is anti-informative.* Against the March bag, known-foreign april scenes score median **0.890** while true march members score median **0.779** — the negatives outrank the positives. The 128×128 grayscale thumbnail descriptor matches generic vineyard-row structure, not scene identity.

*A peak-margin refinement (max − p99 of the scene's own correlation profile) inverts between bags.*

| Peak margin | vs March bag | vs April bag |
|---|---|---|
| `march_*` (positive in March, negative in April) | **+0.081** | **+0.087** |
| `april_*` (negative in March, positive in April) | +0.024 | **+0.028** |
| `may_*` (negative in both) | +0.023 | +0.020 |
| unattributed | +0.038 | +0.059 |

March scenes score high in *both* bags and April scenes low in *both*: the statistic tracks the scene's own visual character, not membership. Against April the known negatives beat the known positives threefold. **No threshold on either metric separates members from non-members**, so all-230 matching would cause large, arbitrary exclusions — strictly worse than the status quo. CP-0 therefore **retains prefix-based scene selection**.

This does not impugn CP-0 itself: CP-0 never used correlation as a *membership* test. Membership comes from the prefix; matching only *localises* the scene within a bag known to contain it. That use is sound and is corroborated by April, where all 10 prefix-selected scenes located at corr 0.883–0.957 (all high-confidence). What is now known is that the `corr` field is a localisation confidence and must not be read as evidence of provenance.

**(d) Accepted limitation, and the gate that follows from it.**
The origin of the 90 unattributed scenes is **unknown and currently undeterminable**. If any belong to an evaluated bag, CP-0 under-excludes there and that bag's results are contaminated.

This risk is **directional, not uniform**. `GEOMETRY_PIPELINE_SPEC.md` §0 characterises the 90 as *canopy* (summer-foliage) imagery, which points at the summer sessions specifically. March and April are bare-vine/early-growth, so their exposure is low; June, July and September are the plausible sources and their exposure is high. Accordingly:

- **March and April proceed now.** The risk is low, pre-existing, and already baked into March's committed results — April neither introduces nor worsens it. It is carried as a stated limitation. May is likewise unblocked: it is prefix-attributed (30 scenes) and not canopy-season.
- **June, July and September are GATED.** None may be evaluated until the attribution of the 90 scenes is resolved by keypoint matching with geometric verification (ORB/SIFT + RANSAC inlier count), which discriminates a true re-observation from mere same-vineyard similarity in a way a global thumbnail descriptor cannot. Tracked as **O019**.

**(e) Stale assertion flagged.** `GEOMETRY_PIPELINE_SPEC.md` §0 asserts the 90 canopy scenes come from *"other sessions not on disk"*. That was written when only March was on disk and was never re-verified; six bags are on disk now, so the assertion is **stale and unsupported**. Under additive preservation the spec text is left as written — this entry is the correction of record, and O019 is what will settle it.

**(f) Methodological observation — the single-bag assumption was pervasive, and April is what exposed it.**
Extending to a second bag surfaced **five** defects that share one shape: code that was *implicitly* correct only while exactly one bag existed, and silently wrong the moment a second did. They were found across the whole pipeline, not in one module:

| Defect | Single-bag assumption | Failure mode with a second bag |
|---|---|---|
| `frame_manifest_build.py` — `assert len(passes) == 11` | march has 11 in-row passes | **Loud** — aborts CP-1 outright on any other bag |
| `extract_frames.py` — QA overlay path | one bag, one overlay directory | **Silent** — april's overlays would overwrite march's committed artefacts |
| `lidar_crosscheck.py` — anchor selector (D047) | corridor ≈ pass | **Silent** — samples row-exit frames; produced two spurious sign disagreements *per bag*, in march as well as april |
| `lidar_crosscheck.py` — `PC2_TOPIC_ID = 28` | topic ids are stable across bags | **Silent, and it did not fire** — april happens to share id 28, so the result was correct **by coincidence, not by construction** |
| `near_seed_sensitivity.py` — cache load path | a run either completes or is re-run from scratch | **Silent** — accepts a partial cache as complete; would have swept 2 of 9 model-streams and emitted a full-looking result |

**Only the first fails loudly.** The other four degrade quietly, and two of them (the anchor selector, the topic id) were *already latent in the committed march results* — the anchor bug had in fact already fired on march and had gone unnoticed because F017 reported only the aggregate for the pooled set. This is the substantive point: **the bugs were not introduced by adding april; april made them visible.**

**Bearing on the project's claims.** D040 described the pipeline from CP-2 onward as bag-agnostic. That claim was *true* but **unproven** — it had never been executed against a second bag, and "runs correctly on the data it was written for" is not evidence of generality. The pre-CP-2 stages were not bag-agnostic at all (see (a)). The honest statement is that bag-agnosticism became an *evidenced* property of the pipeline only after april, and only for the paths april actually exercised.

**Operational rule for may / june / july / september.** Treat **any code path not yet exercised by a second bag as suspect until proven otherwise.** Specifically: do not infer from "it worked on april" that a path is now generally correct — april exercised the in-row geometric and (pending) control paths, but a path is only evidenced for the *conditions it has actually met*. Sanity-check per-bag outputs against march/april rather than accepting them because they parse; prefer loud assertions over silent defaults; and resolve every external identifier (topic names, file paths, thresholds) by lookup rather than by literal. Where a constant is bag-measured rather than chosen, re-derive it per bag and say so.

**For the dissertation (limitations / lessons learned).** This belongs in the write-up as a methodological observation, not buried as engineering trivia: a single-site, single-session evaluation can conceal generalisation defects in the *evaluation code itself*, independently of any defect in the method under test. Four of the five defects here were invisible to every check that ran on march alone — including the reproduction checks — because reproducing a result does not test whether the code generalises. The multi-bag extension therefore functioned as a validation of the *instrument*, not only of the finding.

**Cross-references.** D040 (whole-bag pooling / multi-bag template — the bag-agnostic claim this qualifies), D041 (frame accounting), D047 (the anchor-selector fix and its binding sampling rule), GEOMETRY_PIPELINE_SPEC §0 and §2 (CP-0 contract), O019 (the gating verification), F017 (the finding whose committed numbers the anchor bug affected).

---

## D047 — LiDAR cross-check anchor selection: true mid-pass sampling (bug fix; regenerates a committed artefact)
**Date:** 23 July 2026
**Status:** LOCKED

**Bug.** `lidar_crosscheck.py` documented "mid-pass anchors per corridor" but selected `fs[len(fs)//3 : +2]` from the corridor's *concatenated* frame list. Corridors are traversed multiple times, so this index is unrelated to position within a pass; on both march and april it landed near a row exit.

**Why it matters.** Row-exit frames are the worst place to test sensor agreement: LiDAR return counts fall to ~1/3 of mid-row, and camera cross-model heading SD roughly doubles (0.50° → 1.07° past 90% of a pass). Two anchors per bag produced camera headings within one SD of zero, reading as spurious sign disagreements against a positive LiDAR.

**Fix.** Group each corridor's frames by `pass_id`, take the longest traversal, sample `PER_CORR` frames at its true midpoint. All anchors now fall at 49.8–50.1% of their pass. A second latent bug — `PC2_TOPIC_ID` hardcoded to 28 with a comment noting it needed a per-bag lookup — was fixed in the same pass by resolving the topic by name; april happened to share id 28, so its earlier result was correct by coincidence, not construction.

**Artefact regeneration.** `results/geometric/{march,april}/final/{bag}_evaluation/lidar_crosscheck.json` regenerated. March's committed values change (see the F017 amendment for the full before/after). The `superseded/` val/test cross-checks used hand-picked anchors and are untouched. **Nothing else in either bag's evaluation depends on this artefact** — F017 is its only consumer.

**Binding rule for remaining bags.** Any per-corridor sampling must respect pass structure; indexing a concatenated multi-pass list is not positional sampling. Applies to may/june/july/september.

**Cross-references.** F017 (the finding, amended), D040 (whole-bag pooling), D046 (multi-bag generalisation).

---

## D048 — O019 resolution: ORB+RANSAC scene→bag attribution, and its three-band decision rule
**Date:** 24 July 2026
**Status:** LOCKED

**Purpose.** Resolve O019 — whether any of the 90 unattributed `color_image_*` SemanticBLT scenes (39% of the dataset, no month prefix) contaminate an evaluated bag. The correlation probe was rejected (D046c) as anti-informative. This locks the keypoint-based replacement and its calibrated decision rule.

**Method.** Per (scene, bag): a coarse 128×128 thumbnail bank shortlists the top-30 candidate frames (recall only), then **ORB (nfeatures 3000) + Lowe-ratio match + RANSAC-homography inlier count** verifies identity; score = max inliers over the shortlist. `scripts/geometric/one_time/scene_attribution_orb.py` → `results/geometric/scene_attribution_keypoint.json`.

**Control calibration (march + april).** Known positives = prefix scenes vs own bag; negatives = prefix scenes vs foreign bag; unknowns = the 90. Observed:

| | march | april |
|---|---|---|
| genuine members | 59–1146 | 68–769 |
| cross-session same-place tail | ≤ 39 | ≤ 127 |
| 90 unknowns | ≤ 11 | ≤ 12 |

March separates cleanly; april does **not** — a cross-session same-place tail (fixed vineyard infrastructure re-observed across sessions, visually confirmed, all correctly-prefixed, not mislabels) reaches 127, overlapping the two weakest genuine members (68, 79). **Fine-verify (full-res ±30-frame neighbourhood) was tested and does NOT recover the weak members** (their true source frames are same-place-different-pass, outside the neighbourhood; one has a single dataset version so version-selection cannot help) — so it is not part of the rule. Validation trail: `scripts/geometric/one_time/scene_attribution_{tail_probe,fineverify}.py` + `results/geometric/april/diagnostics/attribution_tail/`.

**Locked decision — three-band rule** (calibrated to the clean floor gap 12→59 and the tail ceiling 127):
- **≤ 40 inliers → absent.** (Above every unknown, below every genuine member.)
- **≥ 200 inliers → present** → exclude from that bag's evaluation. (Above the cross-session tail with margin; at exact-frame re-observation levels.)
- **40–200 → manual visual review.** Weak same-place members and cross-session matches genuinely coexist here; no automatic statistic separates them.

**Result (O019 satisfied for march + april).** All 90 unknowns score ≤ 12 on both bags — confident-absent, zero in the review band. This converts D046d's *reasoned* directional-risk argument into a *measured* result for both evaluated bags: **the 90 unattributed scenes do not contaminate march or april.**

**May confirms (26 Jul 2026).** Running the wired CP-0 gate on `kg_may_06` scores all 90 unattributed scenes ≤ 11 inliers → 0 present / 0 needs_review / 90 absent. The 90 unattributed scenes are now measured-absent from **all three evaluated bags (March, April, May)** — the canopy-adjacency risk did not materialise on May. O019 remains armed for June/July/September.

**Accepted limitation.** ~1–2 of 10 genuine members per bag land in the 40–200 review band (multi-pass same-place siblings / heavy augmentation / single-version scenes); the method confidently confirms strong presence and clear absence but flags this band rather than auto-deciding. Accepted because (i) the march/april unknowns are unambiguous regardless, (ii) June/July/September are optional timeboxed work, and (iii) any future summer-bag unknown scoring ≥ 40 is **flagged, never silently mis-excluded** — a genuinely-present same-season scene would score ≥ 200 anyway.

**CP-0 integration.** The validated function is available as the per-bag gate: at each bag's CP-0, after prefix-scene location, the 90 unattributed scenes are scored against that bag; ≥ 200 → added to the exclusion set, 40–200 → `needs_review` (blocks that bag's evaluation until reviewed), ≤ 40 → absent. The gate is satisfied at each bag's natural CP-0 point (per the D046/O019 revised approach); june/july/september remain blocked until it resolves for them.

> **Amendment (27 July 2026, additive — two-stage rule promoted on june; the coarse three-band gate above is unchanged).** O019 deferred promoting the fine-verify stage into CP-0 "until validated"; the **june bag validated it** — and, unlike march/april/may (all 0/90 absent), june is where the risk actually materialised. The gate is now **two-stage**: (1) coarse ORB shortlist scores every scene (as above); (2) any scene in the 40–200 `needs_review` band is **fine-verified** — the full-resolution ±30-frame *non-strided* neighbourhood around its best coarse hit is decoded and the max inlier count re-classifies it. A true member whose exact frame the `COARSE=10` stride skipped jumps past 200 → present; a same-vineyard look-alike stays down. On june this recovered **30 of 36** needs_review scenes to present with no human input. Scenes **still** in the band are finalised by a committed per-bag **`d048_confirmed.json`** (`{scene: "present"|"absent"}`) recording the visual decision (labelled scene vs best bag frame); an unconfirmed residual **still BLOCKS CP-1 (fail-closed)**. June's 6 residuals were all confirmed present (the same hillside-building corridor; incl. `color_image_6799`, the F007 scene). **Result: june = 88 of the 90 unattributed scenes present** — a localized ~1,350-frame segment (bag frames ~11,390–12,740), 2 absent — confirming the O019 directional risk: the unattributed `color_image_*` scenes are a summer recording that *is* this bag. Thresholds (40/200), `COARSE` (10), and the block-on-unconfirmed rule are unchanged. Artefacts: `scene_attribution.fine_verify` / `apply_confirmations`; `results/geometric/june/d048_confirmed.json`.

**Implemented (25 Jul 2026).** The gate is now wired into the standard pipeline, so a single `prep.py --bag <name>` runs CP-0 (both parts) then CP-1 for every bag. (CP-0 and CP-1 were consolidated into one `prep.py` in the same cleanup batch — see the geometric README; earlier drafts of this note named the pre-merge scripts `contamination_census.py` / `frame_manifest_build.py`.)
- `scripts/geometric/scene_attribution.py` — production module (constants + ORB/RANSAC primitives + `gate()`), the byte-for-byte algorithm of the frozen validation harness `one_time/scene_attribution_orb.py` (kept separate so the committed calibration stays immutable).
- `prep.py` (CP-0) — builds the coarse bank for **every** bag (the no-prefix early-return is gone), runs the gate, folds `present` scenes into the exclusion windows, records `present`/`needs_review`/`absent` under a `d048_gate` block, and stamps `status: clear | needs_review`.
- `prep.py` (CP-1) — hard-stops if `status == needs_review`, listing the scenes to confirm. This is the enforcement of "blocks that bag's evaluation."
- **Regression check:** re-running the gate on march yields 0 present / 0 needs_review / 90 absent (max 11 inliers) — reproducing the validation and adding **zero** exclusion intervals, so march's (and april's, by the same all-absent result) manifest is byte-identical. Prefix-scene matching is unchanged, so prefix exclusions are untouched.

**Cross-references.** D046 (a–f, the multi-bag generalisation and the rejected correlation probe), O019 (the gate this resolves), GEOMETRY_PIPELINE_SPEC §2 (CP-0). Supersedes the correlation-based attribution of D046c.

## D049 — Deterministic CUDA/cuDNN preload guard (reproducibility)
**Date:** 27 July 2026
**Status:** LOCKED

**Purpose.** Make the GPU pipeline stages reproducible on a cold process. On the RTX 5050 Laptop (Blackwell sm_120) WSL2 devcontainer, `extract_detections.py` / `line_fit_infer.py` / `figures.py` / `one_time/near_seed_sensitivity.py` intermittently aborted at the first cuDNN call with `Invalid handle. Cannot load symbol cudnnGetVersion` — a **fresh** terminal crashed while a **warm** process (many prior GPU calls in the same shell) did not, so the fault was invisible until a clean reproduction hit it.

**Root cause.** The pip-wheel CUDA libraries (`nvidia-*-cu12`, incl. `libcudnn.so.9`) install under `site-packages/nvidia/*/lib`, which is **not** on the dynamic-loader path (`ctypes.CDLL('libcudnn.so.9')` fails by name). PyTorch preloads them itself at `import torch`; on this Blackwell + cuDNN 9 stack that cold-start preload is **intermittently missed**, and the first cuDNN symbol lookup then aborts the process. Isolated by construction: adding the wheel lib dirs to `LD_LIBRARY_PATH` makes `libcudnn.so.9` load by name and `cudnnGetVersion()` return `91900`, and the crash does not recur.

**Locked decision.** Do not rely on torch's implicit preload. `scripts/geometric/cuda_preload.py` `CDLL(RTLD_GLOBAL)`-loads **the cuDNN wheel family only** (`nvidia/cudnn/lib`: `libcudnn.so.9` + its `libcudnn_*.so.9` engines) and is imported **before `import torch`** in each GPU stage. Scope is deliberately cuDNN-only: torch reliably preloads the other CUDA wheels (cublas, cudart, …) itself — only cuDNN is flaky — and force-loading the whole `nvidia/` tree also pulls in `libnvblas.so`, a BLAS-interception library that, once in the `RTLD_GLOBAL` namespace, hijacks CPU BLAS and **segfaults** without an `nvblas.conf` (observed on `line_fit_infer` arm B while validating the first, over-broad version of this guard). `libcudnn.so.9`'s declared deps are all system libs, so the cuDNN family loads standalone. The fix is baked into the code, so any reproducer gets it with **no environment setup**, and it is a silent no-op on a CPU-only install. Belt-and-suspenders: `.devcontainer/post-create.sh` also prepends the wheel lib dirs to `LD_LIBRARY_PATH` in `~/.bashrc` — that only makes libs *findable by name* (nothing force-loads `libnvblas`), covering interactive shells and ad-hoc `python3 -c "import torch"` (the surface the in-script guard does not).

**Scope / invariance.** Load-only change: it makes symbol resolution deterministic and touches no numerics — seeds, weights, thresholds, and every committed artefact are unaffected (rule 7 intact). Verified: importing the guard before torch leaves `torch.backends.cudnn.version()` == 91900 and a reference GPU predict unchanged.

**Cross-references.** CLAUDE.md "Environment quirks" (Blackwell sm_120 / PyTorch 2.11+cu128); `requirements.txt` (the cu128 wheel install line); D048 (the june bag whose run surfaced this).

> **Amendment (27 July 2026, additive).** Two follow-ups from the june run, both reproducibility-neutral (no numerics change; rule 7 intact):
> - **Guard completeness.** The guard is imported before torch in **every** GPU entry point, not just the first four: added to `analyze.py` (Stage-C `single_row_analysis`) and `projection_calibration.py` (CP-2 `_validate`), which were found unguarded — the same intermittent-crash exposure. Full guarded set: `extract_detections`, `line_fit_infer`, `figures`, `one_time/near_seed_sensitivity`, `analyze`, `projection_calibration`. Secondary diagnostics/perception scripts rely on the `post-create.sh` `LD_LIBRARY_PATH` belt-and-suspenders rather than an in-script guard.
> - **cuDNN `half`→`quantize` deprecation.** All active `model.predict(...)`/`.val(...)` calls were switched from the deprecated `half=True` to `quantize=16` (23 call sites across the geometric + perception strands + 4 doc/print mentions; `superseded/` left as-is). ultralytics 8.4.90 forwards `half=True`→`quantize=16`→predictor `fp16 = (quantize==16)`, with `half` popped and no other consumer — so the two are the *identical* inference path; **verified bit-identical** (boxes **and** masks) on the Phase-B and Phase-C weights. The switch only stops the per-predict `LOGGER.warning("'half' is deprecated …")` from flooding a reproducer's console (and risking them killing a healthy run). Committed CSVs reproduce exactly.

## D050 — July bag excluded from evaluation: stop-start recording defeats the contiguous-pass detector
**Date:** 28 July 2026
**Status:** LOCKED

**Decision.** `kg_july_13` is **not evaluated**. The evaluated set remains march, april, may, june (two bare-vine, two canopy).

**What was observed.** CP-1 on july yields **1 in-row pass, 40 eligible frames (0.2% of a 17,422-frame bag), 1 corridor** — against 7,308–8,889 eligible frames and 4–6 corridors on the four evaluated bags. The bag is neither empty nor mis-converted: it covers the **same physical block** (x span −14.6→0.6 m, y −44.7→7.2 m, both matching june) and records **452 m of along-row motion** against june's 479 m.

**Mechanism.** CP-1 defines an in-row pass as a **contiguous** run of `|v_y| > VY_INROW` (0.30 m/s) spanning `|Δy| > PASS_MIN_Y` (10 m). July was driven **stop-start**: 350 stationary blocks (longest 167 s; 64% of the 1,366 s session; the first five minutes are 6.2% moving) against june's 59. The traverses are therefore shredded into **289 sub-threshold fragments of median Δy 1.3 m**, only one of which spans 10 m. The detector sees one pass where the robot drove roughly ten.

**Rejected — a global gap tolerance.** Merging runs separated by short pauses would recover july, but it changes the detector for **every** bag: it breaks march's `expected_passes = 11` and april's `= 12` CP-1 assertions, and would silently alter may's and june's committed manifests and every number derived from them. Not acceptable under working rule 1.

**Considered and deferred — a july-only opt-in gap tolerance.** An optional, default-off `pass_gap_frames` field (july = 45 frames ≈ 3.5 s) was simulated: it recovers **10 passes, 4,416 eligible frames, 7 corridors**, with **97% of recovered frames at `|v_y| ≥ 0.30`** — genuine in-row driving, not padding. It is technically sound and provably non-breaking for the other bags. It is **not adopted**, because it introduces a **bag-specific eligibility criterion** that weakens the cross-bag comparability the multi-bag design exists to provide, and because four bags already deliver a complete, consistent result. The simulation is recorded here so the option can be revisited if a later bag shows the same pattern.

**Limitation to report (A2).** The CP-1 pass detector **assumes uninterrupted traverses and is not robust to stop-start driving**. This is a property of the frame-selection stage, not of any perception arm, and affects no committed result — but it bounds which recordings the pipeline can evaluate, and it is why july is absent. F029's canopy characterisation therefore stays at **n = 2 canopy bags**.

**Artefacts (evidence, committed).** `results/geometric/july/{contamination_census_exclusions,dataset_manifest,manifest_summary}.json`. The D048 gate ran **clean** on july (0 present / 0 needs_review / 90 absent), so contamination is not a factor. July's detection cache (140 detections over 40 frames) is gitignored and carries no result.

**Cross-references.** D040/D041 (CP-1 whole-bag eligibility), D046 (multi-bag generalisation), F029 (canopy characterisation, n = 2), `bag_config.expected_passes`.

> **Amendment (29 July 2026, additive — the decision is unchanged; the evidence is upgraded).** D050 asserted that "the recording itself is sound" on the strength of july's matching spatial extent and along-row distance. A supervisor question — whether the stop-start pattern was the operator pausing whenever GNSS quality degraded — prompted direct measurement, which **refutes that mechanism while evidencing the original assertion far more strongly**.
>
> **The GNSS never degraded.** `/health/gps/error_std` on july has median 0.0072 m, p90 0.010 m and **max 0.010 m**, with 0.0% of the session above 10 cm — statistically indistinguishable from june (median 0.009 m, max 0.01 m) and better than march (median 0.259 m). Error during the pauses (0.0072 m) and during motion (0.0080 m) is the same, so the pauses were **not** GNSS-motivated: the hypothesis is refuted with data, and whatever prompted the 350 stops, it was not signal quality.
>
> **The fault is a degraded localisation publisher, not a sensor.** `/robot_pose` publishes at 12.75 Hz but carries only **0.56 Hz of new content** — 95.6% of consecutive messages byte-identical (june: 78.8% at 3.00 Hz). Where it does update, 2.0% of updates imply speeds above 10 m/s, and it diverges from the GNSS-derived `/odometry/gps` by up to **40.6 m**. On june those two topics agree to 0.010 m median / 1.31 m max, so they are the same trajectory — july's `/robot_pose` is a stale, laggy derivative of a sound GNSS.
>
> **Recovery is possible, but needs two bag-specific deviations.** `/odometry/gps` carries july's position at **4.70 Hz with 0.038 m steps**, which would restore spatial resolution. Without it, july's pose resolves position only every **1.31 m** against june's 0.236 m — *coarser than june's own GT-1 decorrelation distance of 1.27 m* — so july cannot measure the lag at which its metric decorrelates, and the block lengths that set every CI width could not be derived at comparable fidelity (its 1.5 m subsample would also fall to 172 frames against june's 255). Recovering july would therefore require **both** the gap-tolerant pass detector **and** a bag-specific pose source — a larger departure than the single deviation D050 weighed, with the comparability objection applying to both.
>
> **Net.** The decision stands, on a better basis. July is excluded not because its data is doubtful — the recording is now positively evidenced as sound, with a millimetre-accurate fix throughout — but because the platform's **localisation publisher** degraded during that session, and recovering the bag would require two bag-specific rules the cross-bag design cannot absorb. Diagnostics read directly from `kg_july_13.bag` (`/health/gps/error_std`, `/robot_pose`, `/odometry/gps`); no committed artefact is affected.

## D051 — September bag excluded from evaluation: RTK-GNSS fix lost, driven-path reference invalid
**Date:** 28 July 2026
**Status:** LOCKED

**Decision.** `kg_september_09` is **not evaluated**. The evaluated set is final at four bags: march, april (bare-vine), may, june (canopy).

**What was observed.** CP-1 yields **250 eligible frames of 28,202 (0.9%)** across 12 nominal passes — but each lasts only **1–3 seconds** (15–33 frames), against ~60 s and ~900 frames on the evaluated bags. The pass count (12) and corridor count (5) look healthy, which is what makes this failure mode deceptive.

**Root cause — the RTK-GNSS fix was lost for a quarter of the session.** `/health/gps/error_std` has median 0.015 m but **p90 15.20 m and max 41.96 m**, with **25.8% of the recording above 1 m** reported error. March (median 0.259, max 0.82) and june (median 0.009, max 0.01) never exceed 1 m.

**Consequence — the pose degenerates into a step function.** Only **496 distinct pose values across 28,202 frames (1.8%)**; **98.2% of consecutive frames carry an identical pose**; effective update rate **0.26 Hz** against june's 3.00 Hz. When it does move it **snaps** — median step 0.880 m, p99 11.90 m, **max 18.84 m** — implying velocities to 302 m/s. The pose stream also starts **33.3 s after** the camera, and the trajectory wanders to x = +19.4 m, outside the −14.6 → +0.6 m block every other bag occupies.

**Why the 12 "passes" are artefacts.** CP-1 accepts a pass when a run of `|v_y| > 0.30` spans `|Δy| > 10 m`. A single 18 m snap satisfies that in **one frame**, manufacturing a 1–3 second "pass". The detector is reporting jumps, not traverses.

**Why this is fatal rather than fixable.** The GT-1/GT-2 *values* do not consume `/robot_pose`: `line_fit_infer.estimate()` derives offset and heading from the base points and the IPM projection alone, in `base_link`, so the driven path enters as the **frame of reference**, not as a numeric input. What the pose determines is (i) **which frames are evaluated at all** — eligibility, corridor assignment, stationary/headland classification, pass segmentation; (ii) the **Δs = 1.5 m subsample** every per-model CI is drawn from; and (iii) the **decorrelation distances and moving-block lengths** that set the width of every reported interval. On september all three collapse: 250 eligible frames of 28,202, twelve "passes" that are pose jumps rather than traverses, and a position stream far too coarse and erratic to support a spatial autocorrelation estimate. Separately, the interpretive premise fails too — reading GT-1 as *agreement with an RTK-guided path* presumes the platform's own localisation held near the **3.8 cm RTK floor** (Polvara 2024 §5.3; the shaded band in `cmp_forest`), and september's p90 of 15.2 m violates that by orders of magnitude. So the bag cannot yield a defensible interval on any metric, nor a defensible reading of one, even though the per-frame values would be arithmetically computable. No change to frame selection, thresholds or pass detection recovers that.

**Alternative reference sources considered and rejected.** The bag carries `/odometry/base_raw`, `/odometry/gps`, `/front/zed_node/odom` and `/tf` at full rate. Substituting one for `/robot_pose` on september alone would (i) change the *definition* of the reference for a single bag, destroying the cross-bag comparability the multi-bag design exists to provide, and (ii) replace an RTK-referenced path with dead-reckoned odometry whose drift over a 32-minute session is unbounded and unquantified here. Not adopted.

**Future work — a pose-quality precondition.** A CP-0 gate on `/health/gps/error_std` (e.g. refuse a bag exceeding a small percentage above ~0.5 m) would have rejected september in seconds, before a 75 GB conversion and a full CP-0 census. Registered, not built.

**Artefacts (evidence, committed).** `results/geometric/september/{contamination_census_exclusions,dataset_manifest,manifest_summary}.json`. The D048 gate ran **clean** (0 present / 0 needs_review / 90 absent). No frames extracted, no detections computed.

**Cross-references.** D050 (july, excluded for a different and less severe reason), F013 (RTK floor as the GT-1 yardstick), D040/D041 (CP-1 eligibility), F029 (canopy characterisation — n = 2 at the time of this decision; extended to n = 4 by D052).

## D052 — July/August 2023 bags adopted for the geometric strand (control strand not run)
**Date:** 29 July 2026
**Status:** LOCKED

**Decision.** Two 2023 sessions are adopted as the fifth and sixth bags of the **geometric** evaluation: **`july2023`** (2023-07-25, one session recorded as two consecutive files, merged at conversion) and **`august2023`** (2023-08-01, single file). The **control strand (F026/F027/F028) is not run** on them. The evaluated set becomes **six bags geometric / four bags control** — bare-vine march + april, canopy may + june + july2023 + august2023.

> **⚠️ Superseded in part (31 July 2026) — august2023 is withdrawn by D054.** That session's camera recorded no imagery: all 8,916 frames are one byte-identical blank white JPEG, discovered when Stage B1 returned zero detections. The adopted set is therefore **`july2023` only**, and the evaluated set is **five bags geometric / four control** — bare-vine march + april, canopy may + june + july2023.
>
> Everything below on **provenance, calibration validation, session selection and merge policy stands unchanged** — it was established from pose, `tf` and topic metadata, none of which the camera fault touches. Read every "both 2023 bags" claim as applying to **july2023 alone** for anything downstream of the camera.

**Provenance (supervisor-confirmed, 29 July 2026).** Same robot, same vineyard, and the **same autonomous topological-navigation configuration** as the 2022 bags — one year later. These are not a second dataset, and not the Riseholme platform (a genuinely different robot, unrelated to these files). What differs between 2022 and 2023 is **which sensors were logged in a given session**, not the platform. Corroborated independently: april 2022 and both 2023 bags publish the *same* URDF frame names (`pipe1_2`, `2_zed2_*`, `sensor_box_1`, `bat0/1`, `corner0-3`, `top0-3`) and identical camera intrinsics.

**O020 framing holds unchanged.** Because the navigation configuration is identical, the driven-path reference means the same thing on these bags as on march/april/may/june: GT-1/GT-2 measure agreement between the vision-estimated centreline and the platform's **autonomous** GPS/topological driven path (Polvara 2024 §3.3.3). No amendment to O020 or D014 is required; the absence of `/closest_node` / `/current_node` from these recordings is a logging difference, not a change of navigation mode.

**Projection calibration validated against onboard data — the hardcode is confirmed correct.** Composing `base_link → pipe1_2 → 2_zed2_camera_center → 2_zed2_left_camera_frame` from `/tf_static` yields **(+0.3450, +0.0600, +0.7630) m** on **april 2022** and on **both 2023 bags**, matching `projection_calibration.T_BASE_CAM` (Polvara 2024 Table 3) to four decimals; intrinsics match exactly (fx = fy = 1057.0, cx = 952.2, cy = 553.6) and pitch is 1.95° vs 2.00°. This is the **first validation of Table 3 against the robot's own description**. March, may and june publish only 2–3 static transforms and cannot supply it, so Table 3 remains their only source — now corroborated via april.
*Method note (a real trap).* The comparison must compose to the **left lens**, the frame `camera_info` actually reports (`*_left_camera_optical_frame`). Every ZED2 in the tree carries `camera_center → left_camera_frame` at (0, **+0.060**, 0) and `→ right_camera_frame` at (0, −0.060, 0) — the 120 mm stereo baseline. Comparing Table 3 against `camera_center` instead produces a spurious 6 cm lateral discrepancy. **Prefer the bag's own `tf` where it exists, falling back to Table 3 only where it does not** — mandatory for any future platform.

**Sensor-logging differences, and what they cost.** These sessions did not log `/imu/data`, `/imu/mag`, `/imu/rpy`, the side camera, `/motor_controller_data`, or the topological-nav node topics. Only `/imu/data` has any consequence: `scripts/geometric/` contains **zero** IMU references, so the whole geometric path (CP-0 census + D048 gate, CP-1 manifest, frame extraction, detection cache, 9-model line-fit inference, the five in-row analyses, non-in-row, and the F022/F023 mitigation analysis) runs unmodified. The **control strand** consumes it in three places — `state_gate_native.py`, `command_generator.py` (F027-A's locked `omega_max` is the IMU-gyro in-row p99) and `gain_kfold.py` — and is therefore **declared not-run** rather than run on a substitute gyro (`/front/zed_node/imu/data` and `/os_cloud_node/imu` are both present), which would silently change the locked gains and break comparability with the four control bags.

**Session selection, and why the multi-file days are merged only within a day.** Seven 2023 files were supplied, forming **four session-days**: 07-11 (1 file, 10.8 min), 07-20 (3 files, 20.0 min), 07-25 (2 files, 20.6 min), 08-01 (1 file, 14.8 min). Files within a day are consecutive recorder restarts, not `--split` fragments (gaps 18–99 s). Every boundary was checked and falls at a **stationary point** — files end at |v| ≈ 0.01 m/s with 0.18–1.90 m of movement across the gap — so **no traverse is cut mid-row** and merging within a day restores exactly one session. `rosbags-convert` accepts several `--src` and writes them into one destination in chronological order, so the merge happens at conversion and the pipeline still reads a single `.db3`. **Merging across days is refused**: the days are 5–9 apart and share 3–5 corridors, so a composite would fabricate a recording that never happened, present repeat visits to the same corridor as repeat passes within one session, and pool between-day variation into a per-bag block bootstrap that assumes a within-session spatial series — a mis-scaling nothing would flag. It would also yield no more bags than simply selecting the best day.
07-25 and 08-01 were selected as the strongest: 07-25 has GPS median **0.008 m** (matching june's 0.009, best of the four days), 20.6 min, 14 passes, 6 corridors, 524 m along-row; 08-01 has the highest pose rate (4.14 Hz), the highest moving fraction (85.2%) and needs no merge. **07-11 and 07-20 are held in reserve** as clean independent sessions, addable later if more canopy breadth is wanted.

**Session characteristics vs the excluded bags.** Both adopted bags are healthier than july-2022 and september-2022 on every quality axis: GPS max 0.02 m / 0.12 m (september p90 15.2 m), pose content rate 3.95 / 4.14 Hz (july-2022 0.56 Hz, june 3.00 Hz), **zero** implausible pose velocities, 70–85% of each session in motion (july-2022 33%). Both traverse the same block as june (x ∈ [−14.6, +0.6], y ∈ [−44.7, +7.8]). At ~10 Hz camera and 15–21 min they will yield fewer eligible frames than the 2022 bags, so expect correspondingly wider per-bag CIs.

**Why this is not a D050/D051-style exception.** The geometric pipeline needs **no bag-specific code path, threshold or calibration** for these bags — the reason july-2022 and september were excluded. The only changes are **additive**: an optional `sources=` list in `bag_config` (multi-file conversion), and a `CONTROL_EXEMPT` set in `check_bag_complete.py` declaring the control artefacts not expected. Both existing single-source behaviour and all four committed bags' completion checks are unchanged and were verified so. The strand asymmetry is itself precedented: F018's configuration sweep is arm-C-only by design (D026).

**Reporting scope for july2023 (29 July 2026, from the Stage-C run — supersedes this entry's earlier expectations).** CP-1 was healthy: 6,595 eligible frames (**54%**, the highest fraction of any bag), 15 passes, 7 corridors, and **zero contaminated frames** — the D048 gate found 0 of the 90 unattributed scenes present, as expected for 2023 footage against a 2022-derived training set, making this the only bag with no exclusions at all. The **analysis** then revealed a domain-shift effect that narrows what the bag can support.

**Withheld — not reported in any form.** The **B–C** contrast and the marginal **A–C GT-2** contrast are **not reported for this bag, as neither a null nor a positive result.** Paired-frame density is 0.752 m mean spacing against 0.070–0.113 m on the 2022 bags, so the block-length estimator hits its resolution floor (L = 2 / 4 against 9–23 / 17–37) and returns artificially narrow intervals. Under a block-length sweep the B–C GT-2 exclusion disappears by L = 23 and A–C GT-2 by L = 9, and both have inconsistent per-seed signs (`-,+,-` and `-,+,+`) where a real effect would be consistent. These are measurement artefacts of sparsity, not findings: the underlying quantity is not measured reliably enough on this bag to state any result in either direction. **july2023 therefore contributes nothing to F013's evidence base**, which remains the four in-distribution bags. The condition is now detected automatically (D053).

**Reported — robust.** The **A–B and A–C GT-1** contrasts are sign-consistent across all three seeds *and* stable at every block length tested (L = 2, 9, 23, 37), at **−19.5 mm and −24.1 mm** — 51–63% of the 3.8 cm RTK floor, an order of magnitude above may's sub-floor 4–5 mm. Coverage separates the arms in the same direction: **38.3% (A) / 34.8% (B) / 25.9% (C)** against june's 67.8 / 64.9 / 61.5, with abstention rising to 43–47% (june 21–26%) and arm-C base points falling to 7.4. This is a **domain-generalisation observation** — the models were trained on 2022 imagery and evaluated on 2023, and the three arms degrade unequally, the U-Net arm retaining the most coverage. It is **explicitly scoped as separate from the class-structure (B–C) research question** and must not be read as bearing on it.

**Correction to this entry's earlier expectation.** july2023 does **not** extend F029's canopy characterisation from n = 2 to n = 4. Its coverage is under half june's and the cause is domain shift, not canopy, so pooling it with may/june would confound the two. **F029 remains at n = 2 canopy bags.** Two further symptoms confirm the domain-shift reading: no config cell reaches the 70% viability floor (max 25.9%, against may's 63.1%), and the LiDAR cross-check disagrees far more than on any 2022 bag (camera +0.85° vs LiDAR +2.99°, Δ −2.15°; june Δ +0.27°).

**Note on O007.** The perception models were trained on 2022 data; these bags are 2023 — a different season *and* year from the training distribution, and therefore a closer approximation to the out-of-distribution evaluation O007 registers than any existing bag. The coverage collapse above is that distribution shift measured, so this is now an evidenced observation rather than a hoped-for one. Not a substitute for labelled OOD data, but worth reporting as partial mitigation.

> **Correction (9 August 2026, additive — the text above is unchanged and the adoption decision stands; three numeric/robustness claims in the "Reported — robust" paragraph are corrected).** Discovered by a prose-vs-artefact reconciliation pass over every D and F entry. **Root cause: D052 cites no artefact at all** — it is the only entry in either document that states measured values with zero `.json`/`.csv` citations, so nothing ever cross-checked it.
>
> **(a) The GT-1 contrasts are wrong.** `results/geometric/july2023/final/july2023_evaluation/paired_crossarm.json` gives **A–B −20.2 mm (−53.1% of the 3.8 cm RTK floor)** and **A–C −19.0 mm (−50.1%)**. The entry's −19.5 mm / −24.1 mm appear nowhere in that artefact, nor in any other; its "51–63% of the floor" is internally consistent with those two figures and inconsistent with the measurement. **Use −20.2 mm and −19.0 mm.** Note the corrected pair is nearly *equal* (−20.2 vs −19.0), where the superseded pair implied A–C was ~25% the larger effect — the corrected numbers weaken any claim that the two contrasts are separable.
>
> **(b) "Robust" is withdrawn.** The same artefact records `ci_reliability: GT1 samples_per_decorr 1.09, reliable: false` (GT2 1.91, also false) against D053's 3.0 minimum — **D053's guard refuses this bag, so no interval estimate on july2023 is admissible.** What survives is sign-consistency, which is computed independently of the CI machinery: `A–B GT1 − − −` and `A–C GT1 − − −` are both `consistent: true` across the three seeds. **july2023's lateral contrasts are therefore a DIRECTIONAL OBSERVATION with no interval estimate** — exactly the status Riseholme's contrasts hold under D059, and for the same reason. B–C GT-1 is sign-inconsistent (`+ − −`) and remains withheld.
>
> **(c) The block-length stability claim is withdrawn as unreproducible.** "Stable at every block length tested (L = 2, 9, 23, 37)" has no persisted computation: the artefact records only `L_GT1 = 2` (strict-threshold variant 3) and `L_GT2 = 4`, and no sweep output exists anywhere under `results/`. The related claim in the same entry — that "the B–C GT-2 exclusion disappears by L = 23 and A–C GT-2 by L = 9" — is unpersisted for the same reason. **Both are withdrawn.** This is a departure from the project's own convention: F025's near-seed sweeps are persisted as `near_seed_sensitivity.json` for march, april and may. Either regenerate the block-length sweep to an artefact and restore the claim, or leave it withdrawn; it must not stand unsupported.
>
> **Unaffected.** F030's coverage ordering (A 38.3 / B 34.8 / C 25.9 %) is a **classification** result — it needs no interval and no block length, and none of (a)–(c) touches it. The domain-generalisation framing, the arm-C base-point collapse, and the scoping of july2023 as separate from the B–C class-structure question all stand.

**Cross-references.** O020/D014 (autonomous driven-path framing), D045 (paths), D048 (unattributed-scene gate — runs on these bags as on every other), D050/D051 (the excluded 2022 bags, for contrast), D026 (arm-C-only precedent for strand asymmetry), F029 (canopy characterisation — **unchanged at n = 2**, see the correction above), D053 (the CI-reliability guard this bag prompted), O007 (OOD evaluation).

## D053 — CI reliability guard: detect when paired-frame density cannot support the block-length estimator
**Date:** 29 July 2026
**Status:** LOCKED

**Purpose.** The moving-block bootstrap sets its block length as `L = max(2, round(2 × decorr / mean_spacing))`, where `decorr` is the first distance-lag at which the paired-difference autocorrelation falls below 0.1. When paired samples are sparse relative to the decorrelation length, the binned autocorrelation cannot locate its own crossing — the first bin already lies at or beyond it — so `decorr` is returned as a **lower bound**, L is under-estimated, and **the resulting CIs are anti-conservative (too narrow)**. Nothing in the pipeline detected this: it produced a confidently narrow interval with no indication the estimate sat at its resolution floor.

**Observed.** On july2023 (D052) the paired mean spacing is **0.752 m** against 0.070–0.113 m on the four 2022 bags, giving **1.09 and 1.91** samples per decorrelation length and L = 2 / 4 against 9–23 / 17–37. Two contrasts crossed into apparent significance purely as a result, and both reverted under a larger block length.

**Why the existing protection was not enough.** The strand-wide L is already the **maximum** across the three arm-pairs — a conservative reduction that protects a *single* sparse pair, and which has been silently doing its job: april's B–C pair scores 0.98, is itself resolution-limited, and is carried by A–B's 11. It failed on july2023 because **all three pairs were sparse simultaneously**, leaving the maximum nothing better to select. This guard closes exactly that gap; it is a narrow extension of an existing protection, not a new mechanism.

**Locked rule.** `block_lengths.py` computes `samples_per_decorr = decorr_m / mean_spacing_m` for every (pair, metric) and reports it **unconditionally** — so the margin is visible on healthy bags too, not only when it trips. A metric is `resolution_limited` when the pair that *sets* its strand-wide L scores below `MIN_SAMPLES_PER_DECORR = 3.0`; the per-metric verdict is recorded in a `ci_reliability` block. `analyze.py` raises a prominent banner at each of the three derivation sites, and the block propagates into `line_fit_report.json`, `paired_crossarm.json` and `config_analysis.json` so the condition is machine-checkable, not merely printed. A `decorr` of `None` (fallback) is **not** flagged: it means the crossing lies beyond the 3 m search window — longer than the sampling can bound, the opposite failure.

**Threshold calibration.** The pair that sets each strand-wide L scores **4.61–18.69** across the four committed bags and **1.09 / 1.91** on july2023, so 3.0 separates with margin on both sides. Equivalent to `L ≥ 6`, since `L = round(2 × ratio)`.

**Scope — this guard changes no result.** It does not delete CIs, substitute a different L, or alter any committed number. All four committed bags re-derive **byte-identical** block lengths and report `ci_reliable: true` on both metrics, confirming their intervals were sound all along. It is **not** a rescue for july2023: that bag's reporting scope (D052) follows from the sparsity itself, which is a property of the data and unchanged by how well it is detected.

**Retrospective backfill.** The new fields were added to the four committed bags' `line_fit_report.json`, `paired_crossarm.json` and `config_analysis.json` (12 files) by `one_time/backfill_ci_reliability.py` — a **pure recalculation**, no pipeline stage re-run. The three reports per bag carry an identical `block_lengths` block, so all three are updated together; `command_evaluation/command_smoothness.json` is deliberately untouched, as it borrows the geometric L rather than deriving a per-pair structure and carries its own separate caveat. Verified four ways: `git` shows +29/−6 per file with every removal a pure comma reflow (`"L": 9` → `"L": 9,`) and **zero genuine deletions**; stripping the added keys reproduces each file **identically to HEAD**; and a live re-derivation matches the written files exactly. The script derives the new fields from that live re-derivation rather than from the stored numbers, because `mean_spacing_m` is stored rounded to 4 dp and the rounded quotient differs from the estimator's by up to 0.01 — enough that the files would not have matched a future re-run. It is idempotent (strips before re-adding).

**Cross-references.** D052 (july2023, the bag that exposed it), F013 (the paired contrasts these CIs support), D040 (whole-bag pooling), Analysis H / `block_lengths.py` (the estimator), `one_time/backfill_ci_reliability.py`.

## D054 — august2023 excluded: the session's camera recorded no imagery
**Date:** 31 July 2026
**Status:** LOCKED

**Decision.** `august2023` (2023-08-01) is **excluded from the evaluation**, reversing its adoption in D052. The evaluated set is **five bags geometric / four control** — bare-vine march + april, canopy may + june + july2023.

**Observed.** Stage B1 returned **0 detections** over 5,961 eligible frames × 3 seeds — against 145,726 on july2023 and 252,741 on june. Every one of the bag's **8,916** frames on `/front/zed_node/rgb/image_rect_color/compressed` is the *same* JPEG: one distinct payload (md5 `5fa51a606949…`) in a constant 33,351-byte message, decoding to 1920×1080 with `mean 255.0, std 0.0` and a single distinct pixel value across all ~6.2 M channel samples. Verified **exhaustively over all 8,916 frames**, not sampled.

**Not an artefact of our tooling.** The ROS1 source bag reads identically (one payload, mean 255.0, std 0.0), so the blank frames were recorded as such — `rosbags-convert` and every pipeline stage are exonerated. The fault covers the whole camera, not just the RGB channel: `/depth_republish/compressedDepth` is likewise one constant 18,285-byte payload across all 8,918 messages. No alternative RGB topic exists in the recording. A saturated *camera* would still vary frame to frame and could not compress to a byte-identical payload; a single repeated payload indicates a fixed placeholder buffer, i.e. the sensor was not imaging at all.

**Why every prior check passed.** D052 selected this session on pose rate (4.14 Hz, highest of the four 2023 days), moving fraction (85.2%), GPS quality and single-file simplicity — all measured on **non-camera** topics, none capable of detecting a dead camera. The pipeline agreed: CP-0 clean (0 of 90 unattributed scenes present), CP-1 healthy with 13 passes, 5 corridors and **5,961 eligible frames (66.9% — the highest eligible fraction of any bag)**. The robot drove the rows correctly and logged it correctly; only the imagery is absent. The 15.6 GB the bag contains is LiDAR and pose.

**Why this is an exclusion, not a narrowed reporting scope.** july2023 (D052) was narrowed because some contrasts were measured too imprecisely to report. august2023 admits no such treatment: with no imagery there is no measurement to qualify — nothing to report, withhold, or call null. It joins D050 (july-2022) and D051 (september-2022) as a data-integrity exclusion.

**Consequences.** F029's canopy characterisation **remains at n = 2** (may, june) — as D052 already concluded for july2023 on separate grounds. O007's out-of-distribution observation rests on july2023 alone.

**Artefacts discarded (31 July 2026).** The ROS2 conversion (15 GB), the 5,961 extracted blank frames (47 MB) and `results/geometric/august2023/` were deleted; the ROS1 source bag is retained by Edosa. The bag's `bag_config` entry is left in place: it is inert once the bag is not run, and removing it would erase the record of what was attempted.

**Cross-references.** D052 (adoption, now amended), D050/D051 (the other integrity exclusions), D048 (CP-0 gate — passed, and correctly so), F029 (canopy characterisation, n = 2), O007.

## D055 — Riseholme adopted as a supplementary generalisation strand
**Date:** 7 August 2026
**Status:** LOCKED

**Decision.** The Riseholme datasets are adopted as a **supplementary generalisation strand**, separate from and subordinate to the five-bag Ktima evaluation. Riseholme is **not** a second evaluation of the research question and produces no result comparable to F013's Ktima contrasts.

**What it is.** Riseholme (University of Lincoln) is a different site, a different camera, and a different viewing direction. Any Ktima↔Riseholme difference therefore confounds **four** factors — site, season, **camera hardware** (Stereolabs ZED2 forward-facing vs Intel RealSense D435I rear-facing), and **viewing direction**. Stated positively this makes it a *hardware-and-viewpoint* generalisation test as well as a site one, which is a stronger claim than site transfer — but only if the confound is stated rather than glossed. It must be stated wherever Riseholme results appear.

**Data and roles.** `Tue-02-Sep` (2025-09-02, 16.65 min recovered, RTK-fixed, manually driven) is the evaluation bag. `part2_2_9_2025` is **94.1% contained within it** and is used for **path validation only**; the two are never pooled, because doing so would double-count the same physical traverses and spuriously shrink the CIs — the failure D040's whole-bag pooling and D053's guard exist to prevent. `rh_july2026` (SBAS GNSS, 29.6% autonomous) is retained as the only autonomous Riseholme data. The August-2024 RealSense/RTK set is parked: camera-only, no robot pose, no established clock synchronisation.

**Out-of-distribution status verified, not assumed.** BLT covers Riseholme (5 sessions, 2023) as well as Ktima, so training contamination was checked rather than presumed. SemanticBLT's 405 month-less images resolve to **exactly 90** source scenes — matching the 90 unattributed scenes of D048 — and all show Mediterranean architecture, arid ground and red row-end roses, matching the Ktima july2023 frames, with none of Riseholme's glasshouse or water tank. **Riseholme is genuinely out-of-distribution and uncontaminated.** D048's "0 of 90 present" on the Riseholme-adjacent bags is correct for the right reason.

**Cross-references.** D048 (contamination gate), D052 (july2023, the first OOD observation), O007 (OOD evaluation), D056–D059, F030, F031.

## D056 — Riseholme camera extrinsics: empirically derived, partially locked
**Date:** 7 August 2026
**Status:** LOCKED

**Decision.** Three of the six degrees of freedom are adopted as an empirically derived, cross-verified calibration. The other three are **explicitly not adopted** and are set to a stated baseline, with every downstream number conditional on them.

| DOF | value | basis |
|---|---|---|
| **height** | **1.269 m** | 1.269 (rh_july2026, n = 59) vs 1.278 (part2, n = 51) — **9 mm** apart, eleven months apart |
| **pitch** | **+5.75° down** | 58 of 59 samples positive; terrain excluded (below) |
| **roll** | **+0.75°** | mean of 0.98 (rh_july2026) and 0.45 (part2) |
| lateral | 0.0 m **ASSUMED** | two estimates **33 mm** apart, exceeding the **19.0–20.2 mm** effect GT-1 resolves (corrected 9 Aug 2026 from 19.5–24.1; see the D052 correction — the argument is unchanged and slightly strengthened, since 33 mm now exceeds the *whole* range rather than sitting inside it) |
| yaw | 0.0° **ASSUMED** | `/scan` gives +3.21°, IQR [+1.89, +5.16], which excludes the collector's stated 0° |
| longitudinal | 0.0 m **ASSUMED** | never estimated; no available method constrains it |

**Why empirical rather than read from the robot.** The camera was never published to this robot's `tf` tree. Verified by exhaustive `frame_id` enumeration over `/tf` and `/tf_static` on three bags across two sessions (25,063 + 11,241 + 8,611 messages), dumping every frame rather than substring-matching for "cam"; the leg and wheel URDF transforms are present and correct throughout, so this is not a reading failure. An earlier hypothesis — that the re-recording's topic list dropped `/tf_static` — was **falsified**: `part1_2_9_2025` captured `/tf_static` from a near-complete replay and the camera was still absent. No original fragment will supply it. No public source documents this mounting either: BLT (Polvara 2024) used two ZED2 cameras at both sites, and every published Thorvald + D435i configuration is forward-facing.

**Terrain excluded as an explanation for the pitch.** The field slopes 3.78° (±2.93° along the row axis), so a camera mounted at 0° pitch would yield a distribution centred on zero with roughly half the samples negative. Observed: median +5.746°, std 1.458, **1 of 59 negative**. The spread is consistent with terrain riding on a fixed tilt; the offset is not.

**The two lateral estimates are not fully independent.** Both derive the camera's view of the rows from the *same* depth ground-plane fit, and both inherit the same ~5% canopy-versus-trunk bias (0.952 on part2, 0.930 on rh_july2026). They differ only in the *second* sensor used to locate the robot — 2D LiDAR in `base_link` for one, RTK GNSS against the surveyed geometry for the other. This is supporting evidence, not two-source confirmation in the sense of D052's four-decimal `tf`-versus-Table-3 match.

**Adopted without the data collector's confirmation.** The checks above stand on their own method. If his account later contradicts these values, that contradiction is a finding to investigate on its own evidence, not grounds for having withheld them.

**Cross-references.** D052 (the contrasting, tf-validated case), D057, D059, F031, `docs/RISEHOLME.md` §§4, 12, 13.

## D057 — Riseholme reference: the surveyed mid-row line, not the driven path
**Date:** 7 August 2026
**Status:** LOCKED

**Decision.** Riseholme lateral error is measured against the **surveyed mid-row line** in `riseholme.geojson`. **O020/D014's autonomous-driven-path framing does not transfer to this site.**

**Why it cannot transfer.** Ktima's GT-1 is the RMS of the vision-estimated centreline offset *about zero*, which is meaningful only because the robot is **autonomously following the row**, so its driven path defines the row centre. The September 2025 Riseholme sessions were **manually driven** — stated by the data collector and corroborated independently by the complete absence of an `/auto_mode` topic on both 2025 bags, where `rh_july2026` records it at 10.84 Hz. A human operator may sit deliberately off-centre, so RMS-about-zero conflates vision error with real driving deviation. That term is **not small**: the robot's true offset from the surveyed line has **std 0.296 m**, comparable to or larger than the vision error being sought.

**What replaces it.** `error = vision_estimated_offset − true_offset_from_surveyed_line`, computed in `scripts/riseholme/rh_evaluate.py`. This is a **new file**: `analyze.py`'s driven-path-is-centre assumption is correct for Ktima and must not be altered, so the shared code is left untouched (D058).

**Row correspondence is solved exactly.** WayPoint *N* lies on the mid-row line between `row_(g+2)` and `row_(g+1)` where `g = ⌊(N−1)/12⌋` — 108 of 108 waypoints assigned, exactly 12 per line, none left over, established geometrically via a map→WGS84 fit (rotation −0.879°, residual 25.6 cm median) against 2.5 m row spacing. The geojson and the robot's topological map are the same map under two naming schemes; no string-level overlap exists between them.

**Limits of this reference.** The geojson lines are `measured_or_calculated: "calculated"` — derived geometry, not surveyed, with accuracy inheriting from a source the file does not name. Robot position comes from RTK GNSS at 39–62 mm short-scale residual. Both bound the **absolute** metric; neither affects the **paired** contrasts, which is the basis of D059.

**Cross-references.** O020, D014 (the Ktima framing that does not transfer), D055, D059, F031.

## D058 — Riseholme code isolation and the algorithm-parity gate
**Date:** 7 August 2026
**Status:** LOCKED

**Decision.** Riseholme runs from `scripts/riseholme/`, which neither imports from nor is imported by `scripts/geometric/`. Every file constituting the measurement is **byte-identical** across the two trees, enforced mechanically by `scripts/riseholme/verify_algorithm_parity.py`, which must pass before any Riseholme evaluation and after any edit to either tree.

**Why.** Two reasons, both load-bearing. First: if the two sites' results differ, that difference must be attributable to the data, never to the code — a comparison whose arms ran different algorithms would be worthless. Second: D046f recorded that extending a single-bag-shaped code path to a second bag produced **five** defects sharing one shape, **four of them silent**, two already latent in committed results. Repeating that against a committed five-bag result weeks from submission is an unacceptable risk.

**Shared, byte-identical (7 files).** `row_model.py`, `cp3_geometry.py`, `block_lengths.py`, `extract_detections.py`, `line_fit_infer.py`, `analyze.py`, `cuda_preload.py`. Consequently the tuned constants — `NEAR_M`, `TOL`, `INL`, `BINS`, `LOOKAHEAD_BIN`, `CONF`, `BLOB_FRAC`, `HALF`, seeds 42/43/44 — are not merely "kept the same"; **they are the same bytes**. They were tuned for a ZED2 at 0.763 m and 1.95° pitch and may well be suboptimal for a D435i at 1.269 m and 5.75°. **Changing them would destroy comparability, so they stand unchanged and the mismatch is recorded as a limitation rather than tuned away.**

**A concrete consequence, quantified.** The two cameras see different ground windows: the image bottom projects to **2.48 m at Riseholme** (1.269 m high, 5.75° down) against **1.76 m at Ktima** (0.763 m, 1.95°). The shared look-ahead bin `BINS[0] = (1.0, 3.0)` m is therefore populated over 2.48–3.0 m at Riseholme and 1.76–3.0 m at Ktima — **26% of the bin against 62%**. The "2 m look-ahead" is consequently **not the same measurand at the two sites**: at Riseholme it is measured further out, over a narrower window, with fewer base points contributing. Retuning `BINS` would make Riseholme internally cleaner but incomparable with Ktima, so the constants stand unchanged and this is recorded as a limitation. It affects only the already-caveated absolute per-arm values (F031); coverage (F030) is classification-based and untouched, and the paired contrasts are unaffected because the measurement zone is identical for all three arms on the same frames and cancels in the difference (D059).

**Permitted differences, all declared and machine-checked.** (a) One `sys.path` token naming the tree a file lives in — hashing normalises it, so identity is still proven over every constant and every line of logic, and the gate prints where it applied. (b) `prep.py`'s CP-1 body, diffed function-by-function and verified to differ in **only two operator-facing error strings**. (c) Five input/output and calibration files: `bag_config.py` (presents the *identical* `resolve()` interface), `projection_calibration.py`, `prep.py`, `extract_frames.py`, `check_bag_complete.py`. Five further Ktima files are declared not-ported. **Any untriaged Ktima file fails the gate**, so a new script cannot slip in unclassified.

**Cross-references.** D046f (the defect class this prevents), D049 (cuDNN preload), D055, D056.

## D059 — Riseholme reporting asymmetry: absolute caveated, paired primary
**Date:** 7 August 2026
**Status:** LOCKED

**Decision.** Both quantities are reported, always together, with the trust asymmetry made explicit in **every** caption, table and passage in which they appear:

- **Per-arm absolute lateral RMS — reported, heavily caveated.** Never to be read as a precise accuracy figure.
- **Paired cross-arm differences (A−B, A−C, B−C) — the primary, defensible Riseholme result**, and the same measurand as F013 at Ktima.

**Why the asymmetry is principled, not presentational.** For two arms evaluated on the *same frame* against the *same* true offset:

```
err_A − err_B  =  (vis_A − true) − (vis_B − true)  =  vis_A − vis_B
```

The unknown true offset cancels **exactly**. So does the calibration bias: lateral and yaw are properties of **one physical camera shared by all three arms**, so an incorrect assumption displaces every arm identically and vanishes in the subtraction. This is the same common-mode-cancellation principle already relied upon elsewhere in this work for RTK bias and for sensor tilt, applied to a new situation.

| quantity | limited by |
|---|---|
| absolute per-arm RMS | **±182 mm** from the assumed extrinsics (70 mm lateral + 112 mm yaw at the 2 m look-ahead), plus 39–62 mm GNSS short-scale and the "calculated" geojson |
| paired differences | sample density only, gated by D053 |

**The individual numbers are not dropped.** Removing them would conceal the absolute accuracy achieved out of distribution, which a reader is entitled to see. They are shown with their uncertainty band and an explicit statement that they are contaminated by the unknown calibration and driving-offset terms.

**Made visible, not merely captioned.** A sensitivity plot reports RMS as a function of the assumed lateral offset and yaw, demonstrating directly that the paired differences stay flat against those assumptions while the absolute values do not. `projection_calibration.sensitivity()` prints the budget alongside every Riseholme GT-1 figure, so the caveat travels with the number rather than living only in prose.

**Cross-references.** D053 (CI reliability guard), D056 (the assumed DOF), D057 (the reference), F013 (the same measurand at Ktima), F031.

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
- Data: `data/yolo_binary/` from `scripts/perception/pipeline/coco_to_yolo.py` (O005); D028 routing (train 721 / val 46 / test 23 representative).
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
- Data: `data/yolo_multiclass/` from `scripts/perception/pipeline/coco_to_yolo.py --mode multiclass` (O005 / D025); D028 routing (train 721 / val 46 / test 23 representative). Training regime **identical to Phase B** (100 epochs, patience 30, batch 16, workers 0, imgsz 640, SGD schedule, augmentation) — only `nc` and the data path differ (verified: B↔C non-cls training losses match to <0.01, F008).
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

**Regeneration recipe** (the overlap PNGs live under `results/runs/` and are gitignored like every other run artefact; they are reproducible on demand, not merely held locally): `python scripts/perception/diagnostics/blob_overlap_6799.py` — default runs are the four blobbing runs (Phase B seeds 42, 43; Phase C seeds 43, 44); predicts 6799 with each run's locked `weights/best.pt` at conf 0.25 (half=True, D029), takes the largest-area mask per run, and writes `overlap_<a>_<b>.png` + `overlap_summary.json`. Provenance: analysis script committed with this multi-seed pass; Phase B seed configs committed at 4044395, Phase C seed configs at d44cccf. Measured 4-way result: largest-mask areas 75,256–76,837 px; pairwise mask IoU mean 0.929 (range 0.924–0.937 across 6 pairs); centroids within ~6 px.

Blob rate across arms: Phase B 2/3, Phase C 2/3. Class-aware supervision does not affect the failure rate. See F007 for full analysis.

Training-run SD on rasterised fg IoU (0.022) is slightly lower than Phase B (0.027) but higher than Phase A (0.008). The intermittent blob failure dominates variance in both YOLO arms.

### O004 — Literature review extension
Supervisor flagged A1's 6 references as thin. Must reach ~12–15 for A2. Extension planned during dissertation writing phase.

### O005 — COCO→YOLO conversion: in-place script
**Date:** 4 July 2026
**Status:** LOCKED. Path B chosen (in-place COCO→YOLO conversion).

Decision: convert COCO polygon annotations to YOLO segmentation format via an in-repo script (`vineyard_nav/scripts/perception/pipeline/coco_to_yolo.py`), parameterised by class-collapse rule. Do NOT re-download from Roboflow in YOLO format.

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