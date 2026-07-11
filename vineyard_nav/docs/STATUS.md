# STATUS.md — Current Pipeline & Progress Tracker

**Purpose:** Quick orientation for the current state of the project. Use this as the primary handover document when starting a new chat. Point to `DECISIONS.md` for rationale on any specific decision.

---

## Project

**Title:** Multiclass Semantic Segmentation for In-Row Vineyard Navigation: A Comparative Study Against the Binary-Mask Baseline
**Student:** Edosa Ebohon (30436293), MSc Robotics and Artificial Intelligence
**Institution:** University of Lincoln
**Module:** CMP9140 Research Project
**Timeline:** 13 June 2026 → 26 August 2026 (A2 submission)
**Working directory:** `/workspaces/dissertation/vineyard_nav/`

---

## Current state (as of 28 June 2026)

**Planning phase — post supervisor feedback design refinement. Implementation not yet started.**

Recent events:
1. A1 proposal submitted (1 July 2026 deadline; submitted early)
2. Supervisor instruction: reproduce baseline first, then novel development
3. Supervisor feedback rounds resulted in three-arm design (U-Net binary + YOLO binary + YOLO multiclass) with Config A/B/C downstream sweep on Phase C
4. Data split changed from Roboflow default (95/5/2) to 70/20/10 stratified resplit
5. Roboflow-trained model verified as `roboflow-3-n-seg` (instance seg, mAP@50 74.1%) — reference only, not reused

---

## Pipeline architecture

Three model arms, all feeding the same downstream (per-side clustering → RANSAC → offline PID) and evaluated by the same three-strand framework (perception, geometric, command-level).

| Arm | Phase | Model | Class structure | Purpose |
|---|---|---|---|---|
| 1 | A | U-Net (SMP + ImageNet pretrained encoder) | Binary (trunk+pole → foreground) | Official baseline — reproduces de Silva 2024 paradigm |
| 2 | B | YOLOv11-seg (COCO pretrained) | Binary (trunk+pole → 1 class) | Modernised binary baseline |
| 3 | C | YOLOv11-seg (COCO pretrained) | Multiclass (trunk, pole distinct) | The contribution — tests class-aware downstream |

**Isolated comparisons:**
- A ↔ B → **architecture effect** at fixed binary labelling
- B ↔ C → **class-structure effect** at fixed YOLO architecture

**Phase C downstream sweep:**
- Config A: trunk primary, pole fallback below threshold T
- Config B: pole primary, trunk fallback below threshold T
- Config C: class-agnostic (trunk + pole treated as one pool)
- Sweep 3 configs × 6 T values on validation; test at locked (config*, T*)

---

## Progress tracker

### Environment ✅
- [x] Devcontainer working (L-CAS ROS2 Humble, PyTorch 2.11+cu128, sm_120 verified)
- [x] SemanticBLT dataset at `/workspaces/dissertation/SemanticBLT.v1-2024-june.coco-segmentation/`
- [x] Data structure verified: COCO polygon format, 26,280 train annotations across 6 classes
- [x] `vineyard_nav/` folder scaffolded

### Design ✅
- [x] Research question and contribution locked
- [x] Three-arm design locked
- [x] Phase C downstream sweep methodology locked
- [x] Reproducibility framework locked
- [x] Working rules locked (including "no directional framing before Results")

### Data preparation (in progress)
- [x] 70/20/10 stratified resplit with augmentation-leakage guard — **scene-level** (D028, supersedes D024). `scripts/resplit_dataset.py` → `data/splits/resplit_70_20_10.json`. 230 scenes → 161/46/23; leakage-verified; deterministic (seed 42). Test = **23 independent scenes** (11 bare-vine + 12 canopy) — honest bootstrap units; see O006 (raise with supervisor).
- [x] Binary labels for U-Net (Phase A) — via `SemanticBLTBinaryDataset` (on-the-fly COCO→mask)
- [x] YOLO binary label files (Phase B) — `scripts/coco_to_yolo.py` (O005, convert_coco + collapse + D028 routing) → `data/yolo_binary/` (721/46/23; 14,894 fg lines audited == COCO cat{3,5}; coords match source to 0.0003px). `data.yaml` + `canopy_state_map.json` written.
- [ ] YOLO multiclass label files (Phase C — trunk + pole only) — same script, `--mode multiclass` (to add)

### Phase A — U-Net binary
- [x] Dataset class + spot-check visualisation — `segmentation/unet_binary/dataset.py`; spot-check gate **passed** (labels verified: red covers trunk+pole, excludes pipes/robot, both canopy states)
- [x] SMP U-Net wrapper (ResNet-34 encoder, ImageNet pretrained) — `segmentation/unet_binary/model.py`; smoke test passes (24.44M params, [B,2,640,640] logits); CUDA+AMP forward verified on sm_120
- [x] Loss (0.5·CE + 0.5·Dice) + metrics (mIoU, per-class IoU/P/R/F1) — `losses.py`, `metrics.py`; unit tests pass. mIoU from accumulated confusion-matrix counts (verified size-1-batch = analytic 1/3, batching-invariant). Dice = equal-weighted multiclass soft Dice (see note below).
- [x] Training loop (AMP, TensorBoard + CSV logging, checkpoint schema) — `train.py` + `configs/phase_a_unet_binary.yaml`. Smoke gate passed; bitwise reproducibility verified (D016). AMP healthy on sm_120. Peak VRAM batch=8: 3.33 GB / 8 GB.
- [x] Full training done — run `results/runs/phase_a_unet_binary_20260704_004105/` (git 5b4f1c05). Early-stopped @ epoch 52 (patience 10). **Best val mIoU 0.8456 @ epoch 42** (fg IoU 0.6991, val_loss 0.0647). ~41 min. Curves: `training_curves.png` (healthy; mild train/val loss gap, val loss flat — no worsening overfit).
- [x] Validation evaluation — per-epoch during training; `best.pt` locked @ epoch 42.
- [x] `evaluate.py` + `visualize.py` built (impl order step 6) — **dry-run on VALIDATION passed**: reproduces best.pt val mIoU 0.8456 exactly; canopy-stratified; 46 GT-vs-pred panels rendered; `valid_metrics.json` (§9.1 schema). Test set untouched.
- [x] Test evaluation (once) — **DONE 4 Jul 2026, not to be re-run (rule 5)**. Overall test mIoU **0.8561**, fg IoU 0.7195 (23 scenes). Bare-vine 0.8414 / canopy 0.8858. `test_metrics.json` + 23 panels in `predictions_test/`.
- [x] Metrics recorded in DECISIONS.md (O003)
- [x] Bootstrap CIs on the 23 test scenes (D020/O006) — `evaluate.py` now emits `test_per_frame_metrics.csv` (additive; test_metrics.json byte-identical, md5 verified); `evaluation/bootstrap.py` (D020 utility, 10k resamples, seed 42) → `test_bootstrap_ci.json`. Overall fg IoU 0.7119 [0.6572, 0.7659]; canopy−bare-vine gap +0.072 **[−0.034, +0.174] (includes 0)**. F001/F002/F003 + D028 updated.

### Phase B — YOLO binary
- [x] Data prep — `scripts/coco_to_yolo.py` (O005 LOCKED); `data/yolo_binary/` built, numeric+visual spot-check passed. ultralytics 8.4.90 installed & pinned (torch unchanged; opencv note in requirements).
- [x] YOLO data.yaml configured — `configs/phase_b_yolo_binary_data.yaml`
- [x] opencv drift reconciled (O008 RESOLVED) — cv2 single-sourced headless 5.0.0.93; requirements pin updated
- [x] Training config + entry point — `configs/phase_b_yolo_binary_train.yaml` (§6.2; workers 4→0 env-forced), `segmentation/yolo_binary/train.py`. **2-epoch smoke PASSED**: no OOM, no NaN, val losses ↓ (box 3.68→3.25, seg 4.32→3.97), GPU 3.77/8 GB @ batch 16, ~9s/epoch, deterministic (identical reruns). Path nesting fixed (absolute project).
- [x] Full training via ultralytics — 100 epochs, 45.2 min, best.pt @ epoch 86. Val mask mAP@50 **0.629** (box 0.709), peak VRAM 4.23/8 GB. No NaN/collapse.
- [x] `evaluate.py` built (§7) — overall + canopy-stratified via temp list-yamls; `half=True` (AMP-consistent, D004). **Val reproduction EXACT**: overall mask mAP@50 0.6291 == training epoch-86 0.6292. Canopy 0.686 > bare-vine 0.606 (replicates F001). `val_metrics.json` written.
- [x] Test evaluation (once) — **DONE 8 Jul 2026, not to be re-run (rule 5)**. Overall mask mAP@50 **0.6161** (box 0.7219); bare-vine 0.6249 / canopy 0.6192. `test_metrics.json` + 23 GT|Pred panels in `predictions_test/`.
- [x] `visualize.py` standalone (§2) — GT|Pred mask panels, parallel to Phase A.
- [x] Bootstrap CIs (D020 reuse) — per-frame foreground **pixel** metrics → `test_per_frame_metrics.csv` + `test_bootstrap_ci.json`. Overall pixel IoU_fg 0.556 [0.466, 0.633]. (mAP has no per-frame CI; per-frame pixel metric parallels Phase A.)
- [x] conf-threshold sweep on val (D030) — `scripts/phase_b_conf_sweep.py`; **conf\* = 0.25** (val argmax, coincides with default → committed test stands). Curve `val_conf_sweep.png`; mildly sensitive (spread 0.020, F006).
- [x] Metrics recorded in DECISIONS.md (O003 Phase B block)

**Phase B complete** (§10): trained + best.pt locked · test once · results.csv preserved · predictions saved · DECISIONS O003 updated · STATUS updated. → Phase C (YOLO multiclass) can begin.
- [ ] Validation evaluation
- [ ] Test evaluation (once)
- [ ] Metrics recorded

### Phase C — YOLO multiclass — **PERCEPTION COMPLETE (downstream deferred, O010)**
- [x] Data prep — `coco_to_yolo.py --mode multiclass` (D025); `data/yolo_multiclass/` (721/46/23; 14,894 fg lines == Phase B; collapse verified numeric + class-coloured spot-check). `configs/phase_c_yolo_multiclass_data.yaml` (nc:2, trunk/pole).
- [x] Training via ultralytics — `segmentation/yolo_multiclass/` (train copy of Phase B's). Smoke passed; full 100 epochs, 49.3 min, best.pt @ ep 94, peak 4.25 GB. Regime identical to Phase B (F008: non-cls losses match <0.01; cls diverges — controlled comparison confirmed).
- [x] Val reproduction — `evaluate.py --split val`: overall mask mAP@50 0.6126 == training 0.613 (half=True, D029).
- [x] Test evaluation (once, conf 0.25) — **DONE 10 Jul 2026, not to be re-run (rule 5)**. Overall mask mAP@50 **0.6378** (box 0.7268); per-class trunk 0.678 / pole 0.598. Rasterised fg IoU 0.619 [0.572, 0.666]. **6799: no blob, fg IoU 0.627** (F007 informant).
- [x] Metrics recorded in DECISIONS.md (O003 Phase C block); F007 updated; F008 added.
- [~] Downstream sweep + test-at-locked-config — **DEFERRED to geometry-pipeline phase (O010)**; needs RANSAC/centreline/trajectory (not built). Phase C best.pt locked & deterministic; sweep runnable later without retraining.

**Phase C perception complete.** Remaining before dissertation-writing consumes results: geometry pipeline → downstream sweep (O010) → 3-way attribution (spec §10); multi-seed robustness (O009).

### Multi-seed robustness (O009) — COMPLETE (all three arms, seeds 42/43/44)
- [x] Phase A U-Net — 3 seeds (42/43/44). Test fg IoU **0.716 ± 0.008**, mIoU 0.858 ± 0.003, canopy>bare gap +0.076 ± 0.004. Highly stable; no blob mode (U-Net per-pixel). Blob rate 0/3.
- [x] Phase B YOLO binary — 3 seeds. Test mask mAP@50 **0.632 ± 0.016**, fg IoU 0.585 ± 0.027. **6799 blob 2/3 seeds** (42,43 blob; 44 clean); when present, same region (cross-seed mask IoU 0.93). Phase B fg-IoU variance ~3.4× Phase A's, blob-driven (F009).
- [x] Phase C YOLO multiclass — 3 seeds. Test mask mAP@50 **0.644 ± 0.008**, fg IoU 0.594 ± 0.022. **6799 blob 2/3 seeds** (43,44 blob; 42 clean). Blob rate identical to Phase B → **6799 blob is class-structure-independent**; class-aware-supervision-prevents-blob hypothesis falsified (F007).
- **O009 status: COMPLETE.** Cross-arm blob analysis confirms the 6799 blob is a YOLOv11-seg architecture-family × scene pathology (0/3 Phase A; 2/3 each Phase B and Phase C; mask geometry mean 0.93 / range 0.92–0.94 across all six pairwise comparisons of the four blobbing runs). Downstream cross-arm perception ranking deferred to the geometric strand (O010).
- Config-copy recipe (seed-specific YAMLs); distinct seed-tagged run dirs; seed-42 artefacts untouched. Rule 5: each seed's best.pt test-evaluated once.

### Cross-arm perception methodology (D031 LOCKED, F005 REVISED)
Perception uses **native metrics per arm** (U-Net: mIoU/IoU; YOLO: mAP@50/per-class). Rasterised fg IoU retained **per-arm only** as internal characterisation (canopy stratification, blob detection, per-arm CIs) — NOT cross-arm ranking. Primary cross-arm comparison DEFERRED to the geometric strand (RMS lateral error, O010 pipeline). Doc changes: F005 revised, D031 locked, F007 refined (multi-seed), F001 strengthened, F009 added.

### Downstream + evaluation
- [ ] Per-side clustering module (works on pixel masks AND instance centroids)
- [ ] RANSAC line fitting module
- [ ] Offline PID controller (hand-tuned, kinematic sanity check)
- [ ] Three-strand evaluation framework
- [ ] Bootstrap CI + effect size utilities
- [ ] Canopy-state stratification

### Dissertation writing
- [ ] Introduction (refined from A1)
- [ ] Literature Review (extend beyond 6 references — supervisor flag)
- [ ] Methodology (documents all refinements from A1)
- [ ] Implementation
- [ ] Results & Discussion
- [ ] Conclusion

---

## Immediate next action

**Data preparation, in this order:**
1. ~~Script the 70/20/10 stratified resplit~~ ✅ done — scene-level resplit (D028), manifest at `data/splits/resplit_70_20_10.json`
2. ~~Verify no augmented duplicates leak across splits~~ ✅ done — leakage guard passes; determinism confirmed
3. Prepare binary mask generation for Phase A U-Net  ← **next**
4. Convert COCO annotations to YOLO format for Phases B and C

**⚠ Before Phase A training:** raise O006 with supervisor — honest test set is 23 scenes (dataset ceiling of 230 unique scenes). D024's ~100-frame target was augmentation-inflated; D028 makes evaluation honest but revives the 23-frame thinness the supervisor flagged.

---

## Key locked decisions (short list)

Full rationale and history in `DECISIONS.md`. Headline items:

- **Three-arm design.** U-Net binary + YOLO binary + YOLO multiclass. No separate robustness check — U-Net-vs-YOLO comparison built into primary design.
- **U-Net binary:** SMP with ResNet-34 encoder, ImageNet pretrained. Not scratch (superseded).
- **YOLO:** YOLOv11-seg, COCO pretrained, via ultralytics.
- **Multiclass classes:** trunk + pole only, not all 6. All-6 kept as optional supplementary experiment.
- **Phase C downstream sweep:** 3 configs × 6 T values (T ∈ {1, 2, 3, 5, 8, 12} instance counts). Selected on val; test evaluated once.
- **Data split:** 70/20/10 stratified by canopy state; all augmentations of a base image stay in same split.
- **Resolution:** native 640×640, no downsampling.
- **Reproducibility:** seed = 42; git commit hash in every checkpoint; versions pinned in `requirements.txt`.
- **Statistics:** bootstrap CIs + effect sizes over per-frame metric differences. No p-values.
- **Framing:** no directional claim about which arm wins before Results chapter.

---

## Environment quirks (must-know)

- Devcontainer has system Python + venv overlay. When installing new packages, always use `pip install --upgrade <pkg>` so it lands in the venv, not the system layer.
- If imports fail with `_ARRAY_API not found`, the overlay is biting — upgrade the failing package into the venv.
- PyTorch is 2.11.0+cu128 (recent) — matters because sm_120 (Blackwell 5050) support only appeared mid-2026.
- Ignore harmless conflicts: `conan PyYAML`, `Axes3D import warning`, `grpcio-tools protobuf`.

---

## Open items

- **O001 (existed):** Threshold T range — currently {1, 2, 3, 5, 8, 12} instance counts for Phase C. May re-anchor after seeing YOLO multiclass detection densities.
- **O002 (existed):** All-6-classes supplementary experiment — only if Phases A/B/C complete on time.
- **O003 (existed):** Phase A + B + C test metrics — appended to DECISIONS.md as each phase completes.
- **O004 (new):** Literature review extension — supervisor flagged 6 references as thin. Must reach ~12–15 for A2.
- **O005 (new):** "Poles remain visible" retraction from A1 — must be openly acknowledged in A2 Methodology or Discussion.
- **O007 (new):** OOD annotation — label Riseholme footage (different vineyard/season) for an out-of-distribution eval set; scheduled last, per Riccardo. See DECISIONS O007.
- **O010 (new):** Geometry pipeline + Phase C downstream sweep — deferred to the geometry-pipeline phase; primary cross-arm comparison lives here (RMS lateral error). See DECISIONS O010.

---

## Related documents

- `PROJECT_PLAN.md` — full project scope, phase design, evaluation framework
- `DECISIONS.md` — running decisions log with rationale (feeds A2 Methodology chapter directly)
- `PHASE_A_SPEC.md` — U-Net binary implementation contract
- `PHASE_B_SPEC.md` — YOLO binary implementation contract
- `FINDINGS.md` — empirical observations from implementation and evaluation (feeds A2 Results and Discussion chapters)
- `PHASE_C_SPEC.md` — YOLO multiclass implementation contract
- `Masters_Dissertation_Proposal.pdf` — A1 proposal (submitted; source of truth for research question)

---

## Continuation protocol (for new chats)

When starting a new chat:
When starting a new chat:
1. Share this STATUS.md file
2. Share `DECISIONS.md` if the new work touches a locked decision
3. Share `FINDINGS.md` if the new work references empirical results (Results chapter, Discussion, or new findings likely to arise)
4. Share the specific PHASE_X_SPEC.md for the phase being worked on
5. State the immediate task

Do not re-open locked decisions unless there's new evidence. If a locked decision needs revisiting, add a new decision entry to `DECISIONS.md` that supersedes the old one — do not overwrite history.
