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

## Current state (as of 23 July 2026)

**Perception complete; March geometric strand closed; CONTROL STRAND CLOSED; April multi-bag evaluation IN PROGRESS.** The two sections after the March bullets below carry the current state; the March detail is retained unchanged for provenance.

**Perception complete; March geometric strand closed and POOLED to whole-bag (D040).** All three arms trained + tested once + multi-seed (O009 complete). The March geometric strand (in-row centreline evaluation on `kg_march_23`) ran CP-0→CP-6; the val/test split was then **pooled into a single whole-bag evaluation** (D040; Commits 2a/2b/3) over all **7,857 in-row eligible frames** (47% of the 16,656-frame bag; D041 frame accounting). The locked **line-fit** pipeline (D036–D038) is now **bag-parametrised** (`--bag march`, Commit 2b), the multi-bag template.

- **Whole-bag line-fit evaluation (9 models × 7,857 frames) complete.** Per-arm GT-1 RMS ~0.19–0.22 m, GT-2 RMS ~2.5°, two-row coverage ~81%; arms **indistinguishable on the primary GT-1 metric** (F013 — paired moving-block bootstrap, all CIs include zero) with a **sub-noise-floor GT-2 offset** that surfaces on pooling (A<C<B, ≤5.6% of the noise floor, sign-inconsistent for 2/3 pairs — reported honestly, below navigation relevance). Artefacts: `results/geometric/march/final/march_evaluation/` (`line_fit_report.json`, `line_fit_per_frame.csv` 12-col, `paired_crossarm.json`, `config_analysis.json`, `lidar_crosscheck.json`).
- **Findings F010–F019 pooled additively (Commit 3)** — the val→held-out-test-confirmation→pooled derivation trail is preserved in each, with pooled numbers as the headline. F013 pooled (GT-1 indistinguishable + honest GT-2 persistence); F015 & **F019 SUPERSEDED** (kept as historical trails); F016 driven-path (BLT autonomous, Polvara 2024); F017 sensor-common tilt (pooled 10-anchor LiDAR); F018 config (class-agnostic locked, Phase-C-multiclass-specific); F010/F014 whole-bag per-side slopes / adjacent rate (enabled by the Commit-2b 12-col schema); F012 pooled regression-residual (A 1.37/B 1.33/C 1.32°). Decomposed reporting throughout (bias / residual-SD).
- **D041 frame accounting:** 7,857 in-row eligible (47%) evaluated + 5,841 non-in-row (35%, → Commit 6) + 2,958 contaminated (18%, perception-leakage, excluded) = 16,656; mutually exclusive + exhaustive (contamination-first).
- **Config sweep + ablations** (D026) re-reported on pooled data — **class-agnostic locked** (F018; re-reported, not re-selected — the design was locked before pooling).
- **Provenance:** the whole-bag artefacts were produced by the committed bag-parametrised scripts (`line_fit_infer` / `extract_detections` / `line_fit_eval` / `paired_crossarm` / `config_analysis` / `lidar_crosscheck` `--bag march`) — verified byte-for-byte (val) / value-equivalent (test) against the val/test-era outputs. The val/test split scripts + artefacts are retained under `scripts/geometric/superseded/` and `results/geometric/march/superseded/march_val_test_split/` (audit trail).
- **March geometric strand FULLY CLOSED (whole-bag pooled):** F010–F019 pooled; F013 GT-1 indistinguishable + GT-2 honest sub-noise-floor persistence; F015/F019 superseded trails; class-agnostic locked. **Rule 5 (single-shot test) now applies at the multi-bag (whole-bag-per-month) level (D040)** — the within-March held-out test is superseded by the seasonal generalisation claim.
- **Remaining strand work (in order):** Commits 2a–10 done (pooling, pipeline consolidation, findings, STATUS + per-month template, manifest cleanup + whole-bag subsample, non-in-row characterisation F020/F021, mitigation demonstration F022/F023, in-row abstention F024, **report figures — 15, O012**, near-seed window sensitivity F025). Next: multi-bag seasonal (April+) and PID.
- **Test set (23 scenes), Phase A/B/C artefacts, CP-0/1/2/3 artefacts untouched.**

### Control strand — CLOSED (F026–F028, D042–D044, `PID_PIPELINE_SPEC.md`)

The command-level strand (D014 strand 3) ran CP-P1→CP-P4 on march and is complete. Design contract: `docs/PID_PIPELINE_SPEC.md`; design intent mirrored in `docs/CONTROL_DESIGN_INTENT.md`. Artefacts under `results/geometric/march/final/command_evaluation/` and `…/mitigation_evaluation/`.

- **F026 — native-twist state gate.** F022's odometry gate re-derived on the *deployable* onboard signal (`/odometry/base_raw.twist`, D042) rejects **97.6%** of spurious non-in-row centrelines at **0.9%** in-row FP, and collapses to a **single forward-speed predicate** (`v_x > 0.30 m/s`); the turn predicate is inactive natively. Amendment: the **IMU z-axis is sign-inverted** relative to `base_link` (corr −0.953 vs pose-derived).
- **F027 — the open-loop tracking objective is degenerate.** Tuning against the BLT run's executed yaw-rate collapses: all 11 pass-level folds pick the near-zero gain corner and beat commanding nothing by 0.4%; centreline→ω_exec is **R² = 0.0070**. Cause: the BLT platform navigated by **GPS/topological waypoints, not vision**. Objective retired.
- **F027-A — first-principles gains.** Gains derived from a unicycle small-angle model (ζ = 1.0, settling distance 20 m) rather than fitted, so no circularity: Kp = 0.0646, Kψ = 0.00778, Kd = Ki = 0; ramp bound derived in closed form. Saturation 0.41%.
- **F028 — command smoothness is arm-indistinguishable.** All paired cross-arm CIs on RMS Δω̂ span zero. **Converges with F013** — perception differences do not survive to the navigation output at *either* the geometric or the command level.

### April multi-bag evaluation — COMPLETE

Second seasonal bag (`kg_april_06`, bare-vine/early-growth). Pre-CP-2 stages are now bag-parametrised and the ROS1→ROS2 conversion is a committed script (**D046**).

- **Done:** conversion (116.5 GB `.db3`, 24,355 camera frames) → CP-0 census (10/10 april-labelled scenes located, corr 0.883–0.957, 310 frames / 1.3% excluded) → CP-1 manifest (**8,889 in-row eligible**, 12 passes, 4 corridors; `expected_passes = 12` locked as a *reproduction* guard, D046a) → frame extraction → detection cache (717,763 dets) → **9-model inference (80,001 rows)** → `line_fit_eval` + `paired_crossarm`.
- **Confirmed on april (bare-vine; each spot-checked against its artefact):**
  - **F013 (headline)** — GT-1 arm-indistinguishable, all CIs include zero; **A–B closes *at* zero** (≤ 3.2 mm). **March's GT-2** sub-noise-floor offset does **not** reproduce → bounded march-specific.
  - **F010** — tilt arm-consistent (~1.85°, SD 0.007°). **F011** — coverage **77%** (A 77.4 / B 78.2 / C 76.5), bound vindicated (not 83–84%). **F012** — geometric noise arm-invariant (GT-2 RMS 2.21–2.23°).
  - **F017** — camera↔LiDAR tilt agree (cam 2.78° / LiDAR 2.85°, Δ −0.06°, n=8), ~2.8°. **F018** — class-agnostic re-locks; poles supplement coverage (+13.6 pp), pole-only degenerates (0.6%).
  - **F020** — spurious two_row on ~20–32% of non-in-row (A 20.1 / B 26.6 / C 32.1, arm-varying). **F021** — driven_path_error 0.34–0.38 m / heading 5.2–6.5° (IPM-invalid degradation).
  - **F022 / F023** — state gate **87–92%** (87.2 / 90.1 / 91.7) @ 0.9% in-row FP; geometry filter **27–43%** (42.7 / 33.9 / 27.2) @ 1.8–2.6% in-row FP.
  - **F024** — in-row abstention 15% single_row (march 13%). **F026** — native gate `v_x>0.348`: reject **87.7 / 90.6 / 92.3%** @ 0.7% in-row FP.
  - **F027** — executed yaw-rate unpredictable from the centreline (best-linear R² **0.0045** pooled) — open-loop objective degenerate. **F028** — command smoothness **arm-indistinguishable** (all paired CIs span zero).
  - **F007** — geometric blob pathology absent (0 / 236k detections > 15% guard). **Neutral / not per-bag:** F014 holds; F016 neutral; F015/F019 superseded; F025 one-time march study.
- **Completion:** all stages ran (in-row, non-in-row, mitigation, control strand, 15 figures); `check_bag_complete.py --bag april` passes.
- **A working checklist of which findings need per-bag re-testing** (and which are one-time) was compiled 23 Jul 2026 — see the multi-bag audit; F001 (canopy-vs-bare-vine) and the decisive halves of F007/F018 **cannot be tested until a canopy bag runs**, which O019 gates.
- **⚠️ Single-bag-assumption defects (D046f).** Extending to april exposed **five** defects sharing one shape — code implicitly correct only while one bag existed. Four fail *silently*, and two were already latent in march's committed results (the D047 anchor selector had in fact already fired on march). **Rule for may/june/july/september: treat any code path not yet exercised by a second bag as suspect until proven otherwise** — "it worked on april" does not make a path generally correct. Carried into the dissertation's limitations material: reproducing a result does not test whether the code generalises.

### May multi-bag evaluation — COMPLETE

Third seasonal bag (`kg_may_06`, the first **canopy** bag). Locked pipeline, one whole-bag run; `check_bag_complete.py --bag may` gates completion.

- **Done:** conversion → CP-0 census (D048 gate: 0/90 unattributed scenes present) → CP-1 manifest (**8,840 in-row eligible**) → frame extraction → detection cache → 9-model inference → in-row analyses (Stage C) → non-in-row (Stage D) → mitigation → 15 report figures (Stage E) → control strand (Stage F). F007 geometric blob audit: **0 blobs** across 3 seeds / 416k detections (F007 amendment).
- **Confirmed on may (canopy; each spot-checked against its artefact):**
  - **F013 (headline)** — GT-1 arm-indistinguishable; **B–C null** (−0.5 mm, CI incl. 0). *First* bag where A-vs-{B,C} GT-1 resolves a **sub-floor** difference (A–B −4.1, A–C −5.4 mm; ≤ 14% of the 3.8 cm RTK floor), arm-A-specific. *[written, F013]*
  - **F010** — tilt **arm-consistent** (A/B/C 3.73–3.74°, spread 0.01°); higher canopy magnitude (march 2.1° / april 1.85°). **F011** — two_row coverage **63–68%** (67.6 / 65.1 / 63.0), canopy-reduced (march 81 / april 77). **F012** — geometric noise **arm-invariant** (GT-2 RMS 4.10–4.11°).
  - **F017** — camera↔LiDAR tilt **agree** (cam 3.30° / LiDAR 3.12°, Δ 0.18°, n=8), sensor-common ~3.2°. **F018** — **no viable downstream config** (max 63.1% < 70% floor), seasonal-coverage result. *[written]*
  - **F020** — spurious two_row on **~27%** of non-in-row frames (26.8 / 26.7 / 26.3, arm-consistent). **F021** — driven_path_error **0.32–0.34 m** / heading 8.4–9.4° (IPM-invalid degradation).
  - **F022 / F023** — state gate **~92%** @ 0.4% in-row FP; geometry filter **~56–62%** @ **~8–9%** in-row FP (canopy threshold-transfer). *[written, F023 amendment]*
  - **F024** — in-row abstention **22%** single_row (canopy-elevated; march 13 / april 15); dominant cause too_few_near_seed 65.5% (seen_far_only 62.7%). **F026** — native gate `v_x>0.363`: reject **94.0 / 93.7 / 94.1%** @ 1.2% in-row FP.
  - **F027** — executed yaw-rate **unpredictable** from the centreline (best-linear R² **0.0017** pooled) — the open-loop objective is degenerate on may too. **F028** — command smoothness: **B–C and A–C null**, but **A–B resolves** a micro-difference (ΔRMS +0.0008 rad/s, CI [+0.0002, +0.0015] excludes 0; ~3% of the 0.027 rad/s signal) — the **same arm-A-specific pattern as F013**, and equally navigation-negligible.
  - **F007** — geometric blob pathology **absent** (0 / 416k detections > 15% guard). *[written, F007 amendment]* **Neutral / not per-bag:** F014 (adjacent logging arm-invariant, scene-driven) holds; F016 neutral; F015/F019 superseded; F025 (near-seed sensitivity) one-time march study — canopy sensitivity not re-run.
- **Cross-bag reading:** may is the first bag where the **arm-A (U-Net) contrast resolves sub-floor micro-differences in *both* the geometric (F013 GT-1) and command (F028) metrics**, while the **B–C class-structure ablation stays null on every metric and every bag** — the cleanest statement that class structure does not change navigation, canopy included. The only differences that ever appear are architecture-level (A vs B/C), a separate question from the class-structure hypothesis. Canopy also lowers coverage (F011), raises tilt (F010/F017), raises abstention (F024), and inflates the fixed geometry-filter's in-row FP (F023).
- **Completion:** all stages ran; `check_bag_complete.py --bag may` passes once this summary is in place.

*Planning-phase history (retained for context):* A1 submitted early; supervisor → three-arm design (U-Net binary + YOLO binary + YOLO multiclass) + Config A/B/C sweep on Phase C; split changed to 70/20/10 stratified; Roboflow `roboflow-3-n-seg` reference-only.

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
- [x] 70/20/10 stratified resplit with augmentation-leakage guard — **scene-level** (D028, supersedes D024). `scripts/perception/pipeline/resplit_dataset.py` → `data/splits/resplit_70_20_10.json`. 230 scenes → 161/46/23; leakage-verified; deterministic (seed 42). Test = **23 independent scenes** (11 bare-vine + 12 canopy) — honest bootstrap units; see O006 (raise with supervisor).
- [x] Binary labels for U-Net (Phase A) — via `SemanticBLTBinaryDataset` (on-the-fly COCO→mask)
- [x] YOLO binary label files (Phase B) — `scripts/perception/pipeline/coco_to_yolo.py` (O005, convert_coco + collapse + D028 routing) → `data/yolo_binary/` (721/46/23; 14,894 fg lines audited == COCO cat{3,5}; coords match source to 0.0003px). `data.yaml` + `canopy_state_map.json` written.
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
- [x] Data prep — `scripts/perception/pipeline/coco_to_yolo.py` (O005 LOCKED); `data/yolo_binary/` built, numeric+visual spot-check passed. ultralytics 8.4.90 installed & pinned (torch unchanged; opencv note in requirements).
- [x] YOLO data.yaml configured — `configs/phase_b_yolo_binary_data.yaml`
- [x] opencv drift reconciled (O008 RESOLVED) — cv2 single-sourced headless 5.0.0.93; requirements pin updated
- [x] Training config + entry point — `configs/phase_b_yolo_binary_train.yaml` (§6.2; workers 4→0 env-forced), `segmentation/yolo_binary/train.py`. **2-epoch smoke PASSED**: no OOM, no NaN, val losses ↓ (box 3.68→3.25, seg 4.32→3.97), GPU 3.77/8 GB @ batch 16, ~9s/epoch, deterministic (identical reruns). Path nesting fixed (absolute project).
- [x] Full training via ultralytics — 100 epochs, 45.2 min, best.pt @ epoch 86. Val mask mAP@50 **0.629** (box 0.709), peak VRAM 4.23/8 GB. No NaN/collapse.
- [x] `evaluate.py` built (§7) — overall + canopy-stratified via temp list-yamls; `half=True` (AMP-consistent, D004). **Val reproduction EXACT**: overall mask mAP@50 0.6291 == training epoch-86 0.6292. Canopy 0.686 > bare-vine 0.606 (replicates F001). `val_metrics.json` written.
- [x] Test evaluation (once) — **DONE 8 Jul 2026, not to be re-run (rule 5)**. Overall mask mAP@50 **0.6161** (box 0.7219); bare-vine 0.6249 / canopy 0.6192. `test_metrics.json` + 23 GT|Pred panels in `predictions_test/`.
- [x] `visualize.py` standalone (§2) — GT|Pred mask panels, parallel to Phase A.
- [x] Bootstrap CIs (D020 reuse) — per-frame foreground **pixel** metrics → `test_per_frame_metrics.csv` + `test_bootstrap_ci.json`. Overall pixel IoU_fg 0.556 [0.466, 0.633]. (mAP has no per-frame CI; per-frame pixel metric parallels Phase A.)
- [x] conf-threshold sweep on val (D030) — `scripts/perception/diagnostics/phase_b_conf_sweep.py`; **conf\* = 0.25** (val argmax, coincides with default → committed test stands). Curve `val_conf_sweep.png`; mildly sensitive (spread 0.020, F006).
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
- **O009 status: COMPLETE.** Cross-arm blob analysis confirms the 6799 blob is a YOLOv11-seg architecture-family × scene pathology (0/3 Phase A; 2/3 each Phase B and Phase C; mask geometry mean 0.93 / range 0.92–0.94 across all six pairwise comparisons of the four blobbing runs). Downstream cross-arm perception ranking deferred to the geometric strand (O010). Blob-overlap artefacts (gitignored) regenerate on demand via `scripts/perception/diagnostics/blob_overlap_6799.py` (recipe + provenance in DECISIONS O003).
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

**March-bag geometric strand and the control strand are both CLOSED; April is mid-flight.** Current work: finish April's in-row analyses → non-in-row branch → April control strand. **June/July/September are BLOCKED on O019.** The March-strand commit history below is retained for provenance.

*(historical)* **March-bag geometric strand is CLOSED and pooled to whole-bag** (Commits 2a/2b/3: pipeline consolidated + bag-parametrised, findings pooled additively, D041 frame accounting). Remaining March-strand commits, in order:
1. **Commit 4 (this)** — STATUS + per-month template + O012 figures scope (both in-row + non-in-row) + terminology consistency sweep (driven-path, O015).
2. **Commit 5 (O013) — DONE** — `dataset_split.py` → `frame_manifest_build.py`; whole-bag `split="eligible"` marker + Δs=1.5 m subsample recompute (285→284; per-model CI shift GT1 ≤ 0.022 m / GT2 ≤ 0.082°, non-headline); `val_test_split_summary.json`→`manifest_summary.json`; samples → `superseded/`.
3. **Commit 6 (O014) — DONE** — non-in-row: `--scope` added; 5,841 category-C frames → `final/non_in_row_evaluation/`; F020 (spurious `two_row` ~48–52%, 76–80% on turns) + F021 (driven-path error ~0.40 m / ~6°, ~2× in-row, IPM-invalid).
4. **Commit 7 (O016) — DONE** — mitigation demonstration: `mitigation_analysis.py` → `final/mitigation_evaluation/`; F022 (odometry runtime state gate — rejects 98.4% of the spurious non-in-row `two_row` at 1.2% in-row FP, arm-invariant; oracle upper bound 100%/0%) + F023 (perception-only geometry filter — ~38–41% rejection at ~3% in-row FP, turn-blind by construction; combined ∪ 98.6% at ~4%). Framing: characterised + mitigated with **measured effectiveness**, not solved.
5. **Commit 8 (O017) — DONE** — in-row **abstention** characterisation: `single_row_analysis.py` → `final/march_evaluation/single_row_analysis.json`; **F024** (pipeline classifies `single_row` on 12.8–13.9% of in-row frames and emits no centreline; second row detected in ~96%, dominant rejection `too_few_near_seed` 67.7–73.2% — fewer than 2 detections within the 5 m near-seed window (D037 requires ≥2 to seed; a count criterion, not "all beyond"); abstention not failure; conservatism evidence-based not context-based, contrasts F020; D-G tier-2 half-spacing extrapolation specified SPEC §10 but never implemented — deferred, reconciled in F024's writeup).
6. **Commit 9 (O012) — DONE** — 15 report figures via `figures.py` (bag-parametrised, self-contained front-end mirroring `line_fit_infer.py`, load-bearing CSV-consistency assertion, `project_ground` inverse D1) → `final/figures/{in_row,non_in_row,mitigation}/`. In-row (6): anatomy 10247, F013 arm-invariance 7397 + paired-CI forest, F017 sensor-common tilt (camera-vs-LiDAR, 10 anchors — summary, not single-frame), F018 Phase-C classes, F024 abstention 13820. Non-in-row (5): F020 stationary/turn/transition + distribution bars, F021 driven_path_error. Mitigation (4): F022 3-up, F023 3-up, turn-blindness 14987, F022∪F023 complementarity. `FIGURE_SPEC.md` finalised (frame-reuse principle; Fig 3 single-frame F017 caveat).
7. **Commit 10 (O018) — DONE** — near-seed window **sensitivity** (F025): `scripts/geometric/one_time/near_seed_sensitivity.py` (infer-once, sweep NEAR over the fit) → `final/march_evaluation/near_seed_sensitivity.json`. The 5 m rule (D037) is conservative but near-optimal — widening to ~6.0 m recovers ~28% of F024 abstentions (+~4 pp coverage) at <5% RMS cost; Opt-A(10%) = 6.5 m (A/C) / 7.0 m (B), Opt-B = 6.0 m; wider is not better (coverage peaks at 6.5 m, RMS rises monotonically, heavy-tail adjacent-row corruption max ~1.5–1.85 m on ~1.8% of fits → needs a D036/F014 adjacency guard). Arm-consistent; 0 CSV-consistency mismatches.

Then the seasonal phase:
5. **Multi-bag seasonal evaluation** — run the locked bag-parametrised pipeline on **April/May (unblocked)**, then **June/July/September (BLOCKED on O019**, D046d — unattributed-scene attribution must be resolved first) (`--bag <month>`); cross-month synthesis (canopy-state generalisation, where a class-structure effect is most likely to appear).
6. **PID / command-level characterisation** — the downstream control strand.
7. **A2 writing** — Methodology (DECISIONS trail + D041 scope framing), Results/Discussion (FINDINGS F010–F019 pooled).

---

## Val/test discipline (geometric strand) — SUPERSEDED by whole-bag pooling (D040)

> **Note (D040, 16 Jul 2026).** The within-March val/test split described below is **superseded**. It served config-lock leakage-control (F018 selected on val, locked before the CP-6 held-out test) and has served its purpose; the March strand is now evaluated whole-bag (all 7,857 in-row frames; D040/D041). Seasonal generalisation is claimed at the multi-bag level — **rule 5 (single-shot test) applies per-month (whole-bag), not within-March**. The split description is retained for historical context.

Same one-shot-test discipline as perception (rule 5):

- **Val** (7 passes p2/4/5/6/7/8/10; 4,708 frames) — all pipeline development, the row-model refinement (CP-4, D036–D038), the 9-model CP-5 evaluation, the paired cross-arm bootstrap, and every methodological analysis (A–I) run here. Val may be inspected freely.
- **Test** (4 passes p0/1/3/9; 3,149 frames) — held out for **one** final evaluation at the locked config (CP-6). Not inspected until then. The labelled 23-scene perception test set is separately untouched.
- **F013 confirmed on val** (statistically + practically indistinguishable at tight autocorrelation-corrected paired-bootstrap bounds); CP-6 is the held-out confirmation, not substituted by the val conclusion.
- **Subsampling rule — supersedes the D-D Δs = 1.5 m default for paired analyses:** the spatial-independence gap or block length for paired-difference CIs is **grounded in per-pair, per-metric measured autocorrelation (Analysis H)**, not a global default. Measured paired-difference decorrelation is 0.22–0.67 m (0.1 threshold), well below the 1.5 m pre-specified default. Δs = 1.5 m remains the conservative fallback where autocorrelation is not measured.
- **PID / command-level characterisation** remains deferred pending full geometric-strand closure (sweep + test).

---

## Multi-bag structure & per-month template

The pipeline is **bag-parametrised** (`--bag march`, `--bag april`, …; Commit 2b, `bag_config.py`). March is the **design / development / evolution bag** (row-model refinement D036–D038, config lock F018, the val/test → pooling methodology) and keeps the evidence-rich structure; **April onward is locked-pipeline application** (one whole-bag run per bag, no per-bag development).

```
results/geometric/
├── march/
│   ├── final/march_evaluation/       (whole-bag pooled — headline)
│   ├── diagnostics/                   (retained — decision-supporting evidence)
│   ├── superseded/                    (retained — dependencies + audit trail)
│   └── (top-level CP-0/1/2/3 artefacts)
├── april/
│   └── final/april_evaluation/        (whole-bag from --bag april)
├── may/, june/, july/, september/     (same as april)
└── cross_month/
    └── final/cross_month_synthesis/   (seasonal generalisation findings)
```

**Multi-bag readiness:**
- **Pipeline** bag-parametrised (`--bag <month>`) — Commit 2b; the 9 model weights are bag-independent (the same three-arm models are evaluated on every bag).
- **Figures module** will be bag-parametrised (O012: `plot_in_row_frame(bag, …)`, `plot_non_in_row_frame(bag, …)`).
- **Per-month `final/{bag}_evaluation/` template** established (Option-1 artefact naming: the path carries the bag, filenames stay bag-agnostic).
- **Cross-month synthesis** structure defined (`cross_month/final/`); seasonal generalisation (rule 5 per-month) is where a class-structure effect is most likely to appear (canopy-state variation).
- Only **March** retains the evidence-rich structure (`diagnostics/`, `superseded/`); April+ carry `final/{bag}_evaluation/` only.

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
- **O010 (RESOLVED — Commits 2a–3):** Geometry pipeline + Phase C downstream sweep — built, run, and pooled to whole-bag; the primary cross-arm comparison is delivered (F013 pooled, RMS lateral error). See DECISIONS O010, D040.
- **O012 (was O011) — RESOLVED (Commit 9):** Report-figure generation — 15 figures, `scripts/geometric/figures.py` (bag-parametrised; self-contained front-end mirroring `line_fit_infer.py`; **load-bearing CSV-consistency assertion** — every per-frame figure recomputes `(cls, offset, heading)` and asserts equality with the committed CSV before plotting, all pass; `project_ground` inverse added to `projection_calibration.py`, D1, round-trip < 1e-13 px, `project_px` untouched) → `results/geometric/march/final/figures/{in_row,non_in_row,mitigation}/` (~24 MB, committed). `FIGURE_SPEC.md` finalised as the as-built contract (frame-reuse design principle; Fig 3 F017 single-frame caveat: the sensor-common near-equal m_L/m_R is a pooled property, not per-frame). Discipline: `line_fit_infer.py` + committed pipeline scripts + all committed findings/artefacts untouched. **Original scope, retained for provenance:** **In-row:** combined view (left panel raw image — blue trunks, yellow poles, red fitted rows, green centreline, optional red-dotted driven-path reference; right panel bird's-eye), representative pipeline-final frames (NOT dev-era 4223/4107/3991), cross-arm comparison, F017 tilt, F018 mechanism. **Non-in-row:** representative frames per category (headland moving, stationary, transitions), same combined-view format, failure modes explicit. **Locked styling module, bag-parametrised** — `plot_in_row_frame(bag, frame_id)`, `plot_non_in_row_frame(bag, frame_id, category)` — for multi-bag reuse. **RMS naming discipline:** `centreline_error_rms` (in-row, headline, comparable) vs `driven_path_error` (non-in-row, degradation characterisation, three conflations noted) — never conflated in a side-by-side "RMS vs RMS". Placeholder dir: `results/geometric/march/final/figures/`.
- **O018 (new — Commit 10, RESOLVED this commit):** Near-seed window **sensitivity analysis** (F025) — a one-time pipeline-design study (→ `scripts/geometric/one_time/`, refactor commit 39b044a). `near_seed_sensitivity.py` (bag-agnostic; base points detected **once** — NEAR is a fitting parameter — then NEAR swept 5.0→10.0 m over the fit alone) → `final/march_evaluation/near_seed_sensitivity.json` (new sibling; no committed artefact modified). **F025** — the D037 5 m near-seed rule is **conservative but near-optimal**: widening to **~6.0 m** recovers **~28%** of the F024 abstentions (+~4 pp two_row coverage, to ~85%) at **<5%** full-set-RMS cost; deployed-system optimum (Opt-A, full-set RMS ≤ 1.10× F013 baseline) = **6.5 m (A/C) / 7.0 m (B)**, marginal optimum (Opt-B) = **6.0 m**, threshold-sensitive (6.0 at 5% / 6.5–7.0 at 10% / 7.5–10 at 15%). **Wider is not better** — coverage peaks at 6.5 m then declines (lost frames → ~6% by 10 m), RMS rises monotonically. Cost is a **heavy-tail adjacent-row corruption** (existing-fit shift max ~1.5–1.85 m on ~1.8% of frames) → a production widen needs a D036/F014 adjacency-rejection guard (future work), plus multi-bag canopy-vs-bare-vine sensitivity. Recovered-frame RMS ~0.27–0.31 m (~30% worse, reported separately); plausibility-fire 3.6–6.3% (bounded flag, not an FP rate); arm-consistent; **NEAR=5 slice reproduces the committed CSV (0 mismatches)**. Discipline: F011/F013/F024 + committed findings/artefacts untouched (F025 cites F011 one-way); `line_fit_infer.py` + pipeline scripts untouched; additive after F024.
- **O013 (RESOLVED — Commit 5):** Manifest cleanup + whole-bag subsample. `dataset_split.py` → `frame_manifest_build.py` (split-free; `split="eligible"`/`"excluded"` marker; whole-bag single-greedy Δs=1.5 m subsample); `val_test_split_summary.json` → `manifest_summary.json`; `dataset_split_samples/` → `superseded/`. Subsample **285 → 284** frames (per-split → whole-bag; 33 differ). Impact confined to the **non-headline per-model CI** in `line_fit_report.json` (re-run for manifest↔report consistency): **GT1 ≤ 0.022 m, GT2 ≤ 0.082°** bound shift, and 7/9 per-model subsample counts changed; per-arm block CIs, F013 paired, F018 config, F017 lidar, and `line_fit_per_frame.csv` all **unchanged**. (Larger than the ≤ 0.03° first estimated — a 12% subsample-set change; documented here for provenance.)
- **O014 (RESOLVED — Commit 6):** Non-in-row characterisation. `--scope eligible|non_in_row` added to `extract_frames.py`/`line_fit_infer.py` (orthogonal to `--bag`); 5,841 category-C frames extracted + inferred → `final/non_in_row_evaluation/`; `non_in_row_analysis.py` → **F020** (the in-row pipeline emits a spurious `two_row` on ~48–52% of headland frames — 76–80% on turns — arm-consistent; does not degrade to `none`) + **F021** (driven-path error ~0.40 m / ~6° on those outputs — ~2× in-row, IPM-invalid degradation characterisation, **not** comparable to in-row RMS). Deployment implication: an in-row/non-in-row state machine (D041). Categories: stationary 3,946 / turn 376 / transition 1,519.
- **O015 (new — Commit 4, RESOLVED this commit):** Terminology consistency sweep — `driven-path` throughout (BLT is autonomous deployment, Polvara 2024 §3.3.3). F016 was fixed in Commit 3; the metric-strand definition (FINDINGS) and the historical DECISIONS references receive the driven-path treatment **additively** in this commit.
- **O016 (new — Commit 7, RESOLVED this commit):** Non-in-row **mitigation demonstration** — a two-layer rejection of the F020/F021 spurious-`two_row` deployment gap, with **measured effectiveness** (not a claim the gap is solved). `mitigation_analysis.py` (bag-agnostic; evaluates all 13,698 in-row + non-in-row two_row outputs) → `final/mitigation_evaluation/mitigation_analysis.json`. **F022** — odometry runtime **state gate** (`speed > 0.10`, `|v_y| > 0.30`, `|heading-rate| < 22°/s` = in-row p99): rejects **98.4%** of the spurious non-in-row `two_row` at **1.2%** in-row FP, **arm-invariant** (odometry-based); oracle upper bound (manifest `eligible` flag) **100% / 0%** by construction; per-category stationary 100% / turn ~95% / transition ~96%. **F023** — perception-only **geometry filter** (in-row p99 thresholds |offset|>0.71 m, |heading|>6.7°, |m_L−m_R|>0.22, n_base<12): only **~38–41%** rejection at **~3%** in-row FP — the low ceiling **is the finding** (most non-in-row failures overlap the in-row distribution; turn-blind by construction — a real row seen mid-manoeuvre). Combined (F022 ∪ F023): **98.6%** non-in-row rejection at **~4%** in-row FP (union budget). Future work (deployment): learned state classifier, sensor fusion, formal state machine with hysteresis. Discipline: F013/in-row + F020/F021 findings untouched; in-row `final/march_evaluation/` + non-in-row `final/non_in_row_evaluation/` artefacts untouched; mitigation artefact isolated in `final/mitigation_evaluation/`.
- **O017 (new — Commit 8, RESOLVED this commit):** In-row **abstention** characterisation. `single_row_analysis.py` (bag-agnostic; class mix from the committed CSV + failing-side rejection reason re-run on all 9,477 single_row model-frames) → `final/march_evaluation/single_row_analysis.json` (new sibling; no committed artefact modified). **F024** — the pipeline classifies `single_row` on **12.8–13.9%** of in-row frames (arm-consistent ≤1.1 pp) and **emits no centreline**; the second row **is detected in ~96%**, dominant rejection **`too_few_near_seed` 67.7–73.2%** (fewer than 2 detections within the 5 m near-seed window — D037 requires ≥2 to seed, a **count** criterion, not "all beyond"; e.g. frame 13820 left has 1 within, 9 beyond); abstention **not** failure (counts in coverage F011, not the centreline metric F013); conservatism **evidence-based not context-based** — contrasts F020's non-in-row confidence; `not_reproduced` = 0/9,477 (deterministic). **D-G tier-2** half-spacing extrapolation (SPEC §10) **specified but never implemented** — deferred (D038 spread collapsed; tier-2 metric conflates prior-bias/projection-narrowing/off-centre without a per-frame reference), reconciliation + adjacent-frame future-work protocol folded into F024's writeup. Discipline: **F011 untouched** (one-way cite F024→F011); no other finding/artefact touched; additive.
- **O019 (new — GATES June / July / September):** Attribute the **90 unattributed SemanticBLT scenes**. 90 of the 230 unique scenes are named `color_image_*` with no month prefix, so their bag of origin is unknown; if any belong to an evaluated bag, CP-0 under-excludes there and that bag is contaminated. Correlation-based matching **cannot decide it** — a controlled probe showed known negatives outranking known positives on raw peak correlation (april-vs-march 0.890 > march-vs-march 0.779) and a peak-margin refinement inverting between bags (D046c). **Required method:** ORB/SIFT keypoint matching + RANSAC geometric verification against each bag's frame stream, with the same known-positive / known-negative controls, so the inlier threshold is read off the data rather than guessed. **Outcome:** any scene confirmed present in a bag is added to that bag's CP-0 exclusion set. **Blocking:** June, July and September may not be evaluated until this resolves — the 90 are canopy (summer-foliage) imagery, so the risk is directional. **Not blocking:** March, April and May (bare-vine / early-growth, prefix-attributed; D046d).
  - **Revised approach (24 Jul 2026) — fold the check into each bag's CP-0, don't run it as a standalone all-bags job.** The ORB/SIFT + RANSAC check is built as a reusable function `(scenes, bag_frames) → per-scene inlier score` and, once validated, becomes a step inside `contamination_census.py` (CP-0): after locating the prefix-attributed scenes it also matches the 90 unattributed scenes against *this* bag's frame stream and adds any confirmed match to the exclusion set. The gate is unchanged — June/July/September still may not be evaluated until this resolves for that bag — but it is satisfied at each bag's natural CP-0 point rather than requiring a separate upfront pass over all six bags.
  - **Method (to be reviewed before implementing):** coarse thumbnail prefilter → top-K candidate frames per scene (recall only); ORB (nfeatures ~3000) + Lowe-ratio match + **RANSAC homography inlier count** as the discriminating statistic; threshold read off the separation between known-positive (prefix scenes vs own bag) and known-negative (vs foreign bag) inlier distributions. Validation-first: if positives don't cleanly separate from negatives the method is rejected too (same discipline that rejected the correlation probe, D046c).
  - **First step — retroactive March/April validation** (both converted, frames on disk): calibrate the threshold on march/april controls, classify the 90 unknowns; expected result is zero matches (canopy scenes, bare-vine bags), turning D046d's *reasoned* directional-risk argument into a *measured* one. Deliverable `scripts/geometric/one_time/scene_attribution_orb.py` → `results/geometric/scene_attribution_keypoint.json`; on success the core function is promoted into CP-0 and a DECISIONS entry is locked (deferred until validated).
  - **RESOLVED for March/April (24 Jul 2026 — D048 LOCKED).** The ORB+RANSAC method validated: it separates true members from cross-session same-place matches where correlation could not. **All 90 unknowns score ≤ 12 inliers on both march and april → confident-absent, measured** (not reasoned). The three-band decision rule is locked in **D048** (≤40 absent / ≥200 present / 40–200 manual review), calibrated on the march/april controls; fine-verify was tested and shown *not* to recover weak same-place-different-pass members, so it is not relied on. Validation trail: `scene_attribution_orb.py` + `scene_attribution_{tail_probe,fineverify}.py` (one_time), `scene_attribution_keypoint.json`, `results/geometric/april/diagnostics/attribution_tail/`. **June/July/September remain gated** — the locked gate runs at each summer bag's CP-0 when it is processed (optional, timeboxed later work). O019 is therefore closed for the evaluated bags and armed for the rest.
  - **WIRED into the pipeline (25 Jul 2026).** The gate is no longer a standalone probe: `prep.py` (CP-0) runs it for every bag via the shared `scene_attribution.py` module, and `prep.py` (CP-1) hard-stops on any `needs_review` scene. (CP-0 census and CP-1 manifest were consolidated into one `prep.py` in the same cleanup batch.) So the summer bags are now *armed at their natural CP-0* — running `prep.py --bag june` scores the 90 unattributed against june and blocks CP-1 if any land in 40–200. Regression: re-running the gate on march reproduces 0 present / 0 needs_review / 90 absent (max 11), adding zero intervals → manifest byte-identical. See D048 "Implemented".
  - **May confirms (26 Jul 2026).** `prep.py --bag may` (CP-0 gate) scores all 90 unattributed scenes ≤ 11 inliers → 0 present / 0 needs_review / 90 absent. The 90 are now measured-absent from **all three evaluated bags (march, april, may)**; the canopy-adjacency risk did not materialise on may. June/July/September remain armed.
  - **⚠️ [SUPERSEDED — DONE 25 Jul 2026: the D048 gate is now wired into `prep.py`; see the WIRED note above. Kept for provenance of the original pre-wiring plan.]** REQUIRED FIRST STEP before processing June / July / September — do NOT run a summer bag's CP-0 until this is done. The D048 gate is validated but **not yet wired into `contamination_census.py`**. That script has an empty-scenes early-return that skips the descriptor bank for bags with no month-prefixed scenes — which is *exactly* June/July/September — so **as the code stands today the attribution gate would silently NOT run for the summer bags** (the ones it exists to protect). Before any summer bag: **(1)** restructure the empty-scenes early-return so the coarse bank builds and the D048 attribution runs for **all** bags (prefix or not); **(2)** wire in the D048 three-band rule (≤40 absent / ≥200 present → exclude / 40–200 → `needs_review`, blocking that bag's evaluation); **(3)** regenerate March/April CP-0 and confirm their **manifest** reproduces byte-identically (the census carries a `generated` timestamp, so the manifest — not the census — is the byte-identical guard). Only then run the summer bag's CP-0. Wiring was deferred deliberately (D048): it is best tested end-to-end on a bag that actually needs it. A code comment at the empty-scenes early-return in `contamination_census.py` also flags this.
- **O020 (new — documentation correction, non-blocking):** The pre-CP-2-era docs — `PROJECT_PLAN.md` (§ metrics table, reproducibility, limitations), `GEOMETRY_PIPELINE_SPEC.md`, `PHASE_C_SPEC.md`, `PID_PIPELINE_SPEC.md`, `CONTROL_DESIGN_INTENT.md`, `DECISIONS.md` — still describe the BLT reference trajectory as **"teleoperator/teleoperated"**. This is a **mischaracterisation**: the platform ran **autonomous GPS/topological navigation** (Polvara et al. 2024 §3.3.3), established by F016 (driven-path framing) and made load-bearing by F027 (the recorded steering is uncorrelated with vision because the platform never steered from vision — R²≈0.007). The new top-level `README.md` uses the corrected term ("autonomous driven-path") and flags the discrepancy in its Honest-notes section. **Required:** an additive-amendment pass over those docs replacing "teleoperator/teleoperated" with the autonomous-driven-path characterisation, following the same additive-preservation pattern as the F017 anchor-selector amendment (correct in place, note the supersession, do not silently rewrite history). Deferred to its own task — out of scope for the multi-bag/README work that surfaced it.

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
