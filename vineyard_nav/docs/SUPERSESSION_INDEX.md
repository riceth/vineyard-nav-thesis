# SUPERSESSION_INDEX.md — what still stands

**Generated** by `scripts/build_supersession_index.py`. Do not edit by hand; edit the source
entry and regenerate. Regenerate after any change to `DECISIONS.md` or `FINDINGS.md`.

---

## The reading contract

`DECISIONS.md` and `FINDINGS.md` are **append-only**. When a position changed, the original text
was left in place and a correction was added beneath it. Superseded text is therefore still
present, still fluent, and still reads as current. It is not.

**Rules, in order of precedence:**

1. **Where an entry carries a correction, amendment, or revision block, that block supersedes
   the text above it.** Cite the corrected position, never the original.
2. **Never cite the body of a `DEAD` entry as a current position.** These entries are retained
   as an audit trail — they record what was believed and why it was abandoned. They are
   legitimate material for a *narrative* of how the design evolved, and illegitimate as evidence
   for what the design *is*.
3. **For `PARTIAL`, the caveat travels with the claim.** Quoting the surviving part without the
   withdrawal is a misrepresentation.
4. **Silence is not currency.** An entry absent from this index has no recorded amendment; that
   is not a warranty that it is correct, only that nothing has contradicted it.

**Using the history well.** The supersessions are not embarrassments to be hidden. A design that
visibly corrected itself under evidence — a refuted hypothesis, a rejected metric, a tightened
guard — is stronger evidence of rigour than one that never changed. Cite the *arc* ("the initial
attribution was refuted by an independent cross-check, and the finding was rewritten") in the
discussion; cite only the *endpoint* in the results.

| state | count | how to use it |
|---|---|---|
| CURRENT | 50 | cite freely |
| AMENDED | 29 | cite the correction block, not the original text |
| PARTIAL | 2 | cite only with the withdrawal attached |
| DEAD | 10 | never cite as current; history only |

---

## DECISIONS.md — 21 of 59 entries carry a status

### DEAD · D002 — U-Net implementation: scratch — SUPERSEDED by D022 (2 Jul 2026)  <sub>(line 22)</sub>

- L24 — Original status: LOCKED
- L25 — Original decision: Scratch-implemented 4-level Ronneberger U-Net, no pretraining.
- L26 — Why superseded: Post-supervisor-feedback three-arm redesign made U-Net binary one of three arms rather than the primary model. The "scratch for educa…

### DEAD · D003 — No pretraining — SUPERSEDED by D022 (2 Jul 2026)  <sub>(line 30)</sub>

- L32 — Original status: LOCKED
- L33 — Original decision: Phase A and Phase B U-Net trained from random initialisation on SemanticBLT.
- L34 — Why superseded: See D002. Pretraining now used (D022).

### AMENDED · D006 — Label collapsing for binary arms (Phases A and B)  <sub>(line 54)</sub>

- L60 — Correction (8 August 2026, additive — the text above is unchanged, and the decision it records stands). The phrase describing A ↔ B as isolating arch…

### DEAD · D008 — Scratch-training fallback rule — SUPERSEDED by D022 (2 Jul 2026)  <sub>(line 72)</sub>

- L74 — Original status: LOCKED
- L75 — Original decision: Pre-committed mIoU thresholds (0.45 / 0.30 / 0.40) for fallback to SMP+ImageNet under scratch training.
- L76 — Why superseded: Scratch U-Net replaced with SMP+ImageNet from day one (D022). Fallback rule no longer needed.

### AMENDED · D009 — Phase A loss function  <sub>(line 80)</sub>

- L85 — Clarification (3 July 2026, not a supersede): "Generalised" in PHASE_A_SPEC §6 means the multiclass generalisation of soft Dice with equal per-class …

### DEAD · D010 — Multiclass loss — SUPERSEDED by D023 (2 Jul 2026)  <sub>(line 89)</sub>

- L91 — Original status: PROVISIONAL
- L92 — Original decision: CrossEntropy + GeneralisedDice with capped class weights for multiclass U-Net.
- L93 — Why superseded: Multiclass arm moved from U-Net to YOLOv11-seg (D021). Ultralytics handles loss internally (D023).

### DEAD · D012 — Data splits — SUPERSEDED by D024 (2 Jul 2026)  <sub>(line 111)</sub>

- L113 — Original status: LOCKED
- L114 — Original decision: Use Roboflow's existing splits (966/46/23).
- L115 — Why superseded: Supervisor feedback — 23 test frames insufficient. Resplit to 70/20/10 stratified (D024).

### AMENDED · D014 — Three-strand evaluation framework  <sub>(line 128)</sub>

- L139 — Amendment (19 July 2026, additive — the original D014 text above is unchanged). D014's "teleoperator commands" / "teleoperator trajectory" language i…

### AMENDED · D016 — Reproducibility setup  <sub>(line 154)</sub>

- L165 — Clarification (3 July 2026, not a supersede) — concrete measures for bitwise reproducibility, found while validating the Phase A smoke run: `cudnn.de…

### DEAD · D017 — Class-aware downstream (Phase B) — SUPERSEDED by D026 (2 Jul 2026)  <sub>(line 174)</sub>

- L176 — Original status: PROVISIONAL
- L177 — Original decision: Trunk-only RANSAC by default; fall back to combined trunk+pole when trunk pixel count < T. T ∈ {50, 100, 200, 400, 800, 1600} pixe…
- L178 — Why superseded: Multiclass moved from Phase B to Phase C after three-arm redesign (D021). Sweep expanded to include Config A (trunk primary), Config …

### AMENDED · D021 — Three-arm design  <sub>(line 206)</sub>

- L216 — Correction (8 August 2026, additive — the text above is unchanged, and the decision it records stands). The phrase describing A ↔ B as isolating arch…

### DEAD · D024 — 70/20/10 stratified resplit with augmentation-leakage guard — SUPERSEDED by D028 (3 Jul 2026)  <sub>(line 247)</sub>

- L249 — Status: SUPERSEDED by D028 (3 Jul 2026)
- L251 — Why superseded: Written before the Roboflow export was inspected. It assumed the 70/20/10 target could be expressed in *image* counts (725/207/103) w…
- **retires:** D012

### AMENDED · D025 — YOLO multiclass: trunk + pole only  <sub>(line 265)</sub>

- L270 — Supplementary experiment: All-6-classes multiclass kept as optional Phase C.2 if time permits. Would test whether richer supervision transfers back t…

### AMENDED · D030 — Phase B conf-threshold selected on validation  <sub>(line 355)</sub>

- L367 — Supplementary median-based analysis (8 July 2026, not a supersede): `scripts/perception/diagnostics/median_conf_sweep.py` computed both mean and medi…

### AMENDED · D034 — Geometric-strand image→world projection (CP-2) + D-G two-value half-spacing prior  <sub>(line 449)</sub>

- L458 — Update (D036–D038, 13 July 2026). The ~22 % narrowing measured here was largely adjacent-corridor + far-field-fan contamination of the row fit (the C…

### DEAD · D035 — Geometric-strand locked pipeline + GT-2 heading redefinition (CP-3)  <sub>(line 466)</sub>

- L468 — Status: SUPERSEDED (13 July 2026) by D036 (hybrid clustering + RANSAC), D037 (far-field extension), D038 (line-fit centreline). The CP-3 artefacts (`…
- L469 — Superseded because: (1) the near-field 5 m cutoff excluded valid same-row detections — on frame 4107, 6 of the 8 left-row dots lie at X > 5 m, leavin…
- L482 — Canopy blob check done (27 Jul 2026 — F007 geometric-stream audit). The "future work" flagged above (does the blob pathology manifest on the leafy ba…

### AMENDED · D042 — PID state-gate signal source: native bag twist (supersedes F022's pose-finite-difference *for the control strand*)  <sub>(line 608)</sub>

- L626 — Amendment (20 July 2026, additive — the D042 decision above stands; this corrects the gate's *predicate*, per the CP-P1 result F026). The native gate…

### AMENDED · D048 — O019 resolution: ORB+RANSAC scene→bag attribution, and its three-band decision rule  <sub>(line 810)</sub>

- L841 — Amendment (27 July 2026, additive — two-stage rule promoted on june; the coarse three-band gate above is unchanged). O019 deferred promoting the fine…

### AMENDED · D049 — Deterministic CUDA/cuDNN preload guard (reproducibility)  <sub>(line 851)</sub>

- L865 — Amendment (27 July 2026, additive). Two follow-ups from the june run, both reproducibility-neutral (no numerics change; rule 7 intact):

### AMENDED · D050 — July bag excluded from evaluation: stop-start recording defeats the contiguous-pass detector  <sub>(line 869)</sub>

- L889 — Amendment (29 July 2026, additive — the decision is unchanged; the evidence is upgraded). D050 asserted that "the recording itself is sound" on the s…
- L893 — The fault is a degraded localisation publisher, not a sensor. `/robot_pose` publishes at 12.75 Hz but carries only 0.56 Hz of new content — 95.6% of …

### PARTIAL · D052 — July/August 2023 bags adopted for the geometric strand (control strand not run)  <sub>(line 923)</sub>

- L929 — ⚠️ Superseded in part (31 July 2026) — august2023 is withdrawn by D054. That session's camera recorded no imagery: all 8,916 frames are one byte-iden…
- L949 — Reporting scope for july2023 (29 July 2026, from the Stage-C run — supersedes this entry's earlier expectations). CP-1 was healthy: 6,595 eligible fr…
- L955 — Correction to this entry's earlier expectation. july2023 does not extend F029's canopy characterisation from n = 2 to n = 4. Its coverage is under ha…

**DECISIONS.md entries with no recorded amendment (cite freely):** 
`D001`, `D004`, `D005`, `D007`, `D011`, `D013`, `D015`, `D018`, `D019`, `D020`, `D022`, `D023`, `D026`, `D027`, `D028`, `D029`, `D031`, `D032`, `D033`, `D036`, `D037`, `D038`, `D039`, `D040`, `D041`, `D043`, `D044`, `D045`, `D046`, `D047`, `D051`, `D053`, `D054`, `D055`, `D056`, `D057`, `D058`, `D059`

---

## FINDINGS.md — 20 of 32 entries carry a status

### AMENDED · F005 — Rasterised fg IoU is a per-arm characterisation metric, not a cross-arm ranking metric  <sub>(line 193)</sub>

- L196 — Status: REVISED. Original F005 framed rasterised fg IoU as a "cross-arm comparability metric." This framing is retracted; F005 is now scoped to per-a…
- L207 — Revised scope. Rasterised fg IoU is retained as an *internal per-arm characterisation metric*, used for:

### AMENDED · F007 — Phase B best.pt exhibits a large false-positive canopy mask on 6799 not present in last.pt  <sub>(line 276)</sub>

- L406 — Amendment (27 July 2026, additive — geometric-stream blob audit; the original 6799 finding stands unchanged). F007 above is a perception-evaluation r…

### AMENDED · F009 — Phase A vs Phase B training-run variance contrast (Phase B intermittent-blob-driven)  <sub>(line 453)</sub>

- L476 — Revised interpretation. Phase A produces continuously varying detection quality across seeds; the SD 0.008 reflects small training-run randomness in …

### AMENDED · F011 — Far-field extension (D037) rescues ~20 pp of two-row coverage with zero loss, arm-independently  <sub>(line 541)</sub>

- L547 — Confirmed on a second bag (april; `results/geometric/april/final/april_evaluation/line_fit_report.json`). Whole-bag two-row coverage A 77.4 ± 0.4 / B…

### DEAD · F015 — Front-camera yaw offset (~2.2–2.3°) not captured in the published extrinsics  <sub>(line 592)</sub>

- L594 — STATUS: SUPERSEDED (kept, not deleted). The camera-yaw attribution proposed in F015 was refuted by independent LiDAR cross-check (F017). Both the Zed…

### AMENDED · F013 — Cross-arm indistinguishability (GT-1) val→test→pooled; a sub-noise-floor GT-2 offset that surfaces on pooling  <sub>(line 652)</sub>

- L662 — Confirmed on a second bag (april; 8,889 in-row frames, ~6,200 both-two-row per pair; block lengths re-derived on april — L_GT1 = 11, L_GT2 = 28, vs m…
- L664 — Confirmed on a third bag (may; canopy season; 26 Jul 2026; 8,840 in-row frames, ~4,600–4,740 both-two-row per pair; block lengths re-derived on may).…
- L678 — Amendment (23 July 2026, additive — april second-bag test; F013's march numbers and conclusions above stand). The sub-noise-floor GT-2 offset that po…

### AMENDED · F017 — Sensor-common ~2.3–3.8° base_link-to-row tilt (mechanism open)  <sub>(line 727)</sub>

- L753 — Whole-bag cross-check (val→test→pooled). The sensor-common tilt was measured across three anchor sets, all consistent in the operative claim (LiDAR n…
- L755 — Amendment (23 July 2026, additive — anchor-selection bug fix; F017's conclusion is unchanged and strengthened). The whole-bag anchor selector contain…

### AMENDED · F018 — Phase C downstream config sweep + single-class ablations: trunks load-bearing; poles supplement coverage, not quality  <sub>(line 785)</sub>

- L805 — Confirmed on a second bag (april; `results/geometric/april/final/april_evaluation/config_analysis.json`). The mechanism reproduces in full and class-…

### DEAD · F019 — CP-6 held-out test: GT-1 indistinguishable (confirms F013); GT-2 a negligible-but-detectable B-vs-others micro-difference  <sub>(line 833)</sub>

- L835 — STATUS: SUPERSEDED (16 July 2026, D040) — kept as historical trail. The March val/test split was pooled into a single whole-bag evaluation (D040); F0…

### AMENDED · F020 — Non-in-row output distribution: the in-row pipeline invents a centreline on ~half of headland frames  <sub>(line 861)</sub>

- L867 — Confirmed on a second bag — with one claim narrowed (april; 15,156 non-in-row frames, 9 models; `results/geometric/april/final/non_in_row_evaluation/…

### AMENDED · F021 — Driven-path error on non-in-row two_row outputs: ~2× the in-row error, IPM-invalid (a degradation characterisation)  <sub>(line 896)</sub>

- L902 — Confirmed on a second bag (april; `results/geometric/april/final/non_in_row_evaluation/non_in_row_analysis.json`). The driven-path error reproduces: …

### PARTIAL · F022 — Runtime state gate: odometry-based rejection recovers ~98% of the deployment gap  <sub>(line 921)</sub>

- L923 — Superseded in part (9 August 2026, additive — the geometric-strand finding below is unchanged and stands). For the control strand only, D042 replaces…
- L929 — Confirmed on a second bag — arm-invariance holds *per category*, overall closure is lower (april; `mitigation_analysis.json`). F022's causal rejectio…

### AMENDED · F023 — Geometry-confidence filter: a perception-only fallback, blind to clean-geometry turns  <sub>(line 956)</sub>

- L966 — Confirmed on a second bag (april). F023's perception-only ceiling reproduces — non-in-row rejection ~27–43 % (arm-varying because it operates on the …
- L968 — Amendment (26 July 2026, additive — may third-bag; F023's march/april numbers stand). The fixed geometry-filter thresholds do not transfer cleanly to…

### AMENDED · F024 — In-row abstention: the pipeline declines a centreline on ~13% of in-row frames; its conservatism is evidence-based, not context-based  <sub>(line 989)</sub>

- L995 — Confirmed on a second bag (april; 12,376 single_row model-frames across 9 models; `results/geometric/april/final/april_evaluation/single_row_analysis…

### AMENDED · F025 — Near-seed window sensitivity: the 5 m rule is conservative but near-optimal; ~6 m is a measured, bounded refinement  <sub>(line 1024)</sub>

- L1036 — Confirmed on a second bag (april; `results/geometric/april/final/april_evaluation/near_seed_sensitivity.json`; NEAR=5 slice reproduces the committed …
- L1058 — Amendment (27 July 2026, additive — may canopy near-seed sensitivity; F025's march numbers stand). The near-seed optimum widens on canopy, but wideni…

### AMENDED · F026 — Native-twist state gate: F022 transfers to the deployable onboard signal, and collapses to a single forward-speed predicate  <sub>(line 1064)</sub>

- L1074 — Confirmed on a second bag — and F026 predicted its own failure mode before this data existed (april; `results/geometric/april/final/mitigation_evalua…
- L1099 — Amendment (20 July 2026, additive — CP-P3 pre-flight; F026's numbers and conclusions above stand). The odom-vs-IMU disagreement reported above is now…

### AMENDED · F027 — The executed yaw-rate is unpredictable from the centreline: the open-loop tracking objective is degenerate, and the strand pivots to fixed principled gains  <sub>(line 1101)</sub>

- L1112 — Confirmed on a second bag — the degeneracy is a property of the BLT navigation mode, not of march (april; `results/geometric/april/final/command_eval…

### AMENDED · F027-A (continuation) — P-4c: first-principles gain derivation, locked values, and the gentle-reference corollary  <sub>(line 1138)</sub>

- L1157 — Confirmed on a second bag — the design point holds; the gains are re-derived per bag, not carried (april; `results/geometric/april/final/command_eval…

### AMENDED · F028 — Command smoothness is arm-indistinguishable; the hold-last cost is real but bounded, and the ramp layer dominates it  <sub>(line 1182)</sub>

- L1195 — Confirmed on a second bag — the convergent null reproduces (april; `results/geometric/april/final/command_evaluation/command_smoothness.json`). Comma…

### AMENDED · F029 — Canopy season: the class-structure mechanism is season-invariant; what canopy removes is base-point availability, and with it deployability  <sub>(line 1223)</sub>

- L1257 — Canopy deployment gap (28 July 2026, additive; in-row scope above unchanged). The same base-point starvation that costs in-row coverage also suppress…

**FINDINGS.md entries with no recorded amendment (cite freely):** 
`F001`, `F002`, `F003`, `F004`, `F006`, `F008`, `F010`, `F014`, `F012`, `F016`, `F030`, `F031`

