# PID_PIPELINE_SPEC.md — Command-level (control) strand: offline PID over the pipeline centreline

**Status:** **LOCKED — design approved; CP-P0 sign-off cleared 20 Jul 2026.** No controller code written yet — **held at CP-P0** pending Edosa's sequencing decision (CP-P1 native state-gate re-validation vs CP-P2 adapter + command-generator dry run). All design questions resolved: D042/D043/D044 (deps), D014 (amended), and **P-1** (→1a), **P-2** (→2a), **P-3** (→3b), **P-4** (→4b), **P-5** (→5a), **P-6** (→ all three sub-choices). This spec is the control-strand analogue of `GEOMETRY_PIPELINE_SPEC.md`; it consumes that pipeline's centreline output and does not modify it.
**Date drafted:** 19 July 2026
**Depends on:** D014 (three-strand eval + its 19 Jul 2026 amendment), D042 (native-twist state gate), D043 (hold-last abstention), D044 (F0xx numbering), D037/D038 (centreline definition — offset @ 2 m, slope heading), F013 (in-row centreline RMS ~19 cm), F017 (sensor-common ~+2° tilt), F022 (pose-difference state gate — the re-validation baseline), F024 (13 % in-row abstention), D041 (frame accounting), PROJECT_PLAN §4.3/§9.4 (offline PID, not closed-loop).

---

## 0. Change of approach (why this spec exists)

The command-level strand is the **third** of the D014 three-strand framework (perception → geometric → command-level). The geometric strand (now closed for March, whole-bag, D040/D041) established *how well each arm's perceived centreline agrees with where the robot actually drove*. The command-level strand asks the next question: **when that centreline is fed to a controller, how smooth and how faithful is the resulting steering command, and does that ranking agree with the geometric ranking?** All three arms feed the **same controller structure** (D014), so — as at the geometric strand — the cross-arm contrast isolates perception.

**Scope is offline and open-loop** (PROJECT_PLAN §9.4; confirmed by the repo survey):
- The bag is a fixed recording. There is **no simulator** in the project (no Gazebo/URDF/launch; `rosbags` reader only) and **no `/cmd_vel`** topic, so the loop **cannot be closed** and Ziegler–Nichols auto-tuning (needs closed-loop oscillation) is excluded.
- The controller is run **per frame over the recorded in-row stream**, producing a *proposed* yaw-rate; evaluation compares that proposed command against the **executed yaw-rate from the BLT autonomy run** (`/odometry/base_raw.twist.angular.z`; D014 amendment — *not* "teleoperator commands") and against the geometric centreline error.

This spec is **bag-agnostic in intent** (like the geometry pipeline): the same design applies to April+ bags when their CP-1 manifests exist. March is the design and first-validation bag.

**This document does not authorise implementation yet.** CP-P0 (design sign-off) is **cleared** and the design is LOCKED, but the strand is **held at CP-P0**; controller code begins only after Edosa greenlights the CP-P1/CP-P2 sequencing.

---

## 1. Inputs inventory (measured — see the PID-strand repo survey for full detail)

Everything the controller needs is already on disk or in the bag, joined by **bag frame index `i`** (the common key across the centreline CSV, the manifest, and every bag topic — all 36 topics carry exactly 16,656 messages with **timestamps byte-identical to the RGB camera topic**, so `i` indexes them 1:1; verified).

**(a) Centreline input — the geometric pipeline's per-frame CSV.**
`results/geometric/march/final/march_evaluation/line_fit_per_frame.csv`, written by `scripts/geometric/line_fit_infer.py`. **12 columns:** `arm,seed,i,cls,offset,heading,mL,mR,mc,n_base,adj,flags`.
- `offset` — signed lateral offset = centreline **Y at X = 2 m** look-ahead, **metres**, +Y = left (D038/D-E). Measured range on March in-row two_row frames: **−1.16 … +1.69 m, mean +0.135 m**.
- `heading` — centreline **slope**, **degrees** (arctan of centreline dY/dX). Range **−14.7 … +15.8°, mean +2.14°** (carries the ~2.3° common tilt, F017/D038).
- `cls ∈ {two_row, single_row, none}` — **only `two_row` carries `offset`/`heading`**; `single_row`/`none` are the F024 abstention set.
- **It is not one row per frame.** The CSV is **9 rows per frame** (3 arms × 3 seeds). March: **70,713 rows = 9 × 7,857 in-row frames** (57,449 `two_row` / 9,477 `single_row` / 3,787 `none`). Reducing these 9 rows to one per-frame control signal is **§2 (P-1 → 1a, locked)**.
- Frame-index range on the in-row scope: `i ∈ [810, 15968]`, 7,857 distinct frames.

**(b) State-gate signals — native bag twist (D042).**
`/odometry/base_raw` (`nav_msgs/Odometry`), `twist.twist`: `linear.x` (v_x, forward), `linear.y` (v_y, lateral), `angular.z` (yaw-rate), frame `base_link`. Verified populated during motion (mid-bag: v_x ≈ 0.80 m/s, yaw-rate ≈ ±0.01–0.02 rad/s; ≈ 0 while stationary at bag start). Independent cross-check: `/imu/data.angular_velocity.z`. *(Note: `/front/zed_node/odom.twist` is all-zero even mid-bag — unusable; `/odometry/base_raw` is the source.)*

**(c) Evaluation reference — executed command (no `/cmd_vel` exists).**
- **Executed yaw-rate** = `/odometry/base_raw.twist.twist.angular.z` — the realized body yaw-rate of the BLT autonomy run. This is the open-loop comparison target for a proposed yaw-rate.
- **Driven trajectory** = `/robot_pose` (`geometry_msgs/Pose`, position + orientation) — for any trajectory-level cross-check.
- **`/motor_controller_data`** (`thorvald_base/msg/ControllerArray`) may hold true commanded motor values but is a **custom type not in the standard ROS2 Humble typestore**; deserialising it requires the `thorvald_base` message definitions. Flagged as a possible future reference, not a dependency.

**(d) Per-frame state & accounting — the CP-1 manifest.**
`results/geometric/march/dataset_manifest.json` (built by `scripts/geometric/frame_manifest_build.py`): per frame `i`, `timestamp_ns`, `t_offset_s`, `x`, `y`, `speed` (m/s), `corridor`, `pass_id`, and flags `eligible/inrow/headland/stationary/contaminated/subsample_1p5m`. Supplies the in-row frame set (7,857; D041), the pass/corridor structure (for block-bootstrap CIs), and the Δs = 1.5 m spatial-independence subsample.

**Timing.** Camera/state rate **14.77 Hz**; run **1127.3 s**. Corridor driving speed ~0.6 m/s (Polvara 2024; ~0.68 m/s measured). These set the controller's effective `dt ≈ 1/14.77 s` for any discrete-time term (derivative/integral) evaluated frame-to-frame.

---

## 2. Centreline input adapter — reducing 9 rows/frame to one control signal **[RESOLVED — P-1 → 1a, locked 20 Jul 2026]**

The controller consumes **one** `(offset, heading, valid?)` triple per frame, but the CSV has **9 rows per frame** (3 arms × 3 seeds). How they reduce is a **methodological choice with cross-arm-fairness consequences**, so it is flagged as an open decision rather than picked here. The choice must preserve the study's spine: **the controller structure is identical across arms; only the perception (arm) differs** (D014).

**The fixed constraint (not open):** the reduction is done **within an arm** — the command-level comparison is *A vs B vs C*, exactly as the geometric strand is. Seeds are a robustness axis, not a comparison axis. So the open question is really *"how are the 3 seeds of a given arm combined into that arm's per-frame control input?"*, evaluated once per arm.

Candidate reductions (per arm, per frame `i`):

| Option | How | Pros | Cons |
|---|---|---|---|
| **P-1a — per-seed then aggregate** | Run the controller **independently on each of the 3 seeds**, get 3 command streams per arm, report the arm's metric as **mean ± SD across seeds** (mirrors O009 / the geometric per-arm cross-seed treatment). | Most faithful to how robustness is reported elsewhere; exposes seed variance in the *command*; no averaging of geometry before control. | 3× the controller runs; a frame may be `two_row` for some seeds and abstaining for others → per-seed hold logic (fine, but must be handled per stream). |
| **P-1b — average the geometry, then control** | Per frame, average `offset` and `heading` over the seeds that returned `two_row`; feed one averaged signal to one controller per arm. | One command stream per arm; smooths seed noise before it hits the controller; simplest downstream. | Averaging across seeds is a non-runtime construct (a deployed robot runs one model); mixes a variable number of seeds per frame; can mask seed disagreement. |
| **P-1c — designated primary seed (42)** | Use seed 42 only (the locked/primary seed used elsewhere), treat 43/44 as a sensitivity check reported separately. | Closest to "one deployed model"; cleanest runtime story; single stream. | Discards 2/3 of the trained models for the headline; seed-42-specific quirks could bias the arm's command metric. |

**Open sub-points within P-1** (resolve together with the option): (i) when seeds disagree on `cls` (some `two_row`, some abstaining), does the frame count as valid for the arm? (ii) does the adapter emit **absolute** offset/heading or the **paired cross-arm difference** the geometric strand uses to cancel the ~2.3° tilt (F017/D038)? — the command-level cross-arm story may want the same bias-cancelling paired treatment.

**→ RESOLVED 20 July 2026 — locked to P-1a** (per-seed then aggregate, mean ± SD). This keeps "one model = one command stream" runtime semantics per seed (at 3× controller runs) and — the decisive reason — **preserves the F007 blob-seed signal in the command stream** (averaging geometry first, P-1b, would erase it). Full rationale and the planned perception-pathology → control-quality cross-strand check are recorded in §10 (P-1). The §2 sub-points resolve as consequences of 1a: (i) seed `cls` disagreement needs no special rule — each seed runs as its own stream with its own hold-last logic (§6); (ii) absolute-vs-paired output follows §7(c)'s paired cross-arm treatment, unchanged.

---

## 3. State-gate re-derivation under D042 (native twist)

D042 switches the state gate from F022's pose-finite-difference to native bag twist. F022's **validated numbers do not carry over** (D042 caveat); this section is the plan to re-fit and re-validate them, producing **F026** (D044).

> **Amendment (20 July 2026, additive — CP-P1 result F026; supersedes the three-predicate plan below).** CP-P1 executed this re-derivation. Two corrections to the plan as written: **(1) Frame error** — "native `|v_y|` replaces the finite-difference `v_y`" (below) is a **world-frame-vs-body-frame mistake**: `/odometry/base_raw.twist.linear.y` is body-lateral **slip** (~0.05 m/s in-row), not the world-frame along-row velocity; the correct native analogue of F022's along-row predicate is **forward `v_x`** (`twist.linear.x`). A literal `|v_y| > 0.30` keep-predicate retains only **1.5 %** of in-row frames. **(2) Turn predicate dropped** — on the native signal it adds **zero** marginal non-in-row rejection and only in-row false positives (F026 decomposition), so keeping it "for F022 parity" is not justified. **The locked native gate is a single forward-speed predicate: `native_gate = v_x > V_MIN`**, V_MIN = in-row p1 of `v_x` = **0.30 m/s**. **Validated (F026): 97.5–97.6 % non-in-row rejection at 0.9 % in-row FP** (arm-invariant), vs F022's 98.4 % / 1.2 % on the pose-difference signal (the ~0.8 pp shortfall is the transition category — a body-frame limitation, not tuning). The three-predicate text below is retained as the original plan + derivation context; `scripts/control/state_gate_native.py` (`fit_forward_floor` / `native_gate`) is the locked implementation. Sensor caveat: odom and IMU yaw-rate disagree (F026) — the gate uses neither.

**Signals (native, causal).** From `/odometry/base_raw.twist.twist` at frame `i`: `v_x`, `v_y`, `yaw_rate = angular.z`. Cross-check `yaw_rate_imu = /imu/data.angular_velocity.z`. No centred smoothing (that was the offline, non-causal step in `mitigation_analysis.py`); if any smoothing is applied it must be **causal** (trailing window / one-sided filter) and documented, because the gate must be deployable.

**The gate (same shape as F022, thresholds re-fit).** F022's causal gate was
`gate_pass = (speed > V_MIN) & (|v_y| > VY_INROW) & (heading_rate < HR_THRESH)` with `V_MIN = 0.10 m/s`, `VY_INROW = 0.30 m/s`, `HR_THRESH = in-row p99 = 22.1 deg/s`. Under D042 the **same three-predicate structure** is retained but every threshold is re-fit on the native signal:
- `v_x` (native forward speed) replaces the manifest `speed` (which was itself a pose-difference proxy).
- `|v_y|` (native lateral) replaces the finite-difference `v_y`.
- `yaw_rate` (native `angular.z`) replaces the reconstructed heading-rate. **HR_THRESH re-fit as the in-row p99 of the native `|yaw_rate|`** (the F022 methodology, applied to the new signal). Report the native HR_THRESH next to F022's 22.1 deg/s.

**Validation procedure (mirrors F022's exactly, so the two are comparable).** Reuse the `mitigation_analysis.py` evaluation frame — in-row (D040 eligible) vs non-in-row (D041 category C) — and, on the pipeline's `two_row` outputs, recompute:
- **non-in-row rejection %** per arm (target reference: F022 = 98.4–98.5 %),
- **in-row false-positive %** per arm (target reference: F022 = 1.2 %),
- per category (stationary / turn / transition) as F022 does,
- the **odometry-vs-IMU yaw-rate agreement** (correlation + mean/max discrepancy) as an F017-style sensor cross-check.

**Reporting.** One new finding **F026** (D044) with the native-twist gate's thresholds and rejection/FP rates, presented **alongside F022** (pose-difference) in the same table so the signal-source change is a controlled, one-variable comparison. The gate is **not** used as a controller safety layer until F026 lands. *(F022 is untouched; F026 cites it one-way, mirroring the F024→F011 discipline.)*

**Artefact contract (when built).** `results/geometric/march/final/mitigation_evaluation/state_gate_native.json` (new sibling; the existing `mitigation_analysis.json` is not modified — additive-preservation rule).

---

## 4. Controller architecture (design draft)

Per in-row frame `i`, within a single arm (and per the P-1 reduction once chosen):

1. **Centreline input** (§2) → `(offset_i, heading_i, valid_i)`; `valid_i = (cls == two_row)`.
2. **State gate** (§3, D042) → `state_ok_i`. A gate rejection is a *state* abstention (robot not in a row-following state); it is distinct from an F024 *perception* abstention and is flagged separately in output.
3. **Abstention / hold-last** (§6, D043 + P-5a) → if `¬valid_i` **or** `¬state_ok_i` (a state-gate rejection triggers the same hold, P-5a), reuse the last valid command; else compute a fresh command.
4. **PID error formulation** (§5, **P-2 → 2a, locked**) → map `(offset_i, heading_i)` to a single steering error via the weighted-sum law, then to a **proposed yaw-rate `ω̂_i`**. Under P-2a the command carries **no `v` term**; native `v_x` is replayed only for the dead-reckoning trajectory cross-check (P-3 → 3b, §8).
5. **Output conditioning** (P-6, locked; structure from `CONTROL_DESIGN_INTENT.md` layers 5–6): **output clamping** at `ω_max` = p99 of the executed yaw-rate (F022 p99 precedent); **conditional-integration anti-windup** (freeze the integral on saturation; adds no gain); and a **ramp/rate limiter implemented as a toggleable layer** — smoothness/jitter reported both with and without it (D043-style dual view), to preserve the perception-artefact signal (F007 blobs, hold transitions) a cap would otherwise hide.
6. **Command log** → per-frame record for evaluation (§7).

**Not in this strand:** the proposal-era "safety layer" (`CONTROL_DESIGN_INTENT.md` layer 4) that *clamps commands against detected row boundaries* and the perception-health monitor — those are deployment-time layers; here they would have no closed loop to protect. Noted as future/deployment scope (§11), not built.

**Gain tuning.** Hand-tuned, offline (D014 "PID command smoothness"; PROJECT_PLAN §9.4 excludes Ziegler–Nichols). The tuning method (manual sweep vs a documented heuristic on the recorded error signal) is an open sub-question (§10, P-4); no gains are chosen in this draft.

**Cross-arm invariance (locked).** Whatever §5 formulation and whatever gains are chosen, they are **identical across arms A/B/C** — only the perception input differs. This is the D014 spine and is non-negotiable in the design.

---

## 5. PID error formulation — how offset + heading become one yaw-rate **[RESOLVED — P-2 → 2a, locked 20 Jul 2026]**

The pipeline gives two error signals per frame: **lateral offset** (Y at 2 m, m) and **heading** (centreline slope, deg). How they combine into a single yaw-rate command is **not covered by any existing decision** and is the central open control-design question. Three candidate laws, with tradeoffs; **none is implemented**:

**P-2a — Weighted-sum proportional (+ optional I, D) on a blended error.**
`error_i = offset_i + λ·(L·tan heading_i)` (or simply `Kp·offset + Kψ·heading`); `ω̂_i = −(Kp·offset_i + Kψ·heading_i) [− Kd·d(offset)/dt − Ki·∫offset]`.
- *Pros:* the most literal reading of PROJECT_PLAN §4.3 / `CONTROL_DESIGN_INTENT.md` layer 6 ("real-time mathematics on the error signal"); full PID (I term can absorb the ~2.3° systematic tilt / per-pass bias, F017); two-to-four interpretable gains; trivial to tune by hand.
- *Cons:* the offset **at a 2 m look-ahead already folds in heading** (a look-ahead point moves with slope), so a separate heading term risks **double-counting**; the blend weight `λ`/`Kψ` is physically arbitrary; units mixing (m and deg) needs care.

**P-2b — Stanley-style (heading error + cross-track arctan), speed-adaptive.**
`δ_i = heading_i + arctan(k · cross_track_i / v_x_i)`, then convert steering `δ` to yaw-rate `ω̂_i` for the platform kinematics.
- *Pros:* well-cited, principled path-tracking law (DARPA/Stanley); **speed-adaptive** via `v_x` (we have native `v_x`, D042); cleanly separates a heading term from a cross-track term (no double-count if cross-track is taken at the robot, 0 m, not the 2 m look-ahead); single intuitive gain `k`.
- *Cons:* Stanley is derived for **front-axle Ackermann steering**; Thorvald is skid-/omni-steer, so the `δ → ω̂` mapping needs an explicit, documented kinematic assumption; wants the cross-track **at the robot (0 m)** — the pipeline's primary output is at 2 m (0 m is available as a secondary look-ahead per D-E but must be plumbed through); no integral term for steady-state bias unless added.

**P-2c — Pure-pursuit curvature on the 2 m look-ahead point.**
`κ_i = 2·offset_i / L²` (L = 2 m, the D-E/D038 look-ahead the offset is already defined at); `ω̂_i = v_x_i · κ_i`.
- *Pros:* the **most natural fit to the existing centreline output** — it consumes the 2 m offset *as-is*, no re-plumbing; a **single geometric parameter** (L, already locked at 2 m); heading enters **implicitly** through the look-ahead geometry (no double-count, no blend weight); minimal tuning.
- *Cons:* purely geometric — **no explicit heading term and no integral**, so a steady-state bias (the ~2.3° tilt, per-pass offset) is not rejected unless the "PID" wraps a correction around `κ` (at which point the "PID" acts on a curvature/offset error, which should be stated); sensitive to `offset` noise scaled by `1/L²`.

**Common notes.** (i) All three assume **constant forward velocity** (PROJECT_PLAN §4.3); `v_x` from native twist can either be held constant or replayed from the bag — a sub-question (§10). (ii) Whichever is chosen is applied **identically across arms** (D014). (iii) The choice interacts with P-1 (e.g. P-2b/2c want a clean per-stream `v_x`, which P-1a provides naturally).

**→ RESOLVED 20 July 2026 — locked to P-2a** (weighted-sum PID on offset + heading), matching PROJECT_PLAN §4.3/§9.4 as assessed in A1/A2. Applied identically across arms (D014). Decision + rationale in §10 (P-2). P-2b/P-2c are retained above as alternatives-considered; the double-counting cost of P-2a and its mitigation are written up next.

### 5.1 Methodological note — offset/heading double-counting at a fixed look-ahead (acknowledged limitation + mitigation)

*(Recorded as its own point because it will need explaining in the dissertation and the viva.)*

**The limitation.** The two error signals are **not independent** at a fixed 2 m look-ahead. The offset is the centreline's lateral position *at* X = 2 m, and along a tilted centreline the look-ahead point's lateral position already moves with the slope: to first order `offset(2 m) ≈ offset_at_robot + 2·tan(heading)`. So heading is **already partially embedded** in the offset term. A weighted sum `Kp·offset + Kψ·heading` that treats the two as co-equal, independent errors therefore **double-counts** the heading component — increasingly so as `Kψ` grows relative to `Kp`.

**The mitigation (gain-tuning treatment).** Rather than weighting heading as a second independent position error, **offset is the primary proportional term and heading is weighted as a damping/derivative-like contribution** — `Kψ` kept small relative to `Kp`, so heading serves to *anticipate and damp* the approach to centre (reducing overshoot / oscillation) rather than to *drive* the command as an equal error source. This keeps the physically-meaningful lateral-offset error dominant while still using the heading signal for its stabilising role. The exact `Kψ : Kp` ratio is a P-4 tuning outcome, not fixed here.

**Why P-2a despite this.** It is the formulation assessed in A1/A2 and the most literal reading of PROJECT_PLAN §4.3; it gives a full PID whose optional integral term can absorb the ~2.3° systematic tilt / per-pass bias (F017); and its gains are interpretable and hand-tunable. The double-counting is a bounded, well-understood cost that the gain treatment above manages, and it is stated wherever the command metric is reported.

---

## 6. Abstention / hold-last logic and held-span flagging (D043)

**Runtime behaviour (D043 + P-5a).** The controller emits its **last valid commanded yaw-rate** whenever it cannot produce a fresh command — either because the pipeline abstains (`cls != two_row`, an F024 *perception* abstention) **or** because the state gate rejects the frame (§3, D042; P-5a → hold-last, *not* a separate fallback). No fresh geometry is fabricated. The two triggers share one behaviour but stay separable in the log: a perception hold carries `abstain_cls`; a state hold carries `state_ok = false`.

**Held-span flagging (required for the §7 dual metric).** Every per-frame command record carries an explicit provenance flag so held frames are never silently absorbed:
- `source ∈ {fresh, held}` — `fresh` = command computed from this frame's centreline; `held` = reused last-valid command.
- `hold_run_len` — for a held frame, its position in the current contiguous hold run (1, 2, 3, …), so long holds (a sustained abstention span) are distinguishable from isolated single-frame holds.
- `abstain_cls` — the abstaining class (`single_row` / `none`) on held frames, so F024's dominant `too_few_near_seed` mechanism can be tied to command-level effects.
- `last_valid_i` — the frame index whose command is being reused (age of the held command = `i − last_valid_i`, in frames; ×(1/14.77) s).

These flags let §7 partition the stream into **inclusive** (all in-row frames) and **exclusive** (fresh only) sets without recomputation, and let the writeup quantify *how long* the controller typically coasts on a stale command (F024 reports ~13 % abstention; the command-level question is whether those frames cluster into long coasts or scatter as singletons).

**Artefact contract (when built).** Per-frame command CSV, e.g. `results/geometric/march/final/command_evaluation/command_per_frame.csv` with columns `arm,seed,i,source,hold_run_len,abstain_cls,last_valid_i,offset_used,heading_used,omega_cmd,omega_executed,state_ok`. New directory; no existing artefact modified (additive-preservation).

---

## 7. Metric definition — command-level, dual-mode (D043)

All metrics are computed **twice** (D043): **inclusive** (all in-row frames, held frames counted) and **exclusive** (fresh-command frames only). The gap between the two *is* the reported cost of abstention; they are two views of one finding, not competing findings.

**(a) Tracking fidelity — proposed vs executed yaw-rate (open-loop).** Compare `ω̂_i` (proposed) against `ω_i^exec = /odometry/base_raw.twist.angular.z` (executed BLT-autonomy yaw-rate; D014 amendment). Report **RMS(ω̂ − ω^exec)** (rad/s) and their correlation. *Interpretation caveat (to state wherever reported):* this is **not** a closed-loop tracking error — the executed yaw-rate is what the BLT autonomy actually did, which is itself only an approximation of "ideal" row-following (the same teleoperator/autonomy-centred assumption as GT-1, D-F). It measures *agreement of our command with the driven behaviour*, cross-arm-comparable because all arms share the same reference.

**(b) Command smoothness (the D014 headline for this strand; no external reference needed).** On `ω̂`: **RMS of Δω̂ between consecutive frames** (jerk proxy), **command jitter** (SD of Δω̂), and **saturation rate** (fraction of frames at |ω̂| = ω_max). These are the PHASE_C_SPEC §232 "PID smoothness (RMS yaw-rate diff, jitter, saturation rate)" metrics. Smoothness is where held-last most affects the number (a held command has Δω̂ = 0), which is exactly why the inclusive/exclusive split is mandatory here.

**(c) Cross-arm comparison.** Paired-difference bootstrap CIs (as F001/F003/geometric strand), computed on the **Δs = 1.5 m spatially-independent subsample** (manifest `subsample_1p5m`) with **pass-level moving-block** CIs for whole-bag figures (reuse `block_lengths.py` / the `line_fit_eval.py` machinery). Point estimates over all in-row frames; CIs over the subsample — the D-D dual-mode convention, carried over from the geometric strand. No p-values (D014).

**(d) Consistency with the geometric ranking.** Report whether the command-level cross-arm ranking agrees with the geometric-strand ranking (F013: RMS ~19 cm, arm-indistinguishable) — a *convergent-evidence* check, not a new claim.

---

## 8. Evaluation design

- **Scope:** March in-row frames (7,857; D041), joined by frame index `i` across the centreline CSV, the manifest, and `/odometry/base_raw`. Non-in-row frames are **not** in the command-level metric (the controller is a row-follower; non-in-row is the deployment-gap strand, F020/F021, and the state gate's job to reject).
- **Reference:** executed yaw-rate `/odometry/base_raw.twist.angular.z` (D014 amendment) for the per-frame tracking metric (§7a). **Dead-reckoning trajectory cross-check (complementary; sole consumer of P-3b `v_x`):** integrate the proposed `ω̂` with the replayed native `v_x` into a predicted path and compare against the `/robot_pose` driven path — a secondary, trajectory-level view of the same command.
- **Per-arm, cross-seed:** per the P-1 decision (§2). Report per-arm mean ± SD across seeds alongside per-run CIs (O009 convention).
- **Reference floor / caveat:** the executed yaw-rate is BLT-autonomy behaviour, not ground truth; state this wherever a tracking number is reported (parallels the GT-1 teleoperator-centred caveat, D-F). RTK-GNSS pose floor (March 3.8 cm, Polvara 2024 §5.3) bounds any trajectory-level cross-check, not the yaw-rate metric directly.
- **Bag-agnostic:** same design applies to April+ when their manifests exist; March is design + first validation.

---

## 9. Implementation checkpoints (gates — hold at each)

*(Mirrors the geometry spec's CP discipline. No code until CP-P0's open items are resolved.)*

- **CP-P0 — Design sign-off. [CLEARED 20 Jul 2026]** All design questions resolved — P-1 (→1a), P-2 (→2a), P-3 (→3b), P-4 (→4b), P-5 (→5a), P-6 (→ all three); the spec is LOCKED. **Held here** pending Edosa's sequencing decision — no controller code until CP-P1/CP-P2 is greenlit.
- **CP-P1 — Native state-gate re-validation (D042 → F026).** Build the native-twist gate (§3), re-fit thresholds, reproduce F022's evaluation on the native signal, write `state_gate_native.json` + F026 alongside F022. Gate: F026 reviewed; rejection/FP rates and odom-vs-IMU agreement acceptable.
- **CP-P2 — Centreline adapter + command generator (dry run, one arm/seed).** Implement the chosen P-1 reduction and P-2 law on Phase C seed 42 over all in-row frames; produce `command_per_frame.csv` with the D043 provenance flags. Gate: sanity — commands finite, clamping/anti-windup behave, held spans flagged correctly, `ω̂` vs `ω^exec` visually tracks on a sample pass.
- **CP-P3 — Full command-level evaluation (all arms/seeds, dual-metric).** Run §7 metrics inclusive + exclusive, per arm cross-seed, with paired-difference block-bootstrap CIs. Gate: review before writing the headline findings.
- **CP-P4 — Findings + writeup (F026 …, D044).** Command-level findings into `FINDINGS.md` (main F0xx series), four-part discipline; consistency check vs the geometric ranking (§7d). Gate: strand closure review.

---

## 10. Decision register (control strand)

**Locked — dependencies (from `DECISIONS.md`):**
- **D042 — State-gate signal source. [LOCKED; thresholds/rates TBD → F026]** Native `/odometry/base_raw.twist` + `/imu/data` cross-check, replacing F022's pose-difference *for the control strand*. F022's 98.4 %/1.2 % do not carry over; re-fit + re-validate (§3).
- **D043 — Abstention handling. [LOCKED]** Hold-last-command at runtime; dual-metric (inclusive/exclusive) evaluation; held-span provenance flags (§6).
- **D044 — Findings numbering. [LOCKED]** Continue main `F0xx` in `FINDINGS.md`; no separate file. First expected: F026.
- **D014 (amended 19 Jul 2026)** — reference is "executed yaw-rate from the BLT autonomy run," not "teleoperator commands."

**Locked — this strand's design (resolved 20 July 2026 at Edosa's direction):**

- **P-1 — Centreline input adapter → 1a (per-seed then aggregate). [LOCKED]** Run the controller independently on each of the 3 seeds per arm; report the arm's command-level metric as **mean ± SD across seeds** (mirrors O009 and the geometric per-arm cross-seed treatment, §2). **Rationale (beyond methodological consistency):** per-seed streams **preserve the ability to test whether F007's blob-failure seeds also degrade PID tracking** — Phases B and C each blob on 2/3 seeds (F007/O009), so keeping the seeds separate keeps that perception-pathology signal intact in the command stream. Option 1b (average the geometry across seeds *before* the controller) would **smooth that signal away**, and was rejected for this reason. **Planned cross-strand check (NOT a committed finding yet):** once command-level results exist, test — within an arm — whether the blobbing seeds show worse tracking/smoothness than the clean seed, i.e. a *perception-pathology → control-quality* consequence; it becomes a finding only if/when the data supports it. **Consequences:** (i) seed `cls` disagreement needs no special rule (each seed is its own stream with its own hold-last, §6); (ii) cross-arm output uses §7(c)'s paired-difference treatment.

- **P-2 — PID error formulation → 2a (weighted-sum PID on offset + heading). [LOCKED]** `ω̂_i = −(Kp·offset_i + Kψ·heading_i) [− Kd·d(offset)/dt − Ki·∫offset]`, matching PROJECT_PLAN §4.3/§9.4 (A1/A2). Applied identically across arms (D014). **Acknowledged limitation + mitigation (full note §5.1):** offset (Y @ 2 m) and heading (slope) are not independent at a fixed look-ahead, so a weighted sum **double-counts** heading; mitigated in the gain treatment — **offset primary, heading weighted as a damping/derivative-like term (`Kψ` small vs `Kp`), not a co-equal error** (exact ratio a P-4 outcome). To be written up for the dissertation/viva.

- **P-3 — Forward-velocity handling → 3b (replay native `v_x`). [LOCKED]** Native `/odometry/base_raw.twist.linear.x` per frame. **It feeds the dead-reckoning trajectory cross-check only** (§8): under P-2a the yaw-rate command carries **no `v` term**, so `v_x` never enters the command and there is **no tuning/measurement circularity**. Same native-twist source already adopted for D042 — no extra data cost.

- **P-5 — State-gate → controller coupling → 5a (gate rejection triggers hold-last). [LOCKED]** A state-gate rejection (§3, D042) drives the **same hold-last mechanism** as an F024 perception abstention (D043); the controller holds its last valid command. Matches the strand's original scoping ("the controller ignores the centreline when odometry indicates a non-in-row state") and **avoids a second, separately-justified fallback** alongside D043's. The two hold triggers stay separable in the per-frame log (§6): a perception hold carries `abstain_cls`; a state hold carries `state_ok = false`.

- **P-6 — Output-conditioning parameters → all three sub-choices locked. [LOCKED]**
  - **`ω_max` clamp = p99 of the executed yaw-rate** (`/odometry/base_raw.twist.angular.z`, in-row) — consistent with the p99 precedent set by F022's HR_THRESH.
  - **Anti-windup = conditional integration** (freeze the integral term while the output is saturated) — the simplest well-understood technique, and it **adds no gain** to the P-4 tuning burden.
  - **Ramp / rate limiter = implemented as an available layer, but jitter/smoothness reported BOTH with and without it** — mirroring the D043 inclusive/exclusive pattern for held frames. *Rationale:* F007's blob failures (2/3 seeds) and hold-last transitions both inject command discontinuities that reflect **perception/data artefacts, not genuine steering need**; a cap alone would smooth these away before they are seen. Reporting both **preserves the diagnostic signal** (a possible cross-strand finding — perception pathology propagating into control-layer instability) while still showing what a practically deployable, capped system looks like.

- **P-4 — Gain-tuning: circularity avoidance → 4b (pass-level k-fold cross-validation). [LOCKED]** Tune gains with the **11 CP-1 passes as CV folds** (tune on k−1, score the held-out fold, rotate); report the **pooled out-of-fold** command metric plus **gain stability across folds**. **Tuning objective: minimise RMS(ω̂ − ω^exec)** against the executed yaw-rate (§7a; matches the earlier decision). **Gain-sensitivity reporting is also implemented** — metrics across a small gain grid (the k-fold machinery already evaluates per-gain, so it is a lightweight add-on), showing the cross-arm ranking is stable over the gain range. **Rationale:** circularity risk is **common-mode across arms** (shared gains, D014), so it threatens the *absolute* numbers, not the *cross-arm ranking* (D031) — but k-fold gives a **real held-out answer, not just the argument**, at negligible extra cost (**frame-level computation, not model training**), and **without reviving a permanent named split** anywhere in the dissertation (folds are a transient analysis construct; pass-level for spatial independence). Alternatives considered: 4a (full-pool, argument-only) and 4c (fixed principled gains, untuned) — detail below.

**P-4 — rationale detail & alternatives considered (locked → 4b above):**

#### P-4 — Gain-tuning: avoiding circularity **[RESOLVED → 4b, 20 Jul 2026]**

*Framing (applies to every option).* The strand's headline is the **cross-arm** comparison (D031), and P-2a's gains are **identical across arms** (D014), tuned once — so any optimism from tuning is **common-mode across A/B/C and largely cancels in the ranking**, exactly as the ~2.3° tilt cancels in the paired geometric comparison. Circularity therefore threatens the *absolute* smoothness/tracking numbers, not the *relative* cross-arm finding. P-2a is also a **low-dimensional fit** (Kp, Kψ + optional Ki, Kd = 2–4 scalars); its capacity to overfit 7,857 frames is minimal. None of the options below revives a permanent named val/test split (they stay consistent with the D031/D040 pooling rationale):

- **P-4b — Pass-level k-fold cross-validation (no permanent named split). ✓ CHOSEN (locked — see the entry above).** Use the 11 CP-1 passes as CV folds: tune on k−1 folds, score the held-out fold, rotate; report the pooled **out-of-fold** metric plus **gain stability across folds**. Gives honest held-out numbers *while* every frame still contributes to the reported estimate (out-of-fold), so the pooling rationale is preserved and no named split is revived; gain stability also empirically tests the low-capacity claim. *Cost:* k× tuning passes × arms × seeds, but this is **frame-level computation, not model training** — negligible. Folds are **pass-level** to respect spatial independence (same reason as the block-bootstrap / Δs = 1.5 m subsample).
- **P-4a — Full-pool tuning, explicitly justified (not chosen).** Tune on the whole pooled in-row set and report on it, defended by the parameter-count + characterisation + common-mode-cancellation argument. Consistent with D031/D040, but it rests on the *argument* alone; 4b delivers a real held-out number at negligible extra cost, so 4b was preferred.
- **P-4c — Fixed principled gains, not data-tuned (not chosen).** Sidesteps circularity entirely (nothing is fit to the data), fully arm-independent — but the absolute numbers then characterise only a *reasonable*, not a genuinely-tuned, controller. 4b was preferred to report a tuned system while still controlling circularity.

*Tuning objective (resolved).* Locked to **minimise RMS(ω̂ − ω^exec)** — tracking against the executed yaw-rate (§7a), matching the earlier decision. The smoothness-target alternative is **not** used as the tuning objective (smoothness is still reported, §7b).

*Gain-sensitivity reporting (resolved — implemented alongside 4b).* Metrics are also reported across a **small documented gain grid**, showing the cross-arm ranking is **stable over the gain range**. The k-fold machinery already evaluates the metric per candidate gain, so this reuses that infrastructure as a **lightweight addition**; it strengthens the absolute-vs-relative framing (the ranking is *shown* robust to the specific tuned gains, not merely argued to be).

---

## 11. What is explicitly out of scope here

Closed-loop control and any simulation (no simulator in the project; no `/cmd_vel`; bag is a fixed recording — PROJECT_PLAN §9.4). Ziegler–Nichols auto-tuning (needs closed-loop oscillation). The proposal-era deployment **safety layer** (row-boundary command clamping, `CONTROL_DESIGN_INTENT.md` layer 4) and **perception-health monitor** (no loop to protect offline; deployment-time, future work). Deserialising `/motor_controller_data` (custom `thorvald_base` type; possible future reference, not a dependency). Multi-bag seasonal command-level evaluation (April+ — the design extends to them, but March is the strand built and validated first). Any modification to the geometric-strand pipeline, its artefacts, or the labelled 23-scene perception test set (untouched).

---

## 12. References

- **Internal:** `GEOMETRY_PIPELINE_SPEC.md` (the pipeline this strand consumes; D036–D038 centreline definition, D-D dual-mode CIs, D-E look-ahead); `DECISIONS.md` D014 (+ amendment), D042–D044, D041 (frame accounting); `FINDINGS.md` F013 (centreline RMS), F017 (sensor-common tilt), F022 (pose-difference state gate — F026 baseline), F024 (abstention); `PROJECT_PLAN.md` §4.3, §9.4; `CONTROL_DESIGN_INTENT.md` (proposal-era 7-layer control design mirrored from the A1 working notes — safety layer, ramp generator, anti-windup/clamping, parameterised offset `d`, transition-dynamics sweep — annotated with what the locked strand adopted).
- **External:** Polvara, R., Molina, S., Hroob, I., et al. (2024). "Bacchus Long-Term (BLT) data set." *Journal of Field Robotics*, **41**(7), 2280–2298. DOI: [10.1002/rob.22228](https://doi.org/10.1002/rob.22228). Source for the robot/rig, corridor driving speed (~0.6 m/s), RTK-GNSS floors, and the autonomous-deployment context grounding the D014 amendment. Candidate control-law references (Stanley/pure-pursuit) to be added to the literature review if P-2b/P-2c is chosen.
