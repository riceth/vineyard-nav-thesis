# CONTROL_DESIGN_INTENT.md — proposal-era control-pipeline design (provenance)

**Status:** PROVENANCE / historical context — **not** the locked design. This mirrors the 7-layer control-pipeline design from the A1 proposal working notes (`.personal/plan.docx`, proposal-era, outside the repo tree), recorded here so the original design intent is part of the committed record. Each element is annotated with **what the locked control strand actually adopted** (`DECISIONS.md` D042–D044; `PID_PIPELINE_SPEC.md` P-1–P-6) versus what changed or was dropped. **Where this document and the locked decisions conflict, the locked decisions govern.** No claim here is load-bearing for results; the locked spec + findings are.
**Date mirrored:** 20 July 2026 (from the A1 working notes, ~June 2026).

---

## The single biggest divergence: closed-loop → offline open-loop

The proposal described a **closed real-time feedback loop** ("robot moves → camera view changes → new mask → Layer 3 extracts new geometry → Layer 6 issues new commands → self-correcting in real time") emitting `cmd_vel` to the motors. The **locked strand is offline and open-loop** (`PID_PIPELINE_SPEC.md` §0, §11; D014 amendment): the project has **no simulator** and the bag has **no `/cmd_vel`**, so the loop cannot be closed. The controller is run **per frame over the recorded stream**, and its *proposed* yaw-rate is compared against the **executed yaw-rate from the BLT autonomy run** (`/odometry/base_raw.twist.angular.z`; D014 amendment — not "teleoperator commands"). Everything below inherits this reframing.

---

## The 7-layer pipeline (proposal) → what the locked strand adopted

| # | Proposal-era layer (`plan.docx`) | Adopted in the locked strand? |
|---|---|---|
| **1** | **Camera** — physical RGB sensor, raw frames, no code | **Yes** — the bag's front ZED RGB stream (`GEOMETRY_PIPELINE_SPEC.md` §1). No code, as intended. |
| **2** | **Deep multiclass segmentation** — U-Net-style, 7 classes {trunk, pole, pipe, building, robot, vehicle, background}, + a binary baseline | **Yes, evolved** — the three-arm study: Phase A U-Net **binary**, Phase B YOLO binary, Phase C YOLO **multiclass** (trunk/pole only — 2 classes, D025), not 7. This is the perception contribution (F001–F009). |
| **3** | **Row extraction** — per-side clustering + RANSAC + **temporal smoothing** (low-pass / 1-D Kalman) on (θ, offset) | **Partly** — clustering + RANSAC + line-fit centreline **adopted** (D036–D038: hybrid clustering, far-field extension, line-fit @ 2 m). **Temporal smoothing NOT adopted** — the geometry strand is strictly per-frame; the controller consumes the per-frame centreline (P-1). A low-pass/Kalman filter across frames is a possible future addition, not in the locked pipeline. |
| **4** | **Safety layer + perception-health monitor** — clamp `d` against row boundaries − margin; health signal → freeze/slow/stop on degraded | **Not adopted** (offline, no loop to protect — §11). **Partial analogues:** the odometry **state gate** (D042 / F022 / **F026**) is a runtime state check, and **hold-last on abstention** (D043) echoes the "freeze last-good `d` on degraded" idea — but the **row-boundary command clamp** itself is deployment/future work. |
| **5** | **Ramp generator** — rate-limits changes to commanded `d` | **Yes** — the **ramp / rate limiter** of P-6, implemented as a **toggleable layer** with smoothness/jitter reported both with and without it (D043-style dual view). |
| **6** | **PID controller** — error on commanded `d` vs perceived offset; **anti-windup + output clamping**; constant linear velocity + PID angular velocity → `cmd_vel` | **Yes, reframed** — P-2 (weighted-sum PID on offset + heading), P-6 (**conditional-integration anti-windup**, **`ω_max` = executed-yaw-rate p99 output clamp**), P-3 (native `v_x`). But **no `cmd_vel` is published** — the *proposed* yaw-rate is compared to the *executed* yaw-rate (open-loop, D014 amendment). The "commanded `d`" is the centreline (drive-to-centre); a *non-zero* `d` setpoint is **not** adopted (see sub-claim 1). |
| **7** | **Motors** — actuate `cmd_vel`, ROS-handled, no code | **Not applicable** — offline, no actuation, no closed loop (§11). |

---

## Three sub-claims (proposal) → status

1. **Parameterised offset framework** — `d` as a configurable design variable; evaluate at `d = 0` against de Silva (2024) and across non-zero `d`. → **Not adopted in the locked strand.** The controller drives to the centreline (`d = 0` implicitly); the parameterised-`d` framework and the de Silva `d = 0` comparison are a **candidate extension**, not in `PID_PIPELINE_SPEC.md`.
2. **Transition dynamics** — with `d` changing mid-traversal, sweep 3 ramp rates × 2 scenarios; quantify peak transition error, settling time, steady-state error. → **Not adopted as a sweep** (it requires `d`-transitions, hence sub-claim 1). The P-6 ramp/rate-limiter *layer* is implemented and characterised (on/off), but the full transition-dynamics sweep is **future work**.
3. **Failure-mode characterisation** — a taxonomy linking perception quality, `d` magnitude, and tracking error. → **Partly adopted / reframed.** The deployment-gap and abstention findings (F020/F021, **F022/F026** state gate, F024) characterise failure modes, and the **planned P-1 cross-strand check** (do F007's blob-failure seeds degrade PID tracking?) is exactly a *perception-quality → tracking-error* link. The `d`-magnitude axis is absent (no `d`); the perception→tracking axis is planned.

## Foundational result (proposal) → adopted

The **perception comparison** (multiclass vs binary, same data/architecture, evaluated for downstream navigation accuracy) is the study's spine and **is adopted**: the three-arm controlled comparison (F001–F009 perception; F010–F019 geometric; the command-level strand this spec begins). The command-level strand feeds **the same controller structure** across arms (D014), so the cross-arm contrast isolates perception — exactly the foundational-result intent.

## Cross-cutting: structured logging

The proposal's per-layer event logging (safety-clamp activations, ramp shifts, health-state changes, anti-windup engagements) is **reduced to what the offline strand needs**: the **per-frame command CSV** (`PID_PIPELINE_SPEC.md` §6) with provenance flags (`source` fresh/held, `hold_run_len`, `abstain_cls`, `state_ok`) is the strand's structured log. Full multi-layer event logging is a deployment concern.

## Data sources note

The proposal anticipated using the bag's `cmd_vel` for trajectory comparison **"if it contains the original cmd_vel."** It **does not** (confirmed at CP-1 / F026): there is no `/cmd_vel` topic. The evaluation reference is therefore the **executed yaw-rate** (`/odometry/base_raw.twist.angular.z`) and the driven `/robot_pose` trajectory (D014 amendment; `PID_PIPELINE_SPEC.md` §1, §8).

---

## Reference map

- **Locked decisions this doc annotates:** D042 (native-twist state gate + 20 Jul amendment: single `v_x` predicate), D043 (hold-last + dual-metric), D044 (F0xx numbering), D014 (+ amendment); P-1…P-6 (`PID_PIPELINE_SPEC.md` §10).
- **Findings referenced:** F001–F009 (perception arms), F013 (centreline RMS), F017 (sensor-common tilt), F020/F021 (deployment gap), F022 / **F026** (state gate — pose-difference and native), F024 (abstention), F007 (blob-failure seeds — the P-1 cross-strand-check target).
- **Source:** `.personal/plan.docx` (A1 proposal working notes, ~June 2026), mirrored here 20 July 2026. External anchors named in the notes: de Silva (2024), Sivakumar (2021), Barnes (2017), Higuti (2019).
