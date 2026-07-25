# Control strand — scripts

Command-level (PID) strand: turns the geometric pipeline's centreline into a
per-frame **yaw-rate command** stream, then measures its smoothness across the
three arms. This is the third and final evaluation strand. Design contract:
`docs/PID_PIPELINE_SPEC.md`; design intent: `docs/CONTROL_DESIGN_INTENT.md`.

Five scripts, all reusable pipeline components (no `diagnostics/` split here).
All are **CPU-only** — they post-process the geometric strand's committed
centreline, so this strand needs **no model weights and no GPU**.

---

## Before you start

**Be in the right directory** — every command runs from `vineyard_nav/`:

```bash
cd vineyard_nav
pwd     # must end in /vineyard_nav
```

**What this strand needs** (weights are *not* among them):

| Script | Reads | So you need… |
|---|---|---|
| `state_gate_native.py` | in-row + non-in-row centreline CSVs, and the bag's native twist | the geometric **in-row and non-in-row** branches done, **plus the bag `.db3`** |
| `command_generator.py`, `gain_kfold.py` | in-row centreline CSV, and the bag's twist + IMU yaw-rate | the geometric **in-row** branch done, **plus the bag `.db3`** |
| `command_smoothness.py` | only the command stream this strand produces | **nothing but the repo** once `command_generator.py` has run (or the committed `command_per_frame.csv`) |

So three of the five scripts read the bag `.db3` (for the robot's own measured
twist and IMU — the "what the robot actually did" reference), but **none re-runs
a model**. The geometric non-in-row branch (`scripts/geometric/README.md`,
Stage D) must be complete before `state_gate_native.py`, because F026's
validation measures rejection over the non-in-row frames.

---

## Reproduce

Using April as the example. Every step is CPU-only and a minute or few — the whole
strand is ~5–10 minutes.

**One command (recommended).** `control.py` runs Steps 1→4 below in the correct
order, so the run-order footgun (Step 3 before Step 2 → placeholder gains) cannot
happen:

```bash
python3 scripts/control/control.py --bag april
```
- **Needs:** the geometric **in-row and non-in-row** branches done + the bag `.db3`.
- **Produces:** everything the four steps produce — `state_gate_native.json`,
  `command_per_frame.csv` + `command_summary.json`, `gain_kfold.json`,
  `command_smoothness.json`.
- **`--only <step[,step]>`** runs a subset, e.g. `--only command_smoothness` to
  re-render just the headline after a tweak (step names: `state_gate_native`,
  `command_generator`, `gain_kfold`, `command_smoothness`).

`control.py` is a thin wrapper that shells out to the scripts **unchanged**, so its
output is byte-identical to running the steps by hand. The four steps below are
exactly what it runs, in order — run them individually to re-run or inspect a
single stage.

### The four steps (individually)

**Step 1 — Validate the native state gate → finding F026**
```bash
python3 scripts/control/state_gate_native.py --bag april
```
Re-derives the runtime check that decides whether the robot is actually driving
along a row (and so whether its centreline should be trusted), this time from the
robot's own measured forward velocity rather than positions differenced offline,
then measures how often it accepts/rejects correctly per category.
- **Needs:** the in-row **and** non-in-row centreline CSVs + the bag `.db3`.
- **Produces:** `results/geometric/april/final/mitigation_evaluation/state_gate_native.json`.
- **Runtime:** ~1 minute. Exports `load_native_signals` / `fit_forward_floor` /
  `native_gate`, imported by Steps 2–3.

**Step 2 — Generate the command stream → finding F027-A**
```bash
python3 scripts/control/centreline_adapter.py --bag april     # optional: prints per-stream counts only
python3 scripts/control/command_generator.py  --bag april
```
`centreline_adapter.py` is a **library** (the 9 independent arm × seed streams);
running it directly just prints per-stream frame/abstention counts.
`command_generator.py` wires the full stack — gate → PID → hold-last → ramp — and
writes the command stream with first-principles gains (see Notes).
- **Needs:** the in-row centreline CSV + the bag `.db3`.
- **Produces:** `command_evaluation/command_per_frame.csv` (April: 80,001 rows)
  + `command_evaluation/command_summary.json`.
- **Runtime:** ~1–2 minutes.

**Step 3 — k-fold degeneracy check → finding F027**
```bash
python3 scripts/control/gain_kfold.py --bag april
```
Demonstrates that tuning the gains against the recorded steering is *degenerate*
(the recorded platform never steered from vision) — it is **evidence, not a
tuning step**, and does not produce usable gains.
- **Needs:** the in-row CSV + `.db3`, **and `command_summary.json` from Step 2**.
- **Produces:** `command_evaluation/gain_kfold.json`.
- **Runtime:** ~3–5 minutes.
- **⚠️ Must run *after* `command_generator.py`.** `gain_kfold.py` reads
  `command_summary.json` for its sim-equivalence self-check. Run out of order it
  still completes (it creates its own output directory), but the self-check falls
  back to the CP-P2 *placeholder* gains instead of the locked P-4c gains — a
  weaker check. This was observed during the April run: `gain_kfold.py` was run
  first, reported the placeholder check, and was re-run after Step 2 to get the
  real one. The degeneracy result itself is unaffected (it is a property of the
  objective, not the gains).

**Step 4 — Command-smoothness comparison → finding F028**
```bash
python3 scripts/control/command_smoothness.py --bag april
```
The strand's headline deliverable: the cross-arm comparison of steering
smoothness (RMS frame-to-frame yaw-rate change, with and without the ramp layer).
- **Needs:** only `command_per_frame.csv` from Step 2 (no bag, no `.db3`).
- **Produces:** `command_evaluation/command_smoothness.json`.
- **Runtime:** ~1 minute.

---

## What each output tells you

- **F026** (`state_gate_native.json`) — the deployable onboard signal reproduces
  the odometry state gate; it collapses to a single forward-speed predicate.
- **F027** (`gain_kfold.json`) — the tracking objective is degenerate; the strand
  therefore uses fixed first-principles gains, not tuned ones.
- **F027-A** (`command_summary.json`) — those gains, derived from the design point.
- **F028** (`command_smoothness.json`) — the arms are indistinguishable on command
  smoothness (converging with the geometric strand's F013).

## Notes that affect how you read the outputs

- **Gains are first-principles, not tuned.** `command_generator.derive_gains()`
  computes Kp/Kψ from the locked design point (ζ = 1.0, settling distance 20 m);
  Kd = Ki = 0. `derive_ramp_rate()` likewise derives the slew limit. Changing
  `ZETA` / `D_SETTLE_M` changes all of them. Thresholds (`V_MIN`, `ω_max`) are
  re-fit per bag from that bag's own data.
- **The command stream is written twice**, with and without the ramp layer
  (`omega_cmd` vs `omega_cmd_ramp`), so the smoothing can be measured rather than
  silently absorbing perception jitter (D043).

## Re-running on another bag

Add the bag to `BAGS` in `scripts/geometric/bag_config.py`, run the geometric
strand (in-row **and** non-in-row) for it, then run this strand with
`python3 scripts/control/control.py --bag <name>` (or the four steps individually).
