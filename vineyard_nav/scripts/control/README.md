# Control strand — scripts

Command-level (PID) strand: consumes the geometric pipeline's centreline and produces a per-frame
yaw-rate command stream. Design contract: `docs/PID_PIPELINE_SPEC.md`. All four scripts here are
**reusable pipeline components** — there is no `diagnostics/` split in this folder.

**Run every command from `vineyard_nav/`.** All scripts are bag-parametrised (`--bag march`,
`--bag april`, …) and resolve per-bag paths through `scripts/geometric/bag_config.py`.

## Prerequisites

The geometric strand must have run first — this strand reads
`results/geometric/<bag>/final/<bag>_evaluation/line_fit_per_frame.csv` and the frame manifest
`dataset_manifest.json`. See `scripts/geometric/README.md`.

## Reproduce, in order

**Step 1 — Validate the native state gate**
```bash
python3 scripts/control/state_gate_native.py --bag march
```
Re-derives the runtime check that decides whether the robot is actually driving along a row — and
so whether its centreline should be trusted — this time from the robot's own measured velocity
rather than from positions differenced offline, then measures how often it accepts and rejects
correctly.
→ `results/geometric/<bag>/final/mitigation_evaluation/state_gate_native.json`
Exports `load_native_signals` · `fit_forward_floor` · `native_gate`, which Step 2 imports.

**Step 2 — Generate the command stream**
```bash
python3 scripts/control/centreline_adapter.py --bag march     # optional: stream summary only
python3 scripts/control/command_generator.py  --bag march
```
`centreline_adapter.py` is a **library** (9 independent arm × seed streams); running it
directly just prints per-stream frame/abstention counts. `command_generator.py` wires the full
stack — gate → PID → hold-last → ramp — and writes the command stream.
→ `results/geometric/<bag>/final/command_evaluation/command_per_frame.csv`
→ `results/geometric/<bag>/final/command_evaluation/command_summary.json`

**Step 3 — Run the k-fold degeneracy check**
```bash
python3 scripts/control/gain_kfold.py --bag march
```
→ `results/geometric/<bag>/final/command_evaluation/gain_kfold.json`

**Order matters here:** `gain_kfold.py` reads `command_summary.json` and `command_per_frame.csv`
to run its sim-equivalence self-check, so **`command_generator.py` must run first**. Running it
against a stale command stream invalidates that check (it compares two different controllers).

## Notes that affect how you read the outputs

- **Gains are first-principles, not tuned.** `command_generator.derive_gains()` computes Kp/Kψ from
  the locked design point (ζ = 1.0, settling distance 20 m); Kd = Ki = 0. `derive_ramp_rate()`
  likewise derives the slew limit. Changing `ZETA` / `D_SETTLE_M` changes all of them.
- `gain_kfold.py` is **evidence that the tracking objective is degenerate, not a tuning step** —
  it does not produce usable gains, by design.
- Both `command_per_frame.csv` outputs are produced with and without the ramp layer
  (`omega_cmd` vs `omega_cmd_ramp`), so the smoothing can be measured rather than silently
  absorbing perception jitter.

## Re-running on another bag

Add the bag to `BAGS` in `scripts/geometric/bag_config.py`, build its frame manifest, then run the
three steps above with `--bag <name>`. Thresholds (`V_MIN`, `ω_max`) are re-fit per bag from that
bag's own data.
