"""Control strand — one-command driver (runs Steps 1→4 in dependency order).

Thin orchestrator: shells out to the four control-strand scripts in the correct
order, so the run-order footgun (gain_kfold before command_generator → placeholder
gains, PID_PIPELINE_SPEC) cannot happen. Each script is run UNCHANGED via its own
CLI, so the output is byte-identical to running the steps by hand in order; all
five scripts remain individually runnable and importable.

  python3 scripts/control/control.py --bag april                     # whole strand, in order
  python3 scripts/control/control.py --bag april --only command_smoothness   # re-run one stage

Steps (detail + per-step outputs in this directory's README):
  1 state_gate_native   -> mitigation_evaluation/state_gate_native.json        (F026)
  2 command_generator   -> command_evaluation/command_per_frame.csv + command_summary.json  (F027-A)
  3 gain_kfold          -> command_evaluation/gain_kfold.json                  (F027; needs Step 2)
  4 command_smoothness  -> command_evaluation/command_smoothness.json          (F028; needs Step 2)
`centreline_adapter.py` is a library (imported by Steps 2–3), not a step, so it is
not run here; run it directly only to inspect per-stream counts.
"""
import sys
import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = [
    ("state_gate_native",  "state_gate_native.py"),
    ("command_generator",  "command_generator.py"),
    ("gain_kfold",         "gain_kfold.py"),
    ("command_smoothness", "command_smoothness.py"),
]


def main():
    names = [n for n, _ in STEPS]
    ap = argparse.ArgumentParser(description="Run the control strand end-to-end (Steps 1->4, in order).")
    ap.add_argument("--bag", default="march", help="bag name (default: march)")
    ap.add_argument("--only", help=f"comma-separated subset to run ({', '.join(names)}); default: all, in order")
    a = ap.parse_args()

    want = None
    if a.only:
        want = [s.strip() for s in a.only.split(",")]
        unknown = [s for s in want if s not in names]
        if unknown:
            raise SystemExit(f"unknown step(s) {unknown}; known: {names}")
    run = [(n, s) for n, s in STEPS if want is None or n in want]

    for i, (name, script) in enumerate(run, 1):
        print(f"\n===== control [{a.bag}] {i}/{len(run)}: {name} =====", flush=True)
        r = subprocess.run([sys.executable, str(HERE / script), "--bag", a.bag])
        if r.returncode != 0:
            raise SystemExit(f"control: step '{name}' failed (exit {r.returncode}) — stopping.")
    print(f"\ncontrol [{a.bag}]: done ({len(run)} step{'s' if len(run) != 1 else ''}).")


if __name__ == "__main__":
    main()
