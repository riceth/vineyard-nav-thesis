"""CP-(-1) ROS1 -> ROS2 bag conversion — the first step of any new bag.

The whole pipeline reads a ROS2 SQLite bag (`.db3`) via `bag_config.resolve()["db3"]`, but the
BLT bags are distributed as ROS1 `.bag` files. This converts one into the ROS2 layout the rest of
the pipeline expects. It is a thin, reproducible wrapper around the `rosbags` package's
`rosbags-convert` (pinned in requirements.txt) — no ROS installation is needed.

  python3 scripts/geometric/convert_bag.py --bag april
    /workspaces/dissertation/kg_april_06.bag  ->  /workspaces/dissertation/kg_april_06_ros2/

Both paths come from `bag_config.BAGS`, so adding a bag there is the only edit needed to convert it.

NOTES
  - This step was previously undocumented: the March conversion existed on disk but no script or
    instruction recorded how it was produced (its metadata.yaml records `ros_distro: rosbags`,
    which is what identified the tool). This script closes that gap so every bag after March is
    reproducible from the downloaded ROS1 file.
  - Conversion is I/O-bound and roughly size-preserving: budget ~1x the source bag in free disk
    (e.g. the 109 GB April bag produces a ~110 GB .db3) and expect tens of minutes.
  - Idempotent-ish: rosbags-convert refuses to overwrite an existing destination, so a completed
    conversion is skipped with a message rather than silently redone.
"""
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
from bag_config import resolve, BAGS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="april", help=f"bag name (known: {sorted(BAGS)})")
    ap.add_argument("--dry-run", action="store_true", help="print the command without running it")
    a = ap.parse_args()
    B = resolve(a.bag)
    src, dst, db3 = B["src_bag"], B["ros2_dir"], B["db3"]

    if not src.exists():
        raise SystemExit(
            f"source ROS1 bag not found: {src}\n"
            f"Download the {a.bag} BLT bag and place it at that path (see the repository README).")
    if db3.exists():
        print(f"[{a.bag}] already converted -> {db3} ({db3.stat().st_size / 1e9:.1f} GB); nothing to do.")
        return

    free = shutil.disk_usage(dst.parent).free
    need = src.stat().st_size
    print(f"[{a.bag}] source {src.name} {need / 1e9:.1f} GB | free {free / 1e9:.1f} GB")
    if free < need * 1.1:
        raise SystemExit(f"not enough free disk: need ~{need * 1.1 / 1e9:.0f} GB, have {free / 1e9:.0f} GB")

    cmd = ["rosbags-convert", "--src", str(src), "--dst", str(dst)]
    print("  " + " ".join(cmd), flush=True)
    if a.dry_run:
        return
    subprocess.run(cmd, check=True)

    if not db3.exists():
        raise SystemExit(f"conversion finished but {db3} is missing — check the rosbags-convert output")
    print(f"[{a.bag}] done -> {db3} ({db3.stat().st_size / 1e9:.1f} GB)")
    print(f"  next: python3 scripts/geometric/prep.py --bag {a.bag}")


if __name__ == "__main__":
    main()
