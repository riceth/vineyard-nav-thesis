"""Per-bag path resolution for the whole-bag geometric evaluation pipeline (D040 multi-bag template).

The pooled pipeline scripts (line_fit_infer, line_fit_eval, extract_detections, paired_crossarm,
config_analysis, lidar_crosscheck) are bag-agnostic: they take `--bag <name>` and resolve every
per-bag input/output here, so the same logic runs on march / april / may / ... unchanged. The 9
model weights are bag-INDEPENDENT (the same three-arm models are evaluated on every seasonal bag)
and stay hardcoded in the scripts. Add a bag by adding one BAGS entry.

Path convention (Option 1 artefact naming): results/geometric/{bag}/ holds the bag's manifest,
detection cache and final/{bag}_evaluation/. Artefact filenames are bag-agnostic (the folder path
carries the bag), e.g. results/geometric/march/final/march_evaluation/line_fit_report.json.
"""
import argparse
from pathlib import Path

PKG = Path(__file__).resolve().parents[2]                        # vineyard_nav/

# per-bag inputs that do NOT follow the {bag} convention (extracted-frames dir, ROS2 bag db3)
BAGS = {
    "march": {
        "frames_dir": PKG / "results/runs/geom_cp1_frames_640",
        "db3": Path("/workspaces/dissertation/kg_march_23_ros2/kg_march_23_ros2.db3"),
    },
    # april / may / june / july / september added as each bag's CP-1 manifest is built
}


def resolve(bag):
    """Return the per-bag path bundle. Raises SystemExit on an unknown bag."""
    if bag not in BAGS:
        raise SystemExit(f"unknown bag '{bag}'; known bags: {sorted(BAGS)}")
    base = PKG / "results" / "geometric" / bag
    out = base / "final" / f"{bag}_evaluation"
    cache = base / "cache"
    return {
        "bag": bag,
        "manifest": base / "dataset_manifest.json",
        "frames_dir": BAGS[bag]["frames_dir"],
        "db3": BAGS[bag]["db3"],
        "cache_dir": cache,
        "detections": cache / "detections.csv",
        "out_dir": out,
        "per_frame_csv": out / "line_fit_per_frame.csv",
        "line_fit_report": out / "line_fit_report.json",
        "paired": out / "paired_crossarm.json",
        "config": out / "config_analysis.json",
        "lidar": out / "lidar_crosscheck.json",
    }


def parse_bag():
    """Standard --bag CLI for the pipeline scripts; returns the resolved path bundle."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="march", help="bag name (default: march)")
    return resolve(ap.parse_args().bag)
