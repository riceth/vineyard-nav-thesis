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


def resolve(bag, scope="eligible"):
    """Return the per-bag path bundle. `scope` selects the evaluation stratum (orthogonal to bag):
    'eligible' = in-row whole-bag (final/{bag}_evaluation/); 'non_in_row' = the D041 category-C
    deployment-gap stratum (final/non_in_row_evaluation/). Raises SystemExit on an unknown bag."""
    if bag not in BAGS:
        raise SystemExit(f"unknown bag '{bag}'; known bags: {sorted(BAGS)}")
    if scope not in ("eligible", "non_in_row"):
        raise SystemExit(f"unknown scope '{scope}'; expected eligible | non_in_row")
    base = PKG / "results" / "geometric" / bag
    out = base / "final" / ("non_in_row_evaluation" if scope == "non_in_row" else f"{bag}_evaluation")
    cache = base / "cache"
    return {
        "bag": bag, "scope": scope,
        "manifest": base / "dataset_manifest.json",
        "frames_dir": BAGS[bag]["frames_dir"],       # shared dir; eligible / non-in-row indices are disjoint
        "db3": BAGS[bag]["db3"],
        "cache_dir": cache,
        "detections": cache / "detections.csv",       # in-row only (config sweep is in-row)
        "out_dir": out,
        "per_frame_csv": out / "line_fit_per_frame.csv",
        "line_fit_report": out / "line_fit_report.json",
        "paired": out / "paired_crossarm.json",
        "config": out / "config_analysis.json",
        "lidar": out / "lidar_crosscheck.json",
        "non_in_row_analysis": out / "non_in_row_analysis.json",
    }


def frames_for_scope(man, scope):
    """Frame-index list for a scope. eligible = in-row (D040); non_in_row = D041 category C
    (headland, non-contaminated) = the deployment-gap stratum."""
    if scope == "non_in_row":
        return [f["i"] for f in man["frames"] if f["headland"] and not f["contaminated"]]
    return [f["i"] for f in man["frames"] if f["eligible"]]


def parse_bag():
    """Standard --bag / --scope CLI for the pipeline scripts; returns the resolved path bundle."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="march", help="bag name (default: march)")
    ap.add_argument("--scope", default="eligible", choices=["eligible", "non_in_row"],
                    help="evaluation stratum (default: eligible = in-row)")
    a = ap.parse_args()
    return resolve(a.bag, a.scope)
