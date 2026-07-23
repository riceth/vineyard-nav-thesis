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
GIT = PKG.parent                                                 # repo root — the raw bags live here

# Per-bag inputs that do NOT follow the {bag} convention. `src_bag` is the downloaded ROS1 bag;
# `ros2_dir` / `db3` are the ROS2 conversion produced by convert_bag.py (the pipeline reads the
# .db3, never the ROS1 bag). `frames_dir` is the extracted-frames cache (gitignored, ~1 GB/bag).
#   scene_prefix    — SemanticBLT filename prefix for this bag's month, used by CP-0 to find the
#                     labelled scenes that must be excluded as perception-training contamination.
#                     None => the dataset contains no labelled scenes from this bag (CP-0 correctly
#                     produces an empty exclusion list rather than failing).
#   expected_passes — optional CP-1 sanity guard on the number of in-row traversals. None => the
#                     count is reported but not asserted (correct for any bag not yet characterised).
#                     It is a REPRODUCTION guard — it catches a silently changed bag file or a
#                     changed threshold constant — NOT a count of physical traversals. A pass is a
#                     maximal run of smoothed |v_y| > VY_INROW with |dy| > PASS_MIN_Y, so a brief dip
#                     under the velocity threshold mid-row splits one traversal into two, and a
#                     partial traversal that still clears PASS_MIN_Y counts as one. Both bags show
#                     this: march's locked 11 contains a split pair (p8/p9, 0.5 s apart), and
#                     april's 12 = 10 full passes + 1 partial (p4: 266 frames, 18 s, followed by a
#                     397 s idle gap) + 1 split pair (p7/p8, divided by a 2-frame / 0.14 s dip under
#                     the threshold while driving straight down the row). The rule behaves
#                     identically on both; the counts are as-built, not idealised. See D046(a).
def _bag(stem, frames, scene_prefix=None, expected_passes=None, qa_samples=None):
    return {"frames_dir": PKG / "results/runs" / frames, "qa_samples": qa_samples,
            "src_bag": GIT / f"{stem}.bag",
            "ros2_dir": GIT / f"{stem}_ros2",
            "db3": GIT / f"{stem}_ros2" / f"{stem}_ros2.db3",
            "scene_prefix": scene_prefix, "expected_passes": expected_passes}


BAGS = {
    # march keeps its original frames-dir name — it is referenced by committed artefacts.
    # march's CP-1 QA overlays are committed under superseded/ (moved there historically) — keep
    # that exact path so the committed artefacts still resolve; new bags use the diagnostics/ convention.
    "march":     _bag("kg_march_23",     "geom_cp1_frames_640",           "march", 11,
                      qa_samples="superseded/dataset_split_samples"),
    "april":     _bag("kg_april_06",     "geom_cp1_frames_640_april",     "april", 12),
    "may":       _bag("kg_may_06",       "geom_cp1_frames_640_may",       "may"),
    "june":      _bag("kg_june_08",      "geom_cp1_frames_640_june",      "june"),
    "july":      _bag("kg_july_13",      "geom_cp1_frames_640_july"),
    "september": _bag("kg_september_09", "geom_cp1_frames_640_september"),
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
        "src_bag": BAGS[bag]["src_bag"],             # downloaded ROS1 bag (input to convert_bag.py)
        "ros2_dir": BAGS[bag]["ros2_dir"],           # ROS2 conversion output dir
        "db3": BAGS[bag]["db3"],
        "qa_samples": base / (BAGS[bag]["qa_samples"] or "diagnostics/frame_samples"),  # CP-1 overlays
        "scene_prefix": BAGS[bag]["scene_prefix"],   # CP-0: SemanticBLT month prefix (None = none labelled)
        "expected_passes": BAGS[bag]["expected_passes"],   # CP-1: optional pass-count guard
        "census": base / "contamination_census_exclusions.json",   # CP-0 output
        "manifest_summary": base / "manifest_summary.json",        # CP-1 output
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
