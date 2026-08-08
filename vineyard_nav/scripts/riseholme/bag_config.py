"""Riseholme configuration: paths, topics, and the camera calibration adopted for this site.

DELIBERATELY SEPARATE FROM scripts/geometric/. The Ktima pipeline is frozen — its five-bag result
is committed and must not regress. Riseholme differs in ways that touch locked geometry (rear-facing
camera, raw Image topic, different intrinsics, 2D scan instead of a 3D cloud), and D046f showed that
extending a single-bag-shaped code path silently breaks it. Nothing here imports from, or is
imported by, scripts/geometric/.
"""
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[2]          # vineyard_nav/
GIT = PKG.parent                                    # repo root; raw bags live here

# --------------------------------------------------------------------------------------------
# Topics — Riseholme publishes raw sensor_msgs/Image, not CompressedImage as Ktima does.
# --------------------------------------------------------------------------------------------
CAM_COLOR = "/camera_link_rear/color/image_raw"
CAM_INFO = "/camera_link_rear/color/camera_info"
CAM_DEPTH = "/camera_link_rear/depth/image_rect_raw"
ODOM = "/odometry/gps"          # map -> base_link, EKF-fused, the densest pose source
GPS_FIX = "/gps/fix"            # WGS84, RTK-fixed on the 2025 sessions
SCAN = "/scan"                  # 2D LaserScan in base_link (no 3D cloud at this site)

# --------------------------------------------------------------------------------------------
# Camera intrinsics — Intel RealSense D435I, serial 050222071152.
# Identical to four decimals across every Riseholme dataset (2024, 2025, 2026), so one set serves.
# --------------------------------------------------------------------------------------------
K_COLOR = np.array([[908.902, 0.0, 650.331],
                    [0.0, 909.155, 363.993],
                    [0.0, 0.0, 1.0]])
COLOR_WH = (1280, 720)
K_DEPTH = np.array([[425.732, 0.0, 426.044],
                    [0.0, 425.732, 238.384],
                    [0.0, 0.0, 1.0]])
DEPTH_WH = (848, 480)

# --------------------------------------------------------------------------------------------
# EXTRINSICS.  Read this block before using any number from it.
#
# The camera was never published to the robot's tf tree — confirmed by exhaustive frame_id dumps
# over /tf and /tf_static on three bags across two sessions (25,063 + 11,241 + 8,611 messages;
# every frame_id enumerated, none matching the camera's own declared optical frames). The values
# below are therefore EMPIRICAL, recovered from the bags' own sensors, not read from a robot
# description. See docs/RISEHOLME.md sections 4 and 12.
# --------------------------------------------------------------------------------------------

# LOCKED — cross-verified on two independent sessions eleven months apart.
CAM_HEIGHT_M = 1.269          # rh_july2026 n=59; part2_2_9_2025 gives 1.278 -> 9 mm agreement
CAM_PITCH_DEG = 5.75          # downward. 58 of 59 samples positive; terrain slope excluded as the
                              # cause (field slopes 3.78 deg, +/-2.93 along-row, which would centre
                              # the distribution on zero and make ~half the samples negative)
CAM_ROLL_DEG = 0.75           # mean of 0.98 (rh_july2026) and 0.45 (part2)

# NOT LOCKED — assumed here so the projection can run at all. Every downstream number is
# CONDITIONAL on these two assumptions and must be reported as such.
#   lateral: two estimates disagree by 33 mm (-0.068 m from /scan, -0.035 m from geojson+GNSS),
#            which exceeds the 19.5-24.1 mm effect size GT-1 resolves on Ktima.
#   yaw:     /scan gives +3.21 deg with IQR [+1.89, +5.16], which excludes the collector's
#            stated 0 deg. Unresolved.
# Both bias GT-1 directly. They are set to the "assume centred" baseline, NOT to a measurement.
CAM_LATERAL_M = 0.0           # ASSUMED (centred). Uncertainty ~ +/-0.07 m
CAM_YAW_DEG = 180.0           # ASSUMED (straight back). Uncertainty ~ +/-3 deg
CAM_LONGITUDINAL_M = 0.0      # ASSUMED — never estimated; no method here constrains it

EXTRINSICS_LOCKED = ("height", "pitch", "roll")
EXTRINSICS_ASSUMED = ("lateral", "yaw", "longitudinal")

# The camera looks BACKWARD along the robot's -x axis. Ktima's forward-facing geometry (near/far
# split at X < 5 m, 2 m look-ahead) assumes +x. Anything reusing that logic must flip sign.
CAMERA_FACES_REAR = True

# --------------------------------------------------------------------------------------------
# Ground truth — the supervisor's surveyed row geometry.
# --------------------------------------------------------------------------------------------
GEOJSON = GIT / "Ground Robot Recordings - Aug 2024" / "riseholme.geojson"
# WayPoint N lies on the mid-row line between row_(g+2) and row_(g+1), g = (N-1)//12.
# Established geometrically (108/108 assigned, 12 per line, none left over) — the geojson and the
# robot's topological map are the same map under different names.
WAYPOINTS_PER_ROW = 12

# The camera under-reads row spacing by ~5% (0.952x on part2, 0.930x on rh_july2026) because it
# measures the canopy envelope while /scan and the geojson measure trunks and posts. Stable across
# sessions, so treated as physical rather than as an error to correct out.
CANOPY_TRUNK_RATIO = 0.94

BAGS = {
    "part2": {
        "src": GIT / "September 2025" / "part2_2_9_2025.bag",
        "ros2_dir": GIT / "rh_part2_ros2",
        "frames_dir": PKG / "results/runs/rh_frames_640_part2",
        "results": PKG / "results/riseholme/part2",
        "session": "2025-09-02",
        "autonomous": False,        # manual driving; O020's autonomous-driven-path framing does
                                    # NOT transfer. Reference must be the geojson line.
        "gnss": "RTK_FIXED",
    },
    "tue02sep": {
        "src": GIT / "September 2025" / "Tue-02-Sep.bag",
        "ros2_dir": GIT / "rh_tue02sep_ros2",
        "frames_dir": PKG / "results/runs/rh_frames_640_tue02sep",
        "results": PKG / "results/riseholme/tue02sep",
        "session": "2025-09-02",
        "autonomous": False,
        "gnss": "RTK_FIXED",
        "note": "trailing index missing (97.5% present) - needs reindex before conversion",
    },
    "july2026": {
        "src": GIT / "rh_july2026.bag",
        "ros2_dir": GIT / "rh_july2026_ros2",
        "frames_dir": PKG / "results/runs/rh_frames_640_july2026",
        "results": PKG / "results/riseholme/july2026",
        "session": "2026-07-17",
        "autonomous": True,         # 29.6% of /auto_mode samples True - the only autonomous RH data
        "gnss": "SBAS",             # ~575 mm; 11x coarser than the 2025 sessions
    },
}

FRAME_PX = 640                 # models were trained at 640x640 stretch; match it exactly
SEEDS = [42, 43, 44]




# --------------------------------------------------------------------------------------------
# resolve() presents the IDENTICAL interface to scripts/geometric/bag_config.resolve(), so the
# shared algorithm files (block_lengths, extract_detections, line_fit_infer, analyze) import this
# module unchanged and cannot diverge. Only the base path and the per-bag inputs differ.
# --------------------------------------------------------------------------------------------
def resolve(bag, scope="eligible"):
    if bag not in BAGS:
        raise SystemExit(f"unknown Riseholme bag '{bag}'; known bags: {sorted(BAGS)}")
    if scope not in ("eligible", "non_in_row"):
        raise SystemExit(f"unknown scope '{scope}'; expected eligible | non_in_row")
    B = BAGS[bag]
    base = PKG / "results" / "riseholme" / bag
    out = base / "final" / ("non_in_row_evaluation" if scope == "non_in_row" else f"{bag}_evaluation")
    cache = base / "cache"
    return {
        "bag": bag, "scope": scope,
        "manifest": base / "dataset_manifest.json",
        "frames_dir": B["frames_dir"],
        "src_bag": B["src"],
        "src_bags": [B["src"]],
        "ros2_dir": B["ros2_dir"],
        "db3": B["ros2_dir"] / f"{B['ros2_dir'].name}.db3",
        "qa_samples": base / "diagnostics/frame_samples",
        "scene_prefix": None,          # no SemanticBLT scenes from this site; CP-0 is vacuous here
        "expected_passes": None,       # no locked pass count for Riseholme
        "census": base / "contamination_census_exclusions.json",
        "manifest_summary": base / "manifest_summary.json",
        "cache_dir": cache,
        "detections": cache / "detections.csv",
        "out_dir": out,
        "per_frame_csv": out / "line_fit_per_frame.csv",
        "line_fit_report": out / "line_fit_report.json",
        "paired": out / "paired_crossarm.json",
        "config": out / "config_analysis.json",
        "lidar": out / "lidar_crosscheck.json",       # never produced: 2D scan only, run with --only
        "non_in_row_analysis": out / "non_in_row_analysis.json",
    }


def frames_for_scope(man, scope):
    """Identical semantics to the Ktima helper of the same name."""
    key = "eligible" if scope == "eligible" else "non_in_row"
    return [f["i"] for f in man["frames"] if f.get(key)]


def parse_bag(default_scope="eligible"):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--scope", default=default_scope, choices=["eligible", "non_in_row"])
    ap.add_argument("--non-in-row", action="store_true")
    a, _ = ap.parse_known_args()
    return resolve(a.bag, "non_in_row" if a.non_in_row else a.scope)


if __name__ == "__main__":
    print("Riseholme configuration")
    print(f"  LOCKED extrinsics : height {CAM_HEIGHT_M} m, pitch {CAM_PITCH_DEG} deg, roll {CAM_ROLL_DEG} deg")
    print(f"  ASSUMED (not measured): lateral {CAM_LATERAL_M} m, yaw {CAM_YAW_DEG} deg, longitudinal {CAM_LONGITUDINAL_M} m")
    print(f"  camera faces rear : {CAMERA_FACES_REAR}")
    for n in BAGS:
        meta, b = BAGS[n], resolve(n)
        print(f"  {n:10} {meta['session']}  autonomous={meta['autonomous']:<5} gnss={meta['gnss']:<9} "
              f"src={'ok' if meta['src'].exists() else 'MISSING':<7} db3={'ok' if b['db3'].exists() else 'not yet'}")
