#!/usr/bin/env python3
"""CP-2 image->ground projection for RISEHOLME. Same module name and same public interface as
scripts/geometric/projection_calibration.py, so cp3_geometry.py imports it byte-identically.

THIS IS THE ONLY FILE IN WHICH THE TWO SITES' GEOMETRY DIFFERS. Everything downstream — row
fitting, consensus sweep, centreline, GT-1/GT-2, the CI estimator — is byte-identical between the
trees and is verified so by verify_algorithm_parity.py. If Ktima and Riseholme results differ, the
cause is the data or this calibration, never the algorithm.

WHAT DIFFERS FROM KTIMA
  camera      Intel RealSense D435I (s/n 050222071152)   vs  Stereolabs ZED2
  native      1280x720                                    vs  1920x1080
  facing      REAR                                        vs  forward
  extrinsics  empirical (see below)                       vs  Polvara 2024 Table 3, tf-validated

THE OUTPUT FRAME — read this before interpreting any number downstream.
The camera looks backward. Returning raw base_link coordinates would put every observed point at
NEGATIVE X, and the shared row model assumes points lie at positive X (NEAR_M, BINS = [(1,3),(3,5)]).
So `project_px` returns coordinates in the CAMERA-VIEW BASE FRAME (CVB):

    origin  = base_link origin (unchanged)
    +X      = the direction the camera looks  (i.e. robot-REARWARD)
    +Y      = left when facing +X
    +Z      = up

CVB is base_link rotated 180 deg about Z. It is still a rigid, robot-fixed frame, so the row model
operates on exactly the kind of input it was written for and needs no modification. The consequence
must be stated wherever Riseholme GT-1 is reported: Riseholme's look-ahead bin measures the
centreline offset 2 m ALONG THE CAMERA'S VIEW, i.e. 2 m BEHIND the robot, where Ktima's measures
2 m ahead. On a straight row these are symmetric; through a turn they are not.

EXTRINSICS PROVENANCE — empirical, not from a robot description.
The camera was never published to this robot's tf tree; confirmed by exhaustive frame_id dumps over
/tf and /tf_static on three bags spanning two sessions (25,063 + 11,241 + 8,611 messages; every
frame enumerated, none matching the camera's declared optical frames). See docs/RISEHOLME.md
sections 4, 12 and 13 for the derivation and the verification log.

  LOCKED  (cross-verified; height agrees to 9 mm across sessions eleven months apart)
      height 1.269 m, pitch 5.75 deg down, roll 0.75 deg
  ASSUMED (NOT measured; every downstream number is conditional on these)
      lateral 0.0 m   - two estimates disagree by 33 mm, which exceeds the 19.5-24.1 mm effect
                        size GT-1 resolves on Ktima
      yaw     0.0 deg residual off straight-rearward - /scan gives +3.21 deg, IQR [+1.89, +5.16],
                        which excludes the collector's stated 0 deg. Unresolved.
      longitudinal 0.0 m - never estimated; no method available here constrains it
"""
from __future__ import annotations
import json, datetime
from pathlib import Path
import numpy as np

# ---- intrinsics (/camera_link_rear/color/camera_info, 1280x720; identical across all RH datasets) ----
W0, H0 = 1280, 720
K = np.array([[908.902, 0.0, 650.331], [0.0, 909.155, 363.993], [0.0, 0.0, 1.0]])
KINV = np.linalg.inv(K)

# ---- extrinsics: CVB <- camera body ----
CAM_HEIGHT_M = 1.269          # LOCKED
CAM_PITCH_DEG = 5.75          # LOCKED, downward
CAM_ROLL_DEG = 0.75           # LOCKED
CAM_LATERAL_M = 0.0           # ASSUMED
CAM_YAW_RESID_DEG = 0.0       # ASSUMED (residual off straight-rearward)
CAM_LONGITUDINAL_M = 0.0      # ASSUMED

EXTRINSICS_LOCKED = ("height", "pitch", "roll")
EXTRINSICS_ASSUMED = ("lateral", "yaw", "longitudinal")
CAMERA_FACES_REAR = True


def _rx(d):
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(d):
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(d):
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# camera body frame is x-forward (along the view), y-left, z-up — the same REP-103 convention the
# Ktima module uses. Positive rotation about +y is nose-down, matching Ktima's PITCH_DEG sign.
# No 180 deg yaw here: CVB is already view-aligned by definition (see the frame note above).
_R_CVB_BODY = _rz(CAM_YAW_RESID_DEG) @ _ry(CAM_PITCH_DEG) @ _rx(CAM_ROLL_DEG)
_R_BODY_OPT = np.array([[0, 0, 1.0], [-1, 0, 0], [0, -1, 0]])      # body <- optical (z-fwd,x-right,y-down)
R_BASE_OPT = _R_CVB_BODY @ _R_BODY_OPT                             # name kept for interface parity

# camera origin in CVB. X and Y flip sign relative to base_link because CVB is rotated 180 deg
# about Z; both are 0 under the current assumptions, so the flip is presently inert but is written
# explicitly so it stays correct if a measured value ever replaces the assumption.
CAM_POS = np.array([-CAM_LONGITUDINAL_M, -CAM_LATERAL_M, CAM_HEIGHT_M])

PITCH_DEG = float(CAM_PITCH_DEG)
NEAR_M = 8.0                    # identical to Ktima: IPM-accuracy cutoff, NOT a tuned parameter


def project_px(u640: float, v640: float, near_m: float = NEAR_M):
    """640x640 (stretched) pixel -> ground (X along camera view, Y left) metres in CVB, or None.

    Identical maths to the Ktima module; only K, W0/H0, R_BASE_OPT and CAM_POS differ."""
    d = R_BASE_OPT @ (KINV @ np.array([u640 * W0 / 640.0, v640 * H0 / 640.0, 1.0]))
    if d[2] >= -1e-6:
        return None
    g = CAM_POS + (-CAM_POS[2] / d[2]) * d
    return np.array([g[0], g[1]]) if (0.0 < g[0] < near_m) else None


def project_ground(X: float, Y: float, Z: float = 0.0):
    """Inverse of project_px, for drawing fits back onto the image. Not part of the metric path."""
    ray_opt = R_BASE_OPT.T @ (np.array([X, Y, Z]) - CAM_POS)
    if ray_opt[2] <= 1e-6:
        return None
    pix = K @ (ray_opt / ray_opt[2])
    return (float(pix[0] * 640.0 / W0), float(pix[1] * 640.0 / H0))


def sensitivity(lateral_m=0.07, yaw_deg=3.2, x_m=2.0):
    """How much the two ASSUMED degrees of freedom move the GT-1 offset at the look-ahead range.

    Reported alongside every Riseholme GT-1 number: a lateral assumption error shifts the
    centreline one-for-one, and a yaw assumption error shifts it by x*tan(yaw)."""
    return {"lookahead_m": x_m,
            "lateral_assumption_error_m": lateral_m,
            "lateral_induced_offset_mm": round(lateral_m * 1000, 1),
            "yaw_assumption_error_deg": yaw_deg,
            "yaw_induced_offset_mm": round(x_m * np.tan(np.radians(yaw_deg)) * 1000, 1),
            "combined_worst_case_mm": round((lateral_m + x_m * np.tan(np.radians(yaw_deg))) * 1000, 1),
            "ktima_effect_size_mm": "19.5-24.1 (A-B / A-C GT-1 on july2023)"}


if __name__ == "__main__":
    print(__doc__.split("\n")[0])
    print(f"  native {W0}x{H0}  fx={K[0,0]} fy={K[1,1]} cx={K[0,2]} cy={K[1,2]}")
    print(f"  LOCKED : height {CAM_HEIGHT_M} m, pitch {CAM_PITCH_DEG} deg down, roll {CAM_ROLL_DEG} deg")
    print(f"  ASSUMED: lateral {CAM_LATERAL_M} m, yaw-residual {CAM_YAW_RESID_DEG} deg, "
          f"longitudinal {CAM_LONGITUDINAL_M} m")
    print(f"  camera faces rear: {CAMERA_FACES_REAR}; +X of the output frame is robot-REARWARD")
    # sanity: image centre and a point low in the frame must land in front of the camera
    for name, (u, v) in [("centre", (320, 320)), ("low", (320, 560)), ("low-left", (120, 560))]:
        g = project_px(u, v)
        print(f"  px {name:9} {str((u,v)):10} -> {None if g is None else np.round(g,3)}")
    print("\n  assumption sensitivity at the 2 m look-ahead:")
    for k, v in sensitivity().items():
        print(f"    {k:32} {v}")
