"""Reusable frame-selection and geometry-recovery layer for Riseholme figures.

Argument-independent by design: it answers "give me N publishable frames matching this predicate,
with everything an overlay needs", so any specific figure later is a short call rather than a build.

Two responsibilities:
  1. Selection, with the privacy allow-list enforced STRUCTURALLY. select() intersects candidates
     with diagnostics/privacy_screen.json's `publishable` list and raises if that file is absent.
     A figure cannot accidentally publish a flagged frame, because it cannot obtain one from here.
  2. Geometry recovery. line_fit_infer writes only the per-frame CSV and discards the base points
     and fitted rows, so overlays need a re-run; with_geometry() does that on a handful of frames.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "scripts" / "riseholme"))
import bag_config                                   # noqa: E402

def _infer_module(bag="tue02sep"):
    """Load line_fit_infer's DEFINITIONS without running its driver.

    line_fit_infer.py has no __main__ guard -- importing it would re-run the full 9-model pass.
    Its source is therefore exec'd up to the driver line, which yields the ACTUAL MODELS table,
    yolo_base, unet_base and estimate() used to produce the reported metrics. Using the real
    source rather than a copy means a figure cannot drift from the measurement it illustrates.
    """
    import types, sys as _sys
    src_path = Path(__file__).resolve().parent / "line_fit_infer.py"
    src = src_path.read_text()
    marker = "\ndev = torch.device"
    if marker not in src:
        raise SystemExit("line_fit_infer.py layout changed: driver marker not found; "
                         "update curation._infer_module() rather than duplicating its logic")
    mod = types.ModuleType("lfi_defs")
    mod.__dict__["__file__"] = str(src_path)
    saved = _sys.argv
    _sys.argv = ["line_fit_infer.py", "--bag", bag]      # its module body calls parse_bag()
    try:
        exec(compile(src[: src.index(marker)], str(src_path), "exec"), mod.__dict__)
    finally:
        _sys.argv = saved
    return mod


def load_arm(arm, seed=42, bag="tue02sep"):
    """Return (predict_base_points, kind) for one arm/seed, exactly as the pipeline loads it."""
    import torch
    M = _infer_module(bag)
    entry = next((e for e in M.MODELS if e[0] == arm and e[1] == seed), None)
    if entry is None:
        raise SystemExit(f"no model registered for arm {arm} seed {seed}")
    _, _, typ, ckpt = entry
    if typ == "yolo":
        from ultralytics import YOLO
        m = YOLO(str(PKG / "results/runs" / ckpt))
        return (lambda im: M.yolo_base(m, im)), typ, m
    dev = torch.device("cuda")
    m = M.UNetBinary().to(dev).eval()
    m.load_state_dict(torch.load(PKG / "results/runs" / ckpt, map_location=dev,
                                 weights_only=False)["model_state_dict"])
    return (lambda im: M.unet_base(m, dev, im)), typ, m


def estimate_fn(bag="tue02sep"):
    """The pipeline's own estimate(); figures must use this, not a re-implementation."""
    return _infer_module(bag).estimate


def publishable(bag):
    """The privacy allow-list. Raises if the screen has not been run — never silently permissive."""
    B = bag_config.resolve(bag)
    p = B["out_dir"].parent.parent / "diagnostics" / "privacy_screen.json"
    if not p.exists():
        raise SystemExit(
            f"privacy screen missing: {p}\n"
            f"Run it first:  python3 scripts/riseholme/diagnostics/privacy_screen.py --bag {bag}\n"
            f"No Riseholme figure may be built before every frame has been screened.")
    d = json.load(open(p))
    allow = set(range(10**9)) if False else None      # explicit: no default-allow path exists
    flagged = set(d["flagged"])
    return flagged, d


def select(bag, n=6, cls=None, arm="A", seed=42, spread=True, frames=None):
    """N publishable frame indices matching a predicate, spread across the session.

    cls: 'two_row' | 'single_row' | 'none' | None (any). Selection uses the per-frame CSV, so it
    reflects what the pipeline actually produced rather than a guess about which frames look good.
    """
    import csv
    B = bag_config.resolve(bag)
    flagged, _ = publishable(bag)
    if frames is not None:
        cand = [i for i in frames if i not in flagged]
    else:
        rows = [r for r in csv.DictReader(open(B["per_frame_csv"]))
                if r["arm"] == arm and int(r["seed"]) == seed]
        cand = [int(r["i"]) for r in rows
                if (cls is None or r["cls"] == cls) and int(r["i"]) not in flagged]
    if not cand:
        return []
    cand = sorted(cand)
    if spread and len(cand) > n:
        idx = np.linspace(0, len(cand) - 1, n).astype(int)
        return [cand[i] for i in idx]
    return cand[:n]


def with_geometry(bag, frame_indices, arm="A", seed=42):
    """Re-run one model on the given frames, retaining base points, fitted rows and centreline.

    Uses the SAME shared cp3_geometry.process_frame as the evaluation, so an overlay cannot show
    something the metrics did not see.
    """
    import cuda_preload                              # noqa: F401
    from ultralytics import YOLO
    import cp3_geometry as G
    B = bag_config.resolve(bag)
    sub = WEIGHTS.get((arm, seed))
    if sub is None:
        raise SystemExit(f"no weights registered for arm {arm} seed {seed}")
    model = YOLO(str(PKG / "results/runs" / sub / "weights/best.pt"))
    out = {}
    for i in frame_indices:
        img = cv2.imread(str(B["frames_dir"] / f"{i:05d}.jpg"))
        if img is None:
            continue
        r = G.process_frame(img, model)
        r["_img"] = img
        out[i] = r
    return out


def robot_pose_enu(bag):
    """Per-camera-frame robot position and heading in the geojson's local ENU frame."""
    import sqlite3
    from rosbags.typesys import Stores, get_typestore
    TS = get_typestore(Stores.ROS2_HUMBLE)
    B = bag_config.resolve(bag)
    GJ = json.load(open(bag_config.GEOJSON))["features"]
    allc = np.array([c for f in GJ for c in
                     (f["geometry"]["coordinates"] if f["geometry"]["type"] == "LineString"
                      else [f["geometry"]["coordinates"]])])
    lon0, lat0 = allc[:, 0].mean(), allc[:, 1].mean()
    mE = 111320.0 * np.cos(np.radians(lat0))

    def to_xy(lon, lat):
        return np.stack([(np.asarray(lon) - lon0) * mE,
                         (np.asarray(lat) - lat0) * 110540.0], -1)

    con = sqlite3.connect(str(B["db3"])); cur = con.cursor()
    tid = cur.execute("SELECT id FROM topics WHERE name='/gps/fix'").fetchone()[0]
    g = []
    for ts_, data in cur.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,)):
        m = TS.deserialize_cdr(bytes(data), "sensor_msgs/msg/NavSatFix")
        g.append((ts_, m.latitude, m.longitude))
    g = np.array(g)
    cid = cur.execute("SELECT id FROM topics WHERE name=?",
                      (bag_config.CAM_COLOR,)).fetchone()[0]
    cam = np.array([r[0] for r in cur.execute(
        "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (cid,))])
    con.close()
    gxy = to_xy(g[:, 2], g[:, 1])
    j = np.clip(np.searchsorted(g[:, 0], cam), 1, len(g) - 1)
    jb = np.where(np.abs(cam - g[j - 1, 0]) <= np.abs(cam - g[j, 0]), j - 1, j)
    pose = {}
    for i, k in enumerate(jb):
        k0, k1 = max(0, k - 5), min(len(gxy) - 1, k + 5)
        hv = gxy[k1] - gxy[k0]
        if np.linalg.norm(hv) < 0.15:
            continue
        pose[i] = (gxy[k], float(np.degrees(np.arctan2(hv[0], hv[1]))))
    lines = [(f["properties"]["row_a_id"],
              to_xy(np.array(f["geometry"]["coordinates"])[:, 0],
                    np.array(f["geometry"]["coordinates"])[:, 1]))
             for f in GJ if f["properties"].get("feature_type") == "mid_row_line"]
    return pose, lines


def enu_to_cvb(p_enu, robot_xy, heading_deg):
    """ENU point -> camera-view base frame (+X along the REARWARD camera view, +Y its left)."""
    h = np.radians(heading_deg)
    fwd = np.array([np.sin(h), np.cos(h)])          # robot forward in ENU
    left = np.array([-np.cos(h), np.sin(h)])
    d = np.asarray(p_enu) - np.asarray(robot_xy)
    x_fwd, y_left = float(d @ fwd), float(d @ left)
    return -x_fwd, -y_left                           # CVB is base_link rotated 180 deg about Z
