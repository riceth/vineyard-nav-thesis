#!/usr/bin/env python3
"""CP-1 image extraction (GEOMETRY_PIPELINE_SPEC.md §9 CP-1).

Reads the CP-1 manifest (frame_manifest_build.py) and, for every ELIGIBLE bag frame, decodes
it from kg_march_23_ros2.db3, resizes to 640x640 with the training STRETCH preprocessing,
and saves a JPEG named by bag frame index. Idempotent — existing frames are skipped.
Also (re)generates a few annotated sample overlays (one per corridor) for the gate.

Frame images go to results/runs/geom_cp1_frames_640/ (GITIGNORED, ~1 GB, not committed);
overlays go to results/geometric/march/superseded/dataset_split_samples/ (small, committed; the
original split-labelled overlays, retained as audit trail). The manifest holds all 16,656
(timestamp, pose, flags) triples; CP-2 consumes only the eligible frames.

Run:  python3 vineyard_nav/scripts/geometric/extract_frames.py
"""
from __future__ import annotations
import sys, sqlite3, json, time, collections
from pathlib import Path
import numpy as np, cv2
from rosbags.typesys import Stores, get_typestore

GIT = Path(__file__).resolve().parents[3]; PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
from bag_config import parse_bag, frames_for_scope
B = parse_bag()
DB3 = B["db3"]
MAN = B["manifest"]
FRAMES = B["frames_dir"]                                    # gitignored; eligible + non-in-row share it
SAMPLES = PKG / "results/geometric/march/superseded/dataset_split_samples"
CAM = "/front/zed_node/rgb/image_rect_color/compressed"
TS = get_typestore(Stores.ROS2_HUMBLE)


def main() -> None:
    FRAMES.mkdir(parents=True, exist_ok=True); SAMPLES.mkdir(parents=True, exist_ok=True)
    man = json.load(open(MAN)); frames = man["frames"]
    sel_ids = set(frames_for_scope(man, B["scope"]))
    elig = [f for f in frames if f["i"] in sel_ids]         # `elig` = the selected scope's frames
    print(f"[{B['bag']}/{B['scope']}] extracting {len(elig)} frames")
    con = sqlite3.connect(str(DB3)); cur = con.cursor()
    tid = cur.execute("SELECT id FROM topics WHERE name=?", (CAM,)).fetchone()[0]
    ids = [r[0] for r in cur.execute("SELECT id FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,))]

    def decode(i):
        data = cur.execute("SELECT data FROM messages WHERE id=?", (ids[i],)).fetchone()[0]
        m = TS.deserialize_cdr(bytes(data), "sensor_msgs/msg/CompressedImage")
        im = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if im.ndim == 3 and im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        return cv2.resize(im, (640, 640))          # STRETCH, matches training

    t0 = time.time(); wrote = skipped = 0
    for f in elig:
        p = FRAMES / f"{f['i']:05d}.jpg"
        if p.exists():
            skipped += 1; continue
        cv2.imwrite(str(p), decode(f["i"]), [cv2.IMWRITE_JPEG_QUALITY, 90]); wrote += 1
        if wrote % 1000 == 0:
            print(f"  extracted {wrote} ({time.time()-t0:.0f}s)", flush=True)
    print(f"frames: wrote {wrote}, skipped {skipped} existing -> {FRAMES.relative_to(GIT)}")

    # (re)generate CP-1 QA overlays: one median frame per corridor (eligible scope only)
    if B["scope"] == "eligible":
        for old in SAMPLES.glob("sample_*.jpg"):
            old.unlink()
        groups = collections.defaultdict(list)
        for f in elig:
            groups[(f["split"], f["corridor"])].append(f)
        for (sp, cor), fs in sorted(groups.items()):
            f = fs[len(fs)//2]; img = decode(f["i"]).copy()
            for k, txt in enumerate([f"frame {f['i']}  t+{f['t_offset_s']:.0f}s  pass {f['pass_id']}",
                                     f"corridor {cor}  {sp}",
                                     f"pose ({f['x']:.1f},{f['y']:.1f})  v {f['speed']:.2f} m/s"]):
                cv2.putText(img, txt, (12, 30+28*k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4)
                cv2.putText(img, txt, (12, 30+28*k), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
            cv2.imwrite(str(SAMPLES / f"sample_{sp}_cor{cor}_f{f['i']}.jpg"), img)
        print(f"overlays: {len(groups)} -> {SAMPLES.relative_to(GIT)} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
