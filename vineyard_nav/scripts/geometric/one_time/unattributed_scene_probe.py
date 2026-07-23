"""One-time correctness probe: which bag do the UNATTRIBUTED SemanticBLT scenes belong to?

CP-0 excludes perception-training contamination by locating each labelled scene inside the bag.
It selected scenes by FILENAME PREFIX (`march_*` for the march bag, etc.). That assumption is
unverified for 39% of the dataset: of 230 unique scenes, 90 are named `color_image_*` with no month
at all (GEOMETRY_PIPELINE_SPEC.md §0 calls them "canopy" scenes and asserts they come from "other
sessions not on disk" — an assertion written when only march was on disk, never re-verified).

If any of those 90 actually live in a bag we evaluate, CP-0 has been silently UNDER-excluding and
that bag's results are contaminated. This probe answers that empirically.

  python3 scripts/geometric/one_time/unattributed_scene_probe.py --bags march april
    -> results/geometric/scene_attribution_probe.json   (cross-bag, not per-bag: hence no {bag} dir)

DESIGN — a controlled comparison, so the threshold is read off the data rather than guessed:
  known POSITIVES : month-prefixed scenes vs their own bag   (march_* vs march)
  known NEGATIVES : month-prefixed scenes vs a foreign bag   (march_* vs april)
  UNKNOWNS        : the 90 `color_image_*` scenes vs every bag
If the unknowns' correlations sit in the negative distribution they are absent from that bag; if any
sit in the positive distribution, that scene is present and must be excluded.

METHOD NOTE: this probe uses COARSE-ONLY matching (the CP-0 descriptor bank at stride COARSE, no
fine local search), so its correlations are a slight LOWER BOUND on CP-0's. That is fine and
deliberate — the same bound applies to positives, negatives and unknowns alike, so the SEPARATION
is unaffected. Absolute values here are not exclusion decisions; CP-0 remains the authority.
"""
import sys
import json
import time
import argparse
import sqlite3
import collections
import re
from pathlib import Path

import numpy as np
import cv2
from rosbags.typesys import Stores, get_typestore

GIT = Path(__file__).resolve().parents[4]
PKG = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
from bag_config import resolve, BAGS

DATASET = GIT / "SemanticBLT.v1-2024-june.coco-segmentation"
CAM = "/front/zed_node/rgb/image_rect_color/compressed"
COARSE = 10
TS = get_typestore(Stores.ROS2_HUMBLE)


def _desc(g):
    g = g.astype(np.float32).ravel(); g -= g.mean(); n = np.linalg.norm(g)
    return g / n if n else g


def scene_table():
    """base scene name -> (group, [image paths]). group = march|april|may|unattributed."""
    files = collections.defaultdict(list)
    for split in ("train", "valid", "test"):
        for im in json.load(open(DATASET / split / "_annotations.coco.json"))["images"]:
            base = re.sub(r"_png\.rf\..*", "", im["file_name"])
            files[base].append(str(DATASET / split / im["file_name"]))
    out = {}
    for base, paths in files.items():
        g = next((m for m in ("march", "april", "may", "june", "july", "september")
                  if base.startswith(m)), "unattributed")
        out[base] = (g, paths)
    return out


def bank_for(bag):
    B = resolve(bag)
    if not B["db3"].exists():
        return None, 0
    con = sqlite3.connect(str(B["db3"])); cur = con.cursor()
    tid = cur.execute("SELECT id FROM topics WHERE name=?", (CAM,)).fetchone()[0]
    ids = [r[0] for r in cur.execute(
        "SELECT id FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,))]
    N = len(ids)
    idx = list(range(0, N, COARSE))
    t0 = time.time()
    descs = []
    for k, i in enumerate(idx):
        data = cur.execute("SELECT data FROM messages WHERE id=?", (ids[i],)).fetchone()[0]
        m = TS.deserialize_cdr(bytes(data), "sensor_msgs/msg/CompressedImage")
        im = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_UNCHANGED)
        if im.ndim == 3 and im.shape[2] == 4:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
        descs.append(_desc(cv2.cvtColor(cv2.resize(im, (128, 128)), cv2.COLOR_BGR2GRAY)))
        if k % 500 == 0:
            print(f"    [{bag}] bank {k}/{len(idx)} ({time.time()-t0:.0f}s)", flush=True)
    con.close()
    print(f"    [{bag}] bank {len(idx)} descriptors over {N} frames ({time.time()-t0:.0f}s)", flush=True)
    return np.stack(descs), N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", default=["march", "april"])
    a = ap.parse_args()

    scenes = scene_table()
    groups = collections.Counter(g for g, _ in scenes.values())
    print(f"scenes: {len(scenes)} {dict(groups)}\n")

    # per-scene descriptor set (all augmented versions + horizontal flips), computed once
    sdesc = {}
    for base, (g, paths) in scenes.items():
        ds = []
        for p in paths:
            im = cv2.imread(p)
            if im is None:
                continue
            gr = cv2.cvtColor(cv2.resize(im, (128, 128)), cv2.COLOR_BGR2GRAY)
            ds += [_desc(gr), _desc(cv2.flip(gr, 1))]
        sdesc[base] = np.stack(ds) if ds else None

    report = {"method": ("coarse-only descriptor matching (CP-0 bank at stride %d, no fine search); "
                         "correlations are a consistent lower bound used for SEPARATION, not for "
                         "exclusion decisions" % COARSE),
              "scene_counts": dict(groups), "per_bag": {}}

    for bag in a.bags:
        print(f"[{bag}] building bank ...", flush=True)
        bank, N = bank_for(bag)
        if bank is None:
            print(f"  [{bag}] no .db3 — skipped (convert it first)"); continue
        best = {}
        for base, ds in sdesc.items():
            best[base] = float((ds @ bank.T).max()) if ds is not None else None
        bygroup = collections.defaultdict(list)
        for base, (g, _) in scenes.items():
            if best[base] is not None:
                bygroup[g].append(best[base])
        stats = {}
        for g, v in bygroup.items():
            v = np.array(v)
            stats[g] = {"n": len(v), "min": round(float(v.min()), 3),
                        "p25": round(float(np.percentile(v, 25)), 3),
                        "median": round(float(np.median(v)), 3),
                        "p75": round(float(np.percentile(v, 75)), 3),
                        "max": round(float(v.max()), 3),
                        "n_ge_0.60": int((v >= 0.60).sum()), "n_ge_0.50": int((v >= 0.50).sum()),
                        "n_ge_0.40": int((v >= 0.40).sum())}
        top = sorted(((best[b], b, scenes[b][0]) for b in best if best[b] is not None), reverse=True)[:15]
        report["per_bag"][bag] = {"bag_frames": N, "coarse_bank": int(np.ceil(N / COARSE)),
                                  "by_group": stats,
                                  "top15": [{"corr": round(c, 3), "scene": s, "group": g} for c, s, g in top]}
        print(f"  [{bag}] by group:")
        for g in ("march", "april", "may", "unattributed"):
            if g in stats:
                s = stats[g]
                print(f"    {g:<14} n={s['n']:<4} median={s['median']:<6} max={s['max']:<6} "
                      f">=0.6:{s['n_ge_0.60']:<4} >=0.5:{s['n_ge_0.50']:<4} >=0.4:{s['n_ge_0.40']}")

    OUT = PKG / "results" / "geometric" / "scene_attribution_probe.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT.relative_to(PKG)}")


if __name__ == "__main__":
    main()
