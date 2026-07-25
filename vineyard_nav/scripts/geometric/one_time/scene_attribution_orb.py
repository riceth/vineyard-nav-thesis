"""O019 validation — ORB + RANSAC scene->bag attribution, with known-positive/negative controls.

SUPERSEDES the correlation probe (`unattributed_scene_probe.py`), which was rejected (D046c) because a
global 128x128 thumbnail descriptor matches generic vineyard-row STRUCTURE, not scene IDENTITY — so
known-foreign scenes scored as high as true members. Keypoint matching with geometric verification
measures identity instead: a true re-observation of the same physical view yields many keypoint
correspondences consistent under a single homography; a different frame of the same vineyard yields
few that survive that geometric constraint.

Pipeline, per (scene, bag):
  1. COARSE PREFILTER (recall only): the CP-0 128x128 zero-mean thumbnail bank shortlists the top-K
     candidate bag frames per scene. Correlation is non-discriminative but not anti-recall — a true
     match still scores high enough to survive the shortlist, so this only avoids ORB-matching every
     scene against the whole stream.
  2. GEOMETRIC VERIFICATION (the decision): ORB (nfeatures=3000) on the scene + each shortlisted
     frame; Lowe-ratio match; RANSAC homography; score = MAX inlier count over the shortlist.

THRESHOLD via controls, same rigor as the rejected probe:
  known POSITIVES  = month-prefixed scenes vs their OWN bag (present)
  known NEGATIVES  = month-prefixed scenes vs a FOREIGN bag (absent)
  UNKNOWNS         = the 90 `color_image_*` scenes vs each bag
The inlier threshold is read off the SEPARATION between the positive and negative distributions.
VALIDATION-FIRST: if positives do not cleanly separate from negatives, this method is rejected too
and that is reported honestly — the hypothesis (geometric inliers separate where correlation did not)
is what the March/April controls test.

  python3 scripts/geometric/one_time/scene_attribution_orb.py --bags march april
    -> results/geometric/scene_attribution_keypoint.json

PRODUCTION HOME: this is the frozen validation harness (controls + calibration) that LOCKED D048.
The gate it validated now runs at every bag's CP-0 via `scripts/geometric/scene_attribution.py`
(same algorithm and constants; kept separate so this validation record stays immutable). Do not
"refactor to share" — that would perturb the committed calibration output.
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
from bag_config import resolve

DATASET = GIT / "SemanticBLT.v1-2024-june.coco-segmentation"
CAM = "/front/zed_node/rgb/image_rect_color/compressed"
TS = get_typestore(Stores.ROS2_HUMBLE)

COARSE = 10            # thumbnail-bank stride (recall prefilter)
SHORTLIST_K = 30       # candidate bag frames verified per scene
MATCH_RES = 640        # both scene + bag frame matched at this resolution (gray)
ORB_N = 3000           # ORB features
LOWE = 0.75            # Lowe ratio test
RANSAC_PX = 5.0        # homography RANSAC reprojection threshold (px)
MIN_MATCH = 8          # need at least this many putative matches to attempt a homography
GROUPS = ("march", "april", "may", "june", "july", "september")

_orb = cv2.ORB_create(nfeatures=ORB_N)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def _thumb(gray128):
    g = gray128.astype(np.float32).ravel(); g -= g.mean(); n = np.linalg.norm(g)
    return g / n if n else g


def scene_table():
    """base scene -> (group, first image path). group = march|…|september|unattributed."""
    files = collections.defaultdict(list)
    for split in ("train", "valid", "test"):
        for im in json.load(open(DATASET / split / "_annotations.coco.json"))["images"]:
            base = re.sub(r"_png\.rf\..*", "", im["file_name"])
            files[base].append(str(DATASET / split / im["file_name"]))
    out = {}
    for base, paths in files.items():
        g = next((m for m in GROUPS if base.startswith(m)), "unattributed")
        out[base] = (g, sorted(paths)[0])          # one representative version per scene
    return out


def orb_of(gray):
    kp, des = _orb.detectAndCompute(gray, None)
    return kp, des


def inliers(kpA, desA, kpB, desB):
    """RANSAC-homography inlier count between two ORB feature sets (0 if too few matches)."""
    if desA is None or desB is None or len(desA) < 2 or len(desB) < 2:
        return 0
    good = []
    for m_n in _bf.knnMatch(desA, desB, k=2):
        if len(m_n) == 2 and m_n[0].distance < LOWE * m_n[1].distance:
            good.append(m_n[0])
    if len(good) < MIN_MATCH:
        return 0
    src = np.float32([kpA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kpB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    return int(mask.sum()) if mask is not None else 0


def bag_stream(bag):
    """(frame ids list, cursor) for the bag's front-camera compressed stream."""
    B = resolve(bag)
    if not B["db3"].exists():
        return None, None
    con = sqlite3.connect(str(B["db3"])); cur = con.cursor()
    tid = cur.execute("SELECT id FROM topics WHERE name=?", (CAM,)).fetchone()[0]
    ids = [r[0] for r in cur.execute(
        "SELECT id FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,))]
    return ids, cur


def decode_gray(cur, msg_id):
    data = cur.execute("SELECT data FROM messages WHERE id=?", (msg_id,)).fetchone()[0]
    m = TS.deserialize_cdr(bytes(data), "sensor_msgs/msg/CompressedImage")
    im = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_UNCHANGED)
    if im.ndim == 3 and im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return cv2.resize(im, (MATCH_RES, MATCH_RES))


def stats(vals):
    v = np.array(vals, float)
    return {"n": len(v), "min": int(v.min()), "p50": float(np.median(v)),
            "p90": float(np.percentile(v, 90)), "max": int(v.max()),
            "mean": round(float(v.mean()), 1)} if len(v) else {"n": 0}


def match_bag(bag, scenes, sdesc, sorb):
    """Return {base: max_inliers} for every scene against this bag."""
    ids, cur = bag_stream(bag)
    if ids is None:
        print(f"  [{bag}] no .db3 — skipped"); return None
    N = len(ids); coarse_idx = list(range(0, N, COARSE))
    t0 = time.time()
    # bank thumbnails are 128x128 (recall prefilter, matches the scene sdesc); decode_gray returns
    # the 640px gray used for ORB, so downsize it to 128 here.
    bank = np.stack([_thumb(cv2.resize(decode_gray(cur, ids[i]), (128, 128))) for i in coarse_idx])
    print(f"  [{bag}] thumbnail bank {len(coarse_idx)} over {N} frames ({time.time()-t0:.0f}s)", flush=True)

    # shortlist candidate coarse-frame positions per scene, then decode+ORB the UNION once
    shortlist = {b: np.argsort(bank @ sdesc[b])[::-1][:SHORTLIST_K] for b in scenes}
    need = sorted({int(p) for sl in shortlist.values() for p in sl})
    forb = {}
    for k, p in enumerate(need):
        forb[p] = orb_of(decode_gray(cur, ids[coarse_idx[p]]))
        if k % 500 == 0:
            print(f"    [{bag}] cand ORB {k}/{len(need)} ({time.time()-t0:.0f}s)", flush=True)
    cur.connection.close()

    best = {}
    for k, b in enumerate(scenes):
        kpS, desS = sorb[b]
        best[b] = max((inliers(kpS, desS, *forb[int(p)]) for p in shortlist[b]), default=0)
        if k % 100 == 0:
            print(f"    [{bag}] verified {k}/{len(scenes)} ({time.time()-t0:.0f}s)", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", default=["march", "april"])
    a = ap.parse_args()

    scenes = scene_table()
    groups = collections.Counter(g for g, _ in scenes.values())
    print(f"scenes: {len(scenes)} {dict(groups)}", flush=True)

    # scene thumbnail (for prefilter) + scene ORB (for verification), computed once
    sdesc, sorb = {}, {}
    for b, (g, path) in scenes.items():
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        sdesc[b] = _thumb(cv2.resize(im, (128, 128)))
        sorb[b] = orb_of(cv2.resize(im, (MATCH_RES, MATCH_RES)))
    scenes_ok = list(sdesc)

    report = {"method": ("ORB(nfeatures=%d)+RANSAC-homography inlier count; thumbnail top-%d prefilter; "
                         "match@%dpx gray; Lowe %.2f; RANSAC %.1fpx. Score = max inliers over shortlist."
                         % (ORB_N, SHORTLIST_K, MATCH_RES, LOWE, RANSAC_PX)),
              "scene_counts": dict(groups), "per_bag": {}, "calibration": {}}

    for bag in a.bags:
        print(f"[{bag}] matching {len(scenes_ok)} scenes ...", flush=True)
        best = match_bag(bag, scenes_ok, sdesc, sorb)
        if best is None:
            continue
        bygroup = collections.defaultdict(list)
        for b in scenes_ok:
            bygroup[scenes[b][0]].append(best[b])
        report["per_bag"][bag] = {g: stats(v) for g, v in bygroup.items()}

        # controls: positives = this bag's own prefix group; negatives = other month-prefixed groups
        pos = bygroup.get(bag, [])
        neg = [x for g, v in bygroup.items() if g not in (bag, "unattributed") for x in v]
        unk = bygroup.get("unattributed", [])
        cal = {"positives": stats(pos), "negatives": stats(neg), "unknowns": stats(unk)}
        if pos and neg:
            # conservative threshold: above every known negative (zero known-neg false positives)
            thr = int(max(neg)) + 1
            cal["threshold_inliers"] = thr
            cal["separates"] = bool(min(pos) > max(neg))
            cal["neg_max"] = int(max(neg)); cal["pos_min"] = int(min(pos))
            cal["unknowns_flagged_present"] = int(sum(1 for x in unk if x >= thr))
        report["calibration"][bag] = cal
        print(f"  [{bag}] pos {cal['positives']} | neg {cal['negatives']} | unk {cal['unknowns']}", flush=True)
        if "threshold_inliers" in cal:
            print(f"  [{bag}] separates={cal['separates']} thr>={cal['threshold_inliers']} "
                  f"(pos_min {cal['pos_min']} vs neg_max {cal['neg_max']}); "
                  f"unknowns flagged present: {cal['unknowns_flagged_present']}/{cal['unknowns']['n']}", flush=True)

    OUT = PKG / "results" / "geometric" / "scene_attribution_keypoint.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT.relative_to(PKG)}")


if __name__ == "__main__":
    main()
