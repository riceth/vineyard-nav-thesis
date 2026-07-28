"""D048 — ORB+RANSAC scene→bag attribution gate (the CP-0 unattributed-scene check).

This is the PRODUCTION home of the gate validated in `one_time/scene_attribution_orb.py`
(D048, O019). The 90 unattributed `color_image_*` SemanticBLT scenes carry no month prefix,
so CP-0's prefix matcher never checks them; this module decides, per bag, whether any of
them were actually recorded in that bag (and must therefore be excluded as perception-training
contamination).

Why keypoints and not correlation: a global 128×128 thumbnail matches generic vineyard-row
STRUCTURE, not scene IDENTITY, so known-foreign scenes scored as high as true members (D046c,
rejected). ORB features + Lowe-ratio match + RANSAC homography measure identity: a true
re-observation yields many correspondences consistent under one homography; a different frame
of the same vineyard yields few that survive that geometric constraint.

Three-band decision rule (LOCKED, D048; calibrated on the march/april controls — the clean
floor gap 12→59 and the cross-session tail ceiling 127):
    inliers ≤ T_ABSENT (40)   → absent        (above every unknown, below every genuine member)
    inliers ≥ T_PRESENT (200) → present       → exclude from that bag's evaluation
    40 < inliers < 200        → needs_review  → BLOCKS that bag's evaluation until reviewed
The wide 40↔200 margins absorb RANSAC's minor run-to-run jitter, so a verdict never flips on it.

The gate reuses CP-0's coarse thumbnail bank for shortlisting (recall only), so wiring it into
`prep.py` (CP-0) adds only the ORB verification of a bounded shortlist — no second bank.
"""
from __future__ import annotations
import sys
import json
import re
import collections
from pathlib import Path

import numpy as np
import cv2
from rosbags.typesys import Stores, get_typestore

GIT_ROOT = Path(__file__).resolve().parents[3]          # /workspaces/dissertation
DATASET = GIT_ROOT / "SemanticBLT.v1-2024-june.coco-segmentation"
CAM = "/front/zed_node/rgb/image_rect_color/compressed"
TS = get_typestore(Stores.ROS2_HUMBLE)

# --- matching parameters (identical to the validated one_time/scene_attribution_orb.py) ---
COARSE = 10            # thumbnail-bank stride (recall prefilter) — must match CP-0's bank stride
SHORTLIST_K = 30       # candidate bag frames ORB-verified per scene
MATCH_RES = 640        # scene + bag frame matched at this resolution (gray) for ORB
ORB_N = 3000           # ORB features
LOWE = 0.75            # Lowe ratio test
RANSAC_PX = 5.0        # homography RANSAC reprojection threshold (px)
MIN_MATCH = 8          # minimum putative matches before attempting a homography
GROUPS = ("march", "april", "may", "june", "july", "september")

# --- three-band decision thresholds (LOCKED, D048) ---
T_ABSENT = 40          # ≤ this → absent
T_PRESENT = 200        # ≥ this → present (exclude)
FINE_HALF = 30         # fine-verify: full-res ±this-many-frames non-strided search around a coarse hit

_orb = cv2.ORB_create(nfeatures=ORB_N)
_bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def classify(n: int) -> str:
    """Three-band verdict for an inlier count (D048). Order matters at the boundaries:
    ≤40 is absent, ≥200 is present, the open interval between is needs_review."""
    if n <= T_ABSENT:
        return "absent"
    if n >= T_PRESENT:
        return "present"
    return "needs_review"


def thumb(gray128: np.ndarray) -> np.ndarray:
    """128×128 zero-mean unit-norm grayscale descriptor — the CP-0 recall prefilter descriptor.
    Identical to prep._desc, so a CP-0 bank row and a scene thumb are comparable."""
    g = gray128.astype(np.float32).ravel(); g -= g.mean(); n = np.linalg.norm(g)
    return g / n if n else g


def orb_of(gray: np.ndarray):
    return _orb.detectAndCompute(gray, None)


def inliers(kpA, desA, kpB, desB) -> int:
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


def decode_gray(cur, msg_id: int) -> np.ndarray:
    """Decode one compressed camera message to MATCH_RES×MATCH_RES grayscale (for ORB)."""
    data = cur.execute("SELECT data FROM messages WHERE id=?", (msg_id,)).fetchone()[0]
    m = TS.deserialize_cdr(bytes(data), "sensor_msgs/msg/CompressedImage")
    im = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_UNCHANGED)
    if im.ndim == 3 and im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return cv2.resize(im, (MATCH_RES, MATCH_RES))


def fine_verify(sorb, center_idx: int, ids: list[int], cur, half: int = FINE_HALF) -> int:
    """Max ORB+RANSAC inliers over the full-res ±`half`-frame neighbourhood (NON-strided) of a coarse
    hit. The coarse bank strides by COARSE, so a true member's exact source frame is usually skipped
    -> its best coarse match is a near-neighbour with a moderate (needs_review) score. Decoding every
    frame within ±half recovers the true frame (a genuine member jumps past T_PRESENT; a look-alike
    from another session stays down). Validated O019 (one_time/scene_attribution_fineverify.py),
    promoted into the production gate on the june bag (D048 two-stage rule)."""
    kpS, desS = sorb
    lo, hi = max(0, center_idx - half), min(len(ids) - 1, center_idx + half)
    best = 0
    for j in range(lo, hi + 1):
        best = max(best, inliers(kpS, desS, *orb_of(decode_gray(cur, ids[j]))))
    return best


def scene_table() -> dict[str, tuple[str, str]]:
    """base scene → (group, first image path). group = march|…|september|unattributed."""
    files: dict[str, list[str]] = collections.defaultdict(list)
    for split in ("train", "valid", "test"):
        for im in json.load(open(DATASET / split / "_annotations.coco.json"))["images"]:
            base = re.sub(r"_png\.rf\..*", "", im["file_name"])
            files[base].append(str(DATASET / split / im["file_name"]))
    out = {}
    for base, paths in files.items():
        g = next((m for m in GROUPS if base.startswith(m)), "unattributed")
        out[base] = (g, sorted(paths)[0])          # one representative version per scene
    return out


def unattributed_scenes() -> dict[str, str]:
    """{base: representative image path} for the 90 no-prefix `color_image_*` scenes."""
    return {b: path for b, (g, path) in scene_table().items() if g == "unattributed"}


def gate(bank: np.ndarray, coarse_idx: list[int], ids: list[int], cur,
         scenes: dict[str, str], fine_half: int = FINE_HALF, log=print) -> list[dict]:
    """Score each scene against one bag and return per-scene attribution rows.

    Reuses the caller's CP-0 coarse bank for shortlisting (recall), then ORB-verifies the
    shortlist union at MATCH_RES. Returns rows sorted by descending inliers:
        {scene, inliers, bag_frame (message index into `ids`, or -1), verdict}
    `bank` rows must be `thumb(...)` descriptors over `[ids[i] for i in coarse_idx]` (i.e. the
    exact bank CP-0 already builds); `cur` is an open sqlite cursor on the bag's `.db3`.
    """
    if not scenes:
        return []
    # scene thumbnail (shortlist) + scene ORB (verification), computed once per scene
    sdesc, sorb = {}, {}
    for b, path in scenes.items():
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        sdesc[b] = thumb(cv2.resize(im, (128, 128)))
        sorb[b] = orb_of(cv2.resize(im, (MATCH_RES, MATCH_RES)))
    ok = list(sdesc)

    # shortlist candidate coarse positions per scene, then decode+ORB the UNION once
    shortlist = {b: np.argsort(bank @ sdesc[b])[::-1][:SHORTLIST_K] for b in ok}
    need = sorted({int(p) for sl in shortlist.values() for p in sl})
    forb = {p: orb_of(decode_gray(cur, ids[coarse_idx[p]])) for p in need}

    rows = []
    for b in ok:
        kpS, desS = sorb[b]
        best_n, best_pos = 0, -1
        for p in shortlist[b]:
            n = inliers(kpS, desS, *forb[int(p)])
            if n > best_n:
                best_n, best_pos = n, int(p)
        bag_frame = coarse_idx[best_pos] if best_pos >= 0 else -1
        rows.append({"scene": b, "coarse_inliers": best_n, "inliers": best_n,
                     "bag_frame": bag_frame, "verdict": classify(best_n)})

    # Stage 2 (D048 two-stage rule, promoted on june): fine-verify ONLY the needs_review band. The
    # coarse stride skips true members' exact frames, so a moderate coarse score is usually a
    # near-neighbour; the full-res ±fine_half search recovers the true frame (genuine member ->
    # present; same-vineyard look-alike -> stays down). No cost when the band is empty.
    nr = [r for r in rows if r["verdict"] == "needs_review" and r["bag_frame"] >= 0]
    if nr:
        log(f"    D048 fine-verify: {len(nr)} needs_review scene(s) (full-res ±{fine_half} frames each) ...")
        for r in nr:
            r["fine_inliers"] = fine_verify(sorb[r["scene"]], r["bag_frame"], ids, cur, fine_half)
            r["inliers"] = max(r["coarse_inliers"], r["fine_inliers"])
            r["verdict"] = classify(r["inliers"])
        log(f"    D048 fine-verify: {sum(r['verdict']=='present' for r in nr)}/{len(nr)} recovered to "
            f"present; {sum(r['verdict']=='needs_review' for r in nr)} still needs_review")

    rows.sort(key=lambda r: r["inliers"], reverse=True)
    counts = collections.Counter(r["verdict"] for r in rows)
    log(f"    D048 gate: {len(rows)} unattributed scenes -> "
        f"{counts['present']} present, {counts['needs_review']} needs_review, {counts['absent']} absent")
    return rows


def load_confirmations(census_path) -> dict:
    """Human present/absent decisions for scenes still in needs_review after fine-verify, read from
    `d048_confirmed.json` beside the bag's census: {scene: "present"|"absent"} (keys starting with
    '_' are ignored as notes). Missing file -> {}. The auditable record of the visual-review step."""
    p = Path(census_path).parent / "d048_confirmed.json"
    return {k: v for k, v in json.load(open(p)).items() if not k.startswith("_")} if p.exists() else {}


def apply_confirmations(rows: list[dict], confirmed: dict, log=print) -> None:
    """Finalise residual needs_review rows IN PLACE from the confirmations. A confirmed scene takes
    'present'/'absent' (marked confirmed_by); an UNconfirmed residual stays needs_review and still
    BLOCKS CP-1 (fail-closed)."""
    resid = [r for r in rows if r["verdict"] == "needs_review"]
    for r in resid:
        v = confirmed.get(r["scene"])
        if v in ("present", "absent"):
            r["verdict"] = v
            r["confirmed_by"] = "d048_confirmed.json"
    left = [r["scene"] for r in rows if r["verdict"] == "needs_review"]
    if resid:
        log(f"    D048 confirmations: {len(resid)-len(left)}/{len(resid)} residual resolved from "
            f"d048_confirmed.json; {len(left)} unconfirmed" + (f" -> {left}" if left else ""))
