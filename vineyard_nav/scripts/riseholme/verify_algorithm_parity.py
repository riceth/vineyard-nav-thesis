"""Gate: the Riseholme and Ktima strands MUST share one algorithm.

If the two sites' results differ, that difference has to be attributable to the data — never to the
code. This asserts that every file in the SHARED set is byte-identical between
scripts/geometric/ and scripts/riseholme/, and that every divergence is declared with a reason.

Run before any Riseholme evaluation, and after any edit to either tree:
    python3 scripts/riseholme/verify_algorithm_parity.py
Exit status 0 = parity holds. Non-zero = a shared file drifted, or a file is untriaged.
"""
import hashlib
import sys
from pathlib import Path

RH = Path(__file__).resolve().parent
KT = RH.parent / "geometric"

# Byte-identical in both trees. These carry the measurement itself: row fitting, ground projection
# consumption, model inference, and the CI estimator. Any drift here invalidates the comparison.
SHARED = {
    "row_model.py":            "row fitting: seeding, consensus sweep, far extension, rejections",
    "cp3_geometry.py":         "detection -> ground points -> centreline (GT-1 / GT-2)",
    "block_lengths.py":        "moving-block CI estimator + the D053 reliability guard",
    "extract_detections.py":   "arm-C detections for the config sweep (CONF, BLOB_FRAC)",
    "line_fit_infer.py":       "9-model inference producing line_fit_per_frame.csv",
    "analyze.py":              "line_fit_eval, paired_crossarm, config_analysis, mitigation",
    "cuda_preload.py":         "cuDNN preload (D049); environment, not algorithm",
}

# Intentionally different. Confined to input/output and site calibration — never to the measurement.
DIVERGENT = {
    "bag_config.py":            "paths + per-bag inputs; presents the IDENTICAL resolve() interface",
    "projection_calibration.py": "D435I intrinsics, 1280x720, and the empirical rear-facing extrinsics",
    "prep.py":                  "CP-0 vacuous here; CP-1 algorithm checked by NEAR_IDENTICAL below",
    "extract_frames.py":        "raw sensor_msgs/Image, not CompressedImage; 1280x720 source",
    "check_bag_complete.py":    "different expected artefacts (no LiDAR cross-check, no control strand)",
}

# Present in the Ktima tree and deliberately absent from Riseholme.
NOT_PORTED = {
    "convert_bag.py":        "conversion is a one-line rosbags-convert call for these bags",
    "scene_attribution.py":  "no SemanticBLT scenes from this site to attribute",
    "figures.py":            "per-bag figure set is Ktima-specific; RH figures are separate",
    "figures_compare.py":    "cross-bag Ktima comparison",
    "figures_supplementary.py": "cross-bag Ktima supplementary set",
}


# The copied stage scripts locate their siblings with
#     sys.path.insert(0, str(PKG / "scripts" / "<tree>"))
# which must name the tree the file lives in. That single token is the only permitted textual
# difference in a SHARED file; hashing is done AFTER normalising it, so identity is still proven
# over everything else including every constant and every line of logic.
_NORM = ('"scripts" / "riseholme"', '"scripts" / "geometric"')


def md5(p):
    return hashlib.md5(p.read_text().replace(*_NORM).encode()).hexdigest()


def raw_md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


# Files that are NOT byte-identical but whose ALGORITHM must be. Each entry names the function
# whose body is compared, and the only substrings allowed to differ (operator-facing text).
NEAR_IDENTICAL = {
    "prep.py": {
        "func": "frame_manifest_build",
        "until": "def main()",
        "allow": ["Convert it first", "Run it first"],
        "why": "CP-1 pass detection, corridor assignment and subsample must match Ktima exactly",
    },
}


def _body(path, func, until):
    s = path.read_text()
    return s[s.index(f"def {func}"):s.index(until)].splitlines()


def check_near_identical():
    import difflib
    bad = []
    print("\nNEAR-IDENTICAL — algorithm must match; only declared text may differ")
    for name, spec in sorted(NEAR_IDENTICAL.items()):
        a, b = KT / name, RH / name
        if not (a.exists() and b.exists()):
            print(f"  SKIP {name} (not present in both)"); continue
        da = _body(a, spec["func"], spec["until"])
        db = _body(b, spec["func"], spec["until"])
        changed = [l[1:].strip() for l in difflib.unified_diff(da, db, lineterm="", n=0)
                   if l[:1] in "+-" and not l.startswith(("+++", "---"))]
        offending = [c for c in changed if not any(t in c for t in spec["allow"])]
        ok = not offending
        print(f"  {'OK  ' if ok else 'DRIFT'} {name}::{spec['func']:22} "
              f"{len(changed)} changed line(s), {len(offending)} not allowed  -- {spec['why']}")
        for c in offending[:5]:
            print(f"        offending: {c[:96]}")
        if not ok:
            bad.append(name)
    return bad


def main():
    fail = []
    print("SHARED — must be byte-identical")
    for name, why in sorted(SHARED.items()):
        a, b = KT / name, RH / name
        if not b.exists():
            print(f"  MISSING in riseholme : {name}"); fail.append(name); continue
        if not a.exists():
            print(f"  MISSING in geometric : {name}"); fail.append(name); continue
        ha, hb = md5(a), md5(b)
        ok = ha == hb
        exact = raw_md5(a) == raw_md5(b)
        tag = "OK  " if ok else "DRIFT"
        note = "" if exact else "  [sys.path token normalised]"
        print(f"  {tag} {name:26} {ha[:12]}  {why}{note}")
        if not ok:
            fail.append(name)

    fail.extend(check_near_identical())

    print("\nDIVERGENT — different by design")
    for name, why in sorted(DIVERGENT.items()):
        state = "present" if (RH / name).exists() else "not written yet"
        print(f"  {name:28} [{state}]  {why}")

    print("\nNOT PORTED")
    for name, why in sorted(NOT_PORTED.items()):
        print(f"  {name:28} {why}")

    # any Ktima file not triaged into one of the three sets is an untracked decision
    known = set(SHARED) | set(DIVERGENT) | set(NOT_PORTED)
    stray = sorted(p.name for p in KT.glob("*.py") if p.name not in known)
    if stray:
        print(f"\nUNTRIAGED Ktima files (classify them in this script): {stray}")
        fail.extend(stray)

    print()
    if fail:
        print(f"PARITY FAILED: {fail}")
        return 1
    print(f"PARITY OK — {len(SHARED)} shared files identical, "
          f"{len(DIVERGENT)} declared divergences, {len(NOT_PORTED)} not ported")
    return 0


if __name__ == "__main__":
    sys.exit(main())
