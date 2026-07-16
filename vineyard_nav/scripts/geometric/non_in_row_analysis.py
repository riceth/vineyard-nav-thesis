"""Non-in-row deployment-gap characterisation (D041 category C; F020, F021). Bag-agnostic.

  python3 non_in_row_analysis.py --bag march --scope non_in_row

Reads the non-in-row per-frame CSV (the `line_fit_infer --scope non_in_row` output) + the manifest,
categorises each frame (stationary / turn / transition), and reports:
  F020 — the pipeline's OUTPUT-CLASS distribution (none / single_row / two_row / fitfail) per
         category and per arm: what the in-row pipeline does when driven over non-in-row frames.
  F021 — on the frames where the pipeline (spuriously) claims two_row, a DRIVEN-PATH ERROR: the RMS
         lateral offset of the IPM-invalid predicted centreline relative to base_link, per category
         and per arm.

The driven-path error is NOT the in-row centreline_error_rms and is not comparable to it. It is a
degradation characterisation carrying three conflations: (1) the flat-ground IPM projection is
invalid on headland slopes; (2) the row centreline is undefined on turns; (3) turn geometry
conflates with the error. See D041 (frame accounting) and F013 (in-row headline).
"""
import sys, json, collections, bisect
from pathlib import Path
import numpy as np

PKG = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
from bag_config import parse_bag, frames_for_scope

B = parse_bag()
MAN = json.load(open(B["manifest"]))
PF = B["per_frame_csv"]
OUT = B["non_in_row_analysis"]
SEEDS = [42, 43, 44]
CATS = ("stationary", "turn", "transition")

# --- categorise each non-in-row frame: stationary / turn / transition -------------------------
frames = {f["i"]: f for f in MAN["frames"]}
elig_idx = sorted(i for i, f in frames.items() if f["eligible"])
elig_corr = {i: frames[i]["corridor"] for i in elig_idx}


def category(i):
    """stationary (headland+stationary); else turn (moving, same flanking corridor) vs
    transition (moving, different flanking corridor / bag edge)."""
    if frames[i]["stationary"]:
        return "stationary"
    pos = bisect.bisect_left(elig_idx, i)
    bc = elig_corr[elig_idx[pos - 1]] if pos > 0 else None
    ac = elig_corr[elig_idx[pos]] if pos < len(elig_idx) else None
    if bc is not None and ac is not None:
        return "turn" if bc == ac else "transition"
    return "transition"   # bag edge (no flanking pass on one side)


cat = {i: category(i) for i in frames_for_scope(MAN, "non_in_row")}
catcount = collections.Counter(cat.values())

# --- parse the non-in-row per-frame CSV -------------------------------------------------------
cls_by = collections.defaultdict(collections.Counter)      # (arm, cat) -> cls counts (all seeds)
two = collections.defaultdict(list)                        # (arm, cat) -> [(offset, heading)] two_row
for ln in PF.read_text().splitlines()[1:]:
    a, s, i, cls, off, hdg, *_ = ln.split(",")
    i = int(i)
    if i not in cat:
        continue
    c = cat[i]
    cls_by[(a, c)][cls] += 1
    cls_by[(a, "ALL")][cls] += 1
    if cls == "two_row" and off and hdg:
        two[(a, c)].append((float(off), float(hdg)))
        two[(a, "ALL")].append((float(off), float(hdg)))


def dist(counter):
    tot = sum(counter.values())
    d = {k: round(100 * counter[k] / tot, 1) for k in ("two_row", "single_row", "none", "fitfail")}
    d["n"] = tot
    return d


def rms(v):
    v = np.asarray(v, float)
    return round(float(np.sqrt(np.mean(v ** 2))), 3) if len(v) else None


def dpe(lst):
    if not lst:
        return {"two_row_n": 0, "driven_path_error_rms_m": None, "driven_path_heading_rms_deg": None}
    return {"two_row_n": len(lst),
            "driven_path_error_rms_m": rms([o for o, h in lst]),
            "driven_path_heading_rms_deg": rms([h for o, h in lst])}


F020 = {"category_counts": dict(catcount),
        "per_arm_overall": {a: dist(cls_by[(a, "ALL")]) for a in "ABC"},
        "per_category": {c: {a: dist(cls_by[(a, c)]) for a in "ABC"} for c in CATS}}
F021 = {"metric_note": ("driven_path_error = RMS lateral offset of the IPM-invalid predicted "
                        "centreline vs base_link on non-in-row two_row outputs; NOT the in-row "
                        "centreline_error_rms and not comparable to it. Conflations: (1) flat-ground "
                        "IPM invalid on headland slopes; (2) row centreline undefined on turns; "
                        "(3) turn geometry conflates with the error."),
        "per_arm_overall": {a: dpe(two[(a, "ALL")]) for a in "ABC"},
        "per_category": {c: {a: dpe(two[(a, c)]) for a in "ABC"} for c in CATS}}

out = {"config": {"bag": B["bag"], "scope": B["scope"], "n_frames": len(cat), "seeds": SEEDS,
                  "categories": "stationary (headland+stationary) / turn (moving, same flanking "
                                "corridor) / transition (moving, different flanking corridor or bag edge)"},
       "F020_output_distribution": F020, "F021_driven_path_error": F021}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, indent=2))

print(f"[{B['bag']}/{B['scope']}] non-in-row frames {len(cat)} | categories {dict(catcount)}")
print("\nF020 output-class distribution (per arm, all non-in-row; % of arm's 3-seed rows):")
for a in "ABC":
    print(f"  {a}: {F020['per_arm_overall'][a]}")
print("\nF021 driven-path error on two_row outputs (per arm, all non-in-row):")
for a in "ABC":
    print(f"  {a}: {F021['per_arm_overall'][a]}")
print(f"\nwrote {OUT}")
