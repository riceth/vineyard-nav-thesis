"""Pooled moving-block length derivation (Analysis-H) for the March pooled strand (D040).

Single source of truth for the paired-difference decorrelation distances and the moving-block
lengths (L_GT1, L_GT2) consumed by the whole-bag CI estimators: line_fit_eval.py (per-arm
RMS CIs), paired_crossarm.py (paired-diff CIs) and config_analysis.py (config RMS
CIs). Keeping the derivation here prevents the three consumers drifting and removes any hardcode
of the val-derived 11/31 (POOLING_SPEC #2: "re-derive Analysis-H ... on the whole-bag data").

Methodology is identical to diagnostics/autocorrelation_block_analysis.py (which stays the
print-only human cross-check): per arm-pair (A-B, A-C, B-C), per metric (GT-1 offset / GT-2
heading), the spatial autocorrelation of the across-seed paired difference vs separation in
metres; the decorrelation distance = first lag where autocorrelation < threshold (0.1, with a
1.5 m fallback); block length L = max(2, round(2*decorr / mean_entry_spacing)). The canonical
strand-wide L_GT1 / L_GT2 = the CONSERVATIVE (maximum) block length across the three pairs per
metric -- this reproduces the single-L convention the val/test production scripts used
(paired_crossarm_test.py / config_sweep_val.py hardcoded one L per metric), now re-derived on
the pooled val+test data rather than fixed at 11/31.

Positions are all eligible frames grouped by pass (whole-bag; no split key); each pass is a
single contiguous spatial series, so a moving block never straddles a pass boundary.
"""
import json
import collections
from pathlib import Path

import numpy as np

SEEDS = [42, 43, 44]
PAIRS = [("A", "B"), ("A", "C"), ("B", "C")]


def _positions(man):
    byp = collections.defaultdict(list)
    for f in man["frames"]:
        if f["eligible"]:                    # whole-bag: eligible only, no split key
            byp[f["pass_id"]].append(f)
    pos = {}
    for pid, fs in byp.items():
        fs = sorted(fs, key=lambda f: f["i"])
        xy = np.array([[f["x"], f["y"]] for f in fs])
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        for f, d in zip(fs, cum):
            pos[f["i"]] = (pid, float(d))
    return pos


def _load(per_frame_csv):
    D = collections.defaultdict(dict)   # (arm,seed) -> {frame_i: (offset, heading)} for two_row
    for ln in Path(per_frame_csv).read_text().splitlines()[1:]:
        a, s, i, cls, off, hdg, *_ = ln.split(",")
        if cls == "two_row" and off and hdg:
            D[(a, int(s))][int(i)] = (float(off), float(hdg))
    return D


def _series(D, pos, x, y):
    cx = set.intersection(*[set(D[(x, s)]) for s in SEEDS])
    cy = set.intersection(*[set(D[(y, s)]) for s in SEEDS])
    rows = []
    for i in (cx & cy):
        if i not in pos:
            continue
        d1 = np.mean([D[(x, s)][i][0] for s in SEEDS]) - np.mean([D[(y, s)][i][0] for s in SEEDS])
        d2 = np.mean([D[(x, s)][i][1] for s in SEEDS]) - np.mean([D[(y, s)][i][1] for s in SEEDS])
        rows.append((pos[i][0], pos[i][1], d1, d2))
    return sorted(rows, key=lambda r: (r[0], r[1]))


def _autocorr(rows, ci, maxd=3.0, bw=0.15):
    v = np.array([r[ci] for r in rows])
    mu, var = v.mean(), v.var()
    nb = int(maxd / bw)
    num = np.zeros(nb)
    cnt = np.zeros(nb)
    bp = collections.defaultdict(list)
    for r in rows:
        bp[r[0]].append(r)
    for _, rs in bp.items():
        p = np.array([r[1] for r in rs])
        val = np.array([r[ci] for r in rs]) - mu
        for a in range(len(rs)):
            b = a + 1
            while b < len(rs) and p[b] - p[a] < maxd:
                k = int((p[b] - p[a]) / bw)
                if k < nb:
                    num[k] += val[a] * val[b]
                    cnt[k] += 1
                b += 1
    ac = np.where(cnt > 0, num / np.maximum(cnt, 1) / var, np.nan)
    return (np.arange(nb) + 0.5) * bw, ac


def _decorr(centres, ac, thr):
    for c, a in zip(centres, ac):
        if not np.isnan(a) and a < thr:
            return round(float(c), 2)
    return None


def _mean_spacing(rows):
    bp = collections.defaultdict(list)
    for r in rows:
        bp[r[0]].append(r[1])
    sps = [(max(v) - min(v)) / (len(v) - 1) for v in bp.values() if len(v) > 1]
    return float(np.mean(sps)) if sps else 0.05


# D053 — CI reliability guard. `decorr` is the first distance-lag at which the paired-difference
# autocorrelation drops below `thr`; it therefore cannot be located more finely than the spacing of
# the paired samples themselves. When there are too few samples per decorrelation length the first
# bin already sits at or beyond the crossing, so `decorr` is returned as a lower bound, L is
# under-estimated, and the resulting CIs are ANTI-CONSERVATIVE (too narrow). The strand-wide L is
# already the max across pairs, which protects a single sparse pair (april's B-C sits at 0.98 and is
# carried by A-B's 11); that protection fails only when every pair is sparse at once, as on july2023.
# Threshold calibrated on the committed bags: the pair that SETS each strand-wide L scores 4.61-18.69
# there, against july2023's 1.09 / 1.91 — so 3.0 separates with margin on both sides.
MIN_SAMPLES_PER_DECORR = 3.0


def pooled_block_lengths(per_frame_csv, man, thr=0.1, maxd=3.0, bw=0.15, fallback=1.5):
    """Return {"L_GT1", "L_GT2", "threshold", "reduction", "fallback_m", "per_pair"}.

    L_GT1 / L_GT2 are the conservative (max) block lengths across the three arm-pairs, in
    per-frame entries, for use with the block-bootstrap CI estimators (block length in entries).
    per_pair records each pair's decorrelation distance, block metres and block-entry length so
    the numbers can be cross-checked against autocorrelation_block_analysis.py's prints.
    """
    pos = _positions(man)
    D = _load(per_frame_csv)
    per_pair = {}
    L1s, L2s = [], []
    for (x, y) in PAIRS:
        rows = _series(D, pos, x, y)
        sp = _mean_spacing(rows)
        entry = {"n_all": len(rows), "mean_spacing_m": round(sp, 4)}
        for ci, name, store in [(2, "GT1", L1s), (3, "GT2", L2s)]:
            centres, ac = _autocorr(rows, ci, maxd, bw)
            dd = _decorr(centres, ac, thr)
            block_m = 2 * (dd if dd else fallback)
            L = max(2, int(round(block_m / sp)))
            spd = (dd / sp) if dd else None          # paired samples per decorrelation length (D053)
            entry[name] = {"decorr_m": dd, "used_fallback": dd is None,
                           "block_m": round(block_m, 3), "L": L,
                           "samples_per_decorr": round(spd, 2) if spd is not None else None,
                           "resolution_limited": bool(spd is not None and spd < MIN_SAMPLES_PER_DECORR)}
            store.append((L, spd))
        per_pair[f"{x}-{y}"] = entry
    # the pair that SETS each strand-wide L is the one whose reliability governs that metric
    win1 = max(L1s, key=lambda t: t[0])
    win2 = max(L2s, key=lambda t: t[0])
    def _rel(w):
        # a fallback decorr (None) means the crossing lies beyond maxd, i.e. LONGER than the sampling
        # can bound — the opposite of the resolution-limited failure, so it is not flagged here
        return {"samples_per_decorr": round(w[1], 2) if w[1] is not None else None,
                "reliable": bool(w[1] is None or w[1] >= MIN_SAMPLES_PER_DECORR)}
    return {"L_GT1": int(win1[0]), "L_GT2": int(win2[0]),
            "threshold": thr, "fallback_m": fallback,
            "reduction": "conservative_max_across_pairs",
            "ci_reliability": {"min_samples_per_decorr": MIN_SAMPLES_PER_DECORR,
                               "GT1": _rel(win1), "GT2": _rel(win2)},
            "per_pair": per_pair}


if __name__ == "__main__":   # standalone sanity check (prints the derived block lengths for a bag)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from bag_config import parse_bag
    B = parse_bag()
    man = json.load(open(B["manifest"]))
    bl = pooled_block_lengths(B["per_frame_csv"], man)
    print(json.dumps(bl, indent=2))
    print(f"\ncanonical block lengths ({B['bag']}): L_GT1={bl['L_GT1']}  L_GT2={bl['L_GT2']}")
