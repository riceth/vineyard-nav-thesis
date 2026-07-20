#!/usr/bin/env python3
"""Bootstrap confidence-interval utility (decision D020).

Percentile bootstrap CIs over per-frame metric rows. This is the shared utility
for all three arms; Phase A feeds it the per-frame CSV emitted by
segmentation/unet_binary/evaluate.py.

Estimand (important, stated honestly): the bootstrap resamples FRAMES with
replacement and takes the MEAN of a per-frame metric. This "mean of per-frame
IoU" is NOT identical to the pooled/aggregate IoU reported in <split>_metrics.json
(a ratio of summed pixel counts). Both are valid; they answer slightly different
questions. Per-frame bootstrap is the right tool for per-scene uncertainty on the
23-scene test set (D028/O006), so the point estimate reported here is the
per-frame mean, and the pooled value is reported alongside for context by the
caller. With only 11/12 frames per canopy stratum, stratified CIs are wide by
construction — the honest consequence of the dataset ceiling (O006), not a defect.

Deterministic: seeded with np.random.default_rng(seed) (default 42, D016).
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_METRICS = ["iou_foreground", "iou_background", "precision_foreground",
                   "recall_foreground", "f1_foreground"]


def bootstrap_mean_ci(values: Sequence[float], n_resamples: int, ci: float,
                      rng: np.random.Generator) -> Dict[str, float]:
    """Percentile-bootstrap CI for the mean of `values`."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan")}
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = v[idx].mean(axis=1)
    lo_p = (100.0 - ci) / 2.0
    hi_p = 100.0 - lo_p
    lo, hi = np.percentile(means, [lo_p, hi_p])
    return {"n": int(n), "mean": float(v.mean()),
            "ci_low": float(lo), "ci_high": float(hi)}


def bootstrap_diff_ci(a: Sequence[float], b: Sequence[float], n_resamples: int,
                      ci: float, rng: np.random.Generator) -> Dict[str, float]:
    """Percentile-bootstrap CI for the difference of independent-sample means
    (mean(a) - mean(b)); a and b are resampled independently each iteration."""
    va = np.asarray(a, dtype=float); va = va[~np.isnan(va)]
    vb = np.asarray(b, dtype=float); vb = vb[~np.isnan(vb)]
    na, nb = len(va), len(vb)
    if na == 0 or nb == 0:
        return {"diff": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    da = va[rng.integers(0, na, size=(n_resamples, na))].mean(axis=1)
    db = vb[rng.integers(0, nb, size=(n_resamples, nb))].mean(axis=1)
    diffs = da - db
    lo_p = (100.0 - ci) / 2.0
    lo, hi = np.percentile(diffs, [lo_p, 100.0 - lo_p])
    return {"diff": float(va.mean() - vb.mean()),
            "ci_low": float(lo), "ci_high": float(hi)}


def load_per_frame(csv_path: str) -> List[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def analyze(csv_path: str, metrics: Sequence[str] = DEFAULT_METRICS,
            stratify_col: str = "canopy_state", n_resamples: int = 10000,
            ci: float = 95.0, seed: int = 42,
            diff_pair: Optional[Tuple[str, str]] = ("canopy", "bare_vine")) -> dict:
    rows = load_per_frame(csv_path)
    strata = sorted({r[stratify_col] for r in rows})
    rng = np.random.default_rng(seed)

    out: Dict[str, dict] = {}
    for metric in metrics:
        block: Dict[str, dict] = {}
        block["overall"] = bootstrap_mean_ci([float(r[metric]) for r in rows],
                                             n_resamples, ci, rng)
        for s in strata:
            vals = [float(r[metric]) for r in rows if r[stratify_col] == s]
            block[s] = bootstrap_mean_ci(vals, n_resamples, ci, rng)
        if diff_pair and diff_pair[0] in strata and diff_pair[1] in strata:
            a = [float(r[metric]) for r in rows if r[stratify_col] == diff_pair[0]]
            b = [float(r[metric]) for r in rows if r[stratify_col] == diff_pair[1]]
            block[f"{diff_pair[0]}_minus_{diff_pair[1]}"] = bootstrap_diff_ci(
                a, b, n_resamples, ci, rng)
        out[metric] = block

    return {
        "meta": {"source_csv": csv_path, "n_resamples": n_resamples,
                 "ci_percent": ci, "seed": seed, "n_rows": len(rows),
                 "stratify_col": stratify_col,
                 "estimand": "per-frame mean (not pooled IoU); see module docstring"},
        "metrics": out,
    }


def _fmt(d: dict) -> str:
    return f"{d['mean']:.4f}  [{d['ci_low']:.4f}, {d['ci_high']:.4f}]  (n={d['n']})"


def print_report(result: dict) -> None:
    m = result["meta"]
    print(f"\nBootstrap {m['ci_percent']:.0f}% CIs — {m['n_resamples']} resamples, "
          f"seed {m['seed']}, {m['n_rows']} frames")
    print(f"source: {m['source_csv']}")
    for metric, block in result["metrics"].items():
        print(f"\n{metric}")
        for key in ("overall", "bare_vine", "canopy"):
            if key in block:
                print(f"  {key:<10} {_fmt(block[key])}")
        for key in block:
            if key.endswith("_minus_bare_vine"):
                d = block[key]
                print(f"  {key:<22} diff {d['diff']:+.4f}  "
                      f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]")


def _self_test() -> None:
    rng = np.random.default_rng(0)
    # Constant values -> CI collapses to the constant.
    c = bootstrap_mean_ci([0.5] * 20, 2000, 95.0, rng)
    assert abs(c["mean"] - 0.5) < 1e-12 and abs(c["ci_low"] - 0.5) < 1e-12
    # Mean lies within its own CI; NaNs ignored.
    r = bootstrap_mean_ci([0.2, 0.4, float("nan"), 0.6, 0.8], 5000, 95.0, rng)
    assert r["n"] == 4 and r["ci_low"] <= r["mean"] <= r["ci_high"]
    # Determinism: same seed -> identical CI.
    a = bootstrap_mean_ci([0.1, 0.3, 0.9, 0.4], 3000, 95.0, np.random.default_rng(42))
    b = bootstrap_mean_ci([0.1, 0.3, 0.9, 0.4], 3000, 95.0, np.random.default_rng(42))
    assert a == b
    print("bootstrap.py self-test passed.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bootstrap CIs over per-frame metrics (D020).")
    ap.add_argument("--per-frame-csv", help="Per-frame metrics CSV from evaluate.py.")
    ap.add_argument("--output", help="Write results JSON here.")
    ap.add_argument("--n-resamples", type=int, default=10000)
    ap.add_argument("--ci", type=float, default=95.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return
    if not args.per_frame_csv:
        ap.error("--per-frame-csv is required (or use --self-test)")

    result = analyze(args.per_frame_csv, n_resamples=args.n_resamples,
                     ci=args.ci, seed=args.seed)
    print_report(result)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
