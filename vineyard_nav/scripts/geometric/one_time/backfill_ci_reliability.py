"""Backfill the D053 CI-reliability fields into the four committed bags' report JSONs.

PURE RECALCULATION — no pipeline stage is re-run. `decorr_m` and `mean_spacing_m` are already
stored in every committed report, and `samples_per_decorr` is their quotient, so nothing here
touches frames, inference or analysis. The three geometric reports per bag carry an IDENTICAL
`config.block_lengths` block, so all three are updated to keep them consistent;
`command_evaluation/command_smoothness.json` is deliberately NOT touched — it borrows the geometric
L rather than deriving its own per-pair structure and carries its own separate caveat.

Key ORDER matches block_lengths.pooled_block_lengths() exactly, so a future re-run of analyze.py
reproduces these files rather than reordering them (verified by --verify against a live re-derivation).

  python3 scripts/geometric/one_time/backfill_ci_reliability.py --dry-run   # show the diff only
  python3 scripts/geometric/one_time/backfill_ci_reliability.py             # write
  python3 scripts/geometric/one_time/backfill_ci_reliability.py --verify    # re-derive and compare
"""
from __future__ import annotations
import sys, json, copy, argparse, difflib
from pathlib import Path

PKG = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PKG / "scripts" / "geometric"))
from block_lengths import MIN_SAMPLES_PER_DECORR  # noqa: E402

BAGS = ["march", "april", "may", "june"]
REPORTS = ["line_fit_report", "paired_crossarm", "config_analysis"]
ADDED_PAIR_KEYS = ("samples_per_decorr", "resolution_limited")


def augment(bl: dict, live: dict) -> dict:
    """Return a NEW block_lengths dict carrying the D053 fields taken from `live`, a fresh
    re-derivation by pooled_block_lengths(). Using `live` rather than recomputing from the stored
    numbers matters: `mean_spacing_m` is stored rounded to 4 dp, and dividing by the rounded value
    differs from the estimator's full-precision quotient by up to 0.01 — enough that the file would
    not match a future re-run. Asserts that every PRE-EXISTING field in `live` equals the committed
    one, so this also proves the committed block reproduces exactly."""
    committed = copy.deepcopy(bl)
    assert strip(live) == committed, "live re-derivation disagrees with the committed block_lengths"
    out = {}
    for k, v in committed.items():
        if k == "per_pair":
            out["ci_reliability"] = copy.deepcopy(live["ci_reliability"])
            pp = {}
            for pair, entry in v.items():
                e = copy.deepcopy(entry)
                for metric in ("GT1", "GT2"):
                    for kk in ADDED_PAIR_KEYS:
                        e[metric][kk] = live["per_pair"][pair][metric][kk]
                pp[pair] = e
            out[k] = pp
            continue
        out[k] = v
    return out


def strip(bl: dict) -> dict:
    """Inverse of augment(): remove every added key, for proving nothing else changed."""
    bl = copy.deepcopy(bl)
    bl.pop("ci_reliability", None)
    for entry in bl["per_pair"].values():
        for metric in ("GT1", "GT2"):
            for k in ADDED_PAIR_KEYS:
                entry[metric].pop(k, None)
    return bl


def paths():
    for b in BAGS:
        for r in REPORTS:
            yield b, PKG / f"results/geometric/{b}/final/{b}_evaluation/{r}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    if a.verify:
        from bag_config import resolve
        from block_lengths import pooled_block_lengths
        print("Re-deriving block lengths live and comparing to the committed files:")
        for b in BAGS:
            B = resolve(b, "eligible")
            man = json.load(open(B["manifest"]))
            live = pooled_block_lengths(B["per_frame_csv"], man)
            filed = json.load(open(PKG / f"results/geometric/{b}/final/{b}_evaluation/line_fit_report.json"))
            filed = filed["config"]["block_lengths"]
            same = json.dumps(live, sort_keys=True) == json.dumps(filed, sort_keys=True)
            print(f"  {b:7} live == committed : {same}")
            if not same:
                for ln in difflib.unified_diff(json.dumps(filed, indent=2).splitlines(),
                                               json.dumps(live, indent=2).splitlines(),
                                               "committed", "live", lineterm="", n=1):
                    print("   ", ln)
        return

    from bag_config import resolve
    from block_lengths import pooled_block_lengths
    live_cache = {}
    nadd = nfile = 0
    for b, p in paths():
        if b not in live_cache:                       # one live re-derivation per bag, reused
            B = resolve(b, "eligible")
            live_cache[b] = pooled_block_lengths(B["per_frame_csv"], json.load(open(B["manifest"])))
        orig_txt = p.read_text()
        doc = json.loads(orig_txt)
        # idempotent: strip any previously-added fields so re-running cannot double-apply or drift
        doc["config"]["block_lengths"] = strip(doc["config"]["block_lengths"])
        orig_txt = json.dumps(doc, indent=2)
        doc["config"]["block_lengths"] = augment(doc["config"]["block_lengths"], live_cache[b])
        new_txt = json.dumps(doc, indent=2)
        # PROOF: stripping the added keys must reproduce the original document exactly
        chk = json.loads(new_txt)
        chk["config"]["block_lengths"] = strip(chk["config"]["block_lengths"])
        assert chk == json.loads(orig_txt), f"{p} — non-additive change detected!"
        d = list(difflib.unified_diff(orig_txt.splitlines(), new_txt.splitlines(),
                                      "before", "after", lineterm="", n=0))
        adds = [l[1:] for l in d if l.startswith("+") and not l.startswith("+++")]
        dels = [l[1:] for l in d if l.startswith("-") and not l.startswith("---")]
        # A removed line is acceptable ONLY as comma reflow: appending a key after an existing
        # entry turns `"L": 9` into `"L": 9,`. Anything else is a real edit and must abort.
        reflow = [x for x in dels if (x + ",") in adds]
        real_dels = [x for x in dels if x not in reflow]
        assert not real_dels, f"{p} — genuine deletions: {real_dels[:3]}"
        genuine_adds = [x for x in adds if x.rstrip(",") not in dels]
        # every genuine addition must mention one of the new keys
        bad = [x for x in genuine_adds
               if not any(k in x for k in ADDED_PAIR_KEYS + ("ci_reliability", "min_samples_per_decorr",
                                                             "reliable", "GT1", "GT2", "{", "}"))]
        assert not bad, f"{p} — unexpected added content: {bad[:3]}"
        nadd += len(genuine_adds); nfile += 1
        print(f"  {b:7} {p.name:20} +{len(genuine_adds)} new lines, "
              f"{len(reflow)} comma-reflow, {len(real_dels)} real deletions")
        if not a.dry_run:
            p.write_text(new_txt)
    print(f"\n{'DRY RUN — nothing written' if a.dry_run else 'WROTE'} "
          f"{nfile} files, {nadd} added lines, 0 deletions, 0 modifications")


if __name__ == "__main__":
    main()
