# scripts/geometric/one_time/

One-shot analyses that back a **pipeline design decision or parameter choice**, rather than a per-bag
measurement. They are kept for reproducibility and are bag-parametrised (`--bag`), but — unlike the
main-directory analyses (`mitigation_analysis.py`, `single_row_analysis.py`, which *are* re-run per
bag to check that the state gate, the geometry filter and the abstention behaviour hold up in a new
season) — these are **not** re-run routinely for each new bag/season: the decision they inform is
made once and applies to the pipeline design.

## Placement criterion
"Would we routinely re-run this on every bag?" — **No** → here. **Yes** → parent `scripts/geometric/`.

## Convention
- Bag-parametrised and reproducible (take `--bag`, resolve paths via `bag_config`).
- Import shared pipeline modules from the parent `scripts/geometric/` (`row_model`, `bag_config`,
  `projection_calibration`, `single_arm_dryrun`). Because `one_time/` adds a directory level, these
  scripts use `PKG = Path(__file__).resolve().parents[3]` and load `row_model.py` from
  `Path(__file__).resolve().parent.parent / "row_model.py"`.
- The findings they produce are **recommendations for the pipeline design**, not per-bag results.

## Inhabitants
- `near_seed_sensitivity.py` — sweeps how far ahead the row fit looks for its seed detections
  (`row_model.NEAR`, default 5 m) and picks the window that best balances the two effects that
  trade off against each other: widening it recovers frames the pipeline would otherwise abstain
  on, but costs accuracy on frames that already fit well. The 5 m window is a one-time pipeline
  design choice, not a per-season quantity.
  → `results/geometric/{bag}/final/{bag}_evaluation/near_seed_sensitivity.json`.
