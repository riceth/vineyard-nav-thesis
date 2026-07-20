# Superseded — geometric scripts

Retired scripts, kept for audit; **not** part of the reproduction path.

## Row model — near-5 m Y-constant

- `yconstant_val_eval.py` — the original nine-model val evaluator, which modelled each row as a
  constant lateral position across the near 5 m instead of fitting a sloped line. Superseded
  by the current row model, which seeds on the densest near-field cluster, extends the fit to
  agreeing far-field detections, and fits a sloped line per row. The current evaluator is
  the whole-bag pair `../line_fit_infer.py` + `../line_fit_eval.py`; the superseded
  output it produced lives at
  `results/geometric/march/superseded/yconstant_val_evaluation/`.

## Val/test-split evaluators — superseded by the whole-bag pipeline

The 11 scripts below were the per-split (val / test) March evaluators. The split was dropped and
every in-row frame pooled into a single evaluation — the split had served its purpose (locking the
configuration without leakage) and pooling gives a larger sample and tighter confidence intervals.
They were consolidated into the bag-agnostic pooled scripts in `../` (`--bag`):

| Superseded (per-split) | Replaced by (whole-bag) |
|---|---|
| `line_fit_val_eval.py`, `line_fit_test_eval.py` | `../line_fit_infer.py` + `../line_fit_eval.py` |
| `paired_crossarm_val.py`, `paired_crossarm_test.py` | `../paired_crossarm.py` |
| `config_sweep_val.py`, `config_ablation_val.py`, `config_ablation_test.py` | `../config_analysis.py` |
| `lidar_crosscheck_val.py`, `lidar_crosscheck_test.py` | `../lidar_crosscheck.py` |
| `extract_detections_val.py`, `extract_detections_test.py` | `../extract_detections.py` |

Their `parents[N]` root resolution and output paths were re-pointed on the move
(outputs → `results/geometric/march/superseded/march_val_test_split/`), so they
still resolve from this location; they are retained only as an audit trail. The
val/test artefacts they produced live at
`results/geometric/march/superseded/march_val_test_split/{val,test}_evaluation/`.

**Note:** `single_arm_dryrun.py` is deliberately **not** here. The **script** stays
in the pipeline because the current drivers import shared constants (`CONF`,
`BLOB_FRAC`, `FRAME_PX`, `side_valid`, `bin_centre`) from it. What is superseded is
only the **report that script produced** — the near-5 m Y-constant row-model
results — **not the script itself**. See `../README.md` for the module/output
split.
