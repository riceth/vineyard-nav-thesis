# Superseded — geometric scripts

Retired scripts, kept for audit; **not** part of the reproduction path.

## Row model — near-5 m Y-constant (D035)

- `yconstant_val_eval.py` — the first CP-5 val evaluator, using the near-5 m
  Y-constant row model (D035). Superseded by the hybrid clustering + far-field
  extension + line-fit centreline pipeline (D036–D038). The current evaluator is
  the whole-bag pair `../line_fit_infer.py` + `../line_fit_eval.py`; the superseded
  output it produced lives at
  `results/geometric/march/superseded/yconstant_val_evaluation/`.

## Val/test-split evaluators — superseded by the whole-bag pipeline (D040)

The 11 scripts below were the per-split (val / test) March evaluators. Under
**D040** the val/test split was pooled into a single whole-bag evaluation, and
these were consolidated into the bag-agnostic pooled scripts in `../` (`--bag`):

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

**Note:** `single_arm_dryrun.py` (the CP-3 dry-run) is deliberately **not** here —
it stays in the pipeline because the current drivers import shared constants
(`CONF`, `BLOB_FRAC`, `FRAME_PX`, `side_valid`, `bin_centre`) from it. Only its
*output* (the CP-3 report) is superseded. See `../README.md` for the module/output
split.
