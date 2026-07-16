# Superseded — geometric scripts

Retired scripts, kept for audit; **not** part of the reproduction path.

- `yconstant_val_eval.py` — the first CP-5 val evaluator, using the near-5 m
  Y-constant row model (D035). Superseded by the hybrid clustering + far-field
  extension + line-fit centreline pipeline (D036–D038). The current evaluator is
  `../line_fit_val_eval.py`; the superseded output it produced lives at
  `results/geometric/march/superseded/yconstant_val_evaluation/`.

**Note:** `single_arm_dryrun.py` (the CP-3 dry-run) is deliberately **not** here —
it stays in the pipeline because the current drivers import shared constants
(`CONF`, `BLOB_FRAC`, `FRAME_PX`, `side_valid`, `bin_centre`) from it. Only its
*output* (the CP-3 report) is superseded. See `../README.md` for the module/output
split.
