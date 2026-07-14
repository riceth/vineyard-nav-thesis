# Superseded — geometric strand, March bag

Retired artefacts, retained as an audit trail. Nothing here feeds a reported
result; each entry records a state the pipeline evolved past.

## `yconstant_val_evaluation/` — buggy Y-constant CP-5 run

- `yconstant_val_report.json`, `yconstant_val_per_frame.csv`

The first CP-5 val evaluation, using the **near-5 m Y-constant median** row fit
(D035). The global-median Y landed in the *gap* between the true-row cluster and
adjacent-corridor detections, biasing the centreline. Superseded by the hybrid
clustering + RANSAC + far-field extension + line-fit centreline pipeline
(**D036–D038**); the corrected run is `../final/val_evaluation/line_fit_val_report.json`
(offset RMS 0.33 → 0.23 m). Kept to document the bug and its magnitude.

## Note — the superseded CP-3 *row model* is elsewhere

The CP-3 dry run also used a now-superseded row model, but its artefacts
(`../single_arm_dryrun_report.json`, `../single_arm_dryrun_samples/`) remain at **top level** as a
locked, committed historical state (commit 798d7d4, DECISIONS.md D035). They are
deliberately **not** moved here — moving committed checkpoint artefacts would
rewrite locked state. See `../README.md`.
