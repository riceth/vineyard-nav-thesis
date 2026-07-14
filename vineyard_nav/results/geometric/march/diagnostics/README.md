# Diagnostics — geometric strand, March bag

Diagnostic evidence generated while developing and validating the geometric
pipeline. Not reported results (those are in `../final/`); kept for traceability
of how the locked pipeline (D036–D038) was arrived at and how the mechanistic
findings were established.

## `figures/rowfit_validation/` — row-fit visual validation

Bird's-eye (X-forward, Y-lateral) plots used at each row-model gate to confirm
the fit tracks the true-row dot trend and rejects adjacent-corridor detections.

| Path | Row-model step | Decision | Findings |
|---|---|---|---|
| `rowfit_f*.png` (top level) | Hybrid clustering + RANSAC seed | D036 | F014 (adjacent-corridor rejection) |
| `far_ext/rowfar_f*.png` | Cluster-consistent far-field extension (project to 10 m) | D037 | F011 (coverage 64 → 83 %) |
| `linefit/linefit_f*.png`, `linefit/slope_hist.png` | Line-fit centreline vs Y-constant | D038 | F010 (systematic tilt) |
| `linefit_final/lfval_f*.png` | Final line-fit val plots | D038 | — |

Sample frames: 3991–3998, 4107, 4223 (single-arm Phase C s42, val).

## Analyses A–I

The numerical diagnostic analyses (A–F noise-floor decomposition and tilt
attribution; G paper benchmark search; H autocorrelation / decorrelation-distance;
I block-bootstrap alternative) were **printed and reasoned through in the session
logs, then written up in `docs/FINDINGS.md` (F010–F019)** — they were not saved as
standalone JSON. The reported statistical artefacts they fed into
(`paired_crossarm_val.json`, `paired_crossarm_test.json`) live in `../final/`.
