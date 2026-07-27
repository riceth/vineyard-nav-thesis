# Geometric strand — scripts

Estimates the vine-row **centreline** the robot would steer down: bag frames →
per-vine detections → ground projection → a straight-line fit to each row →
the midline between them. Runs on any registered bag via `--bag <name>`, and
produces the committed centreline results and figures.

This README is standalone: you can follow it from a clean checkout. **April is
used as the worked example** — every command, runtime, and "you should see"
below is from the actual April run on the project hardware (RTX 5050, 8 GB).
Runtimes are indicative and hardware-dependent.

---

## Before you start

**1. Be in the right directory.** Every command runs from `vineyard_nav/`:

```bash
cd vineyard_nav
pwd     # must end in /vineyard_nav
```

**2. Know which path you're on:**

- **Just verifying the committed numbers?** You need *nothing extra* — skip to
  [Verify committed results without models](#verify-committed-results-without-models).
- **Re-running the pipeline on a bag?** You need two things not in the repo:
  - the bag's ROS1 file at the repo root (e.g. `../kg_april_06.bag`), and
  - the 9 model checkpoints under `results/runs/phase_*` (train them via
    `scripts/perception/README.md`, or obtain them).
  - plus ~150 GB free disk (the ROS2 conversion alone is ~110 GB) and a CUDA GPU.

---

## Verify committed results without models

The per-frame CSVs are committed, so the headline analysis reproduces from the
repo alone — no bag, no weights:

```bash
cd vineyard_nav
python3 scripts/geometric/analyze.py --bag march --only line_fit_eval,paired_crossarm
```

This reads `results/geometric/march/final/march_evaluation/line_fit_per_frame.csv`
(already in the repo) and rewrites `line_fit_report.json` + `paired_crossarm.json`.
Swap `--bag march` for `--bag april` to check the second bag. Runtime: under a
minute. (The other analyses need inputs not in a clean checkout: `config_analysis`
the detection cache from B1, `single_row_analysis`/`lidar_crosscheck` the model
weights / the `.db3` — see the full pipeline below.)

---

## The full pipeline — April worked example

Run these **in order**. Each step names the file it needs from the previous
step and the file it produces, so nothing is ambiguous. Long GPU steps are
marked ⏳ — run them the normal way (foreground is fine); you only need the
detached fallback if your dev-container connection is unreliable and drops
mid-run (see [Gotchas](#gotchas)).

> **Running a different bag?** Every command is bag-agnostic — replace `april`
> with the bag name throughout (e.g. `--bag may`). The bag must be registered in
> `bag_config.py` (march, april, may, june, july, september are) and its ROS1
> `.bag` present at the repo root. The 9 model checkpoints are **bag-independent**
> — no retraining per bag; Stages B–E reuse the same weights. June/July/September
> additionally depend on the D048 gate (A2) clearing `needs_review`.

### Stage A — prepare the bag

**A1. Convert the ROS1 bag to ROS2** (the pipeline reads a ROS2 `.db3`, the BLT
bags ship as ROS1):

```bash
python3 scripts/geometric/convert_bag.py --bag april
```
- **Needs:** the bag's ROS1 file at the repo root (`../kg_<month>_*.bag`); **~1× the bag's size**
  in free disk (the `.db3` is roughly the source size — e.g. April 116.5 GB, May 70 GB).
- **Produces:** `../kg_april_06_ros2/kg_april_06_ros2.db3` (gitignored).
- **Runtime:** tens of minutes, I/O-bound. April's output was a 116.5 GB `.db3`
  containing 24,355 camera frames.
- **You should see:** `[april] done -> …db3 (116.5 GB)` and a "next: prep" hint.

**A2. Prepare the bag — CP-0 census + D048 gate, then CP-1 manifest** (one
command runs both checkpoints in sequence):

```bash
python3 scripts/geometric/prep.py --bag april
```
CP-0 (contamination census) finds which bag frames overlap the perception
training scenes so they can be excluded, in two parts:
1. **Prefix scenes** — the SemanticBLT scenes labelled from this bag's month
   (e.g. `april_*`) are located by thumbnail matching and excluded.
2. **D048 gate** — the 90 unattributed `color_image_*` scenes (no month prefix,
   so part 1 never checks them) are scored against this bag by ORB+RANSAC
   identity: **≥200 inliers → present** (added to the exclusion set), **40–200 →
   `needs_review`**, **≤40 → absent**. Runs for *every* bag, including no-prefix
   bags (june/july/september) — exactly the bags the gate protects. See
   `scene_attribution.py` and DECISIONS.md D048.

CP-1 (frame manifest) then classifies every frame as in-row / headland /
stationary / contaminated and segments the in-row passes — **unless** CP-0
returned `needs_review`, in which case `prep.py` stops after the census and
prints the scenes to confirm (add to exclusions if present, clear if absent,
then re-run).

- **Needs:** the `.db3` from A1; the SemanticBLT dataset at the repo root.
- **Produces (all under `results/geometric/april/`):**
  `contamination_census_exclusions.json` (carries `status: clear | needs_review`
  and a `d048_gate` block), `dataset_manifest.json`, `manifest_summary.json`.
- **Runtime:** ~4–8 minutes (CP-0's gate ORB-verifies a bounded shortlist; CP-1 ~1–2 min).
- **You should see (April):** `prefix 10 located; D048 0 present / 0 needs_review
  / 90 absent; … 310 frames (1.3% of bag)` then `12 in-row passes detected …
  ELIGIBLE 8889 frames`. (All 90 unattributed scenes are confident-absent on
  march/april, D048. The "12" is a *reproduction* count, not 12 physical
  traversals — see D046a; a bag with a configured `expected_passes` asserts that
  count and aborts if it differs.)

**A3. Extract the in-row frames** as 640×640 JPEGs for inference:

```bash
python3 scripts/geometric/extract_frames.py --bag april --scope eligible
```
- **Needs:** the `.db3` (A1) and the manifest (A2).
- **Produces:** `results/runs/geom_cp1_frames_640_april/` (~1.2 GB, gitignored)
  + 4 QA overlay samples under `results/geometric/april/diagnostics/frame_samples/`.
- **Runtime:** ~4–5 minutes.
- **You should see:** `frames: wrote 8889, skipped 0`. Open a QA overlay to
  eyeball that the frame, pose, and flags look sane before spending GPU time.

### Stage B — inference (needs the model weights)

**B1. ⏳ Build the Phase-C detection cache** (runs the 3 multiclass seeds once,
caching per-detection class + base point for the config sweep):

```bash
python3 scripts/geometric/extract_detections.py --bag april
```
- **Needs:** the frames (A3) and the 3 Phase-C checkpoints.
- **Produces:** `results/geometric/april/cache/detections.csv` (gitignored, ~16 MB).
- **Runtime:** ~15 minutes (GPU). April cached 717,763 detections.

**B2. ⏳ Run the 9-model line-fit inference** (the pipeline's longest step — all
three arms × three seeds over every eligible frame):

```bash
python3 scripts/geometric/line_fit_infer.py --bag april
```
- **Needs:** the frames (A3) and all 9 checkpoints.
- **Produces:** `results/geometric/april/final/april_evaluation/line_fit_per_frame.csv`.
- **Runtime:** ~40 minutes (GPU). April wrote 80,001 rows (9 models × 8,889 frames).
- **You should see:** one `[april/eligible][<arm> s<seed>]` line per model, then
  `wrote … (80001 rows = 9 models x 8889 frames)`.

### Stage C — in-row analyses (one command)

All five in-row analyses run from a single driver, `analyze.py`. Each writes the
same report JSON it always did; most read only the per-frame CSV from B2.

```bash
python3 scripts/geometric/analyze.py --bag april
```
Runs, in order:
| name (`--only <name>`) | output | notes |
|---|---|---|
| `line_fit_eval` | `line_fit_report.json` | headline accuracy + bootstrap CIs (CSV only) |
| `paired_crossarm` | `paired_crossarm.json` | same-frame cross-arm CIs (CSV only) |
| `config_analysis` | `config_analysis.json` | class/threshold sweep (also reads `detections.csv` from B1) |
| `single_row_analysis` | `single_row_analysis.json` | abstention mechanism — **⏳ GPU, ~8 min** (re-runs the 9 models on single_row frames) |
| `lidar_crosscheck` | `lidar_crosscheck.json` | camera-vs-LiDAR tilt (reads the `.db3` LiDAR topic) |

- **Runtime:** the four light analyses are seconds each; `single_row_analysis` is
  GPU-bound (~8 min) and dominates. For a fast repo-only subset, select the
  CSV-only ones: `analyze.py --bag april --only line_fit_eval,paired_crossarm`.
- **`--only name[,name]`** runs a subset (comma-separated). `single_row_analysis`
  needs the 9 checkpoints; `lidar_crosscheck` needs the `.db3`; the rest need only
  the committed CSV (and `config_analysis` also the B1 detection cache).

**C6. ⏳ Near-seed window sensitivity** (a one-off study, not a headline step):

```bash
python3 scripts/geometric/one_time/near_seed_sensitivity.py --bag april
```
- **Needs:** the frames (A3) and all 9 checkpoints.
- **Produces:** `final/april_evaluation/near_seed_sensitivity.json`.
- **Runtime:** ~35 minutes (GPU) — builds its own base-point cache then sweeps.
  **Resumable:** if interrupted, re-run the same command and it continues from
  the cached model-streams (this was needed repeatedly — see Gotchas).

### Stage D — non-in-row characterisation (the deployment-gap branch)

This measures what the in-row pipeline does when driven over *headland* frames
it was never meant for. It mirrors A3/B2 with `--scope non_in_row`, then runs the
two non-in-row analyses via `analyze.py --non-in-row`:

```bash
python3 scripts/geometric/extract_frames.py --bag april --scope non_in_row   # D1  -> shared frames dir
python3 scripts/geometric/line_fit_infer.py --bag april --scope non_in_row   # D2 ⏳ -> non_in_row_evaluation/line_fit_per_frame.csv
python3 scripts/geometric/analyze.py        --bag april --non-in-row         # D3+D4 -> non_in_row_analysis.json + mitigation_analysis.json
```
- **D1 runtime:** ~5–6 minutes (April: 15,156 non-in-row frames).
- **D2 runtime:** ⏳ ~45 minutes (GPU) — slower per frame than in-row (headland
  frames yield more spurious detections). April wrote 136,404 rows.

D3+D4 (`analyze.py --non-in-row`) are fast (seconds) and run, in order:
| name (`--only <name>`) | output | notes |
|---|---|---|
| `non_in_row_analysis` | `non_in_row_evaluation/non_in_row_analysis.json` | headland output-class distribution + driven-path error (F020/F021); reads the D2 non-in-row CSV |
| `mitigation` | `mitigation_evaluation/mitigation_analysis.json` | odometry state-gate + geometry-filter rejection (F022/F023); reads **both** the in-row and non-in-row CSVs |

`analyze.py` passes each the correct scope bundle, so the old "forgot `--scope`"
footgun is gone; `--only name[,name]` selects a subset here too.

### Stage E — figures

> **New bag? Per-bag figures need a curated frame registry first.** `figures.py` renders 11
> frame-specific report figures from a hand-curated `FRAMES['<bag>']` entry in `figures.py`
> (representative frames + captions/footers; see the March/April entries). Without it,
> `figures.py --bag <new>` exits gracefully (*"no representative-frame registry for bag …"*) and
> produces no per-bag figures — curate `FRAMES['<bag>']` before this step. `figures_compare.py`
> needs no registry (data-only; runs on any bags whose JSONs exist).

```bash
python3 scripts/geometric/figures.py          --bag april            # 15 per-bag figures
python3 scripts/geometric/figures_compare.py  --bags march april     # 4 cross-bag comparison figures
```
- **`figures.py`** needs the 9 checkpoints (it re-renders per-frame panels) and
  every JSON from Stages C/D. Produces `results/geometric/april/final/figures/`
  (15 PNGs). Runtime ~1–2 minutes. A load-bearing assertion re-checks each
  per-frame figure against the committed CSV and aborts on any mismatch.
- **`figures_compare.py`** is data-only (reads the committed JSONs), extends to
  more bags via `--bags march april may …`. Produces `results/geometric/comparison/figures/`.

> **New bag just finished Stages A–D?** Do both of these in Stage E: (1) render
> that bag's own 15 figures — `python3 scripts/geometric/figures.py --bag <newbag>`;
> and (2) regenerate the cross-bag comparisons *with the new bag added* to the
> `--bags` list — e.g. once May is done:
> `python3 scripts/geometric/figures_compare.py --bags march april may`. Always
> pass **every** evaluated bag to `--bags`: the list is not cumulative across runs,
> each run regenerates the comparison from exactly the bags you name.

### Stage F — hand off to the control strand

The command-level strand runs on this strand's outputs (F026's full validation
needs the Stage-D non-in-row results, so run Stage D first). See
`scripts/control/README.md`.

### Stage G — confirm the bag is done (mandatory gate)

A bag is **not done when its artefacts exist — it is done when its results are also summarised.**

    python3 scripts/geometric/check_bag_complete.py --bag april

Verifies (a) every committed artefact from Stages C–F exists **and** (b) `docs/STATUS.md` carries a
consolidated **"Confirmed on <bag>"** summary bullet (spot-checked, April/May style — one line per
applicable finding F010–F028, control F026–F028 included). Prints `BAG COMPLETE` only when both hold,
a loud `⚠️ NOT COMPLETE` otherwise. `control.py` runs this automatically at the end of Stage F, so you
will see the verdict — **do not consider the bag finished until it passes.** Writing the STATUS summary
is the step this gate exists to enforce (it was missed for May).

---

## Gotchas (all observed during the April run)

- **Fallback if your dev-container connection is unreliable (long ⏳ steps only).**
  Run these steps normally — foreground is fine by default. The one catch is that
  a dropped connection kills a foreground job, and `line_fit_infer.py` is **not**
  resumable (a dropped foreground run means re-running it from scratch), whereas
  `near_seed_sensitivity.py` and `extract_detections.py` **are** resumable. So
  *only if* your connection tends to drop, launch the long steps detached as a
  safety net:
  ```bash
  setsid nohup python3 scripts/geometric/line_fit_infer.py --bag april > /tmp/infer.log 2>&1 < /dev/null &
  ```
  and follow progress with `tail -f /tmp/infer.log`.
- **Non-in-row analyses run via `analyze.py --non-in-row`** (Stage D3+D4) — it
  selects the non-in-row scope internally, so there is no `--scope` to forget.
- **Analysis order matters within a bag** but stages are independent across bags:
  you can run all of March then all of April; you cannot run `analyze.py` before
  `line_fit_infer` (B2) has written the per-frame CSV.

---

## Script reference

**Pipeline (this directory) — reproduces the committed results.** Bag-agnostic
(`--bag <name>`); paths resolved via `bag_config.py`; frames selected on
`eligible` alone (whole-bag pooling, no val/test split, for statistical power).

| Script | Role | Stage |
|---|---|---|
| `convert_bag.py` | ROS1→ROS2 conversion (thin `rosbags-convert` wrapper; disk check; skip-if-done) | A1 |
| `prep.py` | CP-0 (prefix-scene census + D048 unattributed-scene gate) then CP-1 (frame manifest, Δs=1.5 m subsample); stops if the gate flags `needs_review` | A2 |
| `extract_frames.py` | decode eligible / non-in-row frames to 640² JPEGs + QA overlays | A3, D1 |
| `extract_detections.py` | Phase-C per-detection cache → `cache/detections.csv` | B1 |
| `line_fit_infer.py` | 9-model line-fit inference → `line_fit_per_frame.csv` (12-col) | B2, D2 |
| `analyze.py` | all in-row analyses (`line_fit_eval`, `paired_crossarm`, `config_analysis`, `single_row_analysis`, `lidar_crosscheck`); with `--non-in-row`, `non_in_row_analysis` + `mitigation`. `--only name[,name]` selects a subset | C, D3+D4 |
| `figures.py` | the 15 committed per-bag report figures (per-bag frame registry) | E |
| `figures_compare.py` | cross-bag comparison figures (`--bags …`) | E |

**Shared modules (imported, not run directly):** `bag_config.py` (per-bag paths;
add a bag = one `BAGS` entry), `cp3_geometry.py` (the CP-3 locked geometry library —
detection→centreline constants + row-side/centreline functions; imported by the
pipeline and diagnostics), `scene_attribution.py` (the D048 ORB+RANSAC
unattributed-scene gate used by `prep.py` CP-0; validated in
`one_time/scene_attribution_orb.py`), `row_model.py` (the row fit; `exec`'d by
drivers + figures), `projection_calibration.py` (image→ground IPM),
`block_lengths.py` (per-bag CI block lengths, shared so estimators can't drift).

**`one_time/`** — one-off studies, not pipeline steps (`near_seed_sensitivity.py`
= Stage C6; `unattributed_scene_probe.py` = the O019 attribution probe).

**`diagnostics/`** — dev/investigation scripts (`autocorrelation_block_analysis.py`
measures decorrelation distance and cross-checks the block lengths; `slope_analysis.py`
found the ~2.3° common tilt; the `figure_rowfit_*` scripts predate `figures.py`
and regenerate older dev-era validation PNGs, not the committed report figures).

**`superseded/`** — the retired val/test-split evaluators, kept as an audit trail
after the whole-bag pooling change, plus the relocated CP-3 dry-run reproducer
(`single_arm_dryrun.py`, superseded by the real runs) and the legacy `paths.py`
detection-cache shim it uses. Not on the reproduction path.
