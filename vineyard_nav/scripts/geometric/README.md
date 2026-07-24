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
python3 scripts/geometric/line_fit_eval.py    --bag march   # -> line_fit_report.json
python3 scripts/geometric/paired_crossarm.py  --bag march   # -> paired_crossarm.json
python3 scripts/geometric/config_analysis.py  --bag march   # -> config_analysis.json
```

Each reads `results/geometric/march/final/march_evaluation/line_fit_per_frame.csv`
(already in the repo) and rewrites its report JSON. Swap `--bag march` for
`--bag april` to check the second bag. Runtime: under a minute each.

---

## The full pipeline — April worked example

Run these **in order**. Each step names the file it needs from the previous
step and the file it produces, so nothing is ambiguous. Long GPU steps are
marked ⏳ and should be launched detached (see [Gotchas](#gotchas)).

### Stage A — prepare the bag

**A1. Convert the ROS1 bag to ROS2** (the pipeline reads a ROS2 `.db3`, the BLT
bags ship as ROS1):

```bash
python3 scripts/geometric/convert_bag.py --bag april
```
- **Needs:** `../kg_april_06.bag` at the repo root; ~110 GB free disk.
- **Produces:** `../kg_april_06_ros2/kg_april_06_ros2.db3` (gitignored).
- **Runtime:** tens of minutes, I/O-bound. April's output was a 116.5 GB `.db3`
  containing 24,355 camera frames.
- **You should see:** `[april] done -> …db3 (116.5 GB)` and a "next: contamination_census" hint.

**A2. CP-0 — contamination census** (finds which bag frames overlap the
perception training scenes, so they can be excluded from evaluation):

```bash
python3 scripts/geometric/contamination_census.py --bag april
```
- **Needs:** the `.db3` from A1; the SemanticBLT dataset at the repo root.
- **Produces:** `results/geometric/april/contamination_census_exclusions.json`.
- **Runtime:** ~1 minute.
- **You should see (April):** `10/10 located (high 10, low-verified 0); … 310 frames (1.3% of bag)`.

**A3. CP-1 — build the frame manifest** (classifies every frame as in-row /
headland / stationary / contaminated and segments the in-row passes):

```bash
python3 scripts/geometric/frame_manifest_build.py --bag april
```
- **Needs:** the `.db3` (A1) and the census (A2).
- **Produces:** `results/geometric/april/dataset_manifest.json` + `manifest_summary.json`.
- **Runtime:** ~1–2 minutes.
- **You should see (April):** `12 in-row passes detected … ELIGIBLE 8889 frames`.
  (12 is the *reproduction* count, not 12 physical traversals — see D046a. A bag
  with a configured `expected_passes` will instead assert that count and abort if
  it differs, which is the intended safety behaviour.)

**A4. Extract the in-row frames** as 640×640 JPEGs for inference:

```bash
python3 scripts/geometric/extract_frames.py --bag april --scope eligible
```
- **Needs:** the `.db3` (A1) and the manifest (A3).
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
- **Needs:** the frames (A4) and the 3 Phase-C checkpoints.
- **Produces:** `results/geometric/april/cache/detections.csv` (gitignored, ~16 MB).
- **Runtime:** ~15 minutes (GPU). April cached 717,763 detections.

**B2. ⏳ Run the 9-model line-fit inference** (the pipeline's longest step — all
three arms × three seeds over every eligible frame):

```bash
python3 scripts/geometric/line_fit_infer.py --bag april
```
- **Needs:** the frames (A4) and all 9 checkpoints.
- **Produces:** `results/geometric/april/final/april_evaluation/line_fit_per_frame.csv`.
- **Runtime:** ~40 minutes (GPU). April wrote 80,001 rows (9 models × 8,889 frames).
- **You should see:** one `[april/eligible][<arm> s<seed>]` line per model, then
  `wrote … (80001 rows = 9 models x 8889 frames)`.

### Stage C — in-row analyses (repo-only, fast)

Each reads the per-frame CSV from B2. None needs the GPU except C6.

```bash
python3 scripts/geometric/line_fit_eval.py            --bag april   # C1 headline accuracy   -> line_fit_report.json
python3 scripts/geometric/paired_crossarm.py          --bag april   # C2 cross-arm CIs        -> paired_crossarm.json
python3 scripts/geometric/config_analysis.py          --bag april   # C3 class-config sweep   -> config_analysis.json  (reads detections.csv from B1)
python3 scripts/geometric/single_row_analysis.py      --bag april   # C4 abstention mechanism -> single_row_analysis.json
python3 scripts/geometric/lidar_crosscheck.py         --bag april   # C5 camera-vs-LiDAR tilt -> lidar_crosscheck.json  (reads the .db3 LiDAR topic)
```
- **Runtime:** seconds to ~1 minute each.
- **C5 note:** samples the true mid-pass of one traversal per corridor (D047);
  needs the `.db3` for the LiDAR point cloud, so it is the one Stage-C step that
  needs the bag present, not just the CSV.

**C6. ⏳ Near-seed window sensitivity** (a one-off study, not a headline step):

```bash
python3 scripts/geometric/one_time/near_seed_sensitivity.py --bag april
```
- **Needs:** the frames (A4) and all 9 checkpoints.
- **Produces:** `final/april_evaluation/near_seed_sensitivity.json`.
- **Runtime:** ~35 minutes (GPU) — builds its own base-point cache then sweeps.
  **Resumable:** if interrupted, re-run the same command and it continues from
  the cached model-streams (this was needed repeatedly — see Gotchas).

### Stage D — non-in-row characterisation (the deployment-gap branch)

This measures what the in-row pipeline does when driven over *headland* frames
it was never meant for. It mirrors A4/B2 with `--scope non_in_row`:

```bash
python3 scripts/geometric/extract_frames.py   --bag april --scope non_in_row   # D1  -> shared frames dir
python3 scripts/geometric/line_fit_infer.py   --bag april --scope non_in_row   # D2 ⏳ -> non_in_row_evaluation/line_fit_per_frame.csv
python3 scripts/geometric/non_in_row_analysis.py --bag april --scope non_in_row  # D3 -> non_in_row_analysis.json
python3 scripts/geometric/mitigation_analysis.py --bag april                    # D4 -> mitigation_evaluation/mitigation_analysis.json
```
- **D1 runtime:** ~5–6 minutes (April: 15,156 non-in-row frames).
- **D2 runtime:** ⏳ ~45 minutes (GPU) — slower per frame than in-row (headland
  frames yield more spurious detections). April wrote 136,404 rows.
- **⚠️ D3 REQUIRES `--scope non_in_row`.** Without it the script reads the *in-row*
  CSV against the non-in-row frame set (disjoint by construction) and aborts with a
  clear message. This is intentional (a guard added after the mistake was made).
- **D4** reads both the in-row and non-in-row CSVs; it takes `--bag` only, no scope.

### Stage E — figures

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

### Stage F — hand off to the control strand

The command-level strand runs on this strand's outputs (F026's full validation
needs the Stage-D non-in-row results, so run Stage D first). See
`scripts/control/README.md`.

---

## Gotchas (all observed during the April run)

- **The dev-container connection drops.** Long GPU steps (⏳) die if run in the
  foreground and the connection drops. Launch them detached:
  ```bash
  setsid nohup python3 scripts/geometric/line_fit_infer.py --bag april > /tmp/infer.log 2>&1 < /dev/null &
  ```
  and check progress with `tail -f /tmp/infer.log`. `near_seed_sensitivity.py`
  and `extract_detections.py` are resumable; `line_fit_infer.py` is not, so a
  dropped connection means re-running that step from scratch.
- **`non_in_row_analysis.py` needs `--scope non_in_row`** (Stage D3). It fails
  loudly without it.
- **Analysis order matters within a bag** but stages are independent across bags:
  you can run all of March then all of April; you cannot run C1 before B2.

---

## Script reference

**Pipeline (this directory) — reproduces the committed results.** Bag-agnostic
(`--bag <name>`); paths resolved via `bag_config.py`; frames selected on
`eligible` alone (whole-bag pooling, no val/test split, for statistical power).

| Script | Role | Stage |
|---|---|---|
| `convert_bag.py` | ROS1→ROS2 conversion (thin `rosbags-convert` wrapper; disk check; skip-if-done) | A1 |
| `contamination_census.py` | CP-0 — locate + exclude perception-training scenes | A2 |
| `frame_manifest_build.py` | CP-1 — classify frames, segment in-row passes, pick the Δs=1.5 m subsample | A3 |
| `extract_frames.py` | decode eligible / non-in-row frames to 640² JPEGs + QA overlays | A4, D1 |
| `extract_detections.py` | Phase-C per-detection cache → `cache/detections.csv` | B1 |
| `line_fit_infer.py` | 9-model line-fit inference → `line_fit_per_frame.csv` (12-col) | B2, D2 |
| `line_fit_eval.py` | per-frame CSV → per-arm accuracy + bootstrap CIs | C1 |
| `paired_crossarm.py` | same-frame paired cross-arm differences (common error cancels) | C2 |
| `config_analysis.py` | class/threshold config sweep + single-class ablations | C3 |
| `single_row_analysis.py` | in-row abstention mechanism (why a frame yields no centreline) | C4 |
| `lidar_crosscheck.py` | camera row-heading vs LiDAR, mid-pass anchors (D047) | C5 |
| `non_in_row_analysis.py` | headland output distribution + driven-path error (needs `--scope non_in_row`) | D3 |
| `mitigation_analysis.py` | odometry state gate + geometry filter rejection rates | D4 |
| `figures.py` | the 15 committed per-bag report figures (per-bag frame registry) | E |
| `figures_compare.py` | cross-bag comparison figures (`--bags …`) | E |

**Shared modules (imported, not run directly):** `bag_config.py` (per-bag paths;
add a bag = one `BAGS` entry), `row_model.py` (the row fit; `exec`'d by drivers +
figures), `projection_calibration.py` (image→ground IPM), `single_arm_dryrun.py`
(shared constants + the superseded dry-run output), `block_lengths.py` (per-bag
CI block lengths, shared so estimators can't drift), `paths.py` (legacy constant,
superseded scripts only).

**`one_time/`** — one-off studies, not pipeline steps (`near_seed_sensitivity.py`
= Stage C6; `unattributed_scene_probe.py` = the O019 attribution probe).

**`diagnostics/`** — dev/investigation scripts (`autocorrelation_block_analysis.py`
measures decorrelation distance and cross-checks the block lengths; `slope_analysis.py`
found the ~2.3° common tilt; the `figure_rowfit_*` scripts predate `figures.py`
and regenerate older dev-era validation PNGs, not the committed report figures).

**`superseded/`** — the retired val/test-split evaluators (11 scripts), kept as an
audit trail after the whole-bag pooling change. Not on the reproduction path.
