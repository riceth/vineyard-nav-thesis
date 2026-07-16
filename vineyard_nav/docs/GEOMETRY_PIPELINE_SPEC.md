# GEOMETRY_PIPELINE_SPEC.md — Geometric-strand evaluation on bag camera frames

**Status:** APPROVED — all D-A…D-G decisions locked. **D-B resolved 11 July 2026** from Polvara et al. 2024 Table 3 (§6, §12). Not implemented; no code written. **CP-0 is unblocked** — held only for this spec-update review. Supersedes the earlier "map-23-test-scenes-to-bag-timestamps" sketch.
**Date drafted:** 11 July 2026
**Depends on:** D014 (three-strand eval), D031 (cross-arm ranking deferred to geometric strand), D026 (downstream sweep 3 configs × T grid), O010 (this pipeline), PHASE_C_SPEC §8.

---

## 0. Change of approach (why this spec exists)

The original plan evaluated the geometric strand by mapping our **23 labelled test scenes** to bag timestamps and reading the teleoperator pose at each. Task-1 inventory showed this is fragmented and small-N: only **10 of 23 test scenes are March** (bare-vine) and can come from this March bag at all; the other 13 (1 april, 3 may, 9 canopy `color_image_*`) are from **other sessions not on disk**.

**Revised approach (this spec):** evaluate the trained models **directly on the bag's own front-camera stream** — 16,656 continuous frames, each already time-synchronised with a robot pose. This gives large-N, continuous coverage on the exact sensor stream the models will consume in deployment. Training-set frames are **excluded** to avoid contamination; the remainder is split into val (sweep) and test (final).

This approach is **bag-agnostic**: if april/may/summer bags surface later, the same pipeline applies to them (extending coverage to canopy conditions), so it is non-blocking on Riccardo's reply.

---

## 1. Data inventory (measured — see Task-1 report for full detail)

**Bag:** `/workspaces/dissertation/kg_march_23.bag` (ROS1 v2.0, 62.3 GB) — authoritative. ROS2 conversion `kg_march_23_ros2/` (`.db3`, 59 GB) is faithful and **more convenient** (SQLite lets us pull one topic without reading image blobs; verified `/robot_pose` and image data match the ROS1 bag exactly). No other bags on disk.

**Site:** Ktima Gerovassiliou vineyard, Greece (GPS 40.45 N, 22.92 E); robot: Thorvald. Session **2022-03-23 11:50:38 → 12:09:26 UTC**, duration **1127.3 s (18.79 min)**. Bare-vine (March) throughout.

**Capture is synchronised:** all 35 topics carry exactly **16,656 messages at ~14.77 Hz** over the full run (every camera frame has a co-timestamped pose — convenient; but no higher-rate pose for interpolation).

**Camera used = `/front/zed_node/rgb/image_rect_color/compressed`** (confirmed: front frames are the in-row training view; side camera shows a nursery/staging area). **1920×1080 JPEG.** Intrinsics from `/front/zed_node/rgb/camera_info`: **fx = fy = 1057.0, cx = 952.2, cy = 553.6**, frame `front_left_camera_optical_frame`.

**Trajectory = `/robot_pose`** (`geometry_msgs/Pose`, GPS-fused global pose == `/odometry/gps` position). Verified over the full run: **16,656 poses, continuous, 0 gaps > 0.5 s, 0 position jumps > 1 m, z ≡ 0** (2-D). Path length 535.7 m; the run is **~11 pass-traversals over ~5 corridors** (~53 m long, spaced **~2.45 m** centre-to-centre), with headland turns at each end. Substantial stationary periods (31 % of the run; median per-step displacement ≈ 0 m — GPS-stepped).

**Measured parameter — corridor spacing ≈ 2.45 m.** This is a **measured vineyard ground truth** (from the `/robot_pose` pass x-positions ≈ −2.2 / −4.6 / −7.1 / −9.5 / −12.0 m), **not an assumption** (it corrects an earlier ~4 m eyeball estimate). It feeds the **D-G half-spacing prior (~1.2 m, §6a)** and is a useful reference for future **row-spacing-adaptive control**. The precise metric corridor *width* (for the single-row prior) is measured on two-row frames after D-B (§6a).
- Alternatives (not used): `/odometry/base_raw` (wheel — drifts), `/front|side/zed_node/pose` (ZED VO — local, from origin). GPS is **GBAS-augmented** (`/gps/fix` status = 2, RTK-like). **D-A [LOCKED]:** `/robot_pose` is the reference (§10).

**Camera extrinsics are NOT in the bag** (TF tree is `map→odom→base_link→leg/wheel` + `map→topo_map` + an isolated `side_left_camera_frame→side_imu_link`; the front camera has no `base_link` transform). **Resolved via Polvara et al. 2024 Table 3** (§12) — same robot/vineyard/rig: **base_link → Zed2 Front = translation (0.345, 0.060, 0.763) m, quaternion (0, 0.017, 0, 1.000) ≈ 2° downward pitch**, camera 0.763 m above ground (base_link at ground level, paper §3.1.1). See §6, **D-B [RESOLVED]** (§10).

**Cross-checks against Polvara et al. 2024** (same rig/site/campaign): corridor spacing **2.45 m** (our measurement) is consistent with the 5-corridor path in the paper's Fig 3; corridor **driving speed 0.6 m/s** (paper; our when-moving measurement ~0.68 m/s is consistent, mildly GPS-jitter-inflated); **RTK-GNSS ground-truth accuracy ~2–3 cm** (paper — the floor for our RMS lateral error, §5); **GPS→map datum** lat 40.45025 / lon 22.9243 / orientation 0.0 (paper); **March–April are structurally similar** (dormant plants) — useful for interpreting any per-month geometric-error patterns.

**Detection density on bag frames** (Phase C `best.pt`, conf 0.25, 40 frames spread across the run): **trunk mean 17.1 / frame (median 17, range 0–32), pole mean 13.2 (median 13, range 1–21), ≈ 8.5 trunks per side (range 0–28)**. 0/40 frames empty. Model runs cleanly on bag imagery. (Grounds the T-grid, §8.)

---

## 2. Contamination: which bag frames are training data, and how we exclude them

**Finding (verified by frame-matching):** the SemanticBLT **March** frames ARE extracted from this bag. `march_color_image_17` matches bag frame **#7069** (normalised-grayscale corr **0.890**, visually identical scene); `march_color_image_16` → #6997 (0.799). The non-March frames (april/may/canopy) are from other sessions and do **not** appear here.

**Mapping is non-linear** (index 16 → frame 6997, index 17 → frame 7069, ~72 frames / ~5 s apart), so there is no simple index formula — **each labelled March frame must be located by content match**.

**Preprocessing recovered:** the dataset used **stretch-to-640×640** (1920×1080 → 640×640, no crop) — STRETCH beat centre-CROP correlation on every test frame. Bag frames fed to the models must use the **same stretch resize**.

**Exclusion procedure (contamination guard):**
1. Enumerate every **March** SemanticBLT frame across **train + val + test** (unique scenes; the augmented copies collapse to the same scene). [Count to confirm at CP-0; ≈ 420 train + 20 val + 10 test March *frames*, far fewer unique scenes.]
2. For each, frame-match against the bag front stream (coarse every-Nth descriptor search → fine local search, as prototyped) to recover its bag frame index / timestamp `t_k`. **Search heuristic (not a locked assumption):** Riccardo recalls the March frames came from roughly the *first 5–6 min* of the bag — used only to prioritise the search window as an accelerator. **The empirical match governs**: our prototype matches actually landed at ~8 min (`march_color_image_16` → #6997 / t+7.9 min; `_17` → #7069 / t+8.0 min; corr 0.80 / 0.89), *outside* the 5–6 min window — so the full stream is searched and the empirical `t_k` is authoritative wherever it lands.
3. Exclude an **exclusion window** `[t_k − w, t_k + w]` from the eligible set (default `w = 1.0 s` ≈ 15 frames, since adjacent frames are near-duplicate views; **D-C**, §10). Log every excluded interval.
4. Report residual risk: any match with corr below a threshold (e.g. < 0.6) is flagged as "unlocated" and its neighbourhood treated conservatively.

**Note:** train frames were seen in training; val frames drove checkpoint selection; the 23-scene test set is our reserved perception test. **All three** are excluded from the geometric eval set so the geometric numbers are on genuinely unseen frames. Rule 5 (test set untouched) is preserved — we never re-score the labelled test set here.

---

## 3. Eligible-frame selection & val/test split

From 16,656 frames, build the **eligible set**:
- **Remove contamination windows** (§2). **D-C locked: `w` = 1.0 s (≈ ±15 frames)** around each frame-matched March scene. (Contamination confirmed: `march_color_image_17` → bag frame #7069 at corr 0.890; #16 → #6997. All March-labelled train/val/test scenes are frame-matched, each ±1.0 s excluded.)
- **Remove non-in-row frames:** headland turns and manoeuvres. From `/robot_pose` (D-A): 31 % of the run is stationary/slow, in **30 stop segments almost all at row ends** (y ≈ ±44 / +6). Segment into in-row passes vs headland (low speed + high heading-rate at row ends); only in-row frames are eligible. Headland frames → edge case (§7), excluded from the primary metric.
- **Remove stationary frames:** smoothed speed < v_min (≈ 0.10 m/s) — redundant, no lateral-tracking signal.
- **Remove degenerate-perception frames** at eval time (§7: < N detections per side).

**Frame accounting (operational reference; canonical source DECISIONS.md D041).** The 16,656 bag frames partition into exactly three mutually-exclusive, exhaustive buckets (contamination-first ordering; verified against `dataset_manifest.json` — pairwise-disjoint, zero uncovered):

| Category | Count | % | Definition (manifest flags) | Treatment |
|---|---|---|---|---|
| In-row eligible | **7,857** | 47 % | `inrow ∧ ¬contaminated` | evaluated (whole-bag pooled, D040; F010–F018) |
| Contaminated (CP-0 leakage) | **2,958** | 18 % | `contaminated` (taken first) | excluded from all evaluation (§2, D-C) |
| Non-in-row | **5,841** | 35 % | `¬contaminated ∧ headland` | deployment-gap characterisation (Commit 6, F020+) |
| **Total** | **16,656** | 100 % | | |

Non-in-row = 3,946 row-end stops (`headland ∧ stationary`) + 1,895 turns / corridor transitions (`headland ∧ moving`). Sum 7,857 + 5,841 + 2,958 = 16,656. By construction `inrow ⟹ ¬stationary` (in-row |v_y| > 0.30 m/s > v_min 0.10 m/s), so the eligible set = `inrow ∧ ¬contaminated`. Rationale + the metric-interpretation distinction (in-row centreline RMS vs non-in-row *driven-path error*) are in **D041**.

**Dual-mode use of the retained data (D-D — LOCKED; subsample Δs = 1.5 m, data-driven).** The exclusions above define the **eligible set** (~11.8k frames across val + test — order-of-magnitude; exact count at CP-1). The eligible set is used **two ways** — this is a change only to *how retained data is used*, **not** to what is excluded (the 1.0 s contamination window and headland/stationary removal are unchanged):

- **Point estimates** — RMS lateral error per arm, and per sweep combination — are computed over **ALL eligible frames** (no subsampling), to use all available signal for the tightest measurement.
- **Bootstrap 95 % CIs (D020)** are computed on a **spatially-independent subsample at Δs = 1.5 m (~350 frames)**, because bootstrap assumes independent units and adjacent 15 Hz frames are near-duplicates. This preserves honest uncertainty without constraining the underlying measurement.

*Rationale:* subsampling exists only to satisfy the bootstrap independence assumption; it should not shrink the data behind the point estimate. Using both gives tight point estimates **and** honest CIs.

**Why Δs = 1.5 m, spatial not temporal.** Speed varies substantially (31 % stationary/slow; **~0.6 m/s corridor driving**, Polvara et al. 2024 — so **~4.1 cm between adjacent frames** at 14.77 Hz, and **Δs = 1.5 m ≈ 37 bag frames**; `/robot_pose` GPS-stepped), so a fixed Δt would oversample the long headland stops and undersample fast segments. Measured falloff (14 in-row anchors, SSIM + ORB vs pose separation): **SSIM is uninformative** — it saturates ~0.46 even for adjacent frames (high-texture scene decorrelates globally), so the 0.85 rule never applies; **ORB good-match count is the criterion and drops below 100 at Δs ≈ 1.5 m** (438 at < 0.25 m → 117 at 1.0–1.5 m → 91 at 1.5–2.0 m). Conservative alternative 2.0 m. Δs = Euclidean `/robot_pose` displacement.

**Val/test split (D-D → PASS-LEVEL per D033; supersedes the earlier corridor-level split).** The split unit is one **pass** (an individual corridor traversal), not the corridor. The run has **11 passes across 5 corridors** (~2.45 m centres). **Val: 7 passes (p2, p4, p5, p6, p7, p8, p10) = 4,708 frames** (corridors 0, 1, 3); **Test: 4 passes (p0, p1, p3, p9) = 3,149 frames** (corridors 2, 3, 4) — 60/40 by frames; Δs = 1.5 m subsample val 179 / test 106. **Corridor 3** (42 % of eligible, traversed 4×) is **deliberately split across val (1,890) and test (1,412)** so the downstream sweep (config*/T*) cannot overfit to a corridor the test set lacks; **no corridor exceeds 45 %** of its split. Corridors 2 and 4 have a single pass each, so they cannot be in both splits (val-only 0,1; test-only 2,4) — an asymmetry additional bags (April onward) will progressively address (D033). Test evaluated **once** at locked (config*, T*, conf*) — rule 5 analogue. Passes/corridors segmented from `/robot_pose`.

---

## 4. Pipeline architecture

Per eligible frame `i` (independent of arm — A/B/C differ only at step 1):

1. **Perception.** Run the arm's model on the stretch-resized 640×640 frame → foreground masks (U-Net) or instance masks + classes (YOLO B/C) at the locked operating point (conf* = 0.25 for YOLO, D030).
2. **Ground-contact extraction (D039).** The **only arm-specific stage**. **YOLO (B, C):** each instance's **bbox-bottom-centre** (640² → back-mapped to 1920×1080 for projection). **U-Net (A):** connected-component the foreground (8-connectivity), each component ≥ 40 px contributes its **bbox-bottom-centre**. All later steps are arm-agnostic. (U-Net gives ~27 pts/frame vs ~31–33 for YOLO — a genuine binary-vs-instance difference, reported at CP-5.)
3. **Image → ground projection** (§6): map each base point from image pixels to metric ground coordinates in **base_link ground plane** (X forward, Y left, Z = 0).
4. **Per-side clustering** (D026 downstream): assign ground points to **left row** / **right row** (sign of Y, or 1-D clustering on Y). Config A/B/C decide which class drives the fit (trunk-primary / pole-primary / class-agnostic) with fallback threshold **T** (instance count per side).
5. **Row model — hybrid clustering + far-field extension (D036, D037; supersedes the D035 near-5 m Y-constant median).** Project base points out to **X ≤ 10 m** (§6). Per side: (a) **seed** on the near field (X < 5 m) by sliding a **0.5 m Y-window** and taking the densest window's median; (b) **RANSAC-refine** — best row Y over seed ± 0.3 m (0.05 m steps), inliers within ± 0.25 m; (c) **far-field extension** — add dots at 5 m ≤ X ≤ 10 m within **± 0.5 m** of the row Y (same row), rejecting off-Y dots (fan / adjacent corridor); (d) **sanity** — reject if |Y| > 3 m, < 3 inliers, or X-span < 1 m; (e) **adjacent-corridor** clusters (secondary, higher |Y|) are logged and rejected. A **15 % bbox-area blob guard** (D035, retained) drops gross whole-frame blobs. Output: per-side inlier sets → two-row / single-row / none. *Why not the D035 near-5 m median:* the 5 m cutoff discarded valid same-row dots (frame 4107 — 6/8 left-row dots at X > 5 m) and the global median landed in the gap between the true row and adjacent corridors (frame 4223); hybrid + far-extension lifts two-row coverage **64 % → 83 %** (+1 030 frames, 0 lost).
6. **Centreline, offset & heading — line-fit (D038; supersedes the D035 bin-centre).** Fit **Y = mX + c per side (least squares)** on the inlier set (step 5); the centreline is the **midline** of the two fitted lines:
   - **lateral offset** `d̂_i` = centreline Y at **X = 2 m** (Pure-Pursuit look-ahead, **D-E**; +Y = left). Evaluating at the look-ahead removes the ~0.20 m range-bias the D035 range-averaged median incurred under the systematic ~2.3° tilt (offset RMS 0.33 → 0.23 m).
   - **heading** `ψ̂_i` = **centreline slope** in degrees (arctan of the centreline dY/dX) — physically grounded; the far-extension adjacent-rejection makes the per-row slope reliable (only **0.3 %** of frames flag |m| > 0.3), resolving the D035 fan-corruption concern.
   - **width** = mean `Y_L` − mean `Y_R` (rows parallel; for the D-G prior).
   - **quality flags** (logged): steep slope |m| > 0.3, L/R mismatch |m_L − m_R| > 0.2, fit failure.
   A **systematic ~2.3° common tilt** (m_centre = +0.040 ± 0.026 over 3 910 frames), most consistent with a small unmodelled camera yaw (§6), is common to all arms → **cancels in paired cross-arm differences** (D038).
7. **Ground-truth pairing.** Read `/robot_pose` at frame `i`'s timestamp; derive the reference (§5) for `d_i^gt`, `ψ_i^gt`.
8. **Aggregate** over the test set → RMS lateral error, heading error, coverage, with bootstrap CIs (D020). Repeat per arm → cross-arm ranking (D031, the deferred headline comparison).

**Outputs:** a processed `(frame, timestamp, pose, d̂, ψ̂, n_left, n_right, per-side fit residuals)` table per arm; diagnostic overlays for a sample.

---

## 5. Metric definition — RMS lateral error (precise; **D-F DECIDED**)

**Coordinates & units.** All geometry in the **base_link ground plane**, SI metres / degrees. X forward, Y left (ROS REP-103). Signed lateral offset: +Y = row centre is to the robot's left.

**Per-frame estimate.** `d̂_i` = signed Y of the estimated centreline at look-ahead `L_ahead` (§4.6), metres.

**Ground truth (decision D-F — DECIDED 11 July 2026).** We do **not** have surveyed vine-row positions, so an *absolute* lateral ground truth is not directly available: the robot always rides its own driven path, so the trajectory cannot independently say "how far the robot was from the true corridor centre." The adopted resolution:

- **(GT-1) Teleoperator-centred convention — PRIMARY (adopted).** Assume the teleoperator kept the robot on the corridor centre, so `d_i^gt = 0`. Then **RMS lateral error = √(mean_i d̂_i²)** over independent test frames — i.e. how far each arm's perceived centre sits from where a competent human actually drove. A per-pass constant bias (systematic human offset) is estimated as the mean of `d̂` over the pass and reported separately so it is not mistaken for random error. *Limitation:* includes the human's centring imperfection as a common noise floor — **acceptable because all three arms share it, so the cross-arm ranking (the study's actual goal, D031) stays valid.**
- **(GT-2) Heading error — COMPLEMENTARY (adopted; line-fit slope, D038 — supersedes the D035 bin-centre), always reported alongside GT-1.** `ψ̂_i` is the **centreline slope** — arctan of the line-fit centreline's dY/dX (§4.6), degrees. Once the far-extension (D037) rejects adjacent-corridor dots, the per-side line slope is reliable (0.3 % of frames flag |m| > 0.3), so the centreline slope is a physically-grounded heading. Because the centreline is expressed in `base_link` (whose orientation is the robot's `/robot_pose` heading), **GT-2 heading error = √(mean_i ψ̂_i²)** in degrees under the same teleoperator-centred convention as GT-1 — `ψ̂_i` compared against the robot's heading (base_link +X). A **systematic ~2.3° component** — a likely small camera-yaw extrinsic offset (§6, D038) — is common to all three arms and **cancels in the paired cross-arm comparison**; absolute GT-2 includes it. Assumption-light orientation cross-check; adds no scope.
- **(GT-3) LiDAR-referenced centre — DEFERRED to future work.** The bag carries `/scan`, `/merged_cloud`, `/os_cloud_node/points`; a LiDAR row/trunk detector would give an **independent** vine-row geometry → an assumption-free lateral GT. Deferred to preserve the project timeline; documented in Methodology and Discussion as the more rigorous alternative available with additional LiDAR-pipeline work.

**Decision D-F (adopted; to document in the Methodology chapter).** Primary = GT-1 (teleoperator-centred lateral error); complementary = GT-2 (heading error, reported alongside); GT-3 deferred to future work. **Rationale:** GT-1 approximates the practical navigation question ("does our estimated centreline agree with expected driving behaviour?") under an explicitly acknowledged assumption ("the teleoperator drove near the corridor centre"), is feasible with bag data alone, and aligns with standard modular-navigation evaluation practice. GT-2 adds an orientation cross-check at no extra scope. GT-3's LiDAR pipeline is deferred to protect the timeline, flagged as the assumption-free upgrade path. The GT-1 assumption is stated explicitly wherever the lateral number is reported.

**Reporting — dual-mode (D-D).** For each reported split, the RMS is a **point estimate over all that split's eligible frames** (no subsampling) paired with a **95 % bootstrap CI over that split's Δs = 1.5 m subsample**. The **headline cross-arm figures use the test corridors** (all test-eligible frames for the point estimate; the test subsample for the CI); the sweep uses val (§8). Format, e.g.:
> *"Phase B: RMS lateral error 0.XX m (point estimate over all N_test eligible test frames); 95 % bootstrap CI [0.YY, 0.ZZ] computed on the Δs = 1.5 m test subsample."*

For scale, the full eligible set is ≈ 11,800 frames (val + test), ≈ 350 at the Δs = 1.5 m subsample; the test corridors are ~2/5 of these (exact counts at CP-1).

**Ground-truth floor (RTK-GNSS) — per bag.** The `/robot_pose` RTK-GNSS reference has a **published per-session localisation error** (Polvara et al. 2024 **§5.3**, p12 — this is in the §5.3 text, *not* Table 4, which is weather conditions). Use the **operative floor for the bag under evaluation**:

| Session bag | RTK-GNSS ground-truth floor |
|---|---|
| **March (this bag)** | **3.8 cm** (0.038 m) |
| April | 0.8 cm (0.008 m) |
| May | 4.4 cm (0.044 m) |
| June | 1.0 cm (0.010 m) |
| July, September | *not published in the paper's four evaluated sessions* → use the campaign **average ~2–3 cm** (§3.3.3) as a documented estimate, noting the session-specific floor is unpublished |

**Interpretation rule:** any arm's RMS lateral error **at or below the operative floor** for that bag is reported as *within RTK-GNSS ground-truth uncertainty*, not a distinguishing result. **For our March bag the floor is 3.8 cm.** (Thin-structure perception error is expected well above this, but it bounds what the metric can resolve.)

- **Primary (two-row frames, D-G tier 1):** RMS lateral error from the **line-fit centreline at `L_ahead` = 2 m** (D038) [+ 0 m and 3 m by re-evaluating the same lines], + **two-row coverage X %** (both sides fittable — ~83 % on Phase C s42 val after D037).
- **Secondary (single-row frames, D-G tier 2):** RMS lateral error using the half-spacing prior (§6a), reported separately (its own coverage) — at **both** prior values (1.2 m trajectory-anchored primary; projection-consistent sensitivity = half the measured width). *Note (D038):* with the line-fit row model the measured width is now **≈ 2.56 m** (vs D034's 1.91 m), so the projection-consistent half-spacing is **≈ 1.28 m** — it now nearly **coincides** with the 1.2 m trajectory prior, so the D-G two-value spread has largely collapsed.
- **Complementary:** RMS heading error (deg, GT-2); per-pass lateral bias (GT-1).
- **Cross-arm:** paired-difference bootstrap CIs (as in F001/F003, on the paired subsample), not overlapping single-arm CIs.

---

## 6. Image-to-world projection via camera calibration

**Intrinsics (measured, in bag):** K = [[1057.0, 0, 952.2],[0,1057.0,553.6],[0,0,1]] at 1920×1080. If perception runs on the 640×640 **stretched** image, either (a) project using original-resolution pixel coordinates (recommended — map detections back to 1920×1080 before projection, undoing the anisotropic stretch sx=1920/640, sy=1080/640), or (b) rescale K anisotropically to 640×640. **(a) is cleaner** and avoids distorting K.

**Extrinsics (D-B — RESOLVED 11 July 2026, from Polvara et al. 2024 Table 3, §12).** The BLT paper documents this exact robot / vineyard / sensor rig. **base_link → Zed2 Front frame:**
- Translation: **x = 0.345 m (forward), y = 0.060 m (lateral), z = 0.763 m (up)** — camera 34.5 cm forward, 6 cm to one side, **0.763 m above ground** (base_link is at ground level, paper §3.1.1).
- Rotation quaternion (x, y, z, w) = **(0, 0.017, 0, 1.000)** → **~2° downward pitch** (approximately level).

This gives a principled `T_base←cam` from those six DOF. (The mount is in no public repo — the L-CAS `ros2_zed_multi_camera` package only positions cameras relative to a `zed_multi_link`, and the robot-level attachment lives in a non-public BLT bringup — so **Table 3 is the authoritative source**.) The earlier empirical-homography fallback is **no longer needed**.

**Projection model (flat-ground assumption).** With **h = 0.763 m**, **pitch θ ≈ 2° down**, and K, each image base point `(u,v)` back-projects to a ray; intersect with the ground plane `Z = 0` (base frame) → `(X, Y)` metric — equivalently a planar homography `H` from image to ground. **Given the small pitch, either the full 6-DOF model or a simplified pinhole-plus-ground-plane model is valid** (both acceptable per Riccardo). The 0.345 m forward / 0.060 m lateral offsets shift the origin from base_link to the camera and are applied when expressing offsets in the base frame. Validity: **flat local ground** — reasonable within a row segment; headland slopes are an edge case (§7).

**Sanity check (CP-2):** project detections on a frame; confirm the left/right rows land ~2.45 m apart (the trajectory-measured corridor spacing) and parallel.

**Half-spacing prior (D-G secondary tier) — two-value (CP-2 complete, D034).** The single-row fallback places the centre at the detected row ± half the corridor width. The prior is **reported at two values side by side** to make the projection-vs-truth sensitivity transparent: **1.2 m (primary, trajectory-anchored)** — corridors ~2.45 m centre-to-centre from `/robot_pose` (x ≈ −2.2/−4.6/−7.1/−9.5/−12.0 m), reflecting true vineyard geometry — and **0.96 m (sensitivity, projection-consistent)** — half the CP-2 projection-measured corridor width (median 1.91 m over 22 well-detected val frames).

**Projection range (D037).** Base points are projected out to **X ≤ 10 m** (not just the near 5 m). The far-field dots are admitted to a row only if within ± 0.5 m of the near-field row Y (D037), so the longer range adds real same-row support without re-admitting adjacent corridors.

**Known limitation 1 — projection width (CP-2; largely resolved by D036–D038).** The CP-2 sanity check measured corridor width **median 1.91 m** (~22 % narrow vs 2.45 m). This was **largely adjacent-corridor + far-field-fan contamination of the near-8 m line fit used at CP-2**, not a pure projection error: with the revised row model (hybrid D036 + far-extension D037 + line-fit D038) the measured width is **median ≈ 2.56 m, IQR [2.43, 2.76]** — near the true 2.45 m. The narrowing is symmetric and never biased the two-row centreline (midpoint preserved). A residual sub-cm pitch/height offset may remain; true-ground-contact detection is still a future refinement.

**Known limitation 2 — systematic ~2.3° tilt (likely camera yaw; D038).** Across 3 910 two-row val frames the fitted rows share a **common tilt** (m_L +0.036, m_R +0.045, m_centre +0.040 ± 0.026 → centreline heading ≈ **+2.31°**), while the corridor **width stays parallel** (width-slope ≈ 0). A pitch/height error would bend the width (not observed); a **yaw** error rotates the whole ground projection, tilting both rows the same way with width preserved — exactly what is seen. Table 3 (D-B) encodes pitch (q_y = 0.017) with **q_z = 0 (zero yaw)**, so a small unmodelled yaw (~2°) is the leading explanation (partly confounded with the robot's real mean heading-to-row, unresolvable without `/robot_pose` yaw). It is a **projection effect common to all 9 models**, so **paired cross-arm differences cancel it**; absolute GT-1/GT-2 include it and it is stated wherever the number is reported. Removal path: extrinsic re-calibration adding yaw, or the LiDAR GT (GT-3).

---

## 7. Edge cases

| Case | Detection | Handling |
|---|---|---|
| **Few detections on a side** (< N, e.g. N=2) | count per side after clustering | can't fit that side's line → frame **abstains** from the lateral metric (counted in *coverage*, not error). Report coverage per arm — an arm that abstains often is penalised there, not by silently dropping frames. |
| **Both sides sparse / 0 detections** | total < threshold | frame excluded from metric; logged. (0/40 sampled frames were empty, so rare on this bag.) |
| **Single-row / ambiguous** (one side fittable) | exactly one side has ≥ N points | **D-G locked — report BOTH tiers, not abstain.** *Primary* metric uses **two-row frames only** (both sides fittable): report **coverage X %** and RMS lateral error. *Secondary*: on **single-row frames**, estimate centre = detected row ± **measured half corridor-width** (half-spacing prior, §6a) → report RMS lateral error **separately**. Mirrors real deployment where a single-row fallback is meaningful. |
| **Sharp turns / headland** | `/robot_pose` heading-rate high, low speed, row-end | excluded from primary metric (not in-row); optionally reported as a separate "manoeuvre" stratum. Flat-ground projection also least valid here. |
| **Side asymmetry** (one row denser than the other) | per-side count imbalance (seen: 3 vs 28 trunks) | RANSAC per side is independent, so asymmetry is tolerated; but a near-empty side reduces to the single-side case. Config A/B/C fallback (T) governs which class carries a sparse side. |
| **F007 blob** (whole-canopy false mask, YOLO) | one huge mask | its base point projects to a single off-row outlier → **rejected by RANSAC**; verify the RANSAC tolerance actually rejects it (CP-3). This is where instance-seg's failure mode meets the geometry stage. |
| **Stretch distortion** | — | undo anisotropic stretch before projection (§6a). |
| **Stationary frames** | speed < v_min | excluded (redundant). |

---

## 8. Sweep design (D026), grounded in measured bag densities

Measured per-side trunk count on bag frames: **mean 8.5, median 8, range 0–28**; poles fewer. The fallback threshold **T** (use primary class if per-side count ≥ T, else fall back) should span the region where the decision realistically flips.

- **Provisional T grid = {1, 2, 3, 5, 8, 12}** instance counts per side (D026) — well-centred on median 8; covers the sparse-side regime. Given the observed max ~28, optionally extend with {16, 20} to probe dense frames. **Finalise at CP-4** after measuring the per-side count distribution on the **val split** specifically (not just the 40-frame probe).
- **Configs (D026):** A = trunk-primary / pole-fallback; B = pole-primary / trunk-fallback; C = class-agnostic (single pool, no T).
- **Selection (dual-mode, D-D):** rank all {config × T} combinations by **full-frame val-mean RMS lateral error** (all eligible val frames — maximal signal for ranking), then **verify (config*, T*) is robust** against the **Δs = 1.5 m subsampled val bootstrap CIs** (the winner must not be a subsample artefact; if the top combinations' CIs overlap heavily, prefer the simpler config). Total val evaluations: 6 (A) + 6 (B) + 1 (C) = 13 on the base grid (~18 with the extended T-grid).
- **Test:** evaluate **once** at locked (config*, T*) per arm. conf* = 0.25 fixed (D030); no per-frame re-selection.
- **Arm applicability:** the config/T sweep is the **Phase C (multiclass)** contribution mechanism. Phase B (binary) and Phase A (binary) have a single class → Config C (class-agnostic) only; they provide the baselines the Phase C configs are compared against (attribution story, D026).

---

## 9. Implementation checkpoints (gates — hold at each)

- **CP-0 — Contamination census.** Enumerate all March train/val/test frames; frame-match each to the bag (**first-5–6-min heuristic as a search accelerator; empirical match governs** — prototype matches landed at ~8 min); produce the exclusion-window list + an unlocated-frames report. Gate: exclusion coverage acceptable, residual risk quantified.
- **CP-1 — Extraction.** Extract all 16,656 `(frame_640, timestamp, pose)` triples to a processed dataset (~2.3 GB JPEG q90 / ~20 GB raw; ~5–15 min). Gate: continuity check (no missing frames), pose join correct, sample overlays look right.
- **CP-2 — Projection calibration. [DONE]** Applied the **Table 3 extrinsics** (base_link → cam: 0.345, 0.060, 0.763 m; pitch 1.95°) + bag intrinsics + Z=0 ground plane; validated on 22 well-detected val frames — parallel rows, correct centreline. **Row width median 1.91 m, IQR [1.59, 2.45]** (~22 % narrower than the 2.45 m trajectory spacing — *symmetric*, so the two-row centreline metric is unaffected; **known limitation §6, D034**). D-G half-spacing prior: **1.2 m / 0.96 m** (two-value). Module `scripts/geometric/projection_calibration.py`; report `results/geometric/march/projection_calibration_report.json`.
- **CP-3 — Single-arm dry run. [DONE; row model since superseded — D036–D038].** Full pipeline (Phase C seed 42) over all **4 708 val frames**; committed as a locked historical state (commit 32de7c8, `scripts/geometric/single_arm_dryrun.py`, `single_arm_dryrun_samples/`). It locked the **near-5 m Y-constant** row model (D035) at 64.0 % two-row coverage and confirmed the **blob finding** (F007 canopy-blob pathology absent on bare-vine March — largest detections ~10.5 % are real poles; 15 % guard drops 0 real detections, retained). That row model was **superseded** after the CP-5 re-run analysis: near-5 m discarded valid far dots (frame 4107; → **D037** far-extension, 64 → 83 % coverage), the global median landed in inter-cluster gaps (frame 4223; → **D036** hybrid clustering), and the Y-constant model missed a systematic ~2.3° tilt (frame 3998; → **D038** line-fit centreline). Gate: passed (dry-run sanity).
- **CP-4 — Row-model refinement & validation (D036–D038). [DONE]** Between the CP-3 dry run and the 9-model re-run, the row model was rebuilt and validated via single-arm (Phase C s42) val analyses + sample-frame visual review: **D036** (hybrid clustering + RANSAC — frames 4223/3991/4107), **D037** (far-field extension, two-row coverage 64 → 83 %, +1 030 frames, 0 lost — frame 4107), **D038** (line-fit centreline, systematic ~2.3° tilt, offset RMS 0.33 → 0.23 m — frame 3998). Gate: visual confirmation the fit tracks the dot trend (`results/geometric/march/diagnostics/figures/rowfit_validation/`). *(Distinct from the superseded CP-3 row model; this is the step that produced the locked D036–D038 pipeline.)*
- **CP-5 — 9-model val evaluation (line-fit). [DONE]** All 9 checkpoints (A/B/C × seeds 42/43/44) over 4 708 val frames with the locked line-fit pipeline; per-arm/seed coverage, GT-1 (line-fit @ 2 m) + bootstrap CI, GT-2 (slope) + CI, base points, tilt/adjacent/quality-flag stats, deltas vs the superseded Y-constant run. **Its gate analyses (val-side, before test):** the **paired cross-arm difference bootstrap** (the bias-cancelling comparison that neutralises the ~2.3° tilt) and the **Phase C downstream sweep (D026)** that selects the locked (config*, T*). Gate: review val results + locked config before test.
- **CP-6 — Test (once). [pending]** Locked config; all three arms; RMS lateral + heading + coverage + bootstrap CIs; cross-arm paired-difference ranking on the held-out test corridors. Gate: rule-5 analogue (single test evaluation per arm).

**D-F and D-B are both resolved** (§5 metric; §6 extrinsics from Polvara et al. 2024 Table 3). All checkpoint prerequisites are met — **CP-0 is unblocked.**

---

## 10. Decision register (D-A … D-G)

Locked 11 July 2026. **All items resolved and locked**, including D-B (extrinsics from Polvara et al. 2024 Table 3, §6) and the D-D Δs = 1.5 m subsample.

- **D-A — Trajectory reference. [LOCKED]** `/robot_pose` (verified continuous — 0 gaps > 0.5 s, 0 jumps > 1 m). Rationale: `/robot_pose` is the output of the **`robot_localization` EKF (Moore & Stouch 2016, §12)** fusing **wheel odometry + RTK-GNSS**, per Polvara et al. 2024 §3.3.3; the fixed Datum (lat 40.45025, lon 22.9243, orientation 0.0) matches our bag's values. This is not merely methodologically defensible — it is the **published, vineyard-maintainers' own fusion method**. Wheel odom alone drifts; ZED VO is local.
- **D-B — Camera extrinsics source. [RESOLVED 11 Jul 2026]** From **Polvara et al. 2024 Table 3** (§12), same robot/vineyard/rig: base_link → Zed2 Front = translation **(0.345, 0.060, 0.763) m**, quaternion **(0, 0.017, 0, 1.000) ≈ 2° down**, camera 0.763 m above ground. Not in any public repo; Table 3 is authoritative. Empirical-homography fallback no longer needed. Full detail §6; applied at CP-2.
- **D-C — Contamination exclusion window. [LOCKED]** `w` = 1.0 s (≈ ±15 frames) around each frame-matched March scene. Rationale: brackets the near-duplicate neighbourhood of a contaminated frame at 15 Hz. **Window placement is empirical per bag** (via frame-matching, §2); Riccardo's "first 5–6 min" recollection is only a search-space heuristic — our matches actually land at ~8 min, so the empirical `t_k` governs wherever it falls.
- **D-D — Val/test split. [LOCKED — pass-level, D033]** **7 val / 4 test passes** (60/40 by frames; val = p2,p4,p5,p6,p7,p8,p10 → 4,708; test = p0,p1,p3,p9 → 3,149). Split unit is the **pass** (individual traversal), not the corridor — corridor 3 (dominant) is split across both to prevent sweep overfitting; corridors 2,4 are single-pass → split-exclusive. Supersedes the earlier corridor-level split (see D033, §3). Rationale: D028 scene-honest discipline; balances sweep vs test statistical power.
- **D-D — Dual-mode data use. [LOCKED]** **Point estimates over ALL eligible frames (~11.8k)**; **bootstrap CIs over the Δs = 1.5 m spatially-independent subsample (~350)**. Δs = 1.5 m from measured ORB falloff (< 100 matches; SSIM uninformative), spatial not temporal (speed varies, 31 % stationary); conservative alt 2.0 m. Changes *how* retained data is used, not what is excluded.
- **D-E — Look-ahead. [LOCKED]** `L_ahead` = 2 m primary + 0 m (at-robot) secondary; 3 m also reported if cheap.
- **D-F — Ground-truth metric. [LOCKED; GT-2 → line-fit slope, D038 (was D035 centreline-bin)]** GT-1 teleoperator-centred lateral (primary, **line-fit centreline @ 2 m**) + GT-2 heading (complementary, **line-fit centreline slope**); GT-3 LiDAR-referenced deferred. GT-2 evolved D035 (fan-free centreline-bin) → **D038** (line-fit slope) once the far-extension cleaned the inliers. Rationale §5; pipeline lock D036–D038 (DECISIONS.md).
- **D-G — Single-row handling. [LOCKED — two-value prior, D034; spread now collapsed, D038]** Report BOTH tiers: primary two-row (coverage ~83 %, RMS) + secondary single-row with a half-spacing prior at **two values**: **1.2 m** (trajectory-anchored) and the projection-consistent value (half the measured width). *With the line-fit width ≈ 2.56 m (D038, up from CP-2's 1.91 m), the projection-consistent half-spacing is now ≈ **1.28 m** — nearly coinciding with the 1.2 m prior, so the two-value spread has largely collapsed.* See §6 (known limitation 1) + D034 + D038.

---

## 11. What is explicitly out of scope here

Closed-loop control / PID (command-level strand, D014); LiDAR pipeline (unless GT-3 adopted); canopy-condition geometric eval (needs summer bags — pipeline extends to them when available); any change to the labelled 23-scene test set or Phase A/B/C perception artefacts (untouched).

---

## 12. References

- Polvara, R., Molina, S., Hroob, I., et al. (2024). "Bacchus Long-Term (BLT) data set: Acquisition of the agricultural multimodal BLT data set with automated robot deployment." *Journal of Field Robotics*, **41**(7), 2280–2298. DOI: [10.1002/rob.22228](https://doi.org/10.1002/rob.22228). Same robot, vineyard, and sensor rig as this study. Source for: camera extrinsics (**Table 3** → D-B, §6), the robot/sensor rig, corridor driving speed (0.6 m/s, §3.3.2), per-session RTK-GNSS ground-truth floors (**§5.3**: March 3.8 / April 0.8 / May 4.4 / June 1.0 cm → §5) and the campaign average ~2–3 cm (§3.3.3), `robot_localization` wheel-odometry + RTK-GNSS fusion (§3.3.3 → D-A), the 5-corridor path (Fig 3), and the GPS→map datum (lat 40.45025, lon 22.9243, orientation 0.0). Local copy in `.personal`.
- Moore, T. & Stouch, D. (2016). "A generalized extended Kalman filter implementation for the Robot Operating System." In: *Intelligent Autonomous Systems 13*, Springer, pp. 335–348. — The `robot_localization` EKF package Polvara et al. use to fuse wheel odometry + RTK-GNSS into `/robot_pose` (D-A).
