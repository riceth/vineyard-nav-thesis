# Riseholme (RH) data manifest

**Compiled** 4 August 2026 · **Updated** 6 August 2026 with `part1_2_9_2025`
**Status:** reconnaissance complete, evaluation **not started**
**Scope:** consolidates read-only diagnostics. No data converted, processed or modified.

> **Changes in this revision.** A third dataset (`part1_2_9_2025`) is added. Two findings from it
> change earlier conclusions: (a) the geojson↔bag **row correspondence is now solved exactly**;
> (b) the camera mounting is **confirmed never to have been published to `tf`**, so the earlier
> hypothesis that a re-recording dropped it is **withdrawn as falsified** — no original fragment
> will supply it.

---

## 1. Bags and dates

| | `part1_2_9_2025` ⭐ | `rh_july2026` | Aug-2024 set |
|---|---|---|---|
| File | `part1_2_9_2025.bag` | `rh_july2026.bag` (was `17_07_26.bag`) | `gps/row_1_to_6.bag`, `gps/row_6_to_1.bag` |
| Size | 22.35 GB | 53.5 GB | 18.1 / 14.2 GB |
| **Recording date** | **2025-09-02, 11:34:46–11:42:31 UTC** | **2026-07-17, 14:58:21–15:20:37 UTC** | 2024-08-02 |
| Duration | 7.91 min | 22.27 min | 3.80 min (row_6_to_1) |
| Messages / topics | 69,470 / 32 | 175,615 / 28 | 374,117 / 107 |
| Provenance | Replay re-recording (+2.10 d) | Replay re-recording (+11.80 d) | Native RealSense recordings |
| Fragment status | **part 1** of a 262-fragment session | fragment **`_99`** of a split session | — |
| Integrity | ✅ index intact | ✅ intact | ⚠️ `row_1_to_6` truncated (0.38 MB tail; 10,073/10,081 chunks intact, **recoverable by reindex — no re-upload needed**) |

**Date verification.** Both filenames were verified against header stamps rather than assumed —
the RH filename convention is `DD_MM_YY(YY)`, which is easy to misread.
`2_9_2025` → 2 Sept 2025 (source `Tue-02-Sep`; 2 Sept 2025 *is* a Tuesday; 12:34 BST = 11:34 UTC).
`17_07_26` → 17 July 2026 (source `Fri-17-Jul`; 17 July 2026 *is* a Friday; 15:58 BST = 14:58 UTC).

---

## 2. Image and pose topics

`part1_2_9_2025` and `rh_july2026` share the same robot and sensor suite. Differences that matter:

| Topic | `part1_2_9_2025` | `rh_july2026` |
|---|---|---|
| `/camera_link_rear/color/image_raw` | 5,707 @ 12.02 Hz | 13,021 @ 9.74 Hz |
| `/camera_link_rear/depth/image_rect_raw` | 8,008 @ 16.87 Hz | 21,325 @ 15.95 Hz |
| `/odometry/gps` | 6,006 @ 12.65 Hz | 13,419 @ 10.04 Hz |
| `/gps/fix` | 1,983 @ 4.18 Hz | 4,540 @ 3.40 Hz |
| `/scan` (2D LaserScan) | 2,955 @ 6.22 Hz | 6,575 @ 4.92 Hz |
| `/amcl_pose` | 298 @ 0.63 Hz | 507 @ 0.38 Hz |
| **`/tf_static`** | ✅ **1 msg** (contents below) | ❌ absent |
| **`/topological_map_2`** | ✅ **1 msg, 111,586 chars** | ❌ absent |
| **`/vineyard/topological_map_2`** | ✅ 1 msg, 75,250 chars | ❌ absent |
| **`/map`** (OccupancyGrid) | ✅ 2493×2148 @ 0.05 m | ❌ absent |
| **`/auto_mode`** | ❌ **absent** | ✅ 14,493 @ 10.84 Hz, **29.6% True** |

Both: colour **1280×720 rgb8 uncompressed**, depth **848×480 `16UC1`**, single **rear-facing** camera,
**no `/imu/data`**, 2D LaserScan only (no 3D cloud).

**Camera identity — all three datasets share one physical unit.** Intrinsics match to three
decimals across every file: colour `fx 908.902, fy 909.155, cx 650.331, cy 363.993`; depth
`fx 425.732, cx 426.044, cy 238.384`. The Aug-2024 bag names it outright:
**Intel RealSense D435I, serial 050222071152**, on an NVIDIA Jetson USB port.

**Aug-2024 bags carry camera streams only** — no robot pose, no `/tf`, no GNSS, no navigation.

---

## 3. Timestamps and synchronisation ⚠️

**Both robot bags are replay re-recordings; message timing is not the robot's timing.**

| | `part1_2_9_2025` | `rh_july2026` |
|---|---|---|
| header.stamp span | 2025-09-02 11:34:46 → 11:42:31 | 2026-07-17 14:58:21 → 15:20:37 |
| bag clock | 2025-09-04 14:04:54 | 2026-07-29 10:11:30 |
| offset | **+2.10 days** | **+11.80 days** |
| **drift across the bag** | **0.69–0.79 s** | **61 s** ⚠️ |
| source | `part1_9_2/vineyard_Tue-02-Sep---12-34_1..262` | `Fri-17-Jul/vineyard_Fri-17-Jul---15-58_99` |

`/rosout` confirms both: a `/play_` node opening source bags, a `/record_` node capturing the replay.

**Use `header.stamp` for anything time-sensitive on either bag.** For `part1_2_9_2025` sub-second
drift means the timing is otherwise faithful. For `rh_july2026`, 61 s of drift over a 1,337 s bag
means the replay stalled or ran off-rate, so bag-time intervals actively misrepresent behaviour —
its continuity figures (8 colour gaps > 2 s) need re-measuring on `header.stamp` before use.

**Aug-2024 sync is unresolved.** RealSense bags use relative timestamps from zero; the `.LLH` GNSS
tracks are absolute UTC. No shared clock established, so frame↔position pairing is not possible.

---

## 4. Camera-to-base transform — ⛔ **BLOCKING, and confirmed unobtainable from the bags**

### The finding

`part1_2_9_2025` **has** `/tf_static`, the recorder **did** subscribe to it, and the replay covered
**260 of 262 source fragments** — essentially the whole session. What it contains:

```
/tf_static : 1 transform
  map -> topo_map    (0, 0, 0)  identity          <- topological-map alignment, not calibration

/tf : 10 distinct pairs across the entire bag
  base_link -> leg0..leg3       (±0.546, ±0.2415, +0.200)   <- URDF
  leg0..3   -> wheel0..3_link   (0, 0, 0)                   <- URDF
  map -> odom,  odom -> base_link                           <- localisation

transforms involving a camera frame          : 0
camera_info frame_ids                        : camera_link_rear_{color,depth}_optical_frame
are those frames anywhere in the tf tree?    : NO
```

`rh_july2026` shows the same: 10 pairs across all 25,063 `/tf` messages, zero camera transforms.

### ⚠️ Earlier hypothesis withdrawn

The previous revision of this manifest concluded the mounting was *"almost certainly published by
the original robot and lost in the re-record"*, and recommended requesting an original fragment.
**That is now falsified.** `part1_2_9_2025` captured `/tf_static` from a near-complete replay and
the camera is still absent, while the leg and wheel URDF transforms **are** present on `/tf`.

**The camera was never published to the robot's TF tree.** It is not a recording artefact, and
**no original fragment will supply it.** The request should be withdrawn.

### External sources exhausted

No Thorvald/RASberry package on our systems. The only public Thorvald URDF
(`okb6/Thorvald_Grape_Urdf`) is a 2.5 KB stub whose single link is `base_link`.

### Interim empirical recovery (`rh_july2026`, n = 29 frame-pairs)

| DOF | Estimate | Spread | Usable? |
|---|---|---|---|
| Height above ground | **1.269 m** | IQR [1.232, 1.298] | ✅ |
| Pitch | **+5.75°** | IQR [4.85, 6.61] | ✅ |
| Roll | +0.98° | IQR [−0.27, +1.84] | ✅ |
| Lateral offset | −0.068 m | IQR [−0.117, −0.008] | ❌ uncertainty ≳ quantity |
| Yaw | +3.21° | IQR [+1.89, +5.16] | ❌ same |

Method: RANSAC ground-plane fit on depth for height/pitch/roll; rows seen simultaneously by `/scan`
(already in `base_link`) and by depth (camera frame) for lateral/yaw. Cross-check: the two sensors
agree on row spacing to 0.164 m (camera 2.181 m vs `/scan` 2.345 m), the gap shrinking with sample
size (0.286 → 0.274 → 0.164 m at n = 4 → 6 → 29) — consistent with canopy-envelope vs trunk-line
rather than method error.

**Lateral offset and yaw bias GT-1 directly. Those two DOF are the blocker.**

**→ Only remaining path: mounting documentation, or a physical measurement. See §9.**

---

## 5. Ground truth — `riseholme.geojson`

**Format:** GeoJSON `FeatureCollection`, **246 features** (not 9). **No `crs` member → WGS84 /
EPSG:4326** per RFC 7946 — the same frame as `/gps/fix`, so no datum conversion needed.

| feature_type | count | geometry |
|---|---|---|
| `topo_map_line` | 120 | LineString |
| `topo_map_interpolated_node` | 81 | Point |
| `topo_map_point` | 18 | Point |
| `topo_map_node` | 18 | Point |
| **`mid_row_line`** | **9** | LineString (2 vertices) |

The 9 mid-row lines run `west_block_row_10→9` … `row_2→1`, each **16.09–16.12 m** measured
(property `length` 16.20–16.23), heading **171.5°**, **centre-to-centre spacing 2.49–2.52 m**.
Each carries 11 topological nodes (`node_start`, `node_0`…`node_8`, `node_end`) = 99 total.
The "9 interpolated points per line" are separate `Point` features (9 × 9 = 81), not line vertices.

**All 9 are `measured_or_calculated: "calculated"`** — derived geometry, not surveyed. Accuracy
inherits from an underlying source the file does not name.

**Corroboration:** geojson spacing 2.49–2.52 m vs 2.345 m measured independently from `/scan` (6%).

---

## 6. ✅ Row correspondence — **SOLVED**

`part1_2_9_2025` carries `/topological_map_2` (108 WayPoints **with map-frame coordinates**) plus
both `/odometry/gps` (map frame) and `/gps/fix` (WGS84). Pairing those on header stamp gives the
map→WGS84 rigid transform (**rotation −0.879°, residual median 25.6 cm, p90 87.6 cm**), which places
the WayPoints directly onto the geojson lines.

```
WayPoints assigned to a mid-row line : 108 / 108
unassigned                           : 0
```

| geojson mid-row line | WayPoints |
|---|---|
| `west_block_row_2 → row_1` | 1–12 |
| `west_block_row_3 → row_2` | 13–24 |
| `west_block_row_4 → row_3` | 25–36 |
| `west_block_row_5 → row_4` | 37–48 |
| `west_block_row_6 → row_5` | 49–60 |
| `west_block_row_7 → row_6` | 61–72 |
| `west_block_row_8 → row_7` | 73–84 |
| `west_block_row_9 → row_8` | 85–96 |
| `west_block_row_10 → row_9` | 97–108 |

**Exactly 12 WayPoints per line, in unbroken consecutive blocks, none left over.**
Deterministic rule: WayPoint *N* lies on the line between `row_(g+2)` and `row_(g+1)`,
where `g = ⌊(N−1)/12⌋`.

**The geojson and the robot's topological map are the same map under two naming schemes.**
The mapping is robust: 1 m assignment tolerance against 2.5 m row spacing means the 25 cm
transform residual cannot misassign a node.

Independently — and without using the fitted transform — the raw GNSS track passes within
**0.00–0.01 m of all nine lines**, 92–119 fixes inside 1 m of each.

### Naming schemes across the three sources (for reference)

| source | `/current_node` | `/vineyard/current_node` |
|---|---|---|
| `rh_july2026` | `aisle_N_node_M` (N = 0..6) | `WayPoint87–108` |
| `part1_2_9_2025` | `WayPoint15–72` | `WayPoint89–108` |
| geojson | `west_block_row_N_to_..._node_K` | — |

No string-level overlap exists between any pair. Correspondence is geometric only.

---

## 7. Position / GNSS precision

| source | status | absolute σ (E/N) | short-scale residual (5 / 9 / 15-fix) |
|---|---|---|---|
| **Ktima july2023** | GBAS | **8 / 8 mm** | **5.5 / 7.0 / 7.5 mm** |
| Aug-2024 `.LLH` tracks | **RTK fixed, 100% of epochs** | 6–12 mm | — |
| **`part1_2_9_2025`** | **GBAS, `fix_type = 4` (RTK fixed)** | **50 / 49 mm** | **39.0 / 45.8 / 55.9 mm** |
| `rh_july2026` | **SBAS** | 575 / 593 mm | 10.1 / 15.2 / 28.0 mm |

Other estimators, `part1_2_9_2025`: `/gps/filtered` σ 0.109 m; `/odometry/gps` σ 0.105 m;
`/amcl_pose` σ 0.113 / 0.134 m; `/odometry/base_raw` σ 3.36 m (wheel only).
`/health/gps/error_std` median 0.060 m, p90 0.114, max 0.209.

**`part1_2_9_2025` is 11× better absolutely than `rh_july2026`** and is the first robot-borne RTK
fix in the set. Two caveats:

- **The ordering inverts at short scale.** `rh_july2026` is worse absolutely but better at 5-fix
  (10.1 vs 39.0 mm) because SBAS solutions are heavily smoothed — large slow bias, little jitter.
  **Short-scale precision is what a centreline comparison depends on**, and 39.0 mm still exceeds
  the 20 mm effect size.
- 50 mm is **poor for an RTK-fixed solution** (1–2 cm typical), suggesting marginal RTK.

---

## 8. Imagery, health, autonomy

| | `part1_2_9_2025` | `rh_july2026` | Aug-2024 |
|---|---|---|---|
| Blank check | ✅ **5,707 / 5,707 distinct** | ✅ real (mean 91–124, std 58–82) | ✅ **2,000 / 2,000 distinct** |
| Pixel stats | mean 63–129, std 43–90 | mean 91–124 | mean 104–114 |
| Gaps > 2 s (camera) | **0** (max 1.90 s) | 8 (max 3.05 s)¹ | — |
| Autonomy | ❓ no `/auto_mode` | **29.6% True** | — |

¹ measured on bag time; needs re-measuring on `header.stamp` (see §3).

All three verified against the august2023 failure mode (a bag whose every frame was one blank white
JPEG, which passed every non-camera health check). **All three are sound.**

The glasshouse visible in `part1_2_9_2025` imagery matches `rh_july2026` — **same site, visually
confirmed** — with ripe fruit consistent with early September.

**Autonomy on `part1_2_9_2025` cannot be determined.** Active topological navigation
(`/closest_edges` at 1.22 Hz) suggests autonomous operation, but that is inference, not evidence.
This matters: O020/D014 frames GT-1/GT-2 against the platform's **autonomous** driven path.

---

## 9. RH vs Ktima — setup differences

| | **Ktima** (march/april/may/june/july2023) | **Riseholme** |
|---|---|---|
| Robot | Thorvald | Thorvald (same family, different configuration) |
| Camera | ZED2 stereo | RealSense **D435I** s/n 050222071152 |
| **Facing** | **Forward** | **Rear** ⚠️ |
| Image topic | `.../compressed` | `.../image_raw` (raw) |
| Resolution | 1920×1080 BGRA | 1280×720 rgb8 |
| Intrinsics | fx = fy = 1057.0, cx 952.2, cy 553.6 | fx 908.902, fy 909.155, cx 650.331, cy 363.993 |
| **Extrinsics** | `T_BASE_CAM` (0.345, 0.060, 0.763), pitch 1.95° — **validated against onboard `tf` to 4 dp (D052)** | **never published to `tf`** |
| GNSS | GBAS/RTK, σ 8 mm | 50 mm (2025) / 575 mm (2026) |
| LiDAR | 3D `/os_cloud_node/points` | 2D `LaserScan` |
| IMU | `/imu/data` | absent |
| Reference | Autonomous driven path (O020/D014) | Driven path *or* geojson mid-row line |

**Consequences.** Nothing runs unchanged. Frame extraction needs a new topic, type and resolution.
The **rear-facing camera inverts the row-fit geometry** — the near/far split (X < 5 m) and the 2 m
look-ahead both assume a forward view. The 3D LiDAR cross-check has no equivalent. CP-0
contamination screening is vacuous (no SemanticBLT scenes for this site). And **without extrinsics
the geometric estimate cannot be produced at all**, labels or not.

---

## 10. Blocking items

| # | Item | Status |
|---|---|---|
| 1 | **Camera→base_link extrinsics** | ⛔ **BLOCKING — confirmed unobtainable from the bags.** Needs mounting documentation or a physical measurement. |
| 2 | Row-ID mapping geojson ↔ bag | ✅ **SOLVED** (§6) |
| 3 | Position precision vs 20 mm effect size | ⚠️ best available 39 mm short-scale; see below |
| 4 | Remaining `part*` files of the 2025 session | ⚠️ requested — this is part 1 of 262 fragments |
| 5 | Autonomy confirmation for the 2025 session | ⚠️ no `/auto_mode` recorded |
| 6 | Aug-2024 camera↔GNSS time sync | ⚠️ unresolved (lower priority — superseded as a candidate) |
| 7 | Aug-2024 marker provenance (3 of 4 FLOAT) | ⚠️ requested (lower priority) |

**On item 3.** The geojson supplies an absolute row-centre reference, which is a genuine
improvement — comparing against the driven path only works if one assumes the robot drove down the
middle of the row, and that assumption disappears. But it does not improve *precision*: comparing a
vision-estimated centreline to the line still requires knowing **where the robot is relative to it**,
which comes from GNSS/AMCL. **The geojson answers "where is the row centre"; it does not answer
"where is the robot".**

Matching `/scan` to the geojson lines for a tighter pose is possible, but the robot already runs
AMCL — scan-matching localisation — and its own reported lateral uncertainty is **0.113–0.134 m**
on the 2025 bag. It would also convert the measurement into a camera-vs-LiDAR agreement study
rather than a comparison against surveyed truth.

**Recommended candidate: `part1_2_9_2025`** — RTK-fixed, continuous, sub-second replay drift,
carries the topological map and occupancy grid, and its row correspondence to the geojson is
solved. Its weaknesses are duration (7.91 min of a much longer session) and unconfirmed autonomy,
both addressable by obtaining the remaining `part*` files.

---

## 11. Rajitha's reply (6 Aug 2026) — reconciliation against the recorded data

Four statements from the person who collected the data, checked against what the bags contain.

### 11.1 "Mounted facing backwards without any angle"

**Partly usable, partly in conflict — our measurement is NOT overwritten.**

| DOF | Rajitha | Measured (rh_july2026) | Status |
|---|---|---|---|
| Yaw | 0° off straight-back | +3.21°, IQR [1.89, 5.16] | ⚠️ mild tension; **our estimate was already declared unusable**, so adopt his 0° provisionally |
| **Pitch** | implied 0° | **+5.75°, 58 of 59 frames positive** | ⛔ **CONFLICT — see below** |
| Roll | implied 0° | +0.45°, 17/59 negative | ✅ consistent with 0 |
| Lateral | not stated | −0.068 m (unusable) | still unknown |
| Height | not stated | 1.269 m ± 3 cm | still unknown from him |

**The pitch conflict is real and terrain does not explain it.** The obvious reconciliation — that our
5.75° is camera-to-**ground** while his claim is camera-to-**robot body**, with the difference being
the slope of the field — was tested directly and **fails**:

```
measured camera-to-ground pitch, 59 fitted frames
    median +5.746 deg   std 1.458   range -1.45 .. +9.27
    NEGATIVE values: 1 of 59

terrain slope, fitted to the bag's own GNSS altitudes
    magnitude 3.78 deg, steepest ascent bearing 30.6 deg
    component along the row axis (171.5 deg): +/-2.93 deg
```

A camera mounted at 0° pitch would read **pure terrain**: a distribution centred on zero swinging
±2.93° as the robot drives up and down the rows, with roughly half the samples negative. Observed is
a **+5.75° offset with 58 of 59 samples positive**. The ±1.46° spread is consistent with terrain
riding on top of a fixed tilt; the offset itself is not.

**Conclusion: there is a genuine ~5.7° downward tilt that his description does not account for.**
Most likely reading is that *"without any angle"* refers to **yaw** (not angled left or right),
leaving pitch unaddressed — or the bracket has an incidental downward tilt he is not aware of.
**Our measurement stands until physically checked.** This is the single number most worth resolving
with a spirit level or a photograph of the bracket.

### 11.2 "The mounting is in /tf and /tf_static"

⛔ **Contradicted by the recorded data — independently re-verified on both bags.**

The re-check deliberately did not reuse the earlier method (which filtered pairs on the substring
`cam` and could in principle miss a differently-named frame). It dumps **every** distinct `frame_id`
appearing anywhere in `/tf` or `/tf_static` and asks whether the frames the camera itself declares
(via `camera_info` `header.frame_id`) appear among them. Result: see §4 — the tf tree contains only
`base_link`, `leg0–3`, `wheel0–3_link`, `map`, `odom`, `topo_map`, and the camera's own declared
frames are absent from every one of them.

**The robot's own URDF data is present and correct** (leg and wheel transforms, plausible geometry),
so this is not a reading failure on our side — the camera simply was not published.

### 11.3 "September 2025 sessions were manually driven; only 2026 was autonomous"

✅ **Consistent with the data, and it has a real consequence for the evaluation.**

Corroborating evidence: `part1_2_9_2025` and `Tue-02-Sep` (both 2 Sept 2025) carry **no `/auto_mode`
topic at all**, while `rh_july2026` records it at 10.84 Hz and shows 29.6% autonomous. The absence
of the flag on the 2025 bags is what one expects if nothing was driving autonomously.

**⚠️ Consequence — the O020 framing does not transfer to the September 2025 bags.**
For Ktima, GT-1/GT-2 are defined as agreement between the vision-estimated centreline and the
platform's **autonomous** driven path (O020, D014). That reference means something specific: the
output of a navigation system attempting to follow the row. **A manually driven path is a different
kind of reference** — it reflects an operator's steering, including deliberate deviation, and cannot
stand in for a navigation system's output.

For the September 2025 bags this leaves two options, and only one is defensible:

1. ❌ **Do not** compare against the driven path and call it GT-1 as defined for Ktima. That would
   silently redefine the measurand mid-study.
2. ✅ Compare against the **geojson mid-row line** — an absolute, survey-derived row centre that
   does not depend on who or what was steering. This is the only reference these bags can support,
   and it is precisely the reference the geojson supplies (§5, §6).

This is arguably an *improvement* in reference quality (it removes the assumption that the driven
path equals the row centre) but it is **a different measurand from Ktima's GT-1**, and any
cross-site comparison must say so explicitly rather than presenting the two as like-for-like.

### 11.4 "Use the July or September 2025 bags — they have GPS + TF"

**September 2025:** verified — two bags in hand (`part1_2_9_2025`, `Tue-02-Sep`). GPS confirmed good
(RTK-fixed, ~50 mm). **TF confirmed to lack the camera** (§4, §11.2).

**July 2025: ⚠️ NOT IN OUR POSSESSION.** No July 2025 bag has ever been supplied. The only files
matching "july" are `kg_july_13` (Ktima, 2022) and `rh_july2026` (Riseholme, July **2026**) — neither
is the July 2025 session. **This is the one recommendation we cannot check**, and it is worth
checking, because if the July 2025 rig differed it is the only remaining candidate that might carry
camera TF. It should be requested explicitly.

---

## 12. Camera calibration — what is LOCKED, and on what evidence (7 Aug 2026)

Three of the six degrees of freedom are adopted as an **empirically derived, cross-verified
calibration**, independent of any answer from the data collector. If his account later disagrees,
that disagreement is itself a finding to investigate — not grounds for having withheld these.

### 12.1 LOCKED

| DOF | rh_july2026 | part2_2_9_2025 | agreement | basis |
|---|---|---|---|---|
| **Height above ground** | 1.269 m (n = 59) | **1.278 m (n = 51)** | **9 mm** | RANSAC ground-plane fit on depth |
| **Pitch (down)** | **+5.75°** (n = 59) | — | terrain excluded, see §11.1 | 58 of 59 samples positive |
| **Roll** | +0.98° | +0.45° | 0.5° | same plane fit |

Height agreeing to **9 mm across two sessions eleven months apart, on different bags, with
different GNSS hardware quality**, is the strongest single calibration result available here.

### 12.2 NOT LOCKED

| DOF | estimate 1 | estimate 2 | disagreement |
|---|---|---|---|
| Lateral offset | −0.068 m (`/scan`, n = 29) | −0.035 m (geojson+GNSS, n = 51) | **33 mm** |
| Yaw | +3.21°, IQR [+1.89, +5.16] (`/scan`) | 0° (collector's account) | IQR **excludes** 0 |

**33 mm exceeds the 19.5–24.1 mm effect GT-1 must resolve**, and the geojson-anchored estimate's
IQR spans 48 cm. Both DOF bias GT-1 directly, so neither is adopted.

### 12.3 ⚠️ Caveat — the two lateral estimates are NOT fully independent

They must not be read as two clean confirmations. Both:

- derive the camera's view of the rows from **the same depth ground-plane fit**, so any bias in that
  fit propagates into both;
- measure the **canopy envelope**, not the trunk line. The camera consistently under-reads row
  spacing — **0.952× on part2** (2.386 m vs the geojson's 2.505 m) and **0.930× on rh_july2026**
  (2.181 m vs `/scan`'s 2.345 m). That ~5% bias is stable across both sessions, which is good
  evidence it is physical (foliage leaning inward) rather than random — but it is a **shared**
  systematic, present in both estimates.

What genuinely differs between them is the **second sensor** used to locate the robot: 2D LiDAR in
`base_link` for one, RTK GNSS against the survey geometry for the other. So they are independent in
their *reference*, and correlated in their *camera-side measurement*. The agreement is meaningful
support, not two-source confirmation in the sense of D052's `tf`-versus-Table-3 check — which
matched to four decimal places between two authoritative sources.

### 12.4 ⚠️ Hardware difference — this is a hardware generalisation test, not only a site one

**The Riseholme camera is not the same model as Ktima's, and it does not face the same way.**

| | Ktima (5 evaluated bags) | Riseholme (all sessions) |
|---|---|---|
| Camera | **Stereolabs ZED2** (stereo) | **Intel RealSense D435I**, s/n 050222071152 |
| Facing | **Forward** | **Rear** |
| Resolution / encoding | 1920×1080 BGRA, compressed | 1280×720 rgb8, raw |
| Intrinsics | fx = fy = 1057.0, cx 952.2, cy 553.6 | fx 908.902, fy 909.155, cx 650.331, cy 363.993 |
| Mounting | (0.345, 0.060, 0.763), pitch 1.95° — validated against onboard `tf` | height 1.269 m, pitch 5.75° — empirical only |
| Depth | stereo | active IR stereo (structured light assist) |

**This must be stated plainly wherever Riseholme results are reported.** Any performance difference
between Ktima and Riseholme confounds at least four factors: site, season, **camera hardware**, and
**viewing direction**. It cannot be attributed to site alone. Framed positively, this makes
Riseholme a **hardware-and-viewpoint generalisation test** as well as a site one — a stronger claim
than site transfer, but only if the confound is stated rather than glossed.

The rear-facing D435i is also **undocumented in any published source** — the BLT dataset
(Polvara 2024) used two ZED2 cameras at both sites, and every published Thorvald + D435i
configuration is forward-facing. It is a local modification.

### 12.5 Derivation and verification record

How each value was obtained, what was checked, and what was ruled out. Recorded so the calibration
can be defended or refuted on its method rather than taken on assertion.

#### Method A — ground-plane fit on depth  (gives height, pitch, roll)

Depth is metric (`16UC1`, millimetres), so the ground plane recovers camera pose relative to the
ground directly, with no scale assumption.

```
per sampled depth frame:
  subsample ::2 -> 424 x 240      (half-res; ample for a plane)
  keep 0.4 m < Z < 6.0 m, LOWER HALF of the image only   (ground, not canopy)
  RANSAC plane fit: 200 iterations, 0.04 m inlier threshold, seed 42, >=300 inliers required
  SVD refit on inliers
  orient normal downward (+Y is down in the optical frame)
  height = |d| ;  pitch = asin(-n_z) ;  roll = asin(n_x)
```

Sampling: **rh_july2026** — 60 frames spread across the session, 59 fitted.
**part2_2_9_2025** — 120 frames, 51 usable after GNSS matching.

#### Method B — cross-sensor row alignment  (gives lateral, yaw)

`/scan` is published in `base_link`, so the same physical rows are observable in a **known** frame
and in the camera frame. The difference isolates the camera's offset. This is what separates the
camera's mounting from the robot's position in the row — the camera alone cannot distinguish them.

```
from /scan  (base_link)     : fit left/right row lines -> midpoint, heading
from depth  (camera-ground) : same, after the Method A plane defines forward/lateral/up
camera offset = (-camera_midpoint) - base_midpoint      [sign flip: rear-facing]
```

#### Method C — geojson + RTK GNSS  (independent second estimate of lateral)

Same camera-side measurement, but the robot is located by **RTK GNSS against the surveyed row
geometry** instead of by LiDAR. `part2` used because its GNSS is RTK-fixed (~50 mm) rather than
`rh_july2026`'s SBAS (~575 mm). 68 frames matched a fix within 300 ms and a mid-row line within
1.5 m; 51 retained after the quality filter below.

#### Verification log

| # | Check | Result | Conclusion |
|---|---|---|---|
| 1 | **Exhaustive `tf` frame dump**, 3 bags — every `frame_id` in `/tf` and `/tf_static` enumerated rather than substring-matched for "cam" | 25,063 + 11,241 + 8,611 messages; 11–12 distinct frames each; camera's declared optical frames absent from all | Camera genuinely not in the tree; not a search artefact |
| 2 | **Falsification of the first hypothesis** — that the re-recording's topic list dropped `/tf_static` | `part1_2_9_2025` *did* capture `/tf_static` from a near-complete replay; camera still absent while leg/wheel URDF transforms present | Hypothesis **withdrawn**. No original fragment will supply it |
| 3 | **Terrain-slope test on pitch** — could a 0° mount plus a sloping field explain +5.75°? | Field slopes 3.78°, ±2.93° along-row → a 0° mount predicts a distribution centred on zero with ~half negative. Observed: median +5.746°, **1 of 59 negative** | Terrain **excluded**; the offset is a real fixed tilt |
| 4 | **Convergence with sample size** — does the camera↔`/scan` row-spacing gap shrink as n grows? | 0.286 → 0.274 → **0.164 m** at n = 4 → 6 → 29 | Shrinking ⇒ earlier gap was noise, not a systematic method error |
| 5 | **Filter sensitivity** — is the lateral scatter outliers or genuine noise? | Tightening the row-gap filter from 0.3 m to 0.2 m **widened** std (0.592 → 0.656) | Genuine measurement noise; cannot be filtered away |
| 6 | **External documentation search** | BLT (Polvara 2024) used two ZED2s at both sites; every published Thorvald + D435i configuration is forward-facing; the only public Thorvald URDF is a 2.5 KB `base_link`-only stub | No external source can supply the mounting; it is a local modification |
| 7 | **Training-contamination check** — is Riseholme imagery in SemanticBLT? BLT covers Riseholme (5 sessions, 2023) as well as Ktima | The 405 month-less SemanticBLT images resolve to **exactly 90** source scenes, matching D048's 90 unattributed; rendering them shows Mediterranean stone buildings, arid ground and red roses at row ends — matching the Ktima july2023 frames, with none of Riseholme's glasshouse or water tank | The 90 unattributed scenes are **Ktima**. Riseholme is genuinely out-of-distribution and uncontaminated |
| 8 | **Cross-session mount stability** | Height agrees to **9 mm** between sessions eleven months apart (1.269 vs 1.278 m) | Evidence the bracket was **not** moved between 2025 and 2026 |

#### What was NOT verified

- **Longitudinal offset** — no method here constrains it. Assumed 0; never estimated.
- **Depth-sensor absolute scale** — taken as factory calibration. If the D435i's depth scale were
  biased, height and the row-spacing ratio would both shift together, and Method A could not detect
  it. The `Depth_Units` value in the Aug-2024 recording (0.001 m) is consistent with default.
- **Whether the collector's stated 0° yaw is compatible with the measured +3.21°** — unresolved,
  and left unresolved rather than reconciled by assumption.

#### Decision provenance

These values were adopted **without waiting for the data collector's confirmation**, on the basis
of checks 1–8 above. If his account later contradicts them, that contradiction is a finding to
investigate on its own terms — the evidence here stands or falls on its method, not on his
agreement. Pending formal entries: **D056** (this calibration), **D057** (reference frame and the
O020 non-transfer), **D058** (code isolation).

---

## 13. The investigation that ended the wait for the data collector (7 Aug 2026)

The calibration blocker had stalled the strand. Rather than wait indefinitely, four lines of
enquiry were run to establish what could be determined without him. This section records them with
their numbers, because the decision to proceed rests on them.

### 13.1 Geometric reverse-engineering — computed, and found too imprecise to use

Camera midpoint between rows vs the robot's position from RTK GNSS against the surveyed line.
Run on `part2` (RTK ~50 mm) rather than `rh_july2026` (SBAS ~575 mm).

```
usable frames                                                  68 of 120
after quality filter (camera row gap within 0.3 m of 2.505 m)  n = 51

camera row gap          2.386 m  vs geojson 2.505 m   ratio 0.952
camera midpoint        +0.136 m  (std 0.495)
robot offset from line -0.021 m  (std 0.296)

LATERAL OFFSET          median -0.0353 m
                        IQR [-0.261, +0.219]   std 0.592   SEM 83 mm
```

**Uncertainty budget:**

| source | contribution |
|---|---|
| RTK GNSS lateral | ±60 mm |
| geojson lines are *calculated*, not surveyed | unquantified |
| **depth row-fit scatter** | **±495 mm — dominant** |
| line assignment + heading noise | small |

The uncertainty is roughly **seven times the estimate**. It establishes the offset is small and near
zero; it cannot distinguish 0 from ±0.26 m. Tightening the filter to ±0.2 m **widened** the spread
(std 0.592 → 0.656, n = 40), so the scatter is genuine measurement noise, not outliers.
**Not usable for GT-1, which needs ~±10 mm.**

### 13.2 "Assume centred" baseline — a reference point, not a result

Setting lateral := 0.000 m and yaw := 180° exactly, the camera's own view would *be* the robot's
row offset:

```
camera says         -0.136 m
GNSS + geojson say  -0.021 m
disagreement         115 mm
```

### 13.3 Cross-reference of three estimates — reported as they fell, not forced

| method | bag | n | lateral offset | IQR |
|---|---|---|---|---|
| `/scan`-anchored | rh_july2026 | 29 | **−0.068 m** | [−0.117, −0.008] |
| geojson + GNSS | part2 | 51 | **−0.035 m** | [−0.261, +0.219] |
| assume centred | — | — | 0.000 | by definition |

They agree in sign and magnitude — all within 7 cm, across two bags, two sessions a year apart, and
two different second sensors (2D LiDAR vs RTK GNSS + survey geometry). **Mild supporting evidence
for a centred or near-centred mount.**

It is **not** a measurement. The two empirical estimates differ by **33 mm** — larger than the
19.5–24.1 mm effect GT-1 resolves on Ktima — and the geojson estimate's IQR alone spans 48 cm.
*Converging on "small" is not the same as knowing the number.* See §12.3 for why the two are not
fully independent.

Consistent side-result: the camera under-reads row spacing by ~5% in both bags (0.952 on `part2`,
0.930 on `rh_july2026`) — canopy envelope vs trunk line.

### 13.4 Public documentation — definitive negative

| source | camera | facing |
|---|---|---|
| BLT dataset (Polvara 2024) | two ZED2 | forward + side |
| De Silva crop-row work | D435i | forward |
| Semantic-Aware Particle Filter (Sept 2025) | D435i | forward |
| public Thorvald URDF (`okb6/Thorvald_Grape_Urdf`) | none — 2.5 KB `base_link` stub | — |

**No published source documents a rear-facing D435i on a Thorvald.** BLT — including its five
Riseholme 2023 sessions — used ZED2s, not a RealSense at all. The rear D435i is a local
modification, so only the builder holds the number and no amount of searching substitutes.

### 13.5 The decision

**Proceed without waiting.** Height, pitch and roll are adopted (§12.1) because three independent
checks support them and cross-session agreement is 9 mm. Lateral and yaw are **not** adopted; they
are set to the centred baseline as an explicit assumption (§12.2), and every downstream number is
reported as conditional on it.

**What would change this:** a physical measurement or bracket photograph. If the collector's
account later contradicts the locked values, that is a finding to investigate on its own evidence —
not a reason to have withheld them, since the checks here stand independently of his agreement.

**What this does not fix.** Even a perfect calibration leaves the reference-precision limit: GNSS
short-scale residual on `part2` is 39.2 / 44.2 / 56.6 mm (5/9/15-fix) against a 19.5–24.1 mm effect.
The reference is 2–3× coarser than the quantity, so a per-arm GT-1 ranking at Riseholme is expected
to be **indistinguishable regardless of calibration**. The calibration was never the only blocker.
