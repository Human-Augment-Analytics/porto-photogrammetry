# Plan: feed COLMAP SfM into the Meshroom baseline (skip AliceVision's SfM)

Implementation spec; once landed it is historical and the code wins. The durable description
goes in [baseline-meshroom.md](../MEMORY/baseline-meshroom.md).

**Scope.** `baseline/benchmark_meshroom.py --from_sfm` converts a `sfm colmap` scene's
`sparse/0/` to AliceVision SfMData and runs only Meshroom's dense half on it. Meshroom becomes
one more extractor column on the same COLMAP poses as 2DGS/PGSR/GW/…. Native mode is unchanged.

```
skipped:  CameraInit → FeatureExtraction → ImageMatching → FeatureMatching → StructureFromMotion
kept:     PrepareDenseScene → DepthMap → DepthMapFilter → Meshing → MeshFiltering → Texturing → Publish
```

**Researched** 2026-10-30/31 at `b056999`. Sources:
- HPG install `/blue/arthur.porto/srizvi63.gatech/alicevision` (2025.1.0);
- `~/meshroom` (`v2025.1.0`) and `~/alicevision` source (`ed0e2739f`);
- the `conda/meshroom` env;
- `hpg_slurm/{meshroom_benchmark,recon,hull_sfm}.sbatch` and `scene_common.sh`.

Tags: *checked* = run on the login node (CPU) with a throwaway prototype converter on
`sp26_results/36342/all/colmap` (276 images, 6240×4160, SIMPLE_PINHOLE, masks). *source* = read
from code. DepthMap onward has **not** run on converted input.

## Status

| Step | State |
|---|---|
| 1 — converter `baseline/colmap_to_sfmdata.py` | Done; CPU-verified 2026-10-31 |
| 2 — dense-only graph + driver | Done; ran to DepthMap on CPU |
| 3 — `--from_sfm` in `benchmark_meshroom.py` | Done |
| 4 — tests | Done (10 pass; oracle mutation-checked) |
| 5 — `SFM=` mode in `hpg_slurm/meshroom_benchmark.sbatch` (PACE after) | HPG done, dry-run on CPU; PACE not started |
| 6 — pycolmap in `meshroom` env + docs | Done (installed on HPG; numpy/PySide6 unchanged) |
| Verification (GPU) | Done 2026-10-31 (job 48523806); see Verification |

## Goal

`python baseline/benchmark_meshroom.py <colmap_scene> <out> --from_sfm [--keep_cache]` writes
`<out>/textured_mesh/texturedMesh.obj`, with the same layout and `prune_cache()` as today, on
COLMAP's cameras. `eval.mesh` scores it like the native mesh. Unusable input exits non-zero
before any GPU work, naming the cause.

**Non-goals:**
- no changes under `src/libs/` and no AliceVision rebuild;
- no `augenblick recon meshroom` (the wrapper stays in the `meshroom` env, which has no
  `augenblick`);
- no distorted camera models in v1 (step 1b);
- no track-less scenes, i.e. DVLT or VGGT without `--use_ba` (fact 5);
- no metric scale.

## Facts

1. **AliceVision exports COLMAP but cannot import it** (*checked*). `bin/` has
   `exportColmap` and no importer. `sfmDataIO/colmap.cpp` only writes, and the other importers
   don't read COLMAP. → Write `.sfm` JSON ourselves, using AliceVision's loader and exporter as
   the oracle.
2. **Version: the install reads SfMData ≤ 1.2.12; the source checkout is 1.2.14** (*checked*).
   The installed `cameraInit` writes `1.2.12`. The loader rejects newer files, and required
   fields are version-gated (`jsonIO.cpp:loadIntrinsic`). → Pin `SFMDATA_VERSION = (1, 2, 12)`.
   Take required fields from the installed `cameraInit` output, not from the source tree.
3. **Mapping, validated by round trip** (*checked*). Prototype `.sfm` →
   `aliceVision_convertSfMFormat` (rc 0) → `aliceVision_exportColmap` → compared to the
   original:

   | Quantity | Max \|Δ\| |
   |---|---|
   | quaternions | 4e-16 |
   | translations | 5e-15 |
   | points (52,492 of 52,492) | 5e-6 |
   | intrinsics | 0.005 px (the exporter's text precision) |

   All 276 views were exportable.

   | SfMData field | From COLMAP | Note |
   |---|---|---|
   | `viewId` = `poseId` | `image_id` | |
   | `intrinsicId` | `camera_id` | `sfm colmap` uses `CameraMode.PER_IMAGE` |
   | `path` | absolute `images/<name>`, symlinks **not** resolved | `tryLoadMask` matches masks by this filename |
   | `metadata` | `{}` | fact 8 |
   | `type` / `distortionType` / `undistortionType` | `pinhole` / `none` / `none` | |
   | `sensorWidth`, `sensorHeight` | `w`, `h` | so 1 mm = 1 px |
   | `focalLength` = `initialFocalLength` | `fy` | 1.2.11+: `scale_y = fmm·w/sensorWidth` |
   | `pixelRatio` | `fy/fx` | `scale_x = (fmm/pixelRatio)·w/sensorWidth` |
   | `principalPoint` | `(cx − w/2, cy − h/2)` | stored as an offset from the image centre |
   | `serialNumber`, `*Locked`, empty `(un)distortionParams`, `undistortionOffset [0,0]` | constants | required by the 1.2.12 loader |
   | `rotation` | `image.cam_from_world().rotation.matrix()`, **column-major** | `saveMatrix` uses Eigen's linear index |
   | `center` | `image.projection_center()` (= `−Rᵀt`) | `Pose3(R, C)` is world→camera, like COLMAP |
   | `X`, `color`, `descType` | `xyz`, `rgb`, `"sift"` | |
   | `observations[]` | track `(image_id, idx)` → `observationId`, `featureId`, `x` = point2D, `scale: 1.0` | |

   No 0.5 px shift is applied; AliceVision's own exporter applies none either (Open questions).
4. **One observation per view per landmark** (*checked*). `getObservations()` is keyed by view,
   so duplicates are dropped silently. On 36342 that lost exactly the 2,110 same-image
   duplicates (1,222 tracks). → Dedupe by lowest reprojection error, drop landmarks left with
   < 2 views, and log the counts.
5. **The dense stages need tracks, not just poses** (*source*). Three stages read landmarks:
   - DepthMap neighbours come only from `findNearestCamsFromLandmarks`, with no pose fallback;
   - the SGM depth range comes from SfM seeds;
   - Meshing's bounding box uses `Fuser.cpp` with `estimateSpaceMinObservations=3`.

   No kept stage reads observation `scale`. → A track-less scene would give every view zero
   neighbours, so the converter rejects it. `recon colmap` has the same limit.
6. **Run via the Meshroom API or `meshroom_compute`, not `meshroom_batch`** (*checked*).
   `meshroom_batch` exits without `-i` and feeds inputs only to CameraInit-type nodes.
   `photogrammetry.mg` minus the five SfM nodes ran `meshroom_compute --toNode
   PrepareDenseScene_1` to SUCCESS on a 4-view `.sfm`. The `"template": true` header loads, and
   the cache layout (`MeshroomCache/<Node>/<uid>/`) is what `prune_cache()` expects. → The
   `--paramOverrides` handling has to be re-implemented (step 2).
7. **Read COLMAP with pycolmap, installed into the `meshroom` env** (*checked*).
   - The env (Python 3.10, numpy 1.26.4, PySide6 6.8.3) has no pycolmap today.
     `src/utils/colmap_io.py` doesn't read tracks.
   - The `pycolmap==4.1.1` wheel (`cp310`, manylinux) depends only on an unpinned `numpy`, so
     it leaves the env's numpy and PySide6 alone.
   - Tested from a scratch `--target`: it imported and read the 36342 model, giving 276
     cameras, tracks, per-point errors, `img_from_cam` and `projection_center`.
   - It is the version augenblick pins (as `pycolmap-cuda12`) and the library that wrote these
     models, so reading pycolmap-4 output (`frames.bin`, `rigs.bin`) is guaranteed.
   - → Use pycolmap; no hand-written reader. Use the CPU wheel `pycolmap`: the converter needs
     no CUDA, and the tests run in the augenblick env, which already has it.
8. **Empty metadata is safe** (*checked*). PrepareDenseScene logs `Median EV: -nan` but still
   writes `AliceVision:EVComp`. `Texturing.correctEV` defaults to True, and with no EXIF there
   is nothing to correct. → Force `Texturing:correctEV=False`. `checkImageSize` would catch an
   EXIF-rotated mismatch; it passed on 36342.
9. **Masks resolve by stem, with a trap** (*checked*). `tryLoadMask` tries `<viewId>.png`
   before `<stem>.png`. With `<scene>/masks`, alpha was 0 on 84.4% of an EXR. The trap: with
   unpadded numeric names (`5.jpg`), view 5 can load another image's mask. → Fail on any
   `masks/<viewId>.png` that isn't that view's own stem. Masks go only to
   `PrepareDenseScene:masksFolders`.
10. **`--pipeline` is moot** (*checked*). The dense nodes are linked identically in
    `photogrammetry.mg` and `photogrammetryObjectTurntable.mg`, apart from the turntable's
    segmentation-fed `masksFolders`. → Always trim `photogrammetry.mg`; `--pipeline turntable`
    with `--from_sfm` is an error.
11. **HPG's default `MESHROOM_ROOT` (`/blue/…/Meshroom`) doesn't exist**; the frontend is
    `~/meshroom`. This is already documented in `hpg_slurm/README.md` ("Only half-installed
    here"). → Submit with `MESHROOM_ROOT=$HOME/meshroom`; the default is left unchanged.
12. **HPG job conventions** (*checked*):
    - **Paths:** SfM consumers take `SFM=<dir>`, read `$RESULT_ROOT/$SCENE_NAME/$SFM` and
      write `$SFM-<method>` (for example `colmap-2dgs`). The array index → scene mapping is
      shared via `discover_scenes`.
    - **Missing model:** `recon.sbatch` refuses an empty `sparse/0` with "run the $SFM SfM job
      for this scene first".
    - **Fixed flags:** the Meshroom job always passes `--max_threads` (from `--cpus-per-task`,
      24) and `--save_file "$OUT/pipeline.mg"`, then `"$@"`.
    - **Timing CSV:** the method column is `$SLURM_JOB_NAME`. `start_timing`'s third argument
      only adds the `RESULT … sfm=` log marker, and `_timing_finish` greps the `.err` log
      named after the job. → Both modes write `meshroom` rows unless submitted with `-J`; recon
      rows read `<sfm>-<recon>` (`colmap-2dgs`, …), so `--from_sfm` uses `-J colmap-meshroom`.
    - **Array width:** `%1` is about `MeshroomCache` size, which is mostly dense output, so it
      stays.
    - **Data:** the 9-scene subset (`RESULT_ROOT=…/results/neurips/subset`,
      `DATA_ROOT=…/neurips/subset/prepared`) has `colmap/` plus every `colmap-*` recon and a
      native `meshroom/` per scene. All nine pass the converter's guards (*checked*:
      SIMPLE_PINHOLE only, every view registered with observations, `IMG_*` stems, links
      resolve). The `sp26_results/*/colmap/images` links dangle because the data moved.

## Design

### Step 1 — `baseline/colmap_to_sfmdata.py` (pycolmap + stdlib)

`python baseline/colmap_to_sfmdata.py <colmap_scene> <out.sfm> [--sparse sparse/0]`, also
importable.
- Read the model with `pycolmap.Reconstruction(scene/sparse)`, which handles `.bin` and `.txt`.
- `convert(scene_dir, out_path, sparse) -> ConvertStats` applies the fact-3 table and writes
  plain JSON numbers and bools. Boost's loader accepted those (*checked*).
- Per-observation reprojection error for the dedupe comes from
  `camera.img_from_cam(cam_from_world * X)`, so there is no projection code of our own.
- Before writing, it fails on:
  - a non-`SIMPLE_PINHOLE`/`PINHOLE` model (name it, and point to `--camera_model
    SIMPLE_PINHOLE` or COLMAP undistortion);
  - an image missing from `images/`;
  - track-less input: no landmarks with ≥ 2 views, or more than 10% (configurable) of views
    with zero observations. Name DVLT and VGGT-without-BA as the likely cause;
  - a fact-9 mask collision.
- Dedupes (fact 4) and logs views, intrinsics, landmarks kept/dropped, observations
  kept/deduped, and unregistered images, which get no depth map.

**1b (later).** Map `SIMPLE_RADIAL`/`RADIAL` → `radialk1`/`radialk3` only after the exporter
round trip and the step-4 reprojection test both pass on distorted points. Do not guess the
OPENCV ↔ `brown` parameter order.

### Step 2 — dense-only graph + driver

- **Graph:** `write_dense_graph(dest, sfm_path)`, mirroring `write_turntable_graph`:
  - load `photogrammetry.mg` via `find_template`;
  - delete the five SfM nodes and their `nodesVersions` entries;
  - set `PrepareDenseScene_1.inputs.input`;
  - raise on any surviving `{<deleted>.` link.
- **Driver:** `baseline/meshroom_run_graph.py` (~50 lines), launched like `meshroom_batch`
  today (`sys.executable`, `PYTHONPATH=$MESHROOM_ROOT`):
  - `initPlugins/initNodes`, then `graph.initFromTemplate(mg, publishOutputs=True)`;
  - set every Publish node's output to `<out>`;
  - apply overrides with `meshroom_batch`'s regex and semantics (`:243–262`). Typos then raise,
    and dotted group attributes (`Texturing:colorMapping.colorMappingFileType`) keep working;
  - set `cacheDir = <out>/MeshroomCache`, save `--save_file` if given, `executeGraph`;
  - exit non-zero on failure so the cache survives for resume.
- **Fallback:** if `initFromTemplate` proves fragile, edit the JSON and run
  `meshroom_compute` (fact 6).

### Step 3 — `benchmark_meshroom.py --from_sfm`

- `--from_sfm` makes `input_dir` a COLMAP scene. `--sparse` (default `sparse/0`) selects the
  model.
- **Flow:** `convert()` → `<out>/sfm/colmap.sfm` (timed separately) → `write_dense_graph()` →
  driver → `prune_cache()`. The `.sfm` sits outside the cache, so it is kept (45 MB for 276
  views) as a record of the cameras used.
- **Overrides:** drop the five `FeatureExtraction:*` entries, add `Texturing:correctEV=False`,
  and pass masks only via `PrepareDenseScene:masksFolders`.
- **Flags for skipped nodes** (`--describer_*`, `--max_threads`): default them to `None`.
  Native mode fills today's values (`normal`/`normal`/`dspsift`/16); `--from_sfm` warns only
  if one is given. `--pipeline turntable` with `--from_sfm` is an error.
- **Banner:** add the SfM path plus its view and landmark counts. Native mode stays
  byte-for-byte the same.

### Step 4 — `tests/test_meshroom_from_sfm.py` (CPU, no AliceVision)

- **Reprojection oracle:**
  - Input: a synthetic 4-camera `pycolmap.Reconstruction` built in the test, with `fx ≠ fy`, an
    off-centre principal point and non-trivial rotations.
  - Decode the converter's JSON with AliceVision's formulas, written independently of the
    encoder:
    - `R` read column-major;
    - `t = −R·C`;
    - `fx = focalLength/pixelRatio·w/sensorWidth` and `fy = focalLength·w/sensorWidth`;
    - `pp = offset + (w/2, h/2)`.
  - Check: every observation reprojects to within 1e-9 of COLMAP's `POINTS2D`. This catches a
    transposed `R` or a swapped `pixelRatio`, which a self-round-trip would miss.
- **Dedupe:** the lower-error observation is kept, and a < 2-view landmark is dropped.
- **Rejections:** `SIMPLE_RADIAL`, empty tracks, and a mask collision (`masks/2.png`, image
  `2.jpg` with id 5).
- **Graph:** `write_dense_graph` leaves no dangling links and keeps the six dense nodes plus
  Publish. Skip it if no `photogrammetry.mg` is on `MESHROOM_PIPELINE_TEMPLATES_PATH`.
- `baseline/` is imported by path, and the tests run in the augenblick env, which already
  has pycolmap.
- The decoder is deliberately **not** pycolmap: it is the independent side of the oracle.

### Step 5 — `hpg_slurm/meshroom_benchmark.sbatch`

Extend the existing job; both modes share the env, the sm gate, `%1` and pruning.

```bash
MESHROOM_ROOT=$HOME/meshroom sbatch hpg_slurm/meshroom_benchmark.sbatch                                   # native, unchanged
MESHROOM_ROOT=$HOME/meshroom SFM=colmap sbatch -J colmap-meshroom hpg_slurm/meshroom_benchmark.sbatch     # from SfM
```

With `SFM` set:
- **Paths:** `SCENE=$RESULT_ROOT/$SCENE_NAME/$SFM` and `OUT=…/$SFM-meshroom`, beside
  `colmap-2dgs`; native `meshroom/` is never touched.
- **Checks before `start_timing`:** refuse an empty `sparse/0` with `recon.sbatch`'s message,
  and refuse an `images/` that doesn't resolve (`[ -e "$SCENE/images/." ]`). Don't whitelist
  `SFM` values; the converter's track guard decides.
- **Arguments:** pass `--from_sfm`; drop `--max_threads`; keep `--save_file` and `"$@"`.
- **Timing:** `start_timing "$SCENE_NAME" "$N_IMAGES" "$SFM-meshroom"`. Warn, without exiting,
  if `SLURM_JOB_NAME` isn't `$SFM-meshroom`, suggesting `-J $SFM-meshroom`.
- **Banner:** add `input sfm  : $SFM -> $SCENE`. The masks line reads the SfM output's
  `masks/` symlink.
- **Limits:** keep `--time=6:00:00` and tighten it after verification.
- **README:** add `SFM=` to the job's row, and a note on `-J` and `<sfm>-meshroom` under
  "Meshroom baseline".
- **PACE:** mirror the change after HPG verifies, checking its `RESULT_ROOT` layout first.

### Step 6 — env + docs

- `meshroom-setup.md` step 4: add `pip install pycolmap==4.1.1`, the same version augenblick
  pins, to the `meshroom` env build. Install it into the HPG env.
- In `--from_sfm` mode, `benchmark_meshroom.py` imports pycolmap lazily and fails with that
  install line if it's missing. Native mode needs no pycolmap.

- `MEMORY/baseline-meshroom.md`: a "From COLMAP SfM" section with the command, facts 2/4/5/9
  in one line each, and the output path.
- `CLAUDE.md`: Baselines row "`--from_sfm` reuses a COLMAP scene", plus an optional gotcha
  line that it needs tracks, so DVLT and VGGT-without-BA are rejected.

## Verification

**Done (CPU):** facts 2, 3, 4, 6, 7, 8 and 9, first with a numpy prototype on 36342. Re-run with
the pycolmap converter on UF_Mammals_1769_cranium (36342's image links now dangle):
- round trip via `aliceVision_exportColmap`: 432/432 views and 80,397/80,397 points;
  quaternions 3e-16, translations 3e-15, points 5e-6, intrinsics 0.05 px (6-digit text);
- 5,593 duplicates deduped, matching the pre-conversion count;
- `--from_sfm` on a 6-view cut of it ran conversion → graph → driver → PrepareDenseScene SUCCESS
  (alpha 0 on 92.6% of each EXR) and failed at DepthMap on "no CUDA GPU", keeping the cache;
- the `SFM=colmap` job path, dry-run with a scratch `RESULT_ROOT`, wrote a
  `colmap-meshroom,…,FAILED` CSV row and `RESULT … sfm=colmap-meshroom`.

**GPU, run 2026-10-31.** Scene: `results/neurips/subset/UF_Mammals_1769_cranium/colmap`, index 3
in `discover_scenes` (*checked*: its earlier jobs ran as `_3`). It has 432 images, 80,397
points, a mean reprojection error of 0.72 px (the subset's lowest) and 5,593 same-image
duplicates. Beside it are `colmap-{2dgs,pgsr,gw,geosvr,milo}` and a native `meshroom/`, which took
5,451 s:
```bash
RR=/blue/arthur.porto/srizvi63.gatech/results/neurips/subset
DATA_ROOT=/blue/arthur.porto/data/datasets/photogrammetry/neurips/subset/prepared RESULT_ROOT=$RR \
MESHROOM_ROOT=$HOME/meshroom SFM=colmap \
sbatch -J colmap-meshroom --array=3 hpg_slurm/meshroom_benchmark.sbatch
```

1. **End to end:**
   - all six dense nodes reach SUCCESS;
   - DepthMap's tCams count is > 0 per view;
   - Meshing's bounding box encloses the specimen;
   - the mesh exists and the cache is pruned.
2. **Alignment:** render with `src/utils/render_mesh_views.py` next to the native mesh.
   Texture seams or colour bleeding mean pose error.
3. **Score:** `eval.mesh` on native vs `--from_sfm` vs `colmap-2dgs`, all in the same
   registration mode (no `--rigid` unless scaled). Expect `--from_sfm` ≈ native.
4. **Runtime:** expect a drop from the native 5,451 s of about its FeatureExtraction + Matching
   + SfM share.
   Record it in baseline-meshroom.md.
5. **Negative:** `SFM=dvlt` on the full `results/neurips` root (`UF_Mammals_10941_skull`,
   `UF_Mammals_1199_mandible`; the subset has none) exits non-zero at conversion, and the CSV
   says `FAILED`.
6. **Bookkeeping:**
   - the subset's CSV method column reads `colmap-meshroom`;
   - the log has `RESULT … sfm=colmap-meshroom`;
   - native `meshroom/` is untouched.

**Results** (job 48523806, `c0806a-s21`, `--cpus-per-task=16` because the group's 64-CPU QOS
cap was full; native had 24):
- **End to end:** all seven nodes SUCCESS in 1,614 s against native's 5,451 s (3.4x faster, on
  fewer cores); 24.2 GiB cache pruned; CSV row `colmap-meshroom,…,0`; `RESULT … sfm=colmap-meshroom`.
- **Frame:** the OBJ is COLMAP's frame with **Y and Z negated**: AliceVision's `Mesh::save`
  flips them (`mesh/Mesh.cpp:136-137`), native meshes included. After `diag(1,-1,-1)` COLMAP's
  sparse points sit a median 0.045% of the diagonal from the mesh (2DGS: 0.146%); unflipped,
  5.4%. Anything comparing it to COLMAP cameras without registration must apply the flip.
- **Score** vs `gt/UF-M-1769-skull.obj`, similarity registration, 200k samples, seed 0:

  | Mesh | chamfer % diag | F@0.25% | F@0.5% | F@1% | fitted scale |
  |---|---|---|---|---|---|
  | native `meshroom` | 0.251 | 0.908 | 0.939 | 0.960 | 435.1 |
  | `colmap-meshroom` | **0.212** | **0.912** | **0.944** | **0.966** | 116.6 |
  | `colmap-2dgs` | 0.467 | 0.780 | 0.838 | 0.883 | 117.6 |

  The fitted scales of `colmap-meshroom` and `colmap-2dgs` agree (same COLMAP gauge).
- **Negative:** run on the login node instead of a GPU job. `UF_Mammals_1199_mandible/dvlt`
  failed the track guard; `UF_Mammals_10941_skull/dvlt` failed earlier, on 5 images missing
  from its `images/`. Neither wrote a `.sfm`.
- **Not checked:** per-view DepthMap tCams and the Meshing bounding box (node logs are pruned
  with the cache); the render comparison was replaced by the sparse-point distance above.

## Open questions

- **Y/Z flip.** Should `--from_sfm` also write a COLMAP-frame copy of the mesh (`diag(1,-1,-1)`)
  so `--no-align` evaluation and NVS with COLMAP cameras work directly?

- **Pixel centre.** COLMAP puts the first pixel's centre at (0.5, 0.5). AliceVision's
  exporter applies no shift, so either the conventions match or its exporter is 0.5 px off.
  Following the exporter keeps the two consistent, and the bias wouldn't be visible at 6240 px.
  Settle it from `camera/Pinhole.cpp:cam2ima` before claiming sub-pixel parity.
- **Other producers.** `vggt --use_ba`, `turntable` and `hull` on tracked scenes should pass
  the track guard. Confirm with one VGGT+BA run before advertising it.
- **pyalicevision.** It imports in the env (with `ALICEVISION_ROOT` set; otherwise it aborts
  on the OCIO config) and could write the SfMData itself. Not chosen: its SWIG Eigen bindings
  are thin, it ties the converter to that env, and the JSON route is already validated.
  Revisit on an AliceVision upgrade.
