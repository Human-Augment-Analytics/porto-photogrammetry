# Ring-held-out novel-view experiments

Reuse the shared all-view COLMAP model. Train each backend separately with the
explicit manual split in `splits_manual/<specimen>/split.json`. The partition must
contain each registered image exactly once. Holding out SfM views is outside this
protocol. Keep all-view and held-out reconstruction outputs separate.

The benchmark workflow supports 2DGS, PGSR, GW, MILo, GeoSVR and Meshroom. It stages
a private scene per backend with the tree's selected images and masks. COLMAP
binary/text files are read-only links inside a private sparse directory; generated
PLY caches stay private. Computed output directories cannot be adopted symlinks.
Never train into a teammate's adopted reconstruction.

`augenblick recon <backend> --eval --no-score` trains and renders without scoring.
The workflow then scores once, requiring every held-out render, its matching GT
export and a nonempty specimen mask. Standalone `--eval` retains automatic scoring
unless `--no-score` is passed. Scoring uses the shared NVS scorer, including DISTS
where installed; historical frozen runs must not be assigned unmeasured scores.

## Meshroom

Meshroom requires a preconverted all-view SfMData file matching the shared COLMAP
poses and pinhole intrinsics, plus a saved dense pipeline graph. Configure
`backend_args.meshroom` with `--sfmdata`, `--template`, `--meshroom_root`,
`--meshroom_python` and `--alicevision_root`. `{specimen}` and `{tree}` expand in
these arguments. The workflow tracks the SfMData and graph as file dependencies.

Dense preparation removes held-out views, poses, intrinsics that are unused, and
held-out landmark observations; landmarks need at least two training observations.
The graph excludes SfM and Publish nodes, uses selected masks and writes to a
private cache. No new SfM is run. Cache pruning occurs only after successful mesh
publication. Use a fresh output tree for a changed experiment protocol.

Held-out Meshroom views use unlit textured-mesh ray casting. AliceVision 3.3
`Texturing::saveAs` exports vertices as `(x, -y, -z)` relative to SfM coordinates;
the renderer explicitly reverses this fixed basis change with
`--mesh-frame alicevision_obj`. It does not fit an alignment to held-out pixels.
Source: https://github.com/alicevision/AliceVision/blob/v3.3.0/src/aliceVision/mesh/Texturing.cpp

AliceVision UDIM atlas coordinates are localized per triangle before sampling the
corresponding material image. The renderer regression covers a nonzero UDIM tile.

Rendering uses the retained COLMAP calibration and principal point, a black
background, bilinear texture interpolation and bicubic photograph resizing.
Default resolution caps the long side at 1600 pixels. This mesh representation is
different from the native Gaussian/implicit renderers; retain this fact in tables.
Meshroom geometry/metric export needs the same explicit basis conversion before
lineage assumptions can be made. Geometry was not validated by the NVS checks.

## Timing and resources

`command_timing.json` measures the complete reconstruction CLI command, including
its preparation, training, extraction and held-out rendering, excluding caller scene
staging, queue delay and separate scoring. This matches Syed's full command timer
boundary apart from the additional NVS rendering. `stage_timings.json` reports only
backend subprocess stages. Scoring gets a separate timing JSON. Slurm elapsed is a
broader job measurement; do not compare these three scopes interchangeably.

Tree `training_resources` sets `cpus`, `mem_mb` and `minutes`, with optional
`mem_overrides[backend][specimen]`. The November 14 campaign matches 16 CPUs,
96 GB and one RTX PRO 6000; the two fish GeoSVR retries used 128 GB. Preserve
GPU/host records, effective configs, source hashes and retry outcomes per run.
GW held-out training disables exposure compensation while the all-view companion
enables it. A changed condition is not a pure view-withholding ablation.

Run Snakemake from the tree and inspect a dry run before submission. The convenience
`workflow/run_experiments.sh` modifies selected tree configs even with `DRY=1`;
use direct Snakemake commands when reviewing an existing frozen experiment.

## Direct evaluation raster and paper timing export

The paper workflow requires native rendered dimensions to equal the shared
original-photograph evaluation raster (`--require-native-raster`). Dimensions use
floor division; the default caps the long side at 1600 pixels. Resampling a
backend's RGB output is rejected by the paper scorer.

GeoSVR's reconstruction wrapper adds `--canonical_test_resolution` to its native
render command only when held-out evaluation is enabled. Held-out cameras are
cloned at the shared raster with FoV, normalized principal point and pose retained.
Native training cameras, training resolution and train-view renders used by TSDF
remain unchanged. Evaluation exports original-photo RGB ground truth resized with
PIL BICUBIC and records its rendering protocol beside the PNGs. This fixes the
1600x1067 native / 1600x1066 common-raster mismatch without retraining a model.

Ring-held-out workflow commands also append `_timing/sfm_timings.csv` using the
same column order as `hpg_slurm/scene_common.sh`:

```text
method,scene,gpu,n_images,seconds,exit_code,note,jobid,node,finished
```

Rows measure the full reconstruction CLI and preserve fractional seconds. The
precise command JSON remains authoritative. `n_images` is the total input photo
count, consistent with the all-view export; the note records actual train/test
counts and the NVS condition. A file lock protects simultaneous writers. Failed
commands retain their exit code. Queue/setup and shared scoring stay separate;
Slurm accounting is needed for externally killed jobs without a completed timer.
The matched paper request is 16 CPUs, 98304 MiB RAM and one RTX PRO6000, with
`minutes_overrides[backend]` allowing conservative method-specific walltime.

The validated paper launcher freezes source/configuration and records SHA256 for
all input photograph, mask and COLMAP bytes, manual splits and Meshroom auxiliary
files. It validates the native-reader dependency paths through the actual workflow
shell before GPU work, checks identities before every job, and checks all inputs
before resume even for a no-op DAG. Plain directory timestamp dependencies alone
do not guarantee content-aware invalidation. Use fresh output trees for changed
protocols. Keep historical QA runs outside the matched paper cohort.

## GeoSVR follow-up corrections

GeoSVR `--resolution` controls native training size; `--evaluation_resolution`
(default -1) independently controls held-out output size. In the workflow, the
tree's `nvs_resolution` (default -1) is passed to both the GeoSVR held-out renderer
and the canonical scorer. Configure that tree field rather than a conflicting
backend-only evaluation flag. Non-GeoSVR renderers must likewise produce the
configured shared raster; paper scoring still rejects resizing. Default paper
settings are unchanged by this interface correction.

Explicit checkpoint selection resolves `ours_<iteration>[_suffix]` directories.
If more than one variant exists at the requested/latest iteration, scoring fails
rather than choosing directory enumeration order. Use `--render-variant` with the
exact child directory name to select a variant deliberately. Fresh campaign
outputs have one variant; no old result needs to be overwritten for this fix.

The GeoSVR reader now applies its eight-neighbor cap independently per camera.
Previously a sparse-neighbor camera reduced the cap for every later camera; a
zero-neighbor camera could disable their multiview losses entirely. Angular and
distance gates, sorting, weights and train-only neighbor membership are unchanged.
A pose-only audit of the approved 56-specimen roster predicts changed neighbor
sets in 38 held-out training partitions and 31 all-view partitions under that old
code. Those totals are CPU geometric predictions, not measured quality changes.
The two validated QA specimens have at least 8 eligible neighbors everywhere and
are unaffected by the old cap bug. Preserve old outputs/source identities. Before
claiming a pure view-withholding comparison against historical all-view results,
confirm the baseline runtime's reader version and account for this correction;
retraining/quality effects have not been measured by the pose audit. The active
five-method paper campaign uses its immutable earlier snapshot and is unchanged.
