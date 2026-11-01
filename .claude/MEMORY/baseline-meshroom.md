# `baseline/` — Baseline Wrappers

Thin wrappers around third-party photogrammetry tools used as qualitative comparisons. They
match the logging style of the `2dgs` backend (banner, `subprocess.run`,
total-time summary).

## Meshroom (AliceVision)

```bash
python baseline/benchmark_meshroom.py <input_images> <output_dir> \
    [--save_file <path.mg>] [--meshroom_root <path>]
```

Resolves `meshroom_batch` from `$MESHROOM_ROOT` (or `--meshroom_root`), invokes the hardcoded
`photogrammetry` pipeline template, prepends `$MESHROOM_ROOT` to `PYTHONPATH` so its Python
modules resolve, and logs total runtime. Masks are supported via
`--paramOverrides FeatureExtraction:masksFolder=<path>` (see `ff0198b`); texture output is
forced to `.png` (`afdc84b`), and fidelity knobs were added in `87fcdb5`.

Installation and env-var setup: `meshroom-setup.md` (repo root) — it includes a dedicated
`meshroom` conda env for the batch CLI.

RealityScan is the other (commercial) qualitative baseline, run outside this repo.

## From COLMAP SfM (`--from_sfm`)

```bash
python baseline/benchmark_meshroom.py <colmap_scene> <output_dir> --from_sfm [--sparse sparse/0]
```

`baseline/colmap_to_sfmdata.py` (pycolmap, in the `meshroom` env) writes
`<output_dir>/sfm/colmap.sfm`; `write_dense_graph()` strips CameraInit..StructureFromMotion from
`photogrammetry.mg`, and `baseline/meshroom_run_graph.py` runs it (`meshroom_batch` requires a
CameraInit to feed). Output layout and pruning match native mode. Facts behind the converter:

- **SfMData 1.2.12.**
- **One observation per view per landmark.** AliceVision keys observations by view, so COLMAP's
  same-image duplicates are deduped by reprojection error (5,593 on UF_Mammals_1769_cranium).
- **Tracks are required.** DepthMap picks neighbours only from shared landmarks, so DVLT and
  VGGT without BA are rejected before any GPU work.
- **Mask lookup tries `masks/<viewId>.png` before `<stem>.png`.** With numeric image names
  that can load another image's mask; the converter fails on it. View paths stay unresolved
  so the stem lookup matches symlinked images.
- `Texturing:correctEV=False`: converted views carry no EXIF.
- **Output meshes have Y and Z negated** relative to the SfM frame, native runs included:
  AliceVision's `Mesh::save` flips them (`mesh/Mesh.cpp:136-137`). Registered `eval.mesh` absorbs
  it; anything pairing a Meshroom mesh with COLMAP cameras unregistered must apply
  `diag(1,-1,-1)` first. Verified on UF_Mammals_1769_cranium: sparse points to mesh, median
  0.045% of the diagonal after the flip, 5.4% before.

## Running it on a cluster

`pace_slurm/meshroom_benchmark.sbatch` sources `meshroom_common.sh`, not `common.sh`: the
frontend is one GPU-independent env rather than one per GPU, the AliceVision tarball bundles its
own `libcudart` that a CUDA module would shadow, and the `ALICEVISION_*` vars live only in
`~/.bashrc`, which a batch job never sources.

Supported GPUs, the disk-quota constraint and the `%1` array width are documented in
`pace_slurm/README.md`. The three facts behind them, which the code alone does not show:

- **SASS `sm_50`-`sm_90` and zero PTX, yet it runs above sm_90.** The release bundles
  `libnvrtc.so.12.1` and DepthMap compiles kernels at runtime, so absence of PTX is not a
  ceiling. Verified on sm_120 (Blackwell): nvrtc mapped in the live process, valid depth maps.
- **DepthMap is host-bound, so a bigger GPU does not help.** `computeOnMultiGPUs.cpp` runs one
  OMP thread per GPU, not per core, and `ALICEVISION_DEVICE_MAX_CONSTANT_CAMERA_PARAM_SETS`
  clamps batching to the same 64 KB constant-memory budget on every card. An A40 measured
  *faster* than an H200 (10.1 vs 12.9 s/depth map). Extra cores only help FeatureExtraction and
  Texturing. s/depth-map is therefore not a GPU ranking: a `2g.48gb` MIG slice on a 96-core
  node measured 6.1 s/map, beating both. Only compare runs from the same node type.
- **`MeshroomCache` peaks at 50-80 GB per ~660-image scene**, mostly PrepareDenseScene EXRs and
  DepthMap tiles. `prune_cache()` keeps what `Texturing.py` declares (`texturedMesh.*`,
  `texture_*`) and drops the rest, ~84 G -> ~400 M. It refuses to delete a cache holding no mesh,
  so a run that reports success without one keeps everything.

## Reference runtimes

138 images at 6240x4160 with masks, NVIDIA A100 PCIe 40 GB (end-to-end):
COLMAP+SuGaR ~80 min · COLMAP+2DGS ~40 min · COLMAP+PGSR ~70 min · COLMAP+GW ~65 min ·
Meshroom ~60 min.

SfM only, same scene on a B200 80 GB: COLMAP ~10 min · VGGT ~3 min · VGGT+BA ~15 min.
