# SLURM job scripts — HiPerGator

Batch equivalents of the manual `salloc` workflow, so runs survive a dropped SSH session.
This is the **HiPerGator** set; `pace_slurm/` holds the PACE ICE equivalents. The two differ
only in cluster specifics (account, partition, GPU selection, module names, conda root, data
roots) — the job logic is identical.

## Submit

```bash
# One array task per scene found under DATA_ROOT; add --array=N for a single scene.
MASK_METHOD=rembg sbatch hpg_slurm/mask.sbatch
sbatch hpg_slurm/vggt_sfm.sbatch
sbatch hpg_slurm/vggt_ba_sfm.sbatch
sbatch hpg_slurm/colmap_sfm.sbatch
SFM=colmap sbatch hpg_slurm/turntable_sfm.sbatch
SFM=colmap sbatch hpg_slurm/hull_sfm.sbatch
BACKEND=2dgs SFM=vggt sbatch hpg_slurm/recon.sbatch
sbatch hpg_slurm/meshroom_benchmark.sbatch          # baseline; needs the Meshroom frontend
```

Building an env is itself a job — the CUDA rasterizers must compile on the card they will run on:

```bash
GPU=rtx6000 sbatch hpg_slurm/setup_env.sbatch
GPU=b200 sbatch --partition=hpg-b200 hpg_slurm/setup_env.sbatch
```

`recon.sbatch` takes `BACKEND=2dgs|sugar|pgsr|gw` and `SFM=<name>` to pick its input SfM;
`turntable_sfm.sbatch` and `hull_sfm.sbatch` take `SFM=` too and run that SfM first if its
output is missing. Scenes are sorted, so an array index maps to the same scene across
submissions. Extra flags are forwarded to the `augenblick` CLI.

`mask.sbatch` produces the silhouettes the hull and masked-COLMAP paths need, writing a
scene-shaped `images/` + `masks/` pair the SfM jobs consume directly. Output goes to
`$RESULT_ROOT/<method>/<scene>/` so the methods coexist; `OUT_LAYOUT=flat` writes `<scene>/`
instead, where a second method overwrites the first. Both GPU methods fail fast rather than
degrade: `rembg` if onnxruntime cannot load its CUDA provider, `sam3` if no CUDA device is
present at all — `build_sam3_image_model` hardcodes `device="cuda"`. `sam3` takes a text
`--prompt` (default `skeleton`) and a checkpoint from the gated HF repo `facebook/sam3`,
which lands in `$HF_HOME`.

Timing rows take the method from the **job name**, so to keep two methods apart in one CSV:
`MASK_METHOD=threshold sbatch --job-name=mask-threshold hpg_slurm/mask.sbatch`.

`hull_sfm.sbatch` carves the visual hull from the masks as the initial point cloud. Its output
differs from the input SfM only in `sparse/0/points3D.ply`, so running `recon.sbatch` against
both ablates initialisation with poses, intrinsics and schedule held fixed. It requires masks.

For anything not covered, copy `template.sbatch` and edit its command block.

## Files

| File | Role |
|------|------|
| `scene_common.sh` | Roots, scene discovery, banner, timing CSV. Sourced by the two setup files below, never by a job directly |
| `common.sh` | GPU switch, module loads, `HF_HOME`/`TORCH_HOME` exports, conda activate; then sources `scene_common.sh` |
| `meshroom_common.sh` | AliceVision env + sm gate (`MESHROOM_MAX_SM`); then sources `scene_common.sh` |
| `template.sbatch` | Copy-and-edit starting point |
| `mask.sbatch` | `MASK_METHOD=rembg\|threshold\|sam3`; consumes `<scene>/images`, emits a scene |
| `vggt_sfm.sbatch` | VGGT -> COLMAP, one array task per scene |
| `vggt_ba_sfm.sbatch` | Same with `--use_ba` |
| `colmap_sfm.sbatch` | Masked COLMAP SfM |
| `turntable_sfm.sbatch` | Turntable refinement; runs the input SfM first if absent (`SFM=`) |
| `hull_sfm.sbatch` | Visual-hull init; same, and requires masks |
| `recon.sbatch` | `BACKEND=2dgs\|sugar\|pgsr\|gw`, `SFM=<name>` picks the input SfM |
| `meshroom_benchmark.sbatch` | Meshroom baseline |
| `setup_env.sbatch` | Builds `augenblick_<card>` on a node of that GPU |
| `prepare_neurips.sbatch` | Flattens raw NeurIPS objects into `<scene>/{images,masks}` |

## Shared job setup

`scene_common.sh` holds what must be identical across methods for results to compare: roots,
scene discovery, the array-index → scene mapping, the banner, and the timing CSV. No job
sources it directly — `common.sh` (torch/CUDA) and `meshroom_common.sh` (AliceVision) set up
their own environment first. Callers set `GPU` and `CONDA_ENV`, and may set `BANNER_EXTRA`
(`label: value` lines spliced into the banner).

Discovery matches `images/` via `-type d -o -xtype d` because a scene's `images/` is often a
symlink (`prepare_neurips.sbatch` writes them that way); matching only real dirs would drop
those scenes and shift every later array index, so task *N* would mean different scenes in
different jobs.

## Meshroom baseline

`meshroom_benchmark.sbatch` sources `meshroom_common.sh`, not `common.sh`, for three reasons:
the frontend is one GPU-independent env (`$CONDA_ROOT/meshroom`, pure Python) rather than one
per GPU; AliceVision 3.3.0 bundles its own `libcudart.so.12.1.105`, which a CUDA module would
shadow; and the `ALICEVISION_*` variables live only in `~/.bashrc`, which batch jobs never source.

**Only half-installed here:** AliceVision is at `/blue/arthur.porto/srizvi63.gatech/alicevision`,
but the Meshroom frontend checkout is not present, so `meshroom_common.sh` exits 2 pointing at
`meshroom-setup.md` until `$MESHROOM_ROOT/bin/meshroom_batch` exists. Override either path with
`ALICEVISION_ROOT` / `MESHROOM_ROOT`.

Prebuilt kernels are SASS-only to sm_90, but AliceVision bundles `libnvrtc` and compiles
DepthMap kernels at runtime, so newer cards work (verified on sm_120). `MESHROOM_MAX_SM` gates
this; both HPG cards pass (`rtx6000` sm_120, `b200` sm_100).

**One scene at a time (`--array=...%1`), deliberately.** `MeshroomCache` reaches ~50 GB for a
663-image scene; `benchmark_meshroom.py` prunes it to the textured mesh after each success, so
only one scene's intermediates ever exist.

## Data layout

`DATA_ROOT` and `RESULT_ROOT` are defined once in `scene_common.sh` as absolute `/blue/...`
paths (PACE derives its own from the checkout). Override either in the environment:

```
DATA_ROOT   = /blue/arthur.porto/data/datasets/photogrammetry/neurips/prepared
RESULT_ROOT = /blue/arthur.porto/srizvi63.gatech/results/neurips     # <scene>/<sfm>[-<backend>]/
```

Two scene shapes are supported, auto-detected from `DATA_ROOT`; force with
`SCENE_LAYOUT=nested|flat`.

| Layout | Shape | Used by |
|--------|-------|---------|
| `flat` | `<scene>/{images,masks}` | the neurips set (76 scenes, the default root) |
| `nested` | `<scene>/prepared/{images,masks}` | the main set (6 scenes) |

To run against the main dataset instead, export both roots before submitting:

```bash
DATA_ROOT=/blue/arthur.porto/data/datasets/photogrammetry/main/specimens \
RESULT_ROOT=/blue/arthur.porto/srizvi63.gatech/results/main \
  sbatch --array=0-5 hpg_slurm/colmap_sfm.sbatch
```

Note the `specimens/` component: `main/` gained `specimens/` and `realityscan/` above the
scenes. Discovery matches at a **fixed depth**, so pointing `DATA_ROOT` at `main/` finds no
`prepared/` dirs, falls back to `flat`, and then finds nothing — always point it at the
directory whose immediate children are the scenes.

Results no longer carry the old `all/` segment; runs under `results/main/<scene>/all/<sfm>/`
predate these scripts and are not discovered.

## Timing

Every per-scene job appends one row to `$RESULT_ROOT/_timing/sfm_timings.csv` from an `EXIT`
trap, so a scene that OOMs or runs out of wall clock still records:

```
method,scene,gpu,n_images,seconds,exit_code,note,jobid,node,finished
```

`note` classifies a failure from the job's own `.err`: `UNSUPPORTED_ARCH`, `GPU_OOM`,
`HOST_OOM`, `KILLED_OR_TIMEOUT` or `FAILED`. The `gpu` column records the hardware **actually
allocated**, not the `GPU=` target — on a MIG node it reads `<model>_<profile>`.

The three SfM jobs also echo two machine-readable lines per task, for scraping a run summary
out of the logs without re-reading the CSV:

```
SCENE_INFO scene=<name> sfm=<method> images=<n> masks=<n>
RESULT scene=<name> sfm=<method> exit=<code> seconds=<n>
```

`SCENE_INFO` is printed once the scene is resolved, `RESULT` from the same `EXIT` trap that
writes the CSV row — so a job killed mid-run still emits it. Keep both prefixes stable.

## Picking a GPU

`GPU=rtx6000` (default) or `GPU=b200` selects the conda env, CUDA module and arch string. It
does **not** change the partition — pass that too; a mismatch warns rather than aborting:

```bash
GPU=b200 sbatch --partition=hpg-b200 hpg_slurm/vggt_sfm.sbatch
```

VGGT wants >= 80 GB VRAM on large scenes, so `hpg-b200` is often right there. The
reconstruction backends run fine on `hpg-rtx6000`.

## Logs and monitoring

Logs land in `hpg_slurm/logs/<job-name>-<job-id>.{out,err}` (gitignored), each opening with a
banner naming the node, GPU target, conda env and repo commit.

```bash
squeue -u $USER
scancel <jobid>
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS
```

## Gotchas

- **`~/.bashrc` is not sourced in a batch job.** `common.sh` loads the modules and exports
  `HF_HOME`/`TORCH_HOME` explicitly; your interactive environment does not carry over.
- **COLMAP is intentionally not module-loaded.** Every SfM path drives the `pycolmap` Python
  API from the conda env, so no COLMAP binary is needed anywhere.
- **The jobs call the bare `augenblick` console script**, so the package must be installed
  (`pip install -e . --no-deps --no-build-isolation`) into each per-GPU conda env.
- **`GPU=` does not pick the partition.** Pass `--partition=` to match, or the job runs one
  card's env on the other's hardware. The banner warns; it does not abort.
- **rembg can silently fall back to CPU** (~4x slower) when onnxruntime cannot load its CUDA
  provider, and `--only_missing` then caches those slow masks. `common.sh` prepends the env's
  `site-packages/nvidia/*/lib` dirs to `LD_LIBRARY_PATH` to prevent it; `mask.sbatch` aborts if
  the provider is still missing.
- **A timeout is invisible by exit code.** Slurm SIGTERMs at the time limit and bash's EXIT trap
  then sees status 0, so a timed-out job looks successful. `scene_common.sh` greps the `.err`
  for `DUE TO TIME LIMIT` first, recording `KILLED_OR_TIMEOUT` with exit code 124 — any new
  classification must go *before* the `$status -ne 0` check.
- **The timing CSV header is written only when the file is absent**, so a schema change
  silently misaligns every CSV already on disk. Migrate them, or the two writers disagree.
- **Slurm snapshots the batch script at submit time.** Editing an `.sbatch` after `sbatch`
  returns does not affect the queued or running job; save first, then submit.
- **`setup_env.sbatch`'s partition must carry the same card as `GPU=`.** The CUDA rasterizers
  compile for the arch of the node they build on, so a mismatch yields an env that imports
  cleanly and fails at kernel launch; the script refuses it. The env name follows the *card*,
  not the alias: `GPU=rtx6000` builds `augenblick_rtx_pro_6000`.
- **Never run `scripts/setup_*.sh` from two concurrent jobs against one checkout** — they race
  on the same `build/` dirs and silently reuse stale artifacts. Use a separate checkout per
  parallel build; training jobs sharing a checkout are fine.
- **2DGS TSDF extraction needs far more than 64 gb** at `--mesh_res 4096` on a ~660-image scene
  (97 GiB RSS, oom_killed). Nodes carry 515-2063 gb, so raise `--mem` freely. A host-RAM OOM
  reads as `OUT_OF_MEMORY` / `-9` with a `Detected 1 oom_kill event` line — not a CUDA OOM.
