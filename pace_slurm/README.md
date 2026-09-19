# SLURM job scripts — PACE ICE

Batch equivalents of the manual `salloc` workflow, so runs survive a dropped SSH session.
This is the **PACE ICE** set; `hpg_slurm/` holds the HiPerGator equivalents. The two differ
only in cluster specifics (account, partition, gres, module names, conda root) — the job
logic is identical.

## Submit

```bash
# One array task per scene found under DATA_ROOT; add --array=N for a single scene.
MASK_METHOD=rembg sbatch pace_slurm/mask.sbatch
sbatch pace_slurm/vggt_sfm.sbatch
sbatch pace_slurm/vggt_ba_sfm.sbatch
sbatch pace_slurm/colmap_sfm.sbatch
SFM=colmap sbatch pace_slurm/turntable_sfm.sbatch
SFM=colmap sbatch pace_slurm/hull_sfm.sbatch
BACKEND=2dgs SFM=vggt sbatch pace_slurm/recon.sbatch
```

`recon.sbatch` accepts `BACKEND=2dgs|sugar|pgsr|gw` and `SFM=<name>` to pick which SfM result
feeds it; `turntable_sfm.sbatch` takes `SFM=` too and runs that SfM first if its output is
missing. Scenes are discovered under `DATA_ROOT` and sorted, so an array index maps to the same
scene across submissions. Extra flags are forwarded to the `augenblick` CLI.

`mask.sbatch` runs the masking stage and takes `MASK_METHOD=rembg|threshold` (default `rembg`)
and `OUT_LAYOUT=nested|flat` (default `nested`). `nested` writes
`$RESULT_ROOT/<method>/<scene>/` so the two methods coexist; `flat` writes straight to
`$RESULT_ROOT/<scene>/`, which is convenient when `RESULT_ROOT` is already scoped to one run
but means a second method silently overwrites the first.
Unlike every other job here it consumes `<scene>/images` rather than a whole scene, and writes a
new scene-shaped output — an `images/` symlink plus `masks/`. Point the SfM jobs at that
directory to run them masked. It is the stage that *produces* masks, so it is the one job that does not need them
present up front; `--only_missing` makes a requeued job resume where it stopped. The `rembg`
method fails the job outright if onnxruntime cannot load its CUDA provider, rather than falling
back to CPU at ~4x the cost. Its timing rows record the **scene** in the scene column, like
every other job, and take the method from the `method` column — which is the Slurm job name. To
tell two methods apart in one CSV, submit with a matching job name:
`MASK_METHOD=threshold sbatch --job-name=mask-threshold pace_slurm/mask.sbatch`.

`hull_sfm.sbatch` carves the visual hull from the masks and writes it as the initial point
cloud. Its output differs from the input SfM only in `sparse/0/points3D.ply`, so running
`recon.sbatch` against both gives an initialisation ablation with poses, intrinsics and the
training schedule held fixed. It needs masks and will refuse to run without them.

**VGGT is VRAM-hungry.** It loads every image at 1024² in a single batch with no chunking, so
VRAM scales with scene size: on the 48 GB L40S it OOM'd on all 47 NeurIPS scenes, and even on a
141 GB H200 it OOM's above ~276 images. BA needs at least as much again. Run it on the largest
card available, and expect the biggest scenes to fail regardless.

**Unmasked BA fails on most scenes.** 36 of 47 died with *No reconstruction can be built with
BA* — the tracker found too few surviving correspondences. Masking first fixes it: confining
query points to the specimen is what the larger `max_query_pts` budget in `VGGTConfig` is sized
for. Run `mask.sbatch`, then point `vggt_ba_sfm.sbatch` at the masked scenes.

To rerun a method on a different card without clobbering the first run, override the gres and
give the job a distinct name and output dir — the `method` column is the Slurm job name, so the
two runs stay separable in one CSV:

```bash
GPU=h200 RESULT_ROOT=output/h200 sbatch --gres=gpu:h200:1 \
    --job-name=vggt-sfm-h200 pace_slurm/vggt_sfm.sbatch
```

For anything not covered, copy `template.sbatch` and edit its command block.

## Shared job setup

`scene_common.sh` holds what must be identical across methods for their results to be
comparable: root resolution, scene discovery, the array-index → scene mapping, the run
banner, and the timing CSV. It is not sourced directly by a job — `common.sh` (torch/CUDA)
and `meshroom_common.sh` (AliceVision) each set up their own environment and then source it.
Callers set `GPU` and `CONDA_ENV` first, and may set `BANNER_EXTRA` to splice extra
`label: value` lines into the banner.

Scene discovery matches `images/` via `-type d -o -xtype d`: a scene's `images/` is often a
symlink into shared storage, and matching only real dirs would silently drop those scenes and
shift every later array index — so task *N* would mean a different scene in different jobs.

## Meshroom baseline

`meshroom_benchmark.sbatch` sources `meshroom_common.sh`, not `common.sh`, for three reasons:
the frontend is a single GPU-independent env (`$CONDA_ROOT/meshroom`, pure Python) rather than
one env per GPU; AliceVision 3.3.0's tarball bundles its own `libcudart.so.12.1.105`, so
loading a CUDA module would shadow it; and the `ALICEVISION_*` variables live only in
`~/.bashrc`, which a batch job never sources.

**Supported GPUs: `v100`, `rtx_6000`, `a100`, `a40`, `l40s`, `h100`, `h200`.** AliceVision's
prebuilt kernels are SASS-only up to sm_90 with **no PTX** (verified with `cuobjdump` on
`libaliceVision_depthMap.so`, `libpopsift.so`, `libCCTag.so`). No PTX means no JIT, so a newer
device cannot fall back — DepthMap dies with *no kernel image is available for execution on
the device*. `rtx_pro_6000_blackwell` (sm_120) and `mi210` (AMD) are rejected up front.

**It runs one scene at a time (`--array=...%1`), deliberately.** `MeshroomCache` reaches
~50 GB for a 663-image scene against a 300 GB Lustre quota; running 8-wide exhausted the quota
and killed the whole array. `benchmark_meshroom.py` prunes the cache to just the textured mesh
after each successful scene, so only one scene's intermediates exist at a time. Check headroom
with `lfs quota -h -u $USER /storage/ice1` — **not** `df`, which reports the filesystem's free
space and says nothing about your quota.

DepthMap is host-bound: `computeOnMultiGPUs.cpp` runs one OMP thread per GPU, not per core, and
the 64 KB constant-memory limit clamps batching identically on every card. Extra cores only
help FeatureExtraction and Texturing. An A40 measured *faster* than an H200 here (10.1 vs
12.9 s/depth map), so prefer the more available card.

## Cluster specifics

| | PACE ICE | HiPerGator |
|---|---|---|
| Account | `coc-ice` | `arthur.porto` |
| GPU partition | `ice-gpu` (16 h) | `hpg-rtx6000` |
| GPU selection | `--gres=gpu:a100:1` | `--gpus=1` + per-model partition |
| Conda root | `$HOME/scratch/conda` | `/blue/arthur.porto/srizvi63.gatech/conda` |

ICE picks the GPU model through **gres**, not a partition, so switching GPU means changing
`--gres=gpu:<model>:1` — not `--partition`. Models available on `ice-gpu`: `a100`, `l40s`,
`h100`, `h200`, `a40`, `rtx_6000`, `v100`. Check with `sinfo -p ice-gpu -o "%N %G"`.

## Data layout

`DATA_ROOT` and `RESULT_ROOT` are set in `common.sh` and default to paths inside the repo
checkout (which lives on scratch), not to absolute cluster paths:

```
DATA_ROOT   = $REPO_ROOT/data/main      # <scene>/prepared/{images,masks}
RESULT_ROOT = $REPO_ROOT/output         # <scene>/<sfm>[-<backend>]/
```

Override either in the environment to relocate. Prepared scenes are built with:

```bash
python pipeline/preparation/prepare_uf_dataset.py data/main/<scene>/images \
    --out data/main/<scene>/prepared --mode symlink [--include-unmatched]
```

`--mode symlink` avoids duplicating the imagery (home is capped at 30 GB).
`--include-unmatched` is required for scenes that ship **no masks** — without it every image is
dropped and `prepared/images` comes out empty. Of the six scenes, only `TH24-21_Birdsnest`
needs it.

## Timing rows

Every job appends one row to `$RESULT_ROOT/_timing/sfm_timings.csv`:
`method,scene,gpu,n_images,seconds,exit_code,note,jobid,node,finished`. The `gpu` column records
the hardware **actually allocated**, not the `GPU=` target: on a MIG-partitioned node it reads
`<model>_<profile>` (e.g. `rtx_pro_6000_2g.48gb`) and on a whole card just `<model>`.

This matters on `ice-bw-gpu`, where each node advertises **16** `rtx_pro_6000_blackwell` gres
across 4 physical cards — so one gres is a MIG slice (2/12 of the SMs, 48 of 96 GB), not a card.
Recording the bare model there would compare a fraction of a Blackwell against a whole L40S.

## Picking a GPU

`GPU=a100` (default), `l40s`, or `a40` selects the conda env, CUDA module, and arch string in `common.sh` (all three are `cuda/12.9.1` / torch 2.9.1). It does **not** change the gres —
override that too:

```bash
GPU=l40s sbatch --gres=gpu:l40s:1 pace_slurm/vggt_sfm.sbatch
```

VGGT wants >= 80 GB VRAM on large scenes; the `a100` gres on ICE is the 80 GB PCIe model, so the default target is already appropriate. The reconstruction backends run fine on less.

`augenblick_a100`, `augenblick_l40s`, and `augenblick_a40` are built and complete. `common.sh` fails with a build hint for a missing env.

### What's actually on the cluster

Counts from `sinfo` (2026-09-05). Partitions share physical nodes, so these per-partition rows double-count the same hardware — the cluster totals are unique-node counts.

| GPU type | Cluster total | Nodes | On `ice-gpu` |
|---|---|---|---|
| `h100` | 151 | 19 | 48 |
| `h200` | 144 | 18 | 48 |
| `rtx_pro_6000_blackwell` | 64 | 4 | — (`ice-bw-gpu`) |
| `l40s` | 32 | 4 | 32 |
| `v100` | 22 | 11 | 22 |
| `rtx_6000` | 8 | 2 | 8 |
| `a100` | 8 | 4 | 8 |
| `mi210` | 4 | 2 | 4 |
| `a40` | 4 | 2 | 4 |

Other partitions: `coe-gpu` holds the bulk of the H100/H200 fleet (151 + 144) but needs a CoE account; `coc-gpu` mirrors `ice-gpu` minus the H100/H200; `pace-gpu` has no `h100`, `h200`, or
`l40s` at all.

Three things that follow from the table:

- **A100s are the scarce resource** — 8 in total, 2 per node, shared across four partitions. `l40s` (8/node, 48 GB) and `h200` (8/node, 141 GB) queue far faster per GPU.
- **Never submit a bare `--gres=gpu:1`.** The `mi210` nodes are AMD/ROCm; every CUDA backend under `src/libs/` fails there. Always name the model.
- **`rtx_pro_6000_blackwell` is sm_120** and lives on its own `ice-bw-gpu` partition. The CUDA submodules would need a rebuild against a Blackwell-capable toolchain in a separate env.

## Logs and monitoring

Logs land in `pace_slurm/logs/<job-name>-<job-id>.{out,err}` (gitignored). Each starts with a
banner naming the node, GPU target, conda env, and repo commit.

```bash
squeue -u $USER
scancel <jobid>
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS
```

## Gotchas

- **`~/.bashrc` is not sourced in a batch job.** `common.sh` loads the modules explicitly; do not assume your interactive environment carries over.
- **PACE has no `xerces` or `yasm` modules** (HiPerGator does). Nothing in the pipeline needs them, so the PACE `common.sh` simply omits them.
- **CUDA is `cuda/12.9.1` everywhere** — PACE's default, what the envs are built against (torch 2.9.1+cu129), and what `common.sh` loads explicitly. One generation on purpose: `onnxruntime-gpu` links CUDA 12 sonames, so a cu130 torch leaves the masking stage's GPU provider unloadable. Keep the wrapper, the module load, and the torch index aligned when bumping any of them.
- **COLMAP is intentionally not module-loaded.** Every SfM path drives the `pycolmap` Python API from the conda env, so no COLMAP binary is needed anywhere.
- **The jobs call the bare `augenblick` console script**, so the package must be installed (`pip install -e . --no-deps --no-build-isolation`) into each per-GPU conda env.
- **Never run `scripts/setup_*.sh` from two concurrent jobs against one checkout** — they race on the same `build/` dirs and silently reuse stale artifacts. Use a separate checkout per parallel build. Training jobs sharing a checkout are fine.
- **`setup_env.sbatch`'s gres must name the same card as `GPU=`.** The CUDA rasterizers compile for the arch of the node they build on, so a mismatch yields an env that imports cleanly and fails at kernel launch. The script refuses a mismatch rather than letting it through. Wrapper names do not always match the GPU: H200 shares sm_9.0 with H100 and so uses `scripts/setup_h100.sh`.
- **`rembg` falls back to CPU silently** (~4x slower) when onnxruntime cannot load its CUDA provider, and `--only_missing` then reuses those slow masks on every later run. `mask.sbatch` fails the job outright rather than letting that happen; if you drive the masking stage yourself, log the active provider. `rembg` also fetches its model to `~/.u2net/` on first use — warm it on a login node if the compute nodes have no outbound network.
- **A timeout is invisible by exit code.** Slurm SIGTERMs at the time limit, and bash's EXIT trap then sees status 0, so a timed-out job looks like a successful one. `scene_common.sh` greps the job's own `.err` for `DUE TO TIME LIMIT` before trusting the status, and records `KILLED_OR_TIMEOUT` with exit code 124.
