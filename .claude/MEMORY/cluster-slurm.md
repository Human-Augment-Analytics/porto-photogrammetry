# Cluster execution (SLURM)

Batch job scripts live in two parallel sets, one per cluster: `hpg_slurm/` (HiPerGator) and
`pace_slurm/` (PACE ICE). They replace the manual
`salloc -N1 -t8:00:00 --cpus-per-task 32 --ntasks-per-node=1 --partition=hpg-rtx6000 --gpus=1`
workflow. The job logic is identical in both; only the cluster specifics differ (account,
partition, GPU selection, module names, conda root, data roots). Each dir has its own README.

## Site facts

### HiPerGator (`hpg_slurm/`)

- Slurm 25.11.6. Account `arthur.porto` (an `arthur.porto-phenomics` association also exists).
- Partitions in use: `hpg-rtx6000` (45 nodes, `Gres=gpu:rtx_pro_6000:8`, MaxTime 14 days) and
  `hpg-b200` (60 nodes, `Gres=gpu:b200:8`). Others exist: `hpg-default`, `hpg-milan`,
  `hpg-turin`, `bigmem`, `hpg-dev` (12 h).
- Prebuilt conda envs at `/blue/arthur.porto/srizvi63.gatech/conda/`: `augenblick_rtx_pro_6000`,
  `augenblick_b200`, `gaussian_wrapping`, `meshroom`. These are *not* the `augenblick` env named
  in the README quick start.
- Selects a GPU with `--gpus=1` plus a per-model **partition**.

### PACE ICE (`pace_slurm/`)

- No `--account` line: ICE assigns the account (`coc-ice`) itself. Partition `ice-gpu` (16 h),
  with `ice-bw-gpu` (18 h) for the Blackwell nodes.
- Selects a GPU with `--gres=gpu:<model>:1`, **not** a partition. Switching card means changing
  the gres. Per-model availability and counts: `pace_slurm/README.md`.
- Conda root `$HOME/scratch/conda` (home is capped at 30 GB); all envs there are
  cu129 / torch 2.9.1 / numpy 2.x. `conda.sh` is not on the default
  path; `common.sh` sources it from the PACE anaconda install and takes a `CONDA_SH` override.
- PACE has no `xerces`/`yasm` modules; nothing here needs them, so its `common.sh` omits them.
- Never submit a bare `--gres=gpu:1` — the `mi210` nodes are AMD/ROCm and every CUDA backend
  fails there.

## Files

| File | Role |
|------|------|
| `scene_common.sh` | Roots, scene discovery, banner, timing CSV. Sourced by the two setup files below, never by a job directly (PACE only) |
| `common.sh` | GPU switch, module loads, conda activate; then sources `scene_common.sh` |
| `meshroom_common.sh` | AliceVision env + sm ceiling gate; then sources `scene_common.sh` (PACE only) |
| `template.sbatch` | Copy-and-edit starting point |
| `mask.sbatch` | `MASK_METHOD=rembg\|threshold`; consumes `<scene>/images`, emits a scene |
| `vggt_sfm.sbatch` | VGGT -> COLMAP, one array task per scene |
| `vggt_ba_sfm.sbatch` | Same with `--use_ba` |
| `colmap_sfm.sbatch` | Masked COLMAP SfM |
| `turntable_sfm.sbatch` | Turntable refinement; runs the input SfM first if absent (`SFM=`) |
| `hull_sfm.sbatch` | Visual-hull init; same, and requires masks |
| `recon.sbatch` | `BACKEND=2dgs\|sugar\|pgsr\|gw`, `SFM=<name>` picks the input SfM |
| `meshroom_benchmark.sbatch` | Meshroom baseline (PACE only) |

The `scene_common.sh` split exists because scene discovery, the array-index mapping and the
timing CSV must be identical across methods for results to compare; environment setup is the
one thing that legitimately differs, so it stays in the two callers. Callers set `GPU` and
`CONDA_ENV` before sourcing, and may set `BANNER_EXTRA`.

All job scripts take `--scene`/`--output` built from the scene roots, not positionals, and
forward `"$@"` to the `augenblick` CLI. Scenes are **sorted**, so an array index maps to the
same scene across submissions — which is why discovery must not silently skip any: it matches
`images/` with `-type d -o -xtype d`, since a symlinked `images/` would otherwise drop the
scene and shift every later index.

Two layout notes for any script chaining mask -> SfM -> recon over one scene. A flat scene set
(`<scene>/{images,masks}`) is auto-detected, but `DATA_ROOT` must be exported *before* sourcing
if it differs from the default. And keying outputs by GPU (`<root>/<gpu>/<scene>/...`) is what
lets the same scene run on several cards without the runs overwriting each other.

COLMAP on ~660 images costs 1-3 h, so a chained script should reuse a converged `sparse/0`
rather than redo it after a failure in a later stage — otherwise every recon-stage OOM or bad
GPU pays for SfM twice.

## Masking stage

The `mask` stage's default method (`rembg`) needs `rembg` + `onnxruntime-gpu`, both ordinary `requirements.txt` entries, so every `augenblick_*` env carries them and a masking job uses the per-GPU env `common.sh` already selects.

`common.sh` prepends the `site-packages/nvidia/*/lib` dirs to `LD_LIBRARY_PATH` after
`conda activate` (15 dirs). torch preloads those libraries itself, but rembg imports
onnxruntime without torch, so without this hook ORT finds no `libcudnn.so.9`/`libcufft.so.11`
and drops to CPU. A one-line echo of the dir count beside the mask call makes that visible.

## Data roots

HiPerGator sets them per-sbatch as absolute paths; PACE centralises them in `scene_common.sh`
relative to the checkout (which lives on scratch). Override either in the environment to relocate.

| | HiPerGator | PACE ICE |
|---|---|---|
| `DATA_ROOT` | `/blue/arthur.porto/data/datasets/photogrammetry/main` | `$REPO_ROOT/data/main` |
| `RESULT_ROOT` | `/blue/arthur.porto/srizvi63.gatech/results` | `$REPO_ROOT/output` |

In: `<scene>/prepared/{images,masks}` (nested) or `<scene>/{images,masks}` (flat), auto-detected.
Out: `<scene>/<sfm>[-<backend>]/`.

## Timing rows

`scene_common.sh` appends one row per job to `$RESULT_ROOT/_timing/sfm_timings.csv`. Schema,
columns and the MIG caveat: `pace_slurm/README.md`. Two traps that bite when editing it:

- **The header is written only when the file does not exist**, so changing the schema silently
  misaligns every CSV already on disk. Migrate them, or the two writers disagree.
- **A timeout is invisible by exit code** — Slurm SIGTERMs at the time limit and bash's EXIT
  trap then sees status 0. The trap greps the job's own `.err` for `DUE TO TIME LIMIT` before
  trusting the status. Any new classification must go *before* the `$status -ne 0` check.

## GPU switch

`GPU=<name>` selects env, CUDA module, and `TORCH_CUDA_ARCH_LIST`. It does **not** change the
partition or gres; pass those too.

| Cluster | GPU | Env | CUDA module | Arch |
|---|-----|-----|-------------|------|
| HPG | `rtx6000` (default) | `augenblick_rtx_pro_6000` | `cuda/13.0.2` | 12.0 |
| HPG | `b200` | `augenblick_b200` | `cuda/12.8` | 10.0 |
| PACE | `a100` (default) | `augenblick_a100` | `cuda/12.9.1` | 8.0 |
| PACE | `l40s` | `augenblick_l40s` | `cuda/12.9.1` | 8.9 |
| PACE | `a40` | `augenblick_a40` | `cuda/12.9.1` | 8.6 |

Arch strings mirror `GPU_ARCH` in the matching `scripts/setup_<gpu>.sh`; `common.sh` exits 2 with a build hint if the env is absent. Note: `scripts/auto_setup.sh` maps compute cap 8.9 to `setup_l40s.sh`, which serves every Ada
sm_8.9 card (L40S, L40, RTX 6000 Ada) — the RTX 6000 Ada is a *different* card from the Blackwell RTX Pro 6000 on `hpg-rtx6000`.

## Deliberate choices

- **No `module load colmap/3.11`, no `export -f colmap`.** Every SfM path goes through the
  `pycolmap` Python API from the conda env and never shells out, so no COLMAP binary is needed.
- `module purge` first, since a batch shell inherits no `~/.bashrc`.
- `conda activate` requires sourcing `conda.sh` first in a non-interactive shell.
- `--mem` set explicitly (the interactive `salloc` let it default): 24 gb for SfM jobs, 64 gb for
  reconstruction and the template. **64 gb is not enough for 2DGS mesh extraction on a ~660-image
  scene** — TSDF at `--mesh_res 4096` peaked at 97 GiB RSS and was `oom_kill`ed at 64 gb; 192 gb
  cleared it. Nodes here carry 515–2063 gb, so headroom is free. A host-RAM OOM reads as state
  `OUT_OF_MEMORY`, return code `-9`/SIGKILL and a `Detected 1 oom_kill event` line — **not** a
  CUDA OOM, and not a finding about the GPU.
- **Slurm snapshots the batch script at submit time.** Editing an sbatch file after `sbatch`
  returns does not affect the queued or running job — save first, then submit, and never issue
  the edit and the `sbatch` in the same breath.
- `scontrol update MinMemoryNode=` wants **megabytes with no suffix** (`196608`); `192G` is
  rejected as "Invalid MinMemoryNode value".
- Slurm copies the script to `/var/spool`, so `$0` cannot locate the repo; each script resolves
  `common.sh` via `$SLURM_SUBMIT_DIR` and errors out if submitted from elsewhere.
- Logs to `<dir>/logs/%x-%A_%a.{out,err}` (`%x-%j` for the non-array template), gitignored.
