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
  the gres. Models on `ice-gpu`: `a100`, `l40s`, `h100`, `h200`, `a40`, `rtx_6000`, `v100`.
- Conda root `$HOME/scratch/conda` (home is capped at 30 GB); all envs there are
  cu130 / torch 2.9.1 / numpy 2.x. `conda.sh` is not on the default
  path; `common.sh` sources it from the PACE anaconda install and takes a `CONDA_SH` override.
- PACE has no `xerces`/`yasm` modules; nothing here needs them, so its `common.sh` omits them.
- A100s are the scarce card (8 cluster-wide); `l40s`/`h200` queue far faster. Never submit a
  bare `--gres=gpu:1` — the `mi210` nodes are AMD/ROCm and every CUDA backend fails there.

## Files (mirrored in both dirs)

| File | Role |
|------|------|
| `common.sh` | Sourced by every script: GPU switch, module loads, conda activate, banner |
| `template.sbatch` | Copy-and-edit starting point |
| `vggt_sfm.sbatch` | VGGT -> COLMAP, one array task per scene |
| `vggt_ba_sfm.sbatch` | Same with `--use_ba`, writing to `all/vggt_ba` |
| `colmap_sfm.sbatch` | Masked COLMAP SfM |
| `turntable_sfm.sbatch` | Turntable refinement; runs the input SfM first if absent (`SFM=`) |
| `recon.sbatch` | `BACKEND=2dgs\|sugar\|pgsr\|gw`, `SFM=<name>` picks the input SfM |

All job scripts take `--scene`/`--output` built from the scene roots, not positionals, and
forward `"$@"` to the `augenblick` CLI. Scenes are discovered as `<scene>/prepared` dirs under
`DATA_ROOT` and **sorted**, so an array index maps to the same scene across submissions.

## Data roots

HiPerGator sets them per-sbatch as absolute paths; PACE centralises them in `common.sh` relative
to the checkout (which lives on scratch). Override either in the environment to relocate.

| | HiPerGator | PACE ICE |
|---|---|---|
| `DATA_ROOT` | `/blue/arthur.porto/data/datasets/photogrammetry/main` | `$REPO_ROOT/data/main` |
| `RESULT_ROOT` | `/blue/arthur.porto/srizvi63.gatech/results` | `$REPO_ROOT/output` |

Layout is `<scene>/prepared/{images,masks}` in, `<scene>/all/<sfm>[-<backend>]/` out.

## GPU switch

`GPU=<name>` selects env, CUDA module, and `TORCH_CUDA_ARCH_LIST`. It does **not** change the
partition or gres; pass those too.

| Cluster | GPU | Env | CUDA module | Arch |
|---|-----|-----|-------------|------|
| HPG | `rtx6000` (default) | `augenblick_rtx_pro_6000` | `cuda/13.0.2` | 12.0 |
| HPG | `b200` | `augenblick_b200` | `cuda/12.8` | 10.0 |
| PACE | `a100` (default) | `augenblick_a100` | `cuda/13.0.1` | 8.0 |
| PACE | `l40s` | `augenblick_l40s` | `cuda/13.0.1` | 8.9 |
| PACE | `a40` | `augenblick_a40` | `cuda/13.0.1` | 8.6 |

Arch strings mirror `GPU_ARCH` in the matching `scripts/setup_<gpu>.sh`; `common.sh` exits 2 with a build hint if the env is absent. Note: `scripts/auto_setup.sh` maps compute cap 8.9 to `setup_l40s.sh`, which serves every Ada
sm_8.9 card (L40S, L40, RTX 6000 Ada) — the RTX 6000 Ada is a *different* card from the Blackwell RTX Pro 6000 on `hpg-rtx6000`.

## Deliberate choices

- **No `module load colmap/3.11`, no `export -f colmap`.** Every SfM path goes through the
  `pycolmap` Python API from the conda env and never shells out, so no COLMAP binary is needed.
- `module purge` first, since a batch shell inherits no `~/.bashrc`.
- `conda activate` requires sourcing `conda.sh` first in a non-interactive shell.
- `--mem` set explicitly (the interactive `salloc` let it default): 24 gb for SfM jobs, 64 gb for
  reconstruction and the template.
- Slurm copies the script to `/var/spool`, so `$0` cannot locate the repo; each script resolves
  `common.sh` via `$SLURM_SUBMIT_DIR` and errors out if submitted from elsewhere.
- Logs to `<dir>/logs/%x-%A_%a.{out,err}` (`%x-%j` for the non-array template), gitignored.
