# Cluster execution (SLURM)

Batch job scripts live in two parallel sets: `hpg_slurm/` (HiPerGator) and `pace_slurm/`
(PACE ICE), replacing the manual `salloc ... --gpus=1` workflow. The job logic is identical;
only cluster specifics differ (account, partition, GPU selection, modules, conda root, data
roots). Each dir has its own README.

## Site facts

### HiPerGator (`hpg_slurm/`)

- Slurm 25.11.6. Account `arthur.porto` (an `arthur.porto-phenomics` association also exists).
- Partitions in use: `hpg-rtx6000` (45 nodes, `Gres=gpu:rtx_pro_6000:8`, MaxTime 14 days) and
  `hpg-b200` (60 nodes, `Gres=gpu:b200:8`). Others exist: `hpg-default`, `hpg-milan`,
  `hpg-turin`, `bigmem`, `hpg-dev` (12 h).
- Prebuilt envs at `/blue/arthur.porto/srizvi63.gatech/conda/`: `augenblick_rtx_pro_6000`,
  `augenblick_b200`, `gaussian_wrapping`, `meshroom` — *not* the `augenblick` env in the README
  quick start. The name follows the **card**, not the alias: `GPU=rtx6000` selects
  `augenblick_rtx_pro_6000`.
- Selects a GPU with `--gpus=1` plus a per-model **partition**. `scene_common.sh` warns (never
  aborts) when `GPU=` and the partition disagree, since that runs one card's env on another's
  hardware.
- **Meshroom is half-installed**: AliceVision is at
  `/blue/arthur.porto/srizvi63.gatech/alicevision` (binaries, sensor DB, voctree all present),
  but `$MESHROOM_ROOT` does not exist, so `meshroom_common.sh` exits 2 pointing at
  `meshroom-setup.md`. Verified 2026-09-21.

### PACE ICE (`pace_slurm/`)

- No `--account` line: ICE assigns the account (`coc-ice`) itself. Partition `ice-gpu` (16 h),
  with `ice-bw-gpu` (18 h) for the Blackwell nodes.
- Selects a GPU with `--gres=gpu:<model>:1`, **not** a partition. Switching card means changing
  the gres. Per-model availability and counts: `pace_slurm/README.md`.
- Conda root `$HOME/scratch/conda` (home is capped at 30 GB); all envs are cu129 / torch 2.9.1
  / numpy 2.x. `conda.sh` is not on the default path; `common.sh` sources it from the PACE
  anaconda install and takes a `CONDA_SH` override.
- PACE has no `xerces`/`yasm` modules; nothing here needs them, so its `common.sh` omits them.
- Never submit a bare `--gres=gpu:1` — the `mi210` nodes are AMD/ROCm and every CUDA backend
  fails there.

## Files

Both dirs carry the same set unless a row says otherwise.

| File | Role |
|------|------|
| `scene_common.sh` | Roots, scene discovery, banner, timing CSV. Sourced by the two setup files below, never by a job directly |
| `common.sh` | GPU switch, module loads, `HF_HOME`/`TORCH_HOME` exports, conda activate; then sources `scene_common.sh` |
| `meshroom_common.sh` | AliceVision env + sm gate (`MESHROOM_MAX_SM`); then sources `scene_common.sh` |
| `template.sbatch` | Copy-and-edit starting point |
| `mask.sbatch` | `MASK_METHOD=rembg\|threshold\|sam3`; consumes `<scene>/images`, emits a scene. `sam3` preflights CUDA |
| `vggt_sfm.sbatch` | VGGT -> COLMAP, one array task per scene |
| `vggt_ba_sfm.sbatch` | Same with `--use_ba` |
| `colmap_sfm.sbatch` | Masked COLMAP SfM |
| `turntable_sfm.sbatch` | Turntable refinement; runs the input SfM first if absent (`SFM=`) |
| `hull_sfm.sbatch` | Visual-hull init; same, and requires masks |
| `recon.sbatch` | `BACKEND=2dgs\|sugar\|pgsr\|gw`, `SFM=<name>` picks the input SfM |
| `meshroom_benchmark.sbatch` | Meshroom baseline |
| `setup_env.sbatch` | Builds `augenblick_<card>` on a node of that GPU; refuses an arch mismatch |
| `prepare_neurips.sbatch` | Flattens raw NeurIPS objects into `<scene>/{images,masks}` (HPG only) |

The split exists because discovery, the array-index mapping and the timing CSV must be
identical across methods for results to compare; env setup is the one thing that legitimately
differs, so it stays in the callers. They set `GPU` and `CONDA_ENV` first, and may set
`BANNER_EXTRA`.

All jobs take `--scene`/`--output` built from the roots, not positionals, and forward `"$@"`
to the CLI. Scenes are **sorted**, so an array index maps to the same scene across
submissions — hence discovery must skip none: it matches `images/` with `-type d -o -xtype d`,
since a symlinked `images/` would drop the scene and shift every later index.

Two notes for any script chaining mask -> SfM -> recon over one scene: a flat scene set is
auto-detected, but a non-default `DATA_ROOT` must be exported *before* sourcing; and keying
outputs by GPU (`<root>/<gpu>/<scene>/...`) is what lets one scene run on several cards
without the runs overwriting each other.

COLMAP on ~660 images costs 1-3 h, so a chained script should reuse a converged `sparse/0`
rather than redo it after a later-stage failure — else every recon OOM pays for SfM twice.

## Masking stage

The default method (`rembg`) needs `rembg` + `onnxruntime-gpu`, both ordinary
`requirements.txt` entries, so every `augenblick_*` env carries them and a masking job uses the
per-GPU env `common.sh` already selects.

`common.sh` prepends the `site-packages/nvidia/*/lib` dirs (15 of them) to `LD_LIBRARY_PATH`
after `conda activate`. torch preloads those libraries itself, but rembg imports onnxruntime
without torch, so without the hook ORT finds no `libcudnn.so.9`/`libcufft.so.11` and drops to
CPU.

## Data roots

Both centralise them in `scene_common.sh`: HPG uses absolute `/blue/...` paths, PACE derives
its own from the checkout (on scratch). Override either in the environment.

| | HiPerGator | PACE ICE |
|---|---|---|
| `DATA_ROOT` | `/blue/arthur.porto/data/datasets/photogrammetry/neurips/prepared` | `$REPO_ROOT/data/main` |
| `RESULT_ROOT` | `/blue/arthur.porto/srizvi63.gatech/results/neurips` | `$REPO_ROOT/output` |

In: `<scene>/prepared/{images,masks}` (nested) or `<scene>/{images,masks}` (flat), auto-detected.
Out: `<scene>/<sfm>[-<backend>]/`. Runs under an older `<scene>/all/<sfm>/` predate the current
scripts and are not discovered by them.

**Discovery matches at a fixed depth, so `DATA_ROOT` must be the dir whose immediate children
are the scenes.** The HPG `main` set is now `.../photogrammetry/main/specimens`: `main/` gained
`specimens/` and `realityscan/` above the scenes, so pointing `DATA_ROOT` at `main/` finds no
`prepared/` dirs, falls back to `flat`, then finds nothing. Verified 2026-09-21 on disk:
6 scenes under `main/specimens`, 76 under `neurips/prepared`.

## Timing rows

`scene_common.sh` appends one row per job to `$RESULT_ROOT/_timing/sfm_timings.csv`. Schema,
columns and the MIG caveat: either dir's `README.md`. Notes are `UNSUPPORTED_ARCH`, `GPU_OOM`,
`HOST_OOM`, `KILLED_OR_TIMEOUT` or `FAILED`.

On HPG the same hook also prints a `RESULT scene=... sfm=... exit=... seconds=...` line, which
pairs with the `SCENE_INFO` line the three SfM jobs echo so a run summary can be scraped from
the logs. That is why `start_timing` takes an optional third argument (the method) there: one
trap emitting both the CSV row and the marker, rather than two racing to be the EXIT handler.

Two traps that bite when editing it:

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
- For the same reason `common.sh` exports `HF_HOME`/`TORCH_HOME` (both overridable) — to
  `$HOME/scratch/` on PACE, `/blue/arthur.porto/srizvi63.gatech/cache/` on HPG. Without them every checkpoint-fetching job (VGGT ~4 GB, SAM 3 ~2 GB)
  re-downloads into `$HOME/.cache`, against a 30 GB home quota.
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
