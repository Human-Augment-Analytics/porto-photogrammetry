# SLURM job scripts — PACE ICE

Batch equivalents of the manual `salloc` workflow, so runs survive a dropped SSH session.
This is the **PACE ICE** set; `hpg_slurm/` holds the HiPerGator equivalents. The two differ
only in cluster specifics (account, partition, gres, module names, conda root) — the job
logic is identical.

## Submit

```bash
# One array task per scene found under DATA_ROOT; add --array=N for a single scene.
sbatch pace_slurm/vggt_sfm.sbatch
sbatch pace_slurm/vggt_ba_sfm.sbatch
sbatch pace_slurm/colmap_sfm.sbatch
SFM=colmap sbatch pace_slurm/turntable_sfm.sbatch
BACKEND=2dgs SFM=vggt sbatch pace_slurm/recon.sbatch
```

`recon.sbatch` accepts `BACKEND=2dgs|sugar|pgsr|gw` and `SFM=<name>` to pick which SfM result
feeds it; `turntable_sfm.sbatch` takes `SFM=` too and runs that SfM first if its output is
missing. Scenes are discovered under `DATA_ROOT` and sorted, so an array index maps to the same
scene across submissions. Extra flags are forwarded to the `augenblick` CLI.

For anything not covered, copy `template.sbatch` and edit its command block.

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
RESULT_ROOT = $REPO_ROOT/output         # <scene>/all/<sfm>[-<backend>]/
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

## Picking a GPU

`GPU=a100` (default), `l40s`, or `a40` selects the conda env, CUDA module, and arch string in `common.sh` (all three are `cuda/13.0.1` / torch 2.9.1). It does **not** change the gres —
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
- **PACE's default CUDA module is `cuda/12.9.1`**, but the A100 env is built against `cuda/13.0.1`. `common.sh` loads the matching one explicitly.
- **COLMAP is intentionally not module-loaded.** Every SfM path drives the `pycolmap` Python API from the conda env, so no COLMAP binary is needed anywhere.
- **The jobs call the bare `augenblick` console script**, so the package must be installed (`pip install -e . --no-deps --no-build-isolation`) into each per-GPU conda env.
- **Never run `scripts/setup_*.sh` from two concurrent jobs against one checkout** — they race on the same `build/` dirs and silently reuse stale artifacts. Use a separate checkout per parallel build. Training jobs sharing a checkout are fine.
