# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

**Keep this file short.** It is orientation only: what the project is, which entry point to
reach for, and the gotchas that change what you type. All depth — full flag lists, backend
internals, parameter tables, algorithm walk-throughs — lives in `.claude/MEMORY/`. When you
document something new, put the detail in the matching MEMORY file (or add one and index it
below); only add to CLAUDE.md if it changes how the repo is invoked or navigated.

**`.claude/MEMORY/` is the durable record of how this repo works.** When you learn something
about it that would still be true next session and is not already obvious from the code — a
non-obvious flag interaction, a backend quirk, a path that must match elsewhere, a trap that
cost real debugging time — offer to write it to the matching MEMORY file. Verify it before
writing, and say what you verified it against. These files are committed, so propose the edit
rather than making it silently, and keep the entry to the shortest form that stays useful.

Not everything belongs here: one-off session state, personal preferences, and anything the code
or git history already records stay out. A fact about *the repo* goes in `MEMORY/`; a fact about
*how the user works* is Claude's own per-user memory and never lands in this tree.

## MEMORY index

| File | Contents |
|------|----------|
| [environment-and-gpu.md](MEMORY/environment-and-gpu.md) | Install via `scripts/`, per-GPU wrappers, numpy/build gotchas, submodules, GPU notes |
| [data-morphosource.md](MEMORY/data-morphosource.md) | MorphoSource downloader: flags, project 000381689 contents, API gotchas |
| [pipeline-sfm.md](MEMORY/pipeline-sfm.md) | Data prep + all four SfM entry points with full flags (incl. turntable algorithm) |
| [pipeline-reconstruction.md](MEMORY/pipeline-reconstruction.md) | The four reconstruction wrappers, their flags, and output paths |
| [scene-format.md](MEMORY/scene-format.md) | COLMAP scene layout, mask naming, end-to-end data flow |
| [backend-vggt.md](MEMORY/backend-vggt.md) | VGGT architecture, utils, COLMAP conversion, coordinate conventions |
| [backend-sugar.md](MEMORY/backend-sugar.md) | SuGaR four-stage pipeline, extraction, components |
| [backend-2dgs.md](MEMORY/backend-2dgs.md) | 2DGS surfels, losses, TSDF/marching-cubes extraction |
| [backend-pgsr.md](MEMORY/backend-pgsr.md) | PGSR planar Gaussians, multi-view losses, TSDF meshing |
| [backend-gaussian-wrapping.md](MEMORY/backend-gaussian-wrapping.md) | GW three stages, components, CUDA submodule matrix |
| [baseline-meshroom.md](MEMORY/baseline-meshroom.md) | Meshroom wrapper + reference runtimes |
| [repo-conventions.md](MEMORY/repo-conventions.md) | Doc rule, repo layout, legacy code, ID/naming and commit conventions |
| [cluster-slurm.md](MEMORY/cluster-slurm.md) | SLURM job scripts for HiPerGator + PACE ICE, partitions/accounts, GPU switch, batch-env gotchas |
| [augenblick-package.md](MEMORY/augenblick-package.md) | The `src/augenblick` package: ABCs, registry, config bridge, CLI, adding a backend |

`.claude/PLANS/` holds implementation specs for landed refactors — the reasoning and constraints
behind a change, kept as a record of *why* the tree looks as it does. They are historical: where
a plan and the code disagree, the code wins. See [repo-conventions.md](MEMORY/repo-conventions.md).

## Project overview

**Augenblick** is a two-stage photogrammetry pipeline producing 3D meshes from multi-view
images: an SfM initialiser feeds a Gaussian-primitive-based surface reconstructor. The research
question is how different SfM initialisations interact with each mesh extractor.

| Stage | Options |
|-------|---------|
| Data acquisition | `scripts/download_morphosource_project.py` (MorphoSource project 000381689) |
| Data preparation | `pipeline/preparation/prepare_uf_dataset.py` |
| SfM | `augenblick sfm {vggt,colmap,turntable}` (VGGT takes `--use_ba`) |
| Reconstruction | `augenblick recon {sugar,2dgs,pgsr,gw}` |
| Baselines | Meshroom (`baseline/benchmark_meshroom.py`), RealityScan (external) |

Everything between the stages is a COLMAP scene: `images/` + optional `masks/` + `sparse/0/`.

## Quick start

```bash
conda create --name augenblick python=3.10 && conda activate augenblick
git submodule update --init --recursive
bash scripts/auto_setup.sh                  # detects GPU, dispatches to setup_<gpu>.sh
pip install -e . --no-deps --no-build-isolation   # the augenblick CLI; flags are mandatory

augenblick sfm vggt --scene <scene> --output <sfm> --use_ba
augenblick recon 2dgs --scene <sfm> --output <out>
```

There is **no `environment.yml`** — `scripts/auto_setup.sh` (or the manual pip sequence in
README) is the install path. Details and knobs (`BACKENDS`, `SKIP_TETRA`, `NUMPY_GENERATION`,
stale `build/` dirs): [environment-and-gpu.md](MEMORY/environment-and-gpu.md).

## Gotchas worth knowing before you type

- `augenblick sfm turntable` refines an **existing** COLMAP scene; it is a post-SfM step, not a
  standalone SfM. The `SceneRefiner` base class enforces this.
- The `augenblick` CLI comes from `pip install -e . --no-deps --no-build-isolation`, and must be
  installed into **each** per-GPU conda env — the SLURM jobs call the bare console script.
- COLMAP wants masks named `<image_name>.png` (`foo.jpg.png`); the pycolmap scripts build a
  `masks_colmap/` symlink dir to satisfy this.
- Per-backend quirks (GW's no-`cwd` + passthrough, PGSR's `sparse/0/` flattening, SuGaR's
  `--flag True` string booleans) are now class properties — see
  [augenblick-package.md](MEMORY/augenblick-package.md).
- COLMAP IDs are 1-indexed — `+1` offset from VGGT batch indices.
- numpy/scipy/scikit-* versions live in `constraints/numpy{1,2}.txt`, not `requirements.txt`;
  the generation must match the GPU's torch wheel or imports break at runtime, not at install.
- `download_morphosource_project.py` defaults to a **seeded 3-specimen sample** (~3.2 GB), not the
  whole 869 GB project; `--dry-run` costs nothing and needs no API key.
- All third-party backends live under `src/libs/` (`2dgs`, `pgsr`, `sugar`,
  `gaussian_wrapping`, `vggt`, `light_glue`, `pytorch3d`). First-party code is
  `src/augenblick/` (the package), `pipeline/preparation/`, and `src/utils/`.
- Only `src/libs/light_glue`, `src/libs/pytorch3d`, and `src/libs/gaussian_wrapping/submodules/Depth-Anything-V2`
  are git submodules — `src/libs/sugar` and the other backends are vendored in-tree.
