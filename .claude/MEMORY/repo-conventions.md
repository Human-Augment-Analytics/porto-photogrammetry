# Repo Conventions and Documentation Rules

## Documentation layout rule

`.claude/CLAUDE.md` stays **succinct**: orientation, entry points, hard-won gotchas that change
what you type, and pointers into `.claude/MEMORY/`. Anything long — full flag lists, per-backend
internals, parameter tables, algorithm walk-throughs — lives in a topic file under
`.claude/MEMORY/` and is linked from CLAUDE.md's index table.

When adding documentation: put the detail in the right `MEMORY/` file (or add a new one and
index it in CLAUDE.md); add to CLAUDE.md itself only if it changes how someone invokes or
navigates the repo.

## `.claude/PLANS/`

Implementation specs for large refactors, committed once the work lands. They record the
constraints, the verification steps, and the reasoning behind a change — the things a diff does
not preserve. `MEMORY/` describes the tree as it *is*; `PLANS/` describes how it got that way.

- **Plans are historical, not authoritative.** Where a plan contradicts the code, the code wins.
  A landed plan is not updated to track later drift; the matching `MEMORY/` file is.
- Commit a plan when its work lands, in the same commit or right after. A plan for work that was
  abandoned does not get committed.
- Paths inside a committed plan will go stale as the tree moves. That is expected and is not a
  reason to rewrite it — the file is a record of a decision made at a point in time.

Landed: `augenblick-package-architecture.md` (the `src/augenblick` package and CLI),
`move-backends-to-src-libs.md` (relocating third-party backends under `src/libs/`).

## Code comment and docstring style

Comments are **one line, at most 20 words**, and state only the *purpose* of the code below
them — never a restatement of what that code does. Delete narrating comments
(`# Create output directories` above a `mkdir`); keep the ones carrying non-obvious intent:

```python
# Masks end in .png too, so they must be claimed before the image test.
if '.mask.png' in file_name.lower():
```

Docstrings may run to several lines and keep their `Args:` / `Returns:` sections, but stay
concise and describe purpose rather than walking through behaviour — no worked-example blocks.
`pipeline/preparation/prepare_uf_dataset.py` is the reference for both rules.

## Repository layout

```
pipeline/       Data preparation (preparation/); SfM and reconstruction moved to src/augenblick/
src/augenblick/ The pipeline package and its `augenblick` CLI
baseline/       Meshroom wrapper
scripts/        Per-GPU installers (auto_setup.sh + setup_{l40s,a100,a40,h100,b200,rtx_pro_6000,common}.sh)
constraints/    numpy-generation pip pins (numpy1.txt, numpy2.txt), selected by NUMPY_GENERATION
src/libs/       Backends: vggt/, sugar/, 2dgs/, pgsr/, gaussian_wrapping/, light_glue/, pytorch3d/
src/utils/      Standalone helpers (visual_util.py)
hpg_slurm/      HiPerGator job scripts (see cluster-slurm.md)
pace_slurm/     PACE ICE job scripts, same jobs (see cluster-slurm.md)
tests/          Package unit tests
assets/         README result grids
.claude/        CLAUDE.md orientation, MEMORY/ topic docs, PLANS/ landed refactor specs
data/, output/  Local scene data and run outputs (not for commit)
```

## Legacy / non-canonical code

- **`src/pipeline/`** — the first-generation scripts (`run_vggt.py`, `vggt_to_colmap.py`,
  `run_sugar_pipeline.py`, the NeuS2 / instant-ngp exporters, and their helpers) were deleted
  in `ba6d228`. `src/augenblick/` supersedes them; recover anything still needed from git
  history. No NeuS2 or instant-ngp export ships in the tree today.
- **`src/utils/visual_util.py`** — standalone visualisation helper.
- Untracked markdown and scratch directories in the repo root are personal working material,
  not part of the documented pipeline. Keep them out of commits.

## Naming and ID conventions

- COLMAP IDs (image / camera / point3D) are **1-indexed**; VGGT batch index → COLMAP ID has a
  `+1` offset throughout.
- COLMAP mask files must be named `<image_name>.png` (`foo.jpg.png`, not `foo.png`) — hence the
  `masks_colmap/` symlink dir built by `Scene.link_colmap_masks()`.
- `masks/` convention elsewhere in the repo: binary PNG named after the image *stem*, white =
  foreground.
- Turntable camera grouping keys off `--camera_regex` (default `camera\d+`) and orders frames by
  the **last** integer in the filename.

## Commit / branch conventions

Branches are `<author>/<topic>` (e.g. `syed/spring-26-experiments`, `ihor/turntable`) and land on
`main` via PR merges. Commit subjects are imperative and short ("Add masked colmap, Improved
turntable prior").
