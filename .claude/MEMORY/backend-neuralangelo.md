# Neuralangelo (Experimental)

The registered `neuralangelo` backend consumes an existing Augenblick scene
(`images/`, optional `masks/`, and `sparse/0/`) without rerunning SfM. It trains
a neural signed-distance field, extracts a vertex-colored mesh, and optionally
exports held-out neural renders for the shared appearance evaluator.

This is an experimental integration, validated on one skull specimen on PACE
with an L40S and an A100. HPG and other GPU/software combinations have not been
validated. The unmodified upstream source is a Git submodule at
`src/libs/neuralangelo`, pinned to
`94390b64683c067c620d9e075224ccfe582647d0`. Its original license and notices
remain in the submodule. No weights, datasets or compiled extensions are bundled.

## Pinned Setup

Registration and CLI help
do not import Neuralangelo's optional packages. Training requires Linux, an
NVIDIA GPU, a CUDA toolkit matching the selected PyTorch build, and the
architecture-specific tiny-cuda-nn extension.

`git clone --recursive` initializes Neuralangelo alongside the other submodules.
For an existing checkout, initialize it from the Augenblick repository root:

```bash
git submodule update --init --recursive src/libs/neuralangelo
```

Select Neuralangelo through the normal GPU-specific installer:

```bash
BACKENDS="neuralangelo" bash scripts/auto_setup.sh
```

The installer initializes the submodule, installs the normal environment and
calls the Neuralangelo setup helper with that environment's Python. Include
other backend names in `BACKENDS` to install them as well. Neuralangelo remains
opt-in and is not added to the default backend build list.

If Augenblick and its normal dependencies are already installed, run only the
additional setup instead:

```bash
export PYTHON="$(command -v python)"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
bash scripts/setup_neuralangelo.sh
```

Run either setup route inside a GPU allocation. The helper adds pinned Python dependencies
under constraints from the existing environment, builds tiny-cuda-nn at revision
`3c7931caab7b691c6f30dffe20952ed727d935b9`, then checks a hash-grid CUDA
forward/backward pass and upstream imports. It does not clone upstream, install
PyTorch, or run reconstruction. A conflicting installed dependency fails
explicitly rather than silently downgrading the environment. `--python-only`
installs only the additional Python dependencies; it is not a GPU preflight.

The tested base environment used Python 3.10, PyTorch 2.9.1+cu129,
torchvision 0.24.1+cu129, NumPy 2.2.6 and CUDA toolkit 12.9.1. The setup helper
uses the `PYTHON` interpreter (default `python`) and `NEURALANGELO_ROOT`
(default `src/libs/neuralangelo`). The backend uses that submodule by default;
`--upstream /absolute/path` selects an alternative checkout for reconstruction.
Set `NEURALANGELO_ROOT` to the same alternative path for setup. Build for each GPU architecture before
using that GPU; the L40S extension alone does not support an A100.

## GPU Smoke Test and Reconstruction

Set `SCENE` to an existing SfM scene and choose fresh absolute output paths:

```bash
export PYTHON="$(command -v python)"
export NEURALANGELO_ROOT="$(realpath src/libs/neuralangelo)"
SCENE="$(realpath /path/to/scene)"
SMOKE="$PWD/output/neuralangelo-smoke"
OUTPUT="$PWD/output/neuralangelo-full"
"$PYTHON" -m augenblick.cli.main recon neuralangelo \
  --scene "$SCENE" --output "$SMOKE" \
  --eval --iterations 30 --resolution 320 --rays 64 --batch_size 1 \
  --mesh_resolution 64 --block_resolution 32
```

The smoke run exercises preparation, training, checkpoint reload, extraction,
held-out rendering and scoring. Review finite scores, nonempty geometry and
the exported pairs before committing a long allocation. Thirty-step scores
are execution checks, not quality results. A scene needs enough registered
views and reliable sparse points to define bounds and a nonempty held-out set.

For the full default recipe:

```bash
"$PYTHON" -m augenblick.cli.main recon neuralangelo \
  --scene "$SCENE" --output "$OUTPUT" \
  --eval --iterations 500000 --resolution 1600 --rays 512 --batch_size 2 \
  --mesh_resolution 1024 --block_resolution 64
```

Submit these commands through the site's GPU allocation mechanism; no
Neuralangelo-specific cluster wrapper is included. The full schedule can exceed
a single allocation. `--shortened_schedule` explicitly scales warmup, detail
activation and learning-rate milestones for 1000-500000 iterations; without it,
upstream schedule milestones remain unchanged even for short smoke runs.
Changing batch size or schedule is not a hardware-only speed comparison.

## Data and Evaluation Contract

- Evaluation reuses the saved `split.json`; the normal backend creates it when
  absent. Train/test names must form an exact, unique and disjoint partition.
  Without `--eval`, all views train and upstream validation uses a training view.
- SIMPLE_PINHOLE and PINHOLE intrinsics are retained per frame. Distorted models
  are rejected; undistort first. Prepared dimensions must be consistent across
  views. The longest side is capped without upsampling and intrinsics are scaled
  by the actual integer image dimensions.
- If `masks/` exists, every registered image needs a nonempty dimension-matched
  `<stem>.png` mask. Masks threshold at 127 and composite RGB to black; this is
  not a silhouette loss. Appearance embeddings and the background model are off.
- Sparse points with error at most two and track length at least three, and at
  least one training observation, define a median-centered enclosing sphere
  with 10% padding. This excludes held-out-only points, but reuses the existing
  SfM solution. Bounds are not guaranteed to enclose unseen anatomy; inspect
  normalization before interpreting missing structures.
- Preparation refuses to overwrite an existing prepared scene. The manifest
  records upstream revision, RGB/split hashes, normalization and settings.
  Stage timings include setup and validation, not just optimizer steps.
- `mesh/neuralangelo.ply` is restored to SfM coordinates and has vertex colors,
  not a UV atlas. Extraction does not request largest-component filtering.
- Held-out exports are `test/ours_ITERATION/{renders,gt}/<stem>.png`, scored with
  the current shared NVS evaluator. Neural-field RGB is not a mesh-render score.
  Optional `--reference_mesh PATH` invokes the existing shared geometry scorer;
  this PR does not change its alignment or metric implementation.

The runtime wrapper preserves Slurm CPU affinity, disables W&B logging and
enables legacy checkpoint loading for trusted, locally produced checkpoints.
Never load untrusted checkpoints. Resume restores model/optimizer/scheduler,
but upstream does not record all RNG/data-loader state, so it is not a bitwise
replay of interrupted training.

## Checkpoint Continuation and Evaluation

Keep the original output, generated configuration and upstream checkout. Select
an explicit saved checkpoint and its actual iteration:

```bash
CHECKPOINT="$OUTPUT/logs/epoch_02166_iteration_000260000_checkpoint.pt"
ITERATION=260000
"$PYTHON" -m augenblick.reconstruction.neuralangelo_runner \
  --upstream "$NEURALANGELO_ROOT" --output "$OUTPUT" resume-check \
  --resume-from "$CHECKPOINT" --iteration "$ITERATION"
"$PYTHON" -m augenblick.reconstruction.neuralangelo_runner \
  --upstream "$NEURALANGELO_ROOT" --output "$OUTPUT" train \
  --resume-from "$CHECKPOINT"
```

`resume-check` verifies restored state and one finite optimizer step without
saving; continuation reloads the untouched checkpoint. The original target and
schedule stay unchanged. These commands continue training only, not extraction
or evaluation. Upstream saves just before the scheduler step, so a saved
scheduler counter one below the iteration is valid.

Evaluate a checkpoint in a fresh directory outside the training tree:

```bash
EVALUATION="$PWD/output/neuralangelo-checkpoint-evaluation"
"$PYTHON" -m augenblick.reconstruction.neuralangelo_runner \
  --upstream "$NEURALANGELO_ROOT" --output "$OUTPUT" render \
  --checkpoint "$CHECKPOINT" --iteration "$ITERATION" \
  --evaluation-output "$EVALUATION"
"$PYTHON" -m augenblick.reconstruction.neuralangelo_runner \
  --upstream "$NEURALANGELO_ROOT" --output "$OUTPUT" extract \
  --checkpoint "$CHECKPOINT" --evaluation-output "$EVALUATION" \
  --resolution 1024 --block-resolution 64
"$PYTHON" -m augenblick.eval.nvs \
  --scene "$SCENE" --output "$EVALUATION" --iteration "$ITERATION"
```

The render stage checks the loaded iteration and exact held-out names before
recording provenance. Explicit checkpoint extraction requires that provenance
and the same checkpoint path/size. It does not overwrite training outputs.
The shared NVS command scores the exported pairs using the source scene's split
and masks; it does not rerun training or rendering.
Use the same sequence with the final checkpoint to export after a training-only
continuation; do not lower `max_iter` to evaluate an earlier checkpoint.

## License and Limits

Read NVIDIA's upstream `LICENSE.md` before use. It restricts Neuralangelo to
non-commercial research/evaluation on NVIDIA processors. The submodule retains
that license and its notices; Augenblick's license does not override them.
Multi-specimen quality validation, UV photo-baking and HPG deployment remain
follow-up work rather than claims of this integration.