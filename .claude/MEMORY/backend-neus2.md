# NeuS2 (Experimental)

## Method and Scope

This backend integrates the official [NeuS2](https://github.com/19reborn/NeuS2)
static masked-object implementation, pinned as the `src/libs/neus2` submodule at
`081b8191d049158a8296bfcb15c7d9cf065540df`. Its custom `NeuS2_TCNN` dependency is
pinned at `d3a12b91097ddca15f17b1b6dbacb36fa0c768b2`. Initialize recursively.
It is not the separate NeuS2++ (`neuspp`) branch for unmasked/unbounded scenes.
Dynamic-frame reconstruction is outside this initial adapter.

The [paper and supplement](https://arxiv.org/html/2212.05231) explain the core:

- An SDF defines the surface as its zero level set. NeuS-style SDF-to-opacity
  volume rendering supplies image supervision, instead of treating a NeRF
  density isosurface as geometry.
- A multiresolution hash grid feeds a shallow SDF MLP. Geometry features,
  position, SDF, normal and viewing direction feed the RGB network. The released
  DTU preset has 14 hash levels, a one-hidden-layer 64-unit ReLU SDF MLP and a
  two-hidden-layer 64-unit RGB MLP.
- Backpropagation through SDF normals and the Eikonal loss needs mixed
  second-order derivatives. Their ReLU-specific formulation and custom CUDA
  implementation avoid a large general-purpose autograd graph. This is not
  Neuralangelo's numerical finite-difference recipe.
- Progressive hash-level activation improves coarse-to-fine convergence;
  occupancy-grid-guided ray marching avoids empty regions. The paper describes
  foreground-biased ray sampling using masks.
- The dynamic extension fine-tunes consecutive frames with global motion
  prediction. We explicitly stay on the static first-frame path.

The paper reports approximately five minutes for 15k static iterations on an
RTX 3090, using masked DTU scenes with 49/64 images. That is a benchmark claim,
not a measured runtime for our 276-view skull, another GPU or this adapter.
Build time, data preparation, training, meshing and held-out evaluation must be
reported separately. CPU tests alone do not validate native execution; the
separate GPU smoke result is recorded below.

## Backend-Owned Native Execution

The upstream `scripts/run.py` trains then renders training views, extracts a mesh
more than once, and contains a debugger in its optional test-transform path.
`Neus2Backend` owns scene conversion, native training, snapshot loading, mesh
extraction and held-out rendering in the backend module. Each stage imports
that module in an isolated subprocess and dispatches to a backend method; there
is no separate runner module. It drives the same native `pyngp.Testbed`
implementation without reimplementing the SDF, optimizer or CUDA kernels.
`TestbedMode.Nerf` is the correct upstream mode for image-supervised NeuS2.
`TestbedMode.Sdf` is a different mesh-supervised Instant-NGP task.

The binding's `train(batch_size)` argument is a target number of sampled 3D
points, not an iteration count, image batch or ray count. The backend calls it
once per step and verifies each increment of `training_step`. It avoids
`frame()` and its automatic dynamic-frame transitions.

## Setup

After creating/activating the normal Augenblick Python 3.10 environment, use
the same opt-in installer mechanism as the other backends, on a GPU node:

```bash
git submodule update --init --recursive src/libs/neus2
BACKENDS="neus2" bash scripts/auto_setup.sh
```

A recursive Augenblick clone also initializes the source. NeuS2 is not added
to the default backend build list. For an already configured environment:

```bash
export PYTHON="$(command -v python)"
bash scripts/setup_neus2.sh
```

The helper builds only the headless `pyngp` target, with GUI, Vulkan and OptiX
disabled. It requires a working C/C++ toolchain, CMake, CUDA toolkit and Python
development headers. `TCNN_CUDA_ARCHITECTURES` selects the native architecture
(for example `89` for L40S or `80` for A100); otherwise the GPU wrapper or
`nvidia-smi` supplies it. Do not reuse a single-architecture build on another GPU.
`BUILD_JOBS` controls build parallelism.

The 2023 source vendors pybind11 2.7.1, predating NumPy 2 support. A small CMake
wrapper puts pinned pybind11 2.13.6 headers first for the bindings. These headers
are installed only under the build directory, not into the runtime environment.
The wrapper also makes the fork's `parallel_for_gpu` helper visible in the `ngp`
namespace for modern CUDA name lookup, through a forced include rather than
patching the submodule. Native stages run from the upstream root because its
geometric initialization reads `utils/mlp_weights.txt` relative to the cwd.
The helper does not install upstream's old requirements file, replace torch or
downgrade NumPy. It leaves tracked upstream source untouched and checks an
Eigen/NumPy camera-matrix round trip after importing the built module on a GPU.
The source's CUDA/compiler compatibility still needs native validation on each
target toolchain; a mocked CMake test is not a GPU-build result.

## Verified Smoke (November 5, 2026)

The native headless module built on PACE L40S (sm89), CUDA 12.9.1, GCC 12.3,
Python 3.10 and NumPy 2.2.6. The build-only pybind11 camera-matrix round trip
passed, with tracked upstream source unchanged.

A short skull run used the existing 241-training/35-test split, 30 iterations,
320 x 213 prepared images, 8192 target sampled points, a 64-resolution mesh and
one render sample per pixel. Training and inference-only snapshot reload passed;
the exported mesh was finite and nonempty. Every held-out view had paired
exports and the shared scorer reported 35 masked views, zero skips and finite
PSNR, SSIM, LPIPS and DISTS. This validates execution, not reconstruction quality,
full-resolution memory use or the 15k-step runtime. Initial preparation, build
failures and diagnostics are not a method-speed benchmark.

The smoke exposed an upstream optimizer-snapshot restore defect, diagnosed with
CUDA memcheck: EMA deserialization replaces its allocation while the network
retains old inference pointers. Inference-only snapshots, also used by the stock
static script, avoid that path. Do not enable optimizer-state snapshots without
fixing and validating pointer rebinding upstream.

A subsequent full skull run completed 15k iterations at 1600 x 1066 on the same
241/35 split. The training stage took 204.2 seconds including native loading and
snapshot saving; the complete job took 12 minutes 40 seconds including input
preparation, meshing, held-out rendering, scoring and startup. All 35 masked
held-out views were scored with zero skips, and the 512-resolution mesh was
finite and nonempty. These are one-specimen results, not a controlled speedup
benchmark. Both native runs preceded consolidation into `Neus2Backend`; the
class-owned version passed CPU regression and subprocess-entrypoint checks,
but its additional GPU check could not allocate a node. HPG remains unvalidated.

Defaults are `src/libs/neus2` and its `build/` directory. Override setup with
`NEUS2_ROOT`, `NEUS2_BUILD_DIR`, and `PYTHON`. For reconstruction use the matching
`--upstream` and `--build_dir` options. Keep distinct build directories for
different interpreters, GPU architectures or toolchains.

## Reconstruction

Use an existing Augenblick SfM scene with `images/`, `masks/` and `sparse/0/`.
This backend does not rerun SfM. Start with a fresh smoke output on a GPU:

```bash
augenblick recon neus2 \
  --scene /absolute/path/to/scene \
  --output /absolute/path/to/new-smoke-output \
  --eval --iterations 30 --resolution 320 --batch_size 8192 \
  --mesh_resolution 64 --render_spp 1
```

The short schedule tests execution, not reconstruction quality. Inspect finite
losses, a nonempty mesh and complete paired exports before using the default
static schedule:

```bash
augenblick recon neus2 \
  --scene /absolute/path/to/scene \
  --output /absolute/path/to/new-full-output --eval
```

Defaults: 15k steps, a 1600-pixel maximum input dimension, 262144 target sampled
points per step, 512 mesh grid resolution and eight render samples per pixel.
The normal PACE reconstruction wrapper accepts `BACKEND=neus2` after setup.
HPG and other hardware/toolchains are not yet validated.

## Scene and Evaluation Contract

- Train/test names reuse the shared `split.json` and must be an exact, unique,
  disjoint partition of registered image stems. Separate JSON files are written;
  only `transforms_train.json` is loaded during optimization. Without `--eval`,
  all registered views train and no held-out exports are requested.
- SIMPLE_PINHOLE and PINHOLE models retain per-view focal lengths and principal
  points. Distorted cameras fail explicitly. Pose matrices are COLMAP/OpenCV
  camera-to-world transforms, without OpenGL column flips. `from_na=true`
  cancels those flips in the native loader and prevents the ordinary NGP axis cycle.
- Every registered image needs a nonempty dimension-matched mask. Thresholded
  masks become PNG alpha, with RGB outside the mask set to black. Downsampling
  uses nearest-neighbor alpha and scales intrinsics by actual integer dimensions.
  Prepared views currently need a consistent resolution.
- Reliable sparse points (reprojection error <=2, track length >=3) observed by
  at least one training view define a median-centered enclosing sphere, padded
  by 10%. Held-out-only points are excluded. The sphere maps to the native unit
  cube with `scale=1/(2*radius)` and `offset=0.5-center*scale`. Bounds reuse the
  existing SfM solution and may not enclose unseen anatomy.
- The released DTU network/optimizer and progressive-encoding preset are retained;
  only the static iteration target and disabled global-motion flag are set in
  the generated configuration. The preset's Eikonal weight is 0.01 and explicit
  mask-loss weight is zero. Alpha still supplies foreground sampling/compositing.
  We use fixed black background and disable camera, distortion and exposure
  optimization for an unchanged-camera comparison; these controls are explicit
  adapter choices, not a claim of exact paper-recipe reproduction.
- Snapshots are inference-only, matching the upstream static script. Including
  optimizer state exposes an upstream EMA restore bug: deserialization replaces
  the EMA allocation without rebinding the network's inference pointers. This
  initial CLI does not expose continuation or arbitrary-checkpoint evaluation.
  Preserve outputs; do not
  rerun into an existing prepared scene, snapshot, mesh or render directory.
- Extraction explicitly requests the zero SDF level, not a NeRF density threshold.
  Native PLY export undoes `scale/offset`, so vertices are already in the original
  SfM coordinates. Do not undo normalization a second time. Colors are vertex
  colors, not an authored UV atlas; no component filtering is requested.
- Rendering loads test cameras/images only after optimization, restores the
  frozen snapshot and never calls training. The native camera-selection method
  handles noncentral principal points and unequal focal lengths. Linear native
  output is converted to sRGB; target RGB uses the same prepared masked images.
  Every held-out stem is exported to `test/ours_ITERATION/{renders,gt}`.
- The shared Augenblick evaluator scores these pairs with the metrics present
  on the current main branch. These are neural-field scores, not mesh-render
  scores. Geometry scoring can use the existing shared mesh evaluator separately.

`neus2_manifest.json`, `render_provenance.json` and `stage_timings.json` record
the split/image hashes, source revision, normalization, view counts, training
target, snapshot, render settings and per-stage elapsed times. Train timing
includes native setup and snapshot serialization; it is not pure kernel time.

## License

The official `LICENSE.txt` is NVIDIA's Instant-NGP source license with a
non-commercial research/evaluation restriction. Preserve it and all upstream
and nested dependency notices. Augenblick's license does not override them.