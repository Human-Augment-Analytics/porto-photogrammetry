# Plan: add `neus2` as a reconstruction backend

Implementation spec, written before the work lands. Once it lands this file is historical:
where it and the code disagree, the code wins. The durable description belongs in
[pipeline-reconstruction.md](../MEMORY/pipeline-reconstruction.md) and a new
`MEMORY/backend-neus2.md`.

**Scope:** the code, the job scripts and the benchmark workflow treat `neus2` like any other
backend on both clusters. Building and verifying it happens on branch `syed/neus2-inclusion` in its own git worktree,
against a **clone** of the HPG env, `augenblick_rtx_pro_6000_neus2` (step 0). The production env `augenblick_rtx_pro_6000` is not
modified until verification passes and the user approves promotion (step 10). Other envs pick
`neus2` up from `setup_common.sh` (`BACKENDS` default) on their next rebuild. B200 waits on its
env drift being fixed (see environment-and-gpu.md).

Researched 2026-11-25 against the tree at `8ec1aa5` and upstream NeuS2 `main` at
**`081b8191d049158a8296bfcb15c7d9cf065540df`** (2024-03-22). Its one submodule, `neus2_tcnn`, is at
**`d3a12b91097ddca15f17b1b6dbacb36fa0c768b2`** (2023-09-01), which pins CUTLASS **2.10.0**
(`4db6a61`) and fmt (`e150ea0`). Upstream line references are to those commits. Facts marked
*checked* were run against the RTX env or the results tree; everything else was read in the
cloned source. **Nothing here has been compiled or run on a GPU yet.**

Revised 2026-11-26: a branch in its own worktree, an isolated clone of the env, all NeuS2 code
in the vendored tree (one script per stage), four commits with two held for the user, and
`UF_Mammals_19126_cranium` as the verification specimen.

## Status

| Step | State |
|---|---|
| 0 — branch `syed/neus2-inclusion` + clone the test env | Done; runtime proof passed in job 52898251 |
| 1 — vendor + trim into `src/libs/neus2/` | Done, commit `9680ede` |
| 2 — local patches | Done, commit `cb42c0a` (2a–2e + build fix 1) |
| 3 — build `pyngp` into the test env | Done, job 52898251; checks 0 and 3 passed |
| 4 — stage scripts in `src/libs/neus2/` | Done, commit `cb42c0a`; synthetic CPU + GPU checks |
| 5 — adapter `reconstruction/neus2.py` | Done, commit `cb42c0a`; check 1 passed |
| 6 — benchmark workflow + bench tables | Done, commit `fc65fdb` |
| 7 — tests | Done, commit `fc65fdb`; 270 passed + 33 NeuS2 tests |
| 8 — SLURM + setup scripts | Done, commit `fc65fdb` |
| 9 — docs | Done, commit 4 |
| Verification (in the test env) | Done: checks 0–11 pass (results below) |
| Commits 1–2 (steps 1; 2–5) | Done: `9680ede`, `cb42c0a` |
| Commits 3–4 (steps 6–8; 9) | Done after the user's go-ahead: `fc65fdb` + docs |
| 10 — promotion to the production env (needs the user's go-ahead) | Not started |

## Implementation notes (2026-11-26)

What changed against the plan while implementing, newest last. Each was verified as stated.

- **Clone prefix check, substring trap.** `$PROD` is a prefix of `$ENV`
  (`..._6000` vs `..._6000_neus2`), so `grep "$PROD"` matches the clone's own, correct paths:
  it reported 75 files, all false positives. The check must exclude the clone's suffix:
  `grep -rlIP 'augenblick_rtx_pro_6000(?!_neus2)'`, which finds **0** files. `conda create --clone`
  did rewrite every pip-installed shebang. A `sed` run on the false positives was reverted on
  exactly those 75 files, verified by a re-grep. RUNPATHs naming either env: 0.
- **Step 1 trim kept three GUI headers.** `camera_path.h:21-22` includes `imgui/imgui.h` and
  `imguizmo/ImGuizmo.h` with the GUI off (every ImGui call is under `NGP_GUI`). `imgui.h`,
  `imconfig.h`, imgui's licence and `ImGuizmo.h` are vendored as headers only, folded into
  commit 1 (amended before any push), so upstream sources stay unpatched.
- **Patch 2e, `PYBIND11_FINDPYTHON ON`.** Without it pybind11 2.13 used its legacy search and
  found `/usr/bin/python3` 3.9 and `/usr/lib64/libpython3.9.so`, despite NeuS2's own
  `find_package(Python)` finding the env's 3.10.
- **`CMAKE_POLICY_VERSION_MINIMUM` is not needed** with `cmake/3.30.5` (CMake reports it unused).
- **First compile (CPU job 52883192, CUDA 13.0.88, gcc 12.2, sm_120):** `tiny-cuda-nn`,
  CUTLASS 2.10 included, **built** under C++17 with only `double4`-style deprecation warnings.
  The only errors were the imgui includes above.
- **Mesh and render stages load the snapshot alone.** The snapshot carries the dataset metadata
  (`testbed.cu:3162`, per-frame focal/principal point restored at `json_binding.h:177-183`,
  `from_na`, scale/offset), so `extract_mesh.py` and `render.py` never reload images.
- **Faces are used as returned.** Under `from_na`, `save_mesh` gets `invert_normals = from_na`
  (`testbed.cu:305`) and writes faces unreversed, which is the order
  `compute_marching_cubes_mesh` returns.
- **Held-out render metadata is `render_protocol.json`**, the name GeoSVR and `mesh_render` use.
- **`prepare.py` stages only training views**; held-out ground truth comes from the photographs.
- **Build fix 1, `tcnn::parallel_for_gpu` (second compile, CPU job 52891004).** Five calls in
  `nerf_network.h` and one in `takikawa_encoding.cuh` named it unqualified inside `ngp`-namespace
  templates. CUDA 13's front end applies two-phase lookup and rejects them (4 errors in
  `src/testbed.cu`); each is now `tcnn::`-qualified with a marker. The unqualified calls in
  non-template `.cu` code compiled and are untouched.
- **Selection manifest (found by check 8).** The workflow's `eval.nvs --require-complete` needs
  `native_selections/render_*.json`, written by every other backend through
  `augenblick.eval.selection.record_selection`. `render.py` now records the partition it rendered
  (`scene_io.record_native_selection`), folded into commit 2 (now `cb42c0a`).
- **Tests live in new files.** `tests/test_recon_argv.py` does not collect on this branch, so the
  adapter tests are `tests/test_neus2_backend.py`, with a synthetic ring scene in a new
  `tests/conftest.py` instead of moving `test_meshroom_from_sfm.build_scene`. `test_registry`
  also gained `meshroom`, which it was missing.
- **The 30k run used its own `RESULT_ROOT`** (`$V/n30000`, `split.json` copied) so it ran in
  parallel with the 15k run.
- **Pre-existing test failure, not ours:** `tests/test_recon_argv.py:546` repeats `extract=`
  (a `SyntaxError` from merge `8ec1aa5`), so that file does not collect on either branch.

## Goal

`augenblick recon neus2 --scene <sfm> --output <out>` trains NeuS2 on any of the five SfM
methods' COLMAP scenes and writes a marching-cubes mesh **in the input COLMAP frame**. With
`--eval` it also writes held-out renders that `augenblick.eval.nvs` scores like the other
backends' renders, at the shared evaluation raster.

Like the other backends, the adapter runs **one process per stage**, each a script in the
vendored tree (`prepare.py`, `train.py`, `extract_mesh.py`, `render.py`), with that tree as
cwd. All NeuS2-specific code lives in `src/libs/neus2/`. The first-party side is one module
exposing only `Neus2Backend` and `Neus2Config` (step 5): it builds the argv, validates, exports
cameras and cleans up.

Non-goals: no GUI/`testbed` binary, no dynamic-scene (`run_dynamic.py`) support, no NeuS2++
(`neuspp` branch, unmasked/unbounded scenes), no DTU evaluation scripts. The cloned env is a
temporary test bed, not a new permanent env.

## What NeuS2 is

NeuS2 (Wang et al., ICCV 2023, `neus2`) is NeuS rebuilt inside NVIDIA's instant-ngp. The surface
is the zero level set of an SDF MLP fed by a multi-resolution hash grid. It is volume-rendered
with NeuS's unbiased SDF-to-density weights and trained with an Eikonal loss, so the geometry is
an implicit field, not primitives. Everything is CUDA: a fork of tiny-cuda-nn adds second-order
backprop through the fully fused MLP, and a progressive schedule turns on finer hash levels.
Python reaches it only through one pybind11 module, `pyngp`. The mesh comes from marching cubes
on the SDF.

It is the first **implicit-SDF** backend in the comparison, and the first that **ignores the SfM
point cloud entirely**: only the poses, intrinsics and masks reach it. Our `prepare.py` uses the
points only to place a bounding sphere. Hypothesis to test: geometry is insensitive to SfM point
density, and as sensitive to pose error as any photometric method. Expect it to be by far the
fastest backend; the paper reports ~5 min on DTU.

| | GeoSVR (closest existing in structure) | NeuS2 |
|---|---|---|
| Representation | sparse-voxel octree | hash-grid SDF + MLP (implicit) |
| SfM points used for | scene bound | **nothing** (ours: bounding sphere only) |
| Input format | COLMAP `sparse/0` | **NeRF `transforms.json` + RGBA images**, scene normalised into the unit sphere |
| Masks | `masks/` (patched in) | **image alpha channel** |
| Default schedule | 20k iterations | 15k steps (`configs/nerf/dtu.json`) |
| Mesh | TSDF of rendered depths | marching cubes on the SDF |
| Stage scripts | `train.py`, `render.py`, `mesh_extract/tsdf_mesh*.py` (upstream, patched) | `prepare.py`, `train.py`, `extract_mesh.py`, `render.py` (**added**; upstream's `scripts/run.py` is not vendored) |
| Build | `pip install` CUDA extension | **CMake** project → `pyngp*.so`; no torch linkage |
| Licence | NVIDIA SVRaster licence, non-commercial | **NVIDIA Source Code License-NC** (instant-ngp), §3.3: research or evaluation only |

Licence, carry into the docs: `LICENSE.txt` §3.3 is the same research-only footing as GeoSVR and
DVLT, so there is nothing new to clear. The vendored dependencies keep their own permissive
licences (tiny-cuda-nn BSD-3, CUTLASS BSD-3, Eigen MPL-2, pybind11 BSD).

## Upstream facts that shape the design

1. **The scene must be normalised into the unit sphere by the caller.** NeuS2 has no scene
   normaliser. Its DTU converter (`tools/data_format_from_neus.py:59-70, 139-150`) applies NeuS's
   `scale_mat`, which puts the object in the unit sphere, then writes `scale: 0.5`,
   `offset: [0.5, 0.5, 0.5]`, `aabb_scale: 1` so that sphere fills NGP's `[0,1]³` cube. Without
   a `scale` key the loader falls back to `NERF_SCALE` (`nerf_loader.cu:305-306`), tuned for
   Blender scenes, not ours.
   - Our COLMAP gauge is arbitrary (*checked* on `UF_Herp_87980_cranium`, figures below, and on
     the verification specimen `UF_Mammals_19126_cranium`, whose full y extent is 31.5 units). The 1–99 % point
     extent is ~0.75 units, and camera centres sit 3.0–4.4 units away. But the **full** point
     extent is 3.1–3.8 units, from outlier points. A naive bounding box of all points would
     contain the cameras.
   - **Decision:** `prepare.py` computes a robust centre and radius (step 4b) and checks them
     against the masks. It writes the normalised poses, and keeps the transform so the mesh is
     mapped back.
2. **`from_na` makes `transform_matrix` a plain OpenCV camera-to-world.**
   - `nerf_matrix_to_ngp` (`nerf_loader.h:112-133`) flips columns 1 and 2 (OpenGL → NGP), then
     flips them back when `from_na` is set. It also skips the xyz→yzx axis cycle.
   - So under `from_na`, COLMAP's `[R_wc | C]` goes in unchanged, apart from normalising the
     translation, and the world axes are not permuted.
   - **Trap:** `from_na` is tested by key presence only (`nerf_loader.cu:346`), so even
     `"from_na": false` enables it.
3. **Intrinsics must be per frame.** The top-level `cx`/`cy` are parsed (`:426-432`), but the
   line copying them into each frame is commented out (`:708`), and the default is 0.5
   (`nerf_loader.h:44`). The `intrinsic_matrix` branch (`:680-685`) normalises by the
   **top-level** `w`/`h`. Per-frame `cx`/`cy` divide by the **frame's** `w`/`h` (`:692-698`).
   Per-frame `fl_x`/`fl_y` are read only if no top-level `fl_*` exists (`:651-666`).
   **Decision:** every frame carries `fl_x, fl_y, cx, cy, w, h`, at the training raster; there
   are no top-level intrinsics.
4. **Masks travel as the image alpha channel.** The loader reads RGBA (`stbi_load(..., 4)`,
   `:551`). Training composites the target onto a **random background colour** per ray
   (`testbed_nerf.cu:1633-1667`; `random_bg_color = true`, `testbed.h:611`). That forces the
   rendered transmittance to match `1 − alpha`, so it acts as an alpha loss. The explicit
   mask BCE (`mask_loss_weight`, `testbed_nerf.cu:1691-1706`) is 0 in both shipped configs.
   - The `<file>.alpha.<ext>` sidecar (`nerf_loader.cu:554-568`) must share the image's
     extension, so it would be a lossy JPEG here; `dynamic_mask_*.png` (`:571-587`) paints
     hot pink.
   - **Decision:** stage RGBA PNGs, alpha = mask. Keep `mask_loss_weight 0.0`: alpha
     compositing is already the alpha supervision that 2DGS/GW get from their mask channel.
5. **Images stay full-size on the GPU.** Byte RGBA is kept at load resolution
   (`set_training_image`, `:758-817`), and NeuS2 never downscales. At 6240×4160, 432 views
   would need ~45 GB of VRAM. At the shared 1600×1066 raster it is ~2.9 GB.
   - The paper scorer also requires native renders to equal that raster
     (`docs/nvs-protocol.md`, `--require-native-raster`).
   - **Decision:** stage images at `mesh_render.image_size(camera, -r)`, the raster function
     the scorer itself uses, with intrinsics scaled by `mesh_render.calibration`.
6. **The mesh comes out in the normalised frame, not ours.**
   - `save_mesh` writes `(v − offset)/scale` as **5-decimal ASCII**
     (`marching_cubes.cu:905`).
   - `compute_marching_cubes_mesh` (`python_api.cu:99-121`) returns `V, N, C, F` as numpy in
     **NGP space**, with no un-scaling.
   - `thresh` defaults to `m_mesh.thresh = 0.0`, the SDF zero level (`testbed.h:404`). Upstream
     changed it from instant-ngp's 2.5.
   - Faces come out in the file's order under `from_na` (`marching_cubes.cu:912`).
   - **Decision:** `extract_mesh.py` calls `compute_marching_cubes_mesh` and maps
     `x_colmap = c + r·(2·v_ngp − 1)` itself, losing no precision. The benchmark's metric
     lineage assumes every mesh is in the SfM frame.
7. **Upstream's `scripts/run.py` is not usable as our entry point.**
   - It writes to `output/<name>` **relative to cwd** (`run.py:78`).
   - Its `--test_transforms` path stops in **`pdb.set_trace()`** (`:273-274`).
   - It renders a single training view (`:248`), and only via `render_utils.py`, which
     star-imports `pytorch3d_utils` (`render_utils.py:424`). That module imports `termcolor`,
     which is **missing** in the RTX env (*checked*), and `from this import d` (`:98`).
   - It imports torch's tensorboard writer (`:29`). That would load a cu130 torch into the same
     process as `pyngp`'s own CUDA runtime.
   - It writes the mesh as `.obj` in the normalised frame (`:203, 244`).

   **Decision:** do not vendor or patch `scripts/`. Add our own stage scripts at the vendored
   tree's root, beside `configs/` and `utils/`, the way `train.py` and `render.py` sit at the
   root of `src/libs/geosvr` and `src/libs/2dgs`. They call `pyngp` directly and reuse
   upstream's training loop shape (`scripts/run.py:207-216`) and evaluation render settings
   (`render_utils.py:149-152, 272-283`), which they copy rather than import.
8. **Two cwd dependencies inside the C++.**
   - Geometric SDF init reads `utils/mlp_weights*.txt` **relative to cwd**, and `exit(1)`s if
     it is missing (`nerf_network.h:775-793`). It runs from inside the binary, so Python cannot
     catch it.
   - `load_training_data` creates `<data_path>/output/` and opens a log there
     (`testbed.cu:82-98`). For a `.json` path that `mkdir` fails silently and the log handle is
     NULL; every later use is guarded (`:1940-1946, 2009-2015`).
   - **Decision:** the adapter keeps the default `use_cwd = True`, so every stage script runs
     with `cwd = src/libs/neus2`. The shared `open_testbed()` helper (step 4) also checks
     `utils/mlp_weights.txt` exists before building a `Testbed`, with a message that names the
     cwd.
9. **Passing a directory loads every JSON in it as a dynamic-scene frame**
   (`testbed_nerf.cu:2959-2977`). `transforms_test.json` beside the train file would become
   "frame 2". **Decision:** always pass the train JSON's file path.
10. **The step limit is static-safe.** `first_frame_max_training_step` (`testbed.cu:2108`) only
    triggers `training_network_next_frame`, which returns immediately on the last frame
    (`:1991-1998`). So `n_steps` governs. `train.py` still sets it to `n_steps` in the
    effective config, so nothing keyed on it diverges.
11. **Held-out cameras can be set without loading them as training data.**
    - The bindings expose `set_nerf_camera_matrix` (`testbed.cu:228-230`, the same
      `from_na` conversion), `fov_axis`, `fov_xy` and `screen_center` (`python_api.cu:425-439`).
    - `set_camera_to_training_view` (`testbed.cu:254-260`) sets
      `relative_focal = f / res[fov_axis]` and `screen_center = 1 − principal_point`.
    - **Decision:** `render.py` reproduces exactly that for held-out views. Check 6 proves it on
      a training view rendered both ways.
12. **Build system facts** (all read, none compiled):
    - Both `CMakeLists.txt` pin **C++14** (NeuS2 `:58, 65`; tcnn `:63, 71`). CUDA 13's
      CCCL 3 requires C++17. CUDA 12.8/12.9 accept C++14 with deprecation warnings.
    - tcnn picks the GPU arch through `include(FindCUDA/select_compute_arch)` (`:79-84`), unless
      `TCNN_CUDA_ARCHITECTURES` is set (`:90-93`). That module still ships in the env's CMake
      4.4.3 and the `cmake/3.30.5` module (*checked*).
    - Arch ≥ 80 compiles CUTLASS **`Sm80`** kernels (`cutlass_matmul.h:75-79`), and
      `cutlass_mlp.cu` is always built (`src/CMakeLists.txt`). So CUTLASS 2.10 headers must
      compile for sm_120. `static_assert(__CUDA_ARCH__ >= MIN_GPU_ARCH*10)` holds at 120.
    - tcnn hard-codes `CUDA_TOOLKIT_ROOT_DIR /opt/cuda/...` (`:152-158`), but only for an
      unused `find_library(cublasLt)`. It links `cuda cublas` by name.
    - Existence checks `FATAL_ERROR` without `dependencies/glfw/CMakeLists.txt` (NeuS2 `:39-45`)
      or `dependencies/cutlass/CMakeLists.txt` (tcnn `:46-52`). Both checks are unconditional,
      even with the GUI off.
    - `pyngp` is `add_library(pyngp SHARED ...)` over **static** `ngp` + `tiny-cuda-nn`
      (`:322-365`), so the `.so` is self-contained apart from CUDA libs. fmt is never
      included (*checked*: no `fmt/` include anywhere).
13. **The vendored pybind11 is 2.7** (`detail/common.h:12-13`). It reads `PyArray_Descr` through
    a fixed struct with `int elsize` (`numpy.h:59`) and imports `numpy.core.multiarray`
    (`:228`). numpy 2 changed that struct, and the RTX env is numpy **2.2.6** (*checked*).
    pybind11 < 2.12 can then compute wrong itemsizes/strides for the arrays `render()` and
    `compute_marching_cubes_mesh()` return, silently. **Decision:** replace it with upstream
    pybind11 **v2.13.6** (2.12+ supports numpy 2). Patch 2d.
14. **Memory/time scale.** A step is a fixed ray batch (`hyperparams.batch_size`, default
    `1<<18`, read at `testbed.cu:2109-2110`), independent of view count.
    - 15k steps × 262k rays ≈ 3.9 G rays: ~42 visits per pixel on DTU (49 × 1600×1200), but
      only **~5.6** on our 432 × 1600×1066.
    - Upstream's step count may under-train here. Check 9 sweeps it.

## Environment

### The test env and worktree (step 0)

```bash
PROD=/blue/arthur.porto/srizvi63.gatech/conda/augenblick_rtx_pro_6000
ENV=/blue/arthur.porto/srizvi63.gatech/conda/augenblick_rtx_pro_6000_neus2
MAIN=/home/srizvi63.gatech/porto-photogrammetry          # production's checkout, untouched
WT=/home/srizvi63.gatech/porto-photogrammetry-neus2      # worktree on syed/neus2-inclusion
```

Every install step and every verification run uses `$ENV` and `$WT`. `$PROD` and `$MAIN` are
read-only until step 10 and the merge.

- Production is 7.7 GB, with only 56 conda packages; most of it is pip-installed
  (*checked*). `/blue/arthur.porto` has 3.3 TB free (*checked*).
- **Prefix trap.** 100 files in `$PROD/bin` hard-code `$PROD` (*checked*), including the
  `augenblick` console script's shebang (`#!$PROD/bin/python3.10`) that every SLURM job calls.
  If the clone kept that shebang, a "test env" job would silently run production's Python.
  `conda create --clone` rewrites the prefix in text files it copies, pip-installed ones
  included, but step 0 checks this rather than assuming it.
- **Shared-code trap.** All seven editable installs point at `$MAIN` (`src/`,
  `src/libs/dvlt/src`, and the finder modules for vggt, lightglue, pytorch3d, sam3, tetranerf;
  *checked*). A plain clone therefore runs production's working tree, and anything committed
  there reaches production jobs at once. Step 0 breaks this link for the code NeuS2 changes.

#### How the clone is isolated

The clone has two leaks back into production. Both are closed in step 0, and check 11 proves it.

1. **Code: a git worktree, not a branch switch in `$MAIN`.** Switching `$MAIN` to the new branch
   would change what every production job runs. Instead, step 0a checks the branch out at
   `$WT`. Step 0c then re-installs only `augenblick` into `$ENV` from `$WT`, with the repo's own
   `pip install -e . --no-deps --no-build-isolation`. That rewrites the clone's
   `__editable__.augenblick-0.1.0.pth` and its `augenblick` console script, and nothing in
   `$PROD`.
   - `REPO_ROOT`, and so `LIBS_DIR` and every vendored backend, follow `augenblick`'s own path
     (`reconstruction/base.py:18-19`). So `$ENV` runs `$WT/src/libs/neus2`, and production never
     sees `neus2` until the branch merges.
   - The six other editable installs stay pointed at `$MAIN`. vggt, sam3, dvlt, lightglue,
     pytorch3d and tetranerf have in-tree, gitignored builds (`*.so`) and git submodules that a
     fresh worktree does not have, and NeuS2 touches none of them. A diff guard keeps it so:
     `git diff --name-only syed/traditional-pipelines... -- src/libs/{vggt,sam3,dvlt,light_glue,pytorch3d,gaussian_wrapping}`
     must be empty at every commit.
   - SLURM jobs take `REPO_ROOT` from `SLURM_SUBMIT_DIR` (`recon.sbatch:30`,
     `scene_common.sh:8-9`), so verification jobs are submitted **from `$WT`**. The banner's
     `repo :` line (`scene_common.sh:93`) records the path and commit that ran.
   - The worktree is ~176 MB of tracked files; `/home` has 15 GB free (*checked*). It gets its
     own `output/results` symlink to `/blue/arthur.porto/srizvi63.gatech/results/`, as `$MAIN`
     has.
2. **Libraries: rewrite every `RUNPATH` that names `$PROD`.** Compiled extensions keep their
   linker search paths through a clone, so a `.so` in `$ENV` can still load libraries from
   `$PROD`. Step 0b lists them and rewrites each with `patchelf --set-rpath`, replacing the
   `$PROD` prefix with `$ENV`, never changing anything else in the path. `patchelf` is on the
   conda module's path (`/apps/conda/25.7.0/bin/patchelf`, *checked*). Binaries that hard-code
   `$PROD` outside `RUNPATH` cannot be rewritten safely, because the prefixes differ in length.
   The runtime check below catches any that matter.

**The proof is at runtime, not in file listings.** In a GPU job under `$ENV`, import the
`rebuild_env.sbatch:57-62` module list plus `augenblick`, `pyngp` (after step 3) and the
NeuS2 stage scripts' imports, then scan `/proc/self/maps`. No mapped file may live under `$PROD`,
and `augenblick.__file__` must be under `$WT`. That catches every route a file listing misses:
`RUNPATH`, `dlopen` with absolute paths, and stray `.pth` entries.

### Job resources

Every GPU job in this plan (the step 0c and 3 jobs, checks 0, 3 and 6, and every verification
run through `recon.sbatch`) runs with:

```bash
GPU_JOB="--account=arthur.porto-phenomics --partition=hpg-rtx6000 --gpus=1 --cpus-per-task=16 --mem=96gb"
```

- **GPU:** RTX PRO 6000 (`hpg-rtx6000`, `gpu:rtx_pro_6000:8` per node, *checked*).
- **Account:** `arthur.porto-phenomics`, an association this user has (*checked* with
  `sacctmgr`), not the `arthur.porto` default in the job scripts.
- **CPUs and RAM:** 16 and 96 GB, against `recon.sbatch`'s 8 and 64 GB.
- Command-line flags override `#SBATCH` lines, so `recon.sbatch` itself is **not** edited: its
  defaults stay what the benchmark uses. One-off jobs are `sbatch $GPU_JOB --time=... --wrap`
  or small scripts under `$SNAP/jobs/`, never committed.
- The build uses all 16 CPUs (`-j "$SLURM_CPUS_PER_TASK"`).
- The step 0b clone is CPU-only: the same account, no GPU flags, on the default partition.

### Python deps (the stage scripts only)

Only `train.py`, `extract_mesh.py` and `render.py` import `pyngp`. The scripts also use
`numpy`, `PIL`, `open3d` (mesh post-processing) and `pycolmap` (the SfM points), all present
(*checked*: numpy 2.2.6, Pillow 12.3, open3d 0.19.0, pycolmap 4.1.1). **No new pip packages.**
Upstream's `requirements.txt` (numpy~=1.21, scipy~=1.7, ...) is not installed.

No stage script may **import torch**: two CUDA runtimes in one process is a risk we do not
need. They import `augenblick.eval.cameras`, `augenblick.eval.mesh_render`,
`augenblick.eval.split` and `augenblick.core.errors`, so staging and rendering use the scorer's
own raster and split code, as GeoSVR's `render.py:42` already does. None of the four pulls in
torch, open3d or pycolmap at import (*checked*).

### Build toolchain

- `cmake/3.30.5` module or the env's CMake 4.4.3. Either needs
  `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`, as the tetra build already passes, for the old vendored
  sub-projects.
- `gcc/12.2.0`, and the `cuda/13.0.2` module that `hpg_slurm/common.sh:15` loads for rtx6000.
- **Primary:** build with `cuda/13.0.2`, so the runtime `libcublas.so.13` is the one every job
  already has on `LD_LIBRARY_PATH`.
- **Fallback:** if CUTLASS 2.10 cannot be made to compile under CUDA 13, use `cuda/12.9.1`
  (sm_120 needs ≥ 12.8). Then `libcublas.so.12` must resolve without that module: pass
  `-DCMAKE_BUILD_RPATH=$CUDA_HOME/lib64` and check with `ldd` (check 0).
- `pyngp` does not link torch, so the torch/cu130 drift recorded for GeoSVR does not constrain
  it.

## Implementation

Steps in dependency order, all in `$WT`. Steps 1, 2, 4–7 and 9 run on a login node. Step 0b is
I/O-bound and runs as a CPU job. Step 3 needs `nvcc`, so run it as a SLURM job. Step 8's
`common.sh` change must be in `$WT`'s working tree, committed or not, before any verification
job is submitted.

### Commits

Four commits on `syed/neus2-inclusion`, in this order. Each follows the repo's commit
conventions (repo-conventions.md) and ends with the attribution trailer.

| # | Steps | Content | When |
|---|---|---|---|
| 1 | 1 | the vendored, trimmed upstream tree, **unpatched** and with upstream's pybind11 2.7 | after step 1's gitignore check |
| 2 | 2, 3, 4, 5 | the local patches (pybind11 swap included), the six added scripts/modules, the adapter and its `__init__.py` line | after checks 0 and 3 pass (the build and the bindings) and `pytest tests/` still passes |
| 3 | 6, 7, 8 | workflow + bench tables, tests, SLURM + setup scripts | **only on the user's go-ahead** |
| 4 | 9 | the docs and MEMORY edits, plus this plan (repo-conventions.md: a plan is committed when its work lands) | **only on the user's go-ahead** |

- Commit 1 holds no patches, so `git diff <commit 1> <commit 2> -- src/libs/neus2` is exactly
  the set of local divergences from upstream. That is the record a future resync re-applies.
- Step 3 changes no tracked file: the build goes to `$ENV/var/augenblick/neus2-build` and the
  `.so` to `$ENV`'s site-packages. Its build-fix patches are step 2 edits, which is why steps 2
  and 3 share commit 2.
- **Stop after commit 2** and report to the user: what was built and patched, checks 0, 1 and 3,
  and anything that changed against this plan. The user may ask for changes before steps 6–9.
- Steps 6–9 start only after the go-ahead, so the user's changes land before them. The full
  runs (checks 4–10) go through `recon.sbatch` and need step 8, so they also wait. Nothing is
  pushed before the go-ahead.
- The step-10 edits held for promotion (`BACKENDS` default, `pyngp` in `rebuild_env.sbatch`) are
  a fifth commit, after promotion.

### Step 0 — branch `syed/neus2-inclusion`, worktree, and the test env

**0a — branch, in its own worktree.** Every commit lands on a new branch, `syed/neus2-inclusion`,
never on `syed/traditional-pipelines`. `$MAIN` stays on `syed/traditional-pipelines`.

```bash
git -C $MAIN worktree add -b syed/neus2-inclusion $WT syed/traditional-pipelines
ln -s /blue/arthur.porto/srizvi63.gatech/results $WT/output/results   # after mkdir $WT/output
```

- Branch from `syed/traditional-pipelines` (`8ec1aa5`, the tree this plan was verified against),
  **not `main`**. `main` (`11d0f96`, 120 commits behind) lacks GeoSVR, MILo, `workflow/Snakefile`,
  `bench/tree.py` and `hpg_slurm/recon.sbatch`, all of which this plan edits or copies from
  (*checked*).
- `$MAIN`'s untracked files (other plans, `hpg_slurm/` helpers, `scripts/` prep scripts) do not
  appear in `$WT`. Copy this plan into `$WT/.claude/PLANS/`; it stays untracked until commit 4
  (see "Commits").
- Its PR targets `syed/traditional-pipelines` until that branch merges into `main`, then is
  retargeted to `main`.
- After the merge, `git worktree remove $WT`.

**0b — clone the test env.**

```bash
SNAP=$WT/output/results/neurips_verify/neus2/_install && mkdir -p $SNAP
$PROD/bin/python -m pip freeze > $SNAP/prod_freeze_before.txt
touch $SNAP/prod_marker
conda create -p $ENV --clone $PROD --offline --yes
```

Run it as a CPU-only job (no GPU partition): it copies ~7 GB.

Checks, all of which must pass before 0c:

- `head -1 $ENV/bin/augenblick` names `$ENV/bin/python3.10`.
- `grep -rlI "$PROD" $ENV/bin $ENV/lib/python3.10/site-packages` lists **no** text file.
  If any does, fix them with `sed -i "s#$PROD#$ENV#g"` on exactly the listed files. Text files
  only; never binaries, because the prefixes differ in length.
- List every ELF file under `$ENV` whose `RUNPATH`/`RPATH` (`readelf -d`) names `$PROD`, into
  `$SNAP/runpath_before.txt`. Rewrite each with `patchelf --set-rpath` (see "How the clone is
  isolated") and list again into `$SNAP/runpath_after.txt`, which must be empty.
- `$ENV/bin/python -m pip freeze` equals `prod_freeze_before.txt`.

**0c — point `augenblick` at the worktree.**

```bash
cd $WT && $ENV/bin/python -m pip install -e . --no-deps --no-build-isolation
```

- `$ENV`'s `__editable__.augenblick-0.1.0.pth` now names `$WT/src`; `$PROD`'s still names
  `$MAIN/src`.
- `pip freeze` differs from `prod_freeze_before.txt` in the `augenblick` line only.
- On a GPU node, the runtime proof above passes (no `pyngp` yet). Run it as a one-off job, not
  via `rebuild_env.sbatch`. The same job runs the numpy↔torch interop check
  (`rebuild_env.sbatch:51-55`).

### Step 1 — vendor + trim into `src/libs/neus2/`

Copy `081b819` **in-tree, not as a submodule**, like every patched backend. Flatten
`dependencies/neus2_tcnn` (`d3a12b9`) and its CUTLASS into the tree: they are submodules
upstream, but here they are patched (step 2).

- **Keep:** `CMakeLists.txt`, `cmake/`, `include/`, `src/`, `configs/nerf/`, `utils/`
  (fact 8), `LICENSE.txt`, `README.md`, `DATA_CONVENTION.md`, and these `dependencies/`:
  - `args`, `filesystem`, `nanovdb`, `pybind11_json`, `stb_image`, `tinyexr`, `tinylogger`,
    `tinyobjloader`, `cuda-cmake-github-actions`;
  - `eigen/`, trimmed to `Eigen/`, `unsupported/` and its `COPYING.*` (~12 MB of 20 MB);
  - `neus2_tcnn/`: `CMakeLists.txt`, `include/`, `src/`, `bindings/`, `LICENSE.txt`, and its
    `dependencies/{json,pcg32,pybind11_json,stbi}`;
  - `neus2_tcnn/dependencies/cutlass/`: `include/`, `tools/util/include/`, `LICENSE.txt` only
    (~8 MB). tcnn only adds these as include dirs (`:162-163`).
- **Keep `dependencies/pybind11` at upstream's 2.7 in this step.** Its replacement with v2.13.6
  (fact 13) is patch 2d, so it lands in commit 2 with the other divergences.
- **Drop:**
  - GUI/Vulkan, off by flag: `dlss` (187 MB), `glfw`, `imgui`, `imguizmo`, `gl3w`;
  - fmt (unused) and CUTLASS's sources, tests and examples;
  - `scripts/` (fact 7), `tools/`, `docs/`, `AnacondaBat.bat`, `requirements.txt`.
- **Add** (in step 4, so in commit 2, not commit 1) exactly six first-party files at the tree
  root: the stage scripts
  `prepare.py`, `train.py`, `extract_mesh.py`, `render.py`, and the shared modules `scene_io.py`
  and `ngp_session.py`. Each opens with a `LOCAL PATCH (augenblick)` header saying the file is
  ours, not upstream's, so the same `grep -rn` that finds the C++ patches finds them. None of
  the names exists upstream at the root, so a resync cannot overwrite them.
- **Gitignore hazard:** the repo ignores `build/`, `lib/`, `lib64/`, `data`, `output/` and
  `*.so`. No kept path matches (*checked*: the only `data` dirs are in dropped CUTLASS tests and
  Eigen benches). After `git add`, confirm with
  `git status --ignored src/libs/neus2` that nothing kept is ignored.
- Provenance goes in `MEMORY/backend-neus2.md` (step 9), not the vendored tree: URLs, the three
  commits, dropped paths and patch sites.

### Step 2 — local patches

Each carries `LOCAL PATCH (augenblick)`, so `grep -rn` finds every divergence.

- **2a — C++17**, in `CMakeLists.txt:58, 65` and `neus2_tcnn/CMakeLists.txt:63, 71`. Raise
  `CMAKE_CXX_STANDARD`/`CMAKE_CUDA_STANDARD` to 17. CCCL 3 requires it; Eigen 3.4 and CUTLASS
  2.10 support it.
- **2b — existence checks**:
  - wrap NeuS2's glfw check (`:39-45`) in `if (NGP_BUILD_WITH_GUI)`;
  - point tcnn's CUTLASS check (`:46-52`) at `dependencies/cutlass/include/cutlass/cutlass.h`.
- **2c — drop tcnn's `/opt/cuda` hard-code** (`:152-158`). The `cublasLt` it searches for is
  never linked.
- **2d — pybind11 v2.13.6** in place of 2.7 (fact 13). `pybind11_json` and the `pyngp` target
  (`pybind11_extension`, `pybind11::module`) use only APIs 2.13 keeps.
- **Build-fix patches as found.** Fix compile errors under CUDA 13 / sm_120 with minimal edits,
  one marker each, as was done for the other rasterisers: deprecated `cub::`/thrust spellings,
  `__shfl` variants, CUTLASS 2.10 template issues. Record every site in `backend-neus2.md`.
  If CUTLASS 2.10 itself is the blocker, use the fallback toolchain (`cuda/12.9.1`) **before**
  bumping CUTLASS: a CUTLASS swap changes the MLP kernels, so it is a different method.

No C++ source patch is needed for paths: the stage scripts pass absolute paths and run from the
tree root (facts 8, 9), and normalise on the Python side (facts 1–3).

### Step 3 — build `pyngp` into the test env

- **3a — snapshot:** `$ENV/bin/python -m pip freeze > $SNAP/pip_freeze_before.txt`.
- **3b — build** out of tree, **per env**: the repo checkout is shared by every env, and an
  in-tree `build/` would be arch- and prefix-specific.

  ```bash
  B=$ENV/var/augenblick/neus2-build
  TCNN_CUDA_ARCHITECTURES=120 cmake -S src/libs/neus2 -B $B \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNGP_BUILD_WITH_GUI=OFF -DNGP_BUILD_WITH_VULKAN=OFF \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DPython_EXECUTABLE=$ENV/bin/python
  cmake --build $B --target pyngp -j "$SLURM_CPUS_PER_TASK"
  cp $B/pyngp*.so "$($ENV/bin/python -c 'import sysconfig;print(sysconfig.get_paths()["platlib"])')/"
  ```

  - Build only the `pyngp` target; `testbed` is never needed.
  - Install by copy into the test env's site-packages, so `import pyngp` works like every other
    per-env extension, and the step-0 import check can test it.
  - Run as a one-off GPU job reproducing these lines, **not** `rebuild_env.sbatch` or
    `setup_env.sbatch`, which rerun all of `auto_setup.sh` including torch (lesson from GeoSVR).
- **3c — check:** write `pip_freeze_after.txt` and diff it into `$SNAP/pip_diff.txt`. pip should
  see **no** change, since `pyngp` is not a distribution. `ldd pyngp*.so` must resolve every
  CUDA library under the job's module set. `find $PROD -newer $SNAP/prod_marker` must be empty.

### Step 4 — stage scripts in `src/libs/neus2/`

**One process per stage**, like the other backends' vendored scripts. Each stage is a script at
the vendored tree's root, run as `python <script> -s <scene> -m <out> ...` with
cwd = `src/libs/neus2`. Python puts the script's directory on `sys.path`, so the scripts import
the two shared modules as siblings. **No NeuS2 code lives under `src/augenblick/`** except the
adapter of step 5.

| File | Role | Imports `pyngp` |
|---|---|---|
| `scene_io.py` | shared, CPU-only: the bound, the JSON frames, RGBA staging, the vertex inverse map, the cluster filter, held-out intrinsics, the sRGB transfer, reading `normalization.json` | no |
| `ngp_session.py` | shared: the cwd preflight (4a), `open_testbed(snapshot=None)`, the snapshot path for a step count | inside its functions |
| `prepare.py` | stage 1, staging (4b) | no |
| `train.py` | stage 2, training (4c) | yes |
| `extract_mesh.py` | stage 3, marching cubes + mapping back + post-processing (4d) | yes |
| `render.py` | stage 4, held-out renders, with `--eval` only (4e) | yes |

Rules every file follows:

- Top-level imports are the stdlib, `numpy`, `PIL`, the sibling modules and the four
  `augenblick` modules above. `pyngp`, `pycolmap` and `open3d` are imported inside the functions
  that need them, so every file loads on a CPU node without `pyngp` (step 7 tests this). The CPU
  tests load them by path, as `tests/test_geosvr_eval_raster.py:8` loads GeoSVR's `render.py`.
- Each script's argparse defaults equal `Neus2Config`'s (step 5), so a bare invocation is the
  benchmark recipe. The adapter still passes every value explicitly.
- Every path argument is made absolute on entry.
- A `SceneError` exits with code **2** and its message, matching the CLI's own exit code for
  scene errors.

The stages hand over through files in `OUT`, never through process state: `prepare.py` writes
`neus2/scene/` and `neus2/normalization.json`; `train.py` writes the snapshot; `extract_mesh.py`
and `render.py` each load the snapshot into a fresh `Testbed`. The two extra snapshot loads cost
seconds (a msgpack of the hash grid and MLP), and buy per-stage timings in
`stage_timings.json` and the ability to rerun one stage alone.

**4a — preflight** (`ngp_session.py`, called by every `pyngp` script before building a
`Testbed`). `utils/mlp_weights.txt` exists relative to cwd (fact 8); otherwise exit 2 naming the
cwd and the file.

**4b — `prepare.py`** (`-s S -m O -r R --bound_percentile q --bound_margin m [--eval]`):

1. **Cameras.** Read the cameras with `eval.cameras.camera_records(scene.model_dir)`. It returns
   cameras sorted by name, with `position` = centre and `rotation` = camera-to-world rotation,
   which is the lineage convention.
   - Require `SIMPLE_PINHOLE`/`PINHOLE`; otherwise raise `SceneError`. Reuse
     `mesh_render.calibration`, which already raises for other models. (The adapter checks this
     first, in `validate()`; this is the backstop for a direct invocation.)
   - Raster per camera: `size = mesh_render.image_size(rec, R)`; intrinsics from
     `mesh_render.calibration(rec, size)`.
2. **Bound, centre.** Take the SfM points (`points3D.bin` via pycolmap, as GeoSVR does) and their
   `q`/`100−q` percentile box (default `q = 1.0`). `c` is the box centre and
   `r_pts = ½·‖hi − lo‖`.
3. **Bound, mask cover.** For every registered view, take its mask at the raster (NEAREST,
   `> 127`, the scorer's reader), with a pixel stride of 4.
   - Each foreground pixel's ray must pass within `r` of `c`, or the object it sees lies
     outside the sphere.
   - `r_mask` is the **99.9th percentile** of the rays' closest-approach distances to `c`. The
     percentile tolerates stray mask pixels.
4. **Bound, radius.** `r = m · max(r_pts, r_mask)` (default `m = 1.1`, NeuS's
   `scale_mat_scale`).
   - **Gate:** `min_i ‖C_i − c‖ ≥ 1.5·r`, or `SceneError` with the numbers. NeuS assumes
     cameras outside the sphere.
   - Expected on the verification specimen (`UF_Mammals_19126_cranium`): `r_pts` ≈ 0.70 and a
     camera-distance ratio of ≈ 4.2 after the margin (*checked*, points only).
5. **Images.** Write `O/neus2/scene/images/<stem>.png` as RGBA.
   - RGB is the photo `.convert("RGB").resize(size, BICUBIC)`, exactly the scorer's canonical
     GT path (`nvs.py:192-194`), so training and scoring see identical pixels.
   - Alpha is the raster mask × 255.
   - Use `compress_level=1`, because the files are deleted after a successful run (step 5,
     `finalize()`).
6. **JSONs.** Write `O/neus2/scene/transforms_{train,test}.json` (test only with `--eval`; the
   partition comes from `split.json` via `eval.split.read_split`):
   - header: `{"from_na": true, "aabb_scale": 1, "scale": 0.5, "offset": [0.5,0.5,0.5]}`;
   - per frame: `file_path: "images/<stem>.png"`, `transform_matrix` =
     `[[R, (C − c)/r], [0,0,0,1]]`, and `fl_x, fl_y, cx, cy, w, h` (fact 3).
7. **Record.** Write `O/neus2/normalization.json` with `c`, `r`, `r_pts`, `r_mask`, `q`, `m`,
   the camera-distance ratio, the mask-miss fraction at the final `r` (should be ≤ 0.1 %),
   and the raster.

**4c — `train.py`** (`-m O --n_steps N --ek_loss_weight e --mask_loss_weight w`):

1. Write the effective network config, `O/neus2/network.json`: `configs/nerf/dtu.json` with
   `hyperparams.{ek_loss_weight, mask_loss_weight}` overridden and
   `first_frame_max_training_step = N` (fact 10). `dtu.json` is plain JSON; `base.json` has
   `//` comments and is not offered.
2. `t = ngp.Testbed(ngp.TestbedMode.Nerf)`, then `t.load_training_data(<abs train json>)`
   (fact 9), then `t.reload_network_from_file(<network.json>)`, then `t.shall_train = True`.
3. Loop `while t.frame(): ... if t.training_step >= N: break`, as upstream does
   (`scripts/run.py:207-216`). Log the loss every 500 steps.
4. `t.save_snapshot(O/checkpoints/<N>.msgpack, False)`.

**4d — `extract_mesh.py`** (`-m O --n_steps N --mc_res M --num_cluster K`):

1. `t = open_testbed(<snapshot for N>)`; read `c` and `r` from `normalization.json`.
2. `d = t.compute_marching_cubes_mesh(resolution=[M]*3, thresh=0.0)`.
3. Map the vertices back: `V = c + r·(2·d["V"] − 1)` (fact 6). Faces as returned; vertex colours
   from `d["C"]`.
4. Write `O/mesh/neus2_raw.ply`.
5. Post-process with the same rule as 2DGS's `post_process_mesh`
   (`2dgs/utils/mesh_utils.py:22-43`): keep clusters of at least the K-th largest triangle
   count, and at least 50 triangles. Write `O/mesh/neus2_post.ply`. `K = 0` copies the raw
   mesh.

**4e — `render.py`** (`-s S -m O --n_steps N --evaluation_resolution E`; the adapter adds this
stage with `--eval` only):

1. `t = open_testbed(<snapshot for N>)`.
2. For each held-out frame in `transforms_test.json`:
   - `t.set_nerf_camera_matrix(T[:3])`, `t.fov_axis = 1`,
     `t.fov_xy = [fov(fx/H), fov(fy/H)]`, `t.screen_center = [1 − cx/W, 1 − cy/H]`
     (fact 11);
   - `background_color = [0,0,0,1]`, `snap_to_pixel_centers = True`,
     `nerf.rendering_min_transmittance = 1e-4`, `spp = 8`;
   - these are upstream's evaluation settings (`render_utils.py:149-152, 272-283`).
3. Save `linear_to_srgb(img[..., :3])`, clipped, as `O/test/ours_<N>/renders/<stem>.png`.
4. Write the GT as the original photo resized BICUBIC to the same raster, into `.../gt/<stem>.png`.
5. Write `.../protocol.json` (raster, spp, background), as GeoSVR does.
6. `E` (default −1) sets the raster independently of training, mirroring GeoSVR's
   `evaluation_resolution`. NeuS2 renders any size natively, so no camera cloning is needed;
   at `E ≠ R` the intrinsics are rescaled with `mesh_render.calibration` from the scene's
   cameras, not taken from the staged JSON.

### Step 5 — adapter, `src/augenblick/reconstruction/neus2.py`

The module exposes **only `Neus2Backend` and `Neus2Config`** (`__all__`). It holds no NeuS2
logic: no math, no JSON, no image or mesh handling, none of which belongs outside
`src/libs/neus2/` (step 4). Script paths are derived from `backend_dir`
(`self.backend_dir / "train.py"`); there are no public module constants. This is narrower than
GeoSVR's module, which exports `TRAIN_SCRIPT` and friends.

`Neus2Backend(SubprocessBackend)`:

- `name = "neus2"`, `title = "NeuS2 Reconstruction Pipeline"`;
- `backend_dir = LIBS_DIR / "neus2"` and the default `use_cwd = True` (fact 8);
- registered with `@register_reconstruction`, and `neus2` added to the `__init__.py` import line,
  whose import is what fires registration.

**`Neus2Config(EvalParams, ResolutionParams)`**, frozen:

| Field | Default | Goes to |
|---|---|---|
| `n_steps` | `15000` | `train.py`, and `extract_mesh.py`/`render.py` to find the snapshot; upstream's DTU recipe (README) |
| `ek_loss_weight` | `0.01` | `train.py` → effective `network.json` (`dtu.json` value) |
| `mask_loss_weight` | `0.0` | `train.py` → effective `network.json`; alpha compositing already supervises (fact 4) |
| `mc_res` | `1024` | `extract_mesh.py`; upstream default is 512 (below) |
| `num_cluster` | `1` | `extract_mesh.py`; 0 keeps every component |
| `bound_percentile` | `1.0` | `prepare.py`; point-box percentile |
| `bound_margin` | `1.1` | `prepare.py`; sphere radius multiplier |
| `evaluation_resolution` | `-1` | `render.py`, with `--eval` only; same contract as GeoSVR |
| `keep_staged` | `False` | `finalize()`; keep `neus2/scene/images/` after success |

`resolution` (from `ResolutionParams`) goes to `prepare.py` only: the training raster is fixed
at staging. `eval` adds the `render` stage and `--eval` to `prepare.py`.

**Why 1024.** On the verification specimen the sphere radius is ≈ 0.77 units (points only;
the mask cover can only grow it). At 512 the marching-cubes cell is 2r/512 ≈ 0.003 units; at
1024 it is ≈ 0.0015, closer to the ~0.001-unit
TSDF voxels the other backends mesh at. The grid costs ~4.3 GB of VRAM at 1024³. Check 10
compares 512 against 1024.

**`validate()`** (all CPU, before the GPU work starts):

- **A mask for every registered image.** NeuS2 has no unmasked mode on this branch; without
  masks the background is reconstructed inside the sphere. Raise `SceneError` (exit 2) listing
  the count and the first few stems. This is a NeuS2-specific message, not
  `require_registered_masks`, whose text suggests `--lambda_mask 0`.
- **Pinhole cameras only** (fact 3 / 4b).
- **`importlib.util.find_spec("pyngp")` is not `None`.** Otherwise raise `SceneError` naming
  step 3 and the active env, as MILo does for `tetranerf` (`milo.py:37-39, 114-117`). This is
  what production reports until step 10.
- `evaluation_resolution` is −1 or > 0 when `--eval`, as GeoSVR checks.

**No `prepare()` override.** Staging is `prepare.py`, a timed stage of its own, which keeps PIL
and pycolmap work out of the CLI process. The base class writes `split.json` before the stages
run, so `prepare.py` can read it.

**`stages()`** (`L = self.backend_dir`, `py = sys.executable`; every value passed explicitly, so
a change to a script's defaults cannot silently change the benchmark recipe):

```
"NeuS2 staging"      : py L/prepare.py      -s <scene> -m <out> -r R
                                            --bound_percentile q --bound_margin m [--eval]
"NeuS2 training"     : py L/train.py        -m <out> --n_steps N
                                            --ek_loss_weight e --mask_loss_weight w
"Mesh extraction"    : py L/extract_mesh.py -m <out> --n_steps N --mc_res M --num_cluster K
"Held-out rendering" : py L/render.py       -s <scene> -m <out> --n_steps N
                                            --evaluation_resolution E           (with --eval)
```

Rendering comes after mesh extraction, as in GW, so a render failure still leaves a mesh.

**`finalize()`** (runs only after every stage succeeded):

- `export_cameras(scene.model_dir, output_dir / "cameras.json")`, as GeoSVR does
  (`geosvr.py:84-87`). The Snakefile requires it for lineage. The cameras are in the SfM frame
  because the mesh was mapped back (fact 6).
- Then delete `neus2/scene/images/` unless `keep_staged` (~1.5 GB per run at 432 views). The
  JSONs, `network.json` and `normalization.json` stay. A failed run never reaches `finalize()`,
  so it keeps everything for debugging and for rerunning one stage by hand.

**Paths:**

- `mesh_path()` → `<output>/mesh/neus2_post.ply`;
- `test_renders_root()` → the default `<output>/test`;
- renders in `test/ours_<n_steps>/`, named by stem, which `pair_views` accepts.

### Step 6 — benchmark workflow and bench tables

- `workflow/Snakefile`:
  - `MESH_PATHS["neus2"] = "mesh/neus2_post.ply"`;
  - extend the `geosvr`-only `--evaluation_resolution` branch in `backend_arguments` to
    `{"geosvr", "neus2"}`.
- `workflow/run_experiments.sh:20-22`: add `neus2` to the allow-list and its message.
- `src/augenblick/bench/tree.py:17`: `"neus2"` in `RECONSTRUCTION_STAGES`.
- `src/augenblick/bench/links.py`: `("neus2", "reconstructions", "{spec}/colmap-neus2", False)`.

### Step 7 — tests (CPU-only)

Run under `$ENV/bin/python -m pytest`. Tests are listed by file.

**`tests/test_recon_argv.py`:**

- `test_neus2_argv`: the three stages at defaults, each asserted on its whole argv, starting
  `sys.executable, str(<backend_dir>/<script>)`, with every config value passed explicitly.
- `test_neus2_eval_adds_render_stage`: four stages with `--eval`, three without; the render
  stage comes last; `--eval` reaches `prepare.py`; `--evaluation_resolution 2` reaches
  `render.py` only.
- `test_neus2_module_surface`: `neus2.__all__ == ["Neus2Backend", "Neus2Config"]`, and the
  module defines no other public name.
- `test_neus2_finalize_cleans_staging`: on a fake output dir, `finalize()` writes
  `cameras.json` and removes `neus2/scene/images/`, keeping the JSONs; with `keep_staged` the
  images stay.
- `test_neus2_runs_from_vendored_tree`: `backend_dir` is `src/libs/neus2`, `use_cwd` is true,
  and `backend_dir/"utils/mlp_weights.txt"` exists (fact 8).
- `test_neus2_requires_every_mask` / `…_ignores_unregistered_images`: the same pattern as the
  GeoSVR tests.
- `test_neus2_rejects_non_pinhole`: build a `SIMPLE_RADIAL` model with
  `test_meshroom_from_sfm.build_scene` (move it to `tests/conftest.py`).
- `test_neus2_requires_pyngp`: monkeypatch `find_spec` to return `None` → `SceneError`.
- `test_neus2_patches_present`, a grep guard against a wholesale resync:
  - C++17 in both `CMakeLists.txt`;
  - the glfw check gated on the GUI option;
  - the CUTLASS check pointing at `cutlass.h`;
  - no `/opt/cuda`;
  - `PYBIND11_VERSION_MINOR >= 12`;
  - the six added files of step 1 exist and carry the `LOCAL PATCH (augenblick)` header.

**`tests/test_neus2_scene_io.py`** (new; loads `src/libs/neus2/scene_io.py` by path, no `pyngp`):

- **Normalisation round trip.** Random cameras and points → staging helpers → JSON matrices.
  - `T[:3,:3]` equals `rotation`, and `T[:3,3]·r + c` equals the camera centre.
  - Mapping NGP vertices back recovers world points to 1e-6.
  - Through `nerf_matrix_to_ngp`'s `from_na` branch, re-implemented in the test, the camera
    lands at `0.5 + 0.5·(C − c)/r`.
- **JSON contract:**
  - `from_na`, `scale`, `offset` and `aabb_scale` are as specified, with no top-level `fl_*`,
    `cx` or `cy`;
  - every frame has all six intrinsics at the raster;
  - frames are sorted by name;
  - the train/test partition matches `split.json`.
- **Bound:**
  - a point cloud with 2 % far outliers yields `r_pts` from the inliers;
  - a mask wider than the points grows `r` to `r_mask`;
  - a camera inside `1.5·r` → `SceneError`.
- **RGBA staging:** alpha is the NEAREST mask × 255, and RGB is **bit-identical** to the
  scorer's BICUBIC canonical GT for the same photo and raster.
- **Raster:** 6240×4160 at `-r -1` → 1600×1066, the floor rule (`mesh_render.image_size`).
- **sRGB:** `linear_to_srgb(srgb_to_linear(x)) ≈ x`.
- **Post-processing:** two disjoint triangle soups → `K = 1` keeps the larger one; `K = 0`
  keeps both.
- **Held-out intrinsics:** `fov_xy` and `screen_center` from `(fx, fy, cx, cy, W, H)` reproduce
  `relative_focal = f/H` and `1 − c/size` (fact 11).
- **No torch:** importing `scene_io` leaves `torch` out of `sys.modules` (run in a subprocess so
  other tests' imports do not leak in).

**`tests/test_neus2_scripts.py`** (new; loads each of the four stage scripts by path):

- each script loads with `pyngp` unimportable (it is imported only inside functions);
- each script's argparse defaults equal the matching `Neus2Config` field defaults, so the two
  cannot drift;
- `prepare.py` end to end on a tiny synthetic scene (3 cameras, a point cloud, masks) writes
  the JSONs, RGBA PNGs and `normalization.json` the `scene_io` tests specify, with no GPU;
- for `train.py`, `extract_mesh.py` and `render.py`, a missing `utils/mlp_weights.txt` (cwd set
  to a temp dir) exits 2 before `pyngp` is imported.

**Elsewhere:**

- `tests/test_registry.py:52`: add `"neus2"` to the registry set.
- `tests/test_bench.py` / `tests/test_benchmark_workflow.py`: add `"neus2"` to the parametrised
  backend lists, and a dry run with `MESH_PATHS["neus2"]`.

### Step 8 — SLURM and setup scripts

- **`hpg_slurm/common.sh:14`** (lands before any verification job): make the rtx6000 arm
  `CONDA_ENV="${CONDA_ENV:-$CONDA_ROOT/augenblick_rtx_pro_6000}"`, and the b200 arm likewise.
  This is the override `meshroom_common.sh:7` already offers. sbatch exports the submitting
  shell's environment, so `CONDA_ENV=$ENV sbatch ...` points one job at the test env, and the
  banner's `conda env :` line (`scene_common.sh:90`) records which env ran. `CONDA_ENV` is not
  set by conda itself, and is unset in the login shell (*checked*), so existing submissions are
  unchanged. `pace_slurm/common.sh` is left alone: nothing is verified on PACE.
- `hpg_slurm/recon.sbatch:5, 37-40` and `pace_slurm/recon.sbatch:3, 29-31`: add `neus2` to the
  header, the `case "$BACKEND"` allow-list and its error message. The defaults (64 GB,
  3 h, 1 GPU) should suffice for benchmark runs: host RAM is small because images are staged at
  the raster. Verification overrides them (see "Job resources").
- `scripts/setup_common.sh`: a `neus2` arm modelled on the tetra block (`:136-161`).
  - It runs step 3b's commands with `TCNN_CUDA_ARCHITECTURES="${GPU_ARCH/./}"` (`12.0` →
    `120`) and the build dir under `$CONDA_PREFIX/var/augenblick/`.
  - It is **non-fatal** (`|| echo "WARN: ... neus2 unavailable; other backends fine."`), like
    tetra: a C++ build that fails should not abort an env rebuild.
  - Add `neus2` to the `BACKENDS` default (`:14`) **only at step 10**. Until then it is opt-in
    via `BACKENDS=neus2`.
- `hpg_slurm/rebuild_env.sbatch:57-62`: add `"pyngp"` to the import check, also at step 10, so
  production rebuild logs do not show a `FAIL` before it has the module.

### Step 9 — docs (proposed edits, per CLAUDE.md)

- **New `MEMORY/backend-neus2.md`:**
  - what it is; the four stage scripts and the two shared modules; the normalisation and its
    inverse; RGBA staging; outputs; timings from check 8; the licence;
  - a **Vendoring** section: URLs, the three commits, the trimmed/dropped paths, the pybind11
    swap, the six added files, and each `LOCAL PATCH` site, with a reminder to re-apply them on
    resync;
  - the traps: `from_na` presence-only, per-frame intrinsics, the cwd `utils/` weights, a
    directory path loading every JSON.
- **`MEMORY/pipeline-reconstruction.md`:** the new wrapper, its flags and its output paths.
- **`MEMORY/augenblick-package.md`:**
  - `neus2` in the method list;
  - "Per-backend divergences": stage scripts we added to the vendored tree instead of
    upstream's, staging as a timed stage, an adapter module exposing only the backend and its
    config, the mesh mapped back to the SfM frame, `pyngp` as a per-env CMake build.
- **`MEMORY/pipeline-evaluation.md`:** `neus2` in the `EvalParams` list.
- **`MEMORY/environment-and-gpu.md`:**
  - the CMake build (per-env build dir, `TCNN_CUDA_ARCHITECTURES`, non-fatal);
  - the pybind11/numpy-2 trap;
  - `neus2` in `BACKENDS`;
  - cloning an env for a risky install: `conda create --clone`, the shebang/prefix check, and
    `RUNPATH`s that still point at the source env.
- **`MEMORY/cluster-slurm.md`:** the `CONDA_ENV` override in `common.sh`.
- **`docs/nvs-protocol.md`:** add NeuS2 to the supported-backend sentence and the
  `evaluation_resolution` paragraph.
- **`.claude/CLAUDE.md`:**
  - `neus2` in the Reconstruction row and the `src/libs/` list;
  - `backend-neus2.md` in the index;
  - `src/libs/neus2` in the `LOCAL PATCH` gotcha.

## Verification

All of it runs in the test env, from `$WT`. **Specimen:** `UF_Mammals_19126_cranium` with
**COLMAP** SfM, from the live tree `/blue/.../results/neurips/`. Checked 2026-11-26:

- 432/432 images registered, 93,000 points, all `SIMPLE_PINHOLE` at 6240×4160 → raster
  1600×1066;
- 432 images and 433 mask files; the extra is `_provenance.json`, not a mask;
- array index **25** in `neurips/prepared` (76 scenes; 0-based, in `scene_common.sh`'s sorted
  `flat` order);
- **bound, points only:** the 1–99 % extent is 1.14 × 0.49 × 0.66 units (`r_pts` ≈ 0.70), but the
  full extent is 3.5 × **31.5** × 5.9 units. Outliers along y are 10× worse than on the Herp
  specimen, which makes this a sharper test of the robust bound. Camera centres sit 3.21–4.30
  units from the box centre: a camera-distance ratio of ≈ 4.2 after the 1.1 margin, before
  the mask cover can grow `r`;
- reference mesh for check 9: `neurips/gt_20261113/UF_Mammals_19126_cranium_low_poly/mesh.obj`
  (low-poly only; no high-poly drop for this specimen);
- existing colmap-input runs, from `neurips/_timing/sfm_timings.csv` (rtx6000): colmap SfM
  765 s, 2dgs 1337 s, pgsr 2746 s, gw 1922 s, geosvr 3459 s (4092 s on the 2026-11-26 rerun),
  milo 4622 s, meshroom 1670 s.

**Every result goes under `V=$WT/output/results/neurips_verify/neus2`.** Nothing is written into
`neurips/`. Stage the input scene so `--eval`'s `split.json` lands in `$V`:

```bash
SRC=$WT/output/results/neurips/UF_Mammals_19126_cranium/colmap
DST=$V/UF_Mammals_19126_cranium/colmap
mkdir -p $DST/sparse && ln -s "$(readlink -f $SRC/images)" $DST/images \
  && ln -s "$(readlink -f $SRC/masks)" $DST/masks && cp -r $SRC/sparse/0 $DST/sparse/0
```

Symlinks are safe here, unlike GeoSVR: NeuS2 only reads `images/` and `masks/`, and the only
write into the scene root is `split.json`. Runs go through the real job script, submitted from
`$WT` and pointed at the test env:

```bash
cd $WT
export CONDA_ENV=/blue/arthur.porto/srizvi63.gatech/conda/augenblick_rtx_pro_6000_neus2
export DATA_ROOT=/blue/arthur.porto/data/datasets/photogrammetry/neurips/prepared
export RESULT_ROOT=$V BACKEND=neus2 SFM=colmap
sbatch $GPU_JOB --array=25 hpg_slurm/recon.sbatch --eval --n_steps 2000 --keep_staged  # smoke → mv to colmap-neus2-smoke
sbatch $GPU_JOB --array=25 hpg_slurm/recon.sbatch --eval                               # checks 6–8 → colmap-neus2
```

The output dir is always `$RESULT_ROOT/<scene>/colmap-neus2`, so run one at a time and rename
each before the next starts. `--n_steps 2000` is a **smoke value**; never quote its mesh. Every
job log's `conda env :` line must name the test env, and its `repo :` line `$WT` on
`syed/neus2-inclusion`.

| # | Check | Pass condition |
|---|---|---|
| 0 | build + env | step 0's clone checks pass, and the runtime proof passes again with `pyngp` imported; `pyngp` imports in a GPU job under `$ENV`; `ldd` resolves every library; `$SNAP/pip_diff.txt` empty; torch/numpy unchanged; the build log records toolchain, arch 120 and each build-fix patch |
| 1 | registration | `$ENV/bin/augenblick recon --list` shows `neus2`; `--help` shows every field; `validate()` fails with exit 2 on a copy with one mask removed; `$PROD/bin/augenblick recon --list` does **not** show `neus2` |
| 2 | tests | `$ENV/bin/python -m pytest tests/` green |
| 3 | pybind11 / numpy 2 | on a GPU node: a 3×4 matrix through `set_nerf_camera_matrix` then `camera_matrix` round-trips; `render(64, 48, 1)` returns shape `(48, 64, 4)` with C-contiguous strides; `compute_marching_cubes_mesh` arrays have the expected shapes and dtypes |
| 4 | staging | `normalization.json`: camera ratio ≥ 1.5 (expect ≈ 4.2 or less if the mask cover grows `r`), `r_pts` ≈ 0.70 despite the 31.5-unit outliers, mask-miss ≤ 0.1 %; `Loaded 378 images` (train only); RGBA alpha matches masks (the smoke run keeps them) |
| 5 | input untouched | `find $DST -newer <out>/neus2/normalization.json` lists only `split.json`, whose timestamp predates training |
| 6 | held-out camera path | for one **training** view, `set_camera_to_training_view(i)` and the manual matrix + `fov_xy` + `screen_center` path render images within 1e-4 mean absolute difference |
| 7 | mesh frame | mesh bbox inside the SfM 1–99 % box grown 20 %; ≥ 95 % of vertices projected into five training views fall inside the masks; `eval.metric_mesh` lineage check passes on `cameras.json` |
| 8 | full run + NVS | `n_steps 15000 --eval`: mesh at `mesh_path()`; renders at 1600×1066 under `test/ours_15000/`; `n_test` 54; `nvs_metrics.json` scored with `--require-native-raster`; `stage_timings.json` per stage, plus peak RSS/VRAM, against the timings above; staged images gone |
| 9 | schedule | `--n_steps 30000` vs 15000 (fact 14): train PSNR, held-out masked PSNR, F@0.5 % vs the low-poly reference above (similarity fit, no metric claim) |
| 10 | MC resolution | `extract_mesh.py --mc_res 512` run by hand (cwd `$WT/src/libs/neus2`) vs check 8's 1024 mesh, on a copy of the output dir: face count, F@0.5 %, time |
| 11 | production untouched | `find $PROD -newer $SNAP/prod_marker` empty; `$PROD` pip freeze equals `prod_freeze_before.txt`; `git -C $MAIN status` and `HEAD` unchanged; the diff guard of "How the clone is isolated" is empty |

### Verification results (2026-11-26)

All jobs ran in the clone, from `$WT`, with the "Job resources" flags.

| # | Result |
|---|---|
| 0 | job 52898251: every import OK, `pyngp` from the clone, no mapped file under `$PROD`, pip diff empty, `ldd` resolved |
| 1 | `augenblick recon --list` shows `neus2` in the clone, not in `$PROD`; a missing mask exits 2 with the NeuS2 message |
| 2 | 270 passed, 5 skipped; only `test_recon_argv.py`'s pre-existing `SyntaxError` |
| 3 | camera-matrix round trip, `render` `(48,64,4)` float32 C-contiguous, marching-cubes arrays float32/int32 |
| 4 | `r = 0.7721` (`r_pts` 0.702 despite 31.5-unit outliers, `r_mask` 0.583), camera ratio 4.16, mask miss 0 %, 378 train views; staged alpha == mask |
| 5 | nothing in the input scene newer than `normalization.json`; `split.json` predates it |
| 6 | job 52903316: training views 0/100/200, worst mean \|diff\| 3.3e-7 (synthetic: 8.2e-8) |
| 7 | mesh bbox inside the grown SfM box; 96.3–97.9 % of vertices inside the masks in five views; lineage worst 5e-15 |
| 8 | job 52902265: 246 s total (staging 17, training 146, mesh 13, render 69), ~0.8 GB RSS; 54 renders at 1600×1066; strict re-score (job 52903682, `canonical_raw_photo_v1`, `--require-native-raster --require-complete`): PSNR 28.84, SSIM 0.876, LPIPS 0.145; staged images removed |
| 9 | 30k (job 52902268, 391 s): PSNR 29.05, SSIM 0.902, LPIPS 0.113; F@0.1 % 0.774 vs 0.647, F@0.5 % 0.965 vs 0.958, Chamfer 0.109 % vs 0.136 % |
| 10 | `mc_res 512`: 9 s, 686k faces, F@0.5 % 0.958, F@0.1 % 0.647 — the same as 1024 |
| 11 | `$PROD` untouched (0 newer files, freeze unchanged, no `pyngp`); `$MAIN` on `8ec1aa5`; diff guard empty |

Mesh scores use `eval.mesh`'s similarity fit against
`gt_20261113/UF_Mammals_19126_cranium_low_poly/mesh.obj`, as `_eval/gt_20261113` scored the other
backends (F@0.5 %: pgsr 0.971, meshroom 0.965, gw 0.957, 2dgs 0.915, geosvr 0.878, milo 0.804).

## Step 10 — promotion to the production env

Only after every check passes **and the user says go**. Production runs the whole benchmark, so
this is not done as part of verification.

1. Copy the verified `pyngp*.so` from `$ENV`'s site-packages into `$PROD`'s. Both envs have the
   same Python 3.10 and numpy 2.2.6 (step 0), and `pyngp` links nothing from the env but
   libpython, so the verified binary is the one promoted, rather than a fresh build of it.
2. Re-run checks 0 (import + `ldd`) and 3 under `$PROD` in a GPU job.
3. Land the step-8 items held for promotion: `neus2` in the `BACKENDS` default, `pyngp` in
   `rebuild_env.sbatch`'s import check.
4. Deleting `$ENV` (and, after the merge, `$WT`) is the user's call. It is not referenced by anything committed once
   verification is over.

## Open risks

- **CUDA 13 / sm_120 build.** Untested upstream: old tcnn plus CUTLASS 2.10. The likely failures
  are CCCL C++17 fallout and CUTLASS template errors. The fallback is `cuda/12.9.1` with an
  RPATH. If neither builds, the plan stops at step 3. Bumping CUTLASS is a method change and
  needs sign-off.
- **pybind11 2.13 vs NeuS2's bindings.** The swap is required for numpy 2, but the bindings were
  written against Tom94's 2.7 fork (Eigen casters, `py::call_guard`). Expect small compile fixes,
  recorded as patches.
- **Residual sharing with production.** After step 0 the clone shares only the six editable
  libraries left pointed at `$MAIN` (vggt, sam3, dvlt, lightglue, pytorch3d, tetranerf) and
  any non-`RUNPATH` hard-coded paths in binaries. NeuS2 touches neither, the diff guard keeps the
  first true at every commit, and the `/proc/self/maps` proof catches the second.
- **Snapshot reloads.** `extract_mesh.py` and `render.py` each rebuild a `Testbed` and load the
  snapshot. Expected to cost seconds; check 8's `stage_timings.json` shows it. If it is ever
  material, merging the two into one script is local to `src/libs/neus2/` and the adapter.
- **Under-training at 432 views** (fact 14). Check 9 decides whether to change the default.
- **Bound on other SfM methods.** VGGT's 100k points and turntable/hull scenes have different
  outlier profiles. The mask-cover radius is the backstop, and the camera gate fails loudly
  rather than silently clipping. Run check 4 on one VGGT scene before a full sweep.
- **Thin structures** (fish fin rays, bird bones): SDF + Eikonal oversmooths. That is a property
  of the method, worth reporting, not fixing.

## Decisions for the user

The plan is written to these; each is easy to flip before implementation starts.

1. **Our own stage scripts in the vendored tree instead of patching upstream's `scripts/run.py`**
   (fact 7). Upstream fidelity is kept by using upstream's training loop, configs, evaluation
   settings and marching cubes unchanged.
2. **One process per stage** (`prepare`, `train`, `extract_mesh`, `render`), handing over
   through files, with snapshot reloads as the cost (open risks).
3. **`mc_res` 1024 rather than upstream's 512.** Check 10 measures both.
4. **Masks are mandatory.** Unmasked scenes would need the `neuspp` branch, which is out of scope.
5. **Staged PNGs are deleted after success** (`--keep_staged` to keep them).
6. **Promotion copies the verified `.so`** into production rather than rebuilding there.
