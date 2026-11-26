# `src/libs/neus2/` — NeuS2 (hash-grid SDF, marching cubes)

NeuS2 (Wang et al., ICCV 2023, `neus2`) is NeuS rebuilt inside instant-ngp: the surface is the
zero level set of an SDF MLP over a multi-resolution hash grid, volume-rendered with NeuS's
weights and trained with an Eikonal loss. All CUDA; Python reaches it only through the pybind11
module **`pyngp`**, a per-env CMake build (environment-and-gpu.md). It is the only backend that
**ignores the SfM points** except to place a bounding sphere: only poses, intrinsics and masks
reach training.

Wrapper: `augenblick recon neus2` (`src/augenblick/reconstruction/neus2.py`, which exposes only
`Neus2Backend` and `Neus2Config`); flags in [pipeline-reconstruction.md](pipeline-reconstruction.md).

## Stages: one script each, all in the vendored tree

Upstream's `scripts/run.py` is not vendored (it writes relative to cwd, stops in `pdb` on its test
path, and imports torch's tensorboard and `termcolor`). Our scripts sit at the tree root, run with
`cwd = src/libs/neus2`, and hand over through files in the output dir.

| Stage | Script | Does |
|---|---|---|
| NeuS2 staging | `prepare.py` | bound, RGBA training images, `transforms_{train,test}.json`, `normalization.json` |
| NeuS2 training | `train.py` | `configs/nerf/dtu.json` + overrides → `neus2/network.json`; trains; `checkpoints/<n>.msgpack` |
| Mesh extraction | `extract_mesh.py` | marching cubes at SDF 0, vertices mapped back to the SfM frame, 2DGS's cluster filter |
| Held-out rendering | `render.py` (`--eval`) | upstream eval settings (spp 8, black bg, pixel-centre snap), sRGB PNGs + GT, selection manifest |

`scene_io.py` (CPU helpers) and `ngp_session.py` (`open_testbed`, cwd preflight) are shared.
`pyngp` is imported only inside functions, so every script loads (and its CPU tests run) without
it; none imports torch. `extract_mesh.py` and `render.py` load only the snapshot: it carries the
dataset metadata (per-frame focal/principal point, `from_na`, scale/offset), so no images reload.

## Normalisation (the part upstream leaves to the caller)

NeuS2 needs the object inside the unit sphere (`scale 0.5`, `offset 0.5`, `aabb_scale 1` maps it
onto NGP's `[0,1]³`). `prepare.py`: centre `c` = the SfM points' 1–99 % box centre, `r_pts` its
half-diagonal; `r_mask` = 99.9th percentile of mask-pixel rays' closest approach to `c`;
`r = 1.1 · max(r_pts, r_mask)`; **refuses** (exit 2) if a camera is within `1.5 r`. Poses go in
unchanged as OpenCV camera-to-world (`from_na`), translation `(C − c)/r`. Meshes map back with
`x = c + r·(2v − 1)`, so `cameras.json` lineage holds.

## Traps

- **`from_na` is read by key presence only** (`nerf_loader.cu:346`): `"from_na": false` enables it.
- **Intrinsics must be per frame**: top-level `cx`/`cy` are parsed and then ignored (`:708`).
- **A directory path loads every JSON in it as a dynamic-scene frame**: always pass the train
  JSON's path.
- **cwd**: geometric init reads `utils/mlp_weights.txt` relative to cwd and `exit(1)`s inside C++
  if missing; `ngp_session.preflight()` raises first.
- **Masks are mandatory**: they become the alpha channel, and random-background compositing
  turns that into alpha supervision (`mask_loss_weight` stays 0). `validate()` refuses a scene
  with an unmasked registered image, a non-pinhole camera, or no `pyngp`.
- The training log's `mask` value grows while `mask_loss_weight` is 0; it is an unused
  diagnostic, not a loss.
- `load_training_data` tries to create `<json>/output/` and prints `create_directory:` lines; the
  mkdir fails harmlessly because the path is a file.

## Outputs

| Path | What |
|---|---|
| `neus2/scene/transforms_{train,test}.json` | staged frames; `neus2/scene/images/` is deleted by `finalize()` unless `--keep_staged` (~1.5 GB at 432 views) |
| `neus2/normalization.json` | `centre`, `radius`, `r_pts`, `r_mask`, camera ratio, mask-ray miss fraction |
| `neus2/network.json` | the effective network config |
| `checkpoints/<n_steps>.msgpack` | snapshot |
| `mesh/neus2_raw.ply`, `mesh/neus2_post.ply` | raw and cluster-filtered mesh (`mesh_path()`) |
| `test/ours_<n_steps>/{renders,gt,render_protocol.json}` | held-out renders at the scorer's raster |
| `native_selections/render_<res>.json` | the partition `render.py` consumed; `eval.nvs --require-complete` needs it |

## Reference run (2026-11-26)

`UF_Mammals_19126_cranium`, COLMAP SfM, 432 views (378 train / 54 test) at 1600×1066, RTX PRO 6000,
test env clone. Defaults (15k steps, `mc_res 1024`): staging 17 s, training 142 s, meshing 13 s,
rendering 69 s, **246 s total**, ~0.8 GB host RSS; sphere `r = 0.772`, camera ratio 4.16.
Held-out (masked, `canonical_raw_photo_v1`, `--require-native-raster --require-complete`): PSNR
28.84, SSIM 0.876, LPIPS 0.145. Mesh vs the low-poly reference (similarity fit, as
`_eval/gt_20261113`): F@0.5 % 0.958, F@0.1 % 0.647, Chamfer 0.136 %.

- **30k steps**: 391 s; PSNR 29.05; F@0.1 % 0.774, F@0.5 % 0.965, Chamfer 0.109 %. Upstream's
  15k under-trains fine detail at 432 views (fact 14 of the plan); the default is unchanged.
- **`mc_res 512`** on the 15k snapshot: 9 s, 686k faces, F@0.5 % 0.958, F@0.1 % 0.647 — the
  same scores as 1024 at these tolerances.
- Other backends on this specimen, F@0.5 %: pgsr 0.971, meshroom 0.965, gw 0.957, 2dgs 0.915,
  geosvr 0.878, milo 0.804 (1337–4622 s each).

## Licence

NVIDIA Source Code License-NC (instant-ngp), §3.3: research or evaluation only — the same
footing as GeoSVR and DVLT. Vendored dependencies keep their own licences (tiny-cuda-nn BSD-3,
CUTLASS BSD-3, Eigen MPL-2, pybind11 BSD, imgui MIT).

## Vendoring

Upstream `https://github.com/19reborn/NeuS2` at `081b8191d049158a8296bfcb15c7d9cf065540df`
(2024-03-22), with its submodule `neus2_tcnn` at `d3a12b91097ddca15f17b1b6dbacb36fa0c768b2` and
that fork's CUTLASS at `4db6a6140e45c4ffe6339c55b43b159602fa1f35` (2.10.0), all flattened in-tree.
Commit `9680ede` is upstream as-is apart from trimming, so `git diff 9680ede -- src/libs/neus2`
is the full set of local divergences.

**Dropped:** `scripts/`, `tools/`, `docs/`, `AnacondaBat.bat`, `requirements.txt`; the GUI deps
(`dlss`, `glfw`, `gl3w`, imgui and imguizmo sources); fmt; CUTLASS sources, tests and examples;
Eigen outside `Eigen/` and `unsupported/`. **Kept as headers only:** `imgui.h`, `imconfig.h`,
`ImGuizmo.h`, because `camera_path.h` includes them with the GUI off.

**Local patches**, each marked `LOCAL PATCH (augenblick)`; re-apply all of them on any resync.
`tests/test_neus2_backend.py::test_neus2_patches_present` fails if one is lost.

| # | Where | Change | Why |
|---|---|---|---|
| 2a | `CMakeLists.txt`, `dependencies/neus2_tcnn/CMakeLists.txt` | C++/CUDA standard 14 → 17 | CUDA 13's CCCL requires C++17 |
| 2b | same two files | glfw check only with the GUI; CUTLASS check looks for `include/cutlass/cutlass.h` | both were unconditional and need the dropped sources |
| 2c | `neus2_tcnn/CMakeLists.txt` | dropped the `/opt/cuda` hard-code and its `cublasLt` lookup | nothing links `cublasLt` |
| 2d | `dependencies/pybind11/` | upstream pybind11 v2.13.6 (trimmed) replaces 2.7 | 2.7 reads numpy 1's `PyArray_Descr`; the env is numpy 2.2.6 |
| 2e | `CMakeLists.txt` | `set(PYBIND11_FINDPYTHON ON)` before `add_subdirectory(pybind11)` | otherwise pybind11's legacy search links `/usr/lib64/libpython3.9.so` |
| 2f | `nerf_network.h` (5×), `takikawa_encoding.cuh` (1×) | `parallel_for_gpu` → `tcnn::parallel_for_gpu` | CUDA 13 applies two-phase lookup inside templates |
| — | `prepare.py`, `train.py`, `extract_mesh.py`, `render.py`, `scene_io.py`, `ngp_session.py` | first-party files | the stage scripts above |

Built with `cuda/13.0.2`, gcc 12.2, `TCNN_CUDA_ARCHITECTURES=120` (2026-11-26); CUTLASS 2.10 compiles
for sm_120 with only `double4`-style deprecation warnings.
