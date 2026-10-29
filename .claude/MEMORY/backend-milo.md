# `src/libs/milo/` — MILo (Mesh-In-the-Loop Gaussian Splatting)

MILo (Guédon, Gomez et al., SIGGRAPH Asia 2025 / TOG, arXiv 2506.24096, `guedon2025milo`)
extracts the mesh **inside** training. Each Gaussian spawns 9 pivots; a subset feeds a CGAL
Delaunay tetrahedralisation (`tetranerf`, rebuilt every 500 iterations), each pivot carries a
learnable occupancy/SDF, and differentiable marching tetrahedra turns them into a mesh every
iteration of the mesh phase. That mesh is rasterised with nvdiffrast and its depth/normals are
pulled toward the Gaussians'. `mesh_extract_sdf.py` then refines the SDF with the Gaussians
frozen and runs marching tetrahedra once more. No TSDF and no regular grid on the default path.
Same authors and code lineage as SuGaR and GW (GW's own `--milo` mode is a separate port).

Wrapper: `augenblick recon milo` (`src/augenblick/reconstruction/milo.py`); flags in
[pipeline-reconstruction.md](pipeline-reconstruction.md). The plan with full reasoning is
`.claude/PLANS/milo-reconstruction-backend.md`.

## Layout and invocation

Upstream's two-level layout is kept: `src/libs/milo/milo/` is the code (and `backend_dir`,
`use_cwd = True`), `src/libs/milo/submodules/` holds the two rasterisers MILo alone needs. Run
with cwd = `milo/`: `train.py`'s `--config_path` default, `./configs/fast`, is cwd-relative.

## Two recipes: `--densification regular|fast`

| | `regular` (default) | `fast` |
|---|---|---|
| Script | `train_regular_densification.py` | `train.py` |
| Upstream use | DTU, object-centric | the paper's T&T scenes |
| Densification | RaDe-GS/GOF style | Mini-Splatting2 (depth re-init, aggressive cloning, 2 simplifications) |
| Iterations | 30,000 | 18,000 (`configs/fast`) |
| Mesh config | `default_dtu` (mesh loop 20,001–30,000) | `default` (8,001–18,000) |

- **Extraction must match training.** `mesh_extract_sdf.py --iteration` defaults to 18000 and
  `--config` must name the training mesh config. The wrapper derives both from one field each.
- `--mesh_config` (and the extractor's `--config`) accept a `.yaml` path as well as a
  `configs/mesh/` name (patch 6), for custom schedules.
- **Under `regular`, the mesh loop must start after `warn_until_iter` (3,000).** The Gaussian
  sampling helpers call `getTrainCameras_warn_up`, which asks for the half-res camera set
  until then; only `train.py` builds that set, so an earlier start raises `KeyError: 2.0`.
- `train.py`'s Mini-Splatting2 renderer returns no alpha, so under `fast` the mask loss only
  acts from `regularization_from_iter` (3,000) on.
- **Under `fast`, `regularization_from_iter` must be ≥ `densify_until_iter`** (both 3,000 by
  default). From the former on, training renders with RaDe-GS, which returns no `area_max`, but
  Mini-Splatting2 densification reads it until the latter: `KeyError: 'area_max'` (seen
  2026-10-24 with 2000/3000). `validate()` refuses the overlap.

## Extractors: `--extract sdf|tsdf`

- `sdf` (default): `mesh_extract_sdf.py` → `<model>/mesh_learnable_sdf.ply`. Upstream never
  pairs it with the regular trainer (its DTU script uses a TSDF with absolute DTU-scale bounds).
- `tsdf`: `mesh_extract_regular_tsdf.py` → `<model>/mesh_regular_tsdf_res<mesh_res>_post.ply`.
  Scale-free (`depth_trunc = 2 × min camera-to-centre distance`, `voxel = depth_trunc /
  mesh_res`) and zeroes depth outside `gt_mask`. Because the range spans the **camera
  distance**, not the specimen, upstream's `mesh_res 1024` left a turntable specimen ~150 voxels
  wide (98 k faces on `UF_Herp_87980_cranium`); the wrapper defaults to **4096** (1.70 M faces,
  7.6 min, ~56 GiB RSS).
- The Delaunay SDF path **ignores masks** (`depth_fusion`'s `masks` argument is unused), so on
  the `sdf` path the mask shapes the mesh only through the composited GT and the alpha loss.

## Masks, split, input scene

- Masks: `masks/<image_name>.png` is read in `loadCam` at camera resolution; the GT image is
  composited onto the background there, once, and kept as `Camera.gt_mask`.
- `--lambda_mask` (wrapper default 0.1, upstream has no such loss): L1 between RaDe-GS rendered
  alpha and the mask. GOF returns no alpha, so `validate()` refuses `--rasterizer gof` with it.
  `validate()` also requires a mask per **registered** image (shared
  `require_registered_masks` in `reconstruction/base.py`).
- `--eval` reads `split.json` from the scene root; render file names are positional
  (`00000.png`) in sorted-name order, which `nvs.pair_views` maps back to stems.
- The input scene is read in place and **never written**: `points3D.ply` is read if present
  (so an `sfm hull` init cloud reaches MILo), else the cloud is built from `points3D.bin` in
  memory and written only to `<model>/input.ply`.

## Fixed training flags

`--data_device cpu`, `--test_iterations -1` (the end-of-training report would download LPIPS
weights), `--checkpoint_iterations -1` (resume checkpoints carry optimiser state; extraction
reads `point_cloud/iteration_<N>/point_cloud.ply`, which keeps the occupancy fields).

## CUDA extensions

`gaussian_renderer/__init__.py` imports both RaDe-GS and GOF at module level, so **both** must
be built even with one rasteriser selected.

| Module | Source | Note |
|---|---|---|
| `diff_gaussian_rasterization_radegs` | `submodules/diff-gaussian-rasterization-radegs` | **renamed** (patch 8): upstream installs as `diff_gaussian_rasterization`, SuGaR's module name |
| `diff_gaussian_rasterization_gof` | `submodules/diff-gaussian-rasterization_gof` | unique name |
| `diff_gaussian_rasterization_ms`, `fused_ssim`, `tetranerf` | GW's `submodules/` | shared; MILo's copies equal GW's minus our compat edits |
| `simple_knn` | SuGaR's `submodules/simple-knn` | shared; equal minus our `<cfloat>` include |

Both new extensions built **unpatched** on torch 2.9.1+cu130 / `cuda/13.0.2` / sm_120
(2026-10-24, job 47451128): they use `Tensor.data<T>()`, deprecated but not removed in 2.9, and
no `cub::Max`. `SKIP_TETRA=1` removes MILo as well as GW pivot extraction; `validate()` checks
`tetranerf` before training, since upstream only warns and then dies at the first Delaunay.

## Measured (2026-10-24, `UF_Herp_87980_cranium`, COLMAP input, rtx6000)

| Run | Wall time | Peak RSS | Mesh | Background vertices |
|---|---|---|---|---|
| defaults (`regular`, sdf) | 4,952 s (train 4,378 incl. 67 min mesh loop; extract 574) | 56 GiB | 2.88 M faces | 0.01 % |
| `--lambda_mask 0` | 3,953 s | 56 GiB | 2.50 M faces | **1.63 %** |

Background = vertices on a zero mask pixel in ≥ half of 48 views that see them (PGSR 0 %,
GW 0.00 %, 2DGS 0.02 % on the same scene). The mask loss is what keeps background geometry out
of the Delaunay mesh. RSS is dominated by holding every decoded view, so `--mem 64gb` leaves
little headroom. Compare: colmap-2dgs 1186 s, -pgsr 2560 s, -gw 2111 s, -geosvr 3404 s.

## Licence

The repo root carries the Gaussian-Splatting licence (Inria/MPII, non-commercial); `milo/LICENSE`
is Apache-2.0; the rasterisers are under the Gaussian-Splatting licence. Research-only, the same
footing as the other backends.

## Vendoring

Copied in-tree from https://github.com/Anttwo/MILo at
**`0c38aa65f8b768513964f13246ef1ae54f001c70`** (2026-04-03), not as a submodule.

- **Kept in `milo/`:** the 36 modules reachable from the two trainers, `render.py` and both
  extractors, plus `lpipsPyTorch/modules/` (relative imports), `configs/` and `LICENSE`.
- **Dropped:** `blender/`, `eval/`, `scripts/`, `functional/`, `convert.py`, `metrics.py`,
  `mesh_extract_integration.py`, `render_blender_scene.py`, `gaussian_renderer/pgsr.py`,
  `utils/{adjacency,video}_utils.py`; at the root `assets/`, `install.py`, `requirements.txt`,
  `.gitmodules` and the submodules `Depth-Anything-V2`, `nvdiffrast`,
  `diff-gaussian-rasterization_ms`, `fused-ssim`, `simple-knn`, `tetra_triangulation`; and
  `third_party/glm/{doc,test}` from both rasterisers.
- Upstream's `requirements.txt` is not installed: everything MILo imports was already in the env
  (it would also pull plain `opencv-python`).

### `LOCAL PATCH (augenblick)` sites — re-apply on any resync

| # | File | Change | Why |
|---|---|---|---|
| 1 | `scene/dataset_readers.py` `CameraInfo`, `readColmapCameras` | `mask_path` field; resolve `masks/<image_name>.png`; print `Loaded N masks` | upstream takes masks only from an RGBA alpha |
| 2 | `utils/camera_utils.py` `loadCam` | read the mask at camera resolution; composite GT onto the background | one composite for loss, GT export and report; no full-res RGBA decode (~45 GB at 432 × 24 MP) |
| 3 | `scene/dataset_readers.py` `readColmapSceneInfo` | `split.json` under `--eval`, filtered over the sorted list | shared held-out record; keeps positional render order |
| 4 | `scene/dataset_readers.py` `readColmapSceneInfo`, `scene/__init__.py` | read an existing `points3D.ply`, else build in memory; `Scene` writes `input.ply` itself | upstream rewrote `<scene>/sparse/0/points3D.ply` on every load (races, clobbers hull clouds) |
| 5 | `arguments/__init__.py` `read_config` | a flag given on the CLI beats `configs/fast` | upstream overwrote every parsed value, so `--iterations 2000` trained 18000 |
| 6 | both trainers, `mesh_extract_sdf.py` | mesh config ending `.yaml` is a path | custom schedules outside the vendored tree |
| 7 | `arguments/__init__.py` `OptimizationParams`, both trainers | `lambda_mask` + alpha-vs-mask L1 | mask parity with 2DGS/GW |
| 8 | `submodules/diff-gaussian-rasterization-radegs/` (dir, package, `setup.py`), `gaussian_renderer/radegs.py`, `gaussian_renderer/__init__.py` | rename to `diff_gaussian_rasterization_radegs` | the upstream name would replace SuGaR's rasteriser |
| 9 | both trainers | drop `network_gui.init` and the random port | array tasks on one node could collide on the bound port |
| 10 | `utils/regular_tsdf_utils.py` `post_process_mesh` | clamp the cluster index to the cluster count | `--num_cluster 50` on a 7-component specimen mesh raised `IndexError` (2DGS carries the same unpatched function) |

`tests/test_recon_argv.py::test_milo_patches_present` guards these by grep.
