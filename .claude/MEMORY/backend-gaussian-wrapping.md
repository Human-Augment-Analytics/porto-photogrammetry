# `src/libs/gaussian_wrapping/` — Gaussian Wrapping ("Blobs to Spokes")

Watertight, textured surface meshes by treating 3D Gaussians as stochastic oriented surface
elements (Gomez et al., 2026, arXiv:2604.07337). Rasterizer backends: `ours` (median-depth),
`radegs`, `sof`; SDF modes: `ours`, `exact_computation`. The canonical repo path is
`--rasterizer ours` + `--sdf_mode ours`.

> **Layout:** internal modules import as `from scene.gaussian_model import ...` — i.e.
> `src/libs/gaussian_wrapping/` is the package root. the `gw` backend calls each script by absolute path
> with **no `cwd`**; Python prepends the script's own directory to `sys.path`, so the imports
> resolve.

## Three stages (orchestrated by `augenblick recon gw`)

1. **Training** (`train.py`) — hardcoded `--rasterizer ours`, `--exposure_compensation`,
   `--data_device cpu`, `--N_max_gaussians 6000000`. Multi-view NCC + geometric consistency
   losses, normal-field densification (iters ~22k–26k), depth-normal regularization
   (`mask_depth_normal=True` auto-set when `--rasterizer ours`).
2. **Pivot-based mesh extraction** (`pivot_based_mesh_extraction.py`) — hardcoded
   `--sdf_mode ours`, `--dtype int32`, `--use_valid_mask`, `--isosurface_value 0.0`,
   `--n_binary_steps 10`. Marching tetrahedra over a Delaunay tetrahedralisation of pivot
   points, binary-search refinement, optional `--postprocess` (default on, strips floaters) and
   `--filter_large_edges` (default on).
3. **Texture refinement** (`texture_mesh.py`) — bakes per-vertex colours from rendered Gaussian
   views via L1 + fused-SSIM, 1000 iters by default, `sh_degree_for_texturing=0`.

Mesh filename convention: `mesh_{sdf_mode}_{n_pivots}pivots[_post].ply` under `<output_dir>`
(the model_path); the texture stage appends `_texture_refined_{iter}`. The driver derives these
via `get_mesh_path()` / `get_textured_mesh_path()`.

## Output filenames

```
<output_dir>/
├── point_cloud/iteration_<N>/point_cloud.ply   # trained Gaussians
├── cfg_args, time.txt, cameras.json, input.ply # scene metadata
├── mesh_ours_2pivots.ply                       # raw pivot-based mesh
├── mesh_ours_2pivots_post.ply                  # post-processed (floaters stripped)
└── mesh_ours_2pivots_post_texture_refined_<iter-1>.ply
```

`<iter-1>` because `texture_mesh.py` writes the iteration *index*, not the count
(`texture_n_iter=1000` → suffix `_999`).

## Key components

| Path | Role |
|------|------|
| `train.py` | Full optimisation loop; `__main__` CLI selects the rasterizer and loads YAML configs for multiview / MILo / depth-order / normal-field |
| `pivot_based_mesh_extraction.py` | `marching_tetrahedra_with_binary_search()`; SDF mode dispatches to `integrate_ours` or SOF transmittance; `compute_valid_mask` reprojects pivots through every camera and ANDs `gt_mask` where present |
| `texture_mesh.py` | Optimises `_verts_colors` against rendered views; rasterisation via `ScalableMeshRenderer` / `MeshRenderer` (nvdiffrast) |
| `primal_adaptive_meshing_extraction.py` | Alternative extraction: sample candidates from an existing mesh, refine onto the occupancy isosurface, Delaunay reconstruct. `--bounding_box_method {scene,ground_truth,blender}` |
| `scripts/train_and_extract_gw_{ours,radegs}.py` | Upstream end-to-end drivers (`augenblick recon gw` is the configurable local wrapper) |
| `scripts/benchmark_{tnt,mip360}_gw_{ours,radegs}.py` | Dataset batch benchmarks |
| `scene/gaussian_model.py` | `GaussianModel` with `learn_occupancy`, `n_pivots_per_gaussian`, 3D Mip filter, exposure compensation, `densify_and_prune_radegs` |
| `scene/__init__.py` | COLMAP vs Blender auto-detect; checkpoint load or `create_from_pcd` |
| `scene/mesh.py` | `Meshes`, `MeshRasterizer`, `MeshRenderer`, `ScalableMeshRenderer`, QEM utils, `return_delaunay_tets(method="tetranerf")` |
| `gaussian_renderer/ours.py` | `render_ours`, `integrate_ours`, `sample_depth_with_ours` (backed by `diff_gaussian_rasterization_gw`) |
| `gaussian_renderer/radegs.py` | `render_radegs`, `integrate_radegs`; top-level import `try/except`-guarded |
| `gaussian_renderer/sof.py` | `render_sof`, vacancy/transmittance evaluators; only for `--sdf_mode exact_computation` or `--milo` |
| `extraction/pivots.py` | `get_intersecting_pivots_from_normals` (default), `get_pivots_by_scores`, `sample_random_pivots`, `get_searched_pivots` |
| `extraction/mesh.py` | `extract_mesh`, `compute_isosurface_value_from_depth` |
| `regularization/sdf/learnable.py` | `refine_intersections_with_binary_search`, SDF↔occupancy conversions |
| `regularization/sdf/depth_fusion.py` | `AdaptiveTSDF`, `evaluate_mesh_colors_all_vertices`, `frustum_cull_mesh` |
| `regularization/regularizer/multiview.py` | NCC patch matching + geometric consistency (`--multiview` defaults True) |
| `regularization/regularizer/mesh_in_the_loop.py` | MILo depth/normal/occupancy losses vs. a Delaunay mesh rebuilt every `reset_delaunay_every` iters |
| `regularization/regularizer/normal_field.py` | Normal-field init, regularization, densification, non-maximal pruning |
| `regularization/regularizer/depth_order.py` | Depth-Anything-V2 supervision; off unless `--depth_order` |
| `arguments/__init__.py` | `ModelParams`, `PipelineParams`, `OptimizationParams`; `get_combined_args` merges CLI with `cfg_args` from `model_path` |
| `configs/` | YAML presets: `mesh_in_the_loop/`, `multiview/`, `normal_field/`, `depth_order/`, `mesh/` |

## CUDA/C++ submodules (`submodules/`)

| Submodule | Purpose | When loaded |
|-----------|---------|-------------|
| `diff-gaussian-rasterization-gw` | Median-depth rasterizer for `render_ours` / `integrate_ours` | Always on the `ours` path |
| `diff-gaussian-rasterization-ms` | Mini-Splatting2; fused-SSIM `_C` binding + `render_depth`/`render_simp` | Always (top-level import in `utils/loss_utils.py`) |
| `fused-ssim` | Fast SSIM for the photometric loss | Always |
| `warp-patch-ncc` | NCC patch matching for the multiview regularizer | Always |
| `tetra_triangulation` | CGAL Delaunay tetrahedralisation | Pivot extraction stage |
| `nvdiffrast` (vendored) | Mesh rasterisation behind `MeshRasterizer` | Texture refinement |
| `Depth-Anything-V2` (git submodule) | Monocular depth prior | Only with `--depth_order` |
| `diff-gaussian-rasterization` (RaDe-GS) | RaDe-GS rasterizer | Only `--rasterizer radegs`; guarded import |
| `diff-gaussian-rasterization-sof` | SOF transmittance rasterizer | Only `--sdf_mode exact_computation` / `--milo` |

`setup_common.sh` builds the first four plus `tetra_triangulation` (skippable with
`SKIP_TETRA=1`). **MILo reuses three of these builds** — `diff-gaussian-rasterization-ms`,
`fused-ssim` and `tetra_triangulation` (MILo's copies are the same code without our compat
edits) — so a change or fix here applies to both backends, and `SKIP_TETRA=1` disables MILo too
([backend-milo.md](backend-milo.md)). RaDe-GS, SOF, and Depth-Anything-V2 are **not** built in the canonical install;
their imports are `try/except`-guarded or behind `if args.rasterizer == "radegs":`, so this is
safe on the `ours` path.

## Status note

`ed5c733` stripped Gaussian Wrapping from the README; `0ed266e` restored it. The current README
documents GW, and the `gw` backend plus its library are live parts of the pipeline. (The old
`gaussian-wrapping-inclusion.md` audit doc was never committed and no longer exists.)
