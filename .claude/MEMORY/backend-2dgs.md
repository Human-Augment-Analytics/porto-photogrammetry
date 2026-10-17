# `src/libs/2dgs/` — 2D Gaussian Splatting

2D surfel primitives instead of 3D Gaussians for better surface geometry. Custom CUDA rasterizer
`diff-surfel-rasterization` under `submodules/`.

## Training

```bash
cd src/libs/2dgs
python train.py -s <scene_path> -m <model_output_path> \
    [--iterations 30000] [--test_iterations 7000 30000] [--save_iterations 7000 30000]
```

Differences from vanilla 3DGS:
- **Surfel rasterization** — 2D oriented disks; produces `rend_normal`, `rend_alpha`,
  `surf_depth`, `surf_normal`, `rend_dist`.
- **Normal consistency loss** (`lambda_normal`, default 0.05) — rendered normal vs. the
  depth-derived pseudo-surface normal; enabled after iter 7000.
- **Distortion loss** (`lambda_dist`, default 0.0) — ray-along-distortion regularizer for
  tighter depth distributions; enabled after iter 3000.
- **`depth_ratio`** (PipelineParams, default 0.0) — blends expected (0.0) and median (1.0)
  depth. Median for bounded scenes, expected for unbounded.
- **Mask support** — with `gt_alpha_mask`, background pixels are set to `bg_color` in the GT and
  alpha is concatenated as a 4th supervision channel.

### Local additions: silhouette and hull losses (off by default)

Two terms added in-tree (`eb6d464`), both inert at their defaults:

- **`lambda_mask_hull`** (0) — one-sided loss `relu(alpha - dilate(mask))`: penalises opacity
  *outside* the GT silhouette, max-pool-dilated by `mask_dilate_kernel` (9) so it kills floaters
  without eroding concavities. Needs masks.
- **`lambda_hull`** (0) + **`hull_sdf_path`** — penalises Gaussian centres whose signed distance
  to a precomputed visual-hull SDF grid is positive (outside), weighted by opacity, annealed
  linearly from `lambda_hull` at `hull_anneal_start_iter` (500) to 0 at
  `hull_anneal_end_iter` (15000). The grid is an `.npz` with `sdf [D,H,W]` (z,y,x order) and
  `lo`/`hi` bounds; nothing in-tree writes one (`sfm hull` emits a point cloud and mesh, not an
  SDF), so the term is only reachable with an externally built grid.

Both are exposed on `augenblick recon 2dgs` under the same names.

## Rendering + mesh extraction

```bash
cd src/libs/2dgs
python render.py -m <model_path> \
    [--iteration -1] [--skip_train] [--skip_test] [--skip_mesh] \
    [--voxel_size -1.0]   \  # auto if negative
    [--depth_trunc -1.0]  \  # auto = 2 * bounding-sphere radius
    [--sdf_trunc -1.0]    \  # auto = 5 * voxel_size
    [--num_cluster 50] [--unbounded] [--mesh_res 1024] [--render_path]
```

`GaussianExtractor` (`utils/mesh_utils.py`):
- **Bounded (default)** — TSDF fusion over rendered depth, bounding sphere auto-estimated from
  camera poses. Writes `fuse.ply` + `fuse_post.ply`.
- **Unbounded (`--unbounded`)** — marching cubes with spatial contraction
  (`utils/mcube_utils.py`). Writes `fuse_unbounded.ply`. Experimental.

## Key parameters (`arguments/__init__.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_normal` | 0.05 | Normal consistency weight |
| `lambda_dist` | 0.0 | Distortion loss weight |
| `opacity_cull` | 0.05 | Opacity pruning threshold |
| `depth_ratio` | 0.0 | Expected (0) vs median (1) depth blend |
| `render_items` | RGB, Alpha, Normal, Depth, Edge, Curvature | GUI-visualisable quantities |
