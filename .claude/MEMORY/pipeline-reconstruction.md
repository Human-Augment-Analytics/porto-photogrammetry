# Reconstruction Backends (`augenblick recon`)

Six backends, each taking `--scene` and `--output`, orchestrating the underlying `src/libs/`
train + render scripts via `subprocess.run()` with a banner + per-stage + total-time summary.
They are `SubprocessBackend` subclasses; the class machinery is documented in
[augenblick-package.md](augenblick-package.md).

```bash
# SuGaR: vanilla 3DGS → coarse SuGaR → mesh → refine → textured mesh (.obj)
augenblick recon sugar --scene <scene_dir> --output <output_dir> \
    [--gs_iterations 20000] [--gs_densify_grad_threshold] [--gs_densify_until_iter] \
    [--gs_lambda_dssim] [--gs_sh_degree] [--iteration_to_load 7000] \
    [--regularization dn_consistency] [--surface_level] [--n_vertices] \
    [--gaussians_per_triangle] [--refinement_iterations] [--low_poly] [--high_poly] \
    [--refinement_time long] [--square_size] [--postprocess_mesh] [--white_background] [--gpu]

# 2DGS: training + TSDF mesh extraction
augenblick recon 2dgs --scene <scene_dir> --output <output_dir> \
    [-r -1] [--eval] [--iterations 30000] [--test_iterations ...] [--save_iterations ...] \
    [--lambda_dist] [--lambda_normal] [--depth_ratio] \
    [--densify_grad_threshold] [--densify_until_iter] [--opacity_cull] [--white_background] \
    [--voxel_size -1.0] [--depth_trunc -1.0] [--sdf_trunc -1.0] [--num_cluster 50] \
    [--unbounded] [--mesh_res 1024] [--skip_mesh] [--skip_train_export] \
    [--lambda_mask_hull 0] [--mask_dilate_kernel 9] \
    [--hull_sdf_path <grid.npz>] [--lambda_hull 0] [--hull_anneal_start_iter 500] [--hull_anneal_end_iter 15000]

# PGSR: links scene, flattens sparse/0/ → sparse/, trains, TSDF mesh extraction
augenblick recon pgsr --scene <scene_dir> --output <output_dir> \
    [-r -1] [--eval] [--test_iterations ...] [--save_iterations ...] \
    [--iterations 30000] [--max_abs_split_points 0] [--opacity_cull_threshold 0.05] \
    [--lambda_dssim] [--single_view_weight] [--multi_view_ncc_weight] [--multi_view_geo_weight] \
    [--multi_view_num] [--densify_grad_threshold] [--densify_until_iter] [--white_background] \
    [--max_depth 10.0] [--voxel_size 0.001] [--num_cluster] [--use_depth_filter] [--skip_mesh]

# Gaussian Wrapping: train (--rasterizer ours) → pivot mesh extraction → texture refinement
augenblick recon gw --scene <scene_dir> --output <output_dir> \
    [-r -1] [--eval] [--iterations 30000] [--sh_degree 3] [--max_gaussians 6000000] \
    [--densify_until_iter] [--densify_grad_threshold] [--lambda_depth_normal] \
    [--multiview_factor] [--<many>_lr ...] [--extract_iteration] \
    [--n_pivots 2] [--std_factor 3.0] [--n_binary_steps 10] [--isosurface_value 0.0] \
    [--use_searched_pivots] [--use_smallest_axis_as_normal] \
    [--no-postprocess] [--no-filter_large_edges] \
    [--texture_n_iter 1000] [--texture_lambda_dssim] [--texture_lr 0.0025] [--texture_sh_degree]

# GeoSVR: sparse-voxel training (yacs preset + flags) → render → TSDF mesh
augenblick recon geosvr --scene <scene_dir> --output <output_dir> \
    [--preset dtu|tnt|none] [--sche_mult 1.0] [--lambda_depthanythingv2 0.1] \
    [--no-da2_uncertainty] [--power_level_uncertainty 1.0] [--voxel_dropout_min 0.5] \
    [--multi_view_ncc_weight 0.05] [--multi_view_geo_weight 0.01] \
    [--lambda_rectify 1e-5] [--lambda_scaling_penalty 1e-6] [--lambda_mask 0.5] \
    [--white_background] [--tsdf fixed|auto] [--radius_scale 2.0] \
    [--voxel_size 0.001] [--max_depth 5.0] [--sdf_trunc_scale 4.0] \
    [--num_cluster 1] [--use_depth_filter] [-r -1] [--eval]

# MILo: Gaussians with a Delaunay mesh extracted inside training → SDF-refined mesh
augenblick recon milo --scene <scene_dir> --output <output_dir> \
    [--densification regular|fast] [--iterations] [--densify_until_iter] \
    [--regularization_from_iter] [--mesh_config <name|path.yaml>] [--rasterizer radegs|gof] \
    [--imp_metric indoor|outdoor] [--dense_gaussians] [--decoupled_appearance] \
    [--lambda_mask 0.1] [--white_background] [--extract sdf|tsdf] [--refine_iter 1000] \
    [--remove_oof_vertices] [--mesh_res 4096] [--num_cluster 50] [-r -1] [--eval]
```

## Wrapper-specific behaviour

- **SuGaR / 2DGS / PGSR / GeoSVR / MILo** set `cwd` to the backend source dir (`use_cwd = True`;
  MILo's is `src/libs/milo/milo/`).
- **`-r` / `--eval`** come from the shared `ResolutionParams` / `EvalParams` mixins on every
  backend except SuGaR. `-r -1` caps the long side at 1600 px (GeoSVR maps it onto its own
  `res_downscale`); `--eval` writes or reuses `split.json` and scores held-out views at the end
  (see [pipeline-evaluation.md](pipeline-evaluation.md)).
- **2DGS** `--skip_train_export` skips the per-training-view PNG dump (tens of GB at full
  resolution); meshing is unaffected. The `*_hull` flags are local loss terms, off by default —
  see [backend-2dgs.md](backend-2dgs.md).
- **PGSR** builds `<output>/scene/` in its `prepare()` hook: `images/`/`masks/` symlinked,
  `sparse/0/*` symlinked file-by-file into a flat `sparse/` (PGSR expects no `0/`), and
  `split.json` copied. An existing `<output>/scene/` is reused, with `split.json` re-copied.
- **GW is the exception** (`use_cwd = False`): it invokes `train.py`,
  `pivot_based_mesh_extraction.py`, and `texture_mesh.py` under `src/libs/gaussian_wrapping/` by
  **absolute path with no `cwd`**; imports like `from scene.gaussian_model import ...` resolve
  because Python prepends the script's own directory to `sys.path`. Unrecognised flags are
  forwarded **only to the training step** (gated on `accepts_passthrough`). Boolean toggles use
  `argparse.BooleanOptionalAction`, hence the `--no-postprocess` / `--no-filter_large_edges`
  spellings (both default on).
- **GeoSVR** reads the scene in place (no `prepare()`) and writes nothing outside the output dir. It loads a yacs preset (`cfg/dtu_mesh.yaml`), then overrides it with explicit flags, every bool as `--x`/`--no-x`; vendored patches make flags reliably beat the preset. `--lambda_mask` (default 0.5) requires a mask for every image, checked in `validate()`. Default choices: [backend-geosvr.md](backend-geosvr.md). Held-out renders land in `test/ours_<iter>_r<res_downscale>/`, which `find_test_dir` accepts.
- **MILo** reads the scene in place and writes nothing outside the output dir. One field,
  `--densification`, picks the trainer script (`regular` = upstream's object-centric recipe,
  30k; `fast` = the paper's, 18k); the iteration count and mesh config it implies are passed to
  training **and** extraction, which must agree. `--lambda_mask` (default 0.1) needs RaDe-GS and
  a mask per registered image, both checked in `validate()`, as is the `tetranerf` build.
  Details: [backend-milo.md](backend-milo.md).

## Output locations

| Backend | Output | Path |
|---------|--------|------|
| SuGaR | textured mesh `.obj` + `.ply` | `<output>/refined_mesh/<scene>/` |
| 2DGS | triangle mesh | `<model>/train/ours_<iter>/fuse_post.ply` |
| PGSR | triangle mesh | `<model>/mesh/tsdf_fusion_post.ply` |
| GeoSVR | triangle mesh | `<model>/mesh/tsdf/tsdf_fusion_post.ply` |
| GW | raw / post / textured mesh | `<output>/mesh_ours_2pivots{,_post,_post_texture_refined_<iter-1>}.ply` |
| MILo | Delaunay SDF mesh (`sdf`) / TSDF mesh (`tsdf`) | `<output>/mesh_learnable_sdf.ply` / `<output>/mesh_regular_tsdf_res<mesh_res>_post.ply` |

Per-backend internals: [backend-sugar.md](backend-sugar.md), [backend-2dgs.md](backend-2dgs.md),
[backend-pgsr.md](backend-pgsr.md), [backend-gaussian-wrapping.md](backend-gaussian-wrapping.md),
[backend-geosvr.md](backend-geosvr.md), [backend-milo.md](backend-milo.md).
