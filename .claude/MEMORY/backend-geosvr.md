# `src/libs/geosvr/` — GeoSVR (sparse-voxel surface reconstruction)

GeoSVR (Li et al., NeurIPS 2025, `li2025geosvr`) is built on NVIDIA's SVRaster: the scene is a
sparse-voxel octree drawn by the custom CUDA extension `svraster_cuda`, not Gaussians. It adds an
uncertainty-weighted Depth-Anything-V2 depth prior and voxel-surface regularisers (PGSR's
multi-view NCC + geometric losses, surface rectification, scaling penalty). The mesh comes from
TSDF-fusing rendered depths. SfM points set **only the scene bound** (`bound_mode: pcd`); they
never seed primitives.

Wrapper: `augenblick recon geosvr` (`src/augenblick/reconstruction/geosvr.py`); flags in
[pipeline-reconstruction.md](pipeline-reconstruction.md).

## Config: yacs, preset + flags

`src/config.py` defaults ← `--cfg_files` yamls ← CLI flags. The wrapper loads
`cfg/<preset>_mesh.yaml` (default `dtu`, the only object-centric preset: `outside_level: 1`,
`bound_mode: pcd`, black background) and then passes **every** exposed field as a flag, bools as
explicit `--x`/`--no-x`. It always passes `--blend_mask` (DTU turns it off because DTU has no
masks), `--[no-]eval`, and `--res_downscale` (`-r -1` → long side / 1600, matching the other
backends; upstream's own default caps at 1024² pixels). Upstream dumps the resolved config to
`<output>/config.yaml`, which `render.py` and the TSDF scripts read back (scene path included).

Upstream's CLI had two traps, both fixed in `src/config.py` (patch 4 below):
- every bool was `store_true` (`'store_true' if t` tests the *type*), so a `True` default such
  as `blend_mask` or `enable_da2_uncertainty` could not be turned off;
- a flag was applied only if it differed from the `src/config.py` default, so a preset yaml
  silently beat an explicit flag equal to that default.

`procedure.sche_mult` scales `n_iter` (20k at 1.0) and most schedule fields, but **not**
`rectifiy_from`, `scaling_penalty_*`, `W_enlarge_from` or `level_uncertainty_from`: a `0.1`
smoke run never reaches `rectifiy_from: 5000`.

## Masks and the alpha loss

`--lambda_mask` (wrapper default 0.1, upstream 0) is an L2 between rendered transmittance `T`
and `1 - mask`, on for parity with 2DGS/GW, which supervise alpha as a 4th L1+SSIM channel.
Upstream reads `cam.mask.cuda()` unguarded, so `validate()` refuses a scene where any image lacks
`masks/<stem>.png` while `lambda_mask > 0`.

## Outputs

Everything goes under the output dir; the input scene is read in place and never written to.

| Path | What |
|---|---|
| `config.yaml` | resolved config (`train.py`'s first write) |
| `checkpoints/iter<n>_model.pt` | voxel model |
| `mono_priors/depthanythingv2/` | cached DA2 depths (patched; upstream wrote into the scene) |
| `train/ours_<iter>_r<res_downscale>/renders` | train-view renders; the TSDF scripts read **colour** from here |
| `test/ours_<iter>_r<res_downscale>/{renders,gt}` | held-out renders with `--eval`; note the suffix |
| `mesh/tsdf/tsdf_fusion_post.ply` | final mesh (`mesh_path()`) |

`--tsdf auto` (default) runs `tsdf_mesh_360.py`, which sizes the TSDF from the cameras
(`max_depth = radius_scale × min camera-to-focus distance`, `voxel = max_depth / 1536`), so it is
scale-free. `--tsdf fixed` runs `tsdf_mesh.py` with absolute `--voxel_size`/`--max_depth`, which
depend on the SfM's arbitrary scale.

## Runtime gotchas

- **The DA2 prior must be masked on masked scenes** (patch 7). Unpatched, PSNR on
  UF_Herp_87980_cranium reached 30 at iteration 1000, then collapsed to 17 once the prior started,
  on the full schedule too. With the mask, smoke training stays at ~26, about 3 dB under a
  no-prior run (`--lambda_depthanythingv2 0`); whether the prior helps the *mesh* is untested.

- **`render.py` must run before meshing, on every run.** The TSDF scripts render depth
  themselves but load colour from `render.py`'s `train/` renders; without them Open3D fails with
  "RGB image size (0 0)". The wrapper renders with `--skip_test`, or both splits under `--eval`.

- **Host RAM:** upstream keeps every decoded image and mask in memory. 432 × 24 MP views
  peaked at ~71 GiB, so `recon.sbatch`'s 64 GB is OOM-killed; submit with `--mem=128gb`.

- **Every script needs a GPU even for `--help`:** `svraster_cuda/marching_cubes.py` allocates a
  CUDA tensor at import, so imports fail on a login node with "Found no NVIDIA driver".
- **Depth-Anything-V2-Large** (`depth-anything/Depth-Anything-V2-Large-hf`, 1.3 GB,
  CC-BY-NC-4.0) is fetched by `transformers` on the first run with `lambda_depthanythingv2 > 0`.
  It is pre-fetched into the `/blue` `HF_HOME` that `hpg_slurm/common.sh` exports.
- **Multi-view neighbours are scale-dependent:** `data_pack.py` keeps neighbours within an
  absolute `multi_view_max_dis = 1.5`, and `multi_view_num = min(...)` shrinks for *all*
  following cameras once one camera has fewer neighbours (upstream behaviour, unpatched).
- **The DA2 prior has two consumers.** The DTU preset's mono-normal loss (`lambda_normal_dmono`)
  also reads the priors, so `--lambda_depthanythingv2 0` makes the wrapper pass
  `--lambda_normal_dmono 0` too; otherwise training crashes when that loss starts.
- PINHOLE / SIMPLE_PINHOLE cameras only; all five SfM methods emit one of these.

## Licence

Code: NVIDIA Source Code License for SVRaster (§3.3: non-commercial research/evaluation only),
plus `LICENSE_inria.md` for files taken from 3DGS. DA2-Large weights: CC-BY-NC-4.0.

## Vendoring

Upstream `https://github.com/Fictionarry/GeoSVR`, commit
`44bf7e74471e4cbeb2a11a7f86870208a2861bd1` (2026-01-02), vendored in-tree (not a submodule).

**Dropped** (not on our path): `asset/`, `scripts/` (DTU/TnT/360 benchmark drivers; the TnT one
wants `open3d==0.9.0`), `eval.py`, `render_fly_through.py`, `render_mesh.py`, `environment.yml`.

**Local patches**, each marked `LOCAL PATCH (augenblick)`; re-apply all of them on any resync.
`tests/test_recon_argv.py::test_geosvr_patches_present` and `test_geosvr_config_cli_patch` fail
if 3–7 are lost.

| # | File · function | Change | Why |
|---|---|---|---|
| 1 | `src/dataloader/reader_colmap_dataset.py` · `read_cameras_from_colmap`; `src/dataloader/data_pack.py` · `instantiate_a_camera` | load `masks/<stem>.png` beside `images/` into `cam_info.mask` and print `Loaded N masks`; in `instantiate_a_camera`'s non-RGBA branch, set `mask = None` instead of `cam_info.mask` | upstream hard-coded `mask = None`. Use a separate field, not `putalpha`, because `DataPack` rejects alpha + mask. The non-RGBA branch was unreachable upstream, and copying the mask there made the dedicated mask block raise "Duplicated mask" |
| 2 | same file · `read_colmap_dataset` | `split.json` partitions train/test when present | upstream used `idx % test_every`; keeps the held-out set identical across backends |
| 3 | `train.py` (DA2 + MASt3R calls), `render.py` (DA2 call), `src/dataloader/data_pack.py`, `src/dataloader/colmap_loader.py` · `read_colmap_ply` | priors cached under `model_path`; `multi_view.json` write dropped, in-memory neighbours kept; `points3D.ply` + `points_correspondent.json` built in memory, not cached in `sparse/0/` | upstream wrote all of these into the input scene; nothing reads `multi_view.json`, and the ply/json caches were only reloaded by `read_colmap_ply` itself |
| 4 | `src/config.py` · `update_argparser`, `update_config` | bools use `BooleanOptionalAction`; `update_config` parses with `argument_default=SUPPRESS` (fields registered without defaults) and applies every flag given | the two CLI traps above. `set_defaults(k=SUPPRESS)` does **not** work: argparse type-converts the sentinel string |
| 5 | `src/config.py` (`SHORT_FLAGS`), `render.py`, `mesh_extract/tsdf_mesh{,_360}.py` | `-s`/`-m` aliases for training; `-m/--model_path` replaces the positional; mesh scripts put the repo root on `sys.path` | invoke like every other backend: `python <abs script> -s … -m …`, no `PYTHONPATH` |
| 6 | `train.py` · `training_report` (pg_view) | skip the mono-normal panel when the camera has no DA2 prior | upstream read `view.depthanythingv2` unconditionally, so any run with `lambda_depthanythingv2 0` crashed at iteration 1 |
| 7 | `src/utils/loss_utils.py` · `outside_specimen`, `DepthAnythingv2UncertaintyLoss`, `DepthAnythingv2Loss_old` | the DA2 depth loss also ignores pixels outside the segmentation mask | on composited black backgrounds DA2 hallucinates a surface; supervising it there grew an opaque black blob in front of the cameras and collapsed training (PSNR 30 → 17) |

`svraster_cuda` built unpatched on torch 2.9.1+cu130 / `cuda/13.0.2` / sm_120 (2026-10-15).
