# Plan: add `milo` as a reconstruction backend

Implementation spec, written before the work lands. Once it lands this file is historical:
where it and the code disagree, the code wins. The durable description belongs in
[pipeline-reconstruction.md](../MEMORY/pipeline-reconstruction.md) and a new
`MEMORY/backend-milo.md`.

**Scope:** the code and the job scripts treat `milo` like any other backend on both clusters.
Installing and verifying it covers only the HPG env `augenblick_rtx_pro_6000`; other envs pick
it up from `setup_common.sh` (`BACKENDS` default) on their next rebuild.

**GPU account:** every HPG GPU job this plan submits (the step 2c build, checks 0, 1b and 9, and
every verification run) uses **`--account=arthur.porto-phenomics`**. The repo's job scripts
hard-code `--account=arthur.porto` (`hpg_slurm/recon.sbatch:9`), so pass the account on the
`sbatch` command line, which overrides the `#SBATCH` line. `recon.sbatch` sets no `--qos`, so
nothing else needs overriding. The shared scripts keep their account; this applies to MILo's
jobs only. PACE assigns its own account (`coc-ice`) and is unaffected.

Researched 2026-10-20 against the tree at `8106524` (plus the uncommitted GeoSVR
`registered_stems` change) and upstream MILo at
**`0c38aa65f8b768513964f13246ef1ae54f001c70`** (2026-04-03, "Merge pull request #45 from
Anttwo/mesh_eval"). Upstream line references are to that commit, with paths relative to its
`milo/` code dir unless they start with `submodules/`. Facts marked *checked* were run against
the RTX env, our vendored trees or a prototype; everything else was read in the cloned source.

## Status

| Step | State |
|---|---|
| 1 — vendor into `src/libs/milo/` | **Done** 2026-10-24: 39 `.py` + configs from `0c38aa6`; rasterisers without `glm/{doc,test}` |
| 2 — env: build RaDe-GS + GOF rasterisers | **Done**: both built **unpatched** (job 47451128, 1m44s, `cuda/13.0.2`, sm_120); `pip_diff.txt` = the two new dists only |
| 3 — local patches | **Done**: all eight (3a–3h), plus a tenth site found in check 9 (below); 3a/3b/3c/3e exercised on CPU against a synthetic scene and the real 432-view scene |
| 4 — adapter `reconstruction/milo.py` | **Done**, plus a fast-schedule check found in verification (below); `require_registered_masks` lifted into `base.py` |
| 5 — tests | **Done**: 203 passed (17 new MILo/helper tests; GeoSVR mask tests retargeted to `base`) |
| 6 — SLURM + setup scripts | **Done** |
| 7 — docs | **Done** (`backend-milo.md` new; pipeline-reconstruction, augenblick-package, environment-and-gpu, backend-gaussian-wrapping, CLAUDE.md). `repo-conventions.md` "Landed" waits for the commit |
| Verification | **All checks run and pass** except 11's VGGT half (no VGGT scene exists). Visual inspection of the meshes not done |

| # | Result (2026-10-24) |
|---|---|
| 0 | pass: radegs + gof import on a GPU node; SuGaR's `diff_gaussian_rasterization` keeps its vanilla API (no `kernel_size`) |
| 1 / 1b | pass; `--help` exits 0 for both trainers, `render.py` and both extractors on a GPU node |
| 2 | pass: 202/202 |
| 3 | pass: staged `points3D.{bin,ply}` checksums unchanged; only `split.json` (written by `write_split` before training) is new in the scene; subset scene unchanged |
| 4 / 5 | pass: `Loaded 432 masks`; `split.json: 378 train, 54 test` |
| 6 | pass (job 47452067): mesh loop 3501–5000 ran; `mesh_learnable_sdf.ply` 1.49 M faces; Delaunay 5.8 s. Scored separately (job 47455643) after the env fix below: masked **PSNR 24.62, SSIM 0.805, LPIPS 0.167, DISTS 0.129**, `n_test` 54 (smoke schedule; not a quality claim) |
| 6b | pass (job 47456562): the fast recipe trained 5000 iterations, not 18000; no `chkpnt*.pth`; no port errors |
| 7 | pass: no `Downloading` line in any log |
| 8 | pass (job 47456563): **4,952 s** (train 4,378 of which the mesh loop 20,001–30,000 is 67 min; sdf extraction 574); peak RSS **56 GiB** (inside `--mem 64gb`, little headroom); 118 k Gaussians; **2.88 M faces** (PGSR 3.92 M, GW 2.59 M, 2DGS 1.66 M). Peak VRAM not recorded: `recon.sbatch` does not log it |
| 9 | pass after a fix. The TSDF extractor **crashed** in `post_process_mesh` (`--num_cluster 50` vs 7 components) → patch 10. At upstream's `mesh_res 1024` it gave only 98 k faces (voxel 0.0059 vs a 1.21-unit specimen diagonal, because the depth range spans camera distance) → default raised to **4096** (1.70 M faces, 7.6 min). SDF vs TSDF-4096 on shared surface: median 0.25 % / 0.16 % of the diagonal; the SDF mesh carries extra surface (p95 2.3 %) that PGSR and GW lack equally, plausibly closed unobserved regions (unconfirmed without visual inspection). **`sdf` stays the default** |
| 10 | pass (job 47456564): `--lambda_mask 0` → 3,953 s, 2.50 M faces, **1.63 %** of vertices off-silhouette (≥ half of 48 views), vs **0.01 %** at the 0.1 default (PGSR 0, GW 0.00 %, 2DGS 0.02 %). The 0.1 default stays |
| 11 | hull half pass (job 47455685): `Number of points at initialisation: 300000` (the hull cloud); its `points3D.ply` byte-identical after the run. **VGGT half not run**: no VGGT scene exists for any subset specimen (`neurips/UF_Mammals_1199_mandible/vggt` is empty) |

**Found during implementation** (the plan missed these):

- **The plan's fast smoke schedule was invalid.** Under `train.py`, iterations from
  `regularization_from_iter` render with RaDe-GS, which returns no `area_max`; Mini-Splatting2
  densification reads it until `densify_until_iter` (`train.py:361`). `--regularization_from_iter
  2000 --densify_until_iter 3000` therefore died with `KeyError: 'area_max'` at iteration 2000.
  Upstream's fast defaults set both to 3000. `validate()` now refuses
  `regularization_from_iter < densify_until_iter` under `fast` (effective values), and the fast
  smoke runs with the recipe's own 3000/3000. The regular trainer has no such coupling.
- **`--eval` was broken for every backend in the live env**, not by MILo: `main`'s DISTS merge
  added `dists-pytorch==0.1` to `requirements.txt` but the env never got it, so
  `augenblick.eval.metrics` raised `ModuleNotFoundError` after rendering (the smoke's first
  attempt, and 6 pre-existing test errors). Installed the declared pin (dry run: nothing else
  moves; freezes in `$V/_install/pip_freeze_{before,after}_dists.txt`).
- **The adapter's argv was also checked against upstream's real parsers** on CPU: each entry
  script's `__main__` ran up to its first GPU call, with the two unbuilt rasterisers stubbed.
  `-r -1`, `--test_iterations -1` and `--checkpoint_iterations -1` parse as values.
- **Patch 10 / TSDF default** (check 9): `post_process_mesh` indexed `[-num_cluster]` into the
  cluster list and raised when a clean masked mesh had fewer components; and upstream's
  `mesh_res 1024` is far too coarse for turntable captures. Both fixed; 2DGS carries the same
  unpatched `post_process_mesh` (not touched here).
- **Concurrent runs need separate `RESULT_ROOT`s**, not renames: the output dir is fixed
  (`$RESULT_ROOT/<scene>/<sfm>-<backend>`). The fast smoke and check 10 ran under
  `$V/fast/` and `$V/nomask/`, each with a symlink to the staged scene (read-only for runs
  without `--eval`).

## Goal

`augenblick recon milo --scene <sfm> --output <out>` trains MILo on any of the five SfM methods'
COLMAP scenes and writes MILo's own Delaunay mesh (`mesh_extract_sdf.py`). With `--eval` it
also exports held-out renders that `augenblick.eval.nvs` scores like the other backends'.
`--extract tsdf` swaps in MILo's regular-grid TSDF extractor, for a like-for-like comparison
with the 2DGS/PGSR meshes.

Non-goals:
- no new conda env;
- no DTU/T&T evaluation toolboxes, mesh-based NVS (`eval/mesh_nvs/`) or Blender add-on;
- no Depth-Anything depth-order prior (`--depth_order`, not used in the paper);
- no `mesh_extract_integration.py` (GOF opacity-field extraction);
- not GW's `--milo` mode. `src/libs/gaussian_wrapping` carries its own port of MILo's
  regulariser (`regularization/regularizer/mesh_in_the_loop.py`). It runs inside GW's trainer,
  needs the SOF rasteriser we do not build, and is not the published method.

## What MILo is

MILo (Guédon, Gomez et al., SIGGRAPH Asia 2025 / TOG, arXiv 2506.24096, `guedon2025milo`) is
Gaussian splatting with the **mesh extracted inside the training loop**. It shares authors and
code lineage with SuGaR and Gaussian Wrapping.

- **Pivots:** each Gaussian spawns 9 pivot points (centre + 8 box corners); a subset of
  surface Gaussians feeds a CGAL Delaunay tetrahedralisation (`tetranerf`), rebuilt every
  `delaunay_reset_interval` (500) iterations.
- **Learnable SDF:** each pivot carries an occupancy value, stored on the Gaussians
  (`base_occupancy` + learnable `occupancy_shift`). Differentiable marching tetrahedra turns
  pivots + SDF into a mesh **every iteration** of the mesh phase.
- **Losses:** the mesh is rasterised with nvdiffrast; its depth and normals are pulled toward the
  Gaussians' (RaDe-GS median depth). Gradients flow back into the Gaussians' positions, scales
  and rotations. On top come the usual depth-normal consistency and two occupancy losses.
- **Final mesh:** `mesh_extract_sdf.py` refines the SDF for 1,000 more iterations with the
  Gaussians frozen, then runs marching tetrahedra once more. No TSDF, no regular grid.

It is the only backend whose mesh is **the training-time representation itself**, not a
post-hoc fusion. Hypothesis to test: it is less sensitive than the TSDF backends to SfM point
density (densification replaces the init cloud quickly), and its meshes are much lighter at
equal accuracy (the paper's headline claim).

| | GW (closest existing) | MILo |
|---|---|---|
| Primitive | 3D Gaussians, `ours` rasteriser | 3D Gaussians, **RaDe-GS** (or GOF) rasteriser |
| Densification | GW's own | `regular`: RaDe-GS/GOF style; `fast`: Mini-Splatting2 (step 4) |
| Mesh | pivots → Delaunay → marching tets, **after** training | pivots → Delaunay → marching tets, **inside** training, then refined |
| Default iterations | 30,000 | 30,000 (`regular`), 18,000 (`fast`) |
| Mesh path | `mesh_ours_2pivots_post.ply` | `mesh_learnable_sdf.ply` |
| New CUDA builds | — | `diff_gaussian_rasterization_radegs` (renamed, fact 6), `diff_gaussian_rasterization_gof` |
| Licence | GW, non-commercial | **Gaussian-Splatting licence** (Inria/MPII, non-commercial) at the repo root; `milo/LICENSE` is Apache-2.0 |

Licence, carry into the docs: research-only, the same footing as the existing backends. The
rasterisers are under the same Gaussian-Splatting licence.

## Upstream facts that shape the design

1. **Two trainers, two recipes.**
   - `train.py` is the paper's main (T&T) recipe. It uses Mini-Splatting2 densification (depth
     re-init at 2,000, aggressive cloning, two simplification passes) and runs 18,000 iterations.
     Mesh-in-the-loop runs 8,001–18,000 (`configs/mesh/default.yaml`).
   - `train_regular_densification.py` is upstream's recipe for **object-centric** scenes (DTU,
     `scripts/evaluate_dtu.py`). It uses RaDe-GS densification, 30,000 iterations, depth-normal
     regularisation from 15,000 and mesh-in-the-loop 20,001–30,000. Upstream pairs it with
     `--mesh_config default_dtu --imp_metric indoor`.
   - **Decision:** expose both as `--densification {regular,fast}`, default **`regular`**. Our
     specimens are masked, object-centric turntable captures, which is the DTU case. The same
     reasoning picked GeoSVR's DTU preset.

2. **`train.py` silently overrides CLI flags.** `read_config` (`arguments/__init__.py:122-144`)
   loads `--config_path ./configs/fast` (`train.py:553`, cwd-relative) and copies **every** key
   over the parsed CLI value. `configs/fast` sets `iterations`, `densify_until_iter`,
   `aggressive_clone_*`, `warn_until_iter`, `depth_reinit_iter` and `simp_iteration1/2`, so
   `--iterations 2000` trains 18,000 (*checked*, prototype). `train_regular_densification.py`
   never calls `read_config`, so it has no trap.
   **Decision:** patch `read_config` so a flag given on the command line wins (**3e**).

3. **The points cloud is written into the input scene.** `readColmapSceneInfo`
   (`scene/dataset_readers.py:157-163`) **unconditionally** `storePly`s
   `points3D.bin` into `<source>/sparse/0/points3D.ply`, then reads it back. It runs on every
   `Scene()` build: train, render and both extractors.
   - Against a shared SfM dir that is a write into `<sfm>/` that races between array tasks.
   - For a **hull** scene it overwrites the hull init cloud (`sparse/0/points3D.ply`) with the
     raw SfM points, through the symlink if the scene was staged with links.
   - Every other write goes under `model_path` (*checked*: grep of `open(..'w')`, `.write(`,
     `save_ply`, `export`, `write_triangle_mesh`, `save_image`, `savefig`, `torch.save` across
     the reachable files).
   - **Decision:** read an existing `points3D.ply` (the 3DGS/2DGS behaviour, so the hull cloud
     reaches MILo like the other 3DGS backends), else build the cloud in memory; write nothing
     into the scene (**3c**). There is no `prepare()` copy.

4. **Masks are only taken from an RGBA alpha channel.** `loadCam` (`utils/camera_utils.py:51-55`)
   splits a 4th channel into `gt_alpha_mask` → `Camera.gt_mask`; our JPEGs have none.
   - Training never reads `gt_mask`: the loss is plain L1 + SSIM on the raw image
     (`train_regular_densification.py:197-205`, `train.py:216-224`).
   - `gt_mask` is read by the regular TSDF extractor (`utils/regular_tsdf_utils.py:167-168`,
     zeroes depth outside it) and by `integrate_*` SDF evaluation (`regularization/sdf/integration.py:56`).
   - The Delaunay SDF path (`depth_fusion`, the default) takes a `masks` argument and **ignores
     it** (`regularization/sdf/depth_fusion.py:400-460`), so masks cannot carve the MILo mesh.
   - **Decision:** load `masks/<stem>.png` (**3a**), composite GT onto the background, and add
     a rendered-alpha loss (**3g**), on for parity with 2DGS and GW. The mask then shapes the
     Delaunay mesh only through the Gaussians.

5. **The held-out split is `idx % llffhold`** on cameras sorted by `image_name`
   (`dataset_readers.py:146-150`); `--llff` exists on `ModelParams` but is never passed through.
   `augenblick.eval.split.build_split` reproduces this rule, but a hand-written `split.json`
   would be ignored. **Decision:** read `split.json` under `--eval`, as the other backends do
   (**3b**).

6. **MILo's RaDe-GS rasteriser installs as `diff_gaussian_rasterization`**, the module name of
   SuGaR's vanilla 3DGS rasteriser (`submodules/diff-gaussian-rasterization/setup.py:18-22`).
   - The env's copy is SuGaR's (*checked*: `direct_url.json` points into
     `src/libs/sugar/...`; its `GaussianRasterizationSettings` lacks RaDe-GS's `kernel_size`,
     `require_depth`, `require_coord`).
   - Installing MILo's as-is would silently replace it and break SuGaR.
   - `gaussian_renderer/__init__.py:20-21` imports **both** `.radegs` and `.gof` at module
     level, so every MILo script needs both extensions even with one rasteriser selected.
   - **Decision:** rename RaDe-GS's package to `diff_gaussian_rasterization_radegs` (**3h**).
     GOF's `diff_gaussian_rasterization_gof` is already unique.

7. **The other four extensions are ones we already build.** Compared ignoring `build/`,
   `*.egg-info` and `third_party/` (*checked*):
   - `diff-gaussian-rasterization_ms`, `fused-ssim` and `tetra_triangulation` equal GW's copies
     in `src/libs/gaussian_wrapping/submodules/`, **minus our local compatibility edits**
     (`.data<T>()` → `.data_ptr<T>()`, the CUB 3.0 `cub::Max` shim, tetra's CXX11-ABI and
     nvJitLink fixes). After normalising `.data_ptr`, the `_ms` and `fused-ssim` sources are
     otherwise identical.
   - `simple-knn` equals SuGaR's, minus our `#include <cfloat>`.
   - All four install under global module names (`diff_gaussian_rasterization_ms`,
     `fused_ssim`, `tetranerf`, `simple_knn`), and the env already has them.
   - **Decision:** vendor none of them; reuse the GW and SuGaR builds. MILo then depends on
     `src/libs/gaussian_wrapping` (tetra is an editable install from there) and on
     `SKIP_TETRA` being off.

8. **Training opens a TCP listener.** Both trainers draw a random port in 5000–9000 and call
   `network_gui.init` (`train.py:601-603,644`, `train_regular_densification.py:483-485,526`),
   which `bind`s and `listen`s. Two array tasks on one node can draw the same port and the second
   dies with `Address already in use`. GW and PGSR never bind (`pgsr/train.py:501` is commented
   out; GW has no `init`). **Decision:** drop the `init` call (**3d**). `try_connect` on an
   unbound socket already swallows its exception (`gaussian_renderer/network_gui.py:34-41`).

9. **The final test pass downloads LPIPS weights.** Both trainers append `iterations` to
   `test_iterations` unless it contains `-1` (`train.py:595-596`). `training_report` then
   computes LPIPS-VGG (`utils/log_utils.py:142`) through `lpipsPyTorch`, which fetches VGG16
   and LPIPS weights from the network (`lpipsPyTorch/modules/utils.py:17`). **Decision:** the
   wrapper always passes `--test_iterations -1`; held-out scoring is `augenblick.eval.nvs`'s job.

10. **Checkpoints.** `--checkpoint_iterations` defaults to `[8000]` (`train.py:536`) /
    `[19_999]` (`train_regular_densification.py:430`): a full `capture()` with optimiser state,
    used only for resuming. The extractors read `point_cloud/iteration_<N>/point_cloud.ply`,
    which carries the occupancy fields (`scene/gaussian_model.py` `save_ply`/`load_ply`).
    **Decision:** pass `--checkpoint_iterations -1`.

11. **Extraction must match training, and nothing enforces it.**
    - `mesh_extract_sdf.py --iteration` defaults to a hard-coded **18000** (`:551`), so a
      `regular` model (30,000) needs it passed.
    - `--config` (`:561`) must name the **same** mesh config as training (README §3.1 warning).
    - The adapter derives both from one field each, so they agree by construction.
    - `--imp_metric` (`:559`) is used only when there are more Gaussians than Delaunay sites and
      no learned occupancy; the wrapper passes it anyway.

12. **`mesh_extract_sdf.py` after `train_regular_densification.py` is not an upstream
    combination.** Upstream's DTU script meshes with `eval/dtu/mesh_extract_dtu.py`, a TSDF with
    **absolute DTU-scale bounds** (`voxel_size 0.002`, depth 1.0–8.0), unusable at our scales.
    `mesh_extract_regular_tsdf.py` is scale-free: `depth_trunc = radius_factor × min
    camera-to-centre distance`, `voxel = depth_trunc / mesh_res`, and it honours `gt_mask`.
    **Decision:** default `--extract sdf` (the method under study); `--extract tsdf` uses the
    scale-free script. Check 9 compares them.

13. **Camera scales differ between trainers.** `train.py` builds cameras at scales `[1, 2]`
    (`:68`) and uses the half-res set until `warn_until_iter` (3,000). `train_regular_densification.py`
    builds only `[1]` (`:69`), but the Gaussian sampling helpers it calls at mesh start still ask
    `getTrainCameras_warn_up(iteration, warn_until_iter, …)` (`scene/gaussian_model.py:1074-1491`).
    So under `regular`, **mesh-in-the-loop must start after `warn_until_iter`** or the lookup
    hits the missing scale-2 set (`KeyError`). That is harmless at defaults (20,001 > 3,000) and
    constrains the smoke schedule (Verification).

14. **Render file names are positional** (`render.py:33-34`, `00000.png`), from a `Scene` built
    with `shuffle=False`, so test cameras keep the reader's sorted-by-name order.
    `nvs.pair_views` already maps positional names onto the sorted held-out stems (GW renders
    the same way), so no eval change is needed as long as 3b keeps that order. Renders land in
    `<model>/test/ours_<iter>/{renders,gt}/`, which `find_test_dir` reads.

15. **`render_gof` returns `"mask": None`** (`gaussian_renderer/gof.py:125`); only RaDe-GS
    returns rendered alpha (`radegs.py:91`). In `train.py`, iterations before
    `regularization_from_iter` render with Mini-Splatting2's `render_imp`, which returns no alpha
    either (`gaussian_renderer/__init__.py`, `render_imp`). So the mask loss (3g) needs RaDe-GS,
    and under `fast` it starts at iteration 3,000.

16. **The mesh renderer defaults to an OpenGL context** (`scene/mesh.py:172,193-196`,
    `use_opengl=True` → `dr.RasterizeGLContext`). GW's identical `MeshRasterizer` (same default,
    `src/libs/gaussian_wrapping/scene/mesh.py:1059`) runs on HPG rtx6000 nodes: `colmap-gw`'s
    texture stage wrote `mesh_ours_2pivots_post_texture_refined_999.ply` (*checked*). So no patch;
    PACE is unverified.

17. **PINHOLE / SIMPLE_PINHOLE only** (`dataset_readers.py:85-95`); all five SfM methods emit
    these (GeoSVR plan, fact 8). The points loader has **no** error/track-length filter
    (`colmap_loader.py` `read_points3D_binary`, the `track_length<3` test is commented out), so
    VGGT clouds load whole, unlike upstream PGSR.

18. **The script interface already matches the other 3DGS backends.** Training takes
    `-s <scene> -m <model>`; `render.py` and both extractors take `-m` and read `source_path`,
    `eval`, `resolution`, `white_background` and `data_device` back from `<model>/cfg_args`
    (`get_combined_args`, `arguments/__init__.py:98-120`). An `--eval` run therefore extracts
    from training cameras only. `train.py` appends the repo root and `submodules/` to `sys.path`
    itself (`:6-11`); the other scripts import from their own directory. Run with
    cwd = `milo/` (`use_cwd`), `configs/fast` resolves.

## Environment

### Python deps: none new

Imports reachable from the five entry scripts (an AST walk, *checked*), against the RTX env:

| Import | In env? |
|---|---|
| `torch` 2.9.1+cu130, `torchvision`, `numpy` 2.2.6, `PIL`, `plyfile`, `yaml`, `tqdm`, `matplotlib`, `open3d` 0.19.0, `trimesh` | yes |
| `trimesh` | 5.0.0 vs upstream's `4.6.8` pin; the calls `mesh_extract_sdf.py` makes (`Trimesh(..., process=False)`, `update_vertices`, `update_faces`, `export`) work on 5.0.0 (*checked*) |
| `nvdiffrast` | 0.4.0 (stage 6 install); upstream pins a submodule commit |
| `fused_ssim`, `diff_gaussian_rasterization_ms`, `tetranerf`, `simple_knn` | yes (GW / SuGaR builds, fact 7) |
| `diff_gaussian_rasterization_radegs`, `diff_gaussian_rasterization_gof` | **no**: built in step 2 |
| `wandb` | no, and not needed: both imports are `try/except` (`train.py:27-31`, `utils/log_utils.py:11-14`) |
| `depth_anything_v2` | no, and not needed: imported only inside `initialize_depth_order_supervision` (`regularization/regularizer/depth_order.py:30`), i.e. with `--depth_order` |

LPIPS comes from the vendored `lpipsPyTorch/`, not the `lpips` package. Upstream's
`requirements.txt` (`open3d`, `trimesh`, `scikit-image`, `opencv-python`, `plyfile`, `tqdm`) adds
nothing we lack; **do not** install it (it would pull plain `opencv-python`, see
environment-and-gpu.md). `requirements.txt` does not change.

### CUDA extensions

Two new builds, both through `setup_common.sh`'s `build()`:

- `src/libs/milo/submodules/diff-gaussian-rasterization-radegs` (renamed, 3h);
- `src/libs/milo/submodules/diff-gaussian-rasterization_gof`.

Neither has a `cub::Max` (the CUB 3.0 break GW's `_ms` needed a shim for). Both still use
`Tensor.data<T>()` (122 sites, *checked*), which torch 2.9.1 marks **deprecated, not removed**
(`ATen/core/TensorBody.h:246`), so they should build with warnings. On torch 2.9.1 / CUDA 13 /
sm_120 that is **unverified**; fix failures as `LOCAL PATCH`es, with GW's `_ms` diff as the
template.

## Implementation

Steps in dependency order. Run on a login node unless `nvcc` or a GPU is needed.

### Step 1 — vendor into `src/libs/milo/`

Copy the tree at `0c38aa6` **in-tree, not as a submodule**, keeping upstream's two-level
layout, so a resync is a plain diff:

```
src/libs/milo/
├── LICENSE.md, README.md
├── milo/                                    # backend_dir
└── submodules/
    ├── diff-gaussian-rasterization-radegs/  # renamed from diff-gaussian-rasterization (3h)
    └── diff-gaussian-rasterization_gof/
```

- **Keep in `milo/`:** the 36 modules reachable from `train.py`,
  `train_regular_densification.py`, `render.py`, `mesh_extract_sdf.py` and
  `mesh_extract_regular_tsdf.py` (AST walk, *checked*), plus `lpipsPyTorch/modules/` (reached
  by relative import), `configs/` (all of `mesh/`, `fast`, `depth_order/`) and `LICENSE`.
- **Drop from `milo/`:** `blender/`, `eval/`, `scripts/`, `functional/` (the library API, not on
  our path), `convert.py`, `metrics.py`, `mesh_extract_integration.py`,
  `render_blender_scene.py`, `gaussian_renderer/pgsr.py`, `utils/adjacency_utils.py`,
  `utils/video_utils.py`.
- **Drop from the repo root:** `assets/`, `install.py`, `requirements.txt`, `.gitmodules`, and
  the submodules `Depth-Anything-V2`, `nvdiffrast`, `diff-gaussian-rasterization_ms`,
  `fused-ssim`, `simple-knn`, `tetra_triangulation` (fact 7).
- **Drop `third_party/glm/{doc,test}`** from both rasterisers: ~18 MB of the GOF copy's 22 MB;
  the build needs only `third_party/glm/glm/`.
- **Add nothing to the vendored tree.** Provenance (URL, commit, dropped files, the fact-7
  identity check, patch sites) goes in `MEMORY/backend-milo.md` (step 7).

### Step 2 — env: build the two rasterisers

```bash
ENV=/blue/arthur.porto/srizvi63.gatech/conda/augenblick_rtx_pro_6000
SNAP=output/results/neurips_verify/milo/_install   # = $V/_install (Verification)
```

- **2a — snapshot (rollback path):** `$ENV/bin/python -m pip freeze > $SNAP/pip_freeze_before.txt`.
- **2b — `setup_common.sh`:**
  - a `milo` arm in step 7's `case " $BACKENDS "` block that builds the two dirs above;
  - in the same arm, build SuGaR's `simple-knn` **only when `sugar` is not in `BACKENDS`**, and
    GW's `diff-gaussian-rasterization-ms` + `fused-ssim` **only when `gw` is not**, so
    `BACKENDS=milo` alone is complete;
  - step 8's tetra gate becomes `*" gw "*|*" milo "*`; its WARN text names both backends;
  - `milo` in the `BACKENDS` default (line 14) and the header comment (line 5);
  - the verify block gains `diff_gaussian_rasterization_radegs`,
    `diff_gaussian_rasterization_gof` and `tetranerf.utils.extension` (labelled MILo / GW+MILo).
- **2c — build the live env** with a one-off GPU job reproducing `build()` for the two dirs
  only, under `cuda/13.0.2` (the module the live env's extensions were built with), with
  `--account=arthur.porto-phenomics`, **not**
  `rebuild_env.sbatch` (it reruns all of `auto_setup.sh`, torch included). GeoSVR precedent.
- **2d — check:** `$SNAP/pip_freeze_after.txt`, diffed into `$SNAP/pip_diff.txt`: exactly the two
  new distributions; torch, torchvision, numpy and `diff_gaussian_rasterization` (SuGaR's)
  unchanged.
- **2e — `rebuild_env.sbatch`** import check gains the two new modules (as GeoSVR added
  `svraster_cuda._C`).

### Step 3 — local patches

Each patch carries `LOCAL PATCH (augenblick)`, so `grep -rn` finds every divergence. 3a/3b
mirror what `src/libs/pgsr` and `src/libs/2dgs` already do; read theirs first.

**3a — masks** (fact 4):
- `CameraInfo` gains `mask_path: str | None`.
- `readColmapCameras` resolves `<dirname(images_folder)>/masks/<image_name>.png`, or `None`;
  print `Loaded N masks` once, as 2DGS/PGSR do.
- `loadCam` (`utils/camera_utils.py`): when `mask_path` is set and the image has no alpha, open
  it as `L`, `PILtoTorch` it at the same `resolution` as the image → `loaded_mask`, then
  composite `gt_image = gt_image * m + bg * (1 - m)` with `bg` from `args.white_background`.
  Compositing once here means the training loss, `render.py`'s GT export and `training_report`
  all see the same image.
- Do **not** `putalpha` in the reader: `Image.open` is lazy there, and a full-res RGBA decode
  of 432 × 6240×4160 images is ~45 GB of host RAM (the TopoSurfel plan's warning).

**3b — `split.json` under `--eval`** (fact 5), in `readColmapSceneInfo` inside `if eval:`. If
`<path>/split.json` exists, partition by `c.image_name in meta["train"/"test"]` **over the sorted
`cam_infos`** (keeps fact 14's order) and print `split.json: N train, M test`; otherwise keep
`idx % llffhold`.

**3c — no input-scene write** (fact 3), in `readColmapSceneInfo`:
- if `sparse/0/points3D.ply` exists → `fetchPly` it (hull scenes carry their init cloud there);
- else → `read_points3D_binary` into a `BasicPointCloud` in memory (zero normals, colours / 255),
  `ply_path=None`;
- `Scene.__init__` (`scene/__init__.py:52-53`): when `ply_path` is `None`, write
  `<model>/input.ply` with `storePly` from the in-memory cloud instead of copying a file.

**3d — no viewer socket** (fact 8), in both trainers: delete the `network_gui.init(...)` call and
the random-port block above it. Leave the `try_connect` loop; it no-ops on an unbound socket.

**3e — the CLI beats `configs/fast`** (fact 2), in `arguments/__init__.py:read_config`:
collect the `dest` of every action whose option string appears in `sys.argv[1:]` (as `-x`,
`--x` or `--x=v`), and apply a config-file key only when its `dest` is not in that set.
Prototyped on the env's 3.10 (*checked*):
- no flags → every `configs/fast` value, as upstream;
- `--iterations 2000` → 2000 (upstream: 18000);
- `--iterations 18000` (equal to the config) and `--densify_until_iter=500` both win;
- `-r 2` does not leak into other keys.
`get_combined_args` is a different function and stays as it is.

**3f — mesh config by path**, in both trainers (`mesh_config_file = …`, `train.py:617`,
`train_regular_densification.py:499`) and `mesh_extract_sdf.py:584`: if the value ends in
`.yaml`, use it as a path; otherwise keep `configs/mesh/<name>.yaml`. This lets verification
and tuning use a schedule yaml outside the vendored tree without editing it.

**3g — mask loss** (facts 4, 15):
- `OptimizationParams` gains `self.lambda_mask = 0.0` (upstream behaviour when absent).
- Both trainers, after the photometric `loss = …`: if `opt.lambda_mask > 0`, the camera has a
  `gt_mask` and `render_pkg.get("mask")` is not `None`, add
  `opt.lambda_mask * l1_loss(render_pkg["mask"], gt_mask.cuda())`.
- Two-sided L1 on rendered alpha, the same target 2DGS and GW supervise as a 4th loss channel
  (`2dgs/train.py:119-123`). A separate term rather than a 4th channel, because
  `--decoupled_appearance`'s `L1_loss_appearance` assumes 3 channels.

**3h — rename RaDe-GS** (fact 6):
- directory `submodules/diff-gaussian-rasterization` → `…-radegs`; package dir
  `diff_gaussian_rasterization/` → `diff_gaussian_rasterization_radegs/`;
- `setup.py:18-22`: `name`, `packages` and the extension `name` →
  `diff_gaussian_rasterization_radegs[._C]`. `ext.cpp` uses `TORCH_EXTENSION_NAME` and the
  package does `from . import _C`, so nothing else in the extension changes;
- imports at `gaussian_renderer/radegs.py:14` and `gaussian_renderer/__init__.py:32`
  (`render_old`, lazy, but renamed so no path reaches SuGaR's module).

### Step 4 — adapter, `src/augenblick/reconstruction/milo.py`

`MiloBackend(SubprocessBackend)`: `name = "milo"`, `backend_dir = LIBS_DIR / "milo" / "milo"`,
default `use_cwd = True` (fact 18). Module constants `MILO_DIR`, `TRAIN_FAST_SCRIPT`
(`train.py`), `TRAIN_REGULAR_SCRIPT`, `RENDER_SCRIPT`, `SDF_SCRIPT` (`mesh_extract_sdf.py`),
`TSDF_SCRIPT` (`mesh_extract_regular_tsdf.py`). Register with `@register_reconstruction` and add
to the `__init__.py` import line.

**`MiloConfig(EvalParams, ResolutionParams)`**, frozen:

| Field | Default | Flag / note |
|---|---|---|
| `densification` | `"regular"` | `Literal["regular","fast"]` → which trainer (fact 1) |
| `iterations` | `None` → 30,000 / 18,000 | always passed: `--iterations` (train), `--iteration` (sdf extract) |
| `densify_until_iter` | `None` | passed only when set (recipe default otherwise) |
| `regularization_from_iter` | `None` | passed only when set |
| `mesh_config` | `None` → `default_dtu` / `default` | name or `.yaml` path (3f); `--mesh_config` (train) **and** `--config` (sdf extract) |
| `rasterizer` | `"radegs"` | `Literal["radegs","gof"]`; train, render, both extractors |
| `imp_metric` | `"indoor"` | `Literal["indoor","outdoor"]`; train + sdf extract. DTU uses `indoor` |
| `dense_gaussians` | `False` | |
| `decoupled_appearance` | `False` | **off, unlike upstream's DTU recipe** (below) |
| `lambda_mask` | `0.1` | 3g; provisional, check 10 |
| `white_background` | `False` | train only; the rest read `cfg_args` |
| `extract` | `"sdf"` | `Literal["sdf","tsdf"]` (fact 12) |
| `refine_iter` | `1000` | sdf |
| `remove_oof_vertices` | `False` | sdf; drops vertices no training camera sees |
| `mesh_res` / `num_cluster` | `1024` / `50` | tsdf (upstream defaults; 2DGS uses the same) |

**Fixed on the train argv:** `-s <scene> -m <output>`, `-r <resolution>`, `--data_device cpu`
(GW precedent; 432 × 1600 px float images are ~9 GB), `--test_iterations -1` (fact 9),
`--checkpoint_iterations -1` (fact 10), `--eval` when set.

**Why `decoupled_appearance` is off.** It fits a per-training-camera appearance embedding
(GOF). Our captures use a fixed lighting rig, so there is little exposure drift to absorb, and
the extra freedom is a confound against 2DGS/PGSR/GeoSVR, which have none. GW turns its
exposure compensation off under `--eval` for the same reason. Revisit if check 8 shows
per-view colour drift.

**`validate()`**:
- `super().validate(scene)`, then:
- `importlib.util.find_spec("tetranerf") is None` → `SceneError`, naming `SKIP_TETRA`. Without
  it, `regularization/regularizer/mesh.py:33-38` only prints a warning, and training dies with a
  `NameError` at the first triangulation, ~20,000 iterations in.
- `lambda_mask > 0` and `rasterizer == "gof"` → `ValueError` (fact 15), as `dvlt.py` does for
  incompatible flags; the message suggests `--lambda_mask 0`.
- `lambda_mask > 0` → every **registered** image needs `masks/<stem>.png`, else `SceneError`
  with the count and first stems. MILo loads only registered images. **Lift GeoSVR's check**
  (`geosvr.py` `validate`, the uncommitted `registered_stems` version) into a shared helper in
  `reconstruction/base.py`, `require_registered_masks(scene, why: str)`, and call it from both.
- `mesh_config` ending in `.yaml` must exist (`ValueError`).

**No `prepare()` override.** 3c makes reading the scene in place safe; everything goes under
`<output>/`.

**`stages()`** (with `I` = resolved iterations, `C` = resolved mesh config):

```
train : python <TRAIN_REGULAR_SCRIPT | TRAIN_FAST_SCRIPT> -s <scene> -m <output>
          --imp_metric M --rasterizer R --mesh_config C --iterations I
          [--densify_until_iter D] [--regularization_from_iter G]
          --lambda_mask L --data_device cpu --test_iterations -1 --checkpoint_iterations -1
          -r <resolution> [--eval] [--white_background] [--dense_gaussians] [--decoupled_appearance]
render: python <RENDER_SCRIPT> -m <output> --rasterizer R --skip_train          (with --eval)
mesh  : python <SDF_SCRIPT> -m <output> --iteration I --rasterizer R --config C
          --imp_metric M --refine_iter N [--remove_oof_vertices]
      | python <TSDF_SCRIPT> -m <output> --rasterizer R --mesh_res X --num_cluster K
```

- The render stage runs **before** extraction (GW precedent): it takes seconds, and the
  metrics survive a failed extraction.
- `render.py --iteration -1` (its default) loads the newest `point_cloud/iteration_*`.

`mesh_path()` → `<output>/mesh_learnable_sdf.ply` (sdf) or
`<output>/mesh_regular_tsdf_res<mesh_res>_post.ply` (tsdf). `test_renders_root()` keeps the
default `<output>/test`. `header()` adds the trainer, mesh config and extractor.

### Step 5 — tests (CPU-only, `tests/test_recon_argv.py` unless noted)

- `test_milo_argv`: train + sdf-extract argv at defaults, asserted on the whole list. Starts
  `sys.executable, str(TRAIN_REGULAR_SCRIPT), "-s", …, "-m", …`; carries
  `--mesh_config default_dtu --iterations 30000 --test_iterations -1 --checkpoint_iterations -1`;
  extraction carries `--iteration 30000 --config default_dtu`.
- `test_milo_fast_recipe_defaults`: `--densification fast` → `TRAIN_FAST_SCRIPT`,
  `--iterations 18000`, `--mesh_config default`, matching `--iteration`/`--config` on extraction.
- `test_milo_mesh_config_path_reaches_both_stages(tmp_path)`: a `.yaml` path appears verbatim on
  train and extract; a missing one raises in `validate()`.
- `test_milo_eval_adds_render_before_extraction` / `…_no_eval_has_two_stages`.
- `test_milo_tsdf_switches_script_and_mesh_path`: `--extract tsdf --mesh_res 512` →
  `TSDF_SCRIPT` argv and `mesh_regular_tsdf_res512_post.ply`.
- `test_milo_mask_loss_requires_every_registered_mask(tmp_path, monkeypatch)` and
  `…_gof_refuses_mask_loss`, mirroring the GeoSVR tests; `lambda_mask 0` passes both.
- `test_milo_requires_tetranerf(monkeypatch)`: `find_spec` stubbed to `None` → `SceneError`.
- `test_milo_reads_scene_in_place(tmp_path)`: `prepare()` returns the input scene and building
  the stages writes nothing.
- `test_milo_read_config_cli_patch` (3e): a subprocess with `cwd=src/libs/milo/milo`, importing
  only `arguments` (no CUDA), building a parser with `iterations` / `densify_until_iter` and
  calling `read_config` with a temp `--config_path`; asserts the four prototype cases.
- `test_milo_patches_present`, a grep check guarding a careless resync:
  - `LOCAL PATCH (augenblick)` in every patched file;
  - `dataset_readers.py` has no unconditional `storePly(ply_path`;
  - no `network_gui.init(` in either trainer;
  - no `import diff_gaussian_rasterization ` / `from diff_gaussian_rasterization import` left
    under `milo/`, and the RaDe-GS `setup.py` names `diff_gaussian_rasterization_radegs`;
  - both trainers and `mesh_extract_sdf.py` accept a `.yaml` mesh config.
- Registry: add `milo` to `tests/test_registry.py:52`.
- `test_require_registered_masks` (base helper), and GeoSVR's two mask tests keep passing
  through it.

### Step 6 — SLURM and setup scripts

- `hpg_slurm/recon.sbatch` (lines 5, 35-37) and `pace_slurm/recon.sbatch` (lines 3, 29-31):
  `milo` in the `case "$BACKEND"` allow-list, its error message and the header comment.
- **Wall time:** unmeasured. The mesh phase rebuilds a CGAL Delaunay of up to 5.4 M pivots on
  the CPU every 500 iterations (20 times over 20,001–30,000), and runs marching tets plus an
  nvdiffrast pass every iteration. Submit the first full run with `--time=8:00:00`; put the
  measured recommendation in the header comment from check 8. Do not raise the shared 3 h
  default.
- `scripts/setup_common.sh` and `hpg_slurm/rebuild_env.sbatch`: the step 2 changes.

### Step 7 — docs (proposed edits, per CLAUDE.md)

- **New `MEMORY/backend-milo.md`:**
  - what it is; the two recipes and why `regular` is the default; the 3e trap;
  - the extraction-must-match-training rule (fact 11) and the two extractors;
  - fact 13's warm-up constraint on custom schedules;
  - the shared GW/SuGaR builds (fact 7) and the RaDe-GS rename (3h);
  - a **Vendoring** section: URL, commit `0c38aa65f8b768513964f13246ef1ae54f001c70`, the files
    dropped in step 1, the fact-7 identity check, and each `LOCAL PATCH` site (file, function,
    change, why), with a reminder to re-apply them on resync.
- **`MEMORY/pipeline-reconstruction.md`:** the `milo` wrapper, its flags and mesh paths.
- **`MEMORY/augenblick-package.md`:** `milo` in the method list; under "Per-backend
  divergences": two trainer scripts behind one field, scene read in place, and the shared
  `require_registered_masks` helper.
- **`MEMORY/environment-and-gpu.md`:** `milo` in `BACKENDS`; the two new extensions; the
  `diff_gaussian_rasterization` name collision and why the rename exists; `SKIP_TETRA=1` now
  disables MILo too.
- **`MEMORY/backend-gaussian-wrapping.md`:** one line saying `_ms`, `fused-ssim` and
  `tetra_triangulation` are shared with MILo, so a compatibility fix there serves both.
- **`.claude/CLAUDE.md`:** `milo` in the Reconstruction row and the `src/libs/` list;
  `backend-milo.md` in the index; the `LOCAL PATCH` gotcha line gains `milo`. The hull gotcha
  ("reaches every backend but GeoSVR") stays true because of 3c.
- **`MEMORY/repo-conventions.md`:** add this plan to the "Landed" list when it lands.

## Verification

**Specimen:** `UF_Herp_87980_cranium` from `output/results/neurips/subset`, with **COLMAP** as
the SfM input, as for GeoSVR and TopoSurfel: 432/432 registered, 432 masks, 6240×4160
`SIMPLE_PINHOLE`, no `split.json`. Timings to compare against (`neurips/subset/_timing/sfm_timings.csv`,
*checked*): colmap-2dgs 1186 s, colmap-pgsr 2560 s, colmap-gw 2111 s, colmap-geosvr 3404 s.

**Every result goes under `V=output/results/neurips_verify/milo`.** Nothing is written under
`neurips/subset`.

```
$V/_install/          pip_freeze_before.txt, pip_freeze_after.txt, pip_diff.txt (2a/2d)
$V/_checks/           logs for checks 1, 1b, 2, 3; smoke_dtu.yaml
$V/_timing/           sfm_timings.csv, appended by recon.sbatch
$V/UF_Herp_87980_cranium/
    colmap/                  staged input scene (a real copy)
    colmap-milo-smoke/       checks 3–7
    colmap-milo-fast-smoke/  check 6b
    colmap-milo/             check 8 (+ check 9's TSDF mesh, written alongside)
    colmap-milo-nomask/      check 10
```

**Staging (a copy, not symlinks)**, for the GeoSVR plan's reason. Record a checksum of
`points3D.ply` so check 3 can prove 3c:

```bash
SRC=output/results/neurips/subset/UF_Herp_87980_cranium/colmap
DST=$V/UF_Herp_87980_cranium/colmap
mkdir -p $DST/sparse && cp -rL $SRC/images $SRC/masks $DST/ && cp -rL $SRC/sparse/0 $DST/sparse/0
sha256sum $DST/sparse/0/points3D.* > $V/_checks/sparse_before.sha256
```

**Smoke schedule** (`$V/_checks/smoke_dtu.yaml`): a copy of `configs/mesh/default_dtu.yaml` with
`start_iter: 3501`, `stop_iter: 5000`, `learnable_sdf_reset_stop_iter: 4501`. `start_iter` must
stay above `warn_until_iter` (3,000; fact 13). It is a **plumbing value**; never quote its mesh.

**Runs go through the real job script** (`UF_Herp_87980_cranium` is array index 0 of the subset):

```bash
export DATA_ROOT=/blue/arthur.porto/data/datasets/photogrammetry/neurips/subset/prepared
export RESULT_ROOT=$V BACKEND=milo SFM=colmap
ACCT=--account=arthur.porto-phenomics   # overrides recon.sbatch's #SBATCH --account (Scope)
SMOKE="--iterations 5000 --densify_until_iter 3000 --regularization_from_iter 2000 \
       --mesh_config $V/_checks/smoke_dtu.yaml --refine_iter 200"
sbatch $ACCT --array=0 hpg_slurm/recon.sbatch --eval $SMOKE                    # → mv to -smoke
sbatch $ACCT --array=0 hpg_slurm/recon.sbatch --densification fast $SMOKE      # → mv to -fast-smoke
sbatch $ACCT --array=0 --time=8:00:00 hpg_slurm/recon.sbatch                   # check 8 → colmap-milo
sbatch $ACCT --array=0 --time=8:00:00 hpg_slurm/recon.sbatch --lambda_mask 0   # check 10 → mv to -nomask
```

`$ACCT` must come **before** the script path: anything after it is passed to `augenblick`, not
to `sbatch`. The one-off GPU jobs (step 2c, checks 0, 1b and 9) take the same flag.

The output dir is always `$RESULT_ROOT/<scene>/colmap-milo`, so run the jobs one at a time and
rename each before the next starts.

| # | Check | Pass condition |
|---|---|---|
| 0 | env | `pip_diff.txt` = the two new distributions only; on a GPU node `import diff_gaussian_rasterization_radegs, diff_gaussian_rasterization_gof` works **and** `diff_gaussian_rasterization.GaussianRasterizationSettings._fields` still lacks `kernel_size` (SuGaR's survived) |
| 1 | registration | `augenblick recon --list` shows `milo`; `--help` shows every field |
| 1b | script interface | GPU job, cwd `src/libs/milo/milo`: `--help` exits 0 for both trainers, `render.py` and both extractors (proves the imports and the rename, 3h) |
| 2 | tests | `pytest tests/` green |
| 3 | input scene untouched | after the smoke run, `sha256sum -c $V/_checks/sparse_before.sha256` passes and `find $V/UF_Herp_87980_cranium/colmap -newer colmap-milo-smoke/cfg_args` lists nothing but `split.json` (3c); `neurips/subset/.../colmap` unchanged |
| 4 | masks | `Loaded 432 masks`; `test/ours_5000/gt/*.png` have a black background (3a) |
| 5 | split | `split.json: 378 train, 54 test` (3b) |
| 6 | mesh loop + extraction | the smoke log shows mesh regularisation starting at 3501 and `Delaunay triangulation time`; `mesh_learnable_sdf.ply` exists, non-empty; `nvs_metrics.json` with `n_test` 54 |
| 6b | fast recipe + 3e | the fast smoke trains **5000** iterations (`point_cloud/iteration_5000/`), not 18000; no `chkpnt*.pth` in either smoke dir (fact 10); no `Using random port` / bind error (3d) |
| 7 | no network | the smoke log has no `Downloading` line (fact 9) |
| 8 | full run | defaults, no `--eval`, like the baselines. Report per-stage wall time, peak RSS, peak VRAM, Gaussian count and mesh face count, against the four timings above. Visually compare with `colmap-gw` and `colmap-pgsr` (detail, holes, floaters, background) |
| 9 | extractor comparison | in a GPU job, run `mesh_extract_regular_tsdf.py -m colmap-milo --rasterizer radegs` by hand. Compare its `_post` mesh with check 8's sdf mesh (face count, `augenblick.eval.mesh` without `--rigid`, visual), to judge fact 12's untested combination |
| 10 | mask loss | `--lambda_mask 0` full run vs check 8: background floaters and face count. Settles the 0.1 default |
| 11 | other SfM inputs | one `SFM=vggt` and one `SFM=hull` smoke run complete; the hull run's training log reports the hull cloud's point count (read from `points3D.ply`, 3c) |

## Open risks

- **RaDe-GS / GOF builds on sm_120 / CUDA 13 / torch 2.9.1:** untested upstream (CUDA 11.8
  only). If either cannot be made to build, MILo cannot run at all (fact 6: both are imported at
  module level), so the plan stops at step 2.
- **`mesh_extract_sdf.py` after regular densification** is not an upstream combination
  (fact 12). If check 9 shows it is worse than the TSDF mesh, switch the `regular` default to
  `--extract tsdf` and record why, rather than defaulting to `fast`.
- **Runtime and memory:** CGAL triangulation of up to 5.4 M pivots on the CPU, 20 times per run,
  plus `mesh_extract_sdf`'s own. 432 views at 1600 px is ~9× DTU's 49. Check 8 sets `--time` and
  `--mem`. The knobs, in order: a `lowres` mesh config (540 k Delaunay sites), then
  `delaunay_reset_interval`.
- **VRAM on PACE A100 40 GB:** unmeasured; the paper's `veryhighres` notes higher memory. Check 8
  reports peak VRAM on the 96 GB card.
- **Background in the mesh:** the Delaunay SDF ignores masks (fact 4). Masks act only through the
  composited GT and the alpha loss. If check 10 shows background sheets survive at
  `lambda_mask 0.1`, the next step is passing `gt_mask` into `evaluate_mesh_occupancy`'s
  occupancy labels (a deeper patch, not planned).
- **Shared builds:** a change to GW's `_ms`, `fused-ssim` or tetra, or to SuGaR's
  `simple-knn`, now also changes MILo. `SKIP_TETRA=1` silently removes MILo, which `validate()`
  catches before training.
- **Upstream churn:** the pinned commit is six months old and the README still lists TODOs. On a
  resync, re-run the fact-7 identity diff and re-check the `configs/` schedules against facts 1
  and 13.
