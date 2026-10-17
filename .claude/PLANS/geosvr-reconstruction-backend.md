# Plan: add `geosvr` as a fifth reconstruction backend

Implementation spec, written before the work landed. Once it lands this file is historical:
where it and the code disagree, the code wins. The durable description belongs in
[pipeline-reconstruction.md](../MEMORY/pipeline-reconstruction.md) and a new
`MEMORY/backend-geosvr.md`.

**Scope:** the code and the job scripts treat `geosvr` like any other backend on both clusters.
Installing and verifying it covers only the HPG env `augenblick_rtx_pro_6000`. Other envs pick
it up from `setup_common.sh` (`BACKENDS` default) on their next rebuild. B200 waits on its env
drift being fixed (see environment-and-gpu.md). Verified 2026-10-10 against the tree at `2aa09ce` and upstream GeoSVR at
**`44bf7e74471e4cbeb2a11a7f86870208a2861bd1`** (2026-01-02). Upstream line references are to
that commit.

## Status

| Step | State |
|---|---|
| 1 — vendor into `src/libs/geosvr/` | **Done** 2026-10-15: 68 files from `44bf7e7` |
| 2 — env install + CUDA build | **Done**: `pip_diff.txt` adds only the listed packages + `svraster_cuda`; extension built **unpatched** (job 46089322, 4m53s, `cuda/13.0.2`, sm_120); DA2 in `/blue` cache (1.3 GB), home cache untouched |
| 3 — local patches | **Done**, plus two found in verification (below) |
| 4 — adapter `reconstruction/geosvr.py` | **Done** |
| 5 — `find_test_dir` suffix fix | **Done** |
| 6 — tests | **Done**: 151 passed (incl. a pre-existing stale `dvlt` registry assertion, fixed) |
| 7 — SLURM + setup scripts | **Done**; also `svraster_cuda._C` added to `rebuild_env.sbatch`'s import check |
| 8 — docs | **Done** |
| Verification | **Plumbing passes** (checks 0–7). The default config **collapsed** (check 8, full schedule) because the DA2 prior supervised the composited background; patch 7 fixes it at smoke scale (PSNR 27.8 vs 19.4 collapsed, ~30 without the prior). Not yet re-run: check 8 on the full schedule with patch 7, and check 9 |

| # | Result (2026-10-15) |
|---|---|
| 0 | pass: 11 added packages (the planned set + `imageio-ffmpeg` + transitive deps), none removed; `svraster_cuda._C` loads on GPU |
| 1 / 1b | pass; 1b ran as GPU job 46090964 (every script imports CUDA) |
| 2 | pass: 153/153 |
| 3 | pass: DA2 ran under `transformers` 5.17; 432 priors in `<output>/mono_priors/` |
| 3b | pass: nothing in the staged scene newer than `config.yaml`; `neurips/subset` scene unchanged |
| 4 | pass: `Loaded 432 masks`; GT background exactly 0 |
| 5 | pass: `split.json: 378 train, 54 test` |
| 6 | pass: 100% of 53,503 points inside the pcd bound, wide margin |
| 7 | plumbing pass: `test/ours_2000_r3.9`, `n_test` 54, mesh at `mesh_path()`. **Quality: masked PSNR 6.09, SSIM 0.107, LPIPS 0.48** |
| 8 | **fails on quality**, runs fine (job 46103707): train 3596 s, render 279 s, TSDF 242 s, total 4117 s (vs 2DGS 1186 s, PGSR 2560 s, GW 2111 s); peak RSS 74 GiB. Train PSNR 30.15 at it 1000, then 17.07 at 2000, 19.4 at 20000: **the collapse is not a smoke-schedule artefact.** A train-view pg_view shows a large opaque black blob in front of the camera from it 2000 |
| 9 | held: meaningless until the collapse is fixed |
| smoke | job 46095860: train 583 s, render 248 s, TSDF 223 s; peak RSS 72 GiB. Train PSNR peaked at 25.6 (it 200), then fell to ~17 by it 350, before multi-view losses start (500); held-out pg_view shows a large near-camera opaque blob |

**Found during implementation** (the plan missed these):

- **Two more input-scene writes / upstream bugs**, patched and recorded in
  `MEMORY/backend-geosvr.md`. `colmap_loader.read_colmap_ply` cached `points3D.ply` and
  `points_correspondent.json` in `sparse/0/`; both are now built in memory. `data_pack`'s
  non-RGBA branch copied `cam_info.mask`, so the dedicated mask block raised "Duplicated mask".
  That branch was unreachable until patch 3a loaded masks.
- **Every GeoSVR script needs a GPU, even for `--help`:** `svraster_cuda` allocates a CUDA
  tensor at import. Check 1b therefore ran as a GPU job (46090964), not on a login node.
- **Not `rebuild_env.sbatch`:** it reruns all of `auto_setup.sh`, torch included. The extension
  was built by a one-off job reproducing `build()`. `setup_rtx_pro_6000.sh` names
  `cuda/12.9.1`/cu129, but the live RTX env has torch `2.9.1+cu130` (drift), so the build
  matched the env's existing rasterisers (`cuda/13.0.2`, `libcudart.so.13`).
- **The Depth-Anything prior collapsed training on masked scenes.** Smoke-scale ablations:
  mask loss off still collapses; prior off trains cleanly (PSNR ~30 at it 1800). DA2 on the
  composited black background predicts a smooth surface, and the prior's loss (masked only by
  *rendered* alpha) rewards building opaque black geometry there. Patch 7 also masks it by the
  segmentation mask: no collapse (26.4 at it 1000, vs 17 unpatched and 29.6 without the prior).
  Two upstream crashes surfaced on the way: `pg_view` read the prior unconditionally (patch 6),
  and the mono-normal loss needs the prior, so the wrapper now zeroes `lambda_normal_dmono` when
  `--lambda_depthanythingv2 0`.
- **The render stage runs on every run, not only with `--eval`.** The plan assumed the TSDF
  scripts render the train views themselves. They render depth, but read **colour** from
  `render.py`'s `train/ours_*/renders/` (upstream's DTU script always runs `render.py --skip_test`
  first). Without it, meshing failed with "RGB image size (0 0)". The stages are now train →
  `render.py [--skip_test]` → TSDF.
- **`imageio-ffmpeg` is needed after all.** The plan dropped it because nothing imports it by
  name, but imageio loads it as the `.mp4` plugin for `train.py:801`'s test-report video. Without
  it, training ran all 2,000 iterations and then crashed in a TIFF fallback (`fps`). It is now
  pinned (`0.6.0`) and installed.
- **64 GB is not enough at 24 MP:** upstream keeps every decoded image (432 × 6240×4160 RGB ≈
  34 GB) and now every full-res mask (≈ 11 GB) in RAM. The first smoke run was OOM-killed; the
  rerun peaked at ~71 GiB, so the GPU runs use `--mem=128gb`.

## Goal

`augenblick recon geosvr --scene <sfm> --output <out>` trains GeoSVR on any of the five SfM
methods' COLMAP scenes and writes a TSDF mesh. With `--eval` it also exports held-out renders
that `augenblick.eval.nvs` scores like the other backends' renders.

Non-goals: no new conda env, no DTU/TnT evaluation toolboxes, no MASt3R/DepthPro priors, no
fly-through rendering.

## What GeoSVR is

GeoSVR (Li et al., NeurIPS 2025 spotlight, `li2025geosvr`) is built on NVIDIA's **SVRaster**. The
scene is a sparse-voxel octree drawn by a custom CUDA rasteriser. On top, GeoSVR adds an
uncertainty-weighted Depth-Anything-V2 depth constraint and voxel-surface regularisers: PGSR's
multi-view NCC and geometric losses, surface rectification, and a scaling penalty. The mesh
comes from TSDF-fusing rendered depths.

It is the first **non-Gaussian** primitive in the comparison, and it uses SfM points only for
the scene bound, never to seed primitives. Hypothesis to test: it is less sensitive to
point-cloud density than the 3DGS family, and more sensitive to pose error.

| | PGSR (closest existing) | GeoSVR |
|---|---|---|
| Primitive | planar Gaussians | sparse-voxel octree |
| SfM points used for | initial Gaussians | **scene bound only** (`bound_mode: pcd`) |
| Config | argparse | **yacs**: `src/config.py` ← `--cfg_files` yamls ← CLI |
| Default iterations | 30,000 | 20,000 (`procedure.n_iter`) |
| Mesh | `mesh/tsdf_fusion_post.ply` | **`mesh/tsdf/tsdf_fusion_post.ply`** |
| Runtime model | none | Depth-Anything-V2-Large (HF Hub) when `lambda_depthanythingv2 > 0` |
| Licence | Inria, non-commercial | **NVIDIA SVRaster licence §3.3, non-commercial** + `LICENSE_inria.md` for 3DGS files |

Licence, carry into the docs: Depth-Anything-V2-**Large** weights are CC-BY-NC-4.0. That is
research-only, the same footing as the existing backends and DVLT, so nothing new to clear.

## Upstream facts that shape the design

Read in the cloned source, not the README.

1. **Masks are never loaded.** `src/dataloader/reader_colmap_dataset.py:82-84` hard-codes
   `mask = None`; the only other mask route is an RGBA alpha channel (`data_pack.py:204-206`).
   `cam_info.mask` is supported downstream (`blend_mask` compositing at `data_pack.py:124`), so
   a reader patch is enough → **3a**.
2. **The held-out split is `idx % test_every`** on a `natsort` order
   (`reader_colmap_dataset.py:155-160`), not `split.json`. The other backends read `split.json`
   (e.g. `2dgs/scene/dataset_readers.py:157-171`) → **3b**.
3. **Two CLI traps** in `src/config.py:update_argparser`/`update_config` (lines 245-283):
   - `'store_true' if t else …` tests the *type*, which is always truthy. So a `True`-default
     bool (`blend_mask`, `enable_da2_uncertainty`) **cannot be turned off**.
   - A CLI value is applied only if it differs from the `src/config.py` default, so a preset
     yaml silently beats an explicit flag equal to that default.

   **Decision:** fix both in `src/config.py` (**3d**). The wrapper then passes every exposed
   field as a flag, like PGSR, with no generated yaml. Upstream's dumped `config.yaml`
   (`train.py:891-892`) records the resolved config.
4. **Two writes go into the input scene**; everything else goes under `model_path`. Found by
   grepping `source_path` and every file write in `train.py`, `render.py`, both TSDF scripts
   and `src/`:
   - `mono_utils.py:64` caches depth priors in `<source_path>/mono_priors/<model>/`. The
     callers are at `train.py:66-73`, the MASt3R one at `train.py:76-79`, and
     `render.py:68-71`.
   - `data_pack.py:95` writes `<source_path>/multi_view.json` on every `DataPack` build (train,
     render, both TSDF scripts). **Nothing reads it.** The multi-view loss uses the in-memory
     `cur_cam.nearest_id` instead.

   Against a shared SfM dir these would write into `<scene>/colmap/` and race between array
   tasks. **Decision:** all intermediates go directly into the output dir. There is no
   symlinked or copied scene; the scene is read in place → **3c**.
5. **The script interface differs from the other backends.** All four wrappers run
   `python <absolute script path>`, giving training `-s <scene> -m <model>` and the later stages
   `-m <model>` (2DGS and GW also pass `-s`). GeoSVR differs in two ways:
   - Its mesh scripts do `from src.config import …` (`mesh_extract/tsdf_mesh.py:29`,
     `tsdf_mesh_360.py:29`) with no `sys.path` fix. Run by path, `sys.path[0]` is
     `mesh_extract/`, so the import fails; upstream relies on `PYTHONPATH=./`.
     `train.py` and `render.py` sit at the root and are unaffected.
   - Training takes `--source_path`/`--model_path`, and `render.py`, `tsdf_mesh.py` and
     `tsdf_mesh_360.py` take the model as a **positional** (`render.py:183`,
     `tsdf_mesh.py:143`, `tsdf_mesh_360.py:158`).

   **Decision:** patch the vendored scripts so GeoSVR is invoked exactly like the others →
   **3e**. Then there is no `python -m`, no `PYTHONPATH`, and no reliance on cwd. `import src`
   does not resolve in the RTX env (checked), so putting GeoSVR's root first on `sys.path`
   cannot shadow anything.
6. **Render dirs carry a suffix:** `test/ours_<iter>_r<res_downscale>/` whenever
   `res_downscale > 0` (`render.py:214-218`, `tsdf_mesh.py:168-172`). `nvs.find_test_dir`'s
   `int(name.split("_")[-1])` raises on it → **step 5**. A non-empty `--suffix` can't produce
   the bare name either. Render file names are the image stem (`render.py:84`), as `nvs.py`
   expects.
7. **TSDF scale.** `tsdf_mesh.py` uses an absolute `--voxel_size 0.002` / `--max_depth 5.0`, but
   our SfM methods differ in scale. `tsdf_mesh_360.py:70-84, 206-214` sets
   `max_depth = radius_scale × min camera-to-focus distance` and `voxel = max_depth / 1536` from
   the training cameras, so it needs no absolute scale. **Default `tsdf_mesh_360`**, with
   `tsdf_mesh` as the fallback. Both write `mesh/tsdf/tsdf_fusion_post.ply`.
8. **PINHOLE / SIMPLE_PINHOLE only** (`reader_colmap_dataset.py:72`). All five SfM methods
   emit one of these (`sfm/colmap.py:22`, `vggt.py:51,374`, `dvlt.py:284`,
   `turntable.py:439,483,510`), so no undistortion is needed.
9. **`procedure.sche_mult`** (`train.py:896-923`) scales `n_iter` and most `*_from/*_until`
   fields. It does **not** scale `rectifiy_from`, `scaling_penalty_*`, `W_enlarge_from` or
   `level_uncertainty_from`, so a `0.1` smoke run never reaches `rectifiy_from: 5000`. Fine for
   plumbing, meaningless for quality.
10. **Auto-downscale** caps images at 1024² pixels when `res_downscale` and `res_width` are both
    0 (`data_pack.py:22, 186-194`). That is ~0.4× the pixels of the other backends'
    `-r -1` = long side ≤ 1600 px, so the adapter maps `-r` itself (step 4).

## Environment

### Python deps

Imports on the train/render/mesh path, checked by grep and by `import` in the env:

| Import | Where | In env? |
|---|---|---|
| `yacs` | `src/config.py:11` | **no** |
| `natsort` | `reader_colmap_dataset.py:22`, `colmap_loader.py:14` | **no** |
| `lpips` | `loss_utils.py:12`, module level | **no**, although `requirements.txt:28` pins `0.1.4` (env drift) |
| `pytorch_msssim` | `loss_utils.py:13`, module level | **no** |
| `transformers` | `mono_utils.py:73`, only with DA2 on | **no** |
| `fused_ssim`, `cv2`, `imageio`, `open3d`, `plyfile`, `einops`, `scikit-image` | various | yes |

Nothing on this path imports upstream's `shapely`, `gpytoolbox`, `trimesh==4.0.4` or
`imageio-ffmpeg`, so skip them. Upstream's `environment.yml` (py3.9, torch 1.13) is not used.

**Pin `transformers` against `huggingface_hub==1.28.0`.** Unconstrained, pip resolves
`transformers-5.19.0` **and bumps the hub to 2.2.0**, which would move the `requirements.txt` pin
SAM3 and DVLT load weights through. With hub, numpy and torch held, the dry-run is clean, and
torch `2.9.1+cu130`, numpy 2.2.6 and the hub do not move:

```
Would install annotated-doc-0.0.5 lpips-0.1.4 natsort-8.4.0 pytorch-msssim-1.0.0
  shellingham-1.5.4 tokenizers-0.23.3 transformers-5.17.0 typer-0.27.3 yacs-0.1.8
```

Upstream tested `transformers==4.49.0`. Whether 5.x still accepts
`AutoImageProcessor(images=<tensor>, do_rescale=False)` (`mono_utils.py:83`) is
**unverified** (check 3).

### CUDA extension

`cuda/setup.py` builds `svraster_cuda._C` from 9 `.cu` files plus `binding.cpp`, with no
third-party includes. It goes through `setup_common.sh`'s `build()`. On torch 2.9 / CUDA 13 /
sm_120 it is **unverified**. Likely failures are deprecated `.type()`/`AT_DISPATCH` spellings
and CUB API changes; fix them as local patches, as was done for the other rasterisers.

## Implementation

Steps in dependency order. Run on a login node unless `nvcc` is needed; then use
`hpg_slurm/rebuild_env.sbatch`.

### Step 1 — vendor into `src/libs/geosvr/`

Copy the tree at `44bf7e7` **in-tree, not as a submodule**, like every patched backend.

- **Keep:** `train.py`, `render.py`, `mesh_extract/`, `src/`, `cfg/`, `cuda/`, `LICENSE`,
  `LICENSE_inria.md`, `README.md`.
- **Drop** (none is on our path): `asset/`, `scripts/` (benchmark drivers; `tnt_eval_vanilla`
  wants `open3d==0.9.0`), `eval.py`, `render_fly_through.py`, `render_mesh.py`,
  `environment.yml`.
- **Add nothing to the vendored tree.** Provenance (URL, commit, dropped files, patch sites)
  goes in `MEMORY/backend-geosvr.md` (step 8).

### Step 2 — install into the RTX env

```bash
ENV=/blue/arthur.porto/srizvi63.gatech/conda/augenblick_rtx_pro_6000
SNAP=output/results/neurips_verify/geosvr/_install   # = $V/_install (Verification)
```

- **2a — snapshot (rollback path):**
  `$ENV/bin/python -m pip freeze > $SNAP/pip_freeze_before.txt`.
- **2b — deps:** add `yacs==0.1.8`, `natsort==8.4.0`, `pytorch-msssim==1.0.0` and
  `transformers==5.17.0` to `requirements.txt`. Then install with the hub held:
  `$ENV/bin/python -m pip install yacs==0.1.8 natsort==8.4.0 pytorch-msssim==1.0.0
  lpips==0.1.4 transformers==5.17.0 huggingface_hub==1.28.0`.
- **2c — CUDA:** in `scripts/setup_common.sh`, add a `geosvr` arm to the step-7
  `case " $BACKENDS "` block (`build src/libs/geosvr/cuda`), and add `geosvr` to the `BACKENDS`
  default (line 14). Build via `rebuild_env.sbatch` with `BACKENDS=geosvr`.
- **2d — check:** write `$SNAP/pip_freeze_after.txt` and diff the two freezes into
  `$SNAP/pip_diff.txt`. It should list only those packages plus `svraster_cuda`; torch,
  torchvision, numpy and the hub stay unchanged.
- **2e — pre-fetch DA2** (~1.3 GB) into the `/blue` `HF_HOME` (`hpg_slurm/common.sh:42`), so
  array tasks never race on the download. Afterwards `~/.cache/huggingface` should be
  unchanged:

  ```bash
  HF_HOME=/blue/arthur.porto/srizvi63.gatech/cache/huggingface/ \
    $ENV/bin/huggingface-cli download depth-anything/Depth-Anything-V2-Large-hf
  ```

### Step 3 — local patches

Each patch carries `LOCAL PATCH (augenblick)`, so `grep -rn` finds every divergence.

**3a — masks** (fact 1), in `read_cameras_from_colmap`. Load
`<dirname(images_folder)>/masks/<image_name>.png` if present, as
`Image.open(p).convert("L").resize(image.size)`, into `mask`/`mask_path`. Print
`Loaded N masks`, as 2DGS does.
- Use the `mask` field, **not** `putalpha`: the data pack raises if it gets both
  (`data_pack.py:229-230`), and `lambda_mask` reads `cam.mask`.
- `blend_mask: True` then composites the masked-out pixels onto the background colour.

**3b — `split.json`** (fact 2), in `read_colmap_dataset` before `if eval:`. If
`<path>/split.json` exists, partition by `image_name in meta["train"/"test"]`; otherwise keep
`idx % test_every`. `image_name` is `basename.split(".")[0]` (`:79`), the stem
`augenblick.eval.split` writes.

**3c — no input-scene writes** (fact 4):
- **Depth priors:** pass the output dir as `source_path` at the call sites:
  `cfg.model.model_path` at `train.py:68,72,78`, and `args.model_path` at `render.py:70`
  (`cfg.model.model_path` is not yet reassigned there). Leave `mono_utils.py` untouched.
  The cache becomes `<output>/mono_priors/depthanythingv2/`, which `render.py` reuses.
- **`multi_view.json`** (`data_pack.py:95-110`): drop the `open(...)`, `json_d` and
  `file.write` lines, and keep the loop that fills `nearest_id`/`nearest_names`. Nothing reads
  the file, and redirecting it would mean threading `model_path` through four `DataPack`
  callers.

**3d — config CLI traps** (fact 3), in `src/config.py` only. `train.py` already forwards
non-train flags to `update_config(args.cfg_files, cmd_lst)` (`train.py:873-876`).
`render.py` and the TSDF scripts pass only a yaml, so they are unaffected.
- **Bools:** register them with `action=argparse.BooleanOptionalAction`. Bare `--x` still means
  `True`, so upstream's scripts keep working, and `--no-x` now exists.
- **The CLI always wins:** add `with_defaults=True` to `update_argparser`. In `update_config`,
  build `ArgumentParser(argument_default=argparse.SUPPRESS)` and register the fields with
  `with_defaults=False`, so the namespace holds only the flags actually given. Apply each
  unconditionally.
  - Prototyped on py3.9 and the env's 3.10: `--no-blend_mask` → `{blend_mask: False}`;
    `--eval --lambda_mask 0.0` → both; `[]` → `{}`; `nargs="*"` lists parse.
  - `set_defaults(k=SUPPRESS)` on `type=float` arguments **fails**: argparse type-converts the
    sentinel (`invalid float value`). The fields must be registered without a default.
- **Unchanged:** unknown flags still error in the strict `parse_args(cmd_lst)`.

**3e — script interface matches the other backends** (fact 5):
- **Imports in the mesh scripts:** in `tsdf_mesh.py` and `tsdf_mesh_360.py`, insert
  `sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))` after
  `import os` (line 20) and before the first `from src …` (line 29). Their in-function imports
  (`tsdf_mesh_360.py:74`, `from src.utils.render_utils …`) are then covered too.
- **`-s`/`-m` for training:** in `src/config.py:update_argparser`, register `source_path` and
  `model_path` with the short aliases `-s` and `-m`, alongside 3d's change. Prototyped on the
  env's 3.10: `train.py`'s outer `parse_known_args` passes `-s <scene> -m <out>` through to
  `cmd_lst` in order, and the inner parser resolves both.
- **`-m` for the later stages:** in `render.py:183`, `tsdf_mesh.py:143` and
  `tsdf_mesh_360.py:158`, replace the positional `parser.add_argument('model_path')` with
  `parser.add_argument('-m', '--model_path', required=True)`. `dest` stays `model_path`, so
  `args.model_path` is unchanged downstream. Like PGSR, these take no `-s`: they read
  `source_path` back from the model's `config.yaml`.

### Step 4 — adapter, `src/augenblick/reconstruction/geosvr.py`

`GeosvrBackend(SubprocessBackend)` with `name = "geosvr"`, `backend_dir = LIBS_DIR / "geosvr"`
and the default `use_cwd = True`, as in 2DGS and PGSR. Define module constants like theirs:
`GEOSVR_DIR`, `TRAIN_SCRIPT`, `RENDER_SCRIPT`, `TSDF_SCRIPT` (`mesh_extract/tsdf_mesh.py`),
`TSDF_360_SCRIPT` and `CFG_DIR`, all absolute. Register it with `@register_reconstruction` and
add it to the `__init__.py` import line.

**`GeosvrConfig(EvalParams, ResolutionParams)`**, frozen. Every field is passed on every run,
bools as an explicit `--x`/`--no-x`, so the argv is fully pinned whichever preset loads. The
`lambda_*` defaults mirror `cfg/dtu_mesh.yaml`.

| Field | Default | Upstream flag [yacs group] |
|---|---|---|
| `preset` | `"dtu"` | `Literal["dtu","tnt","none"]` → `--cfg_files <CFG_DIR>/<preset>_mesh.yaml` (omitted for `none`) |
| `sche_mult` | `1.0` | `--sche_mult` [procedure] (fact 9) |
| `lambda_depthanythingv2` | `0.1` | `--lambda_depthanythingv2` [regularizer]; 0 disables DA2 |
| `da2_uncertainty` | `True` | `--[no-]enable_da2_uncertainty` [regularizer] |
| `multi_view_ncc_weight` / `multi_view_geo_weight` | `0.05` / `0.01` | same names [regularizer] |
| `lambda_rectify` / `lambda_scaling_penalty` | `1e-5` / `5e-7` | same names [regularizer] |
| `lambda_mask` | `0.1` | `--lambda_mask` [regularizer]; **on for parity with 2DGS and GW** (below) |
| `white_background` | `False` | `--[no-]white_background` plus the opposite `--[no-]black_background` [model] |
| `tsdf` | `"auto"` | `Literal["auto","fixed"]` → `tsdf_mesh_360` / `tsdf_mesh` (fact 7) |
| `radius_scale` | `2.0` | `tsdf_mesh_360` only |
| `voxel_size` / `max_depth` / `sdf_trunc_scale` | `0.002` / `5.0` / `2.0` | `tsdf_mesh` only |
| `num_cluster` / `use_depth_filter` | `1` / `False` | both TSDF scripts |

**Why DTU.** It is the only object-centric preset. It already sets `outside_level: 1` (almost
no background octree), `bound_mode: pcd` (bound from the masked SfM points) and
`black_background: True`. The TnT preset (`outside_level` 5, auto-exposure) is for unmasked
scenes.

**Fixed overrides on the train argv**, which win over the preset via 3d:
- `--blend_mask`: DTU sets it False because its COLMAP path has no masks; ours do.
- `--eval`/`--no-eval`, always explicit: DTU sets `eval: False`, TnT sets `True`.
- `--res_downscale`, computed in `stages()` (fact 10): `-r N>0` → `N`. `-r -1` →
  `max(W, H) / 1600` from the first image if its long side exceeds 1600, else `0`.
  `res_width` stays 0.
- `-s <scene>` (read in place) and `-m <output>` (3e).
- `--lambda_mask 0.1`: upstream's default is 0 and no preset sets it. It is on **for parity
  with 2DGS and GW**, which supervise rendered alpha as a fourth channel inside their
  L1 + SSIM loss (`2dgs/train.py:119-123`, `gaussian_wrapping/train.py:282-286`).
  - PGSR does not, despite its comment at `pgsr/train.py:167`; it only composites the
    ground truth onto the background.
  - GeoSVR's loss (`train.py:274-276`) is a different form: an L2 between transmittance
    `T` and `1 - mask`.
  - `0.1` matches DTU's `lambda_photo` and is provisional.

**`validate()` mask check.** When `lambda_mask > 0`, every image in `images/` needs
`masks/<stem>.png`.
- Otherwise raise `SceneError` (exit 2) with the count and the first few missing stems, and
  suggest `--lambda_mask 0`.
- Why: upstream calls `cam.mask.cuda()` unguarded (an `AttributeError` mid-run). This way a
  bad scene fails before DA2 inference.

**No `prepare()` override.** GeoSVR reads `sparse/0` directly (`reader_colmap_dataset.py:118`)
and `masks/` and `split.json` from the scene root (3a/3b). PGSR copies its scene only to flatten
`sparse/`. The adapter writes nothing itself. The run's output is all under `<output>/`:
`config.yaml`, `checkpoints/`, `mono_priors/`, `test/` and `mesh/tsdf/`.

**`stages()`:**

```
train : python <TRAIN_SCRIPT> -s <scene> -m <output>
          --cfg_files <CFG_DIR>/dtu_mesh.yaml
          --sche_mult S --lambda_depthanythingv2 L --[no-]enable_da2_uncertainty
          --multi_view_ncc_weight N --multi_view_geo_weight G
          --lambda_rectify R --lambda_scaling_penalty P --lambda_mask M
          --[no-]white_background --[no-]black_background
          --blend_mask --[no-]eval --res_downscale D
          [--test_iterations -1]                              (with --eval)
render: python <RENDER_SCRIPT> -m <output> --skip_train       (with --eval)
mesh  : python <TSDF_360_SCRIPT> -m <output> --radius_scale R --num_cluster N [--use_depth_filter]
      | python <TSDF_SCRIPT> -m <output> --voxel_size V --max_depth D
          --sdf_trunc_scale S --num_cluster N [--use_depth_filter]
```

Without `--eval` there is no render stage, because the TSDF scripts render the train views
themselves. `render.py` runs DA2 on the views it renders (`render.py:67-71`); with
`--skip_train` that means only the held-out views, cached.

`mesh_path()` → `<output>/mesh/tsdf/tsdf_fusion_post.ply`. `test_renders_root()` keeps the
default `<output>/test`.

### Step 5 — `find_test_dir` accepts a suffix (fact 6)

In `src/augenblick/eval/nvs.py:34-46`, take the iteration from `re.match(r"ours_(\d+)", name)`
and skip non-matching names. Bare `ours_<n>` behaves as before.

### Step 6 — tests (CPU-only)

In `tests/test_recon_argv.py` unless noted:

- `test_geosvr_argv`: full train and mesh argv at defaults, asserted on the whole list. It
  starts `sys.executable, str(TRAIN_SCRIPT), "-s", …, "-m", …`, like the other backends, and
  includes `--blend_mask`, `--no-eval` and `--lambda_mask 0.1`. The render and mesh argvs
  take `-m` and no positional.
- `test_geosvr_bool_flags`:
  - `--no-da2_uncertainty` → `--no-enable_da2_uncertainty`;
  - `--white_background` flips both background flags;
  - `--eval` → `--eval`.
- `test_geosvr_config_cli_patch` (3d), run in a **subprocess with `cwd=src/libs/geosvr`** so
  upstream's `src` cannot resolve to ours. It needs `yacs`, so it runs only after step 2.
  After loading `cfg/dtu_mesh.yaml`, it checks:
  - `--no-blend_mask` and `--no-enable_da2_uncertainty` → `False`;
  - `--blend_mask` → `True` over DTU's `False`;
  - `--eval` and `--no-eval` both take effect;
  - an explicit value equal to the `src/config.py` default beats the preset;
  - an unknown flag exits non-zero.
- `test_geosvr_eval_adds_render_stage` / `…_no_eval_has_two_stages`.
- `test_geosvr_tsdf_fixed_switches_script`: `TSDF_SCRIPT` gets the absolute flags.
- `test_geosvr_reads_scene_in_place(tmp_path)`: `prepare()` returns the input scene,
  `--source_path` is its root, and building the stages writes nothing anywhere.
- `test_geosvr_patches_present`: a grep check, guarding 3c and 3e on resync:
  - no `mono_utils.prepare_*` call gets `source_path`, and no `multi_view.json` write remains;
  - both TSDF scripts insert their parent dir into `sys.path` before the first `from src`;
  - no `add_argument('model_path')` positional remains in `render.py` or the TSDF scripts.
- `test_geosvr_mask_loss_requires_every_mask(tmp_path)`:
  - one missing mask → `SceneError` at the default;
  - the same scene passes with `lambda_mask 0`.
- `test_geosvr_resolution_maps_long_side`: 4000×3000 → `res_downscale 2.5`.
- `test_find_test_dir_accepts_resolution_suffix` (eval tests): `ours_20000_r2.0` beats
  `ours_7000`.
- Add `geosvr` to any registry-name assertion.

### Step 7 — SLURM and setup scripts

- `hpg_slurm/recon.sbatch` (line 35) and `pace_slurm/recon.sbatch` (line 29): add `geosvr` to
  the `case "$BACKEND"` allow-list and its error message, and to the header comment
  (`pace_slurm/recon.sbatch:3`). Treat it like any other backend. Mesh extraction may need
  ~40 GB RAM (upstream README); 64 GB should do, but raise it on `oom_killed`.
- `scripts/setup_common.sh`: the 2c changes.

### Step 8 — docs (proposed edits, per CLAUDE.md)

- **New `MEMORY/backend-geosvr.md`:**
  - what it is; the CLI traps and patch 3d; the DA2 cache location; TSDF modes; the licence;
  - a **Vendoring** section: URL, commit `44bf7e74471e4cbeb2a11a7f86870208a2861bd1`, the files
    dropped in step 1, and each `LOCAL PATCH` site (file, function, change, why), with a
    reminder to re-apply them on resync.
- **`MEMORY/pipeline-reconstruction.md`:** the fifth wrapper, its flags and output path.
- **`MEMORY/augenblick-package.md`:**
  - add `geosvr` to the method list;
  - "Per-backend divergences": the `--cfg_files` preset overridden by flags (reliable via 3d),
    the scene read in place, and the 3e interface patches that make it invocable like the
    other backends.
- **`MEMORY/environment-and-gpu.md`:** the four new pins, the hub-pin trap, and `geosvr` in
  `BACKENDS`.
- **`.claude/CLAUDE.md`:** `geosvr` in the Reconstruction row and the `src/libs/` list, and
  `backend-geosvr.md` in the index.

## Verification

**Specimen:** `UF_Herp_87980_cranium` from `output/results/neurips/subset`, with **COLMAP** as
the SfM input. Checked 2026-10-15:
- 432/432 images registered, 53,503 points, 432 masks;
- all cameras `SIMPLE_PINHOLE` at 6240×4160, so `-r -1` gives `res_downscale 3.9`;
- it has no `split.json`;
- it already has COLMAP-input 2DGS, PGSR and GW runs, whose timings check 8 compares against.

**Every verification result goes under `V=output/results/neurips_verify/geosvr`**, which
resolves to `/blue/.../results/neurips_verify/geosvr` (3.4 TB free). The `geosvr/` subfolder
keeps it apart from the SAM3 verification already in `neurips_verify/`. Nothing is written
under `neurips/subset`.

```
$V/_install/          pip_freeze_before.txt, pip_freeze_after.txt, pip_diff.txt (step 2a/2d)
$V/_checks/           logs for checks 1, 1b, 2, 3b
$V/_timing/           sfm_timings.csv, appended by recon.sbatch
$V/UF_Herp_87980_cranium/
    colmap/               staged input scene (a real copy)
    colmap-geosvr-smoke/  checks 3–7
    colmap-geosvr/        check 8
    colmap-geosvr-fixed/  check 9
```

**Staging (a copy, not symlinks).** `--eval` makes `write_split` put `split.json` in the
scene root, and the subset scene's `images/`/`masks/` are symlinks into the dataset. So
dereference-copy only what GeoSVR reads, about 1.95 GB:

```bash
SRC=output/results/neurips/subset/UF_Herp_87980_cranium/colmap
DST=$V/UF_Herp_87980_cranium/colmap
mkdir -p $DST/sparse && cp -rL $SRC/images $SRC/masks $DST/ && cp -rL $SRC/sparse/0 $DST/sparse/0
```

`database.db` and `masks_colmap/` are COLMAP-only and are not copied.

**Runs go through the real job script.** In `scene_common.sh`, `DATA_ROOT` drives scene
discovery and `RESULT_ROOT` drives the input and output paths. With the subset dataset,
`UF_Herp_87980_cranium` is array index 0:

```bash
export DATA_ROOT=/blue/arthur.porto/data/datasets/photogrammetry/neurips/subset/prepared
export RESULT_ROOT=$V BACKEND=geosvr SFM=colmap
sbatch --array=0 hpg_slurm/recon.sbatch --eval --sche_mult 0.1   # → colmap-geosvr; mv to colmap-geosvr-smoke
sbatch --array=0 hpg_slurm/recon.sbatch                          # check 8 → colmap-geosvr
sbatch --array=0 hpg_slurm/recon.sbatch --tsdf fixed             # → colmap-geosvr; mv to colmap-geosvr-fixed
```

The output dir is always `$RESULT_ROOT/<scene>/colmap-geosvr`, so run the jobs one at a time
and rename each before the next starts. Check 9 runs last, after check 8's dir has been moved
out of the way.

Checks 3–7 use `sche_mult 0.1` (2,000 its), a **smoke value**; never quote its mesh.

| # | Check | Pass condition |
|---|---|---|
| 0 | env | `$V/_install/pip_diff.txt` = listed packages + `svraster_cuda`; torch/numpy/hub unchanged; `import svraster_cuda._C` works on a GPU node |
| 1 | registration | `augenblick recon --list` shows `geosvr`; `--help` shows every field |
| 1b | script interface (3e) | from `/tmp` in the env, `python <abs>/mesh_extract/tsdf_mesh_360.py --help` and `python <abs>/render.py --help` list `-m MODEL_PATH`. This proves the `sys.path` fix without `PYTHONPATH` |
| 2 | tests | `pytest tests/` green |
| 3 | DA2 on transformers 5.17 | `<output>/mono_priors/depthanythingv2/*` written; depth plausible (specimen = large inverse depth) |
| 3b | input scene untouched | `find $V/UF_Herp_87980_cranium/colmap -newer colmap-geosvr-smoke/config.yaml` is empty after all stages (`config.yaml` is `train.py`'s first write). `split.json` predates it: it is the shared held-out record, not a GeoSVR intermediate. `neurips/subset/.../colmap` is unchanged too |
| 4 | masks | `Loaded 432 masks`; `train/` or `pg_view/` images have a black background |
| 5 | split | train/test counts match `split.json`: 378/54 under `build_split`'s every-8th rule. No existing run used `--eval`, so there is no 2DGS split to compare |
| 6 | bounds | `Heuristic bounding: pcd` logged; bound encloses the specimen (vs SfM point extent) |
| 7 | mesh + eval | mesh at `mesh_path()`; renders land in `test/ours_2000_r3.9/` and `nvs_metrics.json` is written (this exercises step 5's suffix fix); `n_test` = 54 |
| 8 | full run | `sche_mult 1.0`, no `--eval`, like the baselines. Report per-stage wall time and peak RSS against `neurips/subset/_timing/sfm_timings.csv`: colmap-2dgs 1186 s, colmap-pgsr 2560 s, colmap-gw 2111 s (rtx6000, 432 images) |
| 9 | TSDF modes | `--tsdf fixed` at defaults vs check 8's `auto` mesh, on the same COLMAP scene: how far off the absolute defaults are at our scale |

## Open risks

- **CUDA build on sm_120 / CUDA 13 / torch 2.9:** untested upstream. If it can't be made to
  build, the plan stops at step 2.
- **`transformers` 5.x in `mono_utils.py`** (check 3): if it breaks, patch `mono_utils.py`. Do
  not downgrade: 4.49 needs `huggingface_hub < 1.0`.
- **Bound tightness:** `pcd_density_rate: 0.01` was tuned on DTU, and our point clouds vary by
  SfM method (VGGT 100k, COLMAP sparser). Check 6 is the guard. Promote it to a config field if
  it needs tuning.
- **Unmasked scenes with `outside_level: 1`:** the background may bake into the surface. Use
  `--preset tnt` there, and expect to revisit.
