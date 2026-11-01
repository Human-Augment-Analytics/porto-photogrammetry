# Plan: add `dvlt` as a fifth SfM method

## How to use this document

Implementation spec, written before the work landed. It records what was verified against the
tree and why each change is shaped the way it is. Once the work lands this file becomes
historical — where it and the code disagree, the code wins, and the durable description
belongs in [pipeline-sfm.md](../MEMORY/pipeline-sfm.md) and a new `MEMORY/backend-dvlt.md`.

**Target cluster: HiPerGator (`hpg_slurm/`).** Verified against the tree at `0f1483f` and the
live HPG envs on 2026-09-28/29. PACE is out of scope — a `pace_slurm/` twin can be added later
by copying the HPG script and swapping the cluster specifics documented in
[cluster-slurm.md](../MEMORY/cluster-slurm.md).

The DVLT library already exists at **`src/libs/dvlt/`** (package `dvlt`, source under
`src/libs/dvlt/src/dvlt/`). All line references below are to that tree.

## Status

| Step | State |
|---|---|
| 1a — `pip freeze` snapshot | **Done**: `/blue/arthur.porto/srizvi63.gatech/env-snapshots/rtx_pro_6000-before-dvlt.txt` (176 lines) |
| 1b–1d — install | **Done** 2026-09-29: dry-run named only `dvlt-0.0.1`; freeze diff is one `> dvlt` line; torch `2.9.1+cu130`, numpy 2.2.6 unchanged; `setup_common.sh` line added |
| 2 — model cache | **Done**: `models--nvidia--dvlt` under `/blue/.../cache/huggingface/hub/`; `~/.cache` unchanged |
| 3–6 — adapter, masks, registration, sbatch | **Done**: `src/augenblick/sfm/dvlt.py`, `link_dir` in `sfm/base.py`, `hpg_slurm/dvlt_sfm.sbatch` |
| Verification | Done 2026-09-29 (jobs 43808536–42); results below. Check 3's VGGT comparison is **open**: the VGGT baseline (43807181) hit `GPU_OOM` on the RTX at 276 images |

**Verification results** (scene 14 = `UF_Mammals_1199_mandible`, 276 × 6240×4160, Canon EOS RP
100 mm):

| Check | Result |
|---|---|
| 0 env | One `> dvlt` line; torch/numpy unchanged; repo imports |
| 1 registration | `dvlt` listed in both the RTX and B200 envs; `--help` shows every field |
| 2 cache | `models--nvidia--dvlt` on `/blue`; `~/.cache` unchanged |
| 3 poses | Stored 2D observations reproject to < 0.008 px (inversion right); camera centres form rings around the cloud. No VGGT comparison yet |
| 4 crop/scale | CPU subset: each masked view's own points are 98–99% inside its full-res mask |
| 5 confidence | **Fails as a filter**: at 276 views `depths_conf` p5/p50/p95 = 3.5/5.8/8.7, so `>= 2.0` keeps 99.8%. Default still 2.0 |
| 6 masking | Usable pixels 99.8% → 7.0%; cross-view in-mask fraction ~12% → 61.5% |
| 7 missing masks | Scene 11: warning logged, 5 unmasked frames kept, but they supply **44.6%** of the 100k points (all background) |
| 8 `K` | Inference 6.7 s (K=8) / 8.7–10.0 s (12) / 10.8 s (16); jobs ~75 s total, mostly image loading |
| 9 recon | 2DGS, 2000 its: `dvlt-2dgs/train/ours_2000/fuse_post.ply`, 200k verts (smoke only) |

**Found in verification:**

- **`decode_chunk_size`.** The `DVLT` wrapper defaults it to `None` (decode all frames at once),
  which fails at 276 frames with `input tensor must fit into 32-bit index math` in the depth
  head's conv. Now a `DVLTConfig` field, default 32; the decoders are per-frame, so it does not
  change outputs.
- **Focal length is biased high on this data.** EXIF gives f ≈ 17,360 px; DVLT gives fx
  20–24k and fy pinned at 20,800, which is `rays.py`'s `focal_lengths_norm` clamp (10 × H/2).
  ~20° FoV is at the edge of what the model covers. The model is self-consistent, but every
  downstream reconstruction inherits the bias.

**Decisions taken during implementation** (the plan left these open):

- **Missing masks (step 5): fall back to all-keep**, matching `vggt.py`. The adapter pairs masks
  by stem itself (`_load_masks`, using dvlt's `MASK_EXTS`) instead of calling
  `load_masks_for_directory`, which raises on any gap, and logs a warning with the count.
  Cost seen in the smoke run: the one unmasked frame supplied 67% of the 100k points, all
  background, because a masked frame keeps ~8% of its pixels and an unmasked one ~80%.
- **Rescale (step 4b): a DVLT-specific affine map** (`_to_original`), not VGGT's helper. It
  recomputes `preprocess_images`' per-axis resize, crop and pad offsets from each PIL size and
  maps network index `x` to COLMAP `(x + off + 0.5) / s`, for both K and `points2D`. The
  predicted principal point is kept, not replaced by the image centre.
- **dtype** follows `vggt.py` (bf16 on sm ≥ 8, fp16 below, fp32 and no autocast on CPU), so the
  same code path runs the CPU smoke test.
- Points are also filtered by `gradio_valid_pixels` and finite `world_points`.

## Goal

`augenblick sfm dvlt --scene <dir> --output <dir>` produces the same COLMAP scene that
`augenblick sfm vggt` does, from NVIDIA's Déjà View (DVLT) model instead of VGGT.

Non-goals, each justified below: no bundle adjustment, no DVLT training or evaluation, **no
new conda env**, no B200 support in the first pass.

## What DVLT is

DVLT ("Déjà View: Looping Transformers for Multi-View 3D Reconstruction", Burzio & Fischer et
al., `burzio2026dejaview`, NeurIPS 2026 oral) is a **recurrent** transformer for multi-view 3D
reconstruction. It loops one shared block of frame/global attention `K` times, so `K` is an
inference-time compute knob rather than a retrained variant. It is 117M parameters — roughly a
tenth of VGGT-1B — and its authors name "replacing slow SfM pipelines for 3DGS/NeRF
initialisation" as a target use.

Its output contract is the same shape as VGGT's: per-view extrinsics and intrinsics, per-pixel
depth, world points, and confidence. That is exactly the tuple `src/augenblick/sfm/vggt.py`
already converts into a COLMAP model, so this is a new adapter over the *same* conversion path,
not a new pipeline.

| | VGGT | DVLT |
|---|---|---|
| Params | 1B | 117M (counted from the built model) |
| Inference res | 518 square | 504 longest edge, centre-cropped to multiples of 14 (640×480 → 504×378) |
| Weights | `facebook/VGGT-1B` via `torch.hub` (`TORCH_HOME`) | `nvidia/dvlt` via HF Hub (`HF_HOME`) |
| Compute knob | — | `inference_steps` (`K`), 8–16 |
| Extrinsics | world-from-camera | `cameras.camera_to_worlds`, `(S, 3, 4)` **camera-to-world** |
| Weights licence | permissive | **NVIDIA licence — research/evaluation only** |

**Licence note, carry into the docs:** DVLT *code* is Apache 2.0 (portions under the VGGT
licence v1; see `src/libs/dvlt/LICENSE`); the *weights* are NVIDIA-licensed,
research/evaluation only. Weights are fetched at runtime and never committed.

## Environment: `augenblick_rtx_pro_6000` only, and nothing new to install

**Decision: install `dvlt` into the existing HPG env
`/blue/arthur.porto/srizvi63.gatech/conda/augenblick_rtx_pro_6000`** — the default
`GPU=rtx6000` target of `hpg_slurm/common.sh`. The only package added is `dvlt` itself:

- **No compiled CUDA extensions.** Plain `setuptools.build_meta`, no `ext_modules`. The env's
  torch does not move and the four CUDA rasterisers (2DGS, PGSR, SuGaR, GW) are untouched.
- **Every dependency is already in the env.** `src/libs/dvlt/pyproject.toml` lists
  `huggingface-hub`, `numpy`, `open3d`, `opencv-python-headless`, `Pillow`, `requests`,
  `safetensors`, `torch`, `torchvision` — all present. `flash_attn_interface` is a guarded
  optional import (Hopper FA3); `matplotlib`/`rerun` appear only in an
  `if __name__ == "__main__":` demo in `struct/cameras.py`.
- **Verified by import:** every module in `src/libs/dvlt/src/dvlt` imports in the RTX env via
  `PYTHONPATH`, with no stubs. `requires-python` is `>= 3.10`; the env is 3.10.21.

**No `requirements.txt` change is needed.**

### Why not the B200 env too

`augenblick_b200` has **drifted from its setup script**: it holds numpy **1.26.4**, torch
`2.9.0+cu128`, `pycolmap 3.10.0`, scipy 1.10.1, `opencv-python 4.5.5.64`, and
`huggingface_hub 1.4.1` — although `scripts/setup_b200.sh` sets `NUMPY_GENERATION="2"` and
`requirements.txt` pins `pycolmap-cuda12==4.1.1` / `huggingface_hub==1.28.0`. Neither DVLT nor
`vggt.py`'s pycolmap calls have been checked against that. Rebuild the env from
`setup_b200.sh` first (a separate task), then repeat step 1 there. Until then the DVLT sbatch
**refuses `GPU=b200`** (step 6).

## Feed-forward only, no BA (settled)

VGGT's `--use_ba` path routes through `predict_tracks`, which is VGGSfM's tracker wired to
VGGT's confidence maps, its 518→1024 resolution split, and mask erosion
(`_erode_masks`, `mask_erode_px`; see [backend-vggt.md](../MEMORY/backend-vggt.md)). Reusing it
would mean re-deriving that mapping against DVLT's cropped-504 geometry — a silent-bug class
where a wrong scale still produces a valid-looking model. DVLT is being added as an SfM
initialiser *to compare against*, and feed-forward is what makes that comparison like-for-like
with `vggt` without `--use_ba`. `DVLTConfig` has **no `use_ba` field**.

## The `dvlt` API

`K` is a **constructor** parameter (`model/dvlt/model.py:667`, default 12), not a `predict()`
argument. `src/libs/dvlt/README.md`'s quickstart does **not** match this API (it shows an
`Accelerator`); follow the snippet below.

```python
from dvlt.model.dvlt.model import DVLT
from dvlt.util.preprocess import load_sequence, preprocess_images, load_masks_for_directory

model = DVLT(img_size=504, inference_steps=12,   # K is fixed HERE
             load_patch_embed_weights=False)      # skip a DINOv2 fetch load_pretrained overwrites
model.load_pretrained("nvidia/dvlt", strict=True)
model.setup_test("cuda")                          # takes a device

name, frames = load_sequence(image_dir)           # sorted; returns (name, [PIL])
batch = preprocess_images(frames, img_size=504, patch_size=14, device="cuda", pil_masks=None)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictions = model.predict(batch)
```

Precision-sensitive parts opt out of autocast themselves (`common/amp.py`'s `force_fp32`, and
`autocast(enabled=False)` in `model/dvlt/heads.py`), so a plain bf16 autocast is correct.

`DVLT()` defaults match the released checkpoint (`depth_head_type="conv"`).
`load_pretrained` (`model/base.py`) reads only the weights file — a local file, directory, URL
or HF id — and a mismatch fails loudly under `strict=True`.

Prediction keys, from `PredictionField` (`common/constants.py:72`), batched `[B, ...]`, `B == 1`:

| Key | Shape | Note |
|---|---|---|
| `"cameras"` | `List[Cameras]`, each `[S]` | `.camera_to_worlds` `(S, 3, 4)`; `.get_intrinsics_matrices()` `(S, 3, 3)` |
| `"depths"` | `[B, S, H, W]` | |
| **`"depths_conf"`** | `[B, S, H, W]` | **the confidence key to threshold on** |
| `"world_points"` | `[B, S, H, W, 3]` | depth unprojected through the fitted pose |
| `"world_points_direct"` | `[B, S, H, W, 3]` | rays + depth; *not* what we use |
| `"world_points_direct_conf"` | `[B, S, H, W]` | |

Use `"world_points"`: it is the depth-consistent cloud that matches the poses being written,
which is what a COLMAP model must be internally consistent with.

## Implementation

Six steps, in dependency order. Every command runs on an HPG **login node** unless it is an
`sbatch` — `pip` into `/blue` needs no GPU, and the env is on shared storage.

### Step 1 — install into the RTX env

```bash
ENV=/blue/arthur.porto/srizvi63.gatech/conda/augenblick_rtx_pro_6000
SNAP=/blue/arthur.porto/srizvi63.gatech/env-snapshots
```

**1a — snapshot (DONE).** `$SNAP/rtx_pro_6000-before-dvlt.txt` records
`torch==2.9.1+cu130`, `torchvision==0.24.1+cu130`, numpy 2.2.6, and no `dvlt`. Kept on
`/blue`, not `/tmp`: it is the rollback path.

**1b — dry-run.**

```bash
$ENV/bin/python -m pip install --dry-run --no-deps --no-build-isolation -e src/libs/dvlt
```

The output must be exactly `Would install dvlt-0.0.1`. Any other package named means stop.

**1c — install.**

```bash
$ENV/bin/python -m pip install --no-deps --no-build-isolation --no-index -e src/libs/dvlt
```

`--no-deps` is load-bearing even though every dependency is present: it guarantees pip never
re-resolves `torch`/`numpy` against an env shared with four CUDA rasterisers built on its exact
torch ABI. `--no-index` forbids reaching PyPI at all; drop it only if the editable install
itself fails on it.

**1d — prove nothing moved.**

```bash
$ENV/bin/python -m pip freeze > "$SNAP/rtx_pro_6000-after-dvlt.txt"
diff "$SNAP/rtx_pro_6000-before-dvlt.txt" "$SNAP/rtx_pro_6000-after-dvlt.txt"
```

The diff must be **exactly one `>` line, for `dvlt`**. Any `<` line means an existing package
was mutated — restore it with `pip install --no-deps '<pkg>==<old version>'` from the
before-snapshot. Then:

```bash
$ENV/bin/python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"
# must still print 2.9.1+cu130 2.2.6
$ENV/bin/python -c "from dvlt.model.dvlt.model import DVLT; from dvlt.util.preprocess import load_sequence; print('dvlt OK')"
$ENV/bin/python -c "import pycolmap, augenblick; print('repo still OK')"
```

**Fresh builds:** add `$PIP install -e src/libs/dvlt --no-deps --no-build-isolation` to
`scripts/setup_common.sh`, next to the identically-flagged `src/libs/sam3` install
(`setup_common.sh:71`).

### Step 2 — model cache (already in place; verify only)

`load_pretrained("nvidia/dvlt")` downloads from HF Hub on first call (`hf_hub_download`).
`hpg_slurm/common.sh` already exports

```bash
export HF_HOME="${HF_HOME:-/blue/arthur.porto/srizvi63.gatech/cache/huggingface/}"
export TORCH_HOME="${TORCH_HOME:-/blue/arthur.porto/srizvi63.gatech/cache/torch/}"
```

and `$HF_HOME/hub/` already holds `models--facebook--sam3`, so the checkpoint lands beside it
with no code change. Do **not** use `huggingface-cli download --local-dir` — that writes a flat
snapshot outside the cache layout, which `load_pretrained` will not resolve as a cache hit.

Keep `checkpoint` as a config field (step 3) defaulting to `"nvidia/dvlt"`. If a compute node
lacks outbound HTTPS, warm the cache on the login node with `huggingface-cli download
nvidia/dvlt` (no `--local-dir`) and export `HF_HUB_OFFLINE=1` in the job.

> `DVLT()` by default (`load_patch_embed_weights=True`) fetches DINOv2 ViT-B/14-reg weights at
> construction via `torch.hub.load_state_dict_from_url` (`model/dvlt/model.py:365`, into
> `TORCH_HOME`). It is wasted work: `load_pretrained(..., strict=True)` then overwrites every
> weight. Step 3 passes `load_patch_embed_weights=False`, removing a network dependency at
> construction.

> Aside, not part of this work: `~/.cache/torch` holds 5.8 GB of VGGT/VGGSfM checkpoints from
> before the export landed; the same files now exist under `/blue/.../cache/torch`.

### Step 3 — `src/augenblick/sfm/dvlt.py`

New file, mirroring `vggt.py`'s structure — it is the reference implementation and divergence
from it is the main risk.

**`DVLTConfig`**, `@dataclass(frozen=True)`, one `field(metadata={"help": ...})` per knob so
the argparse bridge picks it up with no CLI edit:

| Field | Type / default | Meaning |
|---|---|---|
| `use_masks` | `bool = False` | Pass masks to `preprocess_images`; see step 5 |
| `seed` | `int = 42` | Same seeding block as `vggt.py` |
| `inference_steps` | `int = 12` | DVLT's `K`. The model's default; supported range 8–16 |
| `img_size` | `int = 504` | Longest edge; patch size 14 is fixed, not a flag |
| `conf_thres_value` | `float = 2.0` | Floor on `depths_conf` selecting depth pixels → points |
| `max_points` | `int = 100_000` | Matches `vggt.py`'s hardcoded `max_points_for_colmap` |
| `checkpoint` | `str = "nvidia/dvlt"` | HF id, resolved via `HF_HOME`; a local path also works |

Raise on `inference_steps < 1`; log a warning outside `[8, 16]` but proceed.

**`DVLTSfM(SfMMethod)`**, `@register_sfm`, `name: ClassVar[str] = "dvlt"`,
`config_cls: ClassVar[type] = DVLTConfig`. `SfMMethod` requires only `images/` — DVLT creates
a reconstruction, so **not** `SceneRefiner`.

**Every heavy import goes inside `run()`**, as in `vggt.py`: `torch`, `numpy`, `pycolmap`,
`trimesh`, and all `dvlt.*` imports. This keeps `augenblick sfm --list` importable on a
login node and lets the method register in an env without `dvlt` (e.g. `augenblick_b200`).

Body, in order:

1. `self.validate(scene)`, banner block (same layout as `vggt.py`), `t_start = time.time()`.
2. The seeding block copied from `vggt.py`.
3. `name, frames = load_sequence(scene.images_dir)`. **Sorts internally** by `IMAGE_EXTS`
   (`util/preprocess.py:25`), so order is reproducible — unlike `vggt.py`'s bare `glob.glob`.
   Build `base_image_path_list` with the *same* sort and filter (import `IMAGE_EXTS`, don't
   redefine it): COLMAP image names are basenames and must align index-for-index with
   `frames`.
4. `DVLT(img_size=cfg.img_size, inference_steps=cfg.inference_steps,
   load_patch_embed_weights=False)`, `load_pretrained(cfg.checkpoint, strict=True)`,
   `setup_test(device)`. Log `Model loaded in {t:.1f}s`, the label `vggt.py` uses.
5. `preprocess_images(frames, img_size=cfg.img_size, patch_size=14, device=device,
   pil_masks=...)` (step 5), then `model.predict(batch)` under `torch.no_grad()` and
   `torch.autocast("cuda", dtype=torch.bfloat16)`. Log elapsed inference time.
6. Step 4's conversions, then `batch_np_matrix_to_pycolmap_wo_track` (as `vggt.py`'s no-BA
   branch), the rescale to original resolution (step 4b), `reconstruction.write` into
   `output_dir/sparse/0`, and the `points.ply` export.
7. Link `images/` and (if present) `masks/` into `output_dir`. **Do not copy `vggt.py`'s
   `_link_dir`** — it carries a non-obvious `lexists`-vs-`exists` fix for dangling links. Move
   it to `sfm/base.py` as public `link_dir` and have both adapters import it.
8. Return `SfMResult(output_dir=..., elapsed=..., scene=Scene(output_dir),
   num_images=reconstruction.num_reg_images(), num_points=reconstruction.num_points3D())`.

Log the `Pipeline complete` block with `vggt.py`'s field labels (`Total time:`, `Output:`).

### Step 4 — the three conversions that are easy to get wrong

Each is a silent failure if wrong: the job exits 0 and the mesh is garbage hours later.

**(a) Extrinsics are camera-to-world; COLMAP wants world-to-camera.** DVLT returns
`cameras.camera_to_worlds`, `(S, 3, 4)`, "per-image c2w matrices, in `[R | t]` format"
(`struct/cameras.py:61`). `batch_np_matrix_to_pycolmap_wo_track` expects the `cam_from_world`
convention `run_VGGT` produces. Invert explicitly:

```python
R_c2w, t_c2w = c2w[:, :3, :3], c2w[:, :3, 3]
R_w2c = R_c2w.transpose(0, 2, 1)
t_w2c = -np.einsum("sij,sj->si", R_w2c, t_c2w)
extrinsic = np.concatenate([R_w2c, t_w2c[..., None]], axis=-1)   # (S, 3, 4)
```

"Both are OpenCV convention" refers to axis signs, which match; the transform direction does
not. Check 3 exists solely to catch this.

**(b) Resolution: crop, not pad — and the intrinsics already account for it.**
`preprocess_images` (`util/preprocess.py:149`) resizes the longest side to `img_size`, then
**centre-crops** H and W to multiples of 14 (`preprocess.py:185-186`). Padding happens only
when images differ in aspect ratio; these scenes share one resolution.

- The output is **non-square** (640×480 → 504×378, observed), so `vggt.py`'s square
  `image_size` shortcut does not hold. Take `(H, W)` from `predictions["depths"]`.
- **Do not hand-derive the principal point.** `get_intrinsics_matrices()`
  (`struct/cameras.py:952`) returns K already in the cropped frame. Rescale it to the original
  resolution — but `rename_colmap_recons_and_rescale_camera()` **assumes VGGT's square pad**
  (it takes `original_coords` from `load_and_preprocess_images_square` and a scalar
  `img_size`). Either build an equivalent `original_coords` for DVLT's crop or write a small
  DVLT-specific rescale. Scale is `original_longest_edge / img_size`, plus the crop offset
  `top = (new_h - crop_h) // 2`, `left = (new_w - crop_w) // 2`, exact from the PIL size since
  `preprocess_images` is deterministic. Decide which, and comment it.

AND the batch's `gradio_valid_pixels` (`[1, S, H, W]`, True on real pixels,
`preprocess.py:228`) into the confidence mask — free when no padding occurred.

**(c) Points and colours.** `predictions["world_points"]` is `[1, S, H, W, 3]` **already in
world space** — no `unproject_depth_map_to_point_map` (that would double-transform). Index
`[0]`, flatten, build `points_xyf` with `create_pixel_coordinate_grid(num_frames, H, W)` from
`vggt.utils.helper`.

Colours from `batch[DataField.IMAGES][0]`, `(S, C, H, W)` — the preprocessed tensor, already
aligned with the points. Cast through `.float()` before `.numpy()` (bf16 has no numpy dtype).

Then, as `vggt.py`'s no-BA branch:

```python
conf_mask = depths_conf >= cfg.conf_thres_value
conf_mask = randomly_limit_trues(conf_mask, cfg.max_points)
points_3d, points_xyf, points_rgb = (a[conf_mask] for a in (points_3d, points_xyf, points_rgb))
```

`conf_thres_value = 2.0` is inherited from VGGT, but **DVLT's confidence is a different
quantity** — check 5 calibrates it.

### Step 5 — masks

**DVLT has native mask support; use it.** `preprocess_images` takes `pil_masks` and runs each
through the *identical* resize/crop pipeline as its image with NEAREST interpolation — the
resolution-alignment problem solved by construction.

When `cfg.use_masks`, build the list with
`load_masks_for_directory(scene.masks_dir, scene.images_dir)` (`preprocess.py:110`). It pairs
**by stem** against the same sorted list `load_sequence` uses. HPG scenes match:
`images/IMG_5663.jpg` ↔ `masks/IMG_5663.png`. Point it at `masks/`, never `masks_colmap/`
(COLMAP's `foo.jpg.png` naming).

- It **raises `FileNotFoundError` if any image lacks a mask** (`preprocess.py:142`), unlike
  `vggt.py` where a missing mask means "unconstrained". Real data hits this:
  `UF_Mammals_10941_skull` has 276 images, 271 masks. Either fall back to an all-true mask for
  the missing ones (the repo's convention) or let it raise and document 1:1 coverage. **Decide
  explicitly and comment it** — silently zeroing an unmasked frame is the failure to avoid.
- Masks must be **at original image resolution**; `prepare_uf_dataset.py` and `augenblick mask`
  both satisfy this.

The mask arrives as `batch[SEGMENTATION_MASK_FIELD]` (`"gradio_segmentation_mask"`,
`[1, S, H, W]` bool, True = keep; constant at `preprocess.py:30`). **The model does not read
it** — AND it into `conf_mask` in step 4c. No `mask_erode_px` equivalent: erosion exists in the
VGGT path to protect the *tracker*, and this path has none.

### Step 6 — register it, and add the SLURM job

`src/augenblick/sfm/__init__.py`:

```python
from augenblick.sfm import colmap, dvlt, hull, turntable, vggt  # noqa: F401
```

Registration fires on import; nothing in `cli/main.py` changes.

Add **`hpg_slurm/dvlt_sfm.sbatch`**, copied from `hpg_slurm/vggt_sfm.sbatch`. Changes:

- `--job-name=dvlt-sfm` — the timing CSV's `method` column and `report_sfm_results.py` both
  key off the job name generically; neither needs an edit.
- Keep `--account=arthur.porto`, `--partition=hpg-rtx6000`, `--gpus=1`, `--array=0-75%8`.
- Refuse the B200 until its env is rebuilt, right after `source common.sh`:
  ```bash
  if [ "$GPU" != rtx6000 ]; then
      echo "ERROR: dvlt is installed only in augenblick_rtx_pro_6000 (GPU=$GPU)" >&2; exit 2
  fi
  ```
- `--mem=48gb`, `--time=2:00:00` to start (VGGT gets 96 gb / 4 h); raise on `HOST_OOM` /
  `KILLED_OR_TIMEOUT`.
- `OUT="$RESULT_ROOT/$SCENE_NAME/dvlt"`.
- `SCENE_INFO ... sfm=dvlt ...` and `start_timing "$SCENE_NAME" "$N_IMAGES" dvlt` (the third
  argument emits the `RESULT` marker the scraper joins on).
- final line: `augenblick sfm dvlt --scene "$SCENE" --output "$OUT" "$@"`
- header comment: drop VGGT's "big scenes want GPU=b200" line.

`common.sh` needs **no change**. Add the script to `hpg_slurm/README.md`'s Files table and
Submit block. `SFM=dvlt BACKEND=2dgs sbatch hpg_slurm/recon.sbatch` then works unchanged
(`recon.sbatch` reads `$RESULT_ROOT/<scene>/$SFM`).

**Whole cards, so timings compare directly.** `--gpus=1` on `hpg-rtx6000` is a whole RTX PRO
6000 Blackwell (45 nodes × 8); the last VGGT array logged `gpu actual : rtx6000` on all nine
tasks. VGGT runs on the same partition, so DVLT-vs-VGGT timing is like-for-like. Still read the
CSV's `gpu` column before quoting a number.

## Verification

**Run through `hpg_slurm/`, not `augenblick` directly** — a login node has no GPU, and batch
jobs don't source `~/.bashrc`. Submit from the repo root; save sbatch edits *before*
submitting (Slurm snapshots the script).

### Scene and baseline

**`--array=14`, `UF_Mammals_1199_mandible`** (default neurips `DATA_ROOT`): 276 images, 276
masks, tied smallest. Index confirmed from job logs (`scene : UF_Mammals_1199_mandible (14 of
76)`) — `discover_scenes` uses a locale-dependent `sort`, so trust logs over a local listing.

```bash
sbatch --array=14 hpg_slurm/vggt_sfm.sbatch     # the comparison
sbatch --array=14 hpg_slurm/dvlt_sfm.sbatch     # the new path
```

Outputs: `$RESULT_ROOT/UF_Mammals_1199_mandible/{vggt,dvlt}`. No `vggt/` output exists there
yet, so the baseline must actually run. Re-running a script overwrites in place.

For check 7, `--array=11`, `UF_Mammals_10941_skull` (276 images / 271 masks) is a real
instance of the missing-mask case.

### Checks, in the order they would catch a mistake

0. **Nothing in the env moved.** Step 1d's diff is exactly one `> dvlt` line; the import
   checks pass; `augenblick recon --list` still resolves.
1. **CPU-only registration.** On the login node, `augenblick sfm --list` lists `dvlt` and
   `augenblick sfm dvlt --help` prints every `DVLTConfig` field. Repeat from `augenblick_b200`
   (no `dvlt`) — it must still list `dvlt`. Proves step 3's imports are function-local.
2. **Weights cached to `/blue`.** After the first job,
   `/blue/.../cache/huggingface/hub/` holds `models--nvidia--dvlt`, and `~/.cache/huggingface`
   (48 KB today) and `~/.cache/torch` did not grow.
3. **Poses are not inverted.** Print camera centres from `sparse/0/images.bin`: for a
   turntable scene they lie on a **ring around** the subject. Compare to VGGT's on the same
   scene — similar up to a similarity transform, not mirrored.
4. **Points land on the subject.** Project `points3D.bin` into a few views and confirm points
   fall inside the mask. A crop-offset or scale error (step 4b) shows as a consistent pixel
   offset.
5. **Confidence threshold calibrated.** Log `depths_conf` percentiles on the first run. If
   `>= 2.0` keeps nearly everything or nearly nothing, change the default and record why.
6. **Masking does something.** `--use_masks` gives **substantially** fewer points than
   unmasked. Marginal means the mask AND silently no-op'd.
7. **The missing-mask decision holds.** `--array=11 ... --use_masks` either errors naming the
   missing file or keeps the five unmasked frames contributing — never zero points from them.
8. **`inference_steps` moves compute.** `--inference_steps 8` vs `16` shows materially
   different inference time. If not, `K` isn't reaching the constructor.
9. **Usable scene.** `SFM=dvlt BACKEND=2dgs sbatch --array=14 hpg_slurm/recon.sbatch
   --iterations 2000` produces a mesh. **Smoke value only** — never quote that mesh.

Checks 0–3 first: 0 protects the shared env, 1 is free, 2 catches a misplaced cache, 3 catches
the error hardest to notice downstream.

`scene_common.sh` appends one row per job to `$RESULT_ROOT/_timing/sfm_timings.csv`
(`method,scene,gpu,n_images,seconds,exit_code,note,jobid,node,finished`), classifying failures
as `UNSUPPORTED_ARCH` / `GPU_OOM` / `HOST_OOM` / `KILLED_OR_TIMEOUT` / `FAILED`.
`python hpg_slurm/report_sfm_results.py --job-id <vggt> --job-id <dvlt>` joins both runs.

> A `GPU_OOM` row is **a result, not a run to retry** — record it. At 117M parameters an OOM
> where VGGT succeeded is a real signal worth chasing.

## Risks

- **Shared-env contamination.** One package added, but a forgotten `--no-deps` could still let
  pip re-resolve against the env. Mitigations: `--dry-run` first, and check 0's one-line diff.
- **Extrinsic inversion (step 4a).** Highest-probability failure; silent. Check 3.
- **Crop offset (step 4b).** Highest-*impact* failure: a principal-point error degrades every
  downstream reconstruction subtly. The VGGT rescale helper must not be reused blindly. Check 4.
- **Confidence scale (step 4c).** 2.0 is a guess. Check 5.
- **The GPU path is unexercised.** `dvlt` has been imported and run on CPU in the RTX env only;
  the first job is the first bf16/CUDA-kernel run (attention backend selection in
  `model_components/layers/attention.py`, `force_fp32` islands under autocast).
- **Checkpoint drift.** `checkpoint="nvidia/dvlt"` resolves to the Hub's latest revision. An
  updated upload would change results silently between runs (or fail `strict=True` if the
  architecture changed). `load_pretrained` accepts `revision=`; pin it if results must be
  reproducible across time.
- **B200 env drift.** Out of scope; the sbatch guard keeps a `GPU=b200` submission from
  failing obscurely mid-run.
- **Weights licence.** NVIDIA licence, research/evaluation only. Never commit the checkpoint;
  never quote DVLT results commercially.

## Documentation to update when this lands

Per [repo-conventions.md](../MEMORY/repo-conventions.md), detail goes in `MEMORY/`:

- **New `MEMORY/backend-dvlt.md`** — architecture, `inference_steps` as a constructor arg, the
  `setup_test(device)`/`predict(batch)` API (and that the library's README does not match it),
  the prediction-key table, the three conversions, native masks and the 1:1 coverage rule, the
  licence restriction. Index it in CLAUDE.md's MEMORY table.
- [pipeline-sfm.md](../MEMORY/pipeline-sfm.md) — an `augenblick sfm dvlt` section with the full
  flag list.
- [environment-and-gpu.md](../MEMORY/environment-and-gpu.md) — `dvlt` installs into
  `augenblick_rtx_pro_6000` via `--no-deps` with no added packages, the `setup_common.sh`
  line, and that the B200 env doesn't carry it (and why).
- [cluster-slurm.md](../MEMORY/cluster-slurm.md) — a `dvlt_sfm.sbatch` row, HPG-only and
  RTX-only.
- [augenblick-package.md](../MEMORY/augenblick-package.md) — `_link_dir` moved to
  `sfm/base.py` as `link_dir`.
- [repo-conventions.md](../MEMORY/repo-conventions.md) — `src/libs/dvlt/` in the layout block;
  add this file to the "Landed" list.
- **CLAUDE.md** — `dvlt` in the SfM row of the stage table and in the `src/libs/` list; "all
  four SfM entry points" → five in the MEMORY index.
