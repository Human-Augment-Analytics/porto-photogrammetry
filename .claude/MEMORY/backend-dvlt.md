# `src/libs/dvlt/` — DVLT (NVIDIA Deja View looping transformer)

Feed-forward multi-view geometry: a DINOv2 patch embedding feeds a recurrent alternating-attention
block looped K times (`inference_steps`), then per-frame ray and depth decoders. The fitted
camera-to-world poses are OpenCV-convention, so only the inversion to world-to-camera separates
them from COLMAP. Wrapper: `augenblick sfm dvlt` (`src/augenblick/sfm/dvlt.py`); flags and
behaviour in [pipeline-sfm.md](pipeline-sfm.md).

The adapter reuses VGGT's COLMAP writer (`batch_np_matrix_to_pycolmap_wo_track`) and pixel-grid
helpers, so `src/libs/vggt` must be installed alongside it.

## Install

`scripts/setup_common.sh` stage 4/9 installs it editable with **`--no-deps`**, beside SAM 3: its
dependency list would otherwise pull torch/numpy past the constraint pins. Its real deps
(`huggingface-hub`, `safetensors`, `open3d`, ...) already come from `requirements.txt`. Weights
load through `huggingface_hub`, so the `transformers` pin in
[environment-and-gpu.md](environment-and-gpu.md) matters here too.

On HPG only `augenblick_rtx_pro_6000` carries it; `hpg_slurm/dvlt_sfm.sbatch` refuses
`GPU=b200` because that env predates DVLT (and SAM 3 and GeoSVR).

## Vendoring

Vendored in-tree (not a submodule), **trimmed to the inference path**: model + preprocessing
only. Training, eval, demo code and the baseline wrappers its README describes are not present;
train or benchmark with upstream. No upstream commit is recorded in the tree.

Local patches, each marked `LOCAL PATCH (augenblick)`:

| File | Change |
|---|---|
| `pyproject.toml` | `requires-python` relaxed `>=3.11` → `>=3.10` (pure Python, compiles clean); dependency list cut to what the trimmed code imports |
| `model/base.py` | inference-only, no `accelerate`: stdlib logging, `setup_test(device)`, `predict` without an accelerator; train/save methods removed |
| `model/dvlt/model.py` | training losses removed (`loss` kept in the signature but must be unset); **`use_seg_mask_for_pose`** weights the pose fit by the batch segmentation mask — backs `--mask_pose_fit` |
| `model_components/__init__.py` | `loss` sub-package imports removed |

## Licence

Code: Apache-2.0 (with third-party notices in `LICENSES/`, `THIRD_PARTY_LICENSES.md`).
**Weights** (`nvidia/dvlt`): NVIDIA License, non-commercial research/evaluation only. They are
fetched at runtime into `$HF_HOME` and never committed.
