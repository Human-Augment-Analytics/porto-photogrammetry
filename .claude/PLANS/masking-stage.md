# Plan: a `mask` stage in `src/augenblick/`

Implementation spec, written before the work landed. Historical once it lands — where this and
the code disagree, the code wins; the durable description belongs in
[augenblick-package.md](../MEMORY/augenblick-package.md).

## Goal

A third stage, `augenblick mask <method> --images <dir> --output <dir>`, that **produces** a
scene's `masks/` from a flat folder of photographs, alongside `sfm` and `recon` which
**consume** it.

```bash
augenblick mask rembg --images data/neurips/prepared/UF_Fish_20486_skull/images \
                     --output data/neurips/prepared/UF_Fish_20486_skull_masked
```

writes `<output>/images/` (symlink to the input directory) and `<output>/masks/` holding one
binary PNG per image. The result is a COLMAP-shaped scene ready for
`augenblick sfm vggt --use_ba --use_masks --scene <output>`.

**Input:** a flat directory of `.jpg`/`.JPG`/`.jpeg`. Nothing else read or expected — no
sibling `masks/`, `sparse/0/`, or `split.json`. This is the difference from `sfm`/`recon`,
which take a `--scene` with the full COLMAP layout.

**Non-goals:** changing any mask consumer; interactive segmentation; mutating the input
directory.

## Why a stage

**9 of 47 neurips scenes ship no masks, all >=432 images.** Unmasked BA failed on 36 of 47
scenes with *No reconstruction can be built with BA* — too few surviving correspondences.
Counted directly:

```bash
for d in data/neurips/prepared/*/; do
  i=$(ls "$d/images" | wc -l); m=$(ls "$d/masks" 2>/dev/null | wc -l)
  [ "$m" -eq 0 ] && echo "NOMASK $(basename $d) images=$i"
done
```

`UF_Fish_20486_skull` 663, `UF_Fish_31766_skull` 657, `UF_Mammals_14760_mandible` 474,
`UF_Mammals_11436_cranium` 442, `UF_Mammals_14760_cranium` 438, `UF_Fish_9627_skull` 434,
`UF_Mammals_26153_mandible` 434, `UF_Mammals_1769_mandible` 432,
`UF_Mammals_19126_cranium` 432. `data/main` has the same hole: `TH24-21_Birdsnest`, 159/0.

These scenes fail three code paths: `sfm hull` raises without `masks/` (`hull.py:118-119`),
`eval/nvs.py` scores them unmasked ("mostly reward reproducing the black surround"), masked
BA cannot run. One stage unblocks all three.

**Two neurips scenes are partially masked** — `UF_Mammals_1398_mandible` 275/276,
`UF_Mammals_4991_mandible` 271/276. `Scene.has_masks()` tests only non-emptiness, so
consumers treat them as complete and silently skip gap views. Not in scope: fixing them
requires reading input masks, which this stage refuses to do.

## Verified state of the tree

Line numbers as of the branch this was written on.

**The ABC to extend.** `Method` (`core/method.py:29-62`) gives `name`, `config_cls`,
`validate(scene)`, `run(scene, output_dir) -> StageResult`. `SfMMethod` and
`ReconstructionMethod` are its subclasses; masking is a third sibling.

### The mask contract

Four readers, all agreeing:

| Reader | Location | Resolution |
|---|---|---|
| `hull` | `sfm/hull.py:214-217` | `masks_dir / f"{stem}.png"`, `.convert("L") > 127` |
| `eval/nvs` | `eval/nvs.py:69-73` | same |
| COLMAP SfM | `core/scene.py:65-83` | symlinks `<stem>.png` -> `<stem>.jpg.png` |
| `prepare_uf_dataset.py` | module docstring | writes `masks/<image stem>.png` |

> **A mask is a PNG named `<image stem>.png`, greyscale-readable, pixel > 127 = foreground.**

### Existing masks are RGB and soft-edged

Measured on `data/neurips/prepared/UF_Herp_3998`: image `(6240, 4160) RGB`, mask same shape
and mode, values `[0 2 3 4 5 ...]`, not `{0, 255}`; foreground fraction 0.241.

1. **Write mode `L`, not RGB.** 26 MP x 663 images x RGB is 3x scene size for no info; every
   consumer hard-thresholds at 127.
2. **Emit clean `{0, 255}`.** Compatible with `> 127` readers.
3. **Mask dimensions must equal image dimensions.** `hull._carve` indexes `mask[vi, ui]`
   after scaling; `eval/nvs._load_mask` resizes with NEAREST. A wrong aspect ratio misaligns
   *silently*.

### The CLI's two-stage assumption

`grep -rn 'stage == "sfm"\|SFM_REGISTRY\|RECONSTRUCTION_REGISTRY' src/ tests/` -> four places
in `cli/main.py`: lines 45-46 (two `_add_stage_parser`), 76
(`registry = SFM_REGISTRY if args.stage == "sfm" else RECONSTRUCTION_REGISTRY`), 84
(`cls = get_sfm(...) if args.stage == "sfm" else get_reconstruction(...)`).

Lines 76 and 84 are binaries: a `mask` invocation falls through to the reconstruction
registry. Step 4 replaces both with a table lookup, not a third arm. `registry.py` is
already generic; only its public wrappers are per-stage.

The masking stage takes `--images`, not `--scene` — its input is not a scene. Step 4 threads
that through the CLI via one `input_flag` field per `STAGES` entry.

### What is installed

**Do not use `pace_pip_list.txt`.** Untracked scratch from a different checkout (its `vggt`
points at `/storage/ice1/8/9/srizvi63/augenblick/src`), lists packages this repo never
installs, disagrees on versions. Authoritative: `requirements.txt`, `constraints/numpy{1,2}.txt`,
`scripts/setup_common.sh`, and the envs themselves.

Measured in all three PACE envs (`~/scratch/conda/augenblick_{a100,a40,l40s}`) — **identical**,
matching `requirements.txt` + `constraints/numpy2.txt`:

| Package | Version | Relevance |
|---|---|---|
| Python / `numpy` / `scipy` | 3.10.21 / 2.2.6 / 1.15.3 | `scipy.ndimage` for post-processing |
| `torch` / `torchvision` | 2.9.1+cu130 / 0.24.1+cu130 | working CUDA |
| `opencv-python` | 4.11.0.86 | GrabCut |
| `scikit-image` / `Pillow` | 0.25.2 / 12.3.0 | Otsu / PNG I/O |
| `onnxruntime` | 1.23.2 | CPU-only build; **shared envs stay untouched** |
| `rembg`, `pymatting`, `pooch`, `numba` | *absent* | live only in the new `augenblick_masked` env (step 6) |

Only numpy 2 is built here (no `augenblick_b200`). Write for both; verify under numpy 2.

`scripts/setup_common.sh` step `2/8` is `pip install -r requirements.txt` (line 62), so a
new `requirements.txt` entry is picked up by any env rebuild; `PIP_CONSTRAINT` is exported
globally (line 49) to catch transitive upgrades — anything not pinned there is a live risk.

**`onnxruntime` today is CPU-only:**

```
$ ~/scratch/conda/augenblick_a100/bin/python -c \
    "import onnxruntime as o; print(o.get_available_providers())"
['AzureExecutionProvider', 'CPUExecutionProvider']
```

`requirements.txt:16` pins plain `onnxruntime`; it conflicts on package name with
`onnxruntime-gpu`. **The shared envs are not changed** — step 6 builds a new env,
`augenblick_masked`, from its own installer, with `onnxruntime-gpu` in place of
`onnxruntime`. Steps 8 and 9 activate that new env; every existing SfM/recon job
continues to run against the shared envs unchanged. Precedent for runtime ONNX model
download: `src/utils/visual_util.py:88-120` fetches `skyseg.onnx` on demand.

**`rembg` transitives, measured:**

```bash
PIP_CONSTRAINT=constraints/numpy2.txt \
  ~/scratch/conda/augenblick_a100/bin/python -m pip install --dry-run rembg==2.0.68
# Would install PyMatting-1.1.16 llvmlite-0.49.0 numba-0.67.0
#               opencv-python-headless-5.0.0.93 pooch-1.9.0 rembg-2.0.68
```

Three concerns for step 6:

1. **`rembg` has no default `onnxruntime` dep** — only via extras
   (`Requires-Dist: onnxruntime-gpu; extra == "gpu"`). Bare `rembg` has no inference
   runtime; pin **`onnxruntime-gpu`** explicitly alongside plain `rembg`.
2. **`opencv-python-headless` 5.0.0.93 collides with the pinned `opencv-python` 4.11.0.86.**
   Both own `cv2`; the survivor depends on install order. Because `augenblick_masked` is
   a fresh env, step 6's installer controls the order (`opencv-python==4.11.0.86` after
   `rembg`, so it wins). No edit to the shared constraints files.
3. **`numba` 0.67.0 needs `numpy<2.6,>=1.22`** — satisfied by 2.2.6. Adds `llvmlite` weight
   but no ABI hazard.

### Environment constraints

- No `dependencies` in `pyproject.toml`. Mask deps go in `requirements.txt`.
- Write for both numpy generations. Avoid `np.bool8`-era aliases and numpy-2-only `copy=`
  semantics.
- Function-local `torch`/`cv2`/`skimage`/`numpy`/`rembg` imports (as `hull.py:134-136`,
  `colmap.py:45`). `sfm/__init__.py` imports every method eagerly to fire registration; a
  module-level heavy import breaks `--list` on a login node.
- `Image.MAX_IMAGE_PIXELS = None` in every method, per `hull.py:139`.

## Design

```
src/augenblick/masking/
    __init__.py      imports the modules, firing registration
    base.py          MaskMethod ABC + MaskResult + shared write/IO helpers
    threshold.py     classical CV: Otsu / GrabCut; no new dependency, no GPU
    rembg.py         learned U^2-Net matting via rembg (ONNX)
```

Mirrors `sfm/` in shape and in base class: `MaskMethod` remains a `Method` subclass. What
differs is the **input type** — a flat images directory rather than a `Scene` — captured
by an input mixin: `SceneInputMixin` for `sfm`/`recon`, `ImagesInputMixin` for `mask`.

### Output contract

Every method writes a scene-shaped output directory. The input `--images` directory is never
mutated:

| `<output>/…` | How | Why |
|---|---|---|
| `images/` | symlink to the resolved input | never copy 663 x 26 MP photographs |
| `masks/` | real directory of written PNGs | the stage's product |

`<output>/masks/<image-stem>.png`, mode `L`, `{0, 255}`, dimensions equal to source.

Nothing else is written: no `sparse/0/`, no `split.json`. A later `sfm`/`recon` writes those
into the same output on its own run.

### The input mixins on `Method`

```python
# src/augenblick/core/method.py — reshaped, not rewritten. Same StageResult, same
# `from_namespace`, same registry story.

from typing import Generic, TypeVar
Input = TypeVar("Input")


class Method(ABC, Generic[Input]):
    """Base for any pipeline stage that transforms an input directory."""

    name: ClassVar[str]
    config_cls: ClassVar[type]
    accepts_passthrough: ClassVar[bool] = False

    def __init__(self, config):
        self.config = config

    @classmethod
    def from_namespace(cls, ns):
        return cls(config_from_namespace(cls.config_cls, ns))

    @classmethod
    @abstractmethod
    def build_input(cls, path: Path) -> Input:
        """Wrap the resolved --scene or --images path into the method's input type."""

    @abstractmethod
    def validate(self, inp: Input) -> None: ...

    @abstractmethod
    def run(self, inp: Input, output_dir: Path) -> StageResult: ...


class SceneInputMixin:
    """Input: a COLMAP Scene."""

    @classmethod
    def build_input(cls, path: Path) -> Scene:
        return Scene(path)

    def validate(self, scene: Scene) -> None:
        scene.require_images()


class ImagesInputMixin:
    """Input: a flat directory of images."""

    IMAGE_SUFFIXES: ClassVar[frozenset] = frozenset(
        {".jpg", ".jpeg", ".JPG", ".JPEG"})

    @classmethod
    def build_input(cls, path: Path) -> Path:
        return path

    def validate(self, images_dir: Path) -> None:
        from augenblick.core.errors import SceneError
        if not images_dir.is_dir():
            raise SceneError(f"no images directory at {images_dir}")
        if not any(p.suffix in self.IMAGE_SUFFIXES for p in images_dir.iterdir()):
            raise SceneError(
                f"{images_dir} has no files with an accepted suffix "
                f"({sorted(self.IMAGE_SUFFIXES)})")
```

Knock-on edits (all in step 1):

- `src/augenblick/sfm/base.py`: `class SfMMethod(SceneInputMixin, Method[Scene])`. Delete
  `SfMMethod.validate` — the mixin owns it. `SceneRefiner.validate` still calls
  `super().validate(scene)` then `scene.require_reconstruction()`.
- `src/augenblick/reconstruction/base.py`: `class ReconstructionMethod(SceneInputMixin,
  Method[Scene])`. `validate` keeps `require_reconstruction()` and calls `super().validate`.

Existing tests continue to pass.

### `MaskMethod`

`src/augenblick/masking/base.py`:

```python
"""Shared shape of the masking methods, and the mask-writing contract they all satisfy."""
import logging, os, time
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from augenblick.core.errors import SceneError
from augenblick.core.method import ImagesInputMixin, Method, StageResult
from augenblick.core.timing import StageTimer

logger = logging.getLogger(__name__)


@dataclass
class MaskResult(StageResult):
    """A completed masking run: the output directory plus per-image mask counts."""

    num_images: int = 0
    num_written: int = 0
    num_reused: int = 0
    num_failed: int = 0


@dataclass(frozen=True)
class MaskCommonConfig:
    """Fields every masking method shares."""

    only_missing: bool = field(default=False, metadata={
        "help": "Skip images that already have <stem>.png in <output>/masks/"})
    min_foreground: float = field(default=0.005, metadata={
        "help": "Reject a mask below this foreground fraction (counts as num_failed)"})
    max_foreground: float = field(default=0.95, metadata={
        "help": "Reject a mask above this foreground fraction"})
    keep_largest: bool = field(default=True, metadata={
        "help": "Keep only the largest connected foreground component"})
    fill_holes: bool = field(default=True, metadata={
        "help": "Fill enclosed background holes inside the silhouette"})


class MaskMethod(ImagesInputMixin, Method[Path]):
    """Consumes a directory of images, produces a scene-shaped output with masks/."""

    title: ClassVar[str]

    @abstractmethod
    def mask_for(self, image_path: Path):
        """Return a bool numpy array [H, W], True on the specimen, matching image dims."""

    def run(self, images_dir: Path, output_dir: Path) -> MaskResult:
        """Segment every image; write masks under output_dir/masks/."""
        # Implemented once on the class; see "The shared run loop".
```

`IMAGE_SUFFIXES` and `validate` come from `ImagesInputMixin`; `from_namespace` and the
`config` init come from `Method`. `MaskMethod.run()` is implemented once — as
`SubprocessBackend.run()` is for the four recon backends — so adding a technique is one
`mask_for()`.

### The shared run loop

1. `self.validate(images_dir)`; `t0 = time.time()`; `out_dir = output_dir.resolve()`.
2. `Image.MAX_IMAGE_PIXELS = None`.
3. Skeleton: `out_dir.mkdir(parents=True, exist_ok=True)`. Symlink `out_dir/"images"` ->
   `images_dir.resolve()` (skip if it exists and points there; `SceneError` if elsewhere).
   `mkdir` `out_dir/"masks"`.
4. Enumerate `sorted(p for p in images_dir.iterdir() if p.suffix in IMAGE_SUFFIXES)`.
5. `StageTimer(self.title, 1, header)`, loop inside one `timer.stage("Masking")`.
6. Per image:
   - `only_missing` and mask exists -> `num_reused += 1`, continue.
   - `self.mask_for(path)`; on **any** exception log a warning, `num_failed += 1`, continue.
   - `postprocess_mask()`; check foreground bounds; reject -> `num_failed`, write no file.
   - `write_mask()` otherwise; `num_written += 1`.
7. `timer.summary({...})`, return `MaskResult`.

Consumers treat a missing mask as "skip this view" (`hull._load_masks` drops the camera,
`eval/nvs._load_mask` returns `None`). An all-foreground mask would silently disable
masking while looking successful, hence no-file-on-reject.

`--only-missing` refuses to overwrite: it exists for resume. A full redo is `rm -rf
<output>/masks/`.

### Shared helpers in `base.py`

```python
def postprocess_mask(mask, keep_largest: bool, fill_holes: bool):
    """Drop specks (scipy.ndimage.label), close holes (binary_fill_holes)."""


def foreground_fraction(mask) -> float:
    """Fraction of pixels the mask calls foreground."""


def write_mask(mask, dest: Path) -> None:
    """Write bool mask as 8-bit greyscale PNG; consumers threshold at >127."""
    # Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(dest, optimize=True)
```

`scipy` is pinned in both constraints and already used across the repo; prefer it to `cv2`
here.

### Method 1: `rembg`, the default {#method-rembg}

`masking/rembg.py`, registered `rembg`. Default in the sbatch.

```python
from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass(frozen=True)
class RembgConfig(MaskCommonConfig):
    """U^2-Net matting parameters, run through rembg's ONNX session."""

    model: str = field(default="isnet-general-use", metadata={
        "help": "rembg model, e.g. u2net, isnet-general-use, birefnet-general"})
    alpha_threshold: int = field(default=127, metadata={
        "help": "Alpha above which a pixel is foreground; matches the consumers' >127"})
    providers: Optional[str] = field(default=None, metadata={
        "help": "Comma-separated ONNX providers; default lets onnxruntime pick "
                "(CUDAExecutionProvider first when GPU is available)"})
    batch_log_every: int = field(default=25, metadata={
        "help": "Log progress every N images"})
```

`mask_for()` runs the session, takes the alpha channel, thresholds at `alpha_threshold`.

- Session created **once**, lazily on the instance (as `visual_util.py:112-120` does).
- `alpha_threshold=127` matches `> 127` readers; keeping them equal round-trips.
- **~0.1-0.3 s/image on GPU**, ~1-3 s/image on CPU. If the log shows only
  `CPUExecutionProvider`, the env was built wrong.

`torchvision` segmentation heads are COCO/VOC-class and have no class for *a fish skull on
a turntable*; salient-object matting is the right shape.

### Method 2: `threshold`, the no-dependency fallback {#method-threshold}

`masking/threshold.py`, registered `threshold`. Runs on CPU with `cv2`/`skimage`/`scipy`, all
already pinned.

```python
@dataclass(frozen=True)
class ThresholdConfig(MaskCommonConfig):
    """Classical background-separation."""

    mode: Literal["otsu", "grabcut"] = field(default="otsu", metadata={
        "help": "otsu: global luminance threshold; grabcut: iterative graph cut"})
    polarity: Literal["auto", "dark-background", "light-background"] = field(
        default="auto", metadata={
            "help": "Which side is background. Turntable backdrops are typically lighter "
                    "but occasionally darker; 'auto' picks per image via border-vs-centre "
                    "luminance"})
    blur_px: int = field(default=3, metadata={
        "help": "Gaussian blur radius before thresholding (0 disables)"})
    grabcut_iters: int = field(default=5, metadata={"help": "GrabCut iterations"})
    border_px: int = field(default=32, metadata={
        "help": "Border inset for GrabCut initial rectangle"})
    downscale: int = field(default=4, metadata={
        "help": "Segment at 1/N res then nearest-upsample (1 disables)"})
```

`mask_for()`: load `L`, optional blur, optional downscale. `otsu` -> `threshold_otsu` with
`polarity` deciding which side is foreground. `grabcut` -> `cv2.grabCut` with
`GC_INIT_WITH_RECT`. Upsample with nearest-neighbour (bilinear reintroduces the soft edge).

**Polarity auto:** compare mean luminance of a `border_px`-wide frame vs the central
`border_px`-wide crop. Border brighter → light backdrop, keep darker pixels.
`downscale=4` cuts 26 MP GrabCut from minutes to tractable while preserving the dimensions
contract via nearest upsample.

## Implementation steps

### Step 1 — input mixins on `Method`

Lands **first**, in its own commit. Edit `src/augenblick/core/method.py` per
[The input mixins on `Method`](#the-input-mixins-on-method):

- `Method` becomes `Generic[Input]`; `validate`/`run` are typed on `Input`.
- Add abstract `build_input(cls, path: Path) -> Input`.
- Add `SceneInputMixin` and `ImagesInputMixin`.

Same commit:

- `src/augenblick/sfm/base.py`: `class SfMMethod(SceneInputMixin, Method[Scene])`; delete
  `SfMMethod.validate`. `SceneRefiner` unchanged (`super()` resolves through the mixin).
- `src/augenblick/reconstruction/base.py`: same mixin; `validate` keeps
  `require_reconstruction()` and calls `super().validate`.

`tests/test_scene.py`, `tests/test_recon_argv.py`, `tests/test_registry.py` continue to pass.

### Step 2 — `masking/base.py` and the package

Create `src/augenblick/masking/__init__.py`:

```python
"""Masking methods; importing the modules is what fires their registration."""
from augenblick.masking import rembg, threshold  # noqa: F401
```

Create `base.py` per [MaskMethod](#maskmethod): `MaskResult`, `MaskCommonConfig`,
`MaskMethod`, three helpers. Every heavy import (`numpy`/`scipy`/`PIL`/`cv2`/`skimage`/
`rembg`) is function-local.

### Step 3 — registry gains a third pair

`core/registry.py`:

```python
MASK_REGISTRY: dict[str, type] = {}

def register_mask(cls: type) -> type:
    return _register(MASK_REGISTRY, cls, "mask")

def get_mask(name: str) -> type:
    return _get(MASK_REGISTRY, name, "mask")
```

### Step 4 — generalise the CLI's stage dispatch

`cli/main.py`. Only step that edits existing behaviour.

```python
# One entry per stage; adding a stage is one edit here.
STAGES = {
    "mask": (MASK_REGISTRY, get_mask, "Run a masking method",
             "--images", "Input directory of .jpg/.JPG/.jpeg photographs"),
    "sfm": (SFM_REGISTRY, get_sfm, "Run an SfM method",
            "--scene", "Input scene directory"),
    "recon": (RECONSTRUCTION_REGISTRY, get_reconstruction,
              "Run a reconstruction backend", "--scene", "Input scene directory"),
}
```

Rework `_add_stage_parser` so `input_flag`/`input_help` come from the table and store as
`args.input_dir`. Then in `main()`:

- Lines 20-21: add `import augenblick.masking  # noqa: E402,F401`.
- Lines 45-46 → `for stage, (reg, _, help_text, flag, help_flag) in STAGES.items(): _add_stage_parser(...)`.
- Line 76 → `registry = STAGES[args.stage][0]`.
- Line 84 → `cls = STAGES[args.stage][1](args.method)`.
- Line 96 → `method.run(cls.build_input(args.input_dir.resolve()), args.output.resolve())`.
  No stage branching: each mixin's `build_input` returns the right input type.

### Step 5 — `threshold`

Per [Method 2](#method-threshold). Needs `cv2`/`skimage`/`scipy`, all pinned. Runs in every
env as built today.

### Step 6 — build a new env `augenblick_masked`

**The three shared envs (`augenblick_{a100,a40,l40s}`) do not change.** Every existing SfM
and reconstruction job keeps running against them exactly as today. Instead, step 6 builds
a **new** conda env that supersets one of them with `rembg` + `onnxruntime-gpu`.
Steps 8 and 9 target this new env; nothing else uses it.

New wrapper `scripts/setup_masked.sh`, mirroring `setup_a100.sh` in shape and length —
`GPU_LABEL`/`GPU_ARCH` are overridable so the CUDA rasterizers can be compiled for whichever
card the mask jobs run on (L40S = 8.9):

```bash
#!/bin/bash
set -euo pipefail
export GPU_LABEL="${GPU_LABEL:-A100}"
export GPU_ARCH="${GPU_ARCH:-8.0}"
export CUDA_MODULE="${CUDA_MODULE:-cuda/13.0.1}"
export TORCH_SPEC="torch==2.9.1 torchvision==0.24.1"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
export NUMPY_GENERATION="2"
export EXTRA_REQUIREMENTS="rembg==2.0.68 onnxruntime-gpu==1.23.2 opencv-python-headless==4.11.0.86"
exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_common.sh"
```

One hook in `scripts/setup_common.sh`: after `pip install -r requirements.txt`, run
`pip install $EXTRA_REQUIREMENTS` when that variable is set. It defaults empty, so every
existing wrapper behaves identically. The env is created by whoever activates it before
running the script, as with every other wrapper.

Resolution notes, all verified with `pip install --dry-run` on the exact
`EXTRA_REQUIREMENTS` string:

- **`onnxruntime-gpu` replaces the CPU build in one resolution step.** Installing after
  `requirements.txt` is enough; no line-stripping machinery is needed. Plain `rembg` (not
  `rembg[gpu]`) since the explicit pin already fixes the runtime.
- **`opencv-python-headless==4.11.0.86`** must be pinned explicitly: `rembg`'s unpinned
  floor otherwise floats to 5.0.0.93, the 5.x major this repo avoids. Pinned on the same
  command line, 4.11.0.86 wins.
- **`numba`/`llvmlite`** come along transitively; numpy 2.2.6 satisfies `numba<2.6`.

> **The real trap was upstream, not here.** `src/libs/vggt/requirements.txt` used to list
> plain `onnxruntime`, and `setup_common.sh` installs VGGT (stage 3/8) *after* the extras —
> so VGGT silently reinstalled the CPU build over `onnxruntime-gpu`, costing
> `CUDAExecutionProvider` with no error. Both packages unpack into the same `onnxruntime/`
> directory, so uninstalling the CPU one alone breaks `import onnxruntime` outright. Fixed by
> deleting that line from VGGT's requirements: nothing under `src/libs/vggt/` imports
> onnxruntime, and the top-level `requirements.txt` pin still serves `visual_util.py`'s
> skyseg path. **Keep it out when syncing from upstream VGGT**, exactly as with its
> pinned `numpy`.

> **CUDA-runtime detail.** `onnxruntime-gpu==1.23.2` links CUDA 12.x; the augenblick torch
> wheels use `+cu130`. ORT-GPU wheels bundle their own CUDA libs, so this usually works.
> Provider init **does** fail with cuda/13.0.1 loaded: ORT needs `libcufft.so.11` *and*
> `libcudnn.so.9`, while the cu130 toolchain ships `.so.12`. `mask.sbatch` therefore loads
> `cuda/12.9.1 cudnn/9.2.0.82-12-cuda` after sourcing `common.sh`, for the mask stage only —
> the stage uses no torch, and sfm/recon keep cu130. Do not downgrade torch. Note
> `cuda/12.4.1` does not exist on ICE (`cuda/12.4` has no usable `lib64`); 12.9.1 is the
> newest CUDA 12 module.

### Step 7 — tests

`tests/test_masking.py`, CPU-only, no GPU required, no model download. Runs in
`augenblick_masked` (the env from step 6):

| Test | Pins down |
|---|---|
| `MASK_REGISTRY == {rembg, threshold}` | mirrors `test_registry.py:52-53`; update the set in the same commit that adds a method |
| `write_mask` round-trips through the consumers' reader | **the most valuable test** — write a bool array, reopen `Image.open(p).convert("L")`, assert `(arr > 127) == original` |
| `write_mask` emits mode `L` | guards the "match existing RGB files" instinct |
| `postprocess_mask` keeps largest component, fills holes | two hand-built arrays |
| `foreground_fraction` on a known array | reject bounds depend on it |
| `ImagesInputMixin.validate` raises on empty dir / no matching suffixes | input contract |
| `MaskMethod.run` symlinks `images/`, creates `masks/` on a synthetic dir | output contract |
| `only_missing` skips images with an existing mask | 3 images, 2 pre-existing masks; `num_reused=2, num_written=1` |
| `threshold` polarity heuristic picks each side | one white-on-black, one black-on-white 64x64 |
| `build_parser()` accepts `mask <m> --images x --output y` | catches half-applied step 4 |
| `build_parser()` rejects `mask <m> --scene x --output y` | proves `--scene` is not silently accepted |

Do **not** run `rembg` or `cv2.grabCut` on real data — `tests/` stays seconds-long on a
login node. Update `tests/test_registry.py:52-53` or assert `MASK_REGISTRY` in the new file
— not both.

### Step 8 — the SLURM job (in `pace_slurm_verify/`)

**Disposable sibling directory of `pace_slurm/`.** `pace_slurm/*.sbatch` and
`pace_slurm/common.sh` stay byte-untouched — they belong to the frozen shared envs.
`pace_slurm_verify/` is where the new sbatches live, and its own `common.sh` overrides
just the env selection.

`pace_slurm_verify/common.sh` sources the shared one for its GPU/CUDA/scene-discovery
logic and then swaps `CONDA_ENV` for the new env:

```bash
# pace_slurm_verify/common.sh — activate augenblick_masked instead of augenblick_$GPU.
# Source the shared common.sh for everything else (GPU switch, module load, scene helpers,
# timing trap).
source "$(dirname "${BASH_SOURCE[0]}")/../pace_slurm/common.sh"
CONDA_ENV="$CONDA_ROOT/augenblick_masked"
if [ ! -d "$CONDA_ENV" ]; then
    echo "ERROR: no augenblick_masked env; build it with bash scripts/setup_masked.sh" >&2
    exit 2
fi
conda activate "$CONDA_ENV"
```

Sourcing `pace_slurm/common.sh` runs its `conda activate "$CONDA_ENV"` on the shared env
first; the two extra lines then re-activate `augenblick_masked`. `conda activate` layered
this way replaces the previous activation cleanly.

`pace_slurm_verify/mask.sbatch`, structured like the other per-scene array jobs:

- `--job-name=mask`
- `--gres=gpu:a100:1` — any of a100/a40/l40s; U^2-Net is small.
- `--cpus-per-task=4`, `--time=2:00:00`, `--mem=32gb`.
- Source `pace_slurm_verify/common.sh` (which itself sources the shared one).
- **Input path:** `--images "$SCENE/images"` (not `"$SCENE"`).
- **Output:** `$RESULT_ROOT/$SCENE_NAME/masked/<method>` so two methods don't collide.
- Keep `start_timing "$SCENE_NAME" "$(ls -1 "$SCENE/images" | wc -l)"`.

```bash
augenblick mask "${MASK_METHOD:-rembg}" \
    --images "$SCENE/images" --output "$OUT" "$@"
```

> **Array sizing.** `--array=0-46%6` matches existing scripts. For only the 9 maskless
> scenes, submit specific indices — but the index depends on `discover_scenes()`'s sorted
> order, which renumbers if `DATA_ROOT` changes.

> **The directory is disposable.** Once (if) the shared envs are updated to include
> `rembg`, delete `pace_slurm_verify/` and copy `mask.sbatch` to `pace_slurm/`. The
> two-line env swap in `pace_slurm_verify/common.sh` is the only production concern; a
> future contributor should not have to reason about which sbatch to run.

### Step 9 — end-to-end pipeline verify on one scene

Runs after every earlier step lands, before any 47-scene sweep. Exercises **mask → sfm →
recon** on one scene, so the downstream pipeline is proven against masks this stage
produced. Everything lands under `output/neurips_verify/` (already gitignored via `output/`).

**Scene: `UF_Herp_3998`.** 276 images with authored masks so `hull` also works;
[augenblick-package.md](../MEMORY/augenblick-package.md) records a VGGT baseline of ~2 min
and 2DGS smoke at ~12 min on this scene, so total verify wall time is under an hour on an
A100.

Layout:

```
output/neurips_verify/
    masked/           mask stage output (scene-shaped)
    sfm/vggt_ba/      sfm stage output, --scene points at ../masked/
    recon/2dgs/       recon stage output
    verify.log        concatenated .out+.err from all three jobs
```

`pace_slurm_verify/` holds mask/sfm/recon sbatches for the verify. **Copy** the two
sfm/recon sbatches from `pace_slurm/` into `pace_slurm_verify/` and change one line each —
`source "$SLURM_DIR/common.sh"` -> `source "$SLURM_DIR_VERIFY/common.sh"` so they activate
`augenblick_masked`. Then add the same override branch each already needs to accept a
`VERIFY_SCENE`/`VERIFY_OUT` pair in place of the array index:

```bash
if [ -n "${VERIFY_SCENE:-}" ]; then
    SCENE_NAME="$VERIFY_SCENE"
    SCENE="$DATA_ROOT/$VERIFY_SCENE"
    OUT="${VERIFY_OUT:?VERIFY_OUT is required alongside VERIFY_SCENE}"
else
    # existing discover_scenes / select_scene / OUT path
fi
```

Submit:

```bash
# Do NOT warm the cache by masking a scene on the login node: that is hundreds of images of
# CPU inference on a shared interactive node. rembg fetches its model to ~/.u2net/ on first
# use and the compute node has network access, so the mask job handles it.

M=$(sbatch --parsable --export=ALL,VERIFY_SCENE=UF_Herp_3998,\
VERIFY_OUT=$PWD/output/neurips_verify/masked pace_slurm_verify/mask.sbatch)
S=$(sbatch --parsable --dependency=afterok:$M --export=ALL,VERIFY_SCENE=UF_Herp_3998,\
VERIFY_OUT=$PWD/output/neurips_verify/sfm/vggt_ba pace_slurm_verify/vggt_ba_sfm.sbatch)
sbatch --parsable --dependency=afterok:$S --export=ALL,VERIFY_SCENE=UF_Herp_3998,\
VERIFY_OUT=$PWD/output/neurips_verify/recon/2dgs pace_slurm_verify/recon.sbatch
```

**Success criteria** (all required):

- Mask: `note=""` in `sfm_timings.csv`, 276 PNGs in `masked/masks/`, each passes the
  round-trip contract check.
- SfM: `sfm/vggt_ba/sparse/0/points3D.bin` non-empty; log's `Pipeline complete` block
  reports **>=250 registered images** (unmasked BA on this scene routinely dropped
  below 200 — masks recover the count).
- Recon: `recon/2dgs/train/ours_<iter>/fuse_post.ply` exists, >=50 MB (plumbing check, not
  quality).

Concatenate all three job logs into `verify.log` for one-file review.

## Verification

Run through `pace_slurm/`. Login-node CPU-only checks below are fine on the login node.

```bash
pytest tests/            # step 7, plus proof nothing existing broke
augenblick mask --list   # rembg, threshold
augenblick sfm --list    # unchanged: colmap, hull, turntable, vggt
augenblick recon --list  # unchanged: 2dgs, gw, pgsr, sugar
augenblick sfm vggt --help   | grep -- --scene   # still --scene
augenblick mask rembg --help | grep -- --images  # --images, not --scene
```

The `sfm`/`recon` `--list` lines are the step-4 regression check; the `--help` greps
confirm per-stage input-flag routing.

1. **Contract round-trips.** Reopen a written mask as `eval/nvs._load_mask` does; assert
   shape equals source `(H, W)`. Silent misalignment surfaces as a bad mesh hours later.
2. **`--only-missing` resumes.** Delete every second mask from a completed run, re-run with
   `--only-missing`; expect `num_reused = N/2, num_written = N/2` and byte-identical output.
3. **Maskless scene becomes usable.** Mask `UF_Fish_20486_skull` (663/0), run
   `augenblick sfm hull --scene <masked>` — previously impossible without `masks/`.
4. **Eyeball the masks.** Non-negotiable: overlay ~8 masks spanning the rotation on their
   images. Only this catches inversion; 1-3 pass on an inverted method.
5. **Polarity auto both ways.** Two scenes visibly differing in backdrop brightness with
   `threshold --polarity auto`; confirm from check 4 that masks land on the specimen.
   `--polarity dark-background`/`light-background` is the one-flag fix.
6. **Foreground fraction plausible.** Measured scene sits at 0.241; 0.99 or 0.001 is a
   failure. High `num_failed` is the bounds firing, not a crash.
7. **End-to-end.** Covered by step 9.

`common.sh`'s trap appends per-scene rows to `$RESULT_ROOT/_timing/sfm_timings.csv`, `note`
classifying `GPU_OOM`/`HOST_OOM`/`KILLED_OR_TIMEOUT` without reading a log. A `GPU_OOM` row
is a **result**, not to retry.

## Risks

- **Silently wrong masks dominate.** An inverted or over-eager mask degrades a mesh hours
  downstream where it's misattributed to the reconstructor. Mitigations: foreground bounds,
  no-file-on-reject, check 4 (the only inversion catch).
- **`augenblick_masked` drifts from the shared envs over time.** It is a separate env, so
  a `requirements.txt` bump landing in the shared envs is not automatically reflected
  here. Rebuild `augenblick_masked` via `bash scripts/setup_masked.sh` whenever the
  shared envs are rebuilt; document this in `MEMORY/environment-and-gpu.md`.
- **ORT-GPU vs cu130 module clash.** ORT-GPU 1.23.2 links CUDA 12.x. Usually fine (bundled
  libs) — it does **not**. `mask.sbatch` loads `cuda/12.9.1 cudnn/9.2.0.82-12-cuda` to supply
  the `.so.11`/`.so.9` sonames ORT links. Not a code change, but not optional either.
- **CPU fallback is a silent ~10x slowdown.** If the ONNX session logs
  `CUDAExecutionProvider` fallback, treat as a build error. The shared run loop prints the
  providers before the first image.
- **`rembg`'s runtime model download** — warm `~/.u2net/` on the login node first.
- **Polarity auto misfires on off-centre subjects.** Pin `--polarity` per check 5.
- **Disk.** 26 MP masks x 47 scenes are not free; symlinked `images/` and PNG compression
  keep it tractable.

## Documentation to update when this lands

Detail goes in `MEMORY/`, not CLAUDE.md (per
[repo-conventions.md](../MEMORY/repo-conventions.md)):

- [augenblick-package.md](../MEMORY/augenblick-package.md) — architecture gains `masking/`;
  `Method` is input-generic with `SceneInputMixin`/`ImagesInputMixin`; registry gains
  `register_mask`/`get_mask`; add "adding a masking method" recipe; note CLI dispatch is a
  `STAGES` table.
- **New `MEMORY/pipeline-masking.md`**, indexed in CLAUDE.md: mask contract
  (`<stem>.png`, mode `L`, `>127` foreground, dims equal source), per-method flags,
  `--images` (not `--scene`) input, 9 of 47 neurips scenes ship maskless.
- [scene-format.md](../MEMORY/scene-format.md) — data flow gains a stage upstream of SfM.
- [cluster-slurm.md](../MEMORY/cluster-slurm.md) — `mask.sbatch` and its `MASK_METHOD` env
  var.
- [environment-and-gpu.md](../MEMORY/environment-and-gpu.md) — new env
  `augenblick_masked` (built by `scripts/setup_masked.sh`) supersets `augenblick_a100`
  with `rembg` + `onnxruntime-gpu`; the three shared envs are unchanged. The new
  env is used exclusively by `pace_slurm_verify/`.
- **CLAUDE.md** — stage table gains Masking row; quick-start gains `augenblick mask`;
  gotchas note `mask` takes `--images` instead of `--scene`.
- Add this file to "Landed" in [repo-conventions.md](../MEMORY/repo-conventions.md).
