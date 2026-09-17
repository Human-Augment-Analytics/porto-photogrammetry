# Masking stage (`augenblick mask`)

Produces a scene's `masks/` from a flat folder of photographs, upstream of SfM. It is the
one stage that **produces** masks; `sfm` and `recon` **consume** them.

```bash
augenblick mask rembg --images data/neurips/prepared/<scene>/images \
                      --output data/neurips/prepared/<scene>_masked
```

Writes `<output>/images/` (a symlink to the resolved input) + `<output>/masks/` — a
COLMAP-shaped scene ready for `augenblick sfm vggt --use_ba --use_masks --scene <output>`.

## Input contract — the difference from sfm/recon

`mask` takes `--images` (a flat directory of `.jpg/.JPG/.jpeg`), **not** `--scene`. Nothing
else is read or expected: no sibling `masks/`, `sparse/0/`, or `split.json`. The input
directory is never mutated. This is expressed in the code by an input mixin on `Method`:
`ImagesInputMixin` for `mask`, `SceneInputMixin` for `sfm`/`recon` (see augenblick-package.md).

## Mask contract (shared with all consumers)

A mask is a PNG named `<image stem>.png`, mode `L`, values `{0, 255}`, **dimensions equal to
the source image**, `> 127 = foreground`. Wrong dimensions misalign *silently* (hull indexes
`mask[vi, ui]`; eval/nvs resizes NEAREST), so the dims-equal-source rule is load-bearing.

A missing mask means "skip this view" to every consumer — so a mask that fails the
foreground-fraction bounds is **not written** (writing an all-foreground mask would silently
disable masking while looking successful).

## Methods

- **`rembg`** (default): learned U²-Net / IS-Net / BiRefNet matting via rembg's ONNX runtime.
  GPU when `onnxruntime-gpu` loads its CUDA provider, CPU otherwise (4.0x slower — measured
  2.35 s/mask GPU vs 9.40 s/mask CPU, isnet-general-use, L40S, 663 images). Downloads
  its model to `~/.u2net/` on first use — warm it on the login node.
  Key flags: `--model` (default `isnet-general-use`), `--alpha_threshold` (default 127),
  `--providers`.
- **`threshold`**: classical Otsu / GrabCut, CPU-only, no new dependency (cv2/skimage/scipy,
  all already pinned). Key flags: `--mode {otsu,grabcut}`, `--polarity {auto,dark-background,
  light-background}` (auto picks per image via border-vs-centre luminance), `--downscale`.

Shared flags (both methods): `--only_missing` (underscore, not `--only-missing`; resume, skips
images that already have a mask), `--min_foreground`/`--max_foreground` (reject bounds),
`--keep_largest`, `--fill_holes`.

`--only_missing` reuses whatever is on disk **without checking how it was produced**, so masks
written during a silent CPU fallback survive every later rerun and look like a cache hit. After
fixing a fallback, delete `<output>/masks/` or drop the flag — a rerun alone will not redo them.

## Cost is postprocessing, not inference

`rembg` infers at 1024² but returns a mask at **source resolution**, and `run()` passes that
straight to `postprocess_mask` with no downscale — so `keep_largest`/`fill_holes` do morphology
on 26 MP instead of 1 MP. On 6240×4160 images the per-image budget is ~2.2 s (L40S, measured
over 4370 images):

| Stage | ~Time | Device |
|-------|-------|--------|
| `binary_fill_holes` | 1.70 s | CPU |
| decode 26 MP JPEG → RGB | 0.57 s | CPU |
| `write_mask` (`optimize=True`) | 0.32 s | CPU |
| `ndimage.label` (`keep_largest`) | 0.23 s | CPU |
| U²-Net inference | ~0.1–0.2 s | GPU |

So the GPU is mostly idle and a masking job is CPU-bound — more `--cpus-per-task` helps, a
bigger card does not. Postprocessing at model resolution and upscaling afterwards would cut
~3-4x, but changes mask boundaries (low-res hole-filling can miss thin structures), so it is a
behaviour change to validate visually, not a free win. Verified against base.py and a timed run
of the 10-scene neurips subset.

## Why the stage exists

9 of 47 neurips scenes ship no masks, all ≥432 images. Masked BA needs masks; `sfm hull`
raises without `masks/`; `eval/nvs` scores unmasked scenes against the black surround. One
stage unblocks all three.

## SLURM

Runs against the per-GPU `augenblick_*` env like every other stage — `rembg` and
`onnxruntime-gpu` are ordinary `requirements.txt` entries, so there is no separate masking env.
A masking job should still guard against silent CPU fallback, since ORT reports the CUDA
provider as available even when it cannot load it. See cluster-slurm.md and environment-and-gpu.md.
