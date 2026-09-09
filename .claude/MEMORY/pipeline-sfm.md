# Data Preparation and SfM Entry Points

All SfM paths emit the same COLMAP scene layout — see [scene-format.md](scene-format.md).

## Data preparation

```bash
python pipeline/preparation/prepare_uf_dataset.py <input_dir> \
    [--out <output_dir>] [--mode {copy,move,symlink}] [--include-unmatched]
```

Splits a flat directory of mixed images and masks (`<stem>.JPG` + `<stem>.jpg.mask.png`) into
`images/` + `masks/`. Images normalised to `.jpg`, masks to `.png` named after the image stem.

## VGGT → COLMAP (`augenblick sfm vggt`)

```bash
augenblick sfm vggt \
    --input_dir <dir>            \  # must contain images/
    --output_dir <dir>           \
    [--use_masks] [--mask_erode_px 3] [--seed 42] \
    [--conf_thres_value 2.0]     \  # VGGT depth-confidence floor, both modes
    [--use_ba]                   \  # VGGSfM tracker + pycolmap BA
    [--shared_camera] [--camera_type SIMPLE_PINHOLE] \
    [--max_reproj_error 8.0] [--vis_thresh 0.2] \
    [--min_inlier_per_frame 64] [--min_valid_frames 0.3] \  # BA inlier floors
    [--query_frame_num 12] [--max_query_pts 4096] [--fine_tracking]
```

Runs VGGT inference, writes `sparse/0/`, exports `points.ply`, and **symlinks** `images/` (plus
`masks/`, when the input has them) into the output rather than copying — so an output scene breaks
if its input moves. Logs per-stage runtimes (model load, inference, tracking+BA) and a total.

`--use_masks` does two things: masks zero VGGT's `depth_conf` outside the subject in both modes,
and under `--use_ba` they also filter the tracker's query points, dropping any that land on
background. `mask_erode_px` (default 3) shrinks the foreground first, so detectors firing on the
silhouette seam do not seed tracks that drift onto the turntable. A frame with no mask is
unconstrained, not empty. Loaders never composite masks into the image; RGBA inputs flatten onto
black — 0 reads as "nothing here" downstream — masked or not.

- **No BA (default):** VGGT depth + camera predictions directly; filter by `conf_thres_value`,
  random-subsample to 100k points, write PINHOLE cameras at 518 px, then rescale to original
  resolution.
- **With `--use_ba`:** VGGSfM tracker for correspondences, then `pycolmap.bundle_adjustment()`.
  Operates at 1024 px internally; supports SIMPLE_PINHOLE and shared-camera modes.

`conf_thres_value` is one floor on one quantity — VGGT's `depth_conf` map — read in both modes:
without BA it selects which depth pixels become 3D points; with BA the tracker samples the same
map at each query point and drops the ones below it. Two traps: the BA-path filter is skipped
entirely unless more than 512 query points clear it, so a low-confidence frame keeps all of them;
and it replaced a hardcoded 1.2, so BA runs predating that were filtered more loosely and are not
comparable to ones at the 2.0 default.

BA drops a frame with fewer than `min_inlier_per_frame` surviving tracks instead of aborting, and
skips BA outright unless `min_valid_frames` — a *fraction* of the frame count, not a count — still
clear that floor. Masking starves frames, so these are the knobs to reach for when
`--use_masks --use_ba` produces nothing. A dropped frame still lands in `sparse/0/` at its raw
VGGT pose with no observations, so `num_reg_images()` counts it.

README's benchmarked BA invocation overrides the defaults:
`--use_ba --shared_camera --max_reproj_error 32 --max_query_pts 1048576 --query_frame_num 8`.

Model/geometry internals: [backend-vggt.md](backend-vggt.md).

## Masked COLMAP (`augenblick sfm colmap`)

```bash
augenblick sfm colmap --scene <dir> --output <dir> \
    [--max_image_size 2400] [--camera_model SIMPLE_PINHOLE]
```

pycolmap-based (not the `colmap` CLI): `extract_features` with
`ImageReaderOptions.mask_path` → `match_exhaustive` → `incremental_mapping`, then writes the
model with the most registered images to `sparse/0/`.

- `images/` and `masks/` are **symlinked** into the output dir, not copied.
- COLMAP expects a mask named `<image_name>.png` (i.e. `foo.jpg.png`), so the script builds a
  `masks_colmap/` dir of symlinks renamed `<stem>.jpg.png`. Same trick in the turntable script.
- `camera_mode=PER_IMAGE`, `num_threads=8`, no undistortion step.
- Prints `COLMAP_FAIL` and exits 2 if no model reconstructs; otherwise `COLMAP_DONE`.

## Turntable refinement (`augenblick sfm turntable`, added `44d02b7`, BA in `0b5b069`)

Post-processes an **existing COLMAP scene** (it needs `sparse/0/` as input, so run VGGT/COLMAP
first) by fitting an exact turntable rig — fixed rotation axis, constant angular step — and
re-solving poses on circular orbits.

```bash
augenblick sfm turntable \
    --input_dir <existing_colmap_scene> --output_dir <dir> \
    [--use_masks] [--camera_regex 'camera\d+'] [--step_deg <float>] \
    [--max_image_size 2400] [--retriangulate {auto,tracks,sift}] \
    [--max_reproj <px>] [--rig_ba {auto,on,off}] [--rig_ba_iters 3]
```

Flow:
1. Group images into physical cameras by `--camera_regex` (default `camera\d+`), order within a
   group by the last integer in the filename (`order_key`).
2. `fit_axis_step()` — SVD-fit a plane to each group's camera centres for the axis; angular step
   from a least-squares slope of unwrapped angle vs. frame index, median across groups.
3. Step **sign is ambiguous**: both `+step` and `-step` are fitted and the one with lower centre
   error wins (skipped when `--step_deg` is given).
4. Retriangulation mode: `auto` picks `tracks` when the input mean track length >= 3.0, else
   falls back to masked SIFT (`sift`).
5. `tracks` mode: optional rig-constrained BA (`rig_ba`, resection–intersection rounds) refines
   axis/step/rig, then `apply_track_preserving()` re-triangulates existing tracks against the
   fixed poses with an adaptive reprojection threshold (default `2.5x` the median). Cameras are
   rewritten as SIMPLE_PINHOLE.
6. `sift` mode: masked SIFT + exhaustive matching, per-group shared focal (median), then batched
   DLT triangulation.

`--use_masks` auto-enables when `<input>/masks/` exists. `--rig_ba off` exists for ablations.
