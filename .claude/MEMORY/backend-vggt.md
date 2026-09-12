# `src/libs/vggt/` — VGGT Model

Meta's Visual Geometry Grounded Transformer. Weights (~4 GB) auto-download from
`facebook/VGGT-1B` on HuggingFace on first run. Vendored in-tree (not a git submodule);
installed editable from `src/libs/vggt` (package root), importable package is `src/libs/vggt/vggt/`.

## Model architecture (`src/libs/vggt/vggt/models/`)

**`vggt.py` — VGGT class**
- Input: images `[B, S, 3, H, W]` in `[0, 1]`, optional query points `[B, N, 2]`
- Aggregator produces token lists via alternating frame/global attention
- Four heads over the aggregated tokens:
  - `camera_head` → `pose_enc [B, S, 9]` (translation[3] + quaternion[4] + FoV[2])
  - `depth_head` → `depth [B, S, H, W, 1]`, `depth_conf [B, S, H, W]`
  - `point_head` → `world_points [B, S, H, W, 3]`, `world_points_conf [B, S, H, W]`
  - `track_head` → `track [B, S, N, 2]`, `vis`, `conf` (only when query_points given)

**`aggregator.py` — Aggregator class**
- DINOv2 ViT-L/14 patch embedding (frozen `dinov2_vitl14_reg` weights)
- 24 alternating blocks: frame attention (`[B*S, P, C]`) + global attention (`[B, S*P, C]`)
- 2D RoPE positional encoding (frequency = 100)
- 1 camera token + 4 register tokens prepended per frame
- Outputs concatenated frame+global intermediates `[B, S, P, 2*C]` (2048-dim) per block pair
- Gradient checkpointing enabled during training

## Utilities (`src/libs/vggt/vggt/utils/`)

**`load_fn.py`**
- `load_and_preprocess_images_square()` — main pipeline loader. Square-pads to `max(W,H)`,
  resizes to `target_size` (default 1024). Returns images `[N, 3, T, T]`,
  `original_coords [N, 6]` (x1, y1, x2, y2, W, H for undoing the pad), and transformed **PIL**
  masks. Masks ride alongside the image, never composited into it.
- `load_and_preprocess_images()` — simpler `crop`/`pad` loader at 518 px. No in-repo callers;
  returns **tensor** masks `[1, H, W]`, unlike the square loader's PIL ones.

**`pose_enc.py`**
- `pose_encoding_to_extri_intri()` — `[B, S, 9]` → extrinsic `[B, S, 3, 4]` + intrinsic
  `[B, S, 3, 3]`, in the resolution of `image_size_hw` (typically 518x518); principal point
  assumed at image centre.
- `extri_intri_to_pose_encoding()` — inverse.

**`geometry.py`**
- `unproject_depth_map_to_point_map()` — depth + extrinsics + intrinsics → world points
- `closed_form_inverse_se3()` — batch SE(3) inverse (numpy and torch)

**`helper.py`**
- `randomly_limit_trues()` — subsample True entries of a boolean mask to a budget
- `create_pixel_coordinate_grid()` — `[S, H, W, 3]` grid of (x, y, frame_idx)

## Tracking (`src/libs/vggt/vggt/dependency/track_predict.py`)

- `predict_tracks()` — VGGSfM tracker over the query frames. `masks [S, H, W]` at the loader's
  resolution drop background query points; `_erode_masks` shrinks them first as a chunked
  `max_pool2d` dilation of the background. `conf_thresh` floors query-point confidence, but only
  bites when more than 512 points clear it.

## COLMAP conversion (`src/libs/vggt/vggt/dependency/np_to_pycolmap.py`)

- `batch_np_matrix_to_pycolmap()` — full conversion **with** tracks, used with BA. Applies
  reprojection-error filtering, builds proper Point2D↔Point3D associations. Frames under
  `min_inlier_per_frame` are dropped individually (mask row zeroed) and still register with no
  observations; BA is skipped outright unless `min_valid_frames` (a *fraction*) clear the floor.
- `batch_np_matrix_to_pycolmap_wo_track()` — lightweight, no tracks, feed-forward mode only.
  Points are assigned to the frame they were unprojected from. **Do NOT use this for BA.**
- `pycolmap_to_batch_np_matrix()` — inverse.

## Conventions

- **VGGT outputs**: OpenCV convention (x-right, y-down, z-forward), camera-from-world `[R|t]`.
- **Intrinsics**: initially in 518x518 pixel space; `augenblick.sfm.vggt` rescales them to the
  original image resolution via `rename_colmap_recons_and_rescale_camera()`.
- **COLMAP IDs are 1-indexed** — there is a `+1` offset between batch index and COLMAP
  image/camera ID throughout.
