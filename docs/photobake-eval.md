# Native Texture Versus Photo Re-Bake

`photobake_mesh.py` evaluates the selected mesh's native export and a training-photo
re-bake on the same geometry and held-out scene cameras. This is a native-output
comparison, not automatically a controlled color-source ablation.

## Usage

Run from the repository root with the SuGaR/PyTorch3D/rasterizer environment:

```bash
PYTHONPATH=src python src/libs/sugar/photobake_mesh.py \
    --scene output/specimen/all/colmap \
    --mesh output/specimen/all/colmap-sugar/sugar/refined_mesh/mesh.obj \
    --output output/specimen/all/colmap-sugar \
    --resolution -1 --photo-max-size -1 --square-size 10 --iteration 15000
```

No Gaussian checkpoint, checkpoint camera JSON, or checkpoint iteration is needed.
`--iteration` only labels output directories as `ours_<n>`; it does not select weights.
The batch equivalent is:

```bash
SCENE=output/specimen/all/colmap \
MESH=output/specimen/all/colmap-2dgs/train/ours_30000/fuse_post.ply \
OUTPUT=output/specimen/all/colmap-2dgs \
RESOLUTION=-1 PHOTO_MAX_SIZE=-1 SQUARE_SIZE=10 ITERATION=30000 \
sbatch pace_slurm/photobake_eval.sbatch
```

## Cameras And Images

Poses and per-view FoVs come from the selected scene's `sparse/0` through
`readColmapSceneInfo(eval=True, load_images=False, load_point_cloud=False)`, not from a model trained on a
different scene. `split.json` takes precedence; otherwise the reader uses sorted
image names and `llffhold=8`. Training views alone supply bake colors. The conversion
passes the reader's transposed COLMAP rotation, world-to-camera translation, and
horizontal/vertical FoVs unchanged into `GSCamera` and `CamerasWrapper`.

The existing undistorted, centered-pinhole camera assumptions remain: FoV-based
SuGaR cameras do not add support for distortion or off-center principal points.
The vendored text parser accepts PINHOLE; its binary path also supports SIMPLE_PINHOLE.
Photobake never reads, converts, or writes a point cloud. Missing, duplicate, or
overlapping entries in an explicit split are rejected rather than silently filtered.

The optional `load_images` parameter is added to `readColmapSceneInfo` and
`readColmapCameras`, defaulting to `True` for existing training callers. The scene
reader's optional `load_point_cloud` also defaults to `True`. With `load_images=False`,
`CameraInfo.image` is `None`; callers needing pixel data must load `image_path` lazily.
Photobake and the parent PR's SuGaR camera loader share the same resolution helper.

Evaluation resolution follows 3DGS semantics, independently of baking resolution:

| `--resolution` | Held-out render, GT, and mask size |
|---|---|
| `-1` (default) | Preserve images up to width 1600; cap wider images at width 1600 |
| `1`, `2`, `4`, `8` | Divide original image dimensions by that factor, rounded |
| Any other positive integer | Target width, with aspect ratio preserved |

There is no inherited 1920-pixel cap. The loader reads only image headers while
building cameras, and closes those handles. It loads held-out RGB and masks one
view at a time at the render size; masks use nearest-neighbor resizing, matching
the scorer. Source photos are also loaded one training view at a time.
`--photo-max-size -1` keeps their original dimensions; a positive value caps their
long side and rescales the corresponding raster camera. Evaluation downsampling
does not downsample the bake photos.

## Atlas And Provenance

Native SuGaR baking and photo baking now share `build_square_atlas`, including
the original six inset UV offsets and packing layout. Both require `square_size >= 5`:
smaller sizes can collapse triangles. Native baking now sums repeated face/texel
indices correctly and retains its SH fallback for unseen faces. Photo baking
retains its mean-seen-face fallback, or gray 0.5 when no faces are seen.

The photo default is explicitly 10. Native size is not inferred from current
configuration defaults or from the photo setting. A native size of 8 and a photo
size of 10, for example, are distinct conditions. For UV meshes, the evaluator
searches the nearest three parent directories for `run_manifest.json` and only
trusts `texture_square_size` when `textured_mesh_path` matches the selected mesh.
Without that evidence, native size is `null` with status `unknown`; vertex-colored
meshes have status `not_applicable`. A historical size of 4 is reported honestly,
not repaired. The parent PR changes SugarConfig's default to 8, rejects sizes below
5, and propagates resolution through training and held-out rendering.

Every invocation writes `photobake/run_metadata.json`, including when scoring is
disabled. The same metadata is included in metrics JSON: scene camera source,
split and view names, requested evaluation resolution and actual held-out sizes,
photo maximum size, photo square size, fallback, and texture sampling. Photo
sampling is `nearest`. Native sampling after asset loading is recorded separately
from original export sampling; the latter stays `unknown` unless a matching
manifest records `texture_sampling_mode`.

Outputs retain the existing names:

- `test_mesh_native/ours_<n>/{renders,gt}` and `nvs_metrics_mesh_native.json`.
- `test_mesh_photofull/ours_<n>/{renders,gt}` and `nvs_metrics_mesh_photofull.json`.
- `photobake/photofull_mesh.obj` unless `--no-save-obj` is given.

The `photofull` label is historical; consult `photo_max_size` to establish whether
the run actually used full-resolution photos.

## Interpreting The Comparison

Matching geometry and held-out cameras removes geometry/pose differences, but a
native export versus a photo re-bake can still differ in atlas density, source
resolution, sampling after reloading, accumulation implementation, and unseen-face
fallback. These are not all changes in color source.

A controlled color-source ablation must re-bake both color sources with identical
atlas size/layout, effective training-view resolution, visibility/masks, texel
accumulation, fallback policy, and evaluation sampling/resolution. This command does
not perform that second controlled bake and does not claim such an ablation.

Existing OBJ/PLY assets and textures are never regenerated or repaired by these
fixes. In particular, an old degenerate atlas or lost repeated-index samples remain
in an existing native export; producing corrected native assets requires an explicit
new export. Do not reinterpret old metrics as corrected results.