# Common Scene Format and Data Flow

## Scene layout

Every reconstruction backend consumes the same COLMAP-format scene:

```
<scene>/
├── images/           # required; source images
├── masks/            # optional; binary PNGs (white = foreground, black = background)
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

Each backend's `scene/__init__.py` auto-detects COLMAP format by the presence of `sparse/`
(vs. Blender's `transforms_train.json`) and reads via `dataset_readers.py` → `colmap_loader.py`.
PGSR is the exception: its `prepare()` step flattens `sparse/0/` → `sparse/`.

COLMAP mask naming quirk: COLMAP looks for `<image_name>.png` (e.g. `foo.jpg.png`), which is why
`Scene.link_colmap_masks()` builds a `masks_colmap/` symlink dir for the pycolmap paths.

## Data flow

```
Raw data (mixed images + masks)
    │
    ▼
prepare_uf_dataset.py ──► images/ + masks/
    │
    │   (a flat images folder with no masks? produce them first:)
    │   augenblick mask rembg --images <dir> --output <scene>  ──► images/ symlink + masks/
    │   (see pipeline-masking.md)
    │
    ├──► augenblick sfm vggt   ──► sparse/0/    (VGGT → depth + cameras → optional BA)
    └──► augenblick sfm colmap ──► sparse/0/    (mask-restricted SIFT, pycolmap API)
                 │
                 └──► augenblick sfm turntable ──► sparse/0/  (rig prior refinement of an
                                                                 existing scene)
    │
    ▼
COLMAP scene (images/ + sparse/0/)
    │
    ├──► recon sugar → 3DGS ckpt → coarse → mesh → refine → textured mesh (.obj)
    ├──► recon 2dgs  → 2DGS model → TSDF fusion → mesh (.ply)
    ├──► recon pgsr  → flatten sparse/ → PGSR model → TSDF fusion → mesh (.ply)
    └──► recon gw    → GW train → pivot marching-tetrahedra → texture refine → mesh (.ply)
    │
    ▼
Baseline comparison: baseline/benchmark_meshroom.py (see baseline-meshroom.md)
```
