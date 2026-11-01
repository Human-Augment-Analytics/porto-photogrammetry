# Common Scene Format and Data Flow

## Scene layout

Every reconstruction backend consumes the same COLMAP-format scene:

```
<scene>/
├── images/           # required; source images
├── masks/            # optional; binary PNGs (white = foreground, black = background)
├── split.json        # optional; train/test stems, written by the first `recon --eval`
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        ├── points3D.bin
        └── points3D.ply  # optional; an init cloud (sfm hull) or a backend's cache of the .bin
```

SfM outputs symlink `images/` and `masks/` back to their input rather than copying, so a scene
breaks if its source moves.

The Gaussian backends' `scene/__init__.py` auto-detect COLMAP format by the presence of
`sparse/` (vs. Blender's `transforms_train.json`) and read via `dataset_readers.py` →
`colmap_loader.py`. PGSR is the exception: its `prepare()` step flattens `sparse/0/` →
`sparse/`. GeoSVR has its own loader (`src/dataloader/`) and reads `sparse/0/` in place.

**`points3D.ply` beats `points3D.bin` for 2DGS, SuGaR's 3DGS step, PGSR and GW**: each converts
the `.bin` only when no `.ply` exists. That is what lets `sfm hull` swap the init cloud. It also
means 2DGS and SuGaR **write** `sparse/0/points3D.ply` into the input scene on first use, and a
stale `.ply` left beside an updated `.bin` is silently preferred. GeoSVR ignores the `.ply`.

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
    ├──► augenblick sfm dvlt   ──► sparse/0/    (DVLT feed-forward, no BA)
    └──► augenblick sfm colmap ──► sparse/0/    (mask-restricted SIFT, pycolmap API)
                 │
                 ├──► augenblick sfm turntable ──► sparse/0/  (rig prior refinement of an
                 │                                              existing scene)
                 └──► augenblick sfm hull      ──► sparse/0/points3D.ply  (visual-hull init
                                                                 cloud; poses unchanged)
    │
    ▼
COLMAP scene (images/ + sparse/0/)
    │
    ├──► recon sugar → 3DGS ckpt → coarse → mesh → refine → textured mesh (.obj)
    ├──► recon 2dgs  → 2DGS model → TSDF fusion → mesh (.ply)
    ├──► recon pgsr  → flatten sparse/ → PGSR model → TSDF fusion → mesh (.ply)
    ├──► recon gw    → GW train → pivot marching-tetrahedra → texture refine → mesh (.ply)
    └──► recon geosvr → sparse-voxel train → render → TSDF fusion → mesh (.ply)
    │
    ▼
Evaluation: recon --eval / eval.nvs (held-out views), eval.mesh (vs. a reference mesh),
            eval.scale (metric scale from scale bars) — see pipeline-evaluation.md
Baseline comparison: baseline/benchmark_meshroom.py (see baseline-meshroom.md)
```
