"""Mask-restricted COLMAP SfM via pycolmap: extract, match, map, keep the best model."""
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from augenblick.core.errors import SceneError
from augenblick.core.registry import register_sfm
from augenblick.core.scene import Scene
from augenblick.sfm.base import SfMMethod, SfMResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ColmapConfig:
    """Feature extraction parameters for the COLMAP SfM run."""

    max_image_size: int = field(default=2400, metadata={"help": "Max image dimension for SIFT"})
    camera_model: str = field(default="SIMPLE_PINHOLE", metadata={"help": "COLMAP camera model"})


@register_sfm
class ColmapSfM(SfMMethod):
    """Runs SIFT extraction, exhaustive matching, and incremental mapping through pycolmap."""

    name: ClassVar[str] = "colmap"
    config_cls: ClassVar[type] = ColmapConfig

    def run(self, scene: Scene, output_dir: Path) -> SfMResult:
        """Reconstruct the scene and write the largest model to output_dir/sparse/0.

        Args:
            scene: Input scene with images/ and optionally masks/.
            output_dir: Directory to write the COLMAP scene into.

        Returns:
            An SfMResult for the selected best model.

        Raises:
            SceneError: If COLMAP reconstructs no model at all.
        """
        import pycolmap

        self.validate(scene)
        t0 = time.time()
        out_dir = output_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        if not (out_dir / "images").exists():
            os.symlink(scene.images_dir, out_dir / "images")
        if scene.has_masks() and not (out_dir / "masks").exists():
            os.symlink(scene.masks_dir, out_dir / "masks")

        mask_dir = scene.link_colmap_masks(out_dir / "masks_colmap")
        mask_path = str(mask_dir) if mask_dir else None

        db_path = str(out_dir / "database.db")
        if os.path.exists(db_path):
            os.remove(db_path)

        ro = pycolmap.ImageReaderOptions()
        if mask_path:
            ro.mask_path = mask_path
        ro.camera_model = self.config.camera_model
        eo = pycolmap.FeatureExtractionOptions()
        eo.max_image_size = self.config.max_image_size
        # Not -1. COLMAP reads that as the machine's core count, which ignores the cgroup a
        # batch scheduler puts the job in: an 8-CPU job on a 192-core node spawns 192 SIFT
        # extractors, each holding a full-resolution image and its pyramid, and is killed for
        # running out of memory. It also makes the thread count depend on which node the job
        # lands on, so wall-clock stops being comparable between runs.
        eo.num_threads = len(os.sched_getaffinity(0))

        t = time.time()
        pycolmap.extract_features(db_path, str(out_dir / "images"),
                                  camera_mode=pycolmap.CameraMode.PER_IMAGE,
                                  reader_options=ro, extraction_options=eo)
        logger.info(f"[colmap] extraction {time.time()-t:.0f}s")

        t = time.time()
        pycolmap.match_exhaustive(db_path)
        logger.info(f"[colmap] matching {time.time()-t:.0f}s")

        t = time.time()
        maps_dir = out_dir / "sparse"
        maps_dir.mkdir(exist_ok=True)
        recs = pycolmap.incremental_mapping(db_path, str(out_dir / "images"), str(maps_dir))
        logger.info(f"[colmap] mapping {time.time()-t:.0f}s -> {len(recs)} model(s)")

        if not recs:
            raise SceneError("COLMAP_FAIL: no model reconstructed")

        best_id = max(recs, key=lambda k: recs[k].num_reg_images())
        best = recs[best_id]
        logger.info(f"[colmap] best model {best_id}: {best.num_reg_images()} images, "
                    f"{best.num_points3D()} points")

        final = maps_dir / "0"
        final.mkdir(exist_ok=True)
        best.write(str(final))
        elapsed = time.time() - t0
        logger.info(f"COLMAP_DONE {best.num_reg_images()}img {best.num_points3D()}pts "
                    f"in {elapsed:.0f}s -> {out_dir}")

        return SfMResult(
            output_dir=out_dir,
            elapsed=elapsed,
            scene=Scene(out_dir),
            num_images=best.num_reg_images(),
            num_points=best.num_points3D(),
        )
