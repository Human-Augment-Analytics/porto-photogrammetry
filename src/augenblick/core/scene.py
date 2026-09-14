"""The COLMAP scene contract shared by SfM and reconstruction stages."""
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from augenblick.core.errors import SceneError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Scene:
    """A COLMAP-format scene: images/ plus optional masks/ and sparse/0/.

    Args:
        root: Directory holding the scene's subdirectories.
    """

    root: Path

    @property
    def images_dir(self) -> Path:
        """Directory holding the scene's source images."""
        return self.root / "images"

    @property
    def masks_dir(self) -> Path:
        """Directory holding per-image masks, which may not exist."""
        return self.root / "masks"

    @property
    def sparse_dir(self) -> Path:
        """Directory holding the COLMAP sparse model."""
        return self.root / "sparse" / "0"

    @property
    def model_dir(self) -> Path:
        """The COLMAP model, allowing for the flattened sparse/ layout PGSR prepares."""
        return self.sparse_dir if self.sparse_dir.is_dir() else self.root / "sparse"

    def has_masks(self) -> bool:
        """Whether the scene carries a non-empty masks/ directory.
        """
        return self.masks_dir.is_dir() and any(self.masks_dir.iterdir())

    def has_reconstruction(self) -> bool:
        """Whether sparse/0/ exists and is non-empty; an empty one means SfM failed."""
        return self.sparse_dir.is_dir() and any(self.sparse_dir.iterdir())

    def require_images(self) -> None:
        """Raise SceneError unless the scene has a non-empty images/ directory."""
        if not self.images_dir.is_dir():
            raise SceneError(f"no images/ directory at {self.images_dir}")
        if not any(self.images_dir.iterdir()):
            raise SceneError(f"images/ directory is empty at {self.images_dir}")

    def require_reconstruction(self) -> None:
        """Raise SceneError unless the scene has a non-empty sparse/0/ model."""
        if not self.sparse_dir.is_dir():
            raise SceneError(f"no SfM model at {self.sparse_dir}")
        if not any(self.sparse_dir.iterdir()):
            raise SceneError(f"SfM model directory is empty at {self.sparse_dir}")

    def link_colmap_masks(self, dest: Path) -> Path | None:
        """Build a symlink directory naming masks the way COLMAP expects.

        COLMAP looks for <image_name>.png, so a mask named after the image *stem* must be
        linked under the image's full filename.

        Args:
            dest: Directory to populate with the renamed symlinks.

        Returns:
            The directory path, or None when the scene has no masks.
        """
        if not self.has_masks():
            return None
        images_by_stem = {p.stem: p.name for p in self.images_dir.iterdir() if p.is_file()}
        dest.mkdir(parents=True, exist_ok=True)
        unmatched = []
        for m in os.listdir(self.masks_dir):
            stem = m.rsplit('.', 1)[0]
            image_name = images_by_stem.get(stem)
            if image_name is None:
                unmatched.append(m)
                continue
            link = dest / f"{image_name}.png"
            if not link.exists():
                os.symlink(self.masks_dir / m, link)
        if unmatched:
            logger.warning(
                f"{len(unmatched)} mask(s) in {self.masks_dir} match no image and were "
                f"skipped (e.g. {unmatched[0]})")
        return dest
