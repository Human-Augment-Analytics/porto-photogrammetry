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

        COLMAP looks for <image_name>.png; images are .jpg, so link <stem>.jpg.png.

        Args:
            dest: Directory to populate with the renamed symlinks.

        Returns:
            The directory path, or None when the scene has no masks.
        """
        if not self.has_masks():
            return None
        dest.mkdir(parents=True, exist_ok=True)
        for m in os.listdir(self.masks_dir):
            link = dest / f"{m.rsplit('.', 1)[0]}.jpg.png"
            if not link.exists():
                os.symlink(self.masks_dir / m, link)
        return dest
