"""Shared shape of the SfM methods, and the refiner variant that needs an existing model."""
import logging
import os
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from augenblick.core.method import Method, SceneInputMixin, StageResult
from augenblick.core.scene import Scene

logger = logging.getLogger(__name__)


def link_dir(src: Path, dst: Path) -> None:
    """Symlink dst -> src, replacing a stale link left by an earlier run."""
    # lexists, not exists: a link dangling from a moved scene still blocks os.symlink.
    if dst.is_symlink():
        dst.unlink()
    elif os.path.lexists(dst):
        logger.warning("%s exists and is not a symlink; leaving it in place", dst)
        return
    os.symlink(src, dst)
    logger.info("Linked %s -> %s", dst, src)


@dataclass
class SfMResult(StageResult):
    """A completed SfM run: the output scene plus its registered image and point counts."""

    scene: Scene | None = None
    num_images: int = 0
    num_points: int = 0


class SfMMethod(SceneInputMixin, Method[Scene]):
    """Consumes a scene with images/, produces sparse/0/ in the output directory."""

    @abstractmethod
    def run(self, scene: Scene, output_dir: Path) -> SfMResult:
        """Run the SfM stage, writing a COLMAP model into output_dir/sparse/0.

        Args:
            scene: Input scene providing images/ and optionally masks/.
            output_dir: Directory to write the resulting COLMAP scene into.

        Returns:
            An SfMResult describing the reconstruction produced.
        """


class SceneRefiner(SfMMethod):
    """An SfM step that refines an existing reconstruction rather than creating one."""

    def validate(self, scene: Scene) -> None:
        """Require images/ and an existing non-empty sparse/0/ to refine."""
        super().validate(scene)
        scene.require_reconstruction()
