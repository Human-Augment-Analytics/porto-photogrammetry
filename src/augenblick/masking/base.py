"""Shared shape of the masking methods, and the mask-writing contract they all satisfy.

A mask is a PNG named <image stem>.png, mode L, `> 127 = foreground`, dimensions equal to
the source image. Every downstream reader (hull, eval/nvs, COLMAP SfM) thresholds at 127.
"""
import logging
import os
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from augenblick.core.errors import SceneError
from augenblick.core.method import ImagesInputMixin, Method, StageResult
from augenblick.core.timing import StageTimer

logger = logging.getLogger(__name__)


@dataclass
class MaskResult(StageResult):
    """A completed masking run: the output directory plus per-image mask counts."""

    num_images: int = 0
    num_written: int = 0
    num_reused: int = 0
    num_failed: int = 0


@dataclass(frozen=True)
class MaskCommonConfig:
    """Fields every masking method shares."""

    only_missing: bool = field(default=False, metadata={
        "help": "Skip images that already have <stem>.png in <output>/masks/"})
    min_foreground: float = field(default=0.005, metadata={
        "help": "Reject a mask below this foreground fraction (counts as num_failed)"})
    max_foreground: float = field(default=0.95, metadata={
        "help": "Reject a mask above this foreground fraction"})
    keep_largest: bool = field(default=True, metadata={
        "help": "Keep only the largest connected foreground component"})
    fill_holes: bool = field(default=True, metadata={
        "help": "Fill enclosed background holes inside the silhouette"})


def postprocess_mask(mask, keep_largest: bool, fill_holes: bool):
    """Drop specks (scipy.ndimage.label) and close holes (binary_fill_holes)."""
    import numpy as np
    from scipy import ndimage

    m = mask.astype(bool)
    if keep_largest:
        labels, n = ndimage.label(m)
        if n > 1:
            sizes = ndimage.sum(m, labels, index=range(1, n + 1))
            keep = int(np.argmax(sizes)) + 1
            m = labels == keep
        elif n == 0:
            return m
    if fill_holes:
        m = ndimage.binary_fill_holes(m)
    return m


def foreground_fraction(mask) -> float:
    """Fraction of pixels the mask calls foreground."""
    import numpy as np
    m = np.asarray(mask, dtype=bool)
    if m.size == 0:
        return 0.0
    return float(m.sum()) / float(m.size)


def write_mask(mask, dest: Path) -> None:
    """Write bool mask as 8-bit greyscale PNG; consumers threshold at >127."""
    import numpy as np
    from PIL import Image

    arr = (np.asarray(mask, dtype=bool).astype(np.uint8)) * 255
    Image.fromarray(arr, mode="L").save(dest, optimize=True)


class MaskMethod(ImagesInputMixin, Method[Path]):
    """Consumes a directory of images, produces a scene-shaped output with masks/.

    Subclasses implement `mask_for(image_path)` returning a bool [H, W] numpy array
    matching the image's dimensions. `run()` is implemented here once.
    """

    title: ClassVar[str]

    @abstractmethod
    def mask_for(self, image_path: Path):
        """Return a bool numpy array [H, W], True on the specimen, matching image dims."""

    def header(self, images_dir: Path, output_dir: Path) -> dict[str, object]:
        """Key/value lines logged under the banner; subclasses may extend this."""
        return {"Images": images_dir, "Output": output_dir, "Method": self.name}

    def run(self, images_dir: Path, output_dir: Path) -> MaskResult:
        """Segment every image; write masks under output_dir/masks/.

        Args:
            images_dir: Flat directory of .jpg/.JPG/.jpeg photographs.
            output_dir: Directory to build a scene-shaped output in.

        Returns:
            A MaskResult with per-image counts and elapsed time.
        """
        # Locally imported so `augenblick <stage> --list` does not require these on the
        # login node.
        from PIL import Image

        self.validate(images_dir)
        Image.MAX_IMAGE_PIXELS = None
        images_dir = images_dir.resolve()
        out_dir = output_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # Symlink images/ rather than copying 26 MP photographs.
        images_link = out_dir / "images"
        if images_link.is_symlink() or images_link.exists():
            target = images_link.resolve() if images_link.exists() else None
            if target != images_dir:
                raise SceneError(
                    f"{images_link} already exists and points elsewhere ({target}); "
                    "refusing to overwrite")
        else:
            os.symlink(images_dir, images_link)

        masks_dir = out_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in images_dir.iterdir()
            if p.suffix in self.IMAGE_SUFFIXES and p.is_file())

        cfg = self.config
        result = MaskResult(
            output_dir=out_dir, elapsed=0.0, num_images=len(image_paths))

        t0 = time.time()
        timer = StageTimer(self.title, 1, self.header(images_dir, out_dir))
        with timer.stage("Masking"):
            for i, path in enumerate(image_paths):
                dest = masks_dir / f"{path.stem}.png"
                if cfg.only_missing and dest.exists():
                    result.num_reused += 1
                    continue
                try:
                    mask = self.mask_for(path)
                except Exception as exc:
                    logger.warning(f"{path.name}: mask_for raised {type(exc).__name__}: {exc}")
                    result.num_failed += 1
                    dest.unlink(missing_ok=True)
                    continue

                with Image.open(path) as im:
                    expected = im.size[::-1]
                if mask.shape != expected:
                    logger.warning(
                        f"{path.name}: mask is {mask.shape}, expected {expected} — rejected")
                    result.num_failed += 1
                    dest.unlink(missing_ok=True)
                    continue

                mask = postprocess_mask(mask, cfg.keep_largest, cfg.fill_holes)
                frac = foreground_fraction(mask)
                if not (cfg.min_foreground <= frac <= cfg.max_foreground):
                    logger.warning(
                        f"{path.name}: foreground fraction {frac:.3f} out of "
                        f"[{cfg.min_foreground}, {cfg.max_foreground}] — rejected")
                    result.num_failed += 1
                    dest.unlink(missing_ok=True)
                    continue

                write_mask(mask, dest)
                result.num_written += 1

        timer.summary({
            "Output": out_dir,
            "Images": result.num_images,
            "Written": result.num_written,
            "Reused": result.num_reused,
            "Failed": result.num_failed,
        })
        result.elapsed = time.time() - t0
        return result
