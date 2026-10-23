"""Shared interfaces for methods that prepare raw inputs for later stages."""

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from augenblick.core.errors import SceneError
from augenblick.core.method import Method, StageResult


@dataclass
class PreparationResult(StageResult):
    """A completed preparation run and the number of images it processed."""

    num_images: int = 0
    report_path: Path | None = None


class PreparationMethod(Method[Path]):
    """Consumes a raw input tree and writes a prepared output tree."""

    IMAGE_SUFFIXES: ClassVar[frozenset[str]] = frozenset(
        {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    )

    @classmethod
    def build_input(cls, path: Path) -> Path:
        """Return the raw input directory used by the preparation method."""
        return path

    def validate(self, input_dir: Path) -> None:
        """Require an input directory containing at least one supported image."""
        if not input_dir.is_dir():
            raise SceneError(f"no input directory at {input_dir}")
        if not any(
            path.is_file() and path.suffix.lower() in self.IMAGE_SUFFIXES
            for path in input_dir.rglob("*")
        ):
            raise SceneError(
                f"{input_dir} has no files with an accepted suffix "
                f"({sorted(self.IMAGE_SUFFIXES)})"
            )

    @abstractmethod
    def run(self, input_dir: Path, output_dir: Path) -> PreparationResult:
        """Prepare the input tree and return a summary of the generated output."""
