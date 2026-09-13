"""The Method ABC that every pipeline stage implements."""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

from augenblick.core.config import config_from_namespace
from augenblick.core.errors import SceneError
from augenblick.core.scene import Scene

logger = logging.getLogger(__name__)

Input = TypeVar("Input")


@dataclass
class StageResult:
    """What a completed stage produced, for logging and for chaining stages.

    Args:
        output_dir: Directory the stage wrote its results into.
        elapsed: Wall-clock seconds the stage took.
        details: Backend-specific extras, such as mesh paths or point counts.
    """

    output_dir: Path
    elapsed: float
    details: dict[str, object] = field(default_factory=dict)


class Method(ABC, Generic[Input]):
    """Base for any pipeline stage that transforms an input directory.

    Subclasses set `name` and `config_cls`, then implement `run`. Registration is
    performed by the register_sfm / register_reconstruction / register_mask decorators.
    The input type varies by stage: sfm/recon take a Scene, mask takes a raw images dir.
    A mixin (SceneInputMixin / ImagesInputMixin) supplies build_input + validate.
    """

    name: ClassVar[str]
    config_cls: ClassVar[type]
    accepts_passthrough: ClassVar[bool] = False

    def __init__(self, config):
        self.config = config

    @classmethod
    def from_namespace(cls, ns):
        """Build the method from parsed CLI arguments."""
        return cls(config_from_namespace(cls.config_cls, ns))

    @classmethod
    @abstractmethod
    def build_input(cls, path: Path) -> Input:
        """Wrap the resolved --scene or --images path into the method's input type."""

    @abstractmethod
    def validate(self, inp: Input) -> None:
        """Raise SceneError if the input lacks what this method requires."""

    @abstractmethod
    def run(self, inp: Input, output_dir: Path) -> StageResult:
        """Execute the stage.

        Args:
            inp: The stage's input (a Scene, or a raw images directory).
            output_dir: Directory to write results into.

        Returns:
            A StageResult describing what was produced.
        """


class SceneInputMixin:
    """Input: a COLMAP Scene."""

    @classmethod
    def build_input(cls, path: Path) -> Scene:
        return Scene(path)

    def validate(self, scene: Scene) -> None:
        scene.require_images()


class ImagesInputMixin:
    """Input: a flat directory of images."""

    IMAGE_SUFFIXES: ClassVar[frozenset] = frozenset({".jpg", ".jpeg", ".JPG", ".JPEG"})

    @classmethod
    def build_input(cls, path: Path) -> Path:
        return path

    def validate(self, images_dir: Path) -> None:
        if not images_dir.is_dir():
            raise SceneError(f"no images directory at {images_dir}")
        if not any(p.suffix in self.IMAGE_SUFFIXES for p in images_dir.iterdir()):
            raise SceneError(
                f"{images_dir} has no files with an accepted suffix "
                f"({sorted(self.IMAGE_SUFFIXES)})")
