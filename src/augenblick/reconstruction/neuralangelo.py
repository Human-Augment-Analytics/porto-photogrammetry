"""Neuralangelo surface reconstruction from an Augenblick SfM scene."""
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from augenblick.core.registry import register_reconstruction
from augenblick.core.errors import SceneError
from augenblick.core.scene import Scene
from augenblick.reconstruction.base import LIBS_DIR, REPO_ROOT, EvalParams, Stage, SubprocessBackend


@dataclass(frozen=True)
class NeuralangeloConfig(EvalParams):
    upstream: Path = field(default=LIBS_DIR / "neuralangelo")
    iterations: int = 500_000
    resolution: int = field(default=1600, metadata={"help": "Maximum image dimension"})
    rays: int = 512
    batch_size: int = 2
    shortened_schedule: bool = False
    mesh_resolution: int = 1024
    block_resolution: int = 64
    seed: int = 22
    reference_mesh: str = field(default="", metadata={"help": "Optional geometry reference"})


@register_reconstruction
class NeuralangeloBackend(SubprocessBackend):
    name: ClassVar[str] = "neuralangelo"
    config_cls: ClassVar[type] = NeuralangeloConfig
    backend_dir: ClassVar[Path] = REPO_ROOT
    title: ClassVar[str] = "Neuralangelo Reconstruction Pipeline"

    def validate(self, scene: Scene) -> None:
        super().validate(scene)
        if not (self.config.upstream / "train.py").is_file():
            raise SceneError(
                f"Neuralangelo source not found: {self.config.upstream}. "
                "Run git submodule update --init --recursive src/libs/neuralangelo "
                "or specify --upstream.")
        if self.config.mesh_resolution < 2 or self.config.block_resolution < 2:
            raise SceneError("Extraction resolutions must be at least two")
        if self.config.reference_mesh and not Path(self.config.reference_mesh).is_file():
            raise SceneError(f"Reference mesh not found: {self.config.reference_mesh}")

    def stages(self, scene: Scene, output_dir: Path) -> list[Stage]:
        config = self.config
        common = [sys.executable, "-m", "augenblick.reconstruction.neuralangelo_runner",
                  "--upstream", str(config.upstream), "--output", str(output_dir)]
        prepare = common + ["prepare", "--scene", str(scene.root),
                            "--iterations", str(config.iterations),
                            "--resolution", str(config.resolution), "--rays", str(config.rays),
                            "--batch-size", str(config.batch_size), "--seed", str(config.seed)]
        if config.eval:
            prepare.append("--eval")
        if config.shortened_schedule:
            prepare.append("--shortened-schedule")
        return [
            Stage("Prepare posed images", prepare),
            Stage("Neuralangelo training", common + ["train"]),
            Stage("SDF mesh extraction", common + ["extract", "--resolution",
                  str(config.mesh_resolution), "--block-resolution", str(config.block_resolution)]),
            *([Stage("Held-out surface rendering", common + ["render"])] if config.eval else []),
        ]

    def mesh_path(self, output_dir: Path) -> Path:
        return output_dir / "mesh" / "neuralangelo.ply"

    def run(self, scene: Scene, output_dir: Path):
        result = super().run(scene, output_dir)
        if self.config.reference_mesh:
            from augenblick.core import process

            process.run([sys.executable, "-m", "augenblick.eval.mesh", "--ours",
                         str(self.mesh_path(output_dir)), "--reference", self.config.reference_mesh,
                         "--samples", "200000", "--seeds", "0", "1", "2",
                         "--json-out", str(output_dir / "geometry_metrics.json")])
            result.details["geometry_metrics"] = output_dir / "geometry_metrics.json"
        return result