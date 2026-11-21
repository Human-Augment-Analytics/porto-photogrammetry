"""Meshroom dense reconstruction from shared SfM, followed by held-out textured-mesh rendering."""
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from augenblick.core.errors import SceneError
from augenblick.core.registry import register_reconstruction
from augenblick.core.scene import Scene
from augenblick.reconstruction.base import EvalParams, ResolutionParams, Stage, SubprocessBackend


@dataclass(frozen=True)
class MeshroomConfig(EvalParams, ResolutionParams):
    sfmdata: str = field(default="", metadata={"help": "Preconverted all-view SfMData matching shared COLMAP"})
    template: str = field(default="", metadata={"help": "Saved all-view Meshroom graph or photogrammetry template"})
    meshroom_root: str = field(default="", metadata={"help": "Meshroom source/install root; defaults to MESHROOM_ROOT"})
    meshroom_python: str = field(default="", metadata={"help": "Python for Meshroom; defaults to MESHROOM_PYTHON"})
    alicevision_root: str = field(default="", metadata={"help": "AliceVision install root; defaults to ALICEVISION_ROOT"})
    keep_cache: bool = field(default=False, metadata={"help": "Retain dense cache after successful mesh publication"})


@register_reconstruction
class MeshroomBackend(SubprocessBackend):
    name: ClassVar[str] = "meshroom"
    config_cls: ClassVar[type] = MeshroomConfig
    backend_dir: ClassVar[Path] = Path(".")
    use_cwd: ClassVar[bool] = False
    title: ClassVar[str] = "Meshroom shared-SfM dense pipeline"

    def roots(self):
        c = self.config
        root = c.meshroom_root or os.environ.get("MESHROOM_ROOT", "")
        alice = c.alicevision_root or os.environ.get("ALICEVISION_ROOT", "")
        python = c.meshroom_python or os.environ.get("MESHROOM_PYTHON", sys.executable)
        if not root or not alice:
            raise SceneError("Meshroom needs meshroom_root and alicevision_root (flags or environment)")
        return Path(root), Path(alice), python

    def prepare(self, scene: Scene, output_dir: Path) -> Scene:
        from augenblick.eval.meshroom import prepare
        root, alice, python = self.roots()
        if not (root / "bin" / "meshroom_compute").is_file():
            raise SceneError("meshroom_compute is missing from the configured Meshroom root")
        sfmdata = Path(self.config.sfmdata) if self.config.sfmdata else scene.root / "alicevision.sfm"
        template = Path(self.config.template) if self.config.template else scene.root / "meshroom_template.mg"
        if not template.is_file():
            template = alice / "share" / "meshroom" / "photogrammetry.mg"
        if not sfmdata.is_file() or not template.is_file():
            raise SceneError("Meshroom requires all-view SfMData and a dense pipeline template")
        prepare(scene, output_dir, sfmdata, template, self.config.eval)
        return scene

    def stages(self, scene: Scene, output_dir: Path) -> list[Stage]:
        root, alice, python = self.roots()
        command = [sys.executable, "-m", "augenblick.eval.meshroom", "--output", str(output_dir),
                   "--meshroom-root", str(root), "--meshroom-python", python,
                   "--alicevision-root", str(alice)]
        if self.config.keep_cache:
            command.append("--keep-cache")
        stages = [Stage("Dense reconstruction and texturing", command)]
        if self.config.eval:
            stages.append(Stage("Held-out textured-mesh rendering", [
                sys.executable, "-m", "augenblick.eval.mesh_render", "--scene", str(scene.root),
                "--output", str(output_dir), "--mesh", str(self.mesh_path(output_dir)),
                "--resolution", str(self.config.resolution), "--mesh-frame", "alicevision_obj"]))
        return stages

    def mesh_path(self, output_dir: Path) -> Path:
        return output_dir / "textured_mesh" / "texturedMesh.obj"
