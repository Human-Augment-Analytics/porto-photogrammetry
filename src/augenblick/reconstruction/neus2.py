"""NeuS2: a hash-grid SDF trained by volume rendering, meshed by marching cubes.

All NeuS2 code lives in src/libs/neus2 (prepare, train, extract_mesh, render); this module
only builds their argv, validates the scene, exports cameras and cleans up.
"""
import importlib.util
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from augenblick.core.errors import SceneError
from augenblick.core.registry import register_reconstruction
from augenblick.core.scene import Scene
from augenblick.reconstruction.base import (
    LIBS_DIR, EvalParams, ResolutionParams, Stage, SubprocessBackend)

__all__ = ["Neus2Backend", "Neus2Config"]


@dataclass(frozen=True)
class Neus2Config(EvalParams, ResolutionParams):
    """Training, bounding and mesh-extraction parameters forwarded to the NeuS2 stage scripts."""

    n_steps: int = field(default=30000, metadata={
        "help": "Training steps; upstream's DTU recipe is 15000, which under-trains fine detail at ~400 views"})
    ek_loss_weight: float = field(default=0.01, metadata={"help": "Eikonal loss weight"})
    mask_loss_weight: float = field(default=0.0, metadata={
        "help": "Mask BCE weight; 0 relies on alpha compositing, which already supervises the mask"})
    mc_res: int = field(default=1024, metadata={
        "help": "Marching-cubes grid per axis (upstream default 512)"})
    num_cluster: int = field(default=1, metadata={
        "help": "Connected components to keep in the mesh; 0 keeps every component"})
    bound_percentile: float = field(default=1.0, metadata={
        "help": "SfM-point percentile box that centres the bounding sphere"})
    bound_margin: float = field(default=1.1, metadata={"help": "Bounding-sphere radius multiplier"})
    evaluation_resolution: int = field(default=-1, metadata={
        "help": "Held-out evaluation downscale, independent of training; -1 caps the long side at 1600"})
    keep_staged: bool = field(default=False, metadata={
        "help": "Keep the staged RGBA training images after a successful run"})


@register_reconstruction
class Neus2Backend(SubprocessBackend):
    """Runs NeuS2's prepare, train, extract_mesh and (with --eval) render stage scripts."""

    name: ClassVar[str] = "neus2"
    config_cls: ClassVar[type] = Neus2Config
    backend_dir: ClassVar[Path] = LIBS_DIR / "neus2"
    title: ClassVar[str] = "NeuS2 Reconstruction Pipeline"

    def validate(self, scene: Scene) -> None:
        """Require pyngp, pinhole cameras and a mask for every registered image.

        Raises:
            SceneError: If pyngp is not built into this env, a camera is not pinhole, a
                registered image has no mask, or evaluation_resolution is invalid.
        """
        c = self.config
        if c.eval and (c.evaluation_resolution == 0 or c.evaluation_resolution < -1):
            raise SceneError("evaluation_resolution must be -1 or a positive downscale")
        super().validate(scene)
        if importlib.util.find_spec("pyngp") is None:
            raise SceneError(
                f"NeuS2 needs the pyngp module, which is not built into {sys.prefix}; build "
                "src/libs/neus2's CMake target pyngp for this env and copy it into site-packages")
        # Imported here so `augenblick recon --list` stays light on a login node.
        from augenblick.eval.cameras import camera_records
        from augenblick.eval.split import registered_stems

        models = sorted({r["camera_model"] for r in camera_records(scene.model_dir)}
                        - {"SIMPLE_PINHOLE", "PINHOLE"})
        if models:
            raise SceneError(f"NeuS2 needs undistorted pinhole cameras; this scene has {models}")
        stems = registered_stems(scene.model_dir)
        missing = [s for s in stems if not (scene.masks_dir / f"{s}.png").is_file()]
        if missing:
            raise SceneError(
                f"NeuS2 trains on masked views only (the mask is its alpha channel), but "
                f"{len(missing)} of {len(stems)} registered images lack masks/<stem>.png "
                f"(e.g. {missing[:5]}); run `augenblick mask` first")

    def stages(self, scene: Scene, output_dir: Path) -> list[Stage]:
        """Build the prepare, train, mesh and optional held-out render invocations."""
        c, lib, py = self.config, self.backend_dir, sys.executable
        prepare = [py, str(lib / "prepare.py"), "-s", str(scene.root), "-m", str(output_dir),
                   "-r", str(c.resolution), "--bound_percentile", str(c.bound_percentile),
                   "--bound_margin", str(c.bound_margin)]
        if c.eval:
            prepare.append("--eval")
        stages = [
            Stage("NeuS2 staging", prepare),
            Stage("NeuS2 training", [
                py, str(lib / "train.py"), "-m", str(output_dir), "--n_steps", str(c.n_steps),
                "--ek_loss_weight", str(c.ek_loss_weight),
                "--mask_loss_weight", str(c.mask_loss_weight)]),
            Stage("Mesh extraction", [
                py, str(lib / "extract_mesh.py"), "-m", str(output_dir),
                "--n_steps", str(c.n_steps), "--mc_res", str(c.mc_res),
                "--num_cluster", str(c.num_cluster)]),
        ]
        # After meshing, as in GW, so a failed render still leaves a mesh.
        if c.eval:
            stages.append(Stage("Held-out rendering", [
                py, str(lib / "render.py"), "-s", str(scene.root), "-m", str(output_dir),
                "--n_steps", str(c.n_steps),
                "--evaluation_resolution", str(c.evaluation_resolution)]))
        return stages

    def finalize(self, scene: Scene, output_dir: Path) -> None:
        """Export the lineage cameras, then drop the staged images unless asked to keep them."""
        from augenblick.eval.cameras import export_cameras

        export_cameras(scene.model_dir, output_dir / "cameras.json")
        staged = output_dir / "neus2" / "scene" / "images"
        if not self.config.keep_staged and staged.is_dir():
            shutil.rmtree(staged)

    def mesh_path(self, output_dir: Path) -> Path:
        return output_dir / "mesh" / "neus2_post.ply"
