"""2DGS: surfel training followed by rendering and TSDF mesh extraction."""
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from augenblick.core.registry import register_reconstruction
from augenblick.core.scene import Scene
from augenblick.reconstruction.base import (
    LIBS_DIR, EvalParams, ResolutionParams, Stage, SubprocessBackend)

logger = logging.getLogger(__name__)

TWODGS_DIR = LIBS_DIR / "2dgs"
TRAIN_SCRIPT = TWODGS_DIR / "train.py"
RENDER_SCRIPT = TWODGS_DIR / "render.py"


@dataclass(frozen=True)
class TwoDGSConfig(EvalParams, ResolutionParams):
    """Training and mesh-extraction parameters forwarded to 2DGS."""

    iterations: int = field(default=30_000, metadata={"help": "Training iterations"})
    test_iterations: list[int] = field(default_factory=lambda: [7_000, 30_000])
    save_iterations: list[int] = field(default_factory=lambda: [7_000, 30_000])
    white_background: bool = field(default=False, metadata={"help": "Use white background"})
    lambda_dist: float = field(default=0.0, metadata={
        "help": "Distortion loss weight (upstream default 0.0; 100 bounded / 1000 unbounded recommended)"})
    lambda_normal: float = field(default=0.05, metadata={"help": "Normal consistency loss weight"})
    depth_ratio: float = field(default=0.0, metadata={
        "help": "Expected (0.0) vs median (1.0) depth blend; 1.0 for bounded scenes"})
    densify_grad_threshold: float = field(default=0.0002, metadata={
        "help": "Lower → more Gaussians (more detail, more memory)"})
    densify_until_iter: int = field(default=15_000, metadata={
        "help": "Iteration to stop densifying; extend alongside --iterations"})
    opacity_cull: float = field(default=0.05, metadata={"help": "Opacity threshold for pruning"})
    voxel_size: float = field(default=-1.0, metadata={"help": "TSDF voxel size (auto if negative)"})
    depth_trunc: float = field(default=-1.0, metadata={"help": "Max depth for TSDF (auto if negative)"})
    sdf_trunc: float = field(default=-1.0, metadata={"help": "SDF truncation (auto if negative)"})
    num_cluster: int = field(default=50, metadata={"help": "Connected components to keep in mesh"})
    unbounded: bool = field(default=False, metadata={
        "help": "Use unbounded mesh extraction (marching cubes)"})
    mesh_res: int = field(default=4096, metadata={"help": "Resolution for unbounded mesh extraction"})
    skip_mesh: bool = field(default=False, metadata={"help": "Skip mesh extraction (render only)"})
    skip_train_export: bool = field(default=False, metadata={
        "help": "Skip writing per-training-view PNGs; mesh extraction is unaffected"})


@register_reconstruction
class TwoDGSBackend(SubprocessBackend):
    """Runs 2DGS train.py then render.py against a COLMAP scene."""

    name: ClassVar[str] = "2dgs"
    config_cls: ClassVar[type] = TwoDGSConfig
    backend_dir: ClassVar[Path] = TWODGS_DIR
    title: ClassVar[str] = "2DGS Reconstruction Pipeline"

    def stages(self, scene: Scene, output_dir: Path) -> list[Stage]:
        """Build the train and render invocations."""
        c = self.config
        scene_dir = scene.root
        train_cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "-s", str(scene_dir),
            "-m", str(output_dir),
            "--iterations", str(c.iterations),
            "--test_iterations", *[str(i) for i in c.test_iterations],
            "--save_iterations", *[str(i) for i in c.save_iterations],
            "--lambda_dist", str(c.lambda_dist),
            "--lambda_normal", str(c.lambda_normal),
            "--depth_ratio", str(c.depth_ratio),
            "--densify_grad_threshold", str(c.densify_grad_threshold),
            "--densify_until_iter", str(c.densify_until_iter),
            "--opacity_cull", str(c.opacity_cull),
        ]
        if c.white_background:
            train_cmd.append("--white_background")
        if c.eval:
            train_cmd.append("--eval")
        if c.resolution != -1:
            train_cmd += ["-r", str(c.resolution)]

        render_cmd = [
            sys.executable, str(RENDER_SCRIPT),
            "-s", str(scene_dir),
            "-m", str(output_dir),
            "--voxel_size", str(c.voxel_size),
            "--depth_trunc", str(c.depth_trunc),
            "--sdf_trunc", str(c.sdf_trunc),
            "--num_cluster", str(c.num_cluster),
            "--mesh_res", str(c.mesh_res),
        ]
        # Held-out views only exist to be rendered when there is a split.
        if not c.eval:
            render_cmd.append("--skip_test")
        else:
            render_cmd.append("--eval")
        if c.resolution != -1:
            render_cmd += ["-r", str(c.resolution)]
        if c.skip_train_export:
            render_cmd.append("--skip_train")
        if c.unbounded:
            render_cmd.append("--unbounded")
        if c.skip_mesh:
            render_cmd.append("--skip_mesh")

        return [
            Stage("2DGS training", train_cmd),
            Stage("Rendering + mesh extraction", render_cmd),
        ]

    def mesh_path(self, output_dir: Path) -> Path:
        """TSDF-fused mesh, written under the iteration the run trained to."""
        return output_dir / "train" / f"ours_{self.config.iterations}" / "fuse_post.ply"
