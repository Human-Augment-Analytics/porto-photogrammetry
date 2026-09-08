"""PGSR: planar Gaussians trained on a flattened copy of the scene, then TSDF meshing."""
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from augenblick.core.registry import register_reconstruction
from augenblick.core.scene import Scene
from augenblick.eval.split import copy_split
from augenblick.reconstruction.base import (
    LIBS_DIR, EvalParams, ResolutionParams, Stage, SubprocessBackend)

logger = logging.getLogger(__name__)

PGSR_DIR = LIBS_DIR / "pgsr"
TRAIN_SCRIPT = PGSR_DIR / "train.py"
RENDER_SCRIPT = PGSR_DIR / "render.py"


@dataclass(frozen=True)
class PgsrConfig(EvalParams, ResolutionParams):
    """Training and mesh-extraction parameters forwarded to PGSR."""

    iterations: int = field(default=30_000, metadata={"help": "Training iterations"})
    test_iterations: list[int] = field(default_factory=lambda: [7_000, 30_000])
    save_iterations: list[int] = field(default_factory=lambda: [7_000, 30_000])
    max_abs_split_points: int = field(default=0, metadata={
        "help": "Max absolute split points (0 to disable)"})
    opacity_cull_threshold: float = field(default=0.05, metadata={
        "help": "Opacity threshold for pruning"})
    white_background: bool = field(default=False, metadata={"help": "Use white background"})
    lambda_dssim: float = field(default=0.2, metadata={"help": "SSIM loss weight"})
    single_view_weight: float = field(default=0.015, metadata={
        "help": "Normal consistency weight (raise for sharper surfaces)"})
    multi_view_ncc_weight: float = field(default=0.15, metadata={
        "help": "NCC patch-matching weight (cross-view photometric consistency)"})
    multi_view_geo_weight: float = field(default=0.03, metadata={
        "help": "Multi-view geometric consistency weight"})
    multi_view_num: int = field(default=8, metadata={
        "help": "Number of nearest views per frame for multi-view losses"})
    densify_grad_threshold: float = field(default=0.0002, metadata={
        "help": "Lower → more Gaussians (more detail, more memory)"})
    densify_until_iter: int = field(default=15_000, metadata={
        "help": "Iteration to stop densifying; extend alongside --iterations"})
    max_depth: float = field(default=10.0, metadata={"help": "Max depth for TSDF integration"})
    voxel_size: float = field(default=0.001, metadata={"help": "TSDF voxel size"})
    num_cluster: int = field(default=1, metadata={"help": "Connected components to keep in mesh"})
    use_depth_filter: bool = field(default=False, metadata={
        "help": "Drop grazing-angle depths before TSDF fusion"})
    skip_mesh: bool = field(default=False, metadata={"help": "Skip mesh extraction (render only)"})


@register_reconstruction
class PgsrBackend(SubprocessBackend):
    """Runs PGSR train.py then render.py against a scene whose sparse/ has been flattened."""

    name: ClassVar[str] = "pgsr"
    config_cls: ClassVar[type] = PgsrConfig
    backend_dir: ClassVar[Path] = PGSR_DIR
    title: ClassVar[str] = "PGSR Reconstruction Pipeline"

    def prepare(self, scene: Scene, output_dir: Path) -> Scene:
        """Copy the scene and flatten sparse/0/ into sparse/, as PGSR reads sparse/ directly.

        Args:
            scene: The input COLMAP scene.
            output_dir: Directory the prepared copy is written under.

        Returns:
            A Scene pointing at the prepared copy.
        """
        scene_dir = scene.root
        pgsr_scene = output_dir / "scene"

        if pgsr_scene.exists():
            logger.info(f"Prepared scene already exists at {pgsr_scene}, reusing")
            # A copy prepared by an earlier non-eval run has no split.json, and PGSR would
            # then fall back to its own llffhold rule while the other backends used the file.
            copy_split(scene_dir, pgsr_scene)
            return Scene(pgsr_scene)

        logger.info(f"Copying scene from {scene_dir} to {pgsr_scene}")
        for subdir in ["images", "masks", "sparse"]:
            src = scene_dir / subdir
            if src.is_dir():
                shutil.copytree(src, pgsr_scene / subdir)
        # PGSR reads split.json from the copy, so the held-out set has to travel with it.
        copy_split(scene_dir, pgsr_scene)

        sparse_0 = pgsr_scene / "sparse" / "0"
        sparse = pgsr_scene / "sparse"
        if sparse_0.is_dir():
            logger.info("Flattening sparse/0/ -> sparse/")
            for item in sparse_0.iterdir():
                shutil.move(str(item), str(sparse / item.name))
            sparse_0.rmdir()
        else:
            logger.info("No sparse/0/ found; assuming sparse/ is already flat")

        return Scene(pgsr_scene)

    def stages(self, scene: Scene, output_dir: Path) -> list[Stage]:
        """Build the train and render invocations; render takes only -m, no -s."""
        c = self.config
        train_cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "-s", str(scene.root),
            "-m", str(output_dir),
            "--iterations", str(c.iterations),
            "--test_iterations", *[str(i) for i in c.test_iterations],
            "--save_iterations", *[str(i) for i in c.save_iterations],
            "--max_abs_split_points", str(c.max_abs_split_points),
            "--opacity_cull_threshold", str(c.opacity_cull_threshold),
            "--lambda_dssim", str(c.lambda_dssim),
            "--single_view_weight", str(c.single_view_weight),
            "--multi_view_ncc_weight", str(c.multi_view_ncc_weight),
            "--multi_view_geo_weight", str(c.multi_view_geo_weight),
            "--multi_view_num", str(c.multi_view_num),
            "--densify_grad_threshold", str(c.densify_grad_threshold),
            "--densify_until_iter", str(c.densify_until_iter),
        ]
        if c.white_background:
            train_cmd.append("--white_background")
        if c.eval:
            train_cmd.append("--eval")
        if c.resolution != -1:
            train_cmd += ["-r", str(c.resolution)]

        render_cmd = [
            sys.executable, str(RENDER_SCRIPT),
            "-m", str(output_dir),
            "--max_depth", str(c.max_depth),
            "--voxel_size", str(c.voxel_size),
            "--num_cluster", str(c.num_cluster),
        ]
        # Held-out views only exist to be rendered when there is a split; render.py reads the
        # scene path and resolution back from the model's cfg_args, so they are not repeated.
        if not c.eval:
            render_cmd.append("--skip_test")
        if c.use_depth_filter:
            render_cmd.append("--use_depth_filter")
        # Upstream render.py spells mesh skipping as --skip_train.
        if c.skip_mesh:
            render_cmd.append("--skip_train")

        return [
            Stage("PGSR training", train_cmd),
            Stage("Rendering + mesh extraction", render_cmd),
        ]

    def mesh_path(self, output_dir: Path) -> Path:
        """TSDF-fused mesh written under the model directory."""
        return output_dir / "mesh" / "tsdf_fusion_post.ply"
