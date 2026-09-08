"""Gaussian Wrapping: training, pivot-based mesh extraction, and texture refinement."""
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

from augenblick.core.registry import register_reconstruction
from augenblick.core.scene import Scene
from augenblick.reconstruction.base import LIBS_DIR, EvalParams, Stage, SubprocessBackend

logger = logging.getLogger(__name__)

GW_DIR = LIBS_DIR / "gaussian_wrapping"
TRAIN_SCRIPT = GW_DIR / "train.py"
RENDER_SCRIPT = GW_DIR / "render.py"
EXTRACT_SCRIPT = GW_DIR / "pivot_based_mesh_extraction.py"
TEXTURE_SCRIPT = GW_DIR / "texture_mesh.py"

DEFAULT_TRAIN_FEATURE_DC_LR = 0.0013
DEFAULT_TRAIN_FEATURE_REST_LR = 0.00011
DEFAULT_TRAIN_POSITION_LR_INIT = 0.00016
DEFAULT_TRAIN_POSITION_LR_FINAL = 0.0000016
DEFAULT_TRAIN_POSITION_LR_DELAY_MULT = 0.01
DEFAULT_TRAIN_POSITION_LR_MAX_STEPS = 30_000
DEFAULT_TRAIN_OPACITY_LR = 0.05
DEFAULT_TRAIN_SCALING_LR = 0.005
DEFAULT_TRAIN_ROTATION_LR = 0.001
DEFAULT_TRAIN_APPEARANCE_EMBEDDINGS_LR = 0.001
DEFAULT_TRAIN_APPEARANCE_NETWORK_LR = 0.001
DEFAULT_TRAIN_GAUSSIAN_FEATURES_LR = 0.05 / 2.0
DEFAULT_TRAIN_PGSR_APPEARANCE_LR = 0.001
DEFAULT_MAX_GAUSSIANS = 6_000_000

DEFAULT_EXTRACT_N_PIVOTS = 2
DEFAULT_EXTRACT_STD_FACTOR = 3.0
DEFAULT_EXTRACT_N_BINARY_STEPS = 10
DEFAULT_EXTRACT_ISOSURFACE_VALUE = 0.0

DEFAULT_TEXTURE_N_ITER = 1000
DEFAULT_TEXTURE_LAMBDA_DSSIM = 0.2
DEFAULT_TEXTURE_LR = 0.0025
DEFAULT_TEXTURE_SH_DEGREE = 0


@dataclass(frozen=True)
class GWConfig(EvalParams):
    """Training, extraction, and texture-refinement parameters for Gaussian Wrapping."""

    resolution: Optional[int] = field(default=None, metadata={
        "short": "-r", "help": "Image resolution override forwarded to all stages"})
    iterations: int = field(default=30_000, metadata={"help": "Training iterations"})
    sh_degree: int = field(default=3, metadata={"help": "Max spherical harmonics degree"})
    max_gaussians: int = field(default=DEFAULT_MAX_GAUSSIANS, metadata={
        "help": "Maximum number of gaussians"})
    densify_until_iter: int = field(default=15_000, metadata={
        "help": "Iteration to stop densifying"})
    densify_grad_threshold: float = field(default=0.0002, metadata={
        "help": "Lower → denser Gaussian cloud (more detail, more memory)"})
    lambda_depth_normal: float = field(default=0.05, metadata={
        "help": "Depth-normal consistency loss weight"})
    multiview_factor: float = field(default=1.0, metadata={"help": "Multi-view loss factor"})
    position_lr_init: float = DEFAULT_TRAIN_POSITION_LR_INIT
    position_lr_final: float = DEFAULT_TRAIN_POSITION_LR_FINAL
    position_lr_delay_mult: float = DEFAULT_TRAIN_POSITION_LR_DELAY_MULT
    position_lr_max_steps: int = DEFAULT_TRAIN_POSITION_LR_MAX_STEPS
    feature_dc_lr: float = DEFAULT_TRAIN_FEATURE_DC_LR
    feature_rest_lr: float = DEFAULT_TRAIN_FEATURE_REST_LR
    opacity_lr: float = DEFAULT_TRAIN_OPACITY_LR
    scaling_lr: float = DEFAULT_TRAIN_SCALING_LR
    rotation_lr: float = DEFAULT_TRAIN_ROTATION_LR
    appearance_embeddings_lr: float = DEFAULT_TRAIN_APPEARANCE_EMBEDDINGS_LR
    appearance_network_lr: float = DEFAULT_TRAIN_APPEARANCE_NETWORK_LR
    gaussian_features_lr: float = DEFAULT_TRAIN_GAUSSIAN_FEATURES_LR
    pgsr_appearance_lr: float = DEFAULT_TRAIN_PGSR_APPEARANCE_LR
    extract_iteration: Optional[int] = field(default=None, metadata={
        "help": "Checkpoint iteration to load for extraction and texture refinement. "
                "Defaults to --iterations."})
    n_pivots: int = field(default=DEFAULT_EXTRACT_N_PIVOTS, metadata={
        "help": "Number of pivots for mesh extraction"})
    std_factor: float = field(default=DEFAULT_EXTRACT_STD_FACTOR, metadata={
        "help": "Pivot offset scale relative to Gaussian extent during mesh extraction"})
    use_searched_pivots: bool = field(default=False, metadata={
        "help": "Refine extraction pivots by searching along the normal direction"})
    use_smallest_axis_as_normal: bool = field(default=False, metadata={
        "help": "Use the Gaussian's smallest axis as the extraction normal instead of learned normals"})
    n_binary_steps: int = field(default=DEFAULT_EXTRACT_N_BINARY_STEPS, metadata={
        "help": "Binary search refinement steps"})
    isosurface_value: float = field(default=DEFAULT_EXTRACT_ISOSURFACE_VALUE, metadata={
        "help": "Isosurface value"})
    postprocess: bool = field(default=True, metadata={
        "help": "Postprocess the extracted mesh (default: on)"})
    filter_large_edges: bool = field(default=True, metadata={
        "help": "Filter triangles with large edges (default: on)"})
    texture_n_iter: int = field(default=DEFAULT_TEXTURE_N_ITER, metadata={
        "help": "Texture refinement iterations"})
    texture_lambda_dssim: float = field(default=DEFAULT_TEXTURE_LAMBDA_DSSIM, metadata={
        "help": "SSIM loss weight for texture refinement"})
    texture_lr: float = field(default=DEFAULT_TEXTURE_LR, metadata={
        "help": "Learning rate for texture refinement"})
    texture_sh_degree: int = field(default=DEFAULT_TEXTURE_SH_DEGREE, metadata={
        "help": "SH degree used while baking the texture"})


@register_reconstruction
class GWBackend(SubprocessBackend):
    """Runs GW training, pivot-based extraction, and texture refinement."""

    name: ClassVar[str] = "gw"
    config_cls: ClassVar[type] = GWConfig
    backend_dir: ClassVar[Path] = GW_DIR
    # GW's scripts import from their own directory, which Python adds to sys.path only for the
    # script's own path; setting cwd would not help and upstream expects absolute invocation.
    use_cwd: ClassVar[bool] = False
    title: ClassVar[str] = "Gaussian Wrapping Reconstruction Pipeline"
    accepts_passthrough: ClassVar[bool] = True

    def __init__(self, config, passthrough: list[str] | None = None):
        super().__init__(config)
        self.passthrough = list(passthrough or [])

    @classmethod
    def from_namespace(cls, ns, passthrough: list[str] | None = None):
        """Build the backend from parsed args plus any unrecognised train.py flags."""
        from augenblick.core.config import config_from_namespace

        return cls(config_from_namespace(cls.config_cls, ns), passthrough)

    @property
    def extract_iteration(self) -> int:
        """Checkpoint iteration for extraction and texturing; defaults to the trained iterations."""
        return self.config.extract_iteration or self.config.iterations

    def get_mesh_path(self, model_path, n_pivots, postprocess):
        """Path GW writes the extracted mesh to, named from the pivot and postprocess settings."""
        mesh_name = f"mesh_ours_{n_pivots}pivots"
        if postprocess:
            mesh_name += "_post"
        mesh_name += ".ply"
        return os.path.join(str(model_path), mesh_name)

    def get_textured_mesh_path(self, model_path, mesh_path, texture_n_iter):
        """Path GW writes the texture-refined mesh to, suffixed with the final iteration index."""
        base = os.path.basename(mesh_path)
        mesh_name = base.split(".")[0]
        mesh_extension = base.split(".")[1]
        i_iter = texture_n_iter - 1
        return os.path.join(str(model_path), f"{mesh_name}_texture_refined_{i_iter}.{mesh_extension}")

    def build_train_cmd(self, model_path, scene_dir) -> list[str]:
        """Argv for the training stage, with any passthrough flags appended."""
        c = self.config
        # Exposure compensation fits one exposure per training camera, and a held-out camera
        # has none, so an evaluation run has to train without it to stay comparable.
        exposure_flag = "--no-exposure_compensation" if c.eval else "--exposure_compensation"
        return [
            sys.executable, str(TRAIN_SCRIPT),
            "--rasterizer", "ours",
            "-s", str(scene_dir),
            "-m", str(model_path),
            "--feature_dc_lr", str(c.feature_dc_lr),
            "--feature_rest_lr", str(c.feature_rest_lr),
            "--position_lr_init", str(c.position_lr_init),
            "--position_lr_final", str(c.position_lr_final),
            "--position_lr_delay_mult", str(c.position_lr_delay_mult),
            "--position_lr_max_steps", str(c.position_lr_max_steps),
            "--opacity_lr", str(c.opacity_lr),
            "--scaling_lr", str(c.scaling_lr),
            "--rotation_lr", str(c.rotation_lr),
            "--appearance_embeddings_lr", str(c.appearance_embeddings_lr),
            "--appearance_network_lr", str(c.appearance_network_lr),
            "--gaussian_features_lr", str(c.gaussian_features_lr),
            "--pgsr_appearance_lr", str(c.pgsr_appearance_lr),
            exposure_flag,
            "--data_device", "cpu",
            "--iterations", str(c.iterations),
            "--sh_degree", str(c.sh_degree),
            "--N_max_gaussians", str(c.max_gaussians),
            "--densify_until_iter", str(c.densify_until_iter),
            "--densify_grad_threshold", str(c.densify_grad_threshold),
            "--lambda_depth_normal", str(c.lambda_depth_normal),
            "--multiview_factor", str(c.multiview_factor),
            *(["--eval"] if c.eval else []),
            *(["-r", str(c.resolution)] if c.resolution is not None else []),
            *self.passthrough,
        ]

    def build_render_cmd(self, model_path, scene_dir, extract_iteration) -> list[str]:
        """Argv for the held-out render stage, which only an evaluation run needs."""
        c = self.config
        cmd = [
            sys.executable, str(RENDER_SCRIPT),
            "--rasterizer", "ours",
            "-s", str(scene_dir),
            "-m", str(model_path),
            "--iteration", str(extract_iteration),
            "--eval",
            "--skip_train",
        ]
        if c.resolution is not None:
            cmd += ["-r", str(c.resolution)]
        return cmd

    def build_extract_cmd(self, model_path, scene_dir, extract_iteration) -> list[str]:
        """Argv for the pivot-based mesh extraction stage."""
        c = self.config
        cmd = [
            sys.executable, str(EXTRACT_SCRIPT),
            "--sdf_mode", "ours",
            "--rasterizer", "ours",
            "--dtype", "int32",
            "-s", str(scene_dir),
            "-m", str(model_path),
            "--n_pivots", str(c.n_pivots),
            "--std_factor", str(c.std_factor),
            "--n_binary_steps", str(c.n_binary_steps),
            "--isosurface_value", str(c.isosurface_value),
            "--iteration", str(extract_iteration),
            "--use_valid_mask",
            "--data_device", "cpu",
        ]
        if c.use_searched_pivots:
            cmd.append("--use_searched_pivots")
        if c.use_smallest_axis_as_normal:
            cmd.append("--use_smallest_axis_as_normal")
        if c.postprocess:
            cmd.append("--postprocess")
        if c.filter_large_edges:
            cmd.append("--filter_large_edges")
        if c.resolution is not None:
            cmd += ["-r", str(c.resolution)]
        return cmd

    def build_texture_cmd(self, model_path, scene_dir, mesh_path, extract_iteration) -> list[str]:
        """Argv for the texture refinement stage."""
        c = self.config
        cmd = [
            sys.executable, str(TEXTURE_SCRIPT),
            "--rasterizer", "ours",
            "-s", str(scene_dir),
            "-m", str(model_path),
            "--mesh", mesh_path,
            "--iteration", str(extract_iteration),
            "--n_iter", str(c.texture_n_iter),
            "--lambda_dssim", str(c.texture_lambda_dssim),
            "--lr", str(c.texture_lr),
            "--sh_degree_for_texturing", str(c.texture_sh_degree),
        ]
        if c.resolution is not None:
            cmd += ["-r", str(c.resolution)]
        return cmd

    def stages(self, scene: Scene, output_dir: Path) -> list[Stage]:
        """Build the train, extract, and texture invocations, plus a render stage when evaluating."""
        scene_dir = scene.root
        it = self.extract_iteration
        mesh_path = self.get_mesh_path(output_dir, self.config.n_pivots, self.config.postprocess)
        stages = [Stage("Training", self.build_train_cmd(output_dir, scene_dir))]
        # Rendering the held-out views takes seconds and needs only the trained model, while
        # mesh extraction is the fragile stage. Running it first keeps the metrics even when
        # extraction fails.
        if self.config.eval:
            stages.append(
                Stage("Held-out rendering", self.build_render_cmd(output_dir, scene_dir, it)))
        stages += [
            Stage("Mesh extraction", self.build_extract_cmd(output_dir, scene_dir, it)),
            Stage("Texture refinement",
                  self.build_texture_cmd(output_dir, scene_dir, mesh_path, it)),
        ]
        return stages

    def mesh_path(self, output_dir: Path) -> Path:
        """The extracted (pre-texture) mesh path."""
        return Path(self.get_mesh_path(output_dir, self.config.n_pivots, self.config.postprocess))

    def textured_mesh_path(self, output_dir: Path) -> Path:
        """The texture-refined mesh path."""
        return Path(self.get_textured_mesh_path(
            output_dir, self.mesh_path(output_dir), self.config.texture_n_iter))

    def header(self, scene: Scene, output_dir: Path) -> dict[str, object]:
        """Report the extraction iteration and both mesh paths up front."""
        header = {
            "Scene": scene.root,
            "Output": output_dir,
            "Extract iteration": self.extract_iteration,
            "Extracted mesh": self.mesh_path(output_dir),
            "Textured mesh": self.textured_mesh_path(output_dir),
        }
        if self.passthrough:
            header["Forwarding extra train.py args"] = " ".join(self.passthrough)
        return header

    def footer(self, output_dir: Path) -> dict[str, object]:
        """Report both mesh paths in the closing summary."""
        return {
            "Output": output_dir,
            "Extracted mesh": self.mesh_path(output_dir),
            "Textured mesh": self.textured_mesh_path(output_dir),
        }
