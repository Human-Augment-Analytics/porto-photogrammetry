"""GeoSVR: sparse-voxel training, then TSDF fusion of rendered depths into a mesh."""
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

from augenblick.core.errors import SceneError
from augenblick.core.registry import register_reconstruction
from augenblick.core.scene import Scene
from augenblick.eval.split import registered_stems
from augenblick.reconstruction.base import (
    LIBS_DIR, EvalParams, ResolutionParams, Stage, SubprocessBackend)

logger = logging.getLogger(__name__)

GEOSVR_DIR = LIBS_DIR / "geosvr"
TRAIN_SCRIPT = GEOSVR_DIR / "train.py"
RENDER_SCRIPT = GEOSVR_DIR / "render.py"
TSDF_SCRIPT = GEOSVR_DIR / "mesh_extract" / "tsdf_mesh.py"
TSDF_360_SCRIPT = GEOSVR_DIR / "mesh_extract" / "tsdf_mesh_360.py"
CFG_DIR = GEOSVR_DIR / "cfg"

# The long-side cap the other backends apply at -r -1.
MAX_LONG_SIDE = 1600


@dataclass(frozen=True)
class GeosvrConfig(EvalParams, ResolutionParams):
    """Training and mesh-extraction parameters forwarded to GeoSVR."""

    preset: Literal["dtu", "tnt", "none"] = field(default="dtu", metadata={
        "help": "Upstream cfg/<preset>_mesh.yaml loaded before the flags; dtu suits masked objects"})
    sche_mult: float = field(default=1.0, metadata={
        "help": "Schedule multiplier; 1.0 trains 20k iterations"})
    lambda_depthanythingv2: float = field(default=0.1, metadata={
        "help": "Depth-Anything-V2 prior weight; 0 disables the prior and the mono-normal loss built on it"})
    da2_uncertainty: bool = field(default=True, metadata={
        "help": "Weight the depth prior by voxel uncertainty"})
    power_level_uncertainty: float = field(default=1.0, metadata={
        "help": "Exponent on the voxel-level uncertainty weight; upstream suggests 0.2 for NVS"})
    voxel_dropout_min: float = field(default=0.5, metadata={
        "help": "After iter 1000, draw the render supersampling factor from U(this, ss); 1 disables"})
    multi_view_ncc_weight: float = field(default=0.05, metadata={
        "help": "Multi-view NCC patch-matching weight"})
    multi_view_geo_weight: float = field(default=0.01, metadata={
        "help": "Multi-view geometric consistency weight"})
    lambda_rectify: float = field(default=1e-5, metadata={
        "help": "Surface rectification regulariser weight"})
    lambda_scaling_penalty: float = field(default=1e-6, metadata={
        "help": "Voxel scaling penalty weight"})
    lambda_mask: float = field(default=0.5, metadata={
        "help": "Rendered-alpha vs mask loss weight, on for parity with 2DGS/GW; 0 disables"})
    white_background: bool = field(default=False, metadata={"help": "Use white background"})
    tsdf: Literal["auto", "fixed"] = field(default="fixed", metadata={
        "help": "auto sizes the TSDF from the cameras; fixed uses the absolute flags below"})
    radius_scale: float = field(default=2.0, metadata={
        "help": "auto TSDF: max depth as a multiple of the camera-to-focus distance"})
    voxel_size: float = field(default=0.001, metadata={"help": "fixed TSDF: voxel size"})
    max_depth: float = field(default=5.0, metadata={"help": "fixed TSDF: max depth"})
    sdf_trunc_scale: float = field(default=4.0, metadata={
        "help": "fixed TSDF: truncation as a multiple of the voxel size"})
    num_cluster: int = field(default=1, metadata={"help": "Connected components to keep in mesh"})
    use_depth_filter: bool = field(default=False, metadata={
        "help": "Drop grazing-angle depths before TSDF fusion"})


def _flag(name: str, on: bool) -> str:
    """Spell a bool explicitly, so the preset yaml never decides it."""
    return f"--{name}" if on else f"--no-{name}"


@register_reconstruction
class GeosvrBackend(SubprocessBackend):
    """Runs GeoSVR train.py, render.py, then TSDF mesh extraction from the rendered views."""

    name: ClassVar[str] = "geosvr"
    config_cls: ClassVar[type] = GeosvrConfig
    backend_dir: ClassVar[Path] = GEOSVR_DIR
    title: ClassVar[str] = "GeoSVR Reconstruction Pipeline"

    def validate(self, scene: Scene) -> None:
        """Also require a mask for every registered image when the mask loss is on.

        Raises:
            SceneError: If lambda_mask > 0 and any registered image lacks masks/<stem>.png.
        """
        super().validate(scene)
        if self.config.lambda_mask <= 0:
            return
        # Upstream reads cam.mask unguarded, so one unmasked view would crash training mid-run.
        # Only registered images are loaded: masked COLMAP drops unmasked ones from the model.
        stems = registered_stems(scene.model_dir)
        missing = [s for s in stems if not (scene.masks_dir / f"{s}.png").is_file()]
        if missing:
            raise SceneError(
                f"lambda_mask={self.config.lambda_mask} needs a mask per registered image, but "
                f"{len(missing)} of {len(stems)} lack masks/<stem>.png (e.g. {missing[:5]}); "
                "pass --lambda_mask 0 to train without it")

    def res_downscale(self, scene: Scene) -> float:
        """Map -r onto GeoSVR's res_downscale, matching the other backends' long-side cap."""
        if self.config.resolution > 0:
            return float(self.config.resolution)
        # Imported here so `augenblick recon --list` stays light on a login node.
        from PIL import Image

        first = min(p for p in scene.images_dir.iterdir() if p.is_file())
        with Image.open(first) as image:
            long_side = max(image.size)
        # 0 leaves upstream's own 1024^2-pixel cap off only when no downscale is needed.
        return long_side / MAX_LONG_SIDE if long_side > MAX_LONG_SIDE else 0.0

    def stages(self, scene: Scene, output_dir: Path) -> list[Stage]:
        """Build the train, optional held-out render, and TSDF invocations."""
        c = self.config
        train_cmd = [
            sys.executable, str(TRAIN_SCRIPT),
            "-s", str(scene.root),
            "-m", str(output_dir),
        ]
        if c.preset != "none":
            train_cmd += ["--cfg_files", str(CFG_DIR / f"{c.preset}_mesh.yaml")]
        train_cmd += [
            "--sche_mult", str(c.sche_mult),
            "--lambda_depthanythingv2", str(c.lambda_depthanythingv2),
            _flag("enable_da2_uncertainty", c.da2_uncertainty),
            "--power_level_uncertainty", str(c.power_level_uncertainty),
            "--voxel_dropout_min", str(c.voxel_dropout_min),
            "--multi_view_ncc_weight", str(c.multi_view_ncc_weight),
            "--multi_view_geo_weight", str(c.multi_view_geo_weight),
            "--lambda_rectify", str(c.lambda_rectify),
            "--lambda_scaling_penalty", str(c.lambda_scaling_penalty),
            "--lambda_mask", str(c.lambda_mask),
            _flag("white_background", c.white_background),
            _flag("black_background", not c.white_background),
            # The DTU preset turns blending off because its scenes carry no masks; ours do.
            "--blend_mask",
            _flag("eval", c.eval),
            "--res_downscale", str(self.res_downscale(scene)),
        ]
        # The mono-normal loss reads the same priors, which are never computed when the prior is off.
        if c.lambda_depthanythingv2 == 0:
            train_cmd += ["--lambda_normal_dmono", "0.0"]
        if c.eval:
            train_cmd += ["--test_iterations", "-1"]

        if c.tsdf == "auto":
            mesh_cmd = [
                sys.executable, str(TSDF_360_SCRIPT),
                "-m", str(output_dir),
                "--radius_scale", str(c.radius_scale),
            ]
        else:
            mesh_cmd = [
                sys.executable, str(TSDF_SCRIPT),
                "-m", str(output_dir),
                "--voxel_size", str(c.voxel_size),
                "--max_depth", str(c.max_depth),
                "--sdf_trunc_scale", str(c.sdf_trunc_scale),
            ]
        mesh_cmd += ["--num_cluster", str(c.num_cluster)]
        if c.use_depth_filter:
            mesh_cmd.append("--use_depth_filter")

        # The TSDF scripts render depth themselves but read colour from render.py's train/ renders,
        # so the train split is always rendered; the held-out split only when there is one.
        render_cmd = [sys.executable, str(RENDER_SCRIPT), "-m", str(output_dir)]
        if not c.eval:
            render_cmd.append("--skip_test")

        return [
            Stage("GeoSVR training", train_cmd),
            Stage("Rendering", render_cmd),
            Stage("TSDF mesh extraction", mesh_cmd),
        ]

    def mesh_path(self, output_dir: Path) -> Path:
        """Both TSDF scripts write the post-processed mesh to the same place."""
        return output_dir / "mesh" / "tsdf" / "tsdf_fusion_post.ply"
