"""MILo: Gaussian splatting with a Delaunay mesh extracted inside the training loop."""
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal, Optional

from augenblick.core.errors import SceneError
from augenblick.core.registry import register_reconstruction
from augenblick.core.scene import Scene
from augenblick.reconstruction.base import (
    LIBS_DIR, EvalParams, ResolutionParams, Stage, SubprocessBackend, require_registered_masks)

logger = logging.getLogger(__name__)

# Upstream's code lives one level down; src/libs/milo/submodules/ holds its two rasterisers.
MILO_DIR = LIBS_DIR / "milo" / "milo"
TRAIN_FAST_SCRIPT = MILO_DIR / "train.py"
TRAIN_REGULAR_SCRIPT = MILO_DIR / "train_regular_densification.py"
RENDER_SCRIPT = MILO_DIR / "render.py"
SDF_SCRIPT = MILO_DIR / "mesh_extract_sdf.py"
TSDF_SCRIPT = MILO_DIR / "mesh_extract_regular_tsdf.py"

# Each recipe's own iteration count and mesh config, as upstream runs them: regular is the
# object-centric (DTU) recipe, fast the paper's Mini-Splatting2 one (configs/fast).
RECIPE_DEFAULTS = {
    "regular": {"iterations": 30_000, "mesh_config": "default_dtu"},
    "fast": {"iterations": 18_000, "mesh_config": "default"},
}
# fast's schedule as train.py resolves it (configs/fast and the argparse default): densification
# stops exactly where the depth-normal loss starts.
FAST_DENSIFY_UNTIL_ITER = 3_000
FAST_REGULARIZATION_FROM_ITER = 3_000


def tetranerf_installed() -> bool:
    """Whether GW's tetra_triangulation build, which MILo's Delaunay step uses, is importable."""
    return importlib.util.find_spec("tetranerf") is not None


@dataclass(frozen=True)
class MiloConfig(EvalParams, ResolutionParams):
    """Training and mesh-extraction parameters forwarded to MILo."""

    densification: Literal["regular", "fast"] = field(default="regular", metadata={
        "help": "regular: RaDe-GS densification, upstream's object-centric recipe; "
                "fast: Mini-Splatting2, the paper's scene recipe"})
    iterations: Optional[int] = field(default=None, metadata={
        "help": "Training iterations; defaults to the recipe's (30000 regular, 18000 fast)"})
    densify_until_iter: Optional[int] = field(default=None, metadata={
        "help": "Iteration to stop densifying; defaults to the recipe's"})
    regularization_from_iter: Optional[int] = field(default=None, metadata={
        "help": "Iteration the depth-normal loss starts; defaults to the recipe's"})
    mesh_config: Optional[str] = field(default=None, metadata={
        "help": "Mesh-in-the-loop config: a configs/mesh/ name or a .yaml path, used by training "
                "and extraction alike; defaults to default_dtu (regular) or default (fast)"})
    rasterizer: Literal["radegs", "gof"] = field(default="radegs", metadata={
        "help": "Gaussian rasteriser; the mask loss needs radegs, the only one returning alpha"})
    imp_metric: Literal["indoor", "outdoor"] = field(default="indoor", metadata={
        "help": "Importance sampling for pivot selection; upstream uses indoor for objects"})
    dense_gaussians: bool = field(default=False, metadata={
        "help": "Train more Gaussians and draw Delaunay pivots from a subset"})
    decoupled_appearance: bool = field(default=False, metadata={
        "help": "Per-view appearance network (GOF); off for parity with the other backends"})
    lambda_mask: float = field(default=0.1, metadata={
        "help": "Rendered-alpha vs mask loss weight, on for parity with 2DGS/GW; 0 disables"})
    white_background: bool = field(default=False, metadata={"help": "Use white background"})
    extract: Literal["sdf", "tsdf"] = field(default="tsdf", metadata={
        "help": "sdf: MILo's refined Delaunay mesh; tsdf: regular-grid TSDF of rendered depth"})
    refine_iter: int = field(default=1000, metadata={
        "help": "sdf: SDF refinement iterations before the final marching tetrahedra"})
    remove_oof_vertices: bool = field(default=False, metadata={
        "help": "sdf: drop vertices no training camera sees"})
    # The depth range spans camera distance, not the specimen, so upstream's 1024 left ~150 voxels
    # across a turntable specimen (98k faces); 4096 gives a mesh as fine as the other backends'.
    mesh_res: int = field(default=4096, metadata={"help": "tsdf: voxels across the depth range"})
    num_cluster: int = field(default=50, metadata={
        "help": "tsdf: connected components to keep in the post-processed mesh"})


@register_reconstruction
class MiloBackend(SubprocessBackend):
    """Runs MILo training, optional held-out rendering, then mesh extraction."""

    name: ClassVar[str] = "milo"
    config_cls: ClassVar[type] = MiloConfig
    # cwd matters: train.py reads its --config_path default, ./configs/fast, relative to it.
    backend_dir: ClassVar[Path] = MILO_DIR
    title: ClassVar[str] = "MILo Reconstruction Pipeline"

    @property
    def iterations(self) -> int:
        """Training iterations, which extraction must load: the field, else the recipe's."""
        return self.config.iterations or RECIPE_DEFAULTS[self.config.densification]["iterations"]

    @property
    def mesh_config(self) -> str:
        """Mesh config shared by training and extraction, which must agree."""
        return self.config.mesh_config or RECIPE_DEFAULTS[self.config.densification]["mesh_config"]

    def validate(self, scene: Scene) -> None:
        """Also check the tetra extension, the rasteriser/mask-loss pairing and the masks.

        Raises:
            SceneError: If tetranerf is not installed, or the mask loss is on and a registered
                image has no mask.
            ValueError: If the mask loss is on with the GOF rasteriser, a mesh config path does
                not exist, or a fast schedule regularises before densification ends.
        """
        super().validate(scene)
        # Upstream only warns when tetranerf is missing, then dies at the first Delaunay
        # triangulation, hours into training.
        if not tetranerf_installed():
            raise SceneError(
                "MILo needs the tetranerf extension (GW's tetra_triangulation build), which is "
                "not installed; rebuild without SKIP_TETRA=1")
        c = self.config
        if c.lambda_mask > 0 and c.rasterizer == "gof":
            raise ValueError(
                "--lambda_mask needs --rasterizer radegs: GOF renders no alpha to compare with "
                "the mask; pass --lambda_mask 0 to use gof")
        if self.mesh_config.endswith(".yaml") and not Path(self.mesh_config).is_file():
            raise ValueError(f"--mesh_config {self.mesh_config} does not exist")
        if c.densification == "fast":
            # From regularization_from_iter, train.py renders with RaDe-GS, whose output lacks
            # the area_max that Mini-Splatting2 densification reads until densify_until_iter:
            # an overlap dies with KeyError: 'area_max' at the first regularised iteration.
            densify_until = c.densify_until_iter or FAST_DENSIFY_UNTIL_ITER
            reg_from = c.regularization_from_iter or FAST_REGULARIZATION_FROM_ITER
            if reg_from < densify_until:
                raise ValueError(
                    f"--densification fast needs --regularization_from_iter ({reg_from}) >= "
                    f"--densify_until_iter ({densify_until}): densification reads a buffer "
                    "only the pre-regularisation renderer returns")
        require_registered_masks(scene, c.lambda_mask)

    def build_train_cmd(self, scene: Scene, output_dir: Path) -> list[str]:
        """Argv for training with the selected recipe."""
        c = self.config
        script = TRAIN_REGULAR_SCRIPT if c.densification == "regular" else TRAIN_FAST_SCRIPT
        cmd = [
            sys.executable, str(script),
            "-s", str(scene.root),
            "-m", str(output_dir),
            "--imp_metric", c.imp_metric,
            "--rasterizer", c.rasterizer,
            "--mesh_config", self.mesh_config,
            "--iterations", str(self.iterations),
        ]
        if c.densify_until_iter is not None:
            cmd += ["--densify_until_iter", str(c.densify_until_iter)]
        if c.regularization_from_iter is not None:
            cmd += ["--regularization_from_iter", str(c.regularization_from_iter)]
        cmd += [
            "--lambda_mask", str(c.lambda_mask),
            "--data_device", "cpu",
            # -1 disables the end-of-training test report, which downloads LPIPS weights;
            # held-out scoring is augenblick.eval.nvs's job. No resume checkpoints either.
            "--test_iterations", "-1",
            "--checkpoint_iterations", "-1",
            "-r", str(c.resolution),
        ]
        for flag in ("eval", "white_background", "dense_gaussians", "decoupled_appearance"):
            if getattr(c, flag):
                cmd.append(f"--{flag}")
        return cmd

    def build_render_cmd(self, output_dir: Path) -> list[str]:
        """Argv for rendering the held-out views; source, split and resolution come from cfg_args."""
        return [
            sys.executable, str(RENDER_SCRIPT),
            "-m", str(output_dir),
            "--rasterizer", self.config.rasterizer,
            "--skip_train",
        ]

    def build_mesh_cmd(self, output_dir: Path) -> list[str]:
        """Argv for the selected mesh extractor."""
        c = self.config
        if c.extract == "tsdf":
            return [
                sys.executable, str(TSDF_SCRIPT),
                "-m", str(output_dir),
                "--rasterizer", c.rasterizer,
                "--mesh_res", str(c.mesh_res),
                "--num_cluster", str(c.num_cluster),
            ]
        cmd = [
            sys.executable, str(SDF_SCRIPT),
            "-m", str(output_dir),
            # Upstream defaults to iteration 18000 and the default mesh config, which match only
            # the fast recipe; both must be what training used.
            "--iteration", str(self.iterations),
            "--rasterizer", c.rasterizer,
            "--config", self.mesh_config,
            "--imp_metric", c.imp_metric,
            "--refine_iter", str(c.refine_iter),
        ]
        if c.remove_oof_vertices:
            cmd.append("--remove_oof_vertices")
        return cmd

    def stages(self, scene: Scene, output_dir: Path) -> list[Stage]:
        """Build the train, optional held-out render, and mesh-extraction invocations."""
        stages = [Stage("MILo training", self.build_train_cmd(scene, output_dir))]
        # Rendering takes seconds and needs only the trained model, while extraction is the
        # fragile stage; running it first keeps the metrics even when extraction fails.
        if self.config.eval:
            stages.append(Stage("Held-out rendering", self.build_render_cmd(output_dir)))
        stages.append(Stage("Mesh extraction", self.build_mesh_cmd(output_dir)))
        return stages

    def mesh_path(self, output_dir: Path) -> Path:
        """Where the selected extractor writes its final mesh."""
        if self.config.extract == "tsdf":
            return output_dir / f"mesh_regular_tsdf_res{self.config.mesh_res}_post.ply"
        return output_dir / "mesh_learnable_sdf.ply"

    def header(self, scene: Scene, output_dir: Path) -> dict[str, object]:
        """Report the recipe, mesh config and extractor up front."""
        return {
            "Scene": scene.root,
            "Output": output_dir,
            "Recipe": f"{self.config.densification} ({self.iterations} iterations)",
            "Mesh config": self.mesh_config,
            "Extractor": self.config.extract,
        }
