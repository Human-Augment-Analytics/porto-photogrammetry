"""Shared shape of the reconstruction backends: validate, prepare, run staged subprocesses."""
import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from augenblick.core import process
from augenblick.core.method import Method, StageResult
from augenblick.core.scene import Scene
from augenblick.core.timing import StageTimer
from augenblick.eval.split import write_split

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBS_DIR = REPO_ROOT / "src" / "libs"


@dataclass(frozen=True)
class EvalParams:
    """Held-out-view evaluation, inherited by every backend config that supports it.

    A base class rather than a duck-typed field name, so eval_enabled has a contract to
    check instead of probing for an attribute that may or may not exist.
    """

    eval: bool = field(default=False, metadata={
        "help": "Hold out views for novel-view evaluation, from the scene's split.json"})


@dataclass(frozen=True)
class ResolutionParams:
    """Input downscale, for the backends that forward -r to their training script."""

    resolution: int = field(default=-1, metadata={
        "short": "-r", "help": "Input downscale factor; -1 caps the long side at 1600 px"})


@dataclass
class Stage:
    """One backend invocation: a label and the argv to run for it.

    Args:
        name: Label used in the "Step i/N" log header.
        cmd: Full argv passed to the subprocess.
    """

    name: str
    cmd: list[str]


class ReconstructionMethod(Method):
    """Consumes a COLMAP scene, produces a mesh."""

    def validate(self, scene: Scene) -> None:
        """Require both images/ and a non-empty sparse/0/ model."""
        scene.require_images()
        scene.require_reconstruction()

    @abstractmethod
    def stages(self, scene: Scene, output_dir: Path) -> list[Stage]:
        """Build the ordered list of backend invocations for this run.

        Args:
            scene: The scene the backend will read, after prepare().
            output_dir: Directory the backend writes into.

        Returns:
            The stages to run, in order.
        """

    @abstractmethod
    def mesh_path(self, output_dir: Path) -> Path:
        """Return where this backend writes its final mesh for the given output dir."""

    @property
    def eval_enabled(self) -> bool:
        """Whether this run holds out views; backends opt in by inheriting EvalParams."""
        return isinstance(self.config, EvalParams) and self.config.eval

    def test_renders_root(self, output_dir: Path) -> Path | None:
        """Directory under which this backend writes its ours_<iter>/{renders,gt} exports.

        Every 3DGS-derived backend here writes <output>/test/ours_<iter>/, so the default
        suits all of them; a backend that diverges overrides this, or returns None to
        declare that it cannot produce held-out renders.
        """
        return output_dir / "test"

    def evaluate(self, scene: Scene, output_dir: Path) -> dict | None:
        """Score the held-out renders this run produced, masked to the specimen.

        Args:
            scene: The scene the backend actually read, after prepare().
            output_dir: Directory the backend wrote into.

        Returns:
            The metrics dictionary, or None when this backend exports no held-out renders.
        """
        test_root = self.test_renders_root(output_dir)
        if test_root is None:
            logger.warning(f"{self.name} exports no held-out renders; skipping evaluation")
            return None

        # Imported here because it pulls in torch, which the CPU-only test suite does not need.
        from augenblick.eval.nvs import find_test_dir, resolve_test_stems, score

        # No iteration is named: whichever checkpoint the render stage exported is the one
        # that exists, and the highest wins if a 7k export was left behind.
        test_dir = find_test_dir(test_root)
        return score(
            test_dir,
            scene.masks_dir,
            resolve_test_stems(scene),
            output_dir / "nvs_metrics.json",
            extra={"backend": self.name, "scene": str(scene.root), "model": str(output_dir)},
        )


class SubprocessBackend(ReconstructionMethod):
    """A reconstruction backend driven by running upstream scripts as subprocesses."""

    backend_dir: ClassVar[Path]
    use_cwd: ClassVar[bool] = True
    title: ClassVar[str]

    def prepare(self, scene: Scene, output_dir: Path) -> Scene:
        """Hook for backends needing a modified scene; default returns it unchanged."""
        return scene

    def header(self, scene: Scene, output_dir: Path) -> dict[str, object]:
        """Key/value lines logged under the banner; backends may extend this."""
        return {"Scene": scene.root, "Output": output_dir}

    def footer(self, output_dir: Path) -> dict[str, object]:
        """Key/value lines logged in the closing summary."""
        return {"Output": output_dir, "Mesh": self.mesh_path(output_dir)}

    def run(self, scene: Scene, output_dir: Path) -> StageResult:
        """Validate, prepare the scene, run each stage under a timer, then score any held-out views.

        Args:
            scene: Input COLMAP scene.
            output_dir: Directory the backend writes into.

        Returns:
            A StageResult carrying the total elapsed time, the mesh path, and the metrics
            when the run held views out.
        """
        self.validate(scene)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Written before prepare() so a backend that copies the scene carries the split with it,
        # which is what keeps the held-out set identical across backends.
        if self.eval_enabled:
            write_split(scene.root)
        prepared = self.prepare(scene, output_dir)
        stages = self.stages(prepared, output_dir)

        timer = StageTimer(self.title, len(stages), self.header(prepared, output_dir))
        cwd = str(self.backend_dir) if self.use_cwd else None
        for stage in stages:
            with timer.stage(stage.name):
                process.run(stage.cmd, cwd=cwd)
        timer.summary(self.footer(output_dir))

        details: dict[str, object] = {
            "mesh": self.mesh_path(output_dir),
            "stages": dict(timer.elapsed),
        }
        if self.eval_enabled:
            metrics = self.evaluate(prepared, output_dir)
            if metrics is not None:
                logger.info(f"Held-out views: PSNR {metrics['psnr']:.2f}  "
                            f"SSIM {metrics['ssim']:.4f}  LPIPS {metrics['lpips']:.4f}  "
                            f"({metrics['n_test']} views)")
                details["metrics"] = metrics

        return StageResult(output_dir=output_dir, elapsed=timer.total, details=details)
