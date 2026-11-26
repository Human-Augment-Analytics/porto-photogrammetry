"""LOCAL PATCH (augenblick): first-party file, not upstream NeuS2.

Opens a pyngp Testbed for the stage scripts beside it. pyngp is imported only inside
open_testbed(), so the scripts load (and their CPU tests run) without it.
"""
from pathlib import Path

from augenblick.core.errors import SceneError

# Geometric SDF init reads this relative to cwd and exit(1)s inside C++ if it is missing.
MLP_WEIGHTS = Path("utils") / "mlp_weights.txt"


def preflight() -> None:
    """Fail with a Python error before the C++ would exit without one (nerf_network.h:775-793)."""
    if not MLP_WEIGHTS.is_file():
        raise SceneError(
            f"{MLP_WEIGHTS} not found under cwd {Path.cwd()}; NeuS2 stages must run from "
            "src/libs/neus2")


def open_testbed(snapshot: Path | None = None):
    """A fresh NeRF-mode Testbed, optionally restored from a snapshot.

    Args:
        snapshot: A .msgpack from train.py. It carries the dataset metadata (poses,
            intrinsics, from_na, scale/offset), so no training images are reloaded.

    Returns:
        (testbed, pyngp module).
    """
    preflight()
    import pyngp as ngp

    testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
    if snapshot is not None:
        if not Path(snapshot).is_file():
            raise SceneError(f"no snapshot at {snapshot}; run train.py first")
        testbed.load_snapshot(str(snapshot))
    return testbed, ngp
