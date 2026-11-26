"""The neus2 adapter: argv, validation, finalize, and the vendored tree's local patches."""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

from augenblick.core.errors import SceneError
from augenblick.core.scene import Scene
from augenblick.reconstruction import neus2
from augenblick.reconstruction.base import LIBS_DIR, REPO_ROOT
from augenblick.reconstruction.neus2 import Neus2Backend, Neus2Config

LIB = LIBS_DIR / "neus2"
SCENE = Scene(Path("/scene"))
OUT = Path("/out")


@pytest.fixture
def pyngp_present(monkeypatch):
    """Pretend pyngp is built, so validate() reaches the scene checks on any env."""
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: object() if name == "pyngp" else real(name, *a))


def test_neus2_argv():
    prepare, train, mesh = Neus2Backend(Neus2Config()).stages(SCENE, OUT)
    py = sys.executable
    assert prepare.cmd == [py, str(LIB / "prepare.py"), "-s", "/scene", "-m", "/out", "-r", "-1",
                           "--bound_percentile", "1.0", "--bound_margin", "1.1"]
    assert train.cmd == [py, str(LIB / "train.py"), "-m", "/out", "--n_steps", "15000",
                         "--ek_loss_weight", "0.01", "--mask_loss_weight", "0.0"]
    assert mesh.cmd == [py, str(LIB / "extract_mesh.py"), "-m", "/out", "--n_steps", "15000",
                        "--mc_res", "1024", "--num_cluster", "1"]


def test_neus2_eval_adds_render_stage():
    stages = Neus2Backend(Neus2Config(eval=True, evaluation_resolution=2, n_steps=2000)).stages(SCENE, OUT)
    assert [s.name for s in stages] == [
        "NeuS2 staging", "NeuS2 training", "Mesh extraction", "Held-out rendering"]
    assert stages[0].cmd[-1] == "--eval"
    assert stages[-1].cmd == [sys.executable, str(LIB / "render.py"), "-s", "/scene", "-m", "/out",
                              "--n_steps", "2000", "--evaluation_resolution", "2"]
    assert all("--evaluation_resolution" not in s.cmd for s in stages[:-1])


def test_neus2_module_surface():
    assert neus2.__all__ == ["Neus2Backend", "Neus2Config"]
    defined = {name for name, value in vars(neus2).items()
               if not name.startswith("_") and getattr(value, "__module__", None) == neus2.__name__}
    assert defined == {"Neus2Backend", "Neus2Config"}


def test_neus2_runs_from_vendored_tree():
    assert Neus2Backend.backend_dir == REPO_ROOT / "src" / "libs" / "neus2"
    assert Neus2Backend.use_cwd
    assert (Neus2Backend.backend_dir / "utils" / "mlp_weights.txt").is_file()
    assert Neus2Backend(Neus2Config()).mesh_path(OUT) == OUT / "mesh" / "neus2_post.ply"


def test_neus2_requires_pyngp(ring_scene, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None)
    with pytest.raises(SceneError, match="pyngp"):
        Neus2Backend(Neus2Config()).validate(Scene(ring_scene(n_views=4)))


def test_neus2_requires_every_mask(ring_scene, pyngp_present):
    root = ring_scene(n_views=4)
    (root / "masks" / "v02.png").unlink()
    with pytest.raises(SceneError, match="1 of 4 registered images lack masks"):
        Neus2Backend(Neus2Config()).validate(Scene(root))


def test_neus2_ignores_unregistered_images(ring_scene, pyngp_present):
    root = ring_scene(n_views=4)
    # An image COLMAP never registered is never staged, so it needs no mask.
    (root / "images" / "stray.jpg").write_bytes((root / "images" / "v01.jpg").read_bytes())
    Neus2Backend(Neus2Config()).validate(Scene(root))


def test_neus2_rejects_non_pinhole(ring_scene, pyngp_present):
    with pytest.raises(SceneError, match="pinhole"):
        Neus2Backend(Neus2Config()).validate(Scene(ring_scene(n_views=4, model="SIMPLE_RADIAL")))


def test_neus2_rejects_bad_evaluation_resolution(ring_scene, pyngp_present):
    with pytest.raises(SceneError, match="evaluation_resolution"):
        Neus2Backend(Neus2Config(eval=True, evaluation_resolution=0)).validate(Scene(ring_scene(n_views=4)))


@pytest.mark.parametrize("keep", [False, True])
def test_neus2_finalize_cleans_staging(ring_scene, tmp_path, keep):
    out = tmp_path / "out"
    staged = out / "neus2" / "scene" / "images"
    staged.mkdir(parents=True)
    (staged / "v01.png").write_bytes(b"")
    (out / "neus2" / "scene" / "transforms_train.json").write_text("{}")
    Neus2Backend(Neus2Config(keep_staged=keep)).finalize(Scene(ring_scene(n_views=4)), out)
    assert (out / "cameras.json").is_file()
    assert staged.is_dir() == keep
    assert (out / "neus2" / "scene" / "transforms_train.json").is_file()


def test_neus2_patches_present():
    """Guard against a wholesale resync from upstream dropping the local patches."""
    root = (LIB / "CMakeLists.txt").read_text()
    tcnn = (LIB / "dependencies" / "neus2_tcnn" / "CMakeLists.txt").read_text()
    for text in (root, tcnn):
        assert "set(CMAKE_CXX_STANDARD 17)" in text and "set(CMAKE_CUDA_STANDARD 17)" in text
    assert 'if (NGP_BUILD_WITH_GUI AND NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/glfw' in root
    assert "set(PYBIND11_FINDPYTHON ON)" in root
    assert "dependencies/cutlass/include/cutlass/cutlass.h" in tcnn
    assert "CUDA_TOOLKIT_ROOT_DIR /opt/cuda" not in tcnn and "find_library(" not in tcnn
    common = (LIB / "dependencies" / "pybind11" / "include" / "pybind11" / "detail" / "common.h").read_text()
    assert int(re.search(r"#define PYBIND11_VERSION_MINOR (\d+)", common).group(1)) >= 12
    headers = LIB / "include" / "neural-graphics-primitives"
    for name in ("nerf_network.h", "takikawa_encoding.cuh"):
        assert not re.search(r"^\s*parallel_for_gpu\(", (headers / name).read_text(), re.M)
    for name in ("prepare.py", "train.py", "extract_mesh.py", "render.py", "scene_io.py", "ngp_session.py"):
        assert "LOCAL PATCH (augenblick)" in (LIB / name).read_text().splitlines()[0]


def test_snakefile_mesh_path_matches_backend():
    snakefile = (REPO_ROOT / "workflow" / "Snakefile").read_text()
    path = re.search(r'"neus2": "([^"]+)"', snakefile).group(1)
    assert OUT / path == Neus2Backend(Neus2Config()).mesh_path(OUT)
    assert 'wildcards.backend in {"geosvr", "neus2"}' in snakefile
