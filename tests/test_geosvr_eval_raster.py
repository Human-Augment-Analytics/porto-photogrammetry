"""Exercise actual native camera adapter without importing its GPU renderer."""
import ast
from pathlib import Path
from PIL import Image


def test_native_canonical_adapter_preserves_training_camera(tmp_path):
    path = Path(__file__).parents[1] / "src/libs/geosvr/render.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "canonical_test_view")
    ns = {};exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    class View:
        image_name = "held_out"
        image_width, image_height = 1600, 1067
        fovx, fovy, cx_p, cy_p = 1.2, .9, .51, .49
        def clone_mini(self):
            return View()
    photo = tmp_path / "held_out.jpg";Image.new("RGB", (6240, 4160), "white").save(photo)
    native = View();camera, truth = ns["canonical_test_view"](native, photo, -1)
    assert camera is not native
    assert (camera.image_width, camera.image_height) == truth.size == (1600, 1066)
    assert (native.image_width, native.image_height) == (1600, 1067)
    assert (camera.fovx, camera.fovy, camera.cx_p, camera.cy_p) == (native.fovx, native.fovy, native.cx_p, native.cy_p)


def test_geosvr_eval_stage_requests_direct_raster(tmp_path):
    from types import SimpleNamespace
    from augenblick.reconstruction.geosvr import GeosvrBackend, GeosvrConfig
    backend = GeosvrBackend(GeosvrConfig(eval=True, resolution=2))
    stages = backend.stages(SimpleNamespace(root=tmp_path), tmp_path / "out")
    render = next(stage.cmd for stage in stages if stage.name == "Rendering")
    assert render[render.index("--canonical_test_resolution") + 1] == "-1"
    assert stages[0].cmd[stages[0].cmd.index("--res_downscale") + 1] == "2.0"
    train = stages[0].cmd
    assert "--canonical_test_resolution" not in train


def test_geo_custom_evaluation_resolution_is_independent(tmp_path):
    from types import SimpleNamespace
    from augenblick.reconstruction.geosvr import GeosvrBackend, GeosvrConfig
    backend = GeosvrBackend(GeosvrConfig(eval=True, resolution=2, evaluation_resolution=3))
    stages = backend.stages(SimpleNamespace(root=tmp_path), tmp_path / "out")
    assert stages[0].cmd[stages[0].cmd.index("--res_downscale") + 1] == "2.0"
    assert stages[1].cmd[stages[1].cmd.index("--canonical_test_resolution") + 1] == "3"
