"""Resolution must stay consistent from backend argv to every SuGaR camera loader."""
import argparse
import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from augenblick.core.config import add_dataclass_arguments, config_from_namespace
from augenblick.core.scene import Scene
from augenblick.reconstruction.base import ResolutionParams
from augenblick.reconstruction.sugar import SugarBackend, SugarConfig
from libs.sugar.sugar_utils.camera_resolution import camera_resolution
from libs.sugar.sugar_utils.general_utils import PILtoTorch, str2bool


SUGAR_DIR = Path(__file__).resolve().parents[1] / "src" / "libs" / "sugar"


def load_definition(relative_path, name, namespace):
    """Execute a real definition without importing CUDA-dependent module globals."""
    path = SUGAR_DIR / relative_path
    tree = ast.parse(path.read_text())
    definition = next(node for node in tree.body if getattr(node, "name", None) == name)
    definition.decorator_list = []
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_square_size_default_is_safe():
    assert SugarConfig().square_size == 8
    assert SugarConfig(square_size=5).square_size == 5


@pytest.mark.parametrize("square_size", [-1, 0, 1, 4])
def test_square_size_rejects_degenerate_atlas(square_size):
    with pytest.raises(ValueError, match="square_size must be >= 5"):
        SugarConfig(square_size=square_size)


@pytest.mark.parametrize("flag", ["-r", "--resolution"])
def test_sugar_cli_resolution_and_regularization_are_independent(flag):
    parser = argparse.ArgumentParser()
    add_dataclass_arguments(parser, SugarConfig)
    config = config_from_namespace(SugarConfig, parser.parse_args([
        flag, "1000", "--regularization", "density",
    ]))
    assert config.resolution == 1000
    assert config.regularization == "density"
    assert config.square_size == 8
    invalid = parser.parse_args(["--square_size", "4"])
    with pytest.raises(ValueError, match="square_size must be >= 5"):
        config_from_namespace(SugarConfig, invalid)


@pytest.mark.parametrize("width,height,resolution,expected", [
    (4000, 3000, -1, (1600, 1200)),
    (1200, 2400, -1, (1200, 2400)),
    (2000, 4000, -1, (1600, 3200)),
    (4000, 3000, 1, (4000, 3000)),
    (4001, 3003, 2, (2000, 1502)),
    (4001, 3003, 4, (1000, 751)),
    (4000, 3000, 8, (500, 375)),
    (1200, 2401, 1000, (1000, 2000)),
    (1200, 2400, 2400, (2400, 4800)),
    (4001, 3003, 1920, (1920, 1441)),
    (1, 1, 8, (1, 1)),
])
def test_camera_resolution(width, height, resolution, expected):
    assert camera_resolution(width, height, resolution) == expected


@pytest.mark.parametrize("width,height,resolution", [
    (0, 100, -1), (100, 0, -1), (100, 100, 0), (100, 100, -2),
])
def test_camera_resolution_rejects_invalid_inputs(width, height, resolution):
    with pytest.raises(ValueError):
        camera_resolution(width, height, resolution)


@pytest.mark.parametrize("resolution", [-1, 1, 2, 4, 8, 1000, 2400])
def test_backend_forwards_resolution_without_changing_regularization(resolution):
    config = SugarConfig(resolution=resolution, eval=True, regularization="sdf")
    assert isinstance(config, ResolutionParams)
    vanilla, sugar, render = SugarBackend(config).stages(Scene(Path("/scene")), Path("/out"))
    for stage in (vanilla, sugar):
        assert stage.cmd[stage.cmd.index("--resolution") + 1] == str(resolution)
    assert sugar.cmd[sugar.cmd.index("-r") + 1] == "sdf"
    assert render.cmd[render.cmd.index("--manifest") + 1] == "/out/sugar/run_manifest.json"


@pytest.mark.parametrize("resolution", [-1, 1, 2, 4, 8, 1000, 2400])
@pytest.mark.parametrize("width,height", [(4001, 3003), (1200, 2401), (2001, 4003)])
def test_resolution_matches_vendored_vanilla(width, height, resolution):
    load_camera = load_definition("gaussian_splatting/utils/camera_utils.py", "loadCam", {
        "WARNED": False,
        "Camera": SimpleNamespace,
        "PILtoTorch": lambda image, dimensions: dimensions,
    })
    camera_info = SimpleNamespace(
        image=SimpleNamespace(size=(width, height), split=lambda: [None] * 3),
        uid=0, R=None, T=None, FovX=0.5, FovY=0.5, image_name="view",
    )
    camera = load_camera(SimpleNamespace(resolution=resolution, data_device="cpu"),
                         0, camera_info, 1.0)
    assert camera_resolution(width, height, resolution) == camera.image


@pytest.fixture
def camera_scene(tmp_path):
    images = tmp_path / "scene" / "images"
    masks = tmp_path / "scene" / "masks"
    checkpoint = tmp_path / "checkpoint"
    images.mkdir(parents=True)
    masks.mkdir()
    checkpoint.mkdir()
    dimensions = [(2001, 1003), (801, 1603), (1001, 501)]
    transforms = []
    for index, (width, height) in enumerate(dimensions):
        name = f"view_{index}"
        Image.new("RGB", (width, height), (32, 64, 96)).save(images / f"{name}.png")
        Image.new("L", (width, height), 255).save(masks / f"{name}.png")
        transforms.append({
            "id": index, "img_name": name, "width": width, "height": height,
            "fx": width / 2, "fy": height / 2,
            "rotation": np.eye(3).tolist(), "position": [0, 0, 0],
        })
    (checkpoint / "cameras.json").write_text(json.dumps(transforms))
    return images.parent, checkpoint, dimensions


@pytest.mark.parametrize("resolution", [-1, 1, 2, 4, 8, 1000, 2400])
@pytest.mark.parametrize("load_gt_images", [True, False])
def test_loader_uses_each_cameras_dimensions(camera_scene, resolution, load_gt_images):
    source, checkpoint, dimensions = camera_scene
    load_cameras = load_definition("sugar_scene/cameras.py", "load_gs_cameras", {
        "os": os, "json": json, "np": np, "Image": Image,
        "focal2fov": lambda focal, pixels: 2 * np.arctan(pixels / (2 * focal)),
        "PILtoTorch": PILtoTorch, "camera_resolution": camera_resolution,
        "GSCamera": SimpleNamespace,
    })
    cameras = load_cameras(str(source), str(checkpoint), resolution=resolution,
                           load_gt_images=load_gt_images)
    for camera, (width, height) in zip(cameras, dimensions):
        target_width, target_height = camera_resolution(width, height, resolution)
        if load_gt_images:
            assert camera.image.shape == (3, target_height, target_width)
            assert camera.gt_alpha_mask.shape == (1, target_height, target_width)
        else:
            assert (camera.image_width, camera.image_height) == (target_width, target_height)
            assert camera.image is None
    assert len(cameras) == len(dimensions)


@pytest.mark.parametrize("resolution", [None, -1, 1, 2, 4, 8, 1000])
@pytest.mark.parametrize("regularization", ["sdf", "density", "dn_consistency"])
def test_combined_training_forwards_resolution_and_records_manifest(
        tmp_path, monkeypatch, resolution, regularization):
    stages = {
        "coarse_training_with_sdf_regularization": Mock(return_value="/coarse.pt"),
        "coarse_training_with_density_regularization": Mock(return_value="/coarse.pt"),
        "coarse_training_with_density_regularization_and_dn_consistency": Mock(
            return_value="/coarse.pt"),
        "extract_mesh_from_coarse_sugar": Mock(return_value=["/coarse.ply"]),
        "refined_training": Mock(return_value="/refined.pt"),
        "extract_mesh_and_texture_from_refined_sugar": Mock(return_value="/refined.obj"),
    }
    argv = ["train.py", "-s", "/scene", "-c", "/checkpoint", "-o", str(tmp_path),
            "-r", regularization, "--eval", "True"]
    if resolution is not None:
        argv += ["--resolution", str(resolution)]
    monkeypatch.setattr(sys, "argv", argv)
    path = SUGAR_DIR / "train.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    namespace = {"__name__": "__main__", "os": os, "json": json, "argparse": argparse,
                 "str2bool": str2bool, **stages}
    exec(compile(tree, str(path), "exec"), namespace)
    expected = resolution if resolution is not None else -1
    calls = [stage.call_args.args[0] for stage in stages.values() if stage.called]
    assert len(calls) == 4
    assert all(args.resolution == expected for args in calls)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["resolution"] == expected
    assert manifest["texture_square_size"] == 8
    assert manifest["eval"] is True


@pytest.mark.parametrize("relative_path", [
    "sugar_trainers/coarse_density.py",
    "sugar_trainers/coarse_density_and_dn_consistency.py",
    "sugar_trainers/coarse_sdf.py",
    "sugar_trainers/refine.py",
    "sugar_extractors/coarse_mesh.py",
    "sugar_extractors/refined_mesh.py",
])
@pytest.mark.parametrize("resolution", [None, -1, 2, 1000])
def test_every_training_and_extraction_wrapper_forwards_resolution(relative_path, resolution):
    tree = ast.parse((SUGAR_DIR / relative_path).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "GaussianSplattingWrapper"]
    assert calls
    args = SimpleNamespace() if resolution is None else SimpleNamespace(resolution=resolution)
    for call in calls:
        value = next(keyword.value for keyword in call.keywords if keyword.arg == "resolution")
        forwarded = eval(compile(ast.Expression(value), relative_path, "eval"), {"args": args})
        assert forwarded == (resolution if resolution is not None else -1)


@pytest.mark.parametrize("resolution", [-1, 2, 1000])
def test_gaussian_wrapper_forwards_resolution_to_loader(resolution):
    path = SUGAR_DIR / "sugar_scene/gs_model.py"
    tree = ast.parse(path.read_text())
    wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == "GaussianSplattingWrapper")
    constructor = next(node for node in wrapper.body if getattr(node, "name", None) == "__init__")
    load_cameras = Mock(side_effect=RuntimeError("stop at camera loading"))
    namespace = {"os": os, "ModelParams": SimpleNamespace, "PipelineParams": SimpleNamespace,
                 "OptimizationParams": SimpleNamespace, "load_gs_cameras": load_cameras}
    exec(compile(ast.Module(body=[constructor], type_ignores=[]), str(path), "exec"), namespace)
    instance = SimpleNamespace()
    with pytest.raises(RuntimeError, match="stop at camera loading"):
        namespace["__init__"](instance, "/scene", "/checkpoint", resolution=resolution)
    assert load_cameras.call_args.kwargs["resolution"] == resolution
    assert instance.model_params.resolution == resolution


@pytest.mark.parametrize("resolution", [None, -1, 2, 1000])
def test_held_out_render_loads_manifest_resolution(tmp_path, resolution):
    manifest = {"vanilla_checkpoint_path": "/checkpoint", "iteration_to_load": 7000}
    if resolution is not None:
        manifest["resolution"] = resolution
    manifest_path = tmp_path / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    wrapper = Mock(side_effect=RuntimeError("stop at camera loading"))
    render = load_definition("render_refined.py", "render_held_out", {
        "json": json, "torch": SimpleNamespace(cuda=SimpleNamespace(set_device=Mock())),
        "GaussianSplattingWrapper": wrapper,
    })
    with pytest.raises(RuntimeError, match="stop at camera loading"):
        render("/scene", str(manifest_path), str(tmp_path), 0, False)
    assert wrapper.call_args.kwargs["resolution"] == (resolution if resolution is not None else -1)