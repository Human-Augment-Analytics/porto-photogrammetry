import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from augenblick.core.errors import SceneError
from augenblick.core.scene import Scene
from augenblick.reconstruction.base import LIBS_DIR
from augenblick.reconstruction.neus2 import Neus2Backend, Neus2Config


@pytest.fixture
def scene_args(tmp_path, monkeypatch):
    scene = tmp_path / "source"
    model = scene / "sparse/0"
    model.mkdir(parents=True)
    (scene / "images").mkdir()
    (scene / "masks").mkdir()
    (model / "cameras.txt").write_text("1 SIMPLE_PINHOLE 8 6 8 4 3\n2 PINHOLE 8 6 12 10 3 2\n")
    names = ["a", "b", "c", "x", "y", "z"]
    records = []
    for image_id, name in enumerate(names, start=1):
        camera_id = 2 if name == "b" else 1
        observations = "1 1 1 2 2 2" if image_id <= 3 else "1 1 3"
        records.append(f"{image_id} 1 0 0 0 1 2 3 {camera_id} {name}.png\n{observations}\n")
        Image.new("RGB", (8, 6), (200, 100, 50)).save(scene / "images" / f"{name}.png")
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[:, :4] = 255
        Image.fromarray(mask).save(scene / "masks" / f"{name}.png")
    (model / "images.txt").write_text("".join(records))
    (model / "points3D.txt").write_text(
        "1 1 2 3 255 255 255 0.1 1 0 2 0 3 0\n"
        "2 3 4 5 255 255 255 0.1 1 1 2 1 3 1\n"
        "3 1000 1000 1000 255 255 255 0.1 4 0 5 0 6 0\n")
    (scene / "split.json").write_text(json.dumps({"train": names[:3], "test": names[3:]}))
    upstream = tmp_path / "upstream"
    (upstream / "configs/nerf").mkdir(parents=True)
    (upstream / "configs/nerf/dtu.json").write_text(json.dumps({
        "encoding": {"n_levels": 14}, "hyperparams": {"ek_loss_weight": 0.01, "mask_loss_weight": 0.0}}))
    monkeypatch.setattr("augenblick.reconstruction.neus2.subprocess.check_output",
                        lambda *arguments, **keywords: "synthetic-revision\n")
    backend = Neus2Backend(Neus2Config(upstream=upstream, eval=True, resolution=4,
                                     build_dir=str(tmp_path / "build")))
    return SimpleNamespace(scene=scene, output=tmp_path / "output", backend=backend)


def test_prepare_preserves_intrinsics_cv_poses_masks_and_split(scene_args):
    scene_args.backend.prepare_scene(Scene(scene_args.scene), scene_args.output)
    output = scene_args.output
    train = json.loads((output / "scene/transforms_train.json").read_text())
    test = json.loads((output / "scene/transforms_test.json").read_text())
    assert [frame["name"] for frame in train["frames"]] == ["a", "b", "c"]
    assert [frame["name"] for frame in test["frames"]] == ["x", "y", "z"]
    assert train["from_na"] is True
    intrinsic = np.asarray(train["frames"][1]["intrinsic_matrix"])
    np.testing.assert_allclose(intrinsic[:3, :3], [[6, 0, 1.5], [0, 5, 1], [0, 0, 1]])
    pose = np.asarray(train["frames"][0]["transform_matrix"])
    np.testing.assert_allclose(pose[:3, :3], np.eye(3))
    np.testing.assert_allclose(pose[:3, 3], [-1, -2, -3])
    with Image.open(output / "scene/images/a.png") as image:
        assert image.mode == "RGBA" and image.size == (4, 3)
        pixels = np.asarray(image)
        assert set(np.unique(pixels[:, :, 3])) == {0, 255}
        assert not pixels[:, 2:, :3].any()
    manifest = json.loads((output / "neus2_manifest.json").read_text())
    assert manifest["normalization_points"] == 2
    np.testing.assert_allclose(manifest["sphere_center"], [2, 3, 4])
    assert manifest["train_views"] == manifest["test_views"] == 3
    assert train["scale"] == test["scale"] == manifest["scale"]
    with pytest.raises(SceneError, match="overwrite"):
        scene_args.backend.prepare_scene(Scene(scene_args.scene), scene_args.output)


@pytest.mark.parametrize("invalid", ["empty", "size", "missing"])
def test_invalid_masks_fail_before_training(scene_args, invalid):
    mask = scene_args.scene / "masks/a.png"
    if invalid == "missing":
        mask.unlink()
    else:
        Image.new("L", (4, 3) if invalid == "size" else (8, 6), 0).save(mask)
    with pytest.raises(SceneError, match="mask|dimensions"):
        scene_args.backend.prepare_scene(Scene(scene_args.scene), scene_args.output)


def test_no_eval_trains_all_registered_views(scene_args):
    backend = Neus2Backend(replace(scene_args.backend.config, eval=False))
    backend.prepare_scene(Scene(scene_args.scene), scene_args.output)
    train = json.loads((scene_args.output / "scene/transforms_train.json").read_text())
    assert len(train["frames"]) == 6


def test_bounds_map_to_unit_cube_and_round_trip():
    points = np.array([[1, 2, 3], [3, 4, 5], [-3, 2, 1]])
    center, radius, scale, offset = Neus2Backend.normalization(points)
    mapped = points * scale + offset
    assert np.all(mapped > 0) and np.all(mapped < 1)
    np.testing.assert_allclose((mapped - offset) / scale, points)
    np.testing.assert_allclose(center * scale + offset, [0.5, 0.5, 0.5])
    assert radius > 0


def test_distorted_camera_rejected():
    with pytest.raises(SceneError, match="undistorted"):
        Neus2Backend.camera_matrix(SimpleNamespace(model=SimpleNamespace(name="OPENCV")))


def test_split_must_be_disjoint_and_exact():
    with pytest.raises(SceneError, match="partition"):
        Neus2Backend.partition_names(["a", "b"], {"train": ["a"], "test": ["a", "b"]})
    assert Neus2Backend.image_size(6240, 4160, 1600) == (1600, 1066)


def test_backend_default_and_override_stage_contract(tmp_path):
    config = Neus2Config(eval=True, iterations=30)
    assert config.upstream == LIBS_DIR / "neus2"
    backend = Neus2Backend(config)
    stages = backend.stages(Scene(Path("/scene")), Path("/out"))
    assert len(stages) == 4
    assert "--eval" in stages[0].cmd
    assert stages[0].cmd[stages[0].cmd.index("--iterations") + 1] == "30"
    assert stages[1].cmd[-1] == "train"
    assert backend.mesh_path(Path("/out")) == Path("/out/mesh/neus2.ply")
    custom = Neus2Backend(Neus2Config(upstream=tmp_path / "source", build_dir=str(tmp_path / "build")))
    stages = custom.stages(Scene(Path("/scene")), Path("/out"))
    assert len(stages) == 3
    for stage in stages:
        assert stage.cmd[stage.cmd.index("--upstream") + 1] == str(tmp_path / "source")
        assert stage.cmd[stage.cmd.index("--build-dir") + 1] == str(tmp_path / "build")


class FakeTestbed:
    def __init__(self):
        self.training_step = 0
        self.shall_train = False
        self.loss = self.ek_loss = self.mask_loss = 0.1
        self.nerf = SimpleNamespace(training=SimpleNamespace())
        self.calls = []

    def load_training_data(self, path):
        metadata = json.loads(Path(path).read_text())
        self.frames = metadata["frames"]
        self.nerf.training.dataset = SimpleNamespace(
            n_images=len(self.frames), from_na=metadata["from_na"],
            scale=metadata["scale"], offset=metadata["offset"])
        self.calls.append(("data", Path(path).name))

    def reload_network_from_file(self, path):
        self.calls.append(("config", path))

    def train(self, batch_size):
        self.training_step += 1
        self.calls.append(("train", batch_size))

    def save_snapshot(self, path, optimizer):
        Path(path).write_text(json.dumps({"iteration": self.training_step}))
        self.calls.append(("save", optimizer))

    def load_snapshot(self, path):
        self.training_step = json.loads(Path(path).read_text())["iteration"]
        self.calls.append(("snapshot", path))

    def reset_camera(self):
        self.calls.append(("reset",))

    def set_camera_to_training_view(self, index):
        self.calls.append(("camera", self.frames[index]["name"]))

    def render(self, width, height, spp, linear):
        assert not self.shall_train and linear
        self.calls.append(("render", width, height, spp))
        return np.full((height, width, 4), 0.21404114)


@pytest.fixture
def fake_runtime(scene_args, monkeypatch):
    instances = []

    def factory(mode):
        testbed = FakeTestbed()
        instances.append(testbed)
        return testbed

    monkeypatch.setattr(Neus2Backend, "load_bindings", staticmethod(
        lambda directory: SimpleNamespace(Testbed=factory, TestbedMode=SimpleNamespace(Nerf=0))))
    backend = Neus2Backend(replace(scene_args.backend.config, iterations=3))
    backend.prepare_scene(Scene(scene_args.scene), scene_args.output)
    return scene_args, backend, instances


def test_training_uses_sample_batch_each_step_and_saves_inference_snapshot(fake_runtime):
    args, backend, instances = fake_runtime
    backend.train(args.output)
    testbed = instances[-1]
    assert testbed.calls.count(("train", 262144)) == 3
    assert ("save", False) in testbed.calls
    assert testbed.training_step == 3 and not testbed.shall_train
    assert not testbed.nerf.training.optimize_extrinsics
    with pytest.raises(SceneError, match="overwrite"):
        backend.train(args.output)


def test_held_out_rendering_loads_only_test_views_and_never_trains(fake_runtime):
    args, backend, instances = fake_runtime
    backend.train(args.output)
    backend.render(args.output)
    testbed = instances[-1]
    assert ("data", "transforms_test.json") in testbed.calls
    assert [call[1] for call in testbed.calls if call[0] == "camera"] == ["x", "y", "z"]
    assert not any(call[0] == "train" for call in testbed.calls)
    export = args.output / "test/ours_3"
    assert sorted(path.stem for path in (export / "renders").glob("*.png")) == ["x", "y", "z"]
    with Image.open(export / "renders/x.png") as image:
        np.testing.assert_allclose(np.asarray(image), 128, atol=1)
    assert json.loads((args.output / "render_provenance.json").read_text())["n_test"] == 3
    with pytest.raises(SceneError, match="overwrite"):
        backend.render(args.output)


def test_snapshot_iteration_is_verified(fake_runtime):
    args, backend, instances = fake_runtime
    backend.train(args.output)
    (args.output / "checkpoints/3.msgpack").write_text(json.dumps({"iteration": 2}))
    with pytest.raises(SceneError, match="iteration"):
        backend.load_testbed(args.output, "test", restore=True)


def test_training_stall_and_nonfinite_loss_fail():
    testbed = FakeTestbed()
    testbed.train = lambda batch_size: None
    with pytest.raises(SceneError, match="stalled"):
        Neus2Backend.run_training(testbed, 3, 262144)
    testbed = FakeTestbed()
    testbed.loss = float("nan")
    with pytest.raises(SceneError, match="Nonfinite"):
        Neus2Backend.run_training(testbed, 3, 262144)


def test_binding_selection_rejects_missing_or_ambiguous_extensions(tmp_path):
    with pytest.raises(SceneError, match="pyngp"):
        Neus2Backend.binding_path(tmp_path)
    selected = tmp_path / "pyngp.test.so"
    selected.touch()
    assert Neus2Backend.binding_path(tmp_path) == selected
    (tmp_path / "pyngp.other.so").touch()
    with pytest.raises(SceneError, match="one"):
        Neus2Backend.binding_path(tmp_path)


def test_linear_rgb_conversion_rejects_nonfinite():
    np.testing.assert_allclose(Neus2Backend.srgb_from_linear([0, 0.21404114, 1]), [0, 0.5, 1], atol=1e-6)
    with pytest.raises(SceneError, match="Nonfinite"):
        Neus2Backend.srgb_from_linear([float("nan")])


def test_registration_does_not_import_cuda_bindings():
    source = str(Path(__file__).resolve().parents[1] / "src")
    code = """
import importlib.abc
import sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyngp', 'tinycudann'}:
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, BlockNative())
import augenblick.reconstruction
from augenblick.core.registry import RECONSTRUCTION_REGISTRY
assert 'neus2' in RECONSTRUCTION_REGISTRY
"""
    subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": source},
                   check=True, capture_output=True, text=True)


def test_native_stage_runs_from_upstream_for_initialization_weights(tmp_path, monkeypatch):
    from augenblick.reconstruction.neus2 import main

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["runner", "--upstream", str(upstream), "--output", str(output), "train"])
    calls = []
    monkeypatch.setattr(Neus2Backend, "train", lambda self, output: calls.append((Path.cwd(), output)))
    main()
    assert calls == [(upstream, output)]
    assert Path.cwd() == tmp_path


@pytest.mark.parametrize("stage_index", [0, 1, 2, 3])
def test_stage_subprocess_entrypoints_do_not_duplicate_registration(stage_index):
    backend = Neus2Backend(Neus2Config(eval=True))
    command = backend.stages(Scene(Path("/scene")), Path("/output"))[stage_index].cmd
    assert command[:2] == [sys.executable, "-c"]
    source = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(command + ["--help"],
                            env={**os.environ, "PYTHONPATH": source},
                            check=True, capture_output=True, text=True)
    assert "usage:" in result.stdout


def test_preparation_dispatch_preserves_backend_configuration(scene_args, monkeypatch):
    from augenblick.reconstruction.neus2 import main

    stage = scene_args.backend.stages(Scene(scene_args.scene), scene_args.output)[0]
    monkeypatch.setattr(sys, "argv", ["neus2-stage", *stage.cmd[3:]])
    main()
    manifest = json.loads((scene_args.output / "neus2_manifest.json").read_text())
    assert manifest["resolution"] == [4, 3]
    assert manifest["iterations"] == 15000
    assert manifest["batch_size"] == 262144
    assert manifest["test_views"] == 3
    assert "prepare" in json.loads((scene_args.output / "stage_timings.json").read_text())


@pytest.mark.parametrize("stage_index,method", [(2, "extract"), (3, "render")])
def test_native_dispatch_preserves_stage_options(tmp_path, monkeypatch, stage_index, method):
    from augenblick.reconstruction.neus2 import main

    output = tmp_path / "output"
    output.mkdir()
    backend = Neus2Backend(Neus2Config(upstream=tmp_path, eval=True,
                                     mesh_resolution=96, render_spp=3))
    command = backend.stages(Scene(Path("/scene")), output)[stage_index].cmd
    monkeypatch.setattr(sys, "argv", ["neus2-stage", *command[3:]])
    calls = []
    monkeypatch.setattr(Neus2Backend, method, lambda self, path: calls.append((self.config, path)))
    main()
    config, destination = calls[0]
    assert destination == output
    assert getattr(config, "mesh_resolution" if method == "extract" else "render_spp") == (
        96 if method == "extract" else 3)


def test_native_failure_restores_working_directory(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    monkeypatch.chdir(tmp_path)
    backend = Neus2Backend(Neus2Config(upstream=upstream))

    def fail(output):
        assert Path.cwd() == upstream
        raise SceneError("native failure")

    monkeypatch.setattr(backend, "train", fail)
    with pytest.raises(SceneError, match="native failure"):
        backend.execute_stage("train", tmp_path / "output")
    assert Path.cwd() == tmp_path