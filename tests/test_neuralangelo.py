import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from augenblick.core.errors import SceneError
from augenblick.core.registry import RECONSTRUCTION_REGISTRY
from augenblick.core.scene import Scene
from augenblick.reconstruction.neuralangelo import NeuralangeloBackend, NeuralangeloConfig
from augenblick.reconstruction.neuralangelo_runner import camera_matrix, checkpoint, disabled_logging, image_size, normalization, partition_names
from augenblick.reconstruction.neuralangelo_runner import apply_training_schedule, training_schedule
from augenblick.reconstruction.neuralangelo_runner import render_request
from augenblick.reconstruction.neuralangelo_runner import training_arguments
from augenblick.reconstruction.neuralangelo_runner import extraction_request
from augenblick.reconstruction.neuralangelo_runner import validate_resume_state
from augenblick.reconstruction.neuralangelo_runner import prepare


@pytest.fixture
def prepared_scene_args(tmp_path, monkeypatch):
    scene = tmp_path / "source"
    model = scene / "sparse/0"
    model.mkdir(parents=True)
    (scene / "images").mkdir()
    (scene / "masks").mkdir()
    (model / "cameras.txt").write_text(
        "1 SIMPLE_PINHOLE 8 6 8 4 3\n2 PINHOLE 8 6 12 10 3 2\n")
    names = ["a", "b", "c", "x", "y", "z"]
    image_records = []
    for image_id, name in enumerate(names, start=1):
        camera_id = 2 if name == "b" else 1
        observations = "1 1 1 2 2 2" if image_id <= 3 else "1 1 3"
        image_records.append(
            f"{image_id} 1 0 0 0 1 2 3 {camera_id} {name}.png\n{observations}\n")
        Image.new("RGB", (8, 6), (200, 100, 50)).save(scene / "images" / f"{name}.png")
        mask = np.zeros((6, 8), dtype=np.uint8)
        mask[:, :4] = 255
        Image.fromarray(mask).save(scene / "masks" / f"{name}.png")
    (model / "images.txt").write_text("".join(image_records))
    (model / "points3D.txt").write_text(
        "1 1 2 3 255 255 255 0.1 1 0 2 0 3 0\n"
        "2 3 4 5 255 255 255 0.1 1 1 2 1 3 1\n"
        "3 1000 1000 1000 255 255 255 0.1 4 0 5 0 6 0\n")
    (scene / "split.json").write_text(json.dumps({"train": names[:3], "test": names[3:]}))
    monkeypatch.setattr(
        "augenblick.reconstruction.neuralangelo_runner.subprocess.check_output",
        lambda *arguments, **keywords: "synthetic-upstream-revision\n")
    return SimpleNamespace(scene=scene, output=tmp_path / "output", upstream=tmp_path / "upstream",
                           eval=True, iterations=30, resolution=4, rays=8, batch_size=1,
                           seed=22, shortened_schedule=False)


def test_prepare_preserves_split_intrinsics_poses_masks_and_training_bounds(prepared_scene_args):
    args = prepared_scene_args
    original_split = (args.scene / "split.json").read_bytes()
    prepare(args)
    metadata = json.loads((args.output / "scene/transforms.json").read_text())
    assert [frame["name"] for frame in metadata["frames"]] == ["a", "b", "c"]
    assert [frame["name"] for frame in metadata["val_frames"]] == ["x", "y", "z"]
    np.testing.assert_allclose(metadata["frames"][0]["intrinsic_matrix"],
                               [[4, 0, 2], [0, 4, 1.5], [0, 0, 1]])
    np.testing.assert_allclose(metadata["frames"][1]["intrinsic_matrix"],
                               [[6, 0, 1.5], [0, 5, 1], [0, 0, 1]])
    np.testing.assert_allclose(metadata["frames"][0]["transform_matrix"],
                               [[1, 0, 0, -1], [0, -1, 0, -2],
                                [0, 0, -1, -3], [0, 0, 0, 1]])
    np.testing.assert_allclose(metadata["sphere_center"], [2, 3, 4])
    assert metadata["sphere_radius"] == pytest.approx(np.sqrt(3) * 1.1)
    with Image.open(args.scene / "images/a.png") as photograph:
        with Image.open(args.scene / "masks/a.png") as mask:
            expected = Image.composite(photograph, Image.new("RGB", (8, 6)), mask)
            expected = expected.resize((4, 3), Image.Resampling.LANCZOS)
    with Image.open(args.output / "scene/images/a.png") as prepared:
        np.testing.assert_array_equal(np.asarray(prepared), np.asarray(expected))
    manifest = json.loads((args.output / "neuralangelo_manifest.json").read_text())
    assert (manifest["train_views"], manifest["test_views"]) == (3, 3)
    assert manifest["normalization_points"] == 2
    assert manifest["resolution"] == [4, 3]
    assert len(manifest["prepared_rgb_sha256"]) == len(manifest["split_sha256"]) == 64
    assert (args.scene / "split.json").read_bytes() == original_split
    with pytest.raises(SceneError, match="overwrite prepared scene"):
        prepare(args)


@pytest.mark.parametrize("invalid_mask", ["empty", "dimensions"])
def test_prepare_rejects_invalid_masks(prepared_scene_args, invalid_mask):
    args = prepared_scene_args
    size = (8, 6) if invalid_mask == "empty" else (4, 3)
    Image.new("L", size, 0).save(args.scene / "masks/a.png")
    with pytest.raises(SceneError, match="Empty specimen mask|Mask dimensions disagree"):
        prepare(args)


def test_prepare_without_evaluation_uses_all_registered_views(prepared_scene_args):
    args = prepared_scene_args
    args.eval = False
    prepare(args)
    metadata = json.loads((args.output / "scene/transforms.json").read_text())
    assert len(metadata["frames"]) == 6
    assert metadata["val_frames"] == []


def test_backend_registration_and_stage_contract():
    assert RECONSTRUCTION_REGISTRY["neuralangelo"] is NeuralangeloBackend
    backend = NeuralangeloBackend(NeuralangeloConfig(eval=True, iterations=12))
    stages = backend.stages(Scene(Path("/scene")), Path("/out"))
    assert len(stages) == 4
    assert all(stage.cmd[:2] == [sys.executable, "-m"] for stage in stages)
    assert "--eval" in stages[0].cmd
    assert stages[0].cmd[stages[0].cmd.index("--iterations") + 1] == "12"
    assert stages[1].cmd[-1] == "train"
    assert stages[3].cmd[-1] == "render"
    assert backend.mesh_path(Path("/out")) == Path("/out/mesh/neuralangelo.ply")


def test_registration_does_not_require_optional_upstream_dependencies():
    source = str(Path(__file__).resolve().parents[1] / "src")
    code = """
import importlib.abc
import sys

class BlockOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tinycudann', 'imaginaire', 'projects', 'wandb', 'diskcache'}:
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, BlockOptionalDependencies())
import augenblick.reconstruction
from augenblick.core.registry import RECONSTRUCTION_REGISTRY
assert 'neuralangelo' in RECONSTRUCTION_REGISTRY
"""
    subprocess.run([sys.executable, "-c", code], check=True,
                   env={**os.environ, "PYTHONPATH": source}, capture_output=True, text=True)


def test_non_eval_backend_has_no_held_out_stage():
    backend = NeuralangeloBackend(NeuralangeloConfig())
    assert len(backend.stages(Scene(Path("/scene")), Path("/out"))) == 3


@pytest.mark.parametrize("model,parameters,expected", [
    ("SIMPLE_PINHOLE", [100, 40, 30], [[100, 0, 40], [0, 100, 30], [0, 0, 1]]),
    ("PINHOLE", [100, 120, 40, 30], [[100, 0, 40], [0, 120, 30], [0, 0, 1]]),
])
def test_supported_camera_intrinsics(model, parameters, expected):
    camera = SimpleNamespace(model=SimpleNamespace(name=model), params=parameters)
    np.testing.assert_allclose(camera_matrix(camera), expected)


def test_distorted_camera_rejected():
    camera = SimpleNamespace(model=SimpleNamespace(name="OPENCV"), params=[])
    with pytest.raises(SceneError, match="undistorted"):
        camera_matrix(camera)


@pytest.mark.parametrize("split", [
    {"train": ["a"], "test": ["a"]},
    {"train": ["a", "a"], "test": ["b"]},
    {"train": ["a"], "test": []},
])
def test_invalid_split_rejected(split):
    with pytest.raises(SceneError):
        partition_names(["a", "b"], split)


def test_normalization_contains_points_and_round_trips():
    points = np.array([[1, 2, 3], [3, 4, 5], [-3, 2, 1]])
    center, radius = normalization(points)
    normalized = (points - center) / radius
    assert np.linalg.norm(normalized, axis=1).max() < 1
    np.testing.assert_allclose(normalized * radius + center, points)


def test_image_resolution_matches_existing_held_out_exports():
    assert image_size(6240, 4160, 1600) == (1600, 1066)
    assert image_size(320, 200, 1600) == (320, 200)


def test_checkpoint_selection_uses_iteration_not_modification_time(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    final = logs / "epoch_00004_iteration_000000030_checkpoint.pt"
    final.touch()
    (logs / "epoch_00001_iteration_000000010_checkpoint.pt").touch()
    assert checkpoint(tmp_path) == final


def test_logging_is_disabled_without_legacy_wandb_calls(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=lambda **options: calls.append(options)))
    disabled_logging(None, None, resume=False)
    assert calls == [{"mode": "disabled"}]


def test_shortened_schedule_reaches_all_detail_levels_and_lr_decays():
    config = {"model": {"render": {"rand_rays": 512}}, "max_iter": 100000}
    apply_training_schedule(config, 100000, shortened=True)
    assert config["optim"]["sched"] == {"warm_up_end": 1000, "two_steps": [60000, 80000]}
    step = config["model"]["object"]["sdf"]["encoding"]["coarse2fine"]["step"]
    assert step == 1000
    assert (100000 - 1000) // step >= 16
    assert config["model"]["render"]["rand_rays"] == 512


def test_default_schedule_is_not_changed():
    config = {"model": {"render": {"rand_rays": 512}}}
    assert apply_training_schedule(config, 500000) == config
    assert training_schedule(30) == {}


def test_shortened_batch_four_backend_arguments():
    backend = NeuralangeloBackend(NeuralangeloConfig(
        iterations=100000, batch_size=4, shortened_schedule=True))
    preparation = backend.stages(Scene(Path("/scene")), Path("/out"))[0].cmd
    assert preparation[preparation.index("--batch-size") + 1] == "4"
    assert "--shortened-schedule" in preparation


def test_intermediate_render_preserves_training_target_and_separates_outputs(tmp_path):
    source = tmp_path / "training"
    selected = source / "logs/epoch_01500_iteration_000180000_checkpoint.pt"
    selected.parent.mkdir(parents=True)
    selected.touch()
    destination = tmp_path / "evaluation"
    args = SimpleNamespace(output=source, checkpoint=selected, iteration=180000,
                           evaluation_output=destination)
    manifest = {"iterations": 500000}
    assert render_request(args, manifest) == (selected, destination, 180000)
    assert manifest["iterations"] == 500000
    assert not destination.exists()


@pytest.mark.parametrize("destination", ["training", "training/evaluation"])
def test_intermediate_render_cannot_write_into_training(tmp_path, destination):
    selected = tmp_path / "checkpoint.pt"
    selected.touch()
    args = SimpleNamespace(output=tmp_path / "training", checkpoint=selected,
                           iteration=180000, evaluation_output=tmp_path / destination)
    with pytest.raises(SceneError, match="outside the training output"):
        render_request(args, {"iterations": 500000})


def test_partial_intermediate_render_arguments_rejected(tmp_path):
    args = SimpleNamespace(output=tmp_path, checkpoint=None, iteration=180000,
                           evaluation_output=None)
    with pytest.raises(SceneError, match="requires"):
        render_request(args, {"iterations": 500000})


def test_default_render_still_requires_final_iteration(tmp_path):
    selected = tmp_path / "logs/epoch_01500_iteration_000180000_checkpoint.pt"
    selected.parent.mkdir()
    selected.touch()
    args = SimpleNamespace(output=tmp_path)
    assert render_request(args, {"iterations": 500000}) == (selected, tmp_path, 500000)


def test_resume_restores_checkpoint_with_original_configuration(tmp_path):
    selected = tmp_path / "logs/epoch_02166_iteration_000260000_checkpoint.pt"
    selected.parent.mkdir()
    selected.touch()
    manifest = {"seed": 22, "iterations": 500000}
    arguments = training_arguments(tmp_path, manifest, selected)
    assert arguments[arguments.index("--config") + 1] == str(tmp_path / "neuralangelo.yaml")
    assert arguments[arguments.index("--checkpoint") + 1] == str(selected)
    assert "--resume" in arguments
    assert manifest["iterations"] == 500000


def test_new_training_does_not_resume(tmp_path):
    arguments = training_arguments(tmp_path, {"seed": 22})
    assert "--resume" not in arguments and "--checkpoint" not in arguments


def test_resume_rejects_missing_or_foreign_checkpoint(tmp_path):
    with pytest.raises(SceneError, match="this training run"):
        training_arguments(tmp_path, {"seed": 22}, tmp_path / "missing.pt")
    foreign = tmp_path / "other.pt"
    foreign.touch()
    with pytest.raises(SceneError, match="this training run"):
        training_arguments(tmp_path, {"seed": 22}, foreign)


def test_intermediate_mesh_uses_same_validated_checkpoint(tmp_path):
    source, destination = tmp_path / "training", tmp_path / "evaluation"
    source.mkdir()
    destination.mkdir()
    selected = source / "checkpoint.pt"
    selected.touch()
    args = SimpleNamespace(output=source, checkpoint=selected, evaluation_output=destination)
    with pytest.raises(SceneError, match="before extracting"):
        extraction_request(args)
    provenance = {"training_output": str(source), "checkpoint": str(selected), "checkpoint_bytes": 0}
    (destination / "checkpoint_evaluation.json").write_text(json.dumps(provenance))
    assert extraction_request(args) == (selected, destination)
    other = source / "other.pt"
    other.touch()
    args.checkpoint = other
    with pytest.raises(SceneError, match="does not match"):
        extraction_request(args)


@pytest.mark.parametrize("scheduler_iteration", [259999, 260000])
def test_resume_accepts_upstream_pre_scheduler_checkpoint(scheduler_iteration):
    trainer = SimpleNamespace(checkpointer=SimpleNamespace(resume_iteration=260000),
        optim=SimpleNamespace(state={0: {"step": np.array(260000)}}),
        sched=SimpleNamespace(last_epoch=scheduler_iteration))
    assert validate_resume_state(trainer, 260000, 500000) == (260000, [260000])
    with pytest.raises(SceneError, match="differs"):
        validate_resume_state(trainer, 180000, 500000)
    trainer.sched.last_epoch = 0
    with pytest.raises(SceneError, match="not restored"):
        validate_resume_state(trainer, 260000, 500000)