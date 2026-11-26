"""NeuS2's vendored stage scripts on the CPU: normalisation, staging, and the pieces that need no GPU."""
import dataclasses
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from augenblick.core.errors import SceneError
from augenblick.core.scene import Scene
from augenblick.eval.mesh_render import image_size
from augenblick.eval.split import write_split
from augenblick.reconstruction.base import LIBS_DIR
from augenblick.reconstruction.neus2 import Neus2Config

from conftest import SYNTH_CENTRE, SYNTH_F, SYNTH_H, SYNTH_W

LIB = LIBS_DIR / "neus2"
SCRIPTS = ("prepare", "train", "extract_mesh", "render")


def load(name):
    if str(LIB) not in sys.path:
        sys.path.insert(0, str(LIB))
    spec = importlib.util.spec_from_file_location(f"neus2_{name}", LIB / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scene_io = load("scene_io")


def nerf_matrix_to_ngp_from_na(matrix):
    """nerf_loader.h:112-133 with from_na set, re-implemented independently."""
    result = np.array(matrix, dtype=np.float64)[:3].copy()
    result[:, 1] *= -1
    result[:, 2] *= -1
    result[:, 3] = result[:, 3] * scene_io.NGP_SCALE + np.asarray(scene_io.NGP_OFFSET)
    result[:, 1] *= -1
    result[:, 2] *= -1
    return result


def run_prepare(root, out, *extra):
    return subprocess.run([sys.executable, str(LIB / "prepare.py"), "-s", str(root), "-m", str(out), *extra],
                          cwd=LIB, capture_output=True, text=True,
                          env={**os.environ, "SLURM_CPUS_PER_TASK": "2"})


@pytest.fixture
def staged(ring_scene, tmp_path):
    root = ring_scene(outliers=3, mask_radius=1.2)
    write_split(root)
    out = tmp_path / "out"
    result = run_prepare(root, out, "--eval")
    assert result.returncode == 0, result.stderr
    return root, out


def test_prepare_normalisation_round_trip(staged):
    from augenblick.eval.cameras import camera_records

    root, out = staged
    norm = json.loads((out / scene_io.NORMALIZATION).read_text())
    centre, radius = np.array(norm["centre"]), norm["radius"]
    assert np.linalg.norm(centre - SYNTH_CENTRE) < 0.05
    records = {r["img_name"]: r for r in camera_records(Scene(root).model_dir)}
    frames = json.loads((out / scene_io.SCENE_DIR / "transforms_train.json").read_text())["frames"]
    for frame in frames:
        record, pose = records[Path(frame["file_path"]).stem], np.array(frame["transform_matrix"])
        assert np.allclose(pose[:3, :3], record["rotation"])
        assert np.allclose(pose[:3, 3] * radius + centre, record["position"])
        ngp = nerf_matrix_to_ngp_from_na(pose)
        assert np.allclose(ngp[:, 3], 0.5 + 0.5 * (np.array(record["position"]) - centre) / radius)
        assert np.allclose(scene_io.ngp_to_world(ngp[:, 3][None], centre, radius)[0], record["position"])


def test_prepare_json_contract(staged):
    root, out = staged
    split = json.loads((root / "split.json").read_text())
    for side in ("train", "test"):
        data = json.loads((out / scene_io.SCENE_DIR / f"transforms_{side}.json").read_text())
        assert data["from_na"] is True and data["scale"] == 0.5
        assert data["offset"] == [0.5, 0.5, 0.5] and data["aabb_scale"] == 1
        assert not {"fl_x", "fl_y", "cx", "cy", "w", "h"} & set(data)
        stems = [Path(f["file_path"]).stem for f in data["frames"]]
        assert stems == sorted(split[side])
        for frame in data["frames"]:
            assert (frame["fl_x"], frame["fl_y"]) == (SYNTH_F, SYNTH_F)
            assert (frame["cx"], frame["cy"]) == (SYNTH_W / 2 + 3, SYNTH_H / 2 - 2)
            assert (frame["w"], frame["h"]) == (SYNTH_W, SYNTH_H)


def test_prepare_bound(staged):
    _, out = staged
    norm = json.loads((out / scene_io.NORMALIZATION).read_text())
    # Three points 30 units out must not inflate the box; the wider mask sets the radius.
    assert norm["r_pts"] < 1.0
    assert norm["r_mask"] > norm["r_pts"]
    assert norm["radius"] == pytest.approx(1.1 * norm["r_mask"])
    assert norm["camera_ratio"] >= scene_io.MIN_CAMERA_RATIO and norm["mask_miss_fraction"] <= 0.001


def test_prepare_stages_rgba_like_the_scorer(staged):
    root, out = staged
    split = json.loads((root / "split.json").read_text())
    images = out / scene_io.SCENE_DIR / "images"
    assert sorted(p.stem for p in images.iterdir()) == sorted(split["train"])
    stem = split["train"][0]
    rgba = np.asarray(Image.open(images / f"{stem}.png"))
    assert rgba.shape == (SYNTH_H, SYNTH_W, 4)
    assert np.array_equal(rgba[..., 3] > 0, scene_io.raster_mask(root / "masks" / f"{stem}.png", (SYNTH_W, SYNTH_H)))
    assert set(np.unique(rgba[..., 3])) <= {0, 255}
    canonical = scene_io.canonical_rgb(root / "images" / f"{stem}.jpg", (SYNTH_W, SYNTH_H))
    assert np.array_equal(rgba[..., :3], np.asarray(canonical))


def test_prepare_refuses_a_sphere_reaching_the_cameras(ring_scene, tmp_path):
    result = run_prepare(ring_scene(n_views=8), tmp_path / "out", "--bound_margin", "4.0")
    assert result.returncode == 2
    assert "NeuS2 needs every camera beyond" in result.stderr


def test_camera_gate_and_robust_bound():
    with pytest.raises(SceneError, match="camera"):
        scene_io.check_camera_gate(np.array([[0.0, 0.0, 1.2]]), np.zeros(3), 1.0)
    assert scene_io.check_camera_gate(np.array([[0.0, 0.0, 2.0]]), np.zeros(3), 1.0) == pytest.approx(2.0)
    rng = np.random.default_rng(1)
    points = np.vstack([rng.uniform(-1, 1, (1000, 3)), rng.uniform(-50, 50, (5, 3))])
    centre, radius = scene_io.robust_bound(points, 1.0)
    assert np.all(np.abs(centre) < 0.1) and radius < math.sqrt(3) * 1.05


def test_raster_floor_rule():
    assert image_size({"width": 6240, "height": 4160}, -1) == (1600, 1066)


def test_srgb_round_trip():
    x = np.linspace(0, 1, 257)
    assert np.allclose(scene_io.linear_to_srgb(scene_io.srgb_to_linear(x)), x, atol=1e-6)


def test_cluster_filter_matches_2dgs_rule():
    ids = np.array([0] * 60 + [1] * 55 + [2] * 3)
    triangles = np.zeros((len(ids), 3))
    assert scene_io.keep_largest_clusters(triangles, ids, 1).sum() == 60
    assert scene_io.keep_largest_clusters(triangles, ids, 2).sum() == 115
    assert scene_io.keep_largest_clusters(triangles, ids, 0).all()


def test_held_out_camera_reproduces_training_view():
    intrinsic = np.array([[500.0, 0, 330.0], [0, 520.0, 236.0], [0, 0, 1]])
    fov_xy, screen_center = scene_io.held_out_camera(intrinsic, (640, 480))
    relative = [1 / (2 * math.tan(math.radians(f) / 2)) for f in fov_xy]
    assert relative == pytest.approx([500.0 / 480, 520.0 / 480])
    assert screen_center == pytest.approx([1 - 330.0 / 640, 1 - 236.0 / 480])


@pytest.mark.parametrize("name", SCRIPTS + ("scene_io", "ngp_session"))
def test_scripts_import_without_pyngp_or_torch(name):
    code = (f"import sys; sys.path.insert(0, {str(LIB)!r}); sys.modules['pyngp'] = None; "
            f"import {name}; assert 'torch' not in sys.modules, 'torch imported'")
    result = subprocess.run([sys.executable, "-c", code], cwd=LIB, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_script_defaults_match_config():
    defaults = {f.name: f.default for f in dataclasses.fields(Neus2Config)}
    for name in SCRIPTS:
        parser_args = load(name).parse_args(["-s", "/s", "-m", "/m"] if name in ("prepare", "render") else ["-m", "/m"])
        for key, value in vars(parser_args).items():
            if key in defaults:
                assert value == defaults[key], (name, key)


@pytest.mark.parametrize("name", ["train", "extract_mesh"])
def test_pyngp_scripts_refuse_the_wrong_cwd(tmp_path, name):
    out = tmp_path / "out"
    (out / scene_io.SCENE_DIR).mkdir(parents=True)
    (out / scene_io.SCENE_DIR / "transforms_train.json").write_text("{}")
    (out / scene_io.NORMALIZATION).write_text(json.dumps({"centre": [0, 0, 0], "radius": 1.0}))
    result = subprocess.run([sys.executable, str(LIB / f"{name}.py"), "-m", str(out)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 2
    assert "mlp_weights.txt" in result.stderr


def test_native_selection_manifest_satisfies_the_scorer(staged, monkeypatch):
    root, out = staged
    # record_selection names the manifest after the running script, render.py in production.
    monkeypatch.setattr(sys, "argv", [str(LIB / "render.py")])
    data = lambda side: json.loads((out / scene_io.SCENE_DIR / f"transforms_{side}.json").read_text())["frames"]
    train, test = data("train"), data("test")
    sizes = {Path(f["file_path"]).stem: (80, 60) for f in test}
    scene_io.record_native_selection(root, out, train, test, sizes, -1)
    manifests = list((out / "native_selections").glob("render_*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    split = json.loads((root / "split.json").read_text())
    assert sorted(r["name"] for r in manifest["test"]) == sorted(split["test"])
    assert sorted(r["name"] for r in manifest["train"]) == sorted(split["train"])
    assert {(r["width"], r["height"]) for r in manifest["test"]} == {(80, 60)}
    assert all(r["mask_exists"] for r in manifest["train"] + manifest["test"])
    with pytest.raises(SceneError, match="partition differs"):
        scene_io.record_native_selection(root, out, train[1:], test, sizes, -1)
