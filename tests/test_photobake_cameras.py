"""Photobake uses scene poses and lazy images, not Gaussian checkpoints."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


SUGAR_DIR = Path(__file__).resolve().parents[1] / "src" / "libs" / "sugar"
sys.path.insert(0, str(SUGAR_DIR))
sys.path.insert(0, str(SUGAR_DIR / "gaussian_splatting"))

from sugar_extractors.texture import _load_photo_camera
from sugar_utils import photobake_cameras
from sugar_utils.camera_resolution import camera_resolution
from sugar_utils.photobake_cameras import (
    load_evaluation_photo, load_scene_cameras,
)


@pytest.mark.parametrize("resolution,expected", [
    (-1, (1600, 800)), (1, (4000, 2000)), (2, (2000, 1000)),
    (4, (1000, 500)), (8, (500, 250)), (1000, (1000, 500)),
    (5000, (5000, 2500)), (3, (3, 1)),
])
def test_evaluation_resolution_matches_3dgs(resolution, expected):
    assert camera_resolution(4000, 2000, resolution) == expected


def test_evaluation_auto_caps_width_not_long_side():
    assert camera_resolution(1000, 3000, -1) == (1000, 3000)
    assert camera_resolution(1667, 1001, 2) == (834, 500)
    assert camera_resolution(1667, 1001, 900) == (900, 540)


@pytest.mark.parametrize("resolution", [0, -2])
def test_invalid_evaluation_resolution(resolution):
    with pytest.raises(ValueError, match="resolution must"):
        camera_resolution(100, 100, resolution)


@pytest.fixture
def colmap_scene(tmp_path, monkeypatch):
    sparse = tmp_path / "sparse" / "0"
    sparse.mkdir(parents=True)
    images = tmp_path / "images"
    images.mkdir()
    masks = tmp_path / "masks"
    masks.mkdir()
    (sparse / "cameras.txt").write_text(
        "1 PINHOLE 2400 1200 1800 1700 1200 600\n"
        "2 PINHOLE 6 4 5 5 3 2\n")
    (sparse / "images.txt").write_text(
        "1 0.7071067811865476 0 0 0.7071067811865476 1 2 3 1 train.JPG\n\n"
        "2 1 0 0 0 4 5 6 2 heldout.png\n\n"
        "3 1 0 0 0 7 8 9 2 other.png\n\n")
    Image.new("RGB", (2400, 1200), (50, 100, 150)).save(images / "train.JPG")
    Image.new("RGB", (6, 4), (255, 255, 255)).save(images / "heldout.png")
    Image.new("RGB", (6, 4), (10, 20, 30)).save(images / "other.png")
    mask = np.zeros((4, 6), dtype=np.uint8)
    mask[:, :3] = 255
    Image.fromarray(mask).save(masks / "heldout.png")
    (tmp_path / "split.json").write_text(json.dumps({
        "train": ["train"], "test": ["heldout", "other"],
    }))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
    return tmp_path


@pytest.mark.parametrize("resolution,train_size", [(-1, (1600, 800)), (2, (1200, 600))])
def test_cameras_use_scene_split_poses_without_checkpoint_or_decoding(
        colmap_scene, monkeypatch, resolution, train_size):
    def forbidden_load(*args, **kwargs):
        pytest.fail("camera metadata must not load checkpoint tensors or full photographs")

    monkeypatch.setattr(torch, "load", forbidden_load)
    monkeypatch.setattr(Image.Image, "load", forbidden_load)
    opened_images = []
    original_open = Image.open

    def track_open(*args, **kwargs):
        image = original_open(*args, **kwargs)
        opened_images.append(image)
        return image

    monkeypatch.setattr(Image, "open", track_open)
    training, heldout = load_scene_cameras(colmap_scene, resolution, device="cpu")

    assert [camera.image_name for camera in training.gs_cameras] == ["train"]
    assert [camera.image_name for camera in heldout.gs_cameras] == ["heldout", "other"]
    camera = training.gs_cameras[0]
    assert (camera.image_width, camera.image_height) == train_size
    np.testing.assert_allclose(camera.R, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-7)
    np.testing.assert_equal(camera.T, [1, 2, 3])
    assert camera.FoVx == pytest.approx(2 * np.arctan(2400 / (2 * 1800)))
    assert camera.FoVy == pytest.approx(2 * np.arctan(1200 / (2 * 1700)))
    assert training.fx[0].item() == pytest.approx(1800 * train_size[0] / 2400)
    assert training.fy[0].item() == pytest.approx(1700 * train_size[1] / 1200)
    assert not hasattr(camera, "original_image")
    assert len(opened_images) == 3
    assert all(image.fp is None for image in opened_images)


def test_reader_fallback_split_is_preserved(colmap_scene):
    (colmap_scene / "split.json").unlink()
    training, heldout = load_scene_cameras(colmap_scene, 1, device="cpu")
    assert [camera.image_name for camera in training.gs_cameras] == ["other", "train"]
    assert [camera.image_name for camera in heldout.gs_cameras] == ["heldout"]


@pytest.mark.parametrize("train,test,error", [
    (["train"], ["heldout", "missing"], "do not match"),
    (["train"], ["train", "heldout"], "unique, disjoint"),
    (["train", "train"], ["heldout"], "unique, disjoint"),
])
def test_split_cannot_silently_drop_or_overlap_views(colmap_scene, train, test, error):
    (colmap_scene / "split.json").write_text(json.dumps({"train": train, "test": test}))
    with pytest.raises(ValueError, match=error):
        load_scene_cameras(colmap_scene, device="cpu")


def test_camera_loading_never_reads_or_creates_point_clouds(colmap_scene, monkeypatch):
    from scene import dataset_readers

    def forbidden_point_cloud(*args, **kwargs):
        pytest.fail("camera-only loading must not read or write point clouds")

    for name in ("fetchPly", "storePly", "read_points3D_binary", "read_points3D_text"):
        monkeypatch.setattr(dataset_readers, name, forbidden_point_cloud)
    training, heldout = load_scene_cameras(colmap_scene, device="cpu")
    assert len(training.gs_cameras) == 1
    assert len(heldout.gs_cameras) == 2
    assert not (colmap_scene / "sparse" / "0" / "points3D.ply").exists()


def test_heldout_ground_truth_and_mask_match_render_size(colmap_scene):
    _, heldout = load_scene_cameras(colmap_scene, 2, device="cpu")
    camera = heldout.gs_cameras[0]
    image, mask = load_evaluation_photo(colmap_scene, camera, [0, 0, 0], "cpu")
    assert image.shape == (2, 3, 3)
    assert mask.shape == image.shape[:2]
    assert mask.dtype == torch.bool
    assert torch.equal(image[..., 0], mask.float())
    expected = Image.open(colmap_scene / "masks" / "heldout.png")
    with expected:
        expected_mask = np.asarray(expected.resize((3, 2), Image.Resampling.NEAREST)) > 127
    np.testing.assert_equal(mask.numpy(), expected_mask)


@pytest.mark.parametrize("photo_max_size,photo_size", [(-1, (2400, 1200)), (600, (600, 300))])
def test_training_photo_size_is_independent_of_evaluation_resolution(
    colmap_scene, photo_max_size, photo_size):
    training, _ = load_scene_cameras(colmap_scene, -1, device="cpu")
    camera = training.gs_cameras[0]
    image, mask, photo_camera = _load_photo_camera(colmap_scene, camera, photo_max_size, "cpu")
    assert (camera.image_width, camera.image_height) == (1600, 800)
    assert image.shape == (1, photo_size[1], photo_size[0], 3)
    assert mask is None
    assert (photo_camera.image_width, photo_camera.image_height) == photo_size
    np.testing.assert_equal(photo_camera.R, camera.R)
    np.testing.assert_equal(photo_camera.T, camera.T)
    assert (photo_camera.FoVx, photo_camera.FoVy) == (camera.FoVx, camera.FoVy)


def test_compatibility_reader_images_close_on_conversion_failure(colmap_scene, monkeypatch):
    scene = photobake_cameras.readColmapSceneInfo(
        str(colmap_scene), images=None, eval=True, load_point_cloud=False)
    images = [info.image for info in scene.train_cameras + scene.test_cameras]
    monkeypatch.setattr(photobake_cameras, "readColmapSceneInfo", lambda *args, **kwargs: scene)
    monkeypatch.setattr(photobake_cameras, "GSCamera", lambda **kwargs: (_ for _ in ()).throw(
        RuntimeError("conversion failed")))
    with pytest.raises(RuntimeError, match="conversion failed"):
        load_scene_cameras(colmap_scene, device="cpu")
    for image in images:
        with pytest.raises(ValueError, match="closed image"):
            image.getpixel((0, 0))