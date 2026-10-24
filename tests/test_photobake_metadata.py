"""Native atlas provenance must come from a matching export manifest."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SUGAR_DIR = Path(__file__).resolve().parents[1] / "src" / "libs" / "sugar"
sys.path.insert(0, str(SUGAR_DIR))
sys.path.insert(0, str(SUGAR_DIR / "gaussian_splatting"))

from photobake_mesh import native_atlas_metadata


@pytest.mark.parametrize("square_size", [4, 8, 10])
def test_native_size_comes_from_matching_ancestor_manifest(tmp_path, square_size):
    mesh = tmp_path / "refined_mesh" / "scene" / "mesh.obj"
    mesh.parent.mkdir(parents=True)
    manifest = tmp_path / "run_manifest.json"
    manifest.write_text(json.dumps({
        "textured_mesh_path": str(mesh), "texture_square_size": square_size,
    }))
    result = native_atlas_metadata(mesh, SimpleNamespace(textures=SimpleNamespace(
        sampling_mode="bilinear")), "uv")
    assert result["square_size"] == square_size
    assert result["square_size_status"] == "known"
    assert result["manifest"] == str(manifest)
    assert result["sampling_loaded"] == "bilinear"
    assert result["sampling_original"] == "unknown"


@pytest.mark.parametrize("manifest", [None, {
    "textured_mesh_path": "different.obj", "texture_square_size": 8,
}, {
    "textured_mesh_path": "mesh.obj", "texture_square_size": None,
}])
def test_missing_or_unrelated_manifest_leaves_native_size_unknown(tmp_path, manifest):
    if manifest is not None:
        (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    result = native_atlas_metadata(tmp_path / "mesh.obj", SimpleNamespace(textures=None), "uv")
    assert result["square_size"] is None
    assert result["square_size_status"] == "unknown"


def test_vertex_colors_have_no_native_square_size(tmp_path):
    result = native_atlas_metadata(tmp_path / "mesh.ply", SimpleNamespace(textures=None), "vertex")
    assert result["square_size"] is None
    assert result["square_size_status"] == "not_applicable"


def test_run_needs_no_checkpoint_and_keeps_output_label_independent(tmp_path, monkeypatch):
    import photobake_mesh
    import torch
    from PIL import Image

    class CameraList(list):
        @property
        def gs_cameras(self):
            return self

    training = CameraList([SimpleNamespace(image_name="train")])
    heldout = CameraList([
        SimpleNamespace(image_name="view_a", image_width=6, image_height=4),
        SimpleNamespace(image_name="view_b", image_width=8, image_height=5),
    ])
    calls = []

    def load_cameras(scene, resolution, device):
        calls.append((scene, resolution, device))
        return training, heldout

    def render(rasterizer, mesh, camera_index, background, device):
        assert camera_index == 0
        return torch.zeros(rasterizer.image_height, rasterizer.image_width, 3)

    def load_truth(scene, camera, background, device):
        return torch.ones(camera.image_height, camera.image_width, 3), None

    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("loaded checkpoint"))
    monkeypatch.setattr(photobake_mesh, "load_scene_cameras", load_cameras)
    monkeypatch.setattr(photobake_mesh, "load_method_mesh", lambda *args: (
        SimpleNamespace(textures=None), "uv"))
    monkeypatch.setattr(photobake_mesh, "MeshRasterizer", lambda cameras: cameras)
    monkeypatch.setattr(photobake_mesh, "_render_native", render)
    monkeypatch.setattr(photobake_mesh, "_render_uv", render)
    monkeypatch.setattr(photobake_mesh, "load_evaluation_photo", load_truth)
    monkeypatch.setattr(photobake_mesh, "bake_photo_texture", lambda mesh, *args, **kwargs: mesh)
    monkeypatch.setattr(photobake_mesh, "_score", lambda *args: args[-1])
    scene = tmp_path / "scene"
    scene.mkdir()
    (scene / "split.json").write_text("{}")
    output = tmp_path / "output"
    result = photobake_mesh.run(
        scene, tmp_path / "mesh.obj", output, iteration=123, resolution=2,
        square_size=8, photo_max_size=600, save_baked_obj=False,
    )

    assert calls == [(scene, 2, "cuda:0")]
    metadata = json.loads((output / "photobake" / "run_metadata.json").read_text())
    assert metadata["camera_source"] == str(scene / "sparse" / "0")
    assert metadata["split_source"] == "split.json"
    assert metadata["evaluation_sizes"] == {"view_a": [6, 4], "view_b": [8, 5]}
    assert metadata["output_iteration"] == 123
    assert metadata["photo_max_size"] == 600
    assert metadata["photo_square_size"] == 8
    assert metadata["photo_sampling"] == "nearest"
    assert metadata["native_atlas"]["square_size_status"] == "unknown"
    assert result == {"native": metadata, "photofull": metadata}
    for target in ("test_mesh_native", "test_mesh_photofull"):
        for camera in heldout:
            root = output / target / "ours_123"
            for directory in ("renders", "gt"):
                with Image.open(root / directory / f"{camera.image_name}.png") as image:
                    assert image.size == (camera.image_width, camera.image_height)