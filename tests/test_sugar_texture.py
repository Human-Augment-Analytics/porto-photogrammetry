"""Focused tests for SuGaR's direct-photo texture-baking helpers."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SUGAR_DIR = Path(__file__).resolve().parents[1] / "src" / "libs" / "sugar"
sys.path.insert(0, str(SUGAR_DIR))
sys.path.insert(0, str(SUGAR_DIR / "gaussian_splatting"))

from sugar_extractors.texture import _accumulate_samples, _find_source_image, build_square_atlas


@pytest.mark.parametrize("square_size", [1, 2, 3, 4])
def test_atlas_rejects_degenerate_size_before_allocation(square_size, monkeypatch):
    def unexpected_allocation(*args, **kwargs):
        pytest.fail("allocated before validating square_size")

    monkeypatch.setattr(torch, "arange", unexpected_allocation)
    with pytest.raises(ValueError, match=">= 5"):
        build_square_atlas(2, square_size, "cpu")


@pytest.mark.parametrize("n_triangles", [1, 2, 3, 8, 9, 18])
@pytest.mark.parametrize("square_size", [5, 6, 8, 10, 17])
def test_square_atlas_preserves_offsets_and_valid_faces(n_triangles, square_size):
    faces, verts, face_map, texture_size = build_square_atlas(n_triangles, square_size, "cpu")
    side = int((n_triangles // 2 + 1) ** 0.5 + 1)
    offsets = torch.tensor([
        [square_size - 2, 1], [2, 1], [square_size - 2, square_size - 3],
        [1, square_size - 1], [1, 3], [square_size - 3, square_size - 1],
    ])
    expected = torch.stack([
        offsets + torch.tensor([column, row]) * square_size
        for column in range(side) for row in range(side)
    ]).reshape(-1, 2)[:3 * n_triangles] / texture_size
    assert torch.equal(verts, expected)
    assert torch.equal(faces.flatten(), torch.arange(3 * n_triangles))
    assert texture_size == side * square_size
    assert verts.shape == (3 * n_triangles, 2)
    assert ((verts > 0) & (verts < 1)).all()
    triangles = verts[faces]
    edges = triangles[:, 1:] - triangles[:, :1]
    area = edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]
    assert (area != 0).all()
    assert face_map.shape == (texture_size, texture_size)
    assert face_map.min() == 0
    assert face_map.max() == n_triangles - 1
    centroids = (triangles.mean(dim=1) * texture_size).long()
    assert torch.equal(face_map[centroids[:, 1], centroids[:, 0]], torch.arange(n_triangles))


def test_square_atlas_rejects_empty_mesh():
    with pytest.raises(ValueError, match="at least one triangle"):
        build_square_atlas(0, 5, "cpu")


def test_accumulate_samples_sums_repeated_faces_and_texels():
    face_colors = torch.zeros(2, 3)
    face_count = torch.zeros(2, 1)
    texture = torch.zeros(2, 2, 3)
    texture_count = torch.zeros(2, 2, 1)
    face_indices = torch.tensor([0, 0, 1])
    texture_pixels = torch.tensor([[1, 0], [1, 0], [0, 1]])
    colors = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    _accumulate_samples(
        face_colors, face_count, texture, texture_count,
        face_indices, texture_pixels, colors)

    assert torch.equal(face_count[:, 0], torch.tensor([2.0, 1.0]))
    assert torch.equal(face_colors[0], torch.tensor([1.0, 1.0, 0.0]))
    assert torch.equal(texture_count[0, 1], torch.tensor([2.0]))
    assert torch.equal(texture[0, 1], torch.tensor([1.0, 1.0, 0.0]))


def test_find_source_image_matches_stem_across_extensions(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    expected = images / "camera1_001.JPG"
    expected.touch()

    assert _find_source_image(tmp_path, "camera1_001") == expected


def test_find_source_image_rejects_ambiguous_stem(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "view.jpg").touch()
    (images / "view.png").touch()

    with pytest.raises(FileNotFoundError, match="expected one source image"):
        _find_source_image(tmp_path, "view")


@pytest.mark.parametrize("square_size", [5, 8, 10])
@pytest.mark.parametrize("sizes", [[(1, 2)], [(1, 2), (2, 1)]])
def test_native_and_photo_bakes_share_atlas_and_native_sums_samples(square_size, sizes, monkeypatch):
    import photobake_mesh
    from pytorch3d.structures import Meshes
    from sugar_extractors import texture

    class CameraList(list):
        @property
        def gs_cameras(self):
            return self

    class Rasterizer:
        def __init__(self, cameras, **kwargs):
            self.height, self.width = cameras.image_height, cameras.image_width

        def __call__(self, *args, **kwargs):
            return SimpleNamespace(
                bary_coords=torch.full((1, self.height, self.width, 1, 3), 1 / 3),
                pix_to_face=torch.zeros((1, self.height, self.width, 1), dtype=torch.long),
            )

    surface = Meshes(
        verts=[torch.tensor([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]])],
        faces=[torch.tensor([[0, 1, 2], [1, 2, 3]])],
    )
    cameras = CameraList([
        SimpleNamespace(image_height=height, image_width=width) for height, width in sizes
    ])
    rgb = torch.tensor([[[1., 0., 0.], [0., 1., 0.]]])
    sugar = SimpleNamespace(
        device="cpu", surface_mesh=surface,
        nerfmodel=SimpleNamespace(training_cameras=cameras),
        _sh_coordinates_dc=torch.zeros(2, 1, 3),
        render_image_gaussian_rasterizer=lambda camera_indices, **kwargs: rgb.reshape(
            *sizes[camera_indices], 3),
    )
    atlas_calls = []
    accumulation_calls = []

    def atlas_spy(*args):
        atlas_calls.append(args)
        return build_square_atlas(*args)

    def accumulation_spy(*args):
        _accumulate_samples(*args)
        accumulation_calls.append(tuple(tensor.clone() for tensor in args[:4]))

    monkeypatch.setattr(texture, "build_square_atlas", atlas_spy)
    monkeypatch.setattr(photobake_mesh, "build_square_atlas", atlas_spy)
    monkeypatch.setattr(texture, "_accumulate_samples", accumulation_spy)
    monkeypatch.setattr(texture, "MeshRasterizer", Rasterizer)
    monkeypatch.setattr(photobake_mesh, "MeshRasterizer", Rasterizer)
    monkeypatch.setattr(photobake_mesh, "_load_photo_camera", lambda source, camera, *args: (
        rgb.reshape(1, camera.image_height, camera.image_width, 3), None, camera))

    native = texture.compute_textured_mesh_for_sugar_mesh(sugar, square_size=square_size)
    photo = photobake_mesh.bake_photo_texture(surface, cameras, "unused", "cpu", square_size)

    assert atlas_calls == [(2, square_size, "cpu"), (2, square_size, "cpu")]
    assert len(accumulation_calls) == len(sizes)
    face_colors, face_count, texels, texel_count = accumulation_calls[-1]
    assert torch.equal(face_colors[0], torch.tensor([1., 1., 0.]) * len(sizes))
    assert face_count[0].item() == 2 * len(sizes)
    assert texel_count.max().item() == 2 * len(sizes)
    assert torch.equal(texels[texel_count[..., 0] > 0], torch.tensor([[1., 1., 0.]]) * len(sizes))
    assert torch.equal(native.textures.verts_uvs_padded(), photo.textures.verts_uvs_padded())
    assert torch.equal(native.textures.faces_uvs_padded(), photo.textures.faces_uvs_padded())
    _, _, face_map, _ = build_square_atlas(2, square_size, "cpu")
    native_map = native.textures.maps_padded()[0].flip(0)
    assert torch.equal(native_map[face_map == 1], torch.full_like(native_map[face_map == 1], 0.5))


def test_native_guard_precedes_model_access_or_allocation(monkeypatch):
    from sugar_extractors.texture import compute_textured_mesh_for_sugar_mesh

    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: pytest.fail("allocated too soon"))
    with pytest.raises(ValueError, match=">= 5"):
        compute_textured_mesh_for_sugar_mesh(None, square_size=4)


def test_photo_guard_precedes_mesh_access_or_allocation(monkeypatch):
    from photobake_mesh import bake_photo_texture

    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: pytest.fail("allocated too soon"))
    with pytest.raises(ValueError, match=">= 5"):
        bake_photo_texture(None, None, "unused", "cpu", square_size=4)