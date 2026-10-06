"""Focused tests for SuGaR's direct-photo texture-baking helpers."""
import sys
from pathlib import Path

import pytest
import torch


SUGAR_DIR = Path(__file__).resolve().parents[1] / "src" / "libs" / "sugar"
sys.path.insert(0, str(SUGAR_DIR))
sys.path.insert(0, str(SUGAR_DIR / "gaussian_splatting"))

from sugar_extractors.texture import (
    _accumulate_camera_support,
    _accumulate_samples,
    _coverage_summary,
    _find_source_image,
    _uv_texel_mask,
)


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


def test_uv_texel_mask_excludes_pixels_outside_triangle():
    verts_uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    faces_uv = torch.tensor([[0, 1, 2]])

    valid = _uv_texel_mask(verts_uv, faces_uv, texture_size=4)

    assert valid[0, 0]
    assert not valid[3, 3]
    assert int(valid.sum()) > 0


def test_uv_texel_mask_handles_triangles_larger_than_five_texels():
    verts_uv = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    faces_uv = torch.tensor([[0, 1, 2]])

    valid = _uv_texel_mask(verts_uv, faces_uv, texture_size=10)

    assert valid[0, 0]
    assert not valid[9, 9]
    assert int(valid.sum()) > 10


def test_camera_support_counts_each_texel_once_per_camera():
    support = torch.zeros((2, 2), dtype=torch.int32)
    camera_pixels = torch.tensor([[1, 0], [1, 0], [0, 1]])

    _accumulate_camera_support(support, camera_pixels)
    _accumulate_camera_support(support, camera_pixels[:1])

    assert torch.equal(support, torch.tensor([[0, 2], [1, 0]], dtype=torch.int32))


def test_coverage_summary_uses_only_valid_uv_texels():
    valid_uv = torch.tensor([[True, True, False], [True, True, False]])
    support = torch.tensor([[0, 1, 9], [2, 3, 9]], dtype=torch.int32)

    summary = _coverage_summary(valid_uv, support)

    assert summary["valid_texels"] == 4
    assert summary["covered_texels"] == 3
    assert summary["coverage"] == 0.75
    assert summary["support_distribution"]["zero"]["count"] == 1
    assert summary["support_distribution"]["one"]["count"] == 1
    assert summary["support_distribution"]["two_or_more"]["count"] == 2
    assert summary["maximum_camera_support"] == 3


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