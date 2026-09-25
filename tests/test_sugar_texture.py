"""Focused tests for SuGaR's direct-photo texture-baking helpers."""
import sys
from pathlib import Path

import pytest
import torch


SUGAR_DIR = Path(__file__).resolve().parents[1] / "src" / "libs" / "sugar"
sys.path.insert(0, str(SUGAR_DIR))
sys.path.insert(0, str(SUGAR_DIR / "gaussian_splatting"))

from sugar_extractors.texture import _accumulate_samples, _find_source_image


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