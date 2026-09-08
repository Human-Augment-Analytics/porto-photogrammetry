"""Matching a backend's renders back to the photographs they came from, across naming schemes."""
import pytest

from augenblick.core.errors import SceneError
from augenblick.eval.nvs import find_test_dir, pair_views

STEMS = ["camera1_IMG_0001", "camera1_IMG_0009", "camera2_IMG_0017"]


def _make_test_dir(tmp_path, names):
    """Create a renders/ and gt/ pair populated with the given file names."""
    test_dir = tmp_path / "ours_30000"
    for sub in ("renders", "gt"):
        (test_dir / sub).mkdir(parents=True)
        for name in names:
            (test_dir / sub / f"{name}.png").touch()
    return test_dir


def test_positional_names_resolve_through_the_split(tmp_path):
    # 2DGS and Gaussian Wrapping export by position, so the ordered split is the only link.
    test_dir = _make_test_dir(tmp_path, ["00000", "00001", "00002"])
    assert [stem for stem, _, _ in pair_views(test_dir, STEMS)] == STEMS


def test_image_names_are_used_directly(tmp_path):
    # PGSR exports by image name, which needs no lookup.
    test_dir = _make_test_dir(tmp_path, STEMS)
    assert sorted(stem for stem, _, _ in pair_views(test_dir, STEMS)) == sorted(STEMS)


def test_pairs_point_at_matching_files(tmp_path):
    test_dir = _make_test_dir(tmp_path, ["00000", "00001", "00002"])
    for stem, render, truth in pair_views(test_dir, STEMS):
        assert render.name == truth.name
        assert render.parent.name == "renders" and truth.parent.name == "gt"


def test_missing_ground_truth_is_an_error(tmp_path):
    test_dir = _make_test_dir(tmp_path, ["00000"])
    (test_dir / "gt" / "00000.png").unlink()
    with pytest.raises(SceneError, match="no ground truth"):
        pair_views(test_dir, STEMS)


def test_more_renders_than_held_out_views_is_an_error(tmp_path):
    # A scene and a model that disagree on the split would otherwise score silently wrong.
    test_dir = _make_test_dir(tmp_path, ["00000", "00001", "00002", "00003"])
    with pytest.raises(SceneError, match="the scene and the model disagree"):
        pair_views(test_dir, STEMS)


def test_empty_render_dir_is_an_error(tmp_path):
    test_dir = _make_test_dir(tmp_path, [])
    with pytest.raises(SceneError, match="no renders"):
        pair_views(test_dir, STEMS)


def test_find_test_dir_picks_the_highest_iteration(tmp_path):
    for iteration in (7000, 30000):
        (tmp_path / f"ours_{iteration}").mkdir()
    assert find_test_dir(tmp_path) == tmp_path / "ours_30000"
    assert find_test_dir(tmp_path, 7000) == tmp_path / "ours_7000"


def test_find_test_dir_without_renders_is_an_error(tmp_path):
    with pytest.raises(SceneError, match="no held-out renders"):
        find_test_dir(tmp_path)
