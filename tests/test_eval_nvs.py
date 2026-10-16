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


def test_score_output_discovers_both_sugar_products(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from augenblick.eval import nvs

    for target in ("test_splat", "test_mesh"):
        _make_test_dir(tmp_path / target, STEMS)
        (tmp_path / target / "ours_7000").mkdir()
    manifest = tmp_path / "sugar" / "run_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"resolution": -1, "texture_square_size": 8}))
    scene = SimpleNamespace(root=tmp_path / "scene", masks_dir=tmp_path / "masks")
    monkeypatch.setattr(nvs, "resolve_test_stems", lambda scene: STEMS)
    calls = []

    def fake_score(test_dir, masks_dir, stems, out_path, extra):
        calls.append((test_dir, stems, out_path, extra))
        return {"psnr": 20, "ssim": 0.8, "lpips": 0.2, "n_test": 3}

    monkeypatch.setattr(nvs, "score", fake_score)
    result = nvs.score_output(scene, tmp_path, iteration=7000)
    assert [call[0] for call in calls] == [
        tmp_path / "test_splat" / "ours_7000",
        tmp_path / "test_mesh" / "ours_7000",
    ]
    assert calls[1][3]["texture_square_size"] == 8
    assert all(call[3]["resolution"] == -1 for call in calls)
    assert result["primary_target"] == "refined_splat"
    assert result["psnr"] == result["refined_splat"]["psnr"]
    assert "textured_mesh" in json.loads((tmp_path / "nvs_metrics.json").read_text())


def test_score_output_requires_both_sugar_products(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from augenblick.eval import nvs

    (tmp_path / "test_mesh").mkdir()
    monkeypatch.setattr(nvs, "resolve_test_stems", lambda scene: STEMS)
    with pytest.raises(SceneError, match="no held-out renders"):
        nvs.score_output(SimpleNamespace(root=tmp_path, masks_dir=tmp_path), tmp_path)


def test_sugar_backend_uses_shared_scorer(tmp_path, monkeypatch):
    from augenblick.eval import nvs
    from augenblick.reconstruction.sugar import SugarBackend, SugarConfig

    scene = object()
    expected = {"primary_target": "refined_splat"}
    calls = []

    def fake_score(scene_arg, output_arg):
        calls.append((scene_arg, output_arg))
        return expected

    monkeypatch.setattr(nvs, "score_sugar_output", fake_score)
    assert SugarBackend(SugarConfig()).evaluate(scene, tmp_path) is expected
    assert calls == [(scene, tmp_path)]
