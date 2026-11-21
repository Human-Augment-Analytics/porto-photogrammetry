"""Matching a backend's renders back to the photographs they came from, across naming schemes."""
import pytest

from augenblick.core.errors import SceneError
from augenblick.eval.nvs import find_test_dir, pair_views, render_iteration

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


def test_find_test_dir_accepts_resolution_suffix(tmp_path):
    (tmp_path / "ours_7000").mkdir()
    (tmp_path / "ours_20000_r2.0").mkdir()
    (tmp_path / "ours_notes.txt").write_text("")
    assert find_test_dir(tmp_path) == tmp_path / "ours_20000_r2.0"


def test_render_iteration_ignores_suffixes():
    assert render_iteration("ours_30000") == 30000
    assert render_iteration("ours_2000_r3.9") == 2000
    assert render_iteration("gt") is None


def test_find_test_dir_without_renders_is_an_error(tmp_path):
    with pytest.raises(SceneError, match="no held-out renders"):
        find_test_dir(tmp_path)


def test_canonical_scoring_ignores_native_gt_compositing(tmp_path,monkeypatch):
    import numpy as np
    import torch
    from PIL import Image
    from augenblick.eval import nvs
    class Distance:
        def __call__(self,a,b,mask=None): return float(((a-b)**2).mean())
    monkeypatch.setattr(nvs.metrics,'Lpips',lambda:Distance())
    monkeypatch.setattr(nvs.metrics,'Dists',lambda:Distance())
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    images=tmp_path/'images';masks=tmp_path/'masks';images.mkdir();masks.mkdir()
    rgb=np.full((32,48,3),180,dtype=np.uint8)
    Image.fromarray(rgb).save(images/'view.png')
    Image.fromarray(np.full((32,48),200,dtype=np.uint8)).save(masks/'view.png')
    results=[]
    for i in range(2):
        output=tmp_path/str(i)
        for sub in ('renders','gt'):(output/sub).mkdir(parents=True)
        Image.fromarray(rgb).save(output/'renders/view.png')
        # Backends may export raw or mask-composited photographs.
        Image.fromarray(rgb if i==0 else (rgb*.5).astype(np.uint8)).save(output/'gt/view.png')
        results.append(nvs.score(output,masks,['view'],output/'metrics.json',
                                canonical_images=images,require_masks=True,require_complete=True))
    assert results[0]['psnr']==results[1]['psnr']
    assert results[0]['ssim']==results[1]['ssim']
    assert results[0]['views'][0]['canonical_rgb_sha256']==results[1]['views'][0]['canonical_rgb_sha256']


def test_paper_scoring_rejects_raster_resampling(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from PIL import Image
    from augenblick.eval import nvs
    class Distance:
        def __call__(self, a, b, mask=None): return float(((a-b)**2).mean())
    monkeypatch.setattr(nvs.metrics, "Lpips", lambda: Distance())
    monkeypatch.setattr(nvs.metrics, "Dists", lambda: Distance())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    images=tmp_path/"images";masks=tmp_path/"masks"
    for d in (images,masks,tmp_path/"renders",tmp_path/"gt"):d.mkdir()
    Image.new("RGB",(64,32),"white").save(images/"view.png")
    Image.new("L",(64,32),255).save(masks/"view.png")
    for d in ("renders","gt"):Image.new("RGB",(64,33),"white").save(tmp_path/d/"view.png")
    with pytest.raises(SceneError, match="must match canonical raster"):
        nvs.score(tmp_path,masks,["view"],tmp_path/"scores.json",canonical_images=images,
                  require_native_raster=True,require_complete=True)


def test_explicit_iteration_accepts_geo_suffix(tmp_path):
    target = tmp_path / "ours_20000_r3.9";target.mkdir()
    assert find_test_dir(tmp_path, 20000) == target


def test_ambiguous_same_iteration_requires_variant(tmp_path):
    for n in ("ours_20000_r3.9", "ours_20000_direct_eval"):(tmp_path / n).mkdir()
    for iteration in (None, 20000):
        with pytest.raises(SceneError, match="ambiguous render variants"):
            find_test_dir(tmp_path, iteration)
    assert find_test_dir(tmp_path, 20000, "ours_20000_direct_eval").name == "ours_20000_direct_eval"
    with pytest.raises(SceneError, match="does not match"):
        find_test_dir(tmp_path, 10000, "ours_20000_direct_eval")
    with pytest.raises(SceneError, match="direct subdirectory"):
        find_test_dir(tmp_path, render_variant="../ours_20000_r3.9")
