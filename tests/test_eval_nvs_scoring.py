"""End-to-end scoring behaviour of the novel-view harness."""
import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")

from augenblick.core.errors import SceneError  # noqa: E402
from augenblick.eval import nvs  # noqa: E402

STEMS = ["camera1_IMG_0001", "camera1_IMG_0009"]
HEIGHT, WIDTH = 48, 64
MARGIN = 12


@pytest.fixture(autouse=True)
def _shared_lpips(monkeypatch):
    """Build the LPIPS backbone once for the module instead of once per scored scene."""
    from augenblick.eval import metrics

    if not hasattr(_shared_lpips, "instance"):
        _shared_lpips.instance = metrics.Lpips()
    monkeypatch.setattr(nvs.metrics, "Lpips", lambda *a, **k: _shared_lpips.instance)


def _mask_array():
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[MARGIN:HEIGHT - MARGIN, MARGIN:WIDTH - MARGIN] = 255
    return mask


def _write(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _build_scene(tmp_path, background_value, seed=0):
    """Write renders, photographs and masks; the photographs get the given background."""
    rng = np.random.default_rng(seed)
    test_dir = tmp_path / "ours_30000"
    masks_dir = tmp_path / "masks"
    mask = _mask_array()
    inside = mask > 0

    for position, stem in enumerate(STEMS):
        render = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        truth = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        render[~inside] = 0                       # backends render onto black
        truth[~inside] = background_value         # the photograph's real surround
        _write(test_dir / "renders" / f"{position:05d}.png", render)
        _write(test_dir / "gt" / f"{position:05d}.png", truth)
        _write(masks_dir / f"{stem}.png", mask)
    return test_dir, masks_dir


def _score(tmp_path, background_value, seed=0):
    test_dir, masks_dir = _build_scene(tmp_path, background_value, seed)
    out = tmp_path / "nvs_metrics.json"
    return nvs.score(test_dir, masks_dir, STEMS, out)


def test_scores_do_not_depend_on_the_photograph_background(tmp_path):
    """A bright, a mid-grey and a black surround must all give the same score."""
    bright = _score(tmp_path / "bright", 240)
    grey = _score(tmp_path / "grey", 128)
    black = _score(tmp_path / "black", 0)

    for metric in ("psnr", "ssim", "lpips"):
        assert grey[metric] == pytest.approx(bright[metric], abs=1e-5), metric
        assert black[metric] == pytest.approx(bright[metric], abs=1e-5), metric


def test_a_background_only_change_cannot_improve_the_score(tmp_path):
    """Making the surround match perfectly must not lift the score."""
    mismatched = _score(tmp_path / "mismatched", 240)
    matched = _score(tmp_path / "matched", 0)
    assert matched["psnr"] == pytest.approx(mismatched["psnr"], abs=1e-5)


def test_the_metric_domain_is_recorded(tmp_path):
    """A metrics file must say which convention produced it, or it cannot be read later."""
    result = _score(tmp_path, 240)
    assert result["metric_domain"] == "mask"
    written = json.loads((tmp_path / "nvs_metrics.json").read_text())
    assert written["metric_domain"] == "mask"
    assert written["n_masked"] == len(STEMS)


def test_an_empty_mask_is_skipped_rather_than_scored(tmp_path):
    """An empty mask has no domain, and a masked mean over it would report a perfect match."""
    test_dir, masks_dir = _build_scene(tmp_path, 240)
    _write(masks_dir / f"{STEMS[0]}.png", np.zeros((HEIGHT, WIDTH), dtype=np.uint8))

    result = nvs.score(test_dir, masks_dir, STEMS, tmp_path / "metrics.json")
    assert result["skipped"] == [STEMS[0]]
    assert result["n_test"] == len(STEMS) - 1


def test_a_resolution_mismatch_is_an_error(tmp_path):
    """Silently resizing would compare different pixels and still produce a number."""
    test_dir, masks_dir = _build_scene(tmp_path, 240)
    _write(test_dir / "gt" / "00000.png",
           np.zeros((HEIGHT * 2, WIDTH * 2, 3), dtype=np.uint8))
    with pytest.raises(SceneError, match="but ground truth is"):
        nvs.score(test_dir, masks_dir, STEMS, tmp_path / "metrics.json")


def test_identical_renders_and_photographs_score_perfectly(tmp_path):
    """A sanity anchor: a perfect reconstruction must reach the ceiling of each metric."""
    rng = np.random.default_rng(5)
    test_dir = tmp_path / "ours_30000"
    masks_dir = tmp_path / "masks"
    mask = _mask_array()

    for position, stem in enumerate(STEMS):
        image = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        _write(test_dir / "renders" / f"{position:05d}.png", image)
        _write(test_dir / "gt" / f"{position:05d}.png", image)
        _write(masks_dir / f"{stem}.png", mask)

    result = nvs.score(test_dir, masks_dir, STEMS, tmp_path / "metrics.json")
    assert result["ssim"] == pytest.approx(1.0, abs=1e-4)
    assert result["lpips"] == pytest.approx(0.0, abs=1e-4)
    assert result["psnr"] > 100
