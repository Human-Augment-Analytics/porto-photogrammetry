"""Masking stage: the mask-writing contract, helpers, run loop, and CLI routing.

CPU-only, no GPU, no model download — rembg, cv2.grabCut and SAM 3 are never run on real
data, so the suite stays seconds-long on a login node.
"""
import numpy as np
import pytest
from PIL import Image

from augenblick.cli.main import build_parser
from augenblick.core.errors import SceneError
from augenblick.core.registry import MASK_REGISTRY
from augenblick.masking.base import (
    MaskMethod,
    foreground_fraction,
    postprocess_mask,
    write_mask,
)
from augenblick.masking.sam3 import Sam3Config, Sam3Mask
from augenblick.masking.threshold import ThresholdConfig, ThresholdMask, _decide_polarity

import augenblick.masking  # noqa: F401  (fires registration)


def test_registry_holds_every_method():
    assert set(MASK_REGISTRY) == {"rembg", "sam3", "threshold"}


def test_write_mask_round_trips_through_consumer_reader(tmp_path):
    # Consumers open the PNG, convert to L, threshold at >127. Round-trip must be exact.
    original = np.zeros((16, 24), dtype=bool)
    original[4:12, 6:18] = True
    dest = tmp_path / "m.png"
    write_mask(original, dest)

    arr = np.asarray(Image.open(dest).convert("L"))
    assert ((arr > 127) == original).all()


def test_write_mask_emits_mode_L(tmp_path):
    dest = tmp_path / "m.png"
    write_mask(np.ones((8, 8), dtype=bool), dest)
    assert Image.open(dest).mode == "L"


def test_write_mask_emits_binary_values(tmp_path):
    dest = tmp_path / "m.png"
    m = np.zeros((8, 8), dtype=bool)
    m[2:6, 2:6] = True
    write_mask(m, dest)
    assert set(np.unique(np.asarray(Image.open(dest)))) <= {0, 255}


def test_postprocess_keeps_largest_component():
    m = np.zeros((20, 20), dtype=bool)
    m[2:10, 2:10] = True   # large blob
    m[15, 15] = True       # speck
    out = postprocess_mask(m, keep_largest=True, fill_holes=False)
    assert out[5, 5]
    assert not out[15, 15]


def test_postprocess_fills_holes():
    m = np.zeros((20, 20), dtype=bool)
    m[3:17, 3:17] = True
    m[8:12, 8:12] = False  # enclosed hole
    out = postprocess_mask(m, keep_largest=False, fill_holes=True)
    assert out[10, 10]


def test_foreground_fraction_on_known_array():
    m = np.zeros((10, 10), dtype=bool)
    m[:, :5] = True
    assert foreground_fraction(m) == pytest.approx(0.5)


def test_validate_raises_on_missing_dir(tmp_path):
    method = ThresholdMask(ThresholdConfig())
    with pytest.raises(SceneError, match="no images directory"):
        method.validate(tmp_path / "nope")


def test_validate_raises_on_no_matching_suffix(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    method = ThresholdMask(ThresholdConfig())
    with pytest.raises(SceneError, match="accepted suffix"):
        method.validate(tmp_path)


class _StubMask(MaskMethod):
    """A masking method that returns a fixed centred rectangle, no CV libs needed."""

    name = "_stub"
    title = "Masking (stub)"
    config_cls = ThresholdConfig

    def mask_for(self, image_path):
        with Image.open(image_path) as im:
            w, h = im.size
        m = np.zeros((h, w), dtype=bool)
        m[h // 4:3 * h // 4, w // 4:3 * w // 4] = True
        return m


def _make_images(dirpath, n, size=(40, 30)):
    dirpath.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", size, (10, 10, 10)).save(dirpath / f"img_{i:03d}.jpg")


def test_run_builds_scene_shaped_output(tmp_path):
    images = tmp_path / "in"
    _make_images(images, 3)
    out = tmp_path / "out"

    result = _StubMask(ThresholdConfig()).run(images, out)

    assert (out / "images").is_symlink()
    assert (out / "images").resolve() == images.resolve()
    assert (out / "masks").is_dir()
    assert result.num_images == 3
    assert result.num_written == 3
    assert result.num_failed == 0
    masks = sorted((out / "masks").glob("*.png"))
    assert [m.name for m in masks] == ["img_000.png", "img_001.png", "img_002.png"]
    # Mask dims equal source dims.
    assert np.asarray(Image.open(masks[0])).shape == (30, 40)


def test_only_missing_reuses_existing_masks(tmp_path):
    images = tmp_path / "in"
    _make_images(images, 3)
    out = tmp_path / "out"
    (out / "masks").mkdir(parents=True)
    # Pre-seed 2 of 3 masks.
    for stem in ("img_000", "img_001"):
        write_mask(np.ones((30, 40), dtype=bool), out / "masks" / f"{stem}.png")

    result = _StubMask(ThresholdConfig(only_missing=True)).run(images, out)
    assert result.num_reused == 2
    assert result.num_written == 1


def test_run_rejects_out_of_bounds_foreground(tmp_path):
    images = tmp_path / "in"
    _make_images(images, 1)
    out = tmp_path / "out"

    class _AllForeground(_StubMask):
        def mask_for(self, image_path):
            with Image.open(image_path) as im:
                w, h = im.size
            return np.ones((h, w), dtype=bool)

    result = _AllForeground(ThresholdConfig()).run(images, out)
    assert result.num_written == 0
    assert result.num_failed == 1
    assert not any((out / "masks").glob("*.png"))


def test_run_refuses_conflicting_images_symlink(tmp_path):
    images = tmp_path / "in"
    _make_images(images, 1)
    other = tmp_path / "other"
    other.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    (out / "images").symlink_to(other)

    with pytest.raises(SceneError, match="points elsewhere"):
        _StubMask(ThresholdConfig()).run(images, out)


def test_polarity_heuristic_picks_each_side():
    # Dark specimen on light background -> "light-background".
    light_bg = np.full((64, 64), 220, dtype=np.uint8)
    light_bg[24:40, 24:40] = 20
    assert _decide_polarity(light_bg, border_px=8) == "light-background"

    # Light specimen on dark background -> "dark-background".
    dark_bg = np.full((64, 64), 20, dtype=np.uint8)
    dark_bg[24:40, 24:40] = 220
    assert _decide_polarity(dark_bg, border_px=8) == "dark-background"


def test_parser_accepts_mask_with_images():
    args = build_parser().parse_args(
        ["mask", "threshold", "--images", "x", "--output", "y"])
    assert args.stage == "mask"
    assert str(args.input_dir) == "x"


def test_parser_rejects_mask_with_scene():
    # --images is supplied so the only thing left to fail on is the unrecognised --scene.
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["mask", "threshold", "--images", "x", "--scene", "x", "--output", "y"])


SAM3_SIZE = (6, 8)  # (height, width)


class _FakeProcessor:
    """Stands in for Sam3Processor: returns fixed detections, so no checkpoint is needed."""

    def __init__(self, masks, scores):
        self._masks, self._scores = masks, scores

    def set_image(self, image):
        return {}

    def set_text_prompt(self, prompt, state):
        return {**state, "masks": self._masks, "scores": self._scores}


def _rect(top, left):
    """A 2x2 foreground block, so two of them can be placed disjointly."""
    m = np.zeros(SAM3_SIZE, dtype=bool)
    m[top:top + 2, left:left + 2] = True
    return m


def _stub_sam3(torch, masks, scores, **config):
    """A Sam3Mask whose processor is already built, bypassing the GPU and the download."""
    method = Sam3Mask(Sam3Config(device="cpu", **config))
    # Upstream hands back bool [N, 1, H, W]; the method flattens that itself.
    method._processor = _FakeProcessor(
        torch.from_numpy(np.stack(masks)[:, None]), torch.tensor(scores))
    return method


def _sam3_image(tmp_path):
    path = tmp_path / "img.jpg"
    Image.new("RGB", SAM3_SIZE[::-1], (10, 10, 10)).save(path)
    return path


def test_sam3_unions_every_surviving_detection(tmp_path):
    """A specimen split across detections must come back whole, not as the top-scoring one."""
    torch = pytest.importorskip("torch")
    left, right = _rect(1, 1), _rect(4, 5)

    mask = _stub_sam3(torch, [left, right], [0.9, 0.8]).mask_for(_sam3_image(tmp_path))

    assert (mask == (left | right)).all()


def test_sam3_max_detections_keeps_the_highest_scoring(tmp_path):
    """The cutoff ranks by score, not by the order the detector returned them."""
    torch = pytest.importorskip("torch")
    weak, strong = _rect(1, 1), _rect(4, 5)

    method = _stub_sam3(torch, [weak, strong], [0.4, 0.9], max_detections=1)
    mask = method.mask_for(_sam3_image(tmp_path))

    assert (mask == strong).all()


def test_sam3_raises_when_nothing_matches(tmp_path):
    """An empty result names the prompt and threshold instead of failing on a reshape."""
    torch = pytest.importorskip("torch")
    method = Sam3Mask(Sam3Config(device="cpu", prompt="vertebra"))
    method._processor = _FakeProcessor(
        torch.zeros((0, 1, *SAM3_SIZE), dtype=torch.bool), torch.zeros(0))

    with pytest.raises(SceneError, match="vertebra"):
        method.mask_for(_sam3_image(tmp_path))
