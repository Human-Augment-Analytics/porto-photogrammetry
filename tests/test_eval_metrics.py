"""Analytic and invariance tests for the novel-view metrics."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from augenblick.eval import metrics  # noqa: E402

WINDOW_RADIUS = metrics.SSIM_WINDOW_SIZE // 2


def _rng():
    return np.random.default_rng(20260915)


def _image(rng, height=64, width=80):
    return torch.from_numpy(rng.random((3, height, width), dtype=np.float64)).float()


def _centre_mask(height=64, width=80, margin=16):
    mask = torch.zeros(1, height, width)
    mask[:, margin:height - margin, margin:width - margin] = 1.0
    return mask


def _erode(mask, radius):
    """Shrink a rectangular mask by `radius`, so SSIM windows stay inside it."""
    eroded = torch.zeros_like(mask)
    rows = torch.nonzero(mask[0].any(dim=1)).flatten()
    cols = torch.nonzero(mask[0].any(dim=0)).flatten()
    eroded[:, rows[0] + radius:rows[-1] + 1 - radius,
           cols[0] + radius:cols[-1] + 1 - radius] = 1.0
    return eroded


def test_psnr_constant_offset_is_closed_form():
    """Two images differing by a constant d have PSNR exactly 20*log10(1/d)."""
    rng = _rng()
    truth = _image(rng) * 0.5
    for delta in (0.5, 0.1, 0.01, 0.001):
        pred = truth + delta
        assert metrics.psnr(pred, truth) == pytest.approx(20 * np.log10(1 / delta), abs=1e-3)


def test_psnr_of_identical_images_is_the_clamp():
    """Identical images give the value implied by the 1e-12 MSE clamp, not an inf or a nan."""
    rng = _rng()
    image = _image(rng)
    assert metrics.psnr(image, image) == pytest.approx(20 * np.log10(1 / np.sqrt(1e-12)), abs=1e-3)


def test_psnr_masked_offset_ignores_the_background_entirely():
    """With a mask, only the offset inside the mask sets the score."""
    rng = _rng()
    truth = _image(rng) * 0.5
    mask = _centre_mask()
    pred = truth.clone()
    pred[:, mask[0] > 0] += 0.1
    assert metrics.psnr(pred, truth, mask) == pytest.approx(20 * np.log10(1 / 0.1), abs=1e-3)


def test_psnr_is_exactly_invariant_to_the_background():
    """Randomising everything outside the mask must not move masked PSNR at all."""
    rng = _rng()
    truth, pred = _image(rng), _image(rng)
    mask = _centre_mask()
    before = metrics.psnr(pred, truth, mask)

    outside = mask.expand_as(truth) == 0
    truth_b, pred_b = truth.clone(), pred.clone()
    truth_b[outside] = torch.from_numpy(rng.random(int(outside.sum()))).float()
    pred_b[outside] = torch.from_numpy(rng.random(int(outside.sum()))).float()

    assert metrics.psnr(pred_b, truth_b, mask) == pytest.approx(before, abs=1e-6)


def test_psnr_with_an_all_ones_mask_equals_the_unmasked_path():
    rng = _rng()
    truth, pred = _image(rng), _image(rng)
    ones = torch.ones(1, truth.shape[1], truth.shape[2])
    assert metrics.psnr(pred, truth, ones) == pytest.approx(metrics.psnr(pred, truth), abs=1e-5)


def test_ssim_of_identical_images_is_one():
    rng = _rng()
    image = _image(rng)
    assert metrics.ssim(image, image) == pytest.approx(1.0, abs=1e-5)


def test_ssim_is_symmetric_and_bounded():
    rng = _rng()
    truth, pred = _image(rng), _image(rng)
    forward, backward = metrics.ssim(pred, truth), metrics.ssim(truth, pred)
    assert forward == pytest.approx(backward, abs=1e-6)
    assert -1.0 - 1e-6 <= forward <= 1.0 + 1e-6


def test_ssim_with_an_all_ones_mask_equals_the_unmasked_path():
    rng = _rng()
    truth, pred = _image(rng), _image(rng)
    ones = torch.ones(1, truth.shape[1], truth.shape[2])
    assert metrics.ssim(pred, truth, ones) == pytest.approx(metrics.ssim(pred, truth), abs=1e-5)


def test_ssim_background_reaches_exactly_one_window_radius_inside():
    """Scored on an eroded mask, SSIM is invariant to the background; on the full mask it is not."""
    rng = _rng()
    truth, pred = _image(rng), _image(rng)
    mask = _centre_mask()
    eroded = _erode(mask, WINDOW_RADIUS)

    outside = mask.expand_as(truth) == 0
    truth_b, pred_b = truth.clone(), pred.clone()
    truth_b[outside] = 0.0
    pred_b[outside] = 0.0

    assert metrics.ssim(pred_b, truth_b, eroded) == pytest.approx(
        metrics.ssim(pred, truth, eroded), abs=1e-6)
    assert metrics.ssim(pred_b, truth_b, mask) != pytest.approx(
        metrics.ssim(pred, truth, mask), abs=1e-6)


def test_mask_bbox_is_tight_and_half_open():
    mask = torch.zeros(1, 20, 30)
    mask[0, 5:12, 7:20] = 1.0
    assert metrics.mask_bbox(mask) == (5, 12, 7, 20)


def test_mask_bbox_of_an_empty_mask_is_the_whole_frame():
    """An empty mask must not raise or return a degenerate crop."""
    assert metrics.mask_bbox(torch.zeros(1, 20, 30)) == (0, 20, 0, 30)


def test_psnr_matches_the_3dgs_per_channel_definition():
    """3DGS averages three per-channel PSNRs; it is not one PSNR over a global MSE."""
    rng = _rng()
    truth = _image(rng) * 0.5
    pred = truth.clone()
    for channel, delta in enumerate((0.02, 0.05, 0.2)):
        pred[channel] += delta

    per_channel = np.mean([20 * np.log10(1 / d) for d in (0.02, 0.05, 0.2)])
    global_mse = np.mean(np.square([0.02, 0.05, 0.2]))
    global_psnr = 20 * np.log10(1 / np.sqrt(global_mse))

    assert metrics.psnr(pred, truth) == pytest.approx(per_channel, abs=1e-3)
    assert abs(per_channel - global_psnr) > 1.0
