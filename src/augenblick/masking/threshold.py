"""Classical background separation: Otsu / GrabCut with per-image polarity auto-pick.

Depends only on packages already pinned in the shared envs (cv2, skimage, scipy, numpy,
Pillow). Runs on CPU. All heavy imports are function-local.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

from augenblick.core.registry import register_mask
from augenblick.masking.base import MaskCommonConfig, MaskMethod

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ThresholdConfig(MaskCommonConfig):
    """Classical background-separation."""

    mode: Literal["otsu", "grabcut"] = field(default="otsu", metadata={
        "help": "otsu: global luminance threshold; grabcut: iterative graph cut"})
    polarity: Literal["auto", "dark-background", "light-background"] = field(
        default="auto", metadata={
            "help": "Which side is background. Turntable backdrops are typically lighter "
                    "but occasionally darker; 'auto' picks per image via border-vs-centre "
                    "luminance"})
    blur_px: int = field(default=3, metadata={
        "help": "Gaussian blur radius before thresholding (0 disables)"})
    grabcut_iters: int = field(default=5, metadata={"help": "GrabCut iterations"})
    border_px: int = field(default=32, metadata={
        "help": "Border inset for GrabCut initial rectangle"})
    downscale: int = field(default=4, metadata={
        "help": "Segment at 1/N res then nearest-upsample (1 disables)"})


def _decide_polarity(luma, border_px: int) -> str:
    """Pick 'dark-background' or 'light-background' from border vs centre luminance."""
    import numpy as np

    h, w = luma.shape
    b = max(1, min(border_px, min(h, w) // 4))
    border = np.concatenate([
        luma[:b, :].ravel(), luma[-b:, :].ravel(),
        luma[:, :b].ravel(), luma[:, -b:].ravel()])
    ch, cw = h // 2, w // 2
    centre = luma[max(0, ch - b):ch + b, max(0, cw - b):cw + b]
    return "light-background" if border.mean() > centre.mean() else "dark-background"


@register_mask
class ThresholdMask(MaskMethod):
    """Otsu or GrabCut background separation; no learned model."""

    name: ClassVar[str] = "threshold"
    title: ClassVar[str] = "Masking (threshold)"
    config_cls: ClassVar[type] = ThresholdConfig

    def mask_for(self, image_path: Path):
        import cv2
        import numpy as np
        from PIL import Image
        from skimage.filters import threshold_otsu

        cfg = self.config
        with Image.open(image_path) as im:
            im_rgb = im.convert("RGB")
            w0, h0 = im_rgb.size
            if cfg.downscale > 1:
                small = im_rgb.resize((max(1, w0 // cfg.downscale),
                                        max(1, h0 // cfg.downscale)),
                                       Image.BILINEAR)
            else:
                small = im_rgb
            rgb = np.asarray(small)

        luma = np.asarray(Image.fromarray(rgb).convert("L"))
        if cfg.blur_px > 0:
            k = 2 * cfg.blur_px + 1
            luma = cv2.GaussianBlur(luma, (k, k), 0)

        polarity = cfg.polarity
        if polarity == "auto":
            polarity = _decide_polarity(luma, cfg.border_px)

        if cfg.mode == "otsu":
            t = threshold_otsu(luma)
            mask_small = luma < t if polarity == "light-background" else luma > t
        elif cfg.mode == "grabcut":
            gc_mask = np.full(luma.shape, cv2.GC_PR_BGD, dtype=np.uint8)
            h, w = luma.shape
            b = min(cfg.border_px, min(h, w) // 4)
            rect = (b, b, max(1, w - 2 * b), max(1, h - 2 * b))
            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)
            cv2.grabCut(rgb, gc_mask, rect, bgd, fgd, cfg.grabcut_iters,
                        cv2.GC_INIT_WITH_RECT)
            mask_small = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        else:
            raise ValueError(f"unknown threshold mode {cfg.mode!r}")

        if cfg.downscale > 1:
            mask_img = Image.fromarray(mask_small.astype(np.uint8) * 255, mode="L").resize(
                (w0, h0), Image.NEAREST)
            mask = np.asarray(mask_img) > 127
        else:
            mask = mask_small
        return mask
