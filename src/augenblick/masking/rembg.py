"""Learned U^2-Net matting via rembg's ONNX runtime; the default masking method.

Runs on GPU when onnxruntime-gpu can load its CUDA provider; falls back to CPU with a
~4x slowdown. All heavy imports are function-local so `augenblick mask --list` on a
login node does not require rembg.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

from augenblick.core.registry import register_mask
from augenblick.masking.base import MaskCommonConfig, MaskMethod

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RembgConfig(MaskCommonConfig):
    """U^2-Net matting parameters, run through rembg's ONNX session."""

    model: str = field(default="isnet-general-use", metadata={
        "help": "rembg model, e.g. u2net, isnet-general-use, birefnet-general"})
    alpha_threshold: int = field(default=127, metadata={
        "help": "Alpha above which a pixel is foreground; matches the consumers' >127"})
    providers: Optional[str] = field(default=None, metadata={
        "help": "Comma-separated ONNX providers; default lets onnxruntime pick "
                "(CUDAExecutionProvider first when GPU is available)"})
    batch_log_every: int = field(default=25, metadata={
        "help": "Log progress every N images"})


@register_mask
class RembgMask(MaskMethod):
    """rembg salient-object matting (U^2-Net / IS-Net / BiRefNet)."""

    name: ClassVar[str] = "rembg"
    title: ClassVar[str] = "Masking (rembg)"
    config_cls: ClassVar[type] = RembgConfig

    def __init__(self, config: RembgConfig):
        super().__init__(config)
        self._session = None
        self._count = 0

    def _get_session(self):
        if self._session is not None:
            return self._session
        from rembg import new_session
        import onnxruntime as ort

        available = ort.get_available_providers()
        if self.config.providers:
            providers = [p.strip() for p in self.config.providers.split(",") if p.strip()]
        else:
            providers = None
        logger.info(f"onnxruntime providers available: {available}")
        self._session = new_session(self.config.model, providers=providers)

        # get_available_providers() lists what ORT was compiled with, not what loaded, only the live session reports the truth.
        active = getattr(self._session, "inner_session", None)
        active = active.get_providers() if active is not None else []
        if "CUDAExecutionProvider" not in active:
            logger.warning(
                f"rembg running on CPU (~4x slower) — active providers: {active}; "
                "CUDA provider did not load. Check that the env's nvidia/*/lib dirs are on "
                "LD_LIBRARY_PATH (pace_slurm/common.sh does this)")
        else:
            logger.info("rembg using CUDAExecutionProvider")
        return self._session

    def mask_for(self, image_path: Path):
        import numpy as np
        from PIL import Image
        from rembg import remove

        session = self._get_session()
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            cut = remove(im, session=session, only_mask=True)
        arr = np.asarray(cut)
        if arr.ndim == 3:
            arr = arr[..., -1]
        mask = arr > self.config.alpha_threshold

        self._count += 1
        every = max(1, self.config.batch_log_every)
        if self._count % every == 0:
            logger.info(f"rembg: masked {self._count} images")
        return mask
