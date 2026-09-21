"""Concept-prompted instance segmentation via SAM 3; masks whatever a text prompt names.

The checkpoint is gated on HuggingFace (facebook/sam3); warm $HF_HOME on the login node with
`hf auth login` before submitting, as compute nodes have no interactive TTY.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

from augenblick.core.errors import SceneError
from augenblick.core.registry import register_mask
from augenblick.masking.base import MaskCommonConfig, MaskMethod

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sam3Config(MaskCommonConfig):
    """SAM 3 concept-prompted segmentation parameters."""

    prompt: str = field(default="skeleton", metadata={
        "help": "Text concept to segment, e.g. 'skeleton', 'fish skull', 'bone'"})
    score_threshold: float = field(default=0.5, metadata={
        "help": "Drop detections scoring below this before the masks are unioned"})
    max_detections: int = field(default=0, metadata={
        "help": "Keep only the N highest-scoring detections (0 = keep all above threshold)"})
    checkpoint_path: Optional[str] = field(default=None, metadata={
        "help": "Local SAM 3 checkpoint; default downloads from the gated HF repo"})
    device: str = field(default="cuda", metadata={"help": "Torch device for the model"})
    batch_log_every: int = field(default=25, metadata={
        "help": "Log progress every N images"})


@register_mask
class Sam3Mask(MaskMethod):
    """SAM 3 concept-prompted segmentation (text prompt, default 'skeleton')."""

    name: ClassVar[str] = "sam3"
    title: ClassVar[str] = "Masking (SAM 3)"
    config_cls: ClassVar[type] = Sam3Config

    def __init__(self, config: Sam3Config):
        super().__init__(config)
        self._processor = None
        self._count = 0

    def header(self, images_dir: Path, output_dir: Path) -> dict[str, object]:
        """Adds the prompt and score threshold to the banner."""
        head = super().header(images_dir, output_dir)
        head["Prompt"] = repr(self.config.prompt)
        head["Score threshold"] = self.config.score_threshold
        return head

    def _get_processor(self):
        if self._processor is not None:
            return self._processor
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # A silent CPU fallback plus --only_missing bakes the slow result into every later
        # rerun, so refuse rather than degrade (the lesson rembg's CPU path already encodes).
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise SceneError(
                f"SAM 3 was asked for device='{self.config.device}' but torch reports no CUDA "
                "device; refusing to run on CPU. Pass --device cpu to override deliberately.")

        logger.info(f"building SAM 3 image model on {self.config.device} "
                    f"(checkpoint: {self.config.checkpoint_path or 'HF facebook/sam3'})")
        model = build_sam3_image_model(
            device=self.config.device,
            checkpoint_path=self.config.checkpoint_path,  # None -> HF download
        )
        # The processor applies confidence_threshold inside its grounding forward, before the
        # surviving masks are interpolated to full resolution - cheaper than filtering after.
        self._processor = Sam3Processor(
            model,
            device=self.config.device,
            confidence_threshold=self.config.score_threshold,
        )
        return self._processor

    def mask_for(self, image_path: Path):
        import numpy as np
        import torch
        from PIL import Image

        processor = self._get_processor()
        # The checkpoint's weights are bfloat16, so float32 input raises "mat1 and mat2 must
        # have the same dtype" on the first matmul. Upstream never documents this; every
        # reference script in sam3/scripts/ opens a bfloat16 autocast before inferring.
        device_type = "cuda" if self.config.device.startswith("cuda") else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                state = processor.set_image(im)
            state = processor.set_text_prompt(self.config.prompt, state)

        masks = state["masks"]
        scores = state["scores"]

        # Upstream returns bool masks already interpolated to the source dims, but the docs
        # do not promise the dtype, so normalise defensively to a bool [N, H, W].
        # Cast to float32 on the torch side first: under bfloat16 autocast these can come back
        # as BFloat16, which numpy has no equivalent for ("unsupported ScalarType BFloat16").
        if isinstance(masks, torch.Tensor):
            if masks.dtype not in (torch.bool, torch.uint8):
                masks = masks.float()
            masks = masks.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.float().detach().cpu().numpy()
        masks = np.asarray(masks)
        scores = np.asarray(scores).reshape(-1)

        # Check this before reshaping: an empty [0, 1, H, W] cannot be reshaped, and the
        # ValueError would surface as an opaque num_failed rather than the real reason.
        if masks.shape[0] == 0:
            raise SceneError(
                f"SAM 3 found nothing matching {self.config.prompt!r} above score "
                f"{self.config.score_threshold}")

        if masks.dtype != bool:
            masks = masks > 0.5
        if masks.ndim > 3:
            masks = masks.reshape((masks.shape[0], -1) + masks.shape[-2:])
            masks = masks.any(axis=1)

        if self.config.max_detections > 0 and masks.shape[0] > self.config.max_detections:
            keep = np.argsort(scores)[::-1][:self.config.max_detections]
            masks = masks[keep]
            scores = scores[keep]

        # Union, not argmax: a specimen split across detections stays whole, which is what
        # keep_largest defaulting to False already assumes.
        mask = np.any(masks, axis=0)

        self._count += 1
        every = max(1, self.config.batch_log_every)
        if self._count % every == 0:
            logger.info(f"sam3: masked {self._count} images "
                        f"(last: {masks.shape[0]} detection(s), "
                        f"top score {float(scores.max()):.3f})")
        return mask
