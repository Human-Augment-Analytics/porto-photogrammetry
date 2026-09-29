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
    device: str = field(default="cuda", metadata={"help": "CUDA device for the model (SAM 3 is GPU-only), e.g. cuda:1"})


@register_mask
class Sam3Mask(MaskMethod):
    """SAM 3 concept-prompted segmentation (text prompt, default 'skeleton')."""

    name: ClassVar[str] = "sam3"
    title: ClassVar[str] = "Masking (SAM 3)"
    config_cls: ClassVar[type] = Sam3Config

    def __init__(self, config: Sam3Config):
        super().__init__(config)
        self._processor = None

    def header(self, images_dir: Path, output_dir: Path) -> dict[str, object]:
        """Adds the prompt and score threshold to the banner."""
        head = super().header(images_dir, output_dir)
        head["Prompt"] = repr(self.config.prompt)
        head["Score threshold"] = self.config.score_threshold
        return head

    def prepare(self) -> None:
        """Build the model up front so a CUDA refusal or a gated-repo 401 aborts the run."""
        from huggingface_hub.errors import GatedRepoError, LocalEntryNotFoundError

        try:
            self._get_processor()
        except (GatedRepoError, LocalEntryNotFoundError) as exc:
            raise SceneError(
                f"could not fetch the SAM 3 checkpoint from facebook/sam3 "
                f"({type(exc).__name__}). Request access on huggingface.co, run "
                "`hf auth login` on the login node to warm $HF_HOME, or pass "
                "--checkpoint_path.") from exc

    def _get_processor(self):
        if self._processor is not None:
            return self._processor
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # SAM 3 is GPU-only whatever --device says: a CPU run dies in CUDA init mid-build.
        if not torch.cuda.is_available():
            raise SceneError(
                "SAM 3 is GPU-only and torch reports no CUDA device; run it on a GPU node.")

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

        # The vendored _forward_grounding returns bool [N, 1, H, W] masks at source dims, and
        # scores that come back bfloat16 under the autocast, which numpy cannot represent.
        masks = state["masks"][:, 0].cpu().numpy()
        scores = state["scores"].float().cpu().numpy()

        if masks.shape[0] == 0:
            raise SceneError(
                f"SAM 3 found nothing matching {self.config.prompt!r} above score "
                f"{self.config.score_threshold}")

        if self.config.max_detections > 0 and masks.shape[0] > self.config.max_detections:
            keep = np.argsort(scores)[::-1][:self.config.max_detections]
            masks = masks[keep]
            scores = scores[keep]

        # Union, not argmax: a specimen split across detections stays whole, which is what
        # keep_largest defaulting to False already assumes.
        mask = np.any(masks, axis=0)
        logger.debug(f"{image_path.name}: {masks.shape[0]} detection(s), "
                     f"top score {float(scores.max()):.3f}")
        return mask
