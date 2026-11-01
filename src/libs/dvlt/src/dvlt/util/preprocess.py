# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image / video loading and preprocessing helpers for DVLT inference.

Promoted from ``dvlt.scripts.gradio_demo`` so the same routines can drive both
the Gradio demo and ad-hoc Python usage after ``pip install -e .``.
"""

import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import InterpolationMode, resize

from dvlt.common.constants import DataField


VIDEO_EXTS = {".mp4", ".mov", ".gif", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
# Extensions considered when looking for a mask file paired with an image by stem.
MASK_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")
# Separate from DataField.POINT_MASKS, which is reserved for depth conditioning
# at training time; this one is a UI/inference filter for the demo.
SEGMENTATION_MASK_FIELD = "gradio_segmentation_mask"


def _frames_from_video(path: str, target_fps: Optional[float] = 2.0, video_fps: float = 24.0) -> list[Image.Image]:
    """Decode a video and return frames downsampled to ~target_fps.

    Falls back to ``video_fps`` when the container does not expose frame rate.
    Pass ``target_fps=None`` to keep every frame (stride = 1).
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Failed to open video: {path}")

    if target_fps is None:
        stride = 1
    else:
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or not math.isfinite(src_fps) or src_fps <= 0:
            src_fps = video_fps
        stride = max(1, int(round(src_fps / target_fps)))

    frames: list[Image.Image] = []
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
        idx += 1
    cap.release()

    if not frames:
        raise ValueError(f"No frames decoded from video: {path}")
    return frames


def load_sequence(
    path: str | Path,
    video_target_fps: Optional[float] = 2.0,
    video_fps: float = 24.0,
) -> tuple[str, list[Image.Image]]:
    """Resolve a single path into a (sequence_name, frames) pair.

    - Directory: name = directory basename, frames = images inside (sorted).
    - Video file (extension in :data:`VIDEO_EXTS`): name = file stem, frames
      are sampled at ~``video_target_fps`` (falling back to ``video_fps`` if
      the container does not expose frame rate). Pass
      ``video_target_fps=None`` to keep every frame.
    - Image file (extension in :data:`IMAGE_EXTS`): name = file stem,
      frames = ``[image]``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"path does not exist: {path}")

    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not files:
            raise ValueError(f"directory {path} contains no images (looked for {sorted(IMAGE_EXTS)}).")
        return path.name, [Image.open(p) for p in files]

    ext = path.suffix.lower()
    if ext in VIDEO_EXTS:
        return path.stem, _frames_from_video(str(path), target_fps=video_target_fps, video_fps=video_fps)
    if ext in IMAGE_EXTS:
        return path.stem, [Image.open(path)]

    raise ValueError(
        f"{path}: unsupported file type {ext!r}. "
        f"Expected a directory, an image ({sorted(IMAGE_EXTS)}), or a video ({sorted(VIDEO_EXTS)})."
    )


def _make_divisible(v: int, divisor: int) -> int:
    return max(divisor, (v // divisor) * divisor)


def load_masks_for_directory(mask_dir: str | Path, image_dir: str | Path) -> list[Image.Image]:
    """Load binary masks paired by stem with the sorted images in ``image_dir``.

    Images in ``image_dir`` are enumerated with the same sort/filter that
    :func:`load_sequence` uses for directory inputs, so the returned list is
    aligned with ``load_sequence(image_dir)[1]`` index-for-index. Each image
    must have a sibling mask file under ``mask_dir`` whose stem matches
    (extensions in :data:`MASK_EXTS` are tried). Masks are assumed to be at
    the **original image resolution**; callers are responsible for matching
    the resolution beforehand (see ``scripts/preprocess_dtu_masks.py`` for
    DTU's preprocessing pipeline).
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    if not image_dir.is_dir():
        raise ValueError(f"--mask-dir paired with non-directory input: {image_dir}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"mask dir does not exist: {mask_dir}")
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise ValueError(f"no images under {image_dir} to pair with masks")
    masks: list[Image.Image] = []
    missing: list[str] = []
    for img_path in images:
        for ext in MASK_EXTS:
            mp = mask_dir / f"{img_path.stem}{ext}"
            if mp.exists():
                masks.append(Image.open(mp))
                break
        else:
            missing.append(img_path.name)
    if missing:
        raise FileNotFoundError(
            f"no matching mask in {mask_dir} for {len(missing)} image(s) "
            f"(first: {missing[0]}; looked for extensions {list(MASK_EXTS)})"
        )
    return masks


def preprocess_images(
    pil_images: list[Image.Image],
    img_size: int,
    patch_size: int,
    device: torch.device,
    pil_masks: Optional[list[Image.Image]] = None,
) -> dict:
    """Convert PIL images into a model-ready batch dict.

    Resizes the longest side to *img_size* (keeping aspect ratio) then
    center-crops both dimensions to be divisible by *patch_size*.
    When multiple images differ in aspect ratio, center-pads each to a
    common height and width so they can be batched (pad value 0 / black).
    Returns ``images`` ``[1, S, 3, H, W]`` and ``gradio_valid_pixels``
    ``[1, S, H, W]`` (True on real pixels, False on synthetic pad).

    When ``pil_masks`` is given (one per image, **already at the same H×W as
    the corresponding image**), each mask follows the identical resize / crop /
    pad pipeline as its image but with NEAREST interpolation, and the result
    is emitted as :data:`SEGMENTATION_MASK_FIELD` shaped ``[1, S, H, W]``
    (bool, True = keep).
    """
    if pil_masks is not None and len(pil_masks) != len(pil_images):
        raise ValueError(f"pil_masks length ({len(pil_masks)}) must match pil_images length ({len(pil_images)})")

    tensors: list[torch.Tensor] = []
    mask_tensors: list[torch.Tensor] = []
    for i, img in enumerate(pil_images):
        img = img.convert("RGB")
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        _, h, w = t.shape
        scale = img_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        t = resize(t, [new_h, new_w], antialias=True)

        crop_h = _make_divisible(new_h, patch_size)
        crop_w = _make_divisible(new_w, patch_size)
        top = (new_h - crop_h) // 2
        left = (new_w - crop_w) // 2
        t = t[:, top : top + crop_h, left : left + crop_w]

        tensors.append(t)

        if pil_masks is not None:
            m_arr = np.array(pil_masks[i].convert("L"))
            if m_arr.shape != (h, w):
                raise ValueError(
                    f"mask {i} shape {m_arr.shape} (H, W) does not match "
                    f"image shape ({h}, {w}). Preprocess masks to the image "
                    f"resolution first (e.g. scripts/preprocess_dtu_masks.py)."
                )
            mt = torch.from_numpy((m_arr > 127).astype(np.uint8)).unsqueeze(0).float()  # (1, h, w)
            mt = resize(mt, [new_h, new_w], interpolation=InterpolationMode.NEAREST)
            mt = mt[:, top : top + crop_h, left : left + crop_w]
            mask_tensors.append(mt)

    # Same longest-side scaling still yields different H×W when aspect ratios differ.
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    aligned: list[torch.Tensor] = []
    valid_masks: list[torch.Tensor] = []
    aligned_segs: list[torch.Tensor] = []
    for i, t in enumerate(tensors):
        _, h, w = t.shape
        pad_h, pad_w = max_h - h, max_w - w
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        # F.pad: (left, right, top, bottom) for (C, H, W)
        aligned.append(F.pad(t, (pad_left, pad_right, pad_top, pad_bottom), value=0.0))
        # Same padding on a ones mask; padded regions become False (do not use POINT_MASKS — that field is reserved for depth conditioning).
        vm = torch.ones(1, h, w, dtype=torch.float32)
        vm = F.pad(vm, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        valid_masks.append(vm.squeeze(0).bool())
        if pil_masks is not None:
            sm = F.pad(mask_tensors[i], (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
            aligned_segs.append(sm.squeeze(0).bool())

    images = torch.stack(aligned, dim=0).unsqueeze(0).to(device)  # (1, S, 3, H, W)
    gradio_valid_pixels = torch.stack(valid_masks, dim=0).unsqueeze(0).to(device)  # (1, S, H, W)
    out: dict = {DataField.IMAGES: images, "gradio_valid_pixels": gradio_valid_pixels}
    if pil_masks is not None:
        seg = torch.stack(aligned_segs, dim=0).unsqueeze(0).to(device)  # (1, S, H, W)
        out[SEGMENTATION_MASK_FIELD] = seg
    return out
