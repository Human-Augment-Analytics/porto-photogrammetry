"""DVLT (NVIDIA Deja View looping transformer) feed-forward SfM.

Same output contract as the VGGT adapter without --use_ba: one forward pass gives poses,
intrinsics and a depth-consistent point cloud, written straight into a COLMAP model.
"""
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from augenblick.core.registry import register_sfm
from augenblick.core.scene import Scene
from augenblick.sfm.base import SfMMethod, SfMResult, link_dir

logger = logging.getLogger(__name__)

# DVLT's patch size is fixed by the checkpoint, so it is not a flag.
PATCH_SIZE = 14
# The range the model's authors evaluate the loop count over.
SUPPORTED_STEPS = (8, 16)


@dataclass(frozen=True)
class DVLTConfig:
    """DVLT inference parameters."""

    use_masks: bool = field(default=False, metadata={
        "help": "Drop points outside masks/<stem>.png; an image with no mask is unconstrained"})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    inference_steps: int = field(default=12, metadata={
        "help": "Loop count K of the recurrent block; more is slower and usually better "
                "(evaluated over 8-16)"})
    img_size: int = field(default=504, metadata={
        "help": "Longest image edge fed to the network; H and W are then centre-cropped to "
                "multiples of 14"})
    conf_perc_thresh: float = field(default=25.0, metadata={
        "help": "Percentile (0-100) of DVLT depth confidence, over valid (and, with "
                "--use_masks, masked-in) pixels, below which a pixel does not become a 3D point"})
    mask_pose_fit: bool = field(default=False, metadata={
        "help": "Also weight the camera-pose fit by the masks, so poses come from masked-in "
                "pixels only (needs --use_masks; a frame with an empty mask stays unweighted)"})
    max_points: int = field(default=100_000, metadata={
        "help": "Randomly subsample the point cloud to at most this many points"})
    decode_chunk_size: int = field(default=32, metadata={
        "help": "Frames per ray/depth decoder pass; the model's wrapper default decodes all "
                "at once, which overflows CUDA's 32-bit indexing on ~276-frame scenes"})
    checkpoint: str = field(default="nvidia/dvlt", metadata={
        "help": "HF Hub id (cached under HF_HOME) or a local weights file/directory"})


def _frame_geometry(w, h, img_size, max_w, max_h):
    """Reproduce preprocess_images' resize/crop/pad for one image of size (w, h).

    Returns per-axis resize factors (sx, sy) and the offsets (off_x, off_y) that take a
    network pixel index to a resized-image index: crop offset minus pad offset.
    """
    scale = img_size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    crop_h = max(PATCH_SIZE, (new_h // PATCH_SIZE) * PATCH_SIZE)
    crop_w = max(PATCH_SIZE, (new_w // PATCH_SIZE) * PATCH_SIZE)
    top, left = (new_h - crop_h) // 2, (new_w - crop_w) // 2
    # Padding only happens when aspect ratios differ within the scene.
    pad_top, pad_left = (max_h - crop_h) // 2, (max_w - crop_w) // 2
    return new_w / w, new_h / h, left - pad_left, top - pad_top


def _to_original(reconstruction, image_names, original_sizes, img_size, net_w, net_h):
    """Rename images and move cameras and 2D points from the network frame to the originals.

    preprocess_images resizes (per axis, after rounding) then centre-crops, so this is a
    per-image affine map, not VGGT's square pad; its rescale helper does not apply. DVLT
    indexes pixels on an integer grid (common/projection.py create_meshgrid), while COLMAP
    puts pixel centres at +0.5; with half-pixel-aligned resampling, network index x lands at
    COLMAP coordinate (x + off + 0.5) / s. K and points2D take the same map, so every point
    still reprojects where the network put it.
    """
    for image_id, pyimage in reconstruction.images.items():
        idx = image_id - 1
        w, h = original_sizes[idx]
        sx, sy, off_x, off_y = _frame_geometry(w, h, img_size, net_w, net_h)
        pyimage.name = image_names[idx]

        # One camera per image in the feed-forward path, so each is rescaled exactly once.
        pycamera = reconstruction.cameras[pyimage.camera_id]
        fx, fy, cx, cy = pycamera.params
        pycamera.params = [fx / sx, fy / sy, (cx + off_x + 0.5) / sx, (cy + off_y + 0.5) / sy]
        pycamera.width = w
        pycamera.height = h

        for point2D in pyimage.points2D:
            x, y = point2D.xy
            point2D.xy = [(x + off_x + 0.5) / sx, (y + off_y + 0.5) / sy]
    return reconstruction


def _load_masks(scene, image_paths):
    """Masks aligned with image_paths; a missing mask becomes all-keep, never all-drop.

    dvlt's load_masks_for_directory raises on any missing mask. Real scenes have gaps
    (UF_Mammals_10941_skull: 276 images, 271 masks), and the repo convention (vggt.py) is
    that no mask means no constraint, so pair by stem here instead.
    """
    from dvlt.util.preprocess import MASK_EXTS
    from PIL import Image

    masks, missing = [], []
    for path in image_paths:
        mask_path = next((scene.masks_dir / f"{path.stem}{ext}" for ext in MASK_EXTS
                          if (scene.masks_dir / f"{path.stem}{ext}").exists()), None)
        if mask_path is None:
            missing.append(path.name)
            with Image.open(path) as img:
                masks.append(Image.new("L", img.size, 255))
        else:
            masks.append(Image.open(mask_path))
    logger.info(f"Found {len(image_paths) - len(missing)} corresponding masks")
    if missing:
        logger.warning(f"{len(missing)} image(s) have no mask and stay unconstrained "
                       f"(first: {missing[0]})")
    return masks


@register_sfm
class DVLTSfM(SfMMethod):
    """Runs the DVLT network and converts its output into a COLMAP reconstruction."""

    name: ClassVar[str] = "dvlt"
    config_cls: ClassVar[type] = DVLTConfig

    def run(self, scene: Scene, output_dir: Path) -> SfMResult:
        """Run DVLT over the scene's images and write a COLMAP model to output_dir/sparse/0.

        Args:
            scene: Input scene with images/ and optionally masks/.
            output_dir: Directory to write the COLMAP scene into.

        Returns:
            An SfMResult for the reconstruction produced.
        """
        # torch and dvlt are imported here so the package stays importable without them.
        import numpy as np
        import torch
        import trimesh
        from dvlt.common.constants import DataField, PredictionField
        from dvlt.model.dvlt.model import DVLT
        from dvlt.util.preprocess import (
            IMAGE_EXTS,
            SEGMENTATION_MASK_FIELD,
            load_sequence,
            preprocess_images,
        )
        from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track
        from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues

        self.validate(scene)
        args = self.config
        if args.inference_steps < 1:
            raise ValueError(f"--inference_steps must be >= 1, got {args.inference_steps}")
        if args.decode_chunk_size < 1:
            raise ValueError(f"--decode_chunk_size must be >= 1, got {args.decode_chunk_size}")
        if not 0.0 <= args.conf_perc_thresh <= 100.0:
            raise ValueError(f"--conf_perc_thresh must be in [0, 100], got {args.conf_perc_thresh}")
        if args.mask_pose_fit and not args.use_masks:
            raise ValueError("--mask_pose_fit needs --use_masks")
        if not SUPPORTED_STEPS[0] <= args.inference_steps <= SUPPORTED_STEPS[1]:
            logger.warning(f"--inference_steps {args.inference_steps} is outside the evaluated "
                           f"range {SUPPORTED_STEPS[0]}-{SUPPORTED_STEPS[1]}; proceeding")
        out_dir_str = str(output_dir)

        logger.info("=" * 60)
        logger.info("DVLT to COLMAP Pipeline")
        logger.info(f"  Input:     {scene.root}")
        logger.info(f"  Output:    {out_dir_str}")
        logger.info(f"  Steps (K): {args.inference_steps}")
        logger.info(f"  Use masks: {args.use_masks}")
        logger.info(f"  Mask pose: {args.mask_pose_fit}")
        logger.info("=" * 60)
        t_start = time.time()

        # Set seed for reproducibility
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
        logger.info(f"Setting seed as: {args.seed}")

        # Set device and dtype; the capability query itself needs a CUDA device.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            dtype = torch.float32
        logger.info(f"Using device: {device}")
        logger.info(f"Using dtype: {dtype}")

        t0 = time.time()
        # K is a constructor argument, not a predict() one. The DINOv2 fetch that
        # load_patch_embed_weights=True triggers is overwritten by load_pretrained anyway.
        # The decoders run per frame, so chunking them does not change the output.
        model = DVLT(img_size=args.img_size, inference_steps=args.inference_steps,
                     decode_chunk_size=args.decode_chunk_size, load_patch_embed_weights=False,
                     use_seg_mask_for_pose=args.mask_pose_fit)
        model.load_pretrained(args.checkpoint, strict=True)
        model.setup_test(device)
        logger.info(f"Model loaded in {time.time() - t0:.1f}s")

        # load_sequence sorts internally; build the names with the same sort and filter so
        # COLMAP image i is frame i.
        image_paths = sorted(p for p in scene.images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        _, frames = load_sequence(scene.images_dir)
        assert len(frames) == len(image_paths)
        logger.info(f"Found {len(frames)} images in {scene.images_dir}")
        original_sizes = [f.size for f in frames]  # (w, h)

        masks = _load_masks(scene, image_paths) if args.use_masks else None
        batch = preprocess_images(frames, img_size=args.img_size, patch_size=PATCH_SIZE,
                                  device=device, pil_masks=masks)

        t0 = time.time()
        # Precision-sensitive parts opt out of autocast themselves (force_fp32).
        with torch.no_grad(), torch.autocast(device, dtype=dtype, enabled=device == "cuda"):
            predictions = model.predict(batch)
        logger.info(f"DVLT inference completed in {time.time() - t0:.1f}s")

        cameras = predictions[PredictionField.CAMERAS][0]
        # DVLT gives camera-to-world; COLMAP (and batch_np_matrix_to_pycolmap_wo_track)
        # wants world-to-camera. The axis convention (OpenCV) already matches.
        c2w = cameras.camera_to_worlds.float().cpu().numpy()
        R_w2c = c2w[:, :3, :3].transpose(0, 2, 1)
        t_w2c = -np.einsum("sij,sj->si", R_w2c, c2w[:, :3, 3])
        extrinsic = np.concatenate([R_w2c, t_w2c[..., None]], axis=-1)  # (S, 3, 4)
        # Already in the cropped network frame; _to_original maps it back.
        intrinsic = cameras.get_intrinsics_matrices().float().cpu().numpy()  # (S, 3, 3)

        # world_points is depth unprojected through the fitted poses: already in world space
        # and consistent with the extrinsics being written.
        points_3d = predictions[PredictionField.WORLD_POINTS][0].float().cpu().numpy()  # (S, H, W, 3)
        depth_conf = predictions[PredictionField.DEPTHS_CONF][0].float().cpu().numpy()  # (S, H, W)
        num_frames, height, width, _ = points_3d.shape

        percentiles = np.percentile(depth_conf, [5, 25, 50, 75, 95])
        logger.info("depths_conf p5/p25/p50/p75/p95: " + " / ".join(f"{p:.3g}" for p in percentiles))

        points_rgb = batch[DataField.IMAGES][0].float().cpu().numpy()  # (S, 3, H, W)
        points_rgb = (points_rgb.transpose(0, 2, 3, 1) * 255).astype(np.uint8)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        # Pad pixels (only when aspect ratios differ) are not real image content.
        valid = batch["gradio_valid_pixels"][0].cpu().numpy().astype(bool)
        valid &= np.isfinite(points_3d).all(axis=-1)
        if masks is not None:
            valid &= batch[SEGMENTATION_MASK_FIELD][0].cpu().numpy().astype(bool)
            logger.info(f"After masking {valid.mean():.1%} of pixels remain")
        if not valid.any():
            raise ValueError("No valid pixels remain to threshold on confidence")

        # The percentile is over candidate pixels, so masking does not shift the cutoff.
        conf_thresh = float(np.percentile(depth_conf[valid], args.conf_perc_thresh))
        conf_mask = valid & (depth_conf >= conf_thresh)
        logger.info(f"Confidence >= p{args.conf_perc_thresh:g} ({conf_thresh:.3g}) keeps "
                    f"{conf_mask.mean():.1%} of pixels")
        conf_mask = randomly_limit_trues(conf_mask, args.max_points)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        logger.info("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            np.array([width, height]),
            shared_camera=False,
            camera_type="PINHOLE",
        )
        reconstruction = _to_original(
            reconstruction, [p.name for p in image_paths], original_sizes, args.img_size,
            width, height)

        sparse_reconstruction_dir = os.path.join(out_dir_str, "sparse", "0")
        logger.info("Saving reconstruction to %s", sparse_reconstruction_dir)
        os.makedirs(sparse_reconstruction_dir, exist_ok=True)
        reconstruction.write(sparse_reconstruction_dir)

        # Save point cloud for fast visualization
        trimesh.PointCloud(points_3d, colors=points_rgb).export(
            os.path.join(sparse_reconstruction_dir, "points.ply"))

        link_dir(scene.images_dir, output_dir / "images")
        if scene.has_masks():
            link_dir(scene.masks_dir, output_dir / "masks")

        total_time = time.time() - t_start
        logger.info("=" * 60)
        logger.info("Pipeline complete")
        logger.info(f"  Total time: {total_time:.1f}s")
        logger.info(f"  Output:     {out_dir_str}")
        logger.info("=" * 60)

        return SfMResult(
            output_dir=output_dir,
            elapsed=total_time,
            scene=Scene(output_dir),
            num_images=reconstruction.num_reg_images(),
            num_points=reconstruction.num_points3D(),
        )
