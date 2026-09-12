# Source: https://github.com/facebookresearch/vggt/blob/main/demo_colmap.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""VGGT feed-forward SfM, optionally refined by tracking plus bundle adjustment."""
import copy
import glob
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from augenblick.core.registry import register_sfm
from augenblick.core.scene import Scene
from augenblick.sfm.base import SfMMethod, SfMResult

logger = logging.getLogger(__name__)


# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


@dataclass(frozen=True)
class VGGTConfig:
    """VGGT inference and optional bundle-adjustment parameters."""

    use_masks: bool = field(default=False, metadata={"help": "Use masks for reconstruction"})
    mask_erode_px: int = field(default=3, metadata={
        "help": "Shrink masks by this many pixels before keypoint filtering, dropping the "
                "silhouette seam detectors fire on (0 disables)"})
    seed: int = field(default=42, metadata={"help": "Random seed for reproducibility"})
    use_ba: bool = field(default=False, metadata={"help": "Use BA for reconstruction"})
    max_reproj_error: float = field(default=8.0, metadata={
        "help": "Maximum reprojection error for reconstruction"})
    min_inlier_per_frame: int = field(default=64, metadata={
        "help": "Surviving tracks a frame needs to take part in BA; frames below it are "
                "dropped individually rather than aborting the reconstruction"})
    min_valid_frames: float = field(default=0.3, metadata={
        "help": "Fraction of frames that must clear --min_inlier_per_frame, or BA is skipped "
                "entirely"})
    shared_camera: bool = field(default=False, metadata={
        "help": "Use shared camera for all images"})
    camera_type: str = field(default="SIMPLE_PINHOLE", metadata={
        "help": "Camera type for reconstruction"})
    vis_thresh: float = field(default=0.2, metadata={"help": "Visibility threshold for tracks"})
    query_frame_num: int = field(default=12, metadata={"help": "Number of frames to query"})
    max_query_pts: int = field(default=4096, metadata={
        "help": "Maximum number of query points. The BA inlier floor counts surviving tracks "
                "per frame absolutely, so masked scenes need a large budget to clear it"})
    fine_tracking: bool = field(default=True, metadata={
        "help": "Use fine tracking (slower but more accurate)"})
    conf_thres_value: float = field(default=2.0, metadata={
        "help": "Minimum VGGT depth confidence, applied in both modes: without --use_ba it "
                "selects which depth pixels become 3D points; with --use_ba the tracker "
                "samples the same map at each query point and drops those below it"})


def run_VGGT(model, images, masks, dtype, resolution=518):
    """Run the VGGT network, returning extrinsics, intrinsics, depth, and masked confidence."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    masks_resized = []
    for i in range(len(masks)):
        mask = masks[i]
        if mask is not None:
            mask = mask.resize((resolution, resolution), Image.Resampling.NEAREST)
        masks_resized.append(mask)
    masks = masks_resized

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()

    for i in range(len(masks)):
        if masks[i] is not None:
            mask = np.array(masks[i])
            mask = mask.astype(np.float32)
            mask = mask / 255.0
            depth_conf[i] *= mask

    return extrinsic, intrinsic, depth_map, depth_conf


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False,
    shared_camera=False, original_coords_img_size=1024,
):
    """Rename reconstruction images to their originals and rescale cameras to full resolution."""
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # original_coords was produced by the loader at original_coords_img_size;
            # rescale top_left into the reconstruction's img_size units before shifting.
            coord_scale = img_size / original_coords_img_size
            top_left = original_coords[pyimageid - 1, :2] * coord_scale

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    return reconstruction


def _link_dir(src: Path, dst: Path) -> None:
    """Symlink dst -> src, replacing a stale link left by an earlier run."""
    # lexists, not exists: a link dangling from a moved scene still blocks os.symlink.
    if dst.is_symlink():
        dst.unlink()
    elif os.path.lexists(dst):
        logger.warning("%s exists and is not a symlink; leaving it in place", dst)
        return
    os.symlink(src, dst)
    logger.info("Linked %s -> %s", dst, src)


@register_sfm
class VGGTSfM(SfMMethod):
    """Runs the VGGT network and converts its output into a COLMAP reconstruction."""

    name: ClassVar[str] = "vggt"
    config_cls: ClassVar[type] = VGGTConfig

    def run(self, scene: Scene, output_dir: Path) -> SfMResult:
        """Run VGGT over the scene's images and write a COLMAP model to output_dir/sparse/0.

        Args:
            scene: Input scene with images/ and optionally masks/.
            output_dir: Directory to write the COLMAP scene into.

        Returns:
            An SfMResult for the reconstruction produced.
        """
        # torch and vggt are imported here so the package stays importable without a GPU.
        import numpy as np
        import pycolmap
        import torch
        import torch.nn.functional as F
        import trimesh

        # Configure CUDA settings
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        from vggt.dependency.np_to_pycolmap import (
            batch_np_matrix_to_pycolmap,
            batch_np_matrix_to_pycolmap_wo_track,
        )
        from vggt.dependency.track_predict import predict_tracks
        from vggt.models.vggt import VGGT
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
        from vggt.utils.load_fn import load_and_preprocess_images_square

        self.validate(scene)
        args = self.config
        input_dir = str(scene.root)
        out_dir_str = str(output_dir)

        logger.info("=" * 60)
        logger.info("VGGT to COLMAP Pipeline")
        logger.info(f"  Input:     {input_dir}")
        logger.info(f"  Output:    {out_dir_str}")
        logger.info(f"  Use BA:    {args.use_ba}")
        logger.info(f"  Use masks: {args.use_masks}")
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

        with torch.no_grad():
            # Run VGGT for camera and depth estimation
            t0 = time.time()
            model = VGGT()
            _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
            model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
            model.eval()
            model = model.to(device)
            load_time = time.time() - t0
            logger.info(f"Model loaded in {load_time:.1f}s")

            # Get image paths and preprocess them
            image_dir = os.path.join(input_dir, "images")
            mask_dir = os.path.join(input_dir, "masks")
            image_path_list = glob.glob(os.path.join(image_dir, "*"))
            mask_path_list = None
            if len(image_path_list) == 0:
                raise ValueError(f"No images found in {image_dir}")
            logger.info(f"Found {len(image_path_list)} images in {image_dir}")
            base_image_path_list = [os.path.basename(path) for path in image_path_list]

            if args.use_masks:
                mask_path_list = []
                for base_image_path in base_image_path_list:
                    mask_path = os.path.join(mask_dir, f"{Path(base_image_path).stem}.png")
                    if os.path.exists(mask_path):
                        mask_path_list.append(mask_path)
                    else:
                        mask_path_list.append(None)
                logger.info(f"Found {len([m for m in mask_path_list if m is not None])} corresponding masks")

            # Load images and original coordinates
            # Load Image in 1024, while running VGGT with 518
            vggt_fixed_resolution = 518
            img_load_resolution = 1024

            images, original_coords, masks = load_and_preprocess_images_square(
                image_path_list, mask_path_list, target_size=img_load_resolution)
            images = images.to(device)
            original_coords = original_coords.to(device)
            logger.info(f"Loaded {len(images)} images from {image_dir}")

            # Run VGGT to estimate camera and depth
            # Run with 518x518 images
            t0 = time.time()
            extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(
                model, images, masks, dtype, vggt_fixed_resolution)
            points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
            vggt_time = time.time() - t0
            logger.info(f"VGGT inference completed in {vggt_time:.1f}s")

            if args.use_ba:
                t0 = time.time()
                image_size = np.array(images.shape[-2:])
                scale = img_load_resolution / vggt_fixed_resolution
                shared_camera = args.shared_camera

                # Built from the loader's masks, which match `images`
                track_masks = None
                if args.use_masks and any(m is not None for m in masks):
                    track_masks = torch.zeros(
                        (len(masks), img_load_resolution, img_load_resolution),
                        dtype=torch.bool, device=device)
                    for i, mask in enumerate(masks):
                        if mask is None:
                            # No mask for this frame means no constraint, not an empty frame.
                            track_masks[i] = True
                        else:
                            track_masks[i] = torch.from_numpy(
                                np.array(mask) > 0).to(device)

                logger.info(
                    f"Tracking budget: {args.max_query_pts} query pts x "
                    f"{args.query_frame_num} frames")

                with torch.cuda.amp.autocast(dtype=dtype):
                    # Predicting Tracks
                    # Using VGGSfM tracker instead of VGGT tracker for efficiency
                    # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
                    # Will be fixed in VGGT v2

                    # You can also change the pred_tracks to tracks from any other methods
                    # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
                    pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                        images,
                        conf=depth_conf,
                        points_3d=points_3d,
                        masks=track_masks,
                        mask_erode_px=args.mask_erode_px,
                        max_query_pts=args.max_query_pts,
                        query_frame_num=args.query_frame_num,
                        keypoint_extractor="aliked+sp",
                        fine_tracking=args.fine_tracking,
                        conf_thresh=args.conf_thres_value
                    )

                    torch.cuda.empty_cache()

                # rescale the intrinsic matrix from 518 to 1024
                intrinsic[:, :2, :] *= scale
                track_mask = pred_vis_scores > args.vis_thresh

                # TODO: radial distortion, iterative BA
                reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
                    points_3d,
                    extrinsic,
                    intrinsic,
                    pred_tracks,
                    image_size,
                    masks=track_mask,
                    max_reproj_error=args.max_reproj_error,
                    shared_camera=shared_camera,
                    camera_type=args.camera_type,
                    min_inlier_per_frame=args.min_inlier_per_frame,
                    min_valid_frames=args.min_valid_frames,
                    points_rgb=points_rgb,
                )

                if reconstruction is None:
                    raise ValueError("No reconstruction can be built with BA")

                # Bundle Adjustment
                ba_options = pycolmap.BundleAdjustmentOptions()
                pycolmap.bundle_adjustment(reconstruction, ba_options)

                reconstruction_resolution = img_load_resolution
                ba_time = time.time() - t0
                logger.info(f"Tracking + bundle adjustment completed in {ba_time:.1f}s")
            else:
                max_points_for_colmap = 100000  # randomly sample 3D points
                shared_camera = False  # in the feedforward manner, we do not support shared camera
                camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

                image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
                num_frames, height, width, _ = points_3d.shape

                points_rgb = F.interpolate(
                    images, size=(vggt_fixed_resolution, vggt_fixed_resolution),
                    mode="bilinear", align_corners=False
                )
                points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
                points_rgb = points_rgb.transpose(0, 2, 3, 1)

                # (S, H, W, 3), with x, y coordinates and frame indices
                points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

                conf_mask = depth_conf >= args.conf_thres_value
                conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

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
                    image_size,
                    shared_camera=shared_camera,
                    camera_type=camera_type,
                )

                reconstruction_resolution = vggt_fixed_resolution

            reconstruction = rename_colmap_recons_and_rescale_camera(
                reconstruction,
                base_image_path_list,
                original_coords.cpu().numpy(),
                img_size=reconstruction_resolution,
                shift_point2d_to_original_res=True,
                shared_camera=shared_camera,
            )
            logger.info("Saving reconstruction to %s", os.path.join(out_dir_str, "sparse", "0"))
            sparse_reconstruction_dir = os.path.join(out_dir_str, "sparse", "0")
            os.makedirs(sparse_reconstruction_dir, exist_ok=True)
            reconstruction.write(sparse_reconstruction_dir)

            # Save point cloud for fast visualization
            trimesh.PointCloud(points_3d, colors=points_rgb).export(
                os.path.join(out_dir_str, "sparse/0/points.ply"))

            _link_dir(scene.images_dir, output_dir / "images")
            if scene.has_masks():
                _link_dir(scene.masks_dir, output_dir / "masks")

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
