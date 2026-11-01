#!/usr/bin/env python3
"""
Convert a COLMAP scene's sparse model to AliceVision SfMData JSON (.sfm).

This lets Meshroom's dense half (PrepareDenseScene onward) run on COLMAP's cameras and
tracks instead of its own SfM. AliceVision has a COLMAP exporter but no importer.

Input: a COLMAP scene (images/, optional masks/, sparse/0/) with pinhole cameras.
Output: an SfMData file readable by AliceVision 2025.1 (SfMData version 1.2.12).

Usage:
    python baseline/colmap_to_sfmdata.py <colmap_scene> <out.sfm> [--sparse sparse/0]
"""
import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# The installed AliceVision loader rejects newer files, and required fields are
# version-gated (jsonIO.cpp:loadIntrinsic). Fields below follow its cameraInit output.
SFMDATA_VERSION = (1, 2, 12)

PINHOLE_MODELS = ("SIMPLE_PINHOLE", "PINHOLE")

# AliceVision's UndefinedIndexT; ids must stay below it.
MAX_INDEX = 2**32 - 1

# PrepareDenseScene's default maskExtension.
MASK_EXTENSION = ".png"

PYCOLMAP_INSTALL = "pip install pycolmap==4.1.1"


class ConversionError(RuntimeError):
    """The COLMAP scene cannot drive Meshroom's dense stages."""


@dataclass(frozen=True)
class ConvertStats:
    views: int
    intrinsics: int
    landmarks: int
    landmarks_dropped: int
    observations: int
    observations_deduped: int
    unregistered: int


def import_pycolmap():
    try:
        import pycolmap
    except ImportError:
        raise ConversionError(f"pycolmap is not installed in this env; run `{PYCOLMAP_INSTALL}`") from None
    return pycolmap


def _intrinsic(camera_id, camera):
    """One pinhole intrinsic. sensorWidth = width makes 1 mm == 1 px, so focalLength is fy."""
    w, h = camera.width, camera.height
    fx, fy = camera.focal_length_x, camera.focal_length_y
    return {
        "intrinsicId": camera_id,
        "width": w,
        "height": h,
        "sensorWidth": w,
        "sensorHeight": h,
        "serialNumber": str(camera_id),
        "type": "pinhole",
        "initializationMode": "calibrated",
        "initialFocalLength": fy,
        "focalLength": fy,
        "pixelRatio": fy / fx,
        "pixelRatioLocked": True,
        "offsetLocked": True,
        "scaleLocked": True,
        # Stored as an offset from the image centre. No 0.5 px shift: AliceVision's own
        # COLMAP exporter applies none either.
        "principalPoint": [camera.principal_point_x - w / 2, camera.principal_point_y - h / 2],
        "distortionLocked": True,
        "distortionInitializationMode": "none",
        "distortionParams": [],
        "undistortionOffset": [0, 0],
        "undistortionParams": [],
        "distortionType": "none",
        "undistortionType": "none",
        "locked": True,
    }


def _pose(image):
    """AliceVision's Pose3(R, C) is world->camera like COLMAP; saveMatrix writes R column-major."""
    rotation = image.cam_from_world().rotation.matrix()
    return {
        "poseId": image.image_id,
        "pose": {
            "transform": {
                "rotation": [float(v) for v in rotation.flatten(order="F")],
                "center": [float(v) for v in image.projection_center()],
            },
            "locked": True,
        },
    }


def _check_masks(masks_dir, images):
    """Fail if tryLoadMask would hand a view another image's mask.

    tryLoadMask tries masks/<viewId>.png before masks/<image stem>.png, so with unpadded
    numeric names (5.jpg) view 5 can pick up the wrong file.
    """
    collisions = []
    missing = 0
    for image in images:
        stem = Path(image.name).stem
        by_id = masks_dir / f"{image.image_id}{MASK_EXTENSION}"
        if by_id.exists() and stem != str(image.image_id):
            collisions.append(f"{image.name} (view {image.image_id}) -> {by_id.name}")
        elif not (masks_dir / f"{stem}{MASK_EXTENSION}").exists():
            missing += 1
    if collisions:
        shown = ", ".join(collisions[:5])
        raise ConversionError(
            f"{len(collisions)} view(s) would load another image's mask by view id: {shown}. "
            "Rename the images or masks so no mask is named after a different image's id."
        )
    if missing:
        logger.warning(f"{missing} view(s) have no mask in {masks_dir}; they run unmasked")


def _structure(recon, registered):
    """Landmarks with one observation per view, as AliceVision keys observations by view.

    A same-image duplicate keeps its lowest-reprojection-error observation; landmarks
    left with fewer than two views are dropped.
    """
    cam_from_world = {i: recon.images[i].cam_from_world() for i in registered}
    structure = []
    dropped = deduped = 0
    for point_id, point in recon.points3D.items():
        by_view = {}
        for el in point.track.elements:
            if el.image_id not in cam_from_world:
                continue
            by_view.setdefault(el.image_id, []).append(el.point2D_idx)

        observations = []
        for image_id, idxs in by_view.items():
            image = recon.images[image_id]
            if len(idxs) > 1:
                deduped += len(idxs) - 1
                camera = recon.cameras[image.camera_id]
                projected = camera.img_from_cam(cam_from_world[image_id] * point.xyz)

                def error(idx):
                    if projected is None:
                        return 0.0
                    dx, dy = image.points2D[idx].xy - projected
                    return dx * dx + dy * dy

                idxs = [min(idxs, key=error)]
            x, y = image.points2D[idxs[0]].xy
            observations.append({
                "observationId": image_id,
                "featureId": idxs[0],
                "x": [float(x), float(y)],
                "scale": 1.0,
            })

        if len(observations) < 2:
            dropped += 1
            continue
        if point_id >= MAX_INDEX:
            raise ConversionError(f"point3D id {point_id} does not fit AliceVision's 32-bit IndexT")
        structure.append({
            "landmarkId": point_id,
            "descType": "sift",
            "color": [int(c) for c in point.color],
            "X": [float(v) for v in point.xyz],
            "observations": observations,
        })
    return structure, dropped, deduped


def convert(scene_dir, out_path, sparse="sparse/0", max_trackless_fraction=0.1):
    """Write scene_dir/<sparse> as SfMData JSON to out_path and return its counts.

    Raises ConversionError, before writing anything, on input the dense stages cannot
    use: distorted cameras, missing images, track-less models and mask collisions.
    """
    pycolmap = import_pycolmap()
    scene_dir = Path(scene_dir)
    sparse_dir = scene_dir / sparse
    images_dir = scene_dir / "images"
    if not sparse_dir.is_dir():
        raise ConversionError(f"no COLMAP model at {sparse_dir}")
    if not images_dir.is_dir():
        raise ConversionError(f"no images/ in {scene_dir}")

    recon = pycolmap.Reconstruction(str(sparse_dir))
    registered = sorted(i for i, image in recon.images.items() if image.has_pose)
    if not registered:
        raise ConversionError(f"{sparse_dir} has no registered images")
    images = [recon.images[i] for i in registered]

    camera_ids = sorted({image.camera_id for image in images})
    distorted = sorted({recon.cameras[c].model.name for c in camera_ids} - set(PINHOLE_MODELS))
    if distorted:
        raise ConversionError(
            f"camera model(s) {', '.join(distorted)} are not supported (only "
            f"{'/'.join(PINHOLE_MODELS)}). Re-run SfM with --camera_model SIMPLE_PINHOLE, "
            "or undistort the scene with COLMAP first."
        )

    # Absolute but unresolved: tryLoadMask matches masks by this path's filename, which a
    # symlink target need not share.
    views = []
    missing = []
    for image in images:
        path = images_dir / image.name
        if not path.exists():
            missing.append(image.name)
        camera = recon.cameras[image.camera_id]
        views.append({
            "viewId": image.image_id,
            "poseId": image.image_id,
            "intrinsicId": image.camera_id,
            "path": os.path.abspath(path),
            "width": camera.width,
            "height": camera.height,
            "metadata": {},
        })
    if missing:
        raise ConversionError(
            f"{len(missing)} registered image(s) missing from {images_dir}, e.g. {missing[0]}"
        )

    masks_dir = scene_dir / "masks"
    if masks_dir.is_dir():
        _check_masks(masks_dir, images)

    structure, dropped, deduped = _structure(recon, registered)
    observed = {obs["observationId"] for lm in structure for obs in lm["observations"]}
    unobserved = len(registered) - len(observed)
    hint = "Track-less SfM (DVLT, or VGGT without --use_ba) cannot drive DepthMap, which picks neighbours from landmarks."
    if not structure:
        raise ConversionError(f"{sparse_dir} has no landmark seen by two or more views. {hint}")
    if unobserved > max_trackless_fraction * len(registered):
        raise ConversionError(
            f"{unobserved} of {len(registered)} views have no observations "
            f"(limit {max_trackless_fraction:.0%}). {hint}"
        )

    on_disk = {p.name for p in images_dir.iterdir()}
    unregistered = len(on_disk - {Path(i.name).name for i in images})

    sfm = {
        "version": list(SFMDATA_VERSION),
        "featuresFolders": [],
        "matchesFolders": [],
        "views": views,
        "intrinsics": [_intrinsic(c, recon.cameras[c]) for c in camera_ids],
        "poses": [_pose(image) for image in images],
        "structure": structure,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(sfm, f)

    stats = ConvertStats(
        views=len(views),
        intrinsics=len(camera_ids),
        landmarks=len(structure),
        landmarks_dropped=dropped,
        observations=sum(len(lm["observations"]) for lm in structure),
        observations_deduped=deduped,
        unregistered=unregistered,
    )
    logger.info(
        f"SfMData {out_path}: {stats.views} views, {stats.intrinsics} intrinsics, "
        f"{stats.landmarks} landmarks ({stats.landmarks_dropped} dropped with < 2 views), "
        f"{stats.observations} observations ({stats.observations_deduped} same-view duplicates dropped)"
    )
    if unobserved:
        logger.warning(f"{unobserved} view(s) have no observations and get no DepthMap neighbours")
    if unregistered:
        logger.warning(f"{unregistered} image(s) in {images_dir} are not registered and get no depth map")
    return stats


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Convert a COLMAP scene to AliceVision SfMData JSON.")
    parser.add_argument("scene_dir", type=Path, help="COLMAP scene (images/, sparse/0/, optional masks/)")
    parser.add_argument("out_path", type=Path, help="Output .sfm file")
    parser.add_argument("--sparse", default="sparse/0", help="Model folder, relative to scene_dir")
    parser.add_argument(
        "--max_trackless_fraction",
        type=float,
        default=0.1,
        help="Reject the model if more than this fraction of views has no observations",
    )
    args = parser.parse_args()
    try:
        convert(args.scene_dir, args.out_path, args.sparse, args.max_trackless_fraction)
    except ConversionError as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
