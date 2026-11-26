"""LOCAL PATCH (augenblick): first-party file, not upstream NeuS2.

Stage 4 (with --eval): render the held-out views at the shared evaluation raster, with
upstream's evaluation settings (scripts/render_utils.py:149-152, 272-283).
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np
from PIL import Image

import ngp_session
import scene_io
from augenblick.core.errors import SceneError
from augenblick.core.scene import Scene
from augenblick.eval.cameras import camera_records

logger = logging.getLogger("neus2.render")
SPP = 8
BACKGROUND = [0.0, 0.0, 0.0, 1.0]
MIN_TRANSMITTANCE = 1e-4


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("-s", "--source_path", type=Path, required=True, help="COLMAP scene")
    parser.add_argument("-m", "--model_path", type=Path, required=True, help="Output directory")
    parser.add_argument("--n_steps", type=int, default=30000, help="Snapshot to render")
    parser.add_argument("--evaluation_resolution", type=int, default=-1,
                        help="Held-out downscale, independent of training; -1 caps the long side at 1600")
    args = parser.parse_args(argv)
    args.source_path, args.model_path = args.source_path.resolve(), args.model_path.resolve()
    return args


def configure(testbed) -> None:
    testbed.background_color = BACKGROUND
    testbed.snap_to_pixel_centers = True
    testbed.nerf.rendering_min_transmittance = MIN_TRANSMITTANCE
    testbed.fov_axis = 1
    testbed.zoom = 1.0


def set_camera(testbed, pose: np.ndarray, intrinsic: np.ndarray, size: tuple[int, int]) -> None:
    """Held-out camera without loading it as training data (set_camera_to_training_view's maths)."""
    testbed.set_nerf_camera_matrix(np.asarray(pose, dtype=np.float32)[:3, :4])
    testbed.fov_axis = 1
    fov_xy, screen_center = scene_io.held_out_camera(intrinsic, size)
    testbed.fov_xy = np.asarray(fov_xy, dtype=np.float32)
    testbed.screen_center = np.asarray(screen_center, dtype=np.float32)


def to_srgb8(image: np.ndarray) -> np.ndarray:
    srgb = np.clip(scene_io.linear_to_srgb(image[..., :3]), 0.0, 1.0)
    return (srgb * 255.0 + 0.5).astype(np.uint8)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    test_json = args.model_path / scene_io.SCENE_DIR / "transforms_test.json"
    if not test_json.is_file():
        raise SceneError(f"no {test_json}; run prepare.py with --eval")
    frames = scene_io.read_json(test_json)["frames"]
    scene = Scene(args.source_path)
    by_stem = {record["img_name"]: record for record in camera_records(scene.model_dir)}

    testbed, _ = ngp_session.open_testbed(scene_io.snapshot_path(args.model_path, args.n_steps))
    configure(testbed)
    root = args.model_path / "test" / f"ours_{args.n_steps}"
    (root / "renders").mkdir(parents=True, exist_ok=True)
    (root / "gt").mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    sizes, view_sizes = set(), {}
    for frame in frames:
        stem = Path(frame["file_path"]).stem
        record = by_stem[stem]
        size, intrinsic = scene_io.raster(record, args.evaluation_resolution)
        sizes.add(size)
        view_sizes[stem] = size
        set_camera(testbed, np.asarray(frame["transform_matrix"]), intrinsic, size)
        image = np.asarray(testbed.render(size[0], size[1], SPP, True))
        if image.shape != (size[1], size[0], 4):
            raise SceneError(f"{stem}: render returned {image.shape}, expected {(size[1], size[0], 4)}")
        Image.fromarray(to_srgb8(image)).save(root / "renders" / f"{stem}.png")
        scene_io.canonical_rgb(scene.images_dir / record["image_name"], size).save(root / "gt" / f"{stem}.png")
    scene_io.write_json(root / "render_protocol.json", {
        "held_out_direct_canonical_raster": True,
        "resolution": args.evaluation_resolution, "rasters": [list(s) for s in sorted(sizes)],
        "original_photograph_resize": "PIL BICUBIC; floor dimensions",
        "training_resolution_unchanged": True,
        "camera_intrinsics": "per-view fx, fy, cx, cy scaled to the raster",
        "spp": SPP, "background": "black", "snap_to_pixel_centers": True,
        "rendering_min_transmittance": MIN_TRANSMITTANCE,
        "views": sorted(Path(frame["file_path"]).stem for frame in frames)})
    # The scorer's --require-complete needs the partition this process actually rendered.
    train_frames = scene_io.read_json(args.model_path / scene_io.SCENE_DIR / "transforms_train.json")["frames"]
    scene_io.record_native_selection(scene.root, args.model_path, train_frames, frames, view_sizes,
                                     args.evaluation_resolution)
    logger.info(f"rendered {len(frames)} held-out views at {sorted(sizes)} "
                f"in {time.monotonic() - start:.1f}s -> {root}")


if __name__ == "__main__":
    scene_io.cli(main)
