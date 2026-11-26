"""LOCAL PATCH (augenblick): first-party file, not upstream NeuS2.

Stage 1: stage a COLMAP scene for NeuS2. Normalises the object into the unit sphere,
writes RGBA images (alpha = mask) at the scorer's raster, and the transforms JSONs.
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np

import scene_io
from augenblick.core.errors import SceneError
from augenblick.core.scene import Scene
from augenblick.eval.cameras import camera_records
from augenblick.eval.split import check_split, read_split

logger = logging.getLogger("neus2.prepare")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("-s", "--source_path", type=Path, required=True, help="COLMAP scene")
    parser.add_argument("-m", "--model_path", type=Path, required=True, help="Output directory")
    parser.add_argument("-r", "--resolution", type=int, default=-1,
                        help="Downscale factor; -1 caps the long side at 1600 px")
    parser.add_argument("--bound_percentile", type=float, default=1.0,
                        help="Percentile box of the SfM points that sets the sphere centre")
    parser.add_argument("--bound_margin", type=float, default=1.1, help="Sphere radius multiplier")
    parser.add_argument("--eval", action="store_true", help="Hold out split.json's test views")
    args = parser.parse_args(argv)
    args.source_path, args.model_path = args.source_path.resolve(), args.model_path.resolve()
    return args


def load_points(model_dir: Path) -> np.ndarray:
    import pycolmap

    model = pycolmap.Reconstruction(str(model_dir))
    return np.array([point.xyz for point in model.points3D.values()], dtype=np.float64)


def partition(scene: Scene, stems: list[str], held_out: bool) -> tuple[list[str], list[str]]:
    if not held_out:
        return stems, []
    split = read_split(scene.root)
    if split is None:
        raise SceneError(f"--eval needs {scene.root}/split.json; the augenblick CLI writes it")
    check_split(split, stems)
    return split["train"], split["test"]


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    start = time.monotonic()
    scene = Scene(args.source_path)
    records = camera_records(scene.model_dir)
    by_stem = {record["img_name"]: record for record in records}
    train, test = partition(scene, list(by_stem), args.eval)

    rasters = {stem: scene_io.raster(record, args.resolution) for stem, record in by_stem.items()}
    masks = {stem: scene.masks_dir / f"{stem}.png" for stem in by_stem}
    missing = [stem for stem, path in masks.items() if not path.is_file()]
    if missing:
        raise SceneError(f"{len(missing)} registered images lack masks/<stem>.png (e.g. {missing[:5]})")

    points = load_points(scene.model_dir)
    centre, r_pts = scene_io.robust_bound(points, args.bound_percentile)

    def view_distances(stem: str) -> np.ndarray:
        record = by_stem[stem]
        size, intrinsic = rasters[stem]
        rays = scene_io.pixel_rays(intrinsic, record["rotation"], scene_io.raster_mask(masks[stem], size))
        return scene_io.ray_distances(np.asarray(record["position"]), rays, centre)

    distances = np.concatenate(scene_io.parallel(view_distances, list(by_stem)))
    if len(distances) == 0:
        raise SceneError("every mask is empty; nothing to bound")
    r_mask = float(np.percentile(distances, scene_io.MASK_PERCENTILE))
    radius = args.bound_margin * max(r_pts, r_mask)
    centres = np.array([record["position"] for record in records])
    ratio = scene_io.check_camera_gate(centres, centre, radius)
    miss = float(np.mean(distances > radius))
    logger.info(f"sphere: c={np.round(centre, 4).tolist()} r={radius:.4f} "
                f"(r_pts={r_pts:.4f}, r_mask={r_mask:.4f}); camera ratio {ratio:.2f}; "
                f"mask-ray miss {100 * miss:.3f}%")

    out = args.model_path / scene_io.SCENE_DIR
    (out / "images").mkdir(parents=True, exist_ok=True)

    def stage(stem: str) -> None:
        photo = scene.images_dir / by_stem[stem]["image_name"]
        with scene_io.Image.open(photo) as image:
            if image.size != (by_stem[stem]["width"], by_stem[stem]["height"]):
                raise SceneError(f"{photo.name} is {image.size}, but its COLMAP camera is "
                                 f"{by_stem[stem]['width']}x{by_stem[stem]['height']}")
        scene_io.stage_image(photo, masks[stem], rasters[stem][0], out / "images" / f"{stem}.png")

    # Only training views are loaded by NeuS2; held-out ground truth comes from the photographs.
    scene_io.parallel(stage, sorted(train))

    def frames(stems: list[str]) -> list[dict]:
        return [scene_io.frame_entry(stem, scene_io.normalized_pose(
                    by_stem[stem]["rotation"], by_stem[stem]["position"], centre, radius),
                    rasters[stem][1], rasters[stem][0]) for stem in sorted(stems)]

    scene_io.write_json(out / "transforms_train.json", scene_io.transforms(frames(train)))
    if args.eval:
        scene_io.write_json(out / "transforms_test.json", scene_io.transforms(frames(test)))
    sizes = sorted({rasters[stem][0] for stem in by_stem})
    scene_io.write_json(args.model_path / scene_io.NORMALIZATION, {
        "centre": centre.tolist(), "radius": radius, "r_pts": r_pts, "r_mask": r_mask,
        "bound_percentile": args.bound_percentile, "bound_margin": args.bound_margin,
        "mask_percentile": scene_io.MASK_PERCENTILE, "mask_stride": scene_io.MASK_STRIDE,
        "camera_ratio": ratio, "mask_miss_fraction": miss, "resolution": args.resolution,
        "rasters": [list(size) for size in sizes], "n_train": len(train), "n_test": len(test),
        "ngp_scale": scene_io.NGP_SCALE, "ngp_offset": list(scene_io.NGP_OFFSET)})
    logger.info(f"staged {len(train)} training views ({len(test)} held out) at "
                f"{sizes} in {time.monotonic() - start:.1f}s")


if __name__ == "__main__":
    scene_io.cli(main)
