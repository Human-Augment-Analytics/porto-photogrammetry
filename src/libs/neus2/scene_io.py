"""LOCAL PATCH (augenblick): first-party file, not upstream NeuS2.

CPU-only helpers shared by the stage scripts beside it: the bounding sphere, the
transforms JSONs, RGBA staging, the vertex inverse map, mesh clean-up and held-out
camera intrinsics. Nothing here imports pyngp or torch.
"""
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from augenblick.core.errors import SceneError
from augenblick.eval.mesh_render import calibration, image_size

# NeuS's unit sphere mapped onto NGP's [0,1]^3 cube (tools/data_format_from_neus.py upstream).
NGP_SCALE = 0.5
NGP_OFFSET = (0.5, 0.5, 0.5)
# NeuS assumes every camera sits outside the sphere; demand clear margin, not mere exclusion.
MIN_CAMERA_RATIO = 1.5
MASK_STRIDE = 4
MASK_PERCENTILE = 99.9
NETWORK_CONFIG = Path(__file__).resolve().parent / "configs" / "nerf" / "dtu.json"

STAGE_DIR = "neus2"
SCENE_DIR = "neus2/scene"
NORMALIZATION = "neus2/normalization.json"


def cli(main) -> None:
    """Run a stage's main(), turning a SceneError into exit code 2 like the augenblick CLI."""
    try:
        main()
    except SceneError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)


def workers() -> int:
    return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))


def snapshot_path(model: Path, n_steps: int) -> Path:
    return model / "checkpoints" / f"{n_steps}.msgpack"


def raster_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    """The scorer's mask reader (eval/nvs.py): NEAREST to the raster, then > 127."""
    with Image.open(path) as mask:
        return np.asarray(mask.convert("L").resize(size, Image.NEAREST)) > 127


def canonical_rgb(photo: Path, size: tuple[int, int]) -> Image.Image:
    """The scorer's canonical ground truth (eval/nvs.py): RGB, BICUBIC to the raster."""
    with Image.open(photo) as image:
        return image.convert("RGB").resize(size, Image.Resampling.BICUBIC)


def robust_bound(points: np.ndarray, percentile: float) -> tuple[np.ndarray, float]:
    """Centre and half-diagonal of the percentile box, so far outlier points cannot inflate it."""
    if len(points) == 0:
        raise SceneError("the SfM model has no 3D points to bound the object")
    lo = np.percentile(points, percentile, axis=0)
    hi = np.percentile(points, 100.0 - percentile, axis=0)
    return (lo + hi) / 2.0, float(0.5 * np.linalg.norm(hi - lo))


def pixel_rays(intrinsic: np.ndarray, rotation: np.ndarray, mask: np.ndarray,
               stride: int = MASK_STRIDE) -> np.ndarray:
    """Unit world-space directions through the centres of foreground pixels, every stride-th."""
    ys, xs = np.nonzero(mask[::stride, ::stride])
    u, v = xs * stride + 0.5, ys * stride + 0.5
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    local = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, dtype=np.float64)], axis=1)
    world = local @ np.asarray(rotation, dtype=np.float64).T
    return world / np.linalg.norm(world, axis=1, keepdims=True)


def ray_distances(origin: np.ndarray, directions: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """Closest approach of each forward ray to the centre."""
    offset = centre - origin
    t = np.maximum(directions @ offset, 0.0)
    return np.linalg.norm(origin + t[:, None] * directions - centre, axis=1)


def check_camera_gate(centres: np.ndarray, centre: np.ndarray, radius: float) -> float:
    """Return min camera distance / radius, refusing a sphere that reaches the cameras."""
    ratio = float(np.linalg.norm(centres - centre, axis=1).min() / radius)
    if ratio < MIN_CAMERA_RATIO:
        raise SceneError(
            f"bounding sphere (r={radius:.4g}) reaches within {ratio:.2f}r of a camera; "
            f"NeuS2 needs every camera beyond {MIN_CAMERA_RATIO}r. Check the SfM points "
            "and masks, or lower --bound_margin")
    return ratio


def normalized_pose(rotation, position, centre: np.ndarray, radius: float) -> np.ndarray:
    """OpenCV camera-to-world with the translation mapped into the unit sphere (from_na)."""
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(rotation, dtype=np.float64)
    pose[:3, 3] = (np.asarray(position, dtype=np.float64) - centre) / radius
    return pose


def ngp_to_world(vertices: np.ndarray, centre, radius: float) -> np.ndarray:
    """Invert scale 0.5 / offset 0.5 and the sphere normalisation, back to the SfM frame."""
    unit = (np.asarray(vertices, dtype=np.float64) - np.asarray(NGP_OFFSET)) / NGP_SCALE
    return np.asarray(centre, dtype=np.float64) + radius * unit


def frame_entry(stem: str, pose: np.ndarray, intrinsic: np.ndarray, size: tuple[int, int]) -> dict:
    """One frame with all six intrinsics; NeuS2 ignores top-level cx/cy (nerf_loader.cu:708)."""
    return {"file_path": f"images/{stem}.png", "transform_matrix": pose.tolist(),
            "fl_x": float(intrinsic[0, 0]), "fl_y": float(intrinsic[1, 1]),
            "cx": float(intrinsic[0, 2]), "cy": float(intrinsic[1, 2]),
            "w": int(size[0]), "h": int(size[1])}


def transforms(frames: list[dict]) -> dict:
    """from_na is read by key presence alone (nerf_loader.cu:346), so it is only ever true."""
    return {"from_na": True, "aabb_scale": 1, "scale": NGP_SCALE, "offset": list(NGP_OFFSET),
            "frames": frames}


def held_out_camera(intrinsic: np.ndarray, size: tuple[int, int]) -> tuple[list[float], list[float]]:
    """fov_xy (degrees, fov_axis 1) and screen_center reproducing set_camera_to_training_view."""
    width, height = size
    relative = (intrinsic[0, 0] / height, intrinsic[1, 1] / height)
    fov_xy = [math.degrees(2.0 * math.atan(1.0 / (2.0 * f))) for f in relative]
    screen_center = [1.0 - intrinsic[0, 2] / width, 1.0 - intrinsic[1, 2] / height]
    return fov_xy, screen_center


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Upstream scripts/common.py's transfer, so renders match its evaluation."""
    return np.where(image > 0.0031308, 1.055 * np.power(np.maximum(image, 0), 1 / 2.4) - 0.055,
                    12.92 * image)


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    return np.where(image > 0.04045, np.power((image + 0.055) / 1.055, 2.4), image / 12.92)


def keep_largest_clusters(triangles: np.ndarray, cluster_ids: np.ndarray, n_cluster: int) -> np.ndarray:
    """2DGS post_process_mesh's rule: keep clusters at least the n-th largest, and >= 50 faces."""
    if n_cluster <= 0 or len(triangles) == 0:
        return np.ones(len(triangles), dtype=bool)
    counts = np.bincount(cluster_ids)
    threshold = max(np.sort(counts)[-min(n_cluster, len(counts))], 50)
    return counts[cluster_ids] >= threshold


def read_json(path: Path) -> dict:
    with Path(path).open() as handle:
        return json.load(handle)


def write_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def raster(record: dict, resolution: int) -> tuple[tuple[int, int], np.ndarray]:
    """The scorer's raster and the camera's intrinsics at it; non-pinhole cameras raise."""
    size = image_size(record, resolution)
    intrinsic, _ = calibration(record, size)
    return size, intrinsic


def stage_image(photo: Path, mask_path: Path, size: tuple[int, int], out: Path) -> None:
    """RGB exactly as the scorer reads ground truth, alpha the raster mask."""
    rgb = canonical_rgb(photo, size)
    alpha = Image.fromarray((raster_mask(mask_path, size) * 255).astype(np.uint8), mode="L")
    rgba = rgb.copy()
    rgba.putalpha(alpha)
    # Level 1: the staged files are deleted once the run succeeds.
    rgba.save(out, compress_level=1)


def record_native_selection(scene_root: Path, model_path: Path, train: list[dict],
                            test: list[dict], test_sizes: dict, resolution: int) -> None:
    """The scorer's selection manifest (eval/selection.py): the views this process consumed.

    Args:
        train: transforms_train.json frames, at their staged raster.
        test: transforms_test.json frames.
        test_sizes: stem -> (width, height) each held-out view was rendered at.
        resolution: The evaluation downscale, which names the manifest.
    """
    from types import SimpleNamespace

    from augenblick.eval.selection import record_selection

    def views(frames, sizes=None):
        out = []
        for frame in frames:
            stem = Path(frame["file_path"]).stem
            width, height = sizes[stem] if sizes else (frame["w"], frame["h"])
            out.append(SimpleNamespace(image_name=stem, width=width, height=height))
        return out

    record_selection(scene_root, model_path, views(train), views(test, test_sizes), resolution)


def parallel(function, items) -> list:
    with ThreadPoolExecutor(max_workers=workers()) as pool:
        return list(pool.map(function, items))
