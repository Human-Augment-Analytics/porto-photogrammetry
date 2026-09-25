"""Per-camera colour calibration from corresponding 24-patch chart images."""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
RECTIFIED_SIZE = (600, 400)
PATCH_ROWS = 4
PATCH_COLUMNS = 6


class ColorCalibrationError(ValueError):
    """The chart configuration or source images cannot produce a safe transform."""


@dataclass(frozen=True)
class CameraChart:
    """One camera's chart image and four source-image corners."""

    reference_image: Path
    corners: np.ndarray


@dataclass(frozen=True)
class CalibrationConfig:
    """Relative reference camera and chart information for every camera group."""

    reference_camera: str
    camera_regex: str
    cameras: dict[str, CameraChart]
    ridge: float = 1e-6


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """Convert normalized sRGB values to linear RGB."""
    values = np.asarray(values, dtype=np.float32)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    """Convert linear RGB values to normalized sRGB."""
    values = np.asarray(values, dtype=np.float32)
    return np.where(
        values <= 0.0031308,
        12.92 * values,
        1.055 * np.maximum(values, 0.0) ** (1.0 / 2.4) - 0.055,
    )


def load_config(path: Path) -> CalibrationConfig:
    """Load and validate a calibration JSON file."""
    data = json.loads(path.read_text())
    cameras = {}
    for raw_name, camera in data.get("cameras", {}).items():
        name = raw_name.lower()
        if name in cameras:
            raise ColorCalibrationError(f"duplicate camera name after normalization: {name}")
        corners = np.asarray(camera.get("corners", []), dtype=np.float32)
        if corners.shape != (4, 2):
            raise ColorCalibrationError(f"{name}: corners must contain four [x, y] points")
        reference_image = Path(camera["reference_image"])
        if reference_image.is_absolute() or ".." in reference_image.parts:
            raise ColorCalibrationError(f"{name}: reference_image must stay inside input")
        cameras[name] = CameraChart(reference_image, corners)

    reference_camera = str(data.get("reference_camera", "")).lower()
    if not cameras:
        raise ColorCalibrationError("config has no cameras")
    if reference_camera not in cameras:
        raise ColorCalibrationError("reference_camera must name an entry in cameras")

    ridge = float(data.get("ridge", 1e-6))
    if ridge < 0:
        raise ColorCalibrationError("ridge must be non-negative")

    return CalibrationConfig(
        reference_camera=reference_camera,
        camera_regex=data.get("camera_regex", r"camera\d+"),
        cameras=cameras,
        ridge=ridge,
    )


def rectify_chart(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp a chart quadrilateral to the standard analysis canvas."""
    width, height = RECTIFIED_SIZE
    destination = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), destination)
    return cv2.warpPerspective(image, transform, (width, height))


def _cluster_1d(values: np.ndarray, count: int) -> np.ndarray:
    """Cluster well-separated grid coordinates without an extra dependency."""
    centers = np.linspace(float(values.min()), float(values.max()), count)
    for _ in range(30):
        labels = np.abs(values[:, None] - centers[None, :]).argmin(axis=1)
        updated = centers.copy()
        for index in range(count):
            members = values[labels == index]
            if len(members):
                updated[index] = float(np.median(members))
        if np.allclose(updated, centers, atol=0.01):
            break
        centers = updated
    return np.sort(centers)


def locate_patch_centers(chart: np.ndarray) -> np.ndarray:
    """Locate the 4×6 patch grid on a rectified chart.

    The dark frame is removed with a brightness/saturation mask. Missing
    components are reconstructed from a bilinear grid fit.
    """
    height, width = chart.shape[:2]
    hsv = cv2.cvtColor(chart, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(chart, cv2.COLOR_BGR2GRAY)
    foreground = ((gray > 55) | (hsv[:, :, 1] > 45)).astype(np.uint8) * 255
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )
    count, _, stats, centroids = cv2.connectedComponentsWithStats(foreground)

    candidates = []
    min_area = width * height * 0.004
    max_area = width * height * 0.04
    for index in range(1, count):
        _, _, component_width, component_height, area = stats[index]
        if not min_area <= area <= max_area:
            continue
        if not 0.05 * width <= component_width <= 0.18 * width:
            continue
        if not 0.08 * height <= component_height <= 0.25 * height:
            continue
        candidates.append(centroids[index])

    points = np.asarray(candidates, dtype=np.float32)
    if len(points) < 16:
        raise ColorCalibrationError(
            f"found only {len(points)} patch interiors; check chart corners or lighting"
        )

    column_centers = _cluster_1d(points[:, 0], PATCH_COLUMNS)
    row_centers = _cluster_1d(points[:, 1], PATCH_ROWS)
    assignments = {}
    for point in points:
        column = int(np.abs(column_centers - point[0]).argmin())
        row = int(np.abs(row_centers - point[1]).argmin())
        expected = np.array([column_centers[column], row_centers[row]])
        distance = float(np.linalg.norm(point - expected))
        previous = assignments.get((row, column))
        if previous is None or distance < previous[0]:
            assignments[(row, column)] = (distance, point)

    if len(assignments) < 16:
        raise ColorCalibrationError(
            f"only {len(assignments)} unique grid cells were detected; check chart corners"
        )

    design = []
    observed_x = []
    observed_y = []
    for (row, column), (_, point) in assignments.items():
        design.append([1.0, column, row, column * row])
        observed_x.append(point[0])
        observed_y.append(point[1])
    design_array = np.asarray(design, dtype=np.float32)
    coefficient_x = np.linalg.lstsq(design_array, observed_x, rcond=None)[0]
    coefficient_y = np.linalg.lstsq(design_array, observed_y, rcond=None)[0]

    fitted = []
    residuals = []
    for row in range(PATCH_ROWS):
        for column in range(PATCH_COLUMNS):
            basis = np.array([1.0, column, row, column * row], dtype=np.float32)
            fitted.append([basis @ coefficient_x, basis @ coefficient_y])
    for (row, column), (_, point) in assignments.items():
        index = row * PATCH_COLUMNS + column
        residuals.append(np.linalg.norm(point - fitted[index]))

    if float(np.median(residuals)) > 15.0:
        raise ColorCalibrationError("detected patch grid is inconsistent; review chart corners")

    centers = np.asarray(fitted, dtype=np.float32)
    if (
        np.any(centers[:, 0] < 0)
        or np.any(centers[:, 0] >= width)
        or np.any(centers[:, 1] < 0)
        or np.any(centers[:, 1] >= height)
    ):
        raise ColorCalibrationError("fitted patch centers extend outside the rectified chart")
    return centers


def sample_patches(chart: np.ndarray, centers: np.ndarray, radius: int = 12) -> np.ndarray:
    """Return median normalized RGB values for the 24 patch interiors."""
    samples = []
    for x_float, y_float in centers:
        x, y = round(float(x_float)), round(float(y_float))
        region = chart[y - radius : y + radius + 1, x - radius : x + radius + 1]
        if region.shape[:2] != (2 * radius + 1, 2 * radius + 1):
            raise ColorCalibrationError("a patch sample extends outside the chart")
        samples.append(np.median(region, axis=(0, 1))[::-1])
    return np.asarray(samples, dtype=np.float32) / 255.0


def fit_color_matrix(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Fit a regularized 3×3 transform in linear RGB."""
    source = srgb_to_linear(source_rgb)
    target = srgb_to_linear(target_rgb)
    identity = np.eye(3, dtype=np.float32)
    return np.linalg.solve(
        source.T @ source + ridge * identity,
        source.T @ target + ridge * identity,
    ).astype(np.float32)


def apply_color_matrix(
    image: np.ndarray,
    matrix: np.ndarray,
    chunk_rows: int = 256,
) -> np.ndarray:
    """Apply a linear-RGB matrix in row chunks to bound peak memory."""
    output = np.empty_like(image)
    for start in range(0, image.shape[0], chunk_rows):
        stop = min(start + chunk_rows, image.shape[0])
        rgb = image[start:stop, :, ::-1].astype(np.float32) / 255.0
        corrected = linear_to_srgb(srgb_to_linear(rgb.reshape(-1, 3)) @ matrix)
        corrected = np.clip(corrected, 0.0, 1.0).reshape(rgb.shape)
        output[start:stop] = np.round(corrected[:, :, ::-1] * 255.0).astype(np.uint8)
    return output


def delta_e76(first_rgb: np.ndarray, second_rgb: np.ndarray) -> np.ndarray:
    """Compute CIE76 distance for corresponding normalized RGB samples."""
    first_lab = cv2.cvtColor(
        first_rgb.reshape(1, -1, 3).astype(np.float32),
        cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3)
    second_lab = cv2.cvtColor(
        second_rgb.reshape(1, -1, 3).astype(np.float32),
        cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3)
    return np.linalg.norm(first_lab - second_lab, axis=1)


def camera_name(path: Path, pattern: re.Pattern[str]) -> str | None:
    """Extract a camera group from a capture filename."""
    match = pattern.search(path.name)
    return match.group(0).lower() if match else None


def _write_image(path: Path, image: np.ndarray, source: Path) -> None:
    """Write corrected pixels while retaining JPEG EXIF metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        from PIL import Image

        with Image.open(source) as original:
            exif = original.info.get("exif")
        options = {"quality": 95, "subsampling": 0}
        if exif:
            options["exif"] = exif
        Image.fromarray(image[:, :, ::-1]).save(path, **options)
    elif not cv2.imwrite(str(path), image):
        raise ColorCalibrationError(f"could not write image: {path}")


def _reference_samples(
    input_dir: Path,
    config: CalibrationConfig,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load each reference and return images, samples, and detected centers."""
    images = {}
    samples = {}
    centers = {}
    for name, chart in config.cameras.items():
        path = input_dir / chart.reference_image
        image = cv2.imread(str(path))
        if image is None:
            raise ColorCalibrationError(f"could not read {name} reference image: {path}")
        rectified = rectify_chart(image, chart.corners)
        patch_centers = locate_patch_centers(rectified)
        images[name] = image
        centers[name] = patch_centers
        samples[name] = sample_patches(rectified, patch_centers)
    return images, samples, centers


def calibrate_directory(
    input_dir: Path,
    output_dir: Path,
    config: CalibrationConfig,
    overwrite: bool = False,
) -> dict:
    """Fit per-camera transforms and calibrate a directory tree.

    Reference chart images are used for fitting but excluded from the output.
    Matching mask files are copied unchanged.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise ColorCalibrationError(f"input directory does not exist: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ColorCalibrationError("output directory must not be inside the input tree")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise ColorCalibrationError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_images, samples, centers = _reference_samples(input_dir, config)
    reference_samples = samples[config.reference_camera]
    matrices = {}
    metrics = {}
    for name, camera_samples in samples.items():
        matrix = (
            np.eye(3, dtype=np.float32)
            if name == config.reference_camera
            else fit_color_matrix(camera_samples, reference_samples, config.ridge)
        )
        corrected_samples = linear_to_srgb(srgb_to_linear(camera_samples) @ matrix)
        before = delta_e76(camera_samples, reference_samples)
        after = delta_e76(np.clip(corrected_samples, 0.0, 1.0), reference_samples)
        matrices[name] = matrix
        metrics[name] = {
            "mean_delta_e76_before": float(before.mean()),
            "median_delta_e76_before": float(np.median(before)),
            "mean_delta_e76_after": float(after.mean()),
            "median_delta_e76_after": float(np.median(after)),
        }

        preview_dir = output_dir / "calibration_previews"
        preview_dir.mkdir(exist_ok=True)
        rectified = rectify_chart(reference_images[name], config.cameras[name].corners)
        cv2.imwrite(str(preview_dir / f"{name}_chart.png"), rectified)

    pattern = re.compile(config.camera_regex, re.IGNORECASE)
    reference_paths = {
        (input_dir / camera.reference_image).resolve() for camera in config.cameras.values()
    }
    reference_masks = {
        (path.parent, path.stem.lower()) for path in reference_paths
    }
    counts = {name: 0 for name in config.cameras}
    clipping = {
        name: {"pixels": 0, "before": 0, "after": 0} for name in config.cameras
    }
    skipped = []

    for source in sorted(input_dir.rglob("*")):
        if not source.is_file() or source.resolve() in reference_paths:
            continue
        relative = source.relative_to(input_dir)
        if source.name.lower().endswith(".mask.png"):
            name = camera_name(source, pattern)
            is_reference_mask = any(
                source.parent.resolve() == parent
                and source.name.lower().startswith(f"{stem}.")
                for parent, stem in reference_masks
            )
            if name not in matrices or is_reference_mask:
                continue
            destination = output_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            continue
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        name = camera_name(source, pattern)
        if name is None or name not in matrices:
            skipped.append(str(relative))
            continue
        image = cv2.imread(str(source))
        if image is None:
            skipped.append(str(relative))
            continue
        destination = output_dir / relative
        if name == config.reference_camera:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            corrected = image
        else:
            corrected = apply_color_matrix(image, matrices[name])
            _write_image(destination, corrected, source)

        total = image.shape[0] * image.shape[1]
        clipping[name]["pixels"] += total
        clipping[name]["before"] += int(np.any((image == 0) | (image == 255), axis=2).sum())
        clipping[name]["after"] += int(
            np.any((corrected == 0) | (corrected == 255), axis=2).sum()
        )
        counts[name] += 1

    for values in clipping.values():
        pixels = values.pop("pixels")
        values["percent_before"] = 100.0 * values.pop("before") / pixels if pixels else 0.0
        values["percent_after"] = 100.0 * values.pop("after") / pixels if pixels else 0.0

    report = {
        "reference_camera": config.reference_camera,
        "camera_regex": config.camera_regex,
        "ridge": config.ridge,
        "matrices": {name: matrix.tolist() for name, matrix in matrices.items()},
        "metrics": metrics,
        "patch_centers_rectified": {
            name: value.tolist() for name, value in centers.items()
        },
        "processed_images": counts,
        "clipping": clipping,
        "skipped_images": skipped,
    }
    report_path = output_dir / "color_calibration_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Wrote colour calibration report to %s", report_path)
    return report
