"""Independent metric scale from printed scale bars."""
from __future__ import annotations

import hashlib
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

logger = logging.getLogger(__name__)

KNOWN_LENGTH_MM = 100.0
WINDOW_SIZE = 24
WINDOW_STRIDE = 8
MIN_OBSERVATIONS = 16
WITHHELD = 8
MAX_ERROR_PX = 2.0
MIN_CONSENSUS = 0.6
MIN_RAY_ANGLE_DEG = 2.0
MIN_INLIERS = 8
HELD_OUT_CONSENSUS = 0.6
MIN_PASSING_WINDOWS = 6
MIN_DISTINCT_RINGS = 2
MIN_DISTINCT_OFFSETS = 3
MAX_PRIMARY_SPREAD = 0.005
MAX_CHECK_ERROR = 0.01


def colmap_pixel(opencv_pixel) -> np.ndarray:
    """Convert an OpenCV pixel coordinate to COLMAP's convention."""
    return np.asarray(opencv_pixel, dtype=float) + 0.5


@dataclass(frozen=True)
class Observation:
    """One accepted detection of one target in one image."""

    camera: object
    pose: np.ndarray
    xy: np.ndarray
    image: str


def _linear_point(observations: list[Observation]) -> np.ndarray:
    """Direct linear triangulation from calibrated rays; the hypothesis step."""
    rows = []
    for obs in observations:
        ray = obs.camera.cam_from_img(obs.xy)
        if ray is None or not np.isfinite(ray).all():
            raise ValueError(f"Cannot unproject {obs.image}")
        rows.extend((ray[0] * obs.pose[2] - obs.pose[0], ray[1] * obs.pose[2] - obs.pose[1]))
    _, _, vt = np.linalg.svd(np.asarray(rows))
    if abs(vt[-1, 3]) < 1e-12:
        raise ValueError("Triangulation at infinity")
    return vt[-1, :3] / vt[-1, 3]


def reprojection_residuals(point: np.ndarray, observations: list[Observation]) -> np.ndarray:
    """Per-observation image residuals of a 3D point, flattened to (2n,)."""
    result = []
    for obs in observations:
        camera_point = obs.pose[:, :3] @ point + obs.pose[:, 3]
        projected = obs.camera.img_from_cam(camera_point)
        if camera_point[2] <= 0 or projected is None or not np.isfinite(projected).all():
            result.extend((1e6, 1e6))
        else:
            result.extend(projected - obs.xy)
    return np.asarray(result)


def triangulate_track(observations: list[Observation], *, max_error: float = MAX_ERROR_PX,
                      minimum: int = MIN_INLIERS, min_consensus: float = MIN_CONSENSUS,
                      min_ray_angle_deg: float = MIN_RAY_ANGLE_DEG) -> dict:
    """Robustly triangulate one target from its accepted observations."""
    if len(observations) < minimum:
        raise ValueError(f"Only {len(observations)} views; need {minimum}")
    pairs = list(itertools.combinations(range(len(observations)), 2))
    rng = np.random.default_rng(0)
    if len(pairs) > 256:
        pairs = [pairs[i] for i in rng.choice(len(pairs), 256, replace=False)]
    best, best_key = None, (-1, -np.inf)
    for a, b in pairs:
        try:
            point = _linear_point([observations[a], observations[b]])
        except ValueError:
            continue
        errors = np.linalg.norm(
            reprojection_residuals(point, observations).reshape(-1, 2), axis=1)
        inliers = errors <= max_error
        count = int(inliers.sum())
        key = (count, -float(np.median(errors[inliers])) if count else -np.inf)
        if key > best_key:
            best, best_key = (point, inliers), key
    if best is None or best_key[0] < minimum:
        raise ValueError("No triangulation with enough reprojection-consistent views")
    point, inliers = best
    for _ in range(5):
        retained = [o for o, keep in zip(observations, inliers) if keep]
        fitted = least_squares(reprojection_residuals, point, args=(retained,),
                               loss="soft_l1", f_scale=1.0)
        if not fitted.success or not np.isfinite(fitted.x).all():
            raise ValueError(f"Point refinement failed: {fitted.message}")
        point = fitted.x
        errors = np.linalg.norm(
            reprojection_residuals(point, observations).reshape(-1, 2), axis=1)
        updated = errors <= max_error
        if np.array_equal(updated, inliers):
            break
        inliers = updated
        if inliers.sum() < minimum:
            raise ValueError("Refinement lost the minimum number of inliers")
    else:
        raise ValueError("Triangulation inlier set did not stabilise")
    if inliers.sum() < minimum or inliers.mean() < min_consensus:
        raise ValueError("Track has insufficient cross-view consensus")
    centres = np.array([-o.pose[:, :3].T @ o.pose[:, 3]
                        for o, keep in zip(observations, inliers) if keep])
    rays = point - centres
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    angle = float(np.degrees(np.arccos(np.clip((rays @ rays.T).min(), -1, 1))))
    if angle < min_ray_angle_deg:
        raise ValueError(f"Triangulation angle too small: {angle:.4f} degrees")
    return {
        "xyz": point.tolist(),
        "observations": len(observations),
        "inliers": int(inliers.sum()),
        "inlier_images": [o.image for o, keep in zip(observations, inliers) if keep],
        "rejected_images": [o.image for o, keep in zip(observations, inliers) if not keep],
        "median_error_px": float(np.median(errors[inliers])),
        "max_error_px": float(errors[inliers].max()),
        "max_ray_angle_deg": angle,
    }


@dataclass(frozen=True)
class Gates:
    """The predeclared protocol; every field is fixed before results are seen."""

    window_size: int = WINDOW_SIZE
    stride: int = WINDOW_STRIDE
    min_observations: int = MIN_OBSERVATIONS
    withheld: int = WITHHELD
    max_error_px: float = MAX_ERROR_PX
    min_consensus: float = MIN_CONSENSUS
    min_ray_angle_deg: float = MIN_RAY_ANGLE_DEG
    min_inliers: int = MIN_INLIERS
    held_out_consensus: float = HELD_OUT_CONSENSUS
    min_passing_windows: int = MIN_PASSING_WINDOWS
    min_distinct_rings: int = MIN_DISTINCT_RINGS
    min_distinct_offsets: int = MIN_DISTINCT_OFFSETS
    max_primary_spread: float = MAX_PRIMARY_SPREAD
    max_check_error: float = MAX_CHECK_ERROR
    known_length_mm: float = KNOWN_LENGTH_MM

    def __post_init__(self):
        if self.min_observations - self.withheld < self.min_inliers:
            raise ValueError("Training observations would fall below the inlier minimum")


@dataclass(frozen=True)
class BarTargets:
    """The four target codes defining the two bars, in the detector's own numbering."""

    primary: tuple[str, str]
    check: tuple[str, str]

    def codes(self) -> tuple[str, ...]:
        return (*self.primary, *self.check)


def _window_markers(tracks: dict[str, dict[str, Observation]], names: list[str],
                    gates: Gates) -> dict:
    """Evaluate all four markers inside one window of image names."""
    markers = {}
    for code, observations in tracks.items():
        selected = sorted((observations[n] for n in names if n in observations),
                          key=lambda o: o.image)
        if len(selected) < gates.min_observations:
            markers[code] = {"passed": False,
                             "error": f"only {len(selected)} observations; "
                                      f"need {gates.min_observations}"}
            continue
        test_indices = set(
            np.rint(np.linspace(0, len(selected) - 1, gates.withheld)).astype(int))
        if len(test_indices) != gates.withheld:
            raise ValueError("Holdout indices are not unique")
        train = [o for i, o in enumerate(selected) if i not in test_indices]
        test = [o for i, o in enumerate(selected) if i in test_indices]
        try:
            fitted = triangulate_track(
                train, max_error=gates.max_error_px, minimum=gates.min_inliers,
                min_consensus=gates.min_consensus,
                min_ray_angle_deg=gates.min_ray_angle_deg)
        except ValueError as exc:
            markers[code] = {"passed": False, "error": str(exc)}
            continue
        errors = np.linalg.norm(reprojection_residuals(
            np.array(fitted["xyz"]), test).reshape(-1, 2), axis=1)
        passed = bool(np.mean(errors <= gates.max_error_px) >= gates.held_out_consensus)
        markers[code] = {
            **fitted, "passed": passed, "held_out_n": len(test),
            "held_out_inliers": int(np.sum(errors <= gates.max_error_px)),
            "held_out_errors_px": errors.tolist(),
            "held_out_images": [o.image for o in test],
        }
    return markers


def sliding_window_scale(tracks: dict[str, dict[str, Observation]],
                         rings: list[list[str]], bars: BarTargets,
                         gates: Gates = Gates()) -> dict:
    """Estimate scale from every window of every ring, then apply the acceptance gates."""
    missing = [code for code in bars.codes() if code not in tracks]
    if missing:
        raise ValueError(f"No observations for target codes {missing}")
    windows = []
    for ring_index, ring in enumerate(rings):
        for offset in range(0, len(ring), gates.stride):
            names = [ring[(offset + i) % len(ring)] for i in range(gates.window_size)]
            window = {"ring": ring_index, "offset": offset,
                      "markers": _window_markers(
                          {code: tracks[code] for code in bars.codes()}, names, gates)}
            window["passed"] = all(m["passed"] for m in window["markers"].values())
            if window["passed"]:
                xyz = {code: np.array(m["xyz"]) for code, m in window["markers"].items()}
                window["primary_length_units"] = float(
                    np.linalg.norm(xyz[bars.primary[0]] - xyz[bars.primary[1]]))
                window["check_length_units"] = float(
                    np.linalg.norm(xyz[bars.check[0]] - xyz[bars.check[1]]))
            windows.append(window)
    passing = [w for w in windows if w["passed"]]
    checks: dict = {"passing_windows": len(passing)}
    from dataclasses import asdict
    result = {
        "protocol": "sliding_window_scale",
        "predeclared": asdict(gates) | {"primary": list(bars.primary),
                                        "check": list(bars.check)},
        "windows": windows,
        "acceptance_checks": checks,
        "accepted": False,
    }
    if not passing:
        return result
    primary = np.array([w["primary_length_units"] for w in passing])
    check = np.array([w["check_length_units"] for w in passing])
    scale = gates.known_length_mm / float(np.median(primary))
    check_mm = check * scale
    checks.update({
        "distinct_rings": len({w["ring"] for w in passing}),
        "distinct_offsets": len({w["offset"] for w in passing}),
        "primary_relative_spread": float((primary.max() - primary.min())
                                         / np.median(primary)),
        "check_bar_relative_error_max": float(
            np.abs(check_mm - gates.known_length_mm).max() / gates.known_length_mm),
        "check_bar_relative_error_median": float(
            abs(np.median(check_mm) - gates.known_length_mm) / gates.known_length_mm),
    })
    result.update({
        "scale_mm_per_model_unit": scale,
        "scale_mm_per_model_unit_range": [gates.known_length_mm / float(primary.max()),
                                          gates.known_length_mm / float(primary.min())],
        "primary_bar_lengths_mm": (primary * scale).tolist(),
        "check_bar_lengths_mm": check_mm.tolist(),
    })
    result["accepted"] = bool(
        len(passing) >= gates.min_passing_windows
        and checks["distinct_rings"] >= gates.min_distinct_rings
        and checks["distinct_offsets"] >= gates.min_distinct_offsets
        and checks["primary_relative_spread"] <= gates.max_primary_spread
        and checks["check_bar_relative_error_max"] <= gates.max_check_error
        and checks["check_bar_relative_error_median"] <= gates.max_check_error)
    return result


def model_hashes(model_dir: Path) -> dict[str, str]:
    """SHA-256 of each COLMAP binary file, identifying the gauge a scale belongs to."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(model_dir).glob("*.bin"))}


def apply_scale(source: Path, destination: Path, scale: float) -> dict:
    """Write a metrically scaled copy of a mesh, preserving topology and colours."""
    import open3d as o3d

    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    mesh = o3d.io.read_triangle_mesh(str(source))
    if len(mesh.triangles) == 0:
        raise ValueError("Mesh contains no triangles")
    vertices = np.asarray(mesh.vertices).copy()
    triangles = np.asarray(mesh.triangles).copy()
    colours = np.asarray(mesh.vertex_colors).copy()
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(vertices).all():
        raise ValueError("Invalid scale or vertices")
    mesh.scale(scale, center=(0.0, 0.0, 0.0))
    if not o3d.io.write_triangle_mesh(str(destination), mesh):
        raise OSError(f"Mesh export failed: {destination}")
    reloaded = o3d.io.read_triangle_mesh(str(destination))
    np.testing.assert_allclose(np.asarray(reloaded.vertices), vertices * scale,
                               rtol=1e-6, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(reloaded.triangles), triangles)
    np.testing.assert_allclose(np.asarray(reloaded.vertex_colors), colours, atol=1 / 255)
    return {
        "units": "mm",
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "scale_mm_per_model_unit": scale,
        "vertices": len(vertices),
        "triangles": len(triangles),
        "extent_mm": np.ptp(np.asarray(reloaded.vertices), axis=0).tolist(),
    }


def rings_from_exif(scene_dir: Path, image_names: list[str],
                    positions: dict[str, str] | None = None) -> list[list[str]]:
    """Group images into camera rings by physical body, lens and turntable position."""
    from PIL import Image

    groups: dict[tuple, list[str]] = {}
    for name in image_names:
        with Image.open(Path(scene_dir) / "images" / name) as photo:
            tags = photo.getexif().get_ifd(34665)
            serial, lens, focal = tags.get(42033), tags.get(42037), tags.get(37386)
        if not serial or not lens or focal is None:
            raise ValueError(f"Missing physical-camera metadata: {name}")
        key = (str(serial), str(lens), float(focal),
               positions.get(name) if positions else None)
        groups.setdefault(key, []).append(name)
    return [sorted(names) for _, names in sorted(groups.items())]


def _load_tracks(detections_path: Path, model, codes: tuple[str, ...]
                 ) -> dict[str, dict[str, Observation]]:
    """Build per-code observation maps from a detections JSON and a reconstruction."""
    import json

    records = json.loads(Path(detections_path).read_text())
    by_name = {image.name: image for image in model.images.values()}
    unregistered = [r["image"] for r in records if r["image"] not in by_name]
    if unregistered:
        raise ValueError(f"{len(unregistered)} detection images are not registered: "
                         f"{unregistered[:5]}")
    tracks: dict[str, dict[str, Observation]] = {code: {} for code in codes}
    for record in records:
        image = by_name[record["image"]]
        pose = image.cam_from_world
        pose = pose() if callable(pose) else pose
        for code in codes:
            candidates = [d for d in record["raw_detections"] if str(d["code"]) == code]
            if len(candidates) != 1:
                if len(candidates) > 1:
                    logger.warning("Rejecting duplicate code %s in %s",
                                   code, record["image"])
                continue
            d = candidates[0]
            tracks[code][record["image"]] = Observation(
                model.cameras[image.camera_id], pose.matrix(),
                colmap_pixel([d["x"], d["y"]]), record["image"])
    return tracks


def main(argv: list[str] | None = None) -> int:
    """Estimate, check and optionally apply an independent metric scale."""
    import argparse
    import json

    import pycolmap

    parser = argparse.ArgumentParser(
        prog="python -m augenblick.eval.scale",
        description="Recover metric scale from printed scale bars, with predeclared "
                    "acceptance gates.")
    parser.add_argument("--model", required=True, type=Path,
                        help="COLMAP sparse model directory (the gauge the scale is for)")
    parser.add_argument("--scene", required=True, type=Path,
                        help="Scene directory with images/ and capture_manifest.json")
    parser.add_argument("--detections", required=True, type=Path,
                        help="JSON of per-image target detections in OpenCV pixels")
    parser.add_argument("--primary", required=True, nargs=2, metavar="CODE",
                        help="Detector codes of the primary bar's two targets; the pairing "
                             "of codes to printed labels must be verified on a photograph")
    parser.add_argument("--check", required=True, nargs=2, metavar="CODE",
                        help="Detector codes of the withheld check bar's two targets")
    parser.add_argument("--known-length-mm", type=float, default=KNOWN_LENGTH_MM,
                        help="Printed centre-to-centre distance of each bar")
    parser.add_argument("--apply-to", type=Path,
                        help="Mesh in the model's gauge to export in millimetres; only "
                             "written when the scale is accepted")
    parser.add_argument("--output", type=Path,
                        help="Metric mesh path; defaults to <apply-to stem>_mm.ply")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    model = pycolmap.Reconstruction(str(args.model))
    if not model.is_valid():
        raise SystemExit("COLMAP model fails structural validation")
    manifest_path = args.scene / "capture_manifest.json"
    positions = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        positions = {Path(entry["image"]).name: entry["position"] for entry in manifest}
    bars = BarTargets(primary=tuple(args.primary), check=tuple(args.check))
    tracks = _load_tracks(args.detections, model, bars.codes())
    # Rings cover every image the detector examined, not only those with accepted
    # detections, so a ring the decoder cannot read appears as recorded failures
    # instead of silently vanishing from the protocol.
    examined = sorted({record["image"]
                       for record in json.loads(args.detections.read_text())})
    rings = rings_from_exif(args.scene, examined, positions)
    gates = Gates(known_length_mm=args.known_length_mm)
    result = sliding_window_scale(tracks, rings, bars, gates)
    result["model"] = str(args.model)
    result["model_sha256"] = model_hashes(args.model)
    result["detections"] = str(args.detections)

    checks = result["acceptance_checks"]
    if result["accepted"]:
        print(f"ACCEPTED scale {result['scale_mm_per_model_unit']:.4f} mm/unit  "
              f"windows {checks['passing_windows']}/{len(result['windows'])}  "
              f"rings {checks['distinct_rings']}  "
              f"primary spread {checks['primary_relative_spread'] * 100:.2f}%  "
              f"check bar max error {checks['check_bar_relative_error_max'] * 100:.2f}%")
    else:
        print(f"REJECTED  windows {checks['passing_windows']}/{len(result['windows'])}  "
              f"checks {json.dumps(checks)}")

    if args.apply_to:
        if not result["accepted"]:
            print("No metric mesh exported: the scale was not accepted")
        else:
            output = args.output or args.apply_to.with_name(
                args.apply_to.stem + "_mm.ply")
            record = apply_scale(args.apply_to, output,
                                 result["scale_mm_per_model_unit"])
            record["calibration_model_sha256"] = result["model_sha256"]
            result["metric_mesh"] = record
            print(f"Metric mesh written: {output}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
