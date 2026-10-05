"""Geometric evaluation of a reconstructed mesh against a reference."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

DEFAULT_TOLERANCES = (0.001, 0.0025, 0.005, 0.01)
DEFAULT_SAMPLES = 200_000
ICP_ITERATIONS = 40
ICP_TRIM_PERCENTILE = 80.0
ICP_SAMPLES = 40_000
DECRUFT_PERCENTILES = (100.0, 99.5, 99.0, 98.0, 95.0)


@dataclass(frozen=True)
class Transform:
    """A similarity transform mapping source points into the reference frame."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map points through the transform."""
        return (self.scale * (self.rotation @ points.T).T) + self.translation


def load_mesh(path: Path):
    """Read a triangle mesh in any format Open3D supports."""
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    if len(mesh.triangles) == 0:
        raise ValueError(f"no triangles in {path}")
    return mesh


def sample_surface(mesh, n: int = DEFAULT_SAMPLES, seed: int = 0) -> np.ndarray:
    """Sample n points uniformly over surface area."""
    import open3d as o3d

    o3d.utility.random.seed(seed)
    pcd = mesh.sample_points_uniformly(number_of_points=n, use_triangle_normal=False)
    return np.asarray(pcd.points, dtype=np.float64)


def _subsample(points: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(points) <= n:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), n, replace=False)]


def _decruft(points: np.ndarray, keep_percentile: float) -> np.ndarray:
    """Drop distant floaters, which would otherwise skew the PCA axes and the radius."""
    if keep_percentile >= 100.0:
        return points
    median = np.median(points, axis=0)
    radius = np.linalg.norm(points - median, axis=1)
    return points[radius <= np.percentile(radius, keep_percentile)]


def _normalise(points: np.ndarray, with_scale: bool):
    """Centre, and optionally divide by mean radius."""
    centre = points.mean(axis=0)
    centred = points - centre
    radius = float(np.linalg.norm(centred, axis=1).mean()) if with_scale else 1.0
    return centred / radius, centre, radius


def _pca_axes(points: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(points - points.mean(axis=0), full_matrices=False)
    return vt


def _umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool):
    """Least-squares similarity (or rigid) transform mapping src onto dst."""
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src0, dst0 = src - mu_src, dst - mu_dst
    covariance = dst0.T @ src0 / len(src)
    u, d, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1
    rotation = u @ correction @ vt
    if with_scale:
        variance = float((src0 ** 2).sum() / len(src))
        scale = float(np.trace(np.diag(d) @ correction) / variance) if variance > 0 else 1.0
    else:
        scale = 1.0
    return scale, rotation, mu_dst - scale * rotation @ mu_src


def _icp(src: np.ndarray, dst_tree: cKDTree, dst: np.ndarray, with_scale: bool):
    """Trimmed iterative closest point, rejecting the worst correspondences each round."""
    current = src.copy()
    total_scale, total_rotation, total_translation = 1.0, np.eye(3), np.zeros(3)
    previous = np.inf
    for _ in range(ICP_ITERATIONS):
        distances, indices = dst_tree.query(current, workers=-1)
        keep = distances <= np.percentile(distances, ICP_TRIM_PERCENTILE)
        scale, rotation, translation = _umeyama(current[keep], dst[indices[keep]], with_scale)
        current = (scale * (rotation @ current.T).T) + translation
        total_scale *= scale
        total_rotation = rotation @ total_rotation
        total_translation = scale * (rotation @ total_translation) + translation
        rms = float(np.sqrt((distances[keep] ** 2).mean()))
        if abs(previous - rms) < 1e-7:
            break
        previous = rms
    return total_scale, total_rotation, total_translation


def _symmetric_chamfer(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * (cKDTree(b).query(a, workers=-1)[0].mean()
                  + cKDTree(a).query(b, workers=-1)[0].mean())


def fit_transform(src: np.ndarray, dst: np.ndarray, *, rigid: bool = False,
                  seed: int = 0) -> Transform:
    """Register src onto dst by multi-start PCA initialisation followed by trimmed ICP."""
    with_scale = not rigid
    best_score, best = np.inf, None
    dst_reference = _subsample(_decruft(dst, 100.0), ICP_SAMPLES, seed)
    src_scoring = _subsample(src, ICP_SAMPLES, seed)

    for percentile in DECRUFT_PERCENTILES:
        dst_trimmed = _decruft(dst, percentile)
        src_trimmed = _decruft(src, percentile)
        dst_norm, dst_centre, dst_radius = _normalise(dst_trimmed, with_scale)
        src_norm, src_centre, src_radius = _normalise(src_trimmed, with_scale)
        dst_icp = _subsample(dst_norm, ICP_SAMPLES, seed)
        src_icp = _subsample(src_norm, ICP_SAMPLES, seed)
        tree = cKDTree(dst_icp)
        dst_axes, src_axes = _pca_axes(dst_icp), _pca_axes(src_icp)

        for sign_x in (1, -1):
            for sign_y in (1, -1):
                reflection = np.diag([sign_x, sign_y, sign_x * sign_y])
                initial = dst_axes.T @ reflection @ src_axes
                scale, rotation, translation = _icp(
                    (initial @ src_icp.T).T, tree, dst_icp, with_scale)
                combined = rotation @ initial
                # Fold the two normalisations into a single transform on raw coordinates.
                total_scale = dst_radius * scale / src_radius
                total_rotation = combined
                total_translation = (dst_radius * translation + dst_centre
                                     - total_scale * combined @ src_centre)
                candidate = Transform(float(total_scale), total_rotation, total_translation)
                score_value = _symmetric_chamfer(dst_reference, candidate.apply(src_scoring))
                if score_value < best_score:
                    best_score, best = score_value, candidate

    if best is None:  # pragma: no cover - only reachable with empty input
        raise ValueError("registration produced no candidate")
    logger.debug(f"registered with scale {best.scale:.6f}, chamfer {best_score:.6g}")
    return best


def score(ours: np.ndarray, reference: np.ndarray, *, diag: float | None = None,
          tolerances=DEFAULT_TOLERANCES, absolute_tolerances=None) -> dict:
    """Accuracy, completeness and F-score of a point set against a reference."""
    if diag is None:
        diag = float(np.linalg.norm(reference.max(axis=0) - reference.min(axis=0)))
    distance_ours = cKDTree(reference).query(ours, workers=-1)[0]
    distance_reference = cKDTree(ours).query(reference, workers=-1)[0]
    return _summarise(distance_ours, distance_reference, diag, tolerances,
                      absolute_tolerances, "point_to_point")


def _fscore_entries(distance_ours, distance_reference, threshold, key) -> dict:
    """Precision, recall and F at one threshold, in the Tanks and Temples definition."""
    precision = float((distance_ours < threshold).mean())
    recall = float((distance_reference < threshold).mean())
    fscore = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {f"precision@{key}": precision, f"recall@{key}": recall, f"fscore@{key}": fscore}


def distance_to_mesh(points: np.ndarray, mesh) -> np.ndarray:
    """Exact unsigned distance from each point to the nearest triangle of a mesh."""
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query = o3d.core.Tensor(np.asarray(points, dtype=np.float32))
    return scene.compute_distance(query).numpy().astype(np.float64)


def score_surface(ours_mesh, reference_mesh, *, transform: Transform | None = None,
                  n: int = DEFAULT_SAMPLES, seed: int = 0, diag: float | None = None,
                  tolerances=DEFAULT_TOLERANCES, absolute_tolerances=None) -> dict:
    """Score two meshes using exact point-to-triangle distance in both directions."""
    import copy

    ours_sample = sample_surface(ours_mesh, n, seed=seed)
    reference_sample = sample_surface(reference_mesh, n, seed=seed)

    moved_mesh = ours_mesh
    if transform is not None:
        ours_sample = transform.apply(ours_sample)
        moved_mesh = copy.deepcopy(ours_mesh)
        matrix = np.eye(4)
        matrix[:3, :3] = transform.scale * transform.rotation
        matrix[:3, 3] = transform.translation
        moved_mesh.transform(matrix)

    if diag is None:
        diag = float(np.linalg.norm(reference_sample.max(axis=0) - reference_sample.min(axis=0)))

    distance_ours = distance_to_mesh(ours_sample, reference_mesh)
    distance_reference = distance_to_mesh(reference_sample, moved_mesh)
    return _summarise(distance_ours, distance_reference, diag, tolerances,
                      absolute_tolerances, "point_to_triangle")


def _summarise(distance_ours, distance_reference, diag, tolerances, absolute_tolerances,
               distance_kind) -> dict:
    """Assemble the reported quantities from the two distance arrays."""
    out = {
        "diag": diag,
        "distance": distance_kind,
        "accuracy_mean_pct": float(distance_ours.mean() / diag * 100),
        "accuracy_p95_pct": float(np.percentile(distance_ours, 95) / diag * 100),
        "completeness_mean_pct": float(distance_reference.mean() / diag * 100),
        "completeness_p95_pct": float(np.percentile(distance_reference, 95) / diag * 100),
        "chamfer_mean_pct": float(0.5 * (distance_ours.mean() + distance_reference.mean())
                                  / diag * 100),
        "accuracy_mean_abs": float(distance_ours.mean()),
        "completeness_mean_abs": float(distance_reference.mean()),
        "chamfer_mean_abs": float(0.5 * (distance_ours.mean() + distance_reference.mean())),
    }
    for tolerance in tolerances:
        out.update(_fscore_entries(distance_ours, distance_reference,
                                   tolerance * diag, f"{tolerance * 100:g}pct"))
    for tolerance in absolute_tolerances or ():
        out.update(_fscore_entries(distance_ours, distance_reference,
                                   tolerance, f"abs{tolerance:g}"))
    return out

def main(argv: list[str] | None = None) -> int:
    """Score one mesh against another from the command line."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="python -m augenblick.eval.mesh",
        description="Score a reconstructed mesh against a reference mesh.")
    parser.add_argument("--ours", required=True, type=Path,
                        help="Reconstructed mesh")
    parser.add_argument("--reference", required=True, type=Path,
                        help="Reference mesh; its units and frame define the metric")
    parser.add_argument("--scale-ours", type=float, default=1.0,
                        help="Multiply the reconstruction by this before scoring, to convert "
                             "units (e.g. 1000 for metres against a millimetre reference)")
    parser.add_argument("--rigid", action="store_true",
                        help="Fit rotation and translation only, leaving scale at 1.0. Use "
                             "when the reconstruction already carries metric scale; a "
                             "similarity fit would otherwise absorb the scale error")
    parser.add_argument("--no-align", action="store_true",
                        help="Skip registration, for meshes already in a common frame")
    parser.add_argument("-n", "--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0],
                        help="Sampling seeds; results are averaged and the spread reported")
    parser.add_argument("--absolute-tolerances", type=float, nargs="*", default=None,
                        help="Tolerances in the reference's own unit, reported alongside the "
                             "diagonal-relative ones")
    parser.add_argument("--point-to-point", action="store_true",
                        help="Use the Tanks and Temples point-to-point distance instead of "
                             "exact point-to-triangle. Biased low; kept for comparability")
    parser.add_argument("--label", default="")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)

    ours_mesh = load_mesh(args.ours)
    reference_mesh = load_mesh(args.reference)
    if args.scale_ours != 1.0:
        ours_mesh.scale(args.scale_ours, center=(0.0, 0.0, 0.0))

    reference_vertices = np.asarray(reference_mesh.vertices)
    diag = float(np.linalg.norm(reference_vertices.max(axis=0) - reference_vertices.min(axis=0)))

    transform = None
    if not args.no_align:
        transform = fit_transform(sample_surface(ours_mesh, ICP_SAMPLES, seed=0),
                                  sample_surface(reference_mesh, ICP_SAMPLES, seed=0),
                                  rigid=args.rigid)

    runs = []
    for seed in args.seeds:
        if args.point_to_point:
            points = sample_surface(ours_mesh, args.samples, seed=seed)
            if transform is not None:
                points = transform.apply(points)
            runs.append(score(points, sample_surface(reference_mesh, args.samples, seed=seed),
                              diag=diag, absolute_tolerances=args.absolute_tolerances))
        else:
            runs.append(score_surface(ours_mesh, reference_mesh, transform=transform,
                                      n=args.samples, seed=seed, diag=diag,
                                      absolute_tolerances=args.absolute_tolerances))

    numeric = [k for k in runs[0] if isinstance(runs[0][k], float)]
    result = {k: float(np.mean([r[k] for r in runs])) for k in numeric}
    result["distance"] = runs[0]["distance"]
    result["fscore_seed_spread@0.5pct"] = float(
        max(r["fscore@0.5pct"] for r in runs) - min(r["fscore@0.5pct"] for r in runs))
    result.update(label=args.label, ours=str(args.ours), reference=str(args.reference),
                  n_samples=args.samples, seeds=list(args.seeds),
                  registration=("none" if args.no_align else
                                ("rigid" if args.rigid else "similarity")),
                  scale=(transform.scale if transform is not None else 1.0))

    print(f"{args.label or args.ours.name}: "
          f"acc {result['accuracy_mean_pct']:.3f}%  "
          f"comp {result['completeness_mean_pct']:.3f}%  "
          f"F@0.5% {result['fscore@0.5pct']:.4f}  "
          f"({result['distance']}, {result['registration']}, scale {result['scale']:.6g})")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


