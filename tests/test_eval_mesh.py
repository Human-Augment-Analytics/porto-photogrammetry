"""Analytic and invariance tests for the geometric metrics."""
import numpy as np
import pytest

from augenblick.eval import mesh

N_SPHERE = 200_000
RADIUS = 1.0
GAP = 0.1


def fibonacci_sphere(n: int, radius: float = 1.0, offset: float = 0.5) -> np.ndarray:
    """Evenly distributed points on a sphere; `offset` shifts the spiral phase."""
    indices = np.arange(n) + offset
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    return radius * np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def ellipsoid(n: int, axes=(1.0, 0.6, 0.3), seed: int = 0) -> np.ndarray:
    """An asymmetric blob, so principal axes are well separated."""
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points * np.asarray(axes)


def rotation_from_axis_angle(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def test_identical_clouds_score_perfectly():
    points = fibonacci_sphere(50_000)
    result = mesh.score(points, points)
    assert result["accuracy_mean_pct"] == pytest.approx(0.0, abs=1e-12)
    assert result["completeness_mean_pct"] == pytest.approx(0.0, abs=1e-12)
    for tolerance in mesh.DEFAULT_TOLERANCES:
        assert result[f"fscore@{tolerance * 100:g}pct"] == pytest.approx(1.0)


def test_concentric_spheres_give_the_known_gap():
    """Every nearest-neighbour distance is the radial gap, in both directions."""
    inner = fibonacci_sphere(N_SPHERE, RADIUS)
    outer = fibonacci_sphere(N_SPHERE, RADIUS + GAP, offset=0.37)
    result = mesh.score(outer, inner)
    assert result["accuracy_mean_abs"] == pytest.approx(GAP, rel=0.02)
    assert result["completeness_mean_abs"] == pytest.approx(GAP, rel=0.02)
    assert result["chamfer_mean_abs"] == pytest.approx(GAP, rel=0.02)


def test_fscore_is_a_step_at_the_known_gap():
    """With all distances equal to the gap, precision and recall are 1 above it and 0 below."""
    inner = fibonacci_sphere(N_SPHERE, RADIUS)
    outer = fibonacci_sphere(N_SPHERE, RADIUS + GAP, offset=0.37)
    result = mesh.score(outer, inner, absolute_tolerances=(GAP * 0.5, GAP * 2.0))
    assert result[f"fscore@abs{GAP * 0.5:g}"] == pytest.approx(0.0)
    assert result[f"fscore@abs{GAP * 2.0:g}"] == pytest.approx(1.0)


def test_relative_metrics_are_invariant_to_a_global_rescale():
    """Multiplying both clouds by a constant must not move any percentage metric."""
    inner = fibonacci_sphere(50_000, RADIUS)
    outer = fibonacci_sphere(50_000, RADIUS + GAP, offset=0.37)
    base = mesh.score(outer, inner)
    scaled = mesh.score(outer * 10.0, inner * 10.0)
    for key in ("accuracy_mean_pct", "completeness_mean_pct", "chamfer_mean_pct",
                "fscore@0.5pct", "fscore@1pct"):
        assert scaled[key] == pytest.approx(base[key], rel=1e-9)
    assert scaled["accuracy_mean_abs"] == pytest.approx(base["accuracy_mean_abs"] * 10.0, rel=1e-9)


def test_absolute_tolerances_track_the_rescale():
    """An absolute tolerance is a distance, so it must follow the coordinates, not the shape."""
    inner = fibonacci_sphere(50_000, RADIUS)
    outer = fibonacci_sphere(50_000, RADIUS + GAP, offset=0.37)
    base = mesh.score(outer, inner, absolute_tolerances=(GAP * 2,))
    scaled = mesh.score(outer * 10.0, inner * 10.0, absolute_tolerances=(GAP * 20,))
    assert scaled[f"fscore@abs{GAP * 20:g}"] == pytest.approx(base[f"fscore@abs{GAP * 2:g}"])


def test_explicit_diag_makes_the_tolerance_mean_one_distance():
    """Passing diag pins the threshold, so candidates of different extent stay comparable."""
    inner = fibonacci_sphere(50_000, RADIUS)
    outer = fibonacci_sphere(50_000, RADIUS + GAP, offset=0.37)
    diag = float(np.linalg.norm(inner.max(axis=0) - inner.min(axis=0)))
    assert mesh.score(outer, inner, diag=diag)["diag"] == pytest.approx(diag)
    assert mesh.score(outer, inner, diag=diag * 2)["accuracy_mean_pct"] == pytest.approx(
        mesh.score(outer, inner, diag=diag)["accuracy_mean_pct"] / 2, rel=1e-9)


def test_transform_apply_matches_its_definition():
    rng = np.random.default_rng(0)
    points = rng.normal(size=(100, 3))
    rotation = rotation_from_axis_angle([0.3, 1.0, -0.7], 0.9)
    transform = mesh.Transform(2.5, rotation, np.array([1.0, -2.0, 3.0]))
    expected = 2.5 * (rotation @ points.T).T + np.array([1.0, -2.0, 3.0])
    assert np.allclose(transform.apply(points), expected)


def test_rigid_registration_recovers_a_known_pose():
    """A rotated and translated copy registers back with scale exactly 1 and no residual."""
    reference = ellipsoid(20_000)
    rotation = rotation_from_axis_angle([0.2, -0.5, 1.0], 0.8)
    moved = (rotation @ reference.T).T + np.array([3.0, -1.0, 2.0])

    transform = mesh.fit_transform(moved, reference, rigid=True)
    assert transform.scale == 1.0
    assert mesh.score(transform.apply(moved), reference)["chamfer_mean_abs"] < 1e-3


def test_rigid_registration_refuses_to_absorb_a_scale_error():
    """A 5% oversized mesh must keep scale at 1.0 and show the error as residual distance."""
    reference = ellipsoid(20_000)
    oversized = reference * 1.05

    rigid = mesh.fit_transform(oversized, reference, rigid=True)
    assert rigid.scale == 1.0
    rigid_error = mesh.score(rigid.apply(oversized), reference)["chamfer_mean_abs"]

    similarity = mesh.fit_transform(oversized, reference, rigid=False)
    assert similarity.scale == pytest.approx(1 / 1.05, rel=0.02)
    similarity_error = mesh.score(similarity.apply(oversized), reference)["chamfer_mean_abs"]

    assert rigid_error > 10 * similarity_error


def test_similarity_registration_recovers_a_known_scale():
    reference = ellipsoid(20_000)
    moved = reference * 2.0 + np.array([1.0, 1.0, 1.0])
    transform = mesh.fit_transform(moved, reference, rigid=False)
    assert transform.scale == pytest.approx(0.5, rel=0.02)


def test_sampling_is_uniform_over_area_not_over_vertices():
    """Two triangles differing 100x in area must receive samples in that ratio."""
    o3d = pytest.importorskip("open3d")
    vertices = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],      # area 0.5
        [0.0, 0.0, 5.0], [0.1, 0.0, 5.0], [0.0, 0.1, 5.0],      # area 0.005
    ])
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))

    points = mesh.sample_surface(m, 20_000, seed=0)
    on_big = (points[:, 2] < 2.5).mean()
    assert on_big == pytest.approx(0.5 / 0.505, abs=0.02)


def test_sampling_is_deterministic_for_a_seed():
    o3d = pytest.importorskip("open3d")
    m = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    assert np.array_equal(mesh.sample_surface(m, 5_000, seed=7),
                          mesh.sample_surface(m, 5_000, seed=7))


def test_accuracy_and_completeness_separate_an_outer_bound():
    """An outer bound is complete but inaccurate, and the two numbers must show that."""
    reference = fibonacci_sphere(50_000, RADIUS)
    # Our mesh: the reference surface plus a shell of surplus material further out.
    surplus = fibonacci_sphere(50_000, RADIUS + 0.5, offset=0.21)
    ours = np.vstack([reference, surplus])

    result = mesh.score(ours, reference)
    assert result["completeness_mean_abs"] == pytest.approx(0.0, abs=1e-6)
    assert result["accuracy_mean_abs"] == pytest.approx(0.25, rel=0.05)


def test_precision_and_recall_are_not_interchanged():
    """A partial reconstruction is precise but incomplete; F alone could not tell."""
    reference = fibonacci_sphere(50_000, RADIUS)
    cap = reference[reference[:, 2] > 0.8]
    assert 0 < len(cap) < len(reference) / 5

    result = mesh.score(cap, reference, absolute_tolerances=(0.05,))
    assert result["precision@abs0.05"] == pytest.approx(1.0)
    assert result["recall@abs0.05"] < 0.2
    assert result["accuracy_mean_abs"] < result["completeness_mean_abs"] / 10


def test_the_threshold_is_strictly_less_than():
    """A distance exactly at the tolerance does not count, and the boundary is pinned."""
    distances = np.array([0.5, 1.0, 1.5])
    entries = mesh._fscore_entries(distances, distances, 1.0, "t")
    assert entries["precision@t"] == pytest.approx(1 / 3)
    assert entries["recall@t"] == pytest.approx(1 / 3)


def test_a_zero_denominator_gives_zero_not_a_nan():
    """When nothing falls inside the tolerance, F is 0; a nan would poison every mean."""
    far = np.array([10.0, 10.0])
    entries = mesh._fscore_entries(far, far, 1.0, "t")
    assert entries["fscore@t"] == 0.0


def test_a_mesh_without_triangles_is_rejected():
    """A point cloud read as a mesh would silently sample nothing."""
    o3d = pytest.importorskip("open3d")
    empty = o3d.geometry.TriangleMesh()
    empty.vertices = o3d.utility.Vector3dVector(np.zeros((3, 3)))
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/empty.ply"
        o3d.io.write_triangle_mesh(path, empty)
        with pytest.raises(ValueError, match="no triangles"):
            mesh.load_mesh(path)


def test_scores_survive_a_change_of_tessellation():
    """The same surface described by different triangle counts must score the same."""
    o3d = pytest.importorskip("open3d")
    coarse = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    fine = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=60)
    assert len(fine.triangles) > 5 * len(coarse.triangles)

    reference = mesh.sample_surface(fine, 200_000, seed=0)
    from_coarse = mesh.score(mesh.sample_surface(coarse, 200_000, seed=1), reference)
    from_fine = mesh.score(mesh.sample_surface(fine, 200_000, seed=2), reference)
    assert from_coarse["fscore@1pct"] == pytest.approx(from_fine["fscore@1pct"], abs=0.02)


def test_distance_to_mesh_is_exact():
    """Distance to a sphere of radius r from radius R is exactly R - r, to float precision."""
    o3d = pytest.importorskip("open3d")
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=200)
    probes = fibonacci_sphere(2_000, 1.5)
    distances = mesh.distance_to_mesh(probes, sphere)
    assert distances.mean() == pytest.approx(0.5, rel=2e-4)
    assert distances.std() < 1e-4


def test_point_to_triangle_removes_the_target_discretisation_error():
    """The point-to-point form overestimates by about the target's sample spacing."""
    o3d = pytest.importorskip("open3d")
    inner = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=120)
    outer = o3d.geometry.TriangleMesh.create_sphere(radius=1.1, resolution=120)

    exact = mesh.score_surface(outer, inner, n=20_000, seed=0)
    sparse_points = mesh.sample_surface(inner, 20_000, seed=3)
    approximate = mesh.score(mesh.sample_surface(outer, 20_000, seed=4), sparse_points)

    assert exact["accuracy_mean_abs"] == pytest.approx(GAP, rel=0.01)
    assert approximate["accuracy_mean_abs"] > exact["accuracy_mean_abs"]
    assert exact["distance"] == "point_to_triangle"
    assert approximate["distance"] == "point_to_point"


def test_score_surface_applies_the_transform_to_both_representations():
    """The moved mesh and the moved samples must agree, or the two directions disagree."""
    o3d = pytest.importorskip("open3d")
    reference = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)
    shifted = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=40)
    shifted.translate((5.0, 0.0, 0.0))

    identity = mesh.Transform(1.0, np.eye(3), np.array([-5.0, 0.0, 0.0]))
    result = mesh.score_surface(shifted, reference, transform=identity, n=20_000)
    assert result["accuracy_mean_abs"] < 1e-3
    assert result["completeness_mean_abs"] < 1e-3
