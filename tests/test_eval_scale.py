"""Tests for independent scale recovery from printed bars, on a synthetic turntable rig."""
import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")

from augenblick.eval import scale


RIG_RADIUS = 5.0
UNITS_PER_MM = 0.01  # ground-truth gauge: 1 model unit = 100 mm
BAR_MM = 100.0


def make_camera(width=1000, height=800, focal=1000.0):
    return pycolmap.Camera(model="SIMPLE_PINHOLE", width=width, height=height,
                           params=[focal, width / 2, height / 2])


def ring_pose(angle: float) -> np.ndarray:
    """World-to-camera matrix of a camera on a horizontal circle, looking at the origin."""
    centre = RIG_RADIUS * np.array([np.cos(angle), np.sin(angle), 0.3])
    forward = -centre / np.linalg.norm(centre)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    return np.c_[rotation, -rotation @ centre]


def marker_points():
    """Two bars of nominal length 100 mm in a gauge of 0.01 units per mm."""
    length = BAR_MM * UNITS_PER_MM
    return {
        "A1": np.array([-0.7, -0.4, 0.0]),
        "A2": np.array([-0.7, -0.4, 0.0]) + [length, 0.0, 0.0],
        "B1": np.array([0.5, 0.6, 0.05]),
        "B2": np.array([0.5, 0.6, 0.05]) + [0.0, -length, 0.0],
    }


def synthetic_rig(n_rings=3, views_per_ring=72, noise=0.0, seed=0,
                  drop=None):
    """Observations of the four markers from `n_rings` rings of cameras."""
    camera = make_camera()
    rng = np.random.default_rng(seed)
    points = marker_points()
    tracks = {code: {} for code in points}
    rings = []
    for ring in range(n_rings):
        names = []
        for view in range(views_per_ring):
            angle = 2 * np.pi * view / views_per_ring + ring * 0.15
            pose = ring_pose(angle)
            name = f"r{ring}_v{view:03d}"
            names.append(name)
            for code, point in points.items():
                if drop is not None and drop(ring, view, code):
                    continue
                xy = camera.img_from_cam(pose[:, :3] @ point + pose[:, 3])
                if xy is None:
                    continue
                xy = np.asarray(xy) + rng.normal(0, noise, 2)
                tracks[code][name] = scale.Observation(camera, pose, xy, name)
        rings.append(names)
    return tracks, rings


BARS = scale.BarTargets(primary=("A1", "A2"), check=("B1", "B2"))


class TestColmapPixel:
    def test_adds_half_pixel(self):
        np.testing.assert_array_equal(scale.colmap_pixel([0, 0]), [0.5, 0.5])
        np.testing.assert_array_equal(scale.colmap_pixel([10.25, 3.0]), [10.75, 3.5])


class TestTriangulateTrack:
    def observations(self, truth, n=12, camera=None, noise=0.0, seed=0):
        camera = camera or make_camera()
        rng = np.random.default_rng(seed)
        out = []
        for i, x in enumerate(np.linspace(-1, 1, n)):
            pose = np.c_[np.eye(3), [-x, 0.0, 0.0]]
            xy = np.asarray(camera.img_from_cam(pose[:, :3] @ truth + pose[:, 3]))
            out.append(scale.Observation(camera, pose, xy + rng.normal(0, noise, 2),
                                         str(i)))
        return out

    def test_exact_recovery(self):
        truth = np.array([0.12, -0.2, 4.0])
        result = scale.triangulate_track(self.observations(truth))
        np.testing.assert_allclose(result["xyz"], truth, atol=1e-8)

    def test_gross_outlier_is_rejected(self):
        truth = np.array([0.12, -0.2, 4.0])
        observations = self.observations(truth)
        bad = observations[0]
        observations[0] = scale.Observation(bad.camera, bad.pose, bad.xy + [50, 30],
                                            bad.image)
        result = scale.triangulate_track(observations)
        np.testing.assert_allclose(result["xyz"], truth, atol=1e-8)
        assert result["inliers"] == len(observations) - 1
        assert observations[0].image in result["rejected_images"]

    def test_noise_tolerance(self):
        truth = np.array([0.12, -0.2, 4.0])
        result = scale.triangulate_track(self.observations(truth, noise=0.2))
        np.testing.assert_allclose(result["xyz"], truth, atol=0.01)

    def test_distorted_rotated_cameras(self):
        camera = pycolmap.Camera(model="SIMPLE_RADIAL", width=1000, height=800,
                                 params=[1000.0, 500.0, 400.0, 0.05])
        truth = np.array([0.12, -0.2, 4.0])
        observations = []
        for i, x in enumerate(np.linspace(-1, 1, 12)):
            a = -x * 0.1
            rotation = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0],
                                 [-np.sin(a), 0, np.cos(a)]])
            pose = np.c_[rotation, -rotation @ np.array([x, 0.0, 0.0])]
            xy = np.asarray(camera.img_from_cam(pose[:, :3] @ truth + pose[:, 3]))
            observations.append(scale.Observation(camera, pose, xy, str(i)))
        result = scale.triangulate_track(observations)
        np.testing.assert_allclose(result["xyz"], truth, atol=1e-8)

    def test_too_few_views_raise(self):
        truth = np.array([0.12, -0.2, 4.0])
        with pytest.raises(ValueError, match="need"):
            scale.triangulate_track(self.observations(truth)[:2])

    def test_near_parallel_rays_raise(self):
        truth = np.array([0.12, -0.2, 4.0])
        camera = make_camera()
        observations = []
        for i, x in enumerate(np.linspace(0, 1e-4, 12)):
            pose = np.c_[np.eye(3), [-x, 0.0, 0.0]]
            xy = np.asarray(camera.img_from_cam(pose[:, :3] @ truth + pose[:, 3]))
            observations.append(scale.Observation(camera, pose, xy, str(i)))
        with pytest.raises(ValueError, match="angle"):
            scale.triangulate_track(observations)


class TestSlidingWindowScale:
    def test_recovers_known_scale(self):
        tracks, rings = synthetic_rig(noise=0.1)
        result = scale.sliding_window_scale(tracks, rings, BARS)
        assert result["accepted"]
        expected = 1.0 / UNITS_PER_MM
        assert result["scale_mm_per_model_unit"] == pytest.approx(expected, rel=1e-3)
        checks = result["acceptance_checks"]
        assert checks["passing_windows"] == 27
        assert checks["distinct_rings"] == 3
        assert checks["check_bar_relative_error_max"] < 0.005

    def test_all_windows_recorded(self):
        tracks, rings = synthetic_rig(noise=0.1)
        result = scale.sliding_window_scale(tracks, rings, BARS)
        assert len(result["windows"]) == 27
        assert all("markers" in w for w in result["windows"])

    def test_single_ring_support_is_rejected(self):
        """Consistent windows confined to one ring must not become an accepted scale."""
        tracks, rings = synthetic_rig(
            noise=0.1, drop=lambda ring, view, code: ring != 1)
        result = scale.sliding_window_scale(tracks, rings, BARS)
        assert not result["accepted"]
        assert result["acceptance_checks"]["distinct_rings"] <= 1
        # The rejected result still carries the diagnostic lengths for inspection.
        assert "scale_mm_per_model_unit" in result

    def test_sparse_detections_are_rejected(self):
        """Half-density coverage starves windows below the observation minimum."""
        tracks, rings = synthetic_rig(
            noise=0.1, drop=lambda ring, view, code: view % 2 == 0)
        result = scale.sliding_window_scale(tracks, rings, BARS)
        assert not result["accepted"]

    def test_moved_bar_fails_held_out(self):
        """A bar that shifts mid-ring must fail windows spanning the shift."""
        moved = marker_points()
        camera = make_camera()
        tracks = {code: {} for code in moved}
        rings = []
        names = []
        for view in range(72):
            angle = 2 * np.pi * view / 72
            pose = ring_pose(angle)
            name = f"r0_v{view:03d}"
            names.append(name)
            offset = np.array([0.06, 0.0, 0.0]) if view >= 36 else np.zeros(3)
            for code, point in moved.items():
                shifted = point + (offset if code.startswith("A") else 0.0)
                xy = np.asarray(camera.img_from_cam(pose[:, :3] @ shifted + pose[:, 3]))
                tracks[code][name] = scale.Observation(camera, pose, xy, name)
        rings.append(names)
        result = scale.sliding_window_scale(tracks, rings, BARS)
        failing = [w for w in result["windows"] if not w["passed"]]
        assert failing, "windows spanning the shift must fail"
        assert not result["accepted"]

    def test_missing_code_raises(self):
        tracks, rings = synthetic_rig()
        del tracks["B2"]
        with pytest.raises(ValueError, match="B2"):
            scale.sliding_window_scale(tracks, rings, BARS)

    def test_gates_reject_insufficient_holdout(self):
        with pytest.raises(ValueError, match="inlier minimum"):
            scale.Gates(min_observations=12, withheld=8)


class TestApplyScale:
    def test_scale_topology_and_colours(self, tmp_path):
        o3d = pytest.importorskip("open3d")
        source = tmp_path / "source.ply"
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        sphere.paint_uniform_color([0.2, 0.4, 0.6])
        assert o3d.io.write_triangle_mesh(str(source), sphere)
        record = scale.apply_scale(source, tmp_path / "metric.ply", 100.0)
        np.testing.assert_allclose(record["extent_mm"], [200, 200, 200], atol=1e-5)
        assert record["units"] == "mm"

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_invalid_scale_rejected(self, tmp_path, bad):
        o3d = pytest.importorskip("open3d")
        source = tmp_path / "source.ply"
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        assert o3d.io.write_triangle_mesh(str(source), sphere)
        with pytest.raises(ValueError):
            scale.apply_scale(source, tmp_path / "bad.ply", bad)

    def test_never_overwrites(self, tmp_path):
        o3d = pytest.importorskip("open3d")
        source = tmp_path / "source.ply"
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
        assert o3d.io.write_triangle_mesh(str(source), sphere)
        destination = tmp_path / "metric.ply"
        scale.apply_scale(source, destination, 2.0)
        with pytest.raises(FileExistsError):
            scale.apply_scale(source, destination, 2.0)
