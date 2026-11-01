"""COLMAP -> SfMData conversion for benchmark_meshroom.py --from_sfm, without AliceVision.

The reprojection test decodes the JSON with AliceVision's own formulas, written apart
from the encoder, so a transposed rotation or a swapped pixelRatio fails here even
though a self-round-trip would pass.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

pycolmap = pytest.importorskip("pycolmap")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
import benchmark_meshroom  # noqa: E402
from colmap_to_sfmdata import ConversionError, convert  # noqa: E402

W, H = 640, 480
# Off-centre principal points and fx != fy, so axis or centre mix-ups show.
INTRINSICS = [(500.0, 520.0, 330.0, 236.0), (610.0, 590.0, 315.5, 250.25)]
QUATS_XYZW = [(0.0, 0.0, 0.0, 1.0), (0.1, -0.2, 0.05, 0.97), (-0.15, 0.1, 0.2, 0.96), (0.3, 0.05, -0.1, 0.94)]
TRANSLATIONS = [(0.1, -0.2, 5.0), (-0.3, 0.1, 5.5), (0.2, 0.3, 4.5), (0.0, -0.1, 6.0)]
# Image 1 sees point 0 a second time, 3 px off; the exact one must win.
DUPLICATE = (1, 0, np.array([3.0, -2.0]))


def _points(n=12):
    rng = np.random.default_rng(0)
    return rng.uniform(-1.0, 1.0, size=(n, 3))


def build_scene(root, *, model="PINHOLE", names=None, with_points=True, duplicate=True,
                lonely_point=False, masks=False):
    """Write a 4-view COLMAP scene: images/ (empty files), sparse/0, optional masks/."""
    names = names or [f"img_{i}.jpg" for i in range(1, 5)]
    recon = pycolmap.Reconstruction()
    poses = {}
    for i in range(1, 5):
        fx, fy, cx, cy = INTRINSICS[i % 2]
        if model == "PINHOLE":
            params = [fx, fy, cx, cy]
        elif model == "SIMPLE_RADIAL":
            params = [fx, cx, cy, 0.01]
        camera = pycolmap.Camera(model=model, width=W, height=H, params=params, camera_id=i)
        recon.add_camera_with_trivial_rig(camera)
        q = np.array(QUATS_XYZW[i - 1])
        poses[i] = pycolmap.Rigid3d(pycolmap.Rotation3d(q / np.linalg.norm(q)), np.array(TRANSLATIONS[i - 1]))

    X = _points() if with_points else np.zeros((0, 3))
    keypoints = {i: [] for i in poses}
    tracks = [[] for _ in X]
    for i, cam_from_world in poses.items():
        camera = recon.cameras[i]
        for p, x in enumerate(X):
            tracks[p].append((i, len(keypoints[i])))
            keypoints[i].append(camera.img_from_cam(cam_from_world * x))
    if with_points and duplicate:
        i, p, offset = DUPLICATE
        tracks[p].append((i, len(keypoints[i])))
        keypoints[i].append(keypoints[i][tracks[p][i - 1][1]] + offset)
    if with_points and lonely_point:
        # Two observations, both in image 1: one view after dedupe, so it is dropped.
        x = np.array([0.0, 0.0, 0.5])
        kp = recon.cameras[1].img_from_cam(poses[1] * x)
        tracks.append([(1, len(keypoints[1])), (1, len(keypoints[1]) + 1)])
        keypoints[1] += [kp, kp + 1.0]
        X = np.vstack([X, x])

    for i, cam_from_world in poses.items():
        image = pycolmap.Image(name=names[i - 1], keypoints=np.array(keypoints[i]).reshape(-1, 2),
                               camera_id=i, image_id=i)
        recon.add_image_with_trivial_frame(image, cam_from_world)
    for x, track in zip(X, tracks):
        recon.add_point3D(x, pycolmap.Track([pycolmap.TrackElement(i, k) for i, k in track]))

    (root / "sparse" / "0").mkdir(parents=True)
    recon.write(str(root / "sparse" / "0"))
    (root / "images").mkdir()
    for name in names:
        (root / "images" / name).touch()
    if masks:
        (root / "masks").mkdir()
        for name in names:
            (root / "masks" / f"{Path(name).stem}.png").touch()
    return recon


def project(sfm, view_id, X):
    """AliceVision's pinhole projection, from the SfMData fields alone."""
    view = next(v for v in sfm["views"] if v["viewId"] == view_id)
    intr = next(k for k in sfm["intrinsics"] if k["intrinsicId"] == view["intrinsicId"])
    pose = next(p for p in sfm["poses"] if p["poseId"] == view["poseId"])["pose"]["transform"]
    R = np.array(pose["rotation"], dtype=float).reshape(3, 3, order="F")
    t = -R @ np.array(pose["center"], dtype=float)
    w, h = intr["width"], intr["height"]
    fy = intr["focalLength"] * w / intr["sensorWidth"]
    fx = intr["focalLength"] / intr["pixelRatio"] * w / intr["sensorWidth"]
    cx, cy = np.array(intr["principalPoint"], dtype=float) + (w / 2, h / 2)
    Xc = R @ np.asarray(X, dtype=float) + t
    return np.array([fx * Xc[0] / Xc[2] + cx, fy * Xc[1] / Xc[2] + cy])


def run_convert(tmp_path, **kw):
    scene = tmp_path / "scene"
    recon = build_scene(scene, **kw)
    out = tmp_path / "out.sfm"
    stats = convert(scene, out)
    with open(out) as f:
        return recon, stats, json.load(f)


def test_observations_reproject_to_colmap_keypoints(tmp_path):
    recon, _, sfm = run_convert(tmp_path)
    checked = 0
    for lm in sfm["structure"]:
        for obs in lm["observations"]:
            keypoint = recon.images[obs["observationId"]].points2D[obs["featureId"]].xy
            np.testing.assert_allclose(obs["x"], keypoint, atol=1e-9)
            np.testing.assert_allclose(project(sfm, obs["observationId"], lm["X"]), keypoint, atol=1e-9)
            checked += 1
    assert checked == 12 * 4


def test_header_and_views(tmp_path):
    _, stats, sfm = run_convert(tmp_path)
    assert sfm["version"] == [1, 2, 12]
    assert stats.views == len(sfm["views"]) == len(sfm["poses"]) == 4
    assert stats.intrinsics == len(sfm["intrinsics"]) == 4
    for view in sfm["views"]:
        assert view["viewId"] == view["poseId"]
        assert os.path.isabs(view["path"]) and Path(view["path"]).exists()
        assert view["metadata"] == {}


def test_duplicate_keeps_lowest_error_observation(tmp_path):
    recon, stats, sfm = run_convert(tmp_path)
    i, p, _ = DUPLICATE
    assert stats.observations_deduped == 1
    lm = next(lm for lm in sfm["structure"] if np.allclose(lm["X"], _points()[p]))
    in_view = [o for o in lm["observations"] if o["observationId"] == i]
    assert len(in_view) == 1
    np.testing.assert_allclose(project(sfm, i, lm["X"]), in_view[0]["x"], atol=1e-9)


def test_landmark_left_with_one_view_is_dropped(tmp_path):
    _, stats, sfm = run_convert(tmp_path, lonely_point=True)
    assert stats.landmarks_dropped == 1
    assert stats.landmarks == len(sfm["structure"]) == 12
    assert all(len({o["observationId"] for o in lm["observations"]}) >= 2 for lm in sfm["structure"])


def test_rejects_distorted_camera(tmp_path):
    build_scene(tmp_path / "scene", model="SIMPLE_RADIAL")
    with pytest.raises(ConversionError, match="SIMPLE_RADIAL"):
        convert(tmp_path / "scene", tmp_path / "out.sfm")
    assert not (tmp_path / "out.sfm").exists()


def test_rejects_trackless_model(tmp_path):
    build_scene(tmp_path / "scene", with_points=False)
    with pytest.raises(ConversionError, match="DVLT"):
        convert(tmp_path / "scene", tmp_path / "out.sfm")


def test_rejects_missing_image(tmp_path):
    build_scene(tmp_path / "scene")
    (tmp_path / "scene" / "images" / "img_3.jpg").unlink()
    with pytest.raises(ConversionError, match="img_3.jpg"):
        convert(tmp_path / "scene", tmp_path / "out.sfm")


def test_rejects_mask_named_after_another_views_id(tmp_path):
    # View 1 is 4.jpg, so tryLoadMask would give it masks/1.png, which is 1.jpg's mask.
    build_scene(tmp_path / "scene", names=["4.jpg", "3.jpg", "2.jpg", "1.jpg"], masks=True)
    with pytest.raises(ConversionError, match="another image's mask"):
        convert(tmp_path / "scene", tmp_path / "out.sfm")


def test_masks_named_by_matching_id_pass(tmp_path):
    build_scene(tmp_path / "scene", names=["1.jpg", "2.jpg", "3.jpg", "4.jpg"], masks=True)
    assert convert(tmp_path / "scene", tmp_path / "out.sfm").views == 4


def test_dense_graph_has_no_sfm_nodes(tmp_path):
    folders = os.environ.get("MESHROOM_PIPELINE_TEMPLATES_PATH", "").split(os.pathsep)
    if not any(f and (Path(f) / "photogrammetry.mg").is_file() for f in folders):
        pytest.skip("photogrammetry.mg not on MESHROOM_PIPELINE_TEMPLATES_PATH")
    sfm_path = tmp_path / "colmap.sfm"
    with open(benchmark_meshroom.write_dense_graph(tmp_path / "dense.mg", sfm_path)) as f:
        template = json.load(f)
    graph = template["graph"]
    assert {n["nodeType"] for n in graph.values()} == {
        "PrepareDenseScene", "DepthMap", "DepthMapFilter", "Meshing", "MeshFiltering", "Texturing", "Publish",
    }
    assert graph["PrepareDenseScene_1"]["inputs"]["input"] == str(sfm_path)
    assert not set(template["header"]["nodesVersions"]) & {"CameraInit", "StructureFromMotion"}
    text = json.dumps(graph)
    assert not any(f"{{{n}." in text for n in benchmark_meshroom.SFM_NODES)
