"""Shared fixtures."""
from pathlib import Path

import numpy as np
import pytest

SYNTH_W, SYNTH_H, SYNTH_F = 160, 120, 130.0
SYNTH_CENTRE = np.array([0.3, -0.2, 0.1])
SYNTH_RADIUS = 0.5


def write_ring_scene(root: Path, *, n_views: int = 16, model: str = "SIMPLE_PINHOLE",
                     outliers: int = 0, mask_radius: float = SYNTH_RADIUS) -> Path:
    """A ring of cameras around a textured ball: real images, disc masks, sparse/0 with points."""
    import pycolmap
    from PIL import Image

    rng = np.random.default_rng(0)
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    (root / "sparse" / "0").mkdir(parents=True)
    unit = rng.normal(size=(400, 3))
    points = SYNTH_CENTRE + SYNTH_RADIUS * unit / np.linalg.norm(unit, axis=1, keepdims=True)
    if outliers:
        points = np.vstack([points, SYNTH_CENTRE + rng.uniform(-30, 30, size=(outliers, 3))])
    recon, poses = pycolmap.Reconstruction(), {}
    cx, cy = SYNTH_W / 2 + 3, SYNTH_H / 2 - 2
    params = [SYNTH_F, cx, cy] if model == "SIMPLE_PINHOLE" else [SYNTH_F, cx, cy, 0.01]
    for i in range(1, n_views + 1):
        angle = 2 * np.pi * i / n_views
        centre = SYNTH_CENTRE + np.array([3 * np.cos(angle), 3 * np.sin(angle), 0.6 * np.sin(3 * angle)])
        z = (SYNTH_CENTRE - centre) / np.linalg.norm(SYNTH_CENTRE - centre)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        rotation = np.stack([x, np.cross(z, x), z], 1)
        recon.add_camera_with_trivial_rig(pycolmap.Camera(
            model=model, width=SYNTH_W, height=SYNTH_H, params=params, camera_id=i))
        poses[i] = pycolmap.Rigid3d(pycolmap.Rotation3d(rotation.T), -rotation.T @ centre)
        u, v = np.meshgrid(np.arange(SYNTH_W) + 0.5, np.arange(SYNTH_H) + 0.5)
        rays = np.stack([(u - cx) / SYNTH_F, (v - cy) / SYNTH_F, np.ones_like(u)], -1) @ rotation.T
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
        t = rays @ (SYNTH_CENTRE - centre)
        miss = np.linalg.norm(centre + t[..., None] * rays - SYNTH_CENTRE, axis=-1)
        mask = miss < mask_radius
        colour = np.where(mask[..., None], rng.integers(60, 255, (SYNTH_H, SYNTH_W, 3)), 40)
        Image.fromarray(colour.astype(np.uint8)).save(root / "images" / f"v{i:02d}.jpg", quality=95)
        Image.fromarray((mask * 255).astype(np.uint8)).save(root / "masks" / f"v{i:02d}.png")
    keypoints, tracks = {i: [] for i in poses}, [[] for _ in points]
    for i, pose in poses.items():
        for p, point in enumerate(points):
            local = pose * point
            if local[2] > 0.1:
                tracks[p].append((i, len(keypoints[i])))
                keypoints[i].append(recon.cameras[i].img_from_cam(local))
    for i, pose in poses.items():
        recon.add_image_with_trivial_frame(pycolmap.Image(
            name=f"v{i:02d}.jpg", keypoints=np.array(keypoints[i]), camera_id=i, image_id=i), pose)
    for point, track in zip(points, tracks):
        if len(track) >= 2:
            recon.add_point3D(point, pycolmap.Track([pycolmap.TrackElement(i, k) for i, k in track]))
    recon.write(str(root / "sparse" / "0"))
    return root


@pytest.fixture
def ring_scene(tmp_path):
    """Factory for write_ring_scene under tmp_path."""
    pytest.importorskip("pycolmap")
    count = iter(range(100))
    return lambda **kwargs: write_ring_scene(tmp_path / f"scene{next(count)}", **kwargs)
