"""Scene-owned, split-aware cameras and lazy photographs for mesh evaluation."""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from scene.dataset_readers import readColmapSceneInfo
from sugar_scene.cameras import CamerasWrapper, GSCamera
from sugar_utils.camera_resolution import camera_resolution


def load_scene_cameras(scene_path, resolution=-1, device="cuda"):
    """Use sparse/0 poses and the reader's split without checkpoint or GT tensors."""
    if resolution != -1 and resolution <= 0:
        raise ValueError("resolution must be -1 or a positive integer")
    scene = readColmapSceneInfo(str(scene_path), images=None, eval=True,
                               load_images=False, load_point_cloud=False)
    infos = [*scene.train_cameras, *scene.test_cameras]
    try:
        split_path = Path(scene_path) / "split.json"
        if split_path.is_file():
            manifest = json.loads(split_path.read_text())
            train_names, test_names = manifest["train"], manifest["test"]
            if (len(set(train_names)) != len(train_names)
                    or len(set(test_names)) != len(test_names)
                    or set(train_names) & set(test_names)):
                raise ValueError("split.json must contain unique, disjoint train/test views")
            for label, expected, cameras in (
                ("train", train_names, scene.train_cameras),
                ("test", test_names, scene.test_cameras),
            ):
                if set(expected) != {camera.image_name for camera in cameras}:
                    raise ValueError(f"split.json {label} views do not match the scene cameras")
        if not scene.train_cameras or not scene.test_cameras:
            raise ValueError("photobake requires nonempty training and held-out camera sets")
        wrappers = []
        for split in (scene.train_cameras, scene.test_cameras):
            cameras = []
            for camera_index, info in enumerate(split):
                with Image.open(info.image_path) as source:
                    width, height = camera_resolution(*source.size, resolution)
                camera = GSCamera(
                    colmap_id=info.uid, R=info.R, T=info.T,
                    FoVx=info.FovX, FoVy=info.FovY,
                    image=None, gt_alpha_mask=None, image_name=info.image_name,
                    uid=camera_index, image_height=height, image_width=width,
                    data_device=str(device),
                )
                camera.image_path = str(info.image_path)
                cameras.append(camera)
            wrappers.append(CamerasWrapper(gs_cameras=cameras))
        return tuple(wrappers)
    finally:
        for info in infos:
            if info.image is not None:
                info.image.close()


def load_evaluation_photo(scene_path, camera, background, device):
    """Load one held-out RGB and its nearest-resized mask at the render resolution."""
    size = camera.image_width, camera.image_height
    with Image.open(camera.image_path) as source:
        rgba = source.convert("RGBA").resize(size, Image.Resampling.LANCZOS)
        array = np.asarray(rgba, dtype=np.float32).copy() / 255.0
    foreground = torch.from_numpy(array[..., 3].copy()).to(device)
    mask_path = Path(scene_path) / "masks" / f"{camera.image_name}.png"
    if mask_path.is_file():
        with Image.open(mask_path) as source_mask:
            mask = source_mask.convert("L").resize(size, Image.Resampling.NEAREST)
            foreground = torch.from_numpy((np.asarray(mask) > 127).copy()).to(device)
    image = torch.from_numpy(array[..., :3].copy()).to(device)
    fill = torch.tensor(background, dtype=image.dtype, device=device)
    image = image * foreground[..., None] + fill * (1 - foreground.float()[..., None])
    return image, foreground