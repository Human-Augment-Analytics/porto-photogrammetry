"""Export registered COLMAP cameras in the benchmark camera-coordinate convention."""
import json
from pathlib import Path

from augenblick.core.errors import SceneError


def camera_records(model_dir: Path) -> list[dict]:
    import numpy as np
    import pycolmap

    model = pycolmap.Reconstruction(str(model_dir))
    records = []
    for index, image in enumerate(sorted(model.images.values(), key=lambda i: i.name)):
        camera = model.cameras[image.camera_id]
        pose = image.cam_from_world
        pose = pose() if callable(pose) else pose
        world_to_camera = pose.matrix()
        rotation = world_to_camera[:, :3]
        center = -rotation.T @ world_to_camera[:, 3]
        intrinsic = camera.calibration_matrix()
        records.append({"id": index, "img_name": Path(image.name).stem,
                        "image_name": image.name, "width": camera.width, "height": camera.height,
                        "fx": float(intrinsic[0, 0]), "fy": float(intrinsic[1, 1]),
                        "cx": float(intrinsic[0, 2]), "cy": float(intrinsic[1, 2]),
                        "position": center.tolist(), "rotation": rotation.T.tolist(),
                        "camera_model": camera.model_name, "camera_params": camera.params.tolist()})
    names = [record["img_name"] for record in records]
    if not records or len(names) != len(set(names)):
        raise SceneError("COLMAP cameras must have nonempty, unique image stems")
    return records


def export_cameras(model_dir: Path, output: Path) -> list[dict]:
    records = camera_records(model_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(records, indent=2) + "\n")
    temporary.replace(output)
    return records
