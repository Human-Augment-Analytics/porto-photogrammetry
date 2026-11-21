"""Render textured meshes at retained COLMAP held-out cameras without fitting them.

Uses Open3D CPU ray casting and barycentric texture interpolation; no OpenGL context.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from augenblick.core.errors import SceneError
from augenblick.core.scene import Scene
from augenblick.eval.cameras import camera_records
from augenblick.eval.split import read_split, check_split


def image_size(camera: dict, resolution: int = -1) -> tuple[int, int]:
    width, height = int(camera["width"]), int(camera["height"])
    scale = max(width, height) / 1600 if resolution == -1 else resolution
    scale = max(scale, 1) if resolution == -1 else scale
    if scale <= 0:
        raise SceneError("render resolution must be -1 or a positive downscale")
    return max(1, int(width / scale)), max(1, int(height / scale))


def calibration(camera: dict, size: tuple[int, int]):
    if camera.get("camera_model", "PINHOLE") not in ("SIMPLE_PINHOLE", "PINHOLE"):
        raise SceneError("textured mesh renderer requires undistorted pinhole cameras")
    sx, sy = size[0] / camera["width"], size[1] / camera["height"]
    intrinsic = np.array([[camera["fx"] * sx, 0, camera.get("cx", camera["width"] / 2) * sx],
                          [0, camera["fy"] * sy, camera.get("cy", camera["height"] / 2) * sy],
                          [0, 0, 1]], dtype=np.float64)
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = np.array(camera["rotation"]).T
    extrinsic[:3, 3] = -extrinsic[:3, :3] @ np.array(camera["position"])
    return intrinsic, extrinsic


def sample_texture(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """OBJ UV origin is bottom left; image origin is top left. Bilinear, clamped edges."""
    height, width = image.shape[:2]
    x = np.clip(uv[:, 0], 0, 1) * (width - 1)
    y = (1 - np.clip(uv[:, 1], 0, 1)) * (height - 1)
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = (x - x0)[:, None], (y - y0)[:, None]
    return ((1-wx)*(1-wy)*image[y0, x0] + wx*(1-wy)*image[y0, x1]
            + (1-wx)*wy*image[y1, x0] + wx*wy*image[y1, x1])


class Renderer:
    def __init__(self, mesh_path: Path, mesh_frame: str = "colmap"):
        if mesh_frame not in ("colmap", "alicevision_obj"):
            raise SceneError("unknown mesh coordinate frame")
        import open3d as o3d
        import trimesh

        self.o3d = o3d
        self.mesh_frame = mesh_frame
        self.threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=self.threads)
        self.geometry = {}
        loaded = trimesh.load(str(mesh_path), force="scene", process=False)
        for mesh in loaded.dump():
            if not len(mesh.faces):
                continue
            vertices = np.array(mesh.vertices, dtype=np.float32, copy=True)
            if mesh_frame == "alicevision_obj":
                # AliceVision 3.3 Texturing::saveAs exports (x, -y, -z).
                # Undo this fixed basis change; no fitted alignment or test pixels.
                vertices[:, 1:] *= -1
            faces = np.asarray(mesh.faces, dtype=np.uint32)
            identifier = self.scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))
            visual = mesh.visual
            if visual.kind == "texture":
                material = visual.material
                materials = getattr(material, "materials", [material])
                textures = []
                for item in materials:
                    if not hasattr(item, "image") and hasattr(item, "to_simple"):
                        item = item.to_simple()
                    if getattr(item, "image", None) is None:
                        raise SceneError("textured mesh has a missing material image")
                    textures.append(np.asarray(item.image.convert("RGB"), dtype=np.float32))
                material_ids = getattr(visual, "face_materials", None)
                if material_ids is None:
                    material_ids = np.zeros(len(faces), dtype=int)
                self.geometry[identifier] = (faces, np.asarray(visual.uv), textures,
                                             np.asarray(material_ids), None)
            elif visual.kind == "vertex":
                colors = np.asarray(visual.vertex_colors[:, :3], dtype=np.float32)
                self.geometry[identifier] = (faces, None, None, None, colors)
            else:
                raise SceneError("mesh needs textures or vertex colors; refusing an invented surface color")
        if not self.geometry:
            raise SceneError("mesh has no renderable triangles")

    def render(self, camera: dict, size: tuple[int, int]) -> np.ndarray:
        intrinsic, extrinsic = calibration(camera, size)
        o3d = self.o3d
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            o3d.core.Tensor(intrinsic), o3d.core.Tensor(extrinsic), size[0], size[1])
        hits = self.scene.cast_rays(rays, nthreads=self.threads)
        geometry = hits["geometry_ids"].numpy().reshape(-1)
        primitives = hits["primitive_ids"].numpy().reshape(-1)
        uv = hits["primitive_uvs"].numpy().reshape(-1, 2)
        output = np.zeros((size[0] * size[1], 3), dtype=np.float32)
        for identifier, (faces, texture_uv, textures, material_ids, vertex_colors) in self.geometry.items():
            indices = np.flatnonzero(geometry == identifier)
            if not len(indices):
                continue
            face_ids = primitives[indices]
            barycentric = np.column_stack((1 - uv[indices].sum(axis=1), uv[indices]))
            if vertex_colors is not None:
                output[indices] = (vertex_colors[faces[face_ids]] * barycentric[:, :, None]).sum(axis=1)
            else:
                face_uv = texture_uv[faces[face_ids]]
                if self.mesh_frame == "alicevision_obj":
                    # AliceVision assigns each atlas a UDIM tile. Localize each
                    # triangle before sampling its material's individual image.
                    face_uv = face_uv - np.floor(face_uv.min(axis=1))[:, None, :]
                coordinates = (face_uv * barycentric[:, :, None]).sum(axis=1)
                materials = material_ids[face_ids]
                for material_index, texture in enumerate(textures):
                    selected = materials == material_index
                    output[indices[selected]] = sample_texture(texture, coordinates[selected])
        return np.clip(np.rint(output), 0, 255).astype(np.uint8).reshape(size[1], size[0], 3)


def render_output(scene: Scene, output: Path, mesh: Path, resolution: int = -1, mesh_frame: str = "colmap"):
    records = camera_records(scene.model_dir)
    split = read_split(scene.root)
    if split is None:
        raise SceneError("held-out rendering requires split.json")
    check_split(split, [record["img_name"] for record in records])
    renderer = Renderer(mesh, mesh_frame)
    target = Path(output) / "test" / "mesh"
    for name in ("renders", "gt"):
        directory = target / name
        if directory.is_symlink() or target.is_symlink() or Path(output).is_symlink():
            raise SceneError("render outputs must be private directories")
        directory.mkdir(parents=True, exist_ok=True)
        # A failed render stage may restart, but must not leave PNGs from another split.
        for old in directory.glob("*.png"):
            if old.is_symlink():
                raise SceneError("render output must not contain adopted PNG symlinks")
            old.unlink()
    held_out = set(split["test"])
    for camera in records:
        if camera["img_name"] not in held_out:
            continue
        size = image_size(camera, resolution)
        rendered = renderer.render(camera, size)
        Image.fromarray(rendered).save(target / "renders" / (camera["img_name"] + ".png"))
        with Image.open(scene.images_dir / camera["image_name"]) as photograph:
            truth = photograph.convert("RGB").resize(size, Image.Resampling.BICUBIC)
            truth.save(target / "gt" / (camera["img_name"] + ".png"))
    (target / "render_protocol.json").write_text(json.dumps({
        "representation": "textured_mesh", "renderer": "Open3D CPU raycasting, unlit texture",
        "mesh_frame": mesh_frame, "udim_tiles": mesh_frame == "alicevision_obj", "resolution": resolution, "test": sorted(held_out), "mesh": str(mesh),
        "photograph_resize": "PIL BICUBIC", "background": "black"}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--mesh-frame", choices=("colmap", "alicevision_obj"), default="colmap")
    args = parser.parse_args()
    render_output(Scene(args.scene), args.output, args.mesh, args.resolution, args.mesh_frame)


if __name__ == "__main__":
    main()
