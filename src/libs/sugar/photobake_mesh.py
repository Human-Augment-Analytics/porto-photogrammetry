"""Full-resolution photo UV-atlas bake for ANY reconstructed mesh, scored against ground truth.

Method-agnostic by design: the held-out cameras come directly from the shared COLMAP scene,
without a Gaussian checkpoint, so every backend's extracted mesh is
evaluated on the identical held-out split. For one mesh this renders, at each held-out camera:

  * the mesh's own *native* texture (UV map for OBJ meshes, per-vertex colours for the TSDF
    PLY meshes 2DGS/PGSR produce), and
  * a *full-resolution photo re-bake*: the same square-packed UV atlas SuGaR uses, but filled
    by projecting the full-res source photographs from the training views onto this geometry.

Both render sets are scored with augenblick.eval.nvs.score against the held-out photographs,
so the two textures are compared to ground truth on the same geometry and cameras.

Usage (run from the repo root so this file's dir is on sys.path):
  PYTHONPATH=src python src/libs/sugar/photobake_mesh.py \
    --scene <colmap> --resolution -1 \
      --mesh <method-mesh.ply|obj> --output <model-dir>
"""
import argparse
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import TexturesUV, TexturesVertex
from pytorch3d.structures import Meshes

SUGAR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SUGAR_DIR / "gaussian_splatting"))
# augenblick lives under <repo>/src; add it so scoring can reuse the shared NVS protocol.
REPO_SRC = SUGAR_DIR.parents[1]
sys.path.insert(0, str(REPO_SRC))

from sugar_utils.mesh_rasterization import MeshRasterizer
# These helpers are pure (no SuGaR-model state), so they are safe to reuse for any mesh.
from sugar_extractors.texture import _accumulate_samples, _load_photo_camera, build_square_atlas
from sugar_utils.photobake_cameras import load_evaluation_photo, load_scene_cameras


def _output_dirs(output_path, target, iteration):
    root = Path(output_path) / target / f"ours_{iteration}"
    renders = root / "renders"
    ground_truth = root / "gt"
    renders.mkdir(parents=True, exist_ok=True)
    ground_truth.mkdir(parents=True, exist_ok=True)
    return root, renders, ground_truth


def _save_rgb(image, path):
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3].permute(2, 0, 1)
    torchvision.utils.save_image(image[:3].clamp(0, 1), path)


def native_atlas_metadata(mesh_path, mesh, native_kind):
    """Report native export settings only when a matching nearby manifest records them."""
    result = {
        "square_size": None,
        "square_size_status": "unknown" if native_kind == "uv" else "not_applicable",
        "manifest": None,
        "sampling_original": "unknown",
        "sampling_loaded": getattr(mesh.textures, "sampling_mode", None),
    }
    if native_kind != "uv":
        return result
    mesh_path = Path(mesh_path).resolve()
    for directory in list(mesh_path.parents)[:3]:
        candidate = directory / "run_manifest.json"
        if not candidate.is_file():
            continue
        try:
            with candidate.open() as handle:
                manifest = json.load(handle)
        except (OSError, ValueError) as error:
            warnings.warn(f"Cannot read native manifest {candidate}: {error}", RuntimeWarning)
            continue
        recorded_mesh = manifest.get("textured_mesh_path")
        if not recorded_mesh:
            continue
        recorded_path = Path(recorded_mesh)
        if not recorded_path.is_absolute():
            recorded_path = candidate.parent / recorded_path
        if recorded_path.resolve() != mesh_path:
            continue
        result["manifest"] = str(candidate)
        square_size = manifest.get("texture_square_size")
        if type(square_size) is int and square_size > 0:
            result["square_size"] = square_size
            result["square_size_status"] = "known"
        result["sampling_original"] = manifest.get("texture_sampling_mode") or "unknown"
        break
    return result


def load_method_mesh(mesh_path, device):
    """Load a backend's mesh into a pytorch3d Meshes, keeping its native texture.

    OBJ meshes carry a UV map + material image (SuGaR, GW textured export); PLY meshes from
    TSDF fusion (2DGS, PGSR) carry per-vertex colours. Returns the Meshes and a short label
    describing which native texture kind was found.
    """
    path = Path(mesh_path)
    if path.suffix.lower() == ".obj":
        mesh = load_objs_as_meshes([str(path)], device=device)
        kind = "uv" if isinstance(mesh.textures, TexturesUV) else "none"
        return mesh, kind

    import trimesh

    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    verts = torch.tensor(np.asarray(loaded.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(loaded.faces), dtype=torch.int64, device=device)

    textures = None
    kind = "none"
    visual = getattr(loaded, "visual", None)
    # A UV-mapped PLY (e.g. a textured GW export) round-trips through trimesh as a
    # TextureVisuals with a material image; a TSDF PLY exposes per-vertex colours instead.
    if visual is not None and getattr(visual, "uv", None) is not None and \
            getattr(getattr(visual, "material", None), "image", None) is not None:
        uv = torch.tensor(np.asarray(visual.uv), dtype=torch.float32, device=device)
        image = np.asarray(visual.material.image.convert("RGB"), dtype=np.float32) / 255.0
        maps = torch.tensor(image, device=device)[None]
        faces_uvs = faces.to(torch.int64)
        textures = TexturesUV(maps=maps, verts_uvs=uv[None], faces_uvs=faces_uvs[None])
        kind = "uv"
    elif visual is not None and getattr(visual, "vertex_colors", None) is not None:
        colours = torch.tensor(
            np.asarray(visual.vertex_colors)[:, :3], dtype=torch.float32, device=device) / 255.0
        textures = TexturesVertex(verts_features=colours[None])
        kind = "vertex"

    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    return mesh, kind


def _render_native(rasterizer, mesh, camera_index, background, device):
    """Render one held-out view of the mesh's own texture (UV map or per-vertex colours)."""
    fragments = rasterizer(mesh, cam_idx=camera_index)
    pix_to_face = fragments.pix_to_face.long()
    foreground = pix_to_face[0, :, :, 0] >= 0

    if isinstance(mesh.textures, TexturesUV):
        p3d_fragments = SimpleNamespace(
            pix_to_face=pix_to_face, bary_coords=fragments.bary_coords,
            zbuf=fragments.zbuf, dists=None)
        colors = mesh.textures.sample_textures(p3d_fragments)[0, :, :, 0, :3]
    elif isinstance(mesh.textures, TexturesVertex):
        faces = mesh.faces_packed()
        verts_features = mesh.textures.verts_features_packed()
        faces_features = verts_features[faces]
        colors = interpolate_face_attributes(
            pix_to_face, fragments.bary_coords, faces_features)[0, :, :, 0, :3]
    else:
        raise ValueError("mesh has no native texture to render (no UV map or vertex colours)")

    fill = torch.tensor(background, device=device, dtype=colors.dtype)
    return torch.where(foreground.unsqueeze(-1), colors, fill)


def _render_uv(rasterizer, mesh, camera_index, background, device):
    """Render one held-out view of a UV-textured mesh (used for the photo re-bake)."""
    fragments = rasterizer(mesh, cam_idx=camera_index)
    pix_to_face = fragments.pix_to_face.long()
    p3d_fragments = SimpleNamespace(
        pix_to_face=pix_to_face, bary_coords=fragments.bary_coords,
        zbuf=fragments.zbuf, dists=None)
    colors = mesh.textures.sample_textures(p3d_fragments)[0, :, :, 0, :3]
    foreground = pix_to_face[0, :, :, 0] >= 0
    fill = torch.tensor(background, device=device, dtype=colors.dtype)
    return torch.where(foreground.unsqueeze(-1), colors, fill)


@torch.no_grad()
def bake_photo_texture(surface_mesh, training_cameras, source_path, device,
                       square_size=10, photo_max_size=-1):
    """Bake a UV texture onto surface_mesh by projecting full-res training photos.

    This is SuGaR's square-packed per-triangle atlas bake (sugar_extractors.texture), lifted
    off the SuGaR model so it runs on any triangle mesh: it rasterises the mesh from each
    training camera, projects the masked source photograph, and averages the samples per texel.
    Faces never seen get the mean colour of the seen faces instead of SuGaR's SH fallback.
    """
    if square_size < 5:
        raise ValueError("square_size must be >= 5")
    faces = surface_mesh.faces_packed()
    n_triangles = len(faces)
    faces_uv, verts_uv, uv_to_faces, texture_size = build_square_atlas(
        n_triangles, square_size, device)
    print(f"Photo texture atlas: square_size={square_size}, texture_size={texture_size}, sampling=nearest")

    texture_img = torch.zeros(texture_size, texture_size, 3, device=device)
    texture_count = torch.zeros(texture_size, texture_size, 1, device=device)
    face_colors = torch.zeros(n_triangles, 3, device=device)
    face_count = torch.zeros(n_triangles, 1, device=device)

    rasterizer = None

    for cam_idx in range(len(training_cameras)):
        rgb_img, source_mask, render_camera = _load_photo_camera(
            source_path, training_cameras.gs_cameras[cam_idx], photo_max_size, device)
        height, width = rgb_img.shape[1:3]
        if rasterizer is None or (rasterizer.height, rasterizer.width) != (height, width):
            rasterizer = MeshRasterizer(cameras=render_camera, use_nvdiffrast=True)
        fragments = rasterizer(surface_mesh, cameras=render_camera, cam_idx=0)
        bary_coords = fragments.bary_coords.view(1, height, width, 3)
        pix_to_face = fragments.pix_to_face.view(1, height, width)

        mask = pix_to_face > -1
        if source_mask is not None:
            mask &= source_mask.view(1, height, width)
        face_indices = pix_to_face[mask]
        bary_coords = bary_coords[mask]
        colors = rgb_img[mask]

        pixel_idx_0 = ((verts_uv[faces_uv[face_indices]] * bary_coords[:, :, None]).sum(dim=1)
                       * texture_size).int()
        pixel_idx_0.clamp_(0, texture_size - 1)
        _accumulate_samples(face_colors, face_count, texture_img, texture_count,
                            face_indices, pixel_idx_0, colors)
        del fragments, bary_coords, pix_to_face, colors, rgb_img, render_camera

    filled_mask = texture_count[..., 0] > 0
    texture_img[filled_mask] = texture_img[filled_mask] / texture_count[filled_mask]

    visited_faces_mask = face_count[..., 0] > 0
    face_colors[visited_faces_mask] = face_colors[visited_faces_mask] / face_count[visited_faces_mask]
    if visited_faces_mask.any():
        face_colors[~visited_faces_mask] = face_colors[visited_faces_mask].mean(dim=0)
    else:
        face_colors[~visited_faces_mask] = 0.5
    texture_img[~filled_mask] = face_colors[uv_to_faces[~filled_mask]]
    texture_img = texture_img.flip(0)

    textures_uv = TexturesUV(
        maps=texture_img[None].float(),
        verts_uvs=verts_uv[None],
        faces_uvs=faces_uv[None],
        sampling_mode="nearest",
    )
    return Meshes(
        verts=[surface_mesh.verts_list()[0]],
        faces=[surface_mesh.faces_list()[0]],
        textures=textures_uv,
    )


@torch.no_grad()
def run(scene_path, mesh_path, output_path,
        gpu=0, white_background=False, iteration=0, square_size=10, photo_max_size=-1,
        skip_native=False, skip_bake=False, save_baked_obj=True, score=True, resolution=-1):
    if square_size < 5:
        raise ValueError("square_size must be >= 5")
    torch.cuda.set_device(gpu)
    device = f"cuda:{gpu}"
    background = [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0]

    training_cameras, test_cameras = load_scene_cameras(scene_path, resolution, device)

    mesh, native_kind = load_method_mesh(mesh_path, device)
    print(f"Loaded mesh {mesh_path} (native texture: {native_kind})")
    metadata = {
        "camera_source": str(Path(scene_path).resolve() / "sparse" / "0"),
        "camera_loader": "readColmapSceneInfo(eval=True, load_images=False, load_point_cloud=False)",
        "split_source": "split.json" if (Path(scene_path) / "split.json").is_file() else "llffhold=8",
        "train_views": [camera.image_name for camera in training_cameras.gs_cameras],
        "test_views": [camera.image_name for camera in test_cameras.gs_cameras],
        "evaluation_resolution": resolution,
        "output_iteration": iteration,
        "white_background": white_background,
        "evaluation_sizes": {
            camera.image_name: [camera.image_width, camera.image_height]
            for camera in test_cameras.gs_cameras
        },
        "photo_max_size": photo_max_size,
        "photo_square_size": square_size,
        "photo_sampling": "nearest",
        "photo_unseen_face_fallback": "mean_seen_faces_or_gray_0.5",
        "native_atlas": native_atlas_metadata(mesh_path, mesh, native_kind),
        "comparison": "native_export_vs_photo_rebake_not_controlled_color_source_ablation",
    }
    metadata_path = Path(output_path) / "photobake" / "run_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Photobake settings: {json.dumps(metadata, sort_keys=True)}")

    metrics_written = {}
    if not skip_native:
        if native_kind == "none":
            print("Mesh carries no native texture; skipping the native-texture render")
        else:
            _, renders_dir, gt_dir = _output_dirs(output_path, "test_mesh_native", iteration)
            for camera in test_cameras.gs_cameras:
                rasterizer = MeshRasterizer(cameras=camera)
                stem = Path(camera.image_name).stem
                image = _render_native(rasterizer, mesh, 0, background, device)
                _save_rgb(image, renders_dir / f"{stem}.png")
                ground_truth, _ = load_evaluation_photo(scene_path, camera, background, device)
                _save_rgb(ground_truth, gt_dir / f"{stem}.png")
                del image, ground_truth
            print(f"Wrote native-texture renders -> {renders_dir}")
            if score:
                metrics_written["native"] = _score(
                    scene_path, output_path, "test_mesh_native", iteration,
                    "nvs_metrics_mesh_native.json", mesh_path, native_kind, metadata)

    if not skip_bake:
        textured = bake_photo_texture(
            mesh, training_cameras, scene_path, device,
            square_size=square_size, photo_max_size=photo_max_size)
        if save_baked_obj:
            obj_path = Path(output_path) / "photobake" / "photofull_mesh.obj"
            obj_path.parent.mkdir(parents=True, exist_ok=True)
            tex = textured.textures
            save_obj(
                str(obj_path),
                verts=textured.verts_list()[0],
                faces=textured.faces_list()[0],
                verts_uvs=tex.verts_uvs_list()[0],
                faces_uvs=tex.faces_uvs_list()[0],
                texture_map=tex.maps_padded()[0],
            )
            print(f"Saved photo-baked mesh -> {obj_path}")

        _, renders_dir, gt_dir = _output_dirs(output_path, "test_mesh_photofull", iteration)
        for camera in test_cameras.gs_cameras:
            rasterizer = MeshRasterizer(cameras=camera)
            stem = Path(camera.image_name).stem
            image = _render_uv(rasterizer, textured, 0, background, device)
            _save_rgb(image, renders_dir / f"{stem}.png")
            ground_truth, _ = load_evaluation_photo(scene_path, camera, background, device)
            _save_rgb(ground_truth, gt_dir / f"{stem}.png")
            del image, ground_truth
        print(f"Wrote photo-bake renders -> {renders_dir}")
        if score:
            metrics_written["photofull"] = _score(
                scene_path, output_path, "test_mesh_photofull", iteration,
                "nvs_metrics_mesh_photofull.json", mesh_path, native_kind, metadata)

    return metrics_written


def _score(scene_path, output_path, target, iteration, out_name, mesh_path, native_kind,
           metadata=None):
    """Score one render set against the held-out photos with the shared NVS protocol."""
    from augenblick.core.scene import Scene
    from augenblick.eval.nvs import find_test_dir, resolve_test_stems
    from augenblick.eval.nvs import score as nvs_score

    scene = Scene(Path(scene_path))
    test_dir = find_test_dir(Path(output_path) / target, iteration)
    result = nvs_score(
        test_dir,
        scene.masks_dir,
        resolve_test_stems(scene),
        Path(output_path) / out_name,
        extra={
            **(metadata or {}),
            "target": target,
            "mesh": str(mesh_path),
            "native_texture": native_kind,
            "scene": str(scene.root),
            "model": str(output_path),
        },
    )
    print(f"{target}: psnr={result['psnr']:.4f} ssim={result['ssim']:.4f} "
          f"lpips={result['lpips']:.4f} (n={result['n_test']})")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, help="COLMAP scene (images/, masks/, split.json)")
    parser.add_argument("--mesh", required=True, help="The backend's mesh (.obj or .ply)")
    parser.add_argument("--output", required=True, help="Model directory to write renders/metrics into")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--white-background", action="store_true")
    parser.add_argument("--iteration", type=int, default=0, help="ours_<n> label for the render dirs")
    parser.add_argument("--square-size", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=-1,
                        help="Held-out resolution: -1 caps width at 1600; 1/2/4/8 downsample; otherwise width")
    parser.add_argument("--photo-max-size", type=int, default=-1,
                        help="Cap the long side of source photos; -1 keeps full resolution")
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--skip-bake", action="store_true")
    parser.add_argument("--no-save-obj", action="store_true")
    parser.add_argument("--no-score", action="store_true")
    args = parser.parse_args()

    run(
        args.scene,
        args.mesh,
        args.output,
        gpu=args.gpu,
        white_background=args.white_background,
        iteration=args.iteration,
        square_size=args.square_size,
        photo_max_size=args.photo_max_size,
        resolution=args.resolution,
        skip_native=args.skip_native,
        skip_bake=args.skip_bake,
        save_baked_obj=not args.no_save_obj,
        score=not args.no_score,
    )


if __name__ == "__main__":
    main()
