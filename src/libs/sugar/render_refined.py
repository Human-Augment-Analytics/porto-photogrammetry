"""Render held-out views from refined SuGaR splats and the final textured mesh."""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torchvision
from pytorch3d.io import load_objs_as_meshes

SUGAR_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SUGAR_DIR / "gaussian_splatting"))

from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import load_refined_model
from sugar_utils.mesh_rasterization import MeshRasterizer as SugarMeshRasterizer


def _output_dirs(output_path, target, iteration):
    root = Path(output_path) / target / f"ours_{iteration}"
    renders = root / "renders"
    ground_truth = root / "gt"
    renders.mkdir(parents=True, exist_ok=True)
    ground_truth.mkdir(parents=True, exist_ok=True)
    return renders, ground_truth


def _save_rgb(image, path):
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        image = image[..., :3].permute(2, 0, 1)
    torchvision.utils.save_image(image[:3].clamp(0, 1), path)


def _render_textured_mesh(rasterizer, mesh, camera_index, background):
    """Render one held-out view of the UV-textured mesh through the GS-camera rasterizer."""
    fragments = rasterizer(mesh, cam_idx=camera_index)
    pix_to_face = fragments.pix_to_face.long()
    p3d_fragments = SimpleNamespace(
        pix_to_face=pix_to_face,
        bary_coords=fragments.bary_coords,
        zbuf=fragments.zbuf,
        dists=None,
    )
    colors = mesh.textures.sample_textures(p3d_fragments)[0, :, :, 0, :3]
    foreground = pix_to_face[0, :, :, 0] >= 0
    fill = torch.tensor(background, device=colors.device, dtype=colors.dtype)
    return torch.where(foreground.unsqueeze(-1), colors, fill)


@torch.no_grad()
def render_held_out(scene_path, manifest_path, output_path, gpu, white_background,
                    textured_mesh_path=None, mesh_target="test_mesh", skip_splat=False):
    with open(manifest_path) as handle:
        manifest = json.load(handle)

    torch.cuda.set_device(gpu)
    nerfmodel = GaussianSplattingWrapper(
        source_path=scene_path,
        output_path=manifest["vanilla_checkpoint_path"],
        iteration_to_load=manifest["iteration_to_load"],
        load_gt_images=True,
        eval_split=True,
        white_background=white_background,
    )
    refined_sugar = None
    if not skip_splat:
        refined_sugar = load_refined_model(manifest["refined_model_path"], nerfmodel)
        refined_sugar.eval()

    iteration = manifest["refinement_iterations"]
    if not skip_splat:
        splat_renders, splat_gt = _output_dirs(output_path, "test_splat", iteration)
    mesh_renders, mesh_gt = _output_dirs(output_path, mesh_target, iteration)
    background = [1.0, 1.0, 1.0] if white_background else [0.0, 0.0, 0.0]

    textured_mesh_path = textured_mesh_path or manifest.get("textured_mesh_path")
    if not textured_mesh_path or not os.path.isfile(textured_mesh_path):
        raise FileNotFoundError("SuGaR evaluation requires the final UV-textured OBJ")
    textured_mesh = load_objs_as_meshes([textured_mesh_path], device=nerfmodel.device)
    mesh_rasterizer = SugarMeshRasterizer(cameras=nerfmodel.test_cameras)

    for camera_index, camera in enumerate(nerfmodel.test_cameras.gs_cameras):
        stem = Path(camera.image_name).stem
        ground_truth = camera.original_image[:3]
        if refined_sugar is not None:
            splat = refined_sugar.render_image_gaussian_rasterizer(
                nerf_cameras=nerfmodel.test_cameras,
                camera_indices=camera_index,
                verbose=False,
                bg_color=torch.tensor(background, device=nerfmodel.device),
                sh_deg=refined_sugar.sh_levels - 1,
                compute_color_in_rasterizer=True,
            )
        mesh = _render_textured_mesh(mesh_rasterizer, textured_mesh, camera_index, background)

        if refined_sugar is not None:
            _save_rgb(splat, splat_renders / f"{stem}.png")
            _save_rgb(ground_truth, splat_gt / f"{stem}.png")
        _save_rgb(mesh, mesh_renders / f"{stem}.png")
        _save_rgb(ground_truth, mesh_gt / f"{stem}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument("--textured_mesh", default=None,
                        help="Alternate textured OBJ; defaults to the manifest mesh")
    parser.add_argument("--mesh_target", default="test_mesh",
                        help="Output test-root name for the mesh renders")
    parser.add_argument("--skip_splat", action="store_true",
                        help="Render only the textured mesh")
    args = parser.parse_args()
    render_held_out(
        args.scene,
        args.manifest,
        args.output,
        args.gpu,
        args.white_background,
        textured_mesh_path=args.textured_mesh,
        mesh_target=args.mesh_target,
        skip_splat=args.skip_splat,
    )


if __name__ == "__main__":
    main()