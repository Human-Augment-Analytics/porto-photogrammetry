import numpy as np
import torch
from pathlib import Path
from PIL import Image
from sugar_scene.sugar_model import SuGaR
from sugar_scene.cameras import GSCamera
from sugar_utils.spherical_harmonics import SH2RGB
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV
from sugar_utils.mesh_rasterization import MeshRasterizer, RasterizationSettings


def _find_source_image(source_path, image_name):
    """Find one source photograph by camera stem, independent of its extension."""
    images_dir = Path(source_path) / "images"
    matches = [path for path in images_dir.iterdir() if path.is_file() and path.stem == image_name]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one source image for {image_name!r} under {images_dir}, found {len(matches)}")
    return matches[0]


def _load_photo_camera(source_path, camera, max_image_size, device):
    """Load one source photograph and clone its camera at the photograph's resolution."""
    image_path = _find_source_image(source_path, camera.image_name)
    with Image.open(image_path) as source:
        source = source.convert("RGB")
        width, height = source.size
        if max_image_size > 0 and max(width, height) > max_image_size:
            scale = max_image_size / max(width, height)
            width, height = round(width * scale), round(height * scale)
            source = source.resize((width, height), Image.Resampling.LANCZOS)
        image = torch.from_numpy(np.asarray(source, dtype=np.float32).copy()).to(device) / 255.0

    foreground = None
    mask_path = Path(source_path) / "masks" / f"{camera.image_name}.png"
    if mask_path.is_file():
        with Image.open(mask_path) as source_mask:
            source_mask = source_mask.convert("L").resize((width, height), Image.Resampling.NEAREST)
            foreground = torch.from_numpy((np.asarray(source_mask) > 127).copy()).to(device)

    full_resolution_camera = GSCamera(
        colmap_id=camera.colmap_id,
        R=camera.R,
        T=camera.T,
        FoVx=camera.FoVx,
        FoVy=camera.FoVy,
        image=None,
        gt_alpha_mask=None,
        image_name=camera.image_name,
        uid=camera.uid,
        image_height=height,
        image_width=width,
        data_device=str(device),
    )
    return image.view(1, height, width, 3), foreground, full_resolution_camera


def _accumulate_samples(face_colors, face_count, texture_img, texture_count,
                        face_indices, texture_pixels, colors):
    """Accumulate projected samples, including repeated face and texel indices."""
    ones = torch.ones((len(face_indices), 1), dtype=colors.dtype, device=colors.device)
    face_colors.index_add_(0, face_indices, colors)
    face_count.index_add_(0, face_indices, ones)

    texture_size = texture_img.shape[1]
    flat_pixels = texture_pixels[:, 1] * texture_size + texture_pixels[:, 0]
    texture_img.view(-1, 3).index_add_(0, flat_pixels, colors)
    texture_count.view(-1, 1).index_add_(0, flat_pixels, ones)


@torch.no_grad()
def compute_textured_mesh_for_sugar_mesh(
    sugar:SuGaR,
    square_size:int=10,
    n_sh=0,
    texture_with_gaussian_renders=True,
    bg_color=[0., 0., 0.],
):  
    device = sugar.device
    
    if sugar.nerfmodel is None:
        raise ValueError("You must provide a NerfModel to use this function.")

    if square_size < 3:
        raise ValueError("square_size must be >= 3")

    surface_mesh = sugar.surface_mesh
    faces = surface_mesh.faces_packed()

    n_triangles = len(faces)
    n_squares = n_triangles // 2 + 1
    n_square_per_axis = int(np.sqrt(n_squares) + 1)
    texture_size = square_size * (n_square_per_axis)

    # Build faces UV.
    # Each face will have 3 corresponding vertices in the UV map
    faces_uv = torch.arange(3 * n_triangles, device=device).view(n_triangles, 3)  # n_triangles, 3

    # Build corresponding vertices UV
    vertices_uv = torch.cartesian_prod(
        torch.arange(n_square_per_axis, device=device), 
        torch.arange(n_square_per_axis, device=device))
    bottom_verts_uv = torch.cat(
        [vertices_uv[n_square_per_axis:-1, None], vertices_uv[:-n_square_per_axis-1, None], vertices_uv[n_square_per_axis+1:, None]],
        dim=1)
    top_verts_uv = torch.cat(
        [vertices_uv[1:-n_square_per_axis, None], vertices_uv[:-n_square_per_axis-1, None], vertices_uv[n_square_per_axis+1:, None]],
        dim=1)

    vertices_uv = torch.cartesian_prod(
        torch.arange(n_square_per_axis, device=device), 
        torch.arange(n_square_per_axis, device=device))[:, None]
    u_shift = torch.tensor([[1, 0]], dtype=torch.int32, device=device)[:, None]
    v_shift = torch.tensor([[0, 1]], dtype=torch.int32, device=device)[:, None]
    bottom_verts_uv = torch.cat(
        [vertices_uv + u_shift, vertices_uv, vertices_uv + u_shift + v_shift],
        dim=1)
    top_verts_uv = torch.cat(
        [vertices_uv + v_shift, vertices_uv, vertices_uv + u_shift + v_shift],
        dim=1)

    verts_uv = torch.cat([bottom_verts_uv, top_verts_uv], dim=1)
    verts_uv = verts_uv * square_size
    verts_uv[:, 0] = verts_uv[:, 0] + torch.tensor([[-2, 1]], device=device)
    verts_uv[:, 1] = verts_uv[:, 1] + torch.tensor([[2, 1]], device=device)
    verts_uv[:, 2] = verts_uv[:, 2] + torch.tensor([[-2, -3]], device=device)
    verts_uv[:, 3] = verts_uv[:, 3] + torch.tensor([[1, -1]], device=device)
    verts_uv[:, 4] = verts_uv[:, 4] + torch.tensor([[1, 3]], device=device)
    verts_uv[:, 5] = verts_uv[:, 5] + torch.tensor([[-3, -1]], device=device)
    
    verts_uv = verts_uv.reshape(-1, 2) / texture_size
    print("Building UV map done.")
    
    # Get, for each pixel, the corresponding face
    uvs_coords = torch.cartesian_prod(
        torch.arange(texture_size, device=device, dtype=torch.int32), 
        torch.arange(texture_size, device=device, dtype=torch.int32),
    ).view(texture_size, texture_size, 2)

    square_of_uvs = uvs_coords // square_size
    square_of_uvs = square_of_uvs[..., 0] * n_square_per_axis + square_of_uvs[..., 1]

    uvs_in_top_triangle = uvs_coords % square_size
    uvs_in_top_triangle = uvs_in_top_triangle[..., 0] < uvs_in_top_triangle[..., 1]

    uv_to_faces = 2 * square_of_uvs + uvs_in_top_triangle
    uv_to_faces = uv_to_faces.transpose(0, 1).clamp_max(n_triangles - 1)
    
    # Build Texture
    texture_img = torch.zeros(texture_size, texture_size, 3, device=device)
    texture_count = torch.zeros(texture_size, texture_size, 1, device=device)
    
    # Average color of visited faces 
    face_colors = torch.zeros(n_triangles, 3, device=device)
    face_count = torch.zeros(n_triangles, 1, device=device)
    
    # Color of non visited faces computed using SH
    non_visited_face_colors = SH2RGB(sugar._sh_coordinates_dc[:, 0]).clamp(0., 1.).view(n_triangles, -1, 3)
    non_visited_face_colors = non_visited_face_colors.mean(dim=1)
    
    # Build rasterizer
    height = sugar.nerfmodel.training_cameras.gs_cameras[0].image_height
    width = sugar.nerfmodel.training_cameras.gs_cameras[0].image_width
    raster_settings = RasterizationSettings(image_size=(height, width))
    rasterizer = MeshRasterizer(
        cameras=sugar.nerfmodel.training_cameras,
        raster_settings=raster_settings,
        use_nvdiffrast=True,
    )
    
    print(f"Processing images...")
    for cam_idx in range(len(sugar.nerfmodel.training_cameras)):
        if texture_with_gaussian_renders:
            rgb_img = sugar.render_image_gaussian_rasterizer(
                nerf_cameras=sugar.nerfmodel.training_cameras,
                camera_indices=cam_idx,
                sh_deg=n_sh,
                bg_color=torch.tensor(bg_color, device=device),
                compute_color_in_rasterizer=True,
            ).nan_to_num().clamp(min=0, max=1)
            rgb_img = rgb_img.view(1, height, width, 3)
        else:
            raise NotImplementedError("Should use GT RGB image if texture_with_gaussian_renders is False.")
        fragments = rasterizer(sugar.surface_mesh, cam_idx=cam_idx)
        bary_coords = fragments.bary_coords.view(1, height, width, 3)
        pix_to_face = fragments.pix_to_face.view(1, height, width)

        mask = pix_to_face > -1
        face_indices = pix_to_face[mask]
        bary_coords = bary_coords[mask]
        colors = rgb_img[mask]
        
        face_count[face_indices] = face_count[face_indices] + 1 
        face_colors[face_indices] = face_colors[face_indices] + colors

        pixel_idx_0 = ((verts_uv[faces_uv[face_indices]] * bary_coords[:, :, None]).sum(dim=1) * texture_size).int()
        texture_img[pixel_idx_0[:, 1], pixel_idx_0[:, 0]] = texture_img[pixel_idx_0[:, 1], pixel_idx_0[:, 0]] + colors
        texture_count[pixel_idx_0[:, 1], pixel_idx_0[:, 0]] = texture_count[pixel_idx_0[:, 1], pixel_idx_0[:, 0]] + 1
        
    # For visited UV points, we just average the colors from the rendered images
    filled_mask = texture_count[..., 0] > 0
    texture_img[filled_mask] = texture_img[filled_mask] / texture_count[filled_mask]

    # For non visited UV points belonging to visited faces, we use the average color of the face (computed from visited pixels)
    visited_faces_mask = face_count[..., 0] > 0
    face_colors[visited_faces_mask] = face_colors[visited_faces_mask] / face_count[visited_faces_mask]
    
    # For non visited UV points belonging to non visited faces, we use the averaged SH color
    face_colors[~visited_faces_mask] = non_visited_face_colors[~visited_faces_mask]
    
    # We fill the unvisited UV points with the corresponding face color
    texture_img[~filled_mask] = face_colors[uv_to_faces[~filled_mask]]

    texture_img = texture_img.flip(0)
    
    # Return the textured mesh
    textures_uv = TexturesUV(
        maps=texture_img[None].float(), #texture_img[None]),
        verts_uvs=verts_uv[None],
        faces_uvs=faces_uv[None],
        sampling_mode='nearest',
    )
    textured_mesh = Meshes(
        verts=[surface_mesh.verts_list()[0]],   
        faces=[surface_mesh.faces_list()[0]],
        textures=textures_uv,
    )
    
    return textured_mesh