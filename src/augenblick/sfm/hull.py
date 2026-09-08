"""Visual-hull space carving: a silhouette-derived point cloud to initialise 2DGS from."""
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np

from augenblick.core.errors import SceneError
from augenblick.core.registry import register_sfm
from augenblick.core.scene import Scene
from augenblick.eval.split import read_split
from augenblick.sfm.base import SceneRefiner, SfMResult

logger = logging.getLogger(__name__)

_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")


@dataclass(frozen=True)
class HullConfig:
    """Space-carving grid, voting, and init-cloud sampling parameters."""

    res: int = field(default=256, metadata={"help": "Voxel grid resolution per axis"})
    tau: float = field(default=0.92, metadata={
        "help": "Fraction of seeing cameras that must vote a voxel inside the silhouette"})
    min_seen_frac: float = field(default=0.5, metadata={
        "help": "Voxel must be inside the frustum of at least this fraction of cameras"})
    mask_downscale: int = field(default=4, metadata={"help": "Mask downsample factor for carving"})
    img_downscale: int = field(default=4, metadata={"help": "Image downsample factor for colouring"})
    pad: float = field(default=0.08, metadata={"help": "Bounding-box padding fraction"})
    bbox_percentile: float = field(default=0.5, metadata={
        "help": "Percentile trimmed off each end of the SfM cloud when sizing the grid"})
    n_points: int = field(default=300_000, metadata={"help": "Points sampled from the hull surface"})
    save_mesh: bool = field(default=True, metadata={"help": "Also write visual_hull.ply"})
    use_split: bool = field(default=True, metadata={
        "help": "Carve and colour from training views only when the scene has a split.json"})
    mask_pool: Literal["max", "decimate"] = field(default="decimate", metadata={
        "help": "Mask downsampling rule; max is conservative for an outer bound but measurably "
                "inflates the hull, costing more accuracy than the thin structures it saves"})
    soft_surface: bool = field(default=False, metadata={
        "help": "Mesh the continuous vote margin instead of the binarised volume; sub-voxel in "
                "principle, but the margin field is not smooth and it scores slightly worse"})
    component_frac: float = field(default=0.005, metadata={
        "help": "Keep mesh components at least this fraction of the largest; 0 keeps only the largest"})
    occlusion_aware: bool = field(default=True, metadata={
        "help": "Z-buffer visibility and a mask test before a camera may colour a point"})
    auto_bounds: bool = field(default=True, metadata={
        "help": "Grow the grid and re-carve when the hull reaches the grid boundary"})
    max_expansions: int = field(default=3, metadata={"help": "Most times auto_bounds may grow the grid"})
    refine_bounds: bool = field(default=True, metadata={
        "help": "Re-carve on a grid tightened onto the hull, raising effective resolution"})
    write_normals: bool = field(default=True, metadata={
        "help": "Write hull surface normals into points3D.ply instead of zeros"})


def _cameras_from_model(rec) -> list[dict]:
    """Flatten a pycolmap reconstruction into per-image pose and intrinsic dictionaries."""
    cams = []
    for image in rec.images.values():
        camera = rec.cameras[image.camera_id]
        # Method in pycolmap 4.1, property in 4.2.
        cam_from_world = image.cam_from_world
        if callable(cam_from_world):
            cam_from_world = cam_from_world()
        world_to_cam = cam_from_world.matrix()
        params = dict(zip(camera.params_info.split(", "), camera.params))
        fx = params.get("fx", params.get("f"))
        fy = params.get("fy", fx)
        cams.append(dict(
            name=image.name,
            R=np.asarray(world_to_cam[:, :3], np.float64),
            t=np.asarray(world_to_cam[:, 3], np.float64),
            fx=float(fx), fy=float(fy),
            cx=float(params.get("cx", camera.width / 2)),
            cy=float(params.get("cy", camera.height / 2)),
            W=int(camera.width), H=int(camera.height)))
    return cams


def _project(points, cam):
    """Project world points into one camera, returning pixel coordinates and depth."""
    x_cam = points @ cam["R_t"].T + cam["t_t"]
    z = x_cam[:, 2]
    safe_z = z + 1e-9
    u = cam["fx"] * x_cam[:, 0] / safe_z + cam["cx"]
    v = cam["fy"] * x_cam[:, 1] / safe_z + cam["cy"]
    return u, v, z


def _pixel_lookup(u, v, z, width, height, scale):
    """Bounds-test in continuous pixel coordinates, then floor to a safe integer index.

    Testing after an integer cast lets a point at u/scale in (-1, 0) pass as column 0,
    because casting truncates toward zero rather than flooring.
    """
    uf, vf = u / scale, v / scale
    in_view = (z > 0) & (uf >= 0) & (uf < width) & (vf >= 0) & (vf < height)
    ui = uf.floor().long().clamp_(0, width - 1)
    vi = vf.floor().long().clamp_(0, height - 1)
    return ui, vi, in_view


@register_sfm
class VisualHullInit(SceneRefiner):
    """Carves a visual hull from masks and poses and writes it as the 2DGS init cloud."""

    name: ClassVar[str] = "hull"
    config_cls: ClassVar[type] = HullConfig

    def validate(self, scene: Scene) -> None:
        """Require images/, a sparse/0 model to take poses from, and masks to carve with."""
        super().validate(scene)
        if not scene.has_masks():
            raise SceneError(f"hull carving needs masks/, none at {scene.masks_dir}")

    def run(self, scene: Scene, output_dir: Path) -> SfMResult:
        """Carve the hull, colour a point cloud from it, and emit a ready-to-train scene.

        Args:
            scene: Input scene with images/, masks/, and a COLMAP model in sparse/0.
            output_dir: Directory to write the hull-initialised scene into.

        Returns:
            An SfMResult whose num_points is the size of the hull init cloud.

        Raises:
            SceneError: If no mask matches an image, or the carve collapses to an empty volume.
        """
        import pycolmap
        import torch
        from PIL import Image

        # Some source photographs exceed PIL's decompression-bomb guard.
        Image.MAX_IMAGE_PIXELS = None
        self.validate(scene)
        t0 = time.time()
        out_dir = output_dir.resolve()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        rec = pycolmap.Reconstruction(str(scene.sparse_dir))
        cams = _cameras_from_model(rec)
        n_registered = len(cams)
        cams, held_out = self._apply_split(scene, cams)
        logger.info(f"[hull] {len(cams)}/{n_registered} cameras after split filter "
                    f"({held_out} held out), device={device}")

        cams, masks = self._load_masks(scene, cams, device)
        if not cams:
            raise SceneError(f"no mask matched an image name under {scene.masks_dir}")

        margin, lo, hi = self._carve_bounded(rec, cams, masks, device)
        hull = self._to_mesh(margin, lo, hi)
        logger.info(f"[hull] mesh {len(hull.vertices)} verts, {len(hull.triangles)} faces")

        points, normals, colours, coloured = self._sample_and_colour(
            hull, scene, cams, masks, device)

        self._write_scene(scene, out_dir, points, normals, colours, hull)
        elapsed = time.time() - t0
        logger.info(f"HULL_DONE {len(points)}pts ({coloured} coloured) "
                    f"in {elapsed:.0f}s -> {out_dir}")

        return SfMResult(
            output_dir=out_dir,
            elapsed=elapsed,
            scene=Scene(out_dir),
            num_images=len(cams),
            num_points=len(points),
            details={"hull_faces": len(hull.triangles), "coloured_points": coloured,
                     "carve_cameras": len(cams), "held_out": held_out},
        )

    def _apply_split(self, scene, cams):
        """Drop held-out cameras so the initialisation cannot see the evaluation views."""
        if not self.config.use_split:
            return cams, 0
        split = read_split(scene.root)
        if split is None:
            logger.info("[hull] no split.json; carving from every registered camera")
            return cams, 0
        train = set(split["train"])
        kept = [c for c in cams if Path(c["name"]).stem.split(".")[0] in train]
        if not kept:
            raise SceneError(
                f"split.json at {scene.root} matched none of the registered image names")
        return kept, len(cams) - len(kept)

    def _downsample_mask(self, mask):
        """Reduce a boolean mask; max pooling keeps thin structures a stride can skip over."""
        import torch

        step = self.config.mask_downscale
        if step <= 1:
            return mask
        if self.config.mask_pool == "decimate":
            return mask[::step, ::step]
        # A hull is an outer bound, so the conservative reduction is "inside if any pixel is".
        pooled = torch.nn.functional.max_pool2d(
            mask[None, None].float(), kernel_size=step, stride=step, ceil_mode=True)
        return pooled[0, 0] > 0.5

    def _load_masks(self, scene, cams, device):
        """Load each camera's mask as a downsampled boolean tensor, dropping cameras without one."""
        import torch
        from PIL import Image

        kept, masks = [], []
        for cam in cams:
            path = scene.masks_dir / f"{Path(cam['name']).stem}.png"
            if not path.exists():
                continue
            mask = torch.from_numpy(np.asarray(Image.open(path).convert("L")) > 127).to(device)
            masks.append(self._downsample_mask(mask))
            cam["R_t"] = torch.tensor(cam["R"], device=device, dtype=torch.float64)
            cam["t_t"] = torch.tensor(cam["t"], device=device, dtype=torch.float64)
            kept.append(cam)
        logger.info(f"[hull] matched {len(kept)}/{len(cams)} masks, "
                    f"pooled by {self.config.mask_pool}")
        return kept, masks

    def _grid_bounds(self, rec):
        """Size the voxel grid from the SfM cloud, trimming outliers so floaters cannot inflate it."""
        xyz = np.array([p.xyz for p in rec.points3D.values()], np.float64)
        if len(xyz) == 0:
            raise SceneError("SfM model has no 3D points to size the hull grid from")
        q = self.config.bbox_percentile
        lo = np.percentile(xyz, q, axis=0)
        hi = np.percentile(xyz, 100 - q, axis=0)
        centre, radius = (lo + hi) / 2, (hi - lo) / 2 * (1 + self.config.pad)
        return centre - radius, centre + radius

    def _carve_bounded(self, rec, cams, masks, device):
        """Carve, first growing the grid off the hull, then tightening it onto the hull.

        The SfM cloud sizes the first grid, which is both a clipping risk (an extremity
        feature matching missed can fall outside it) and a resolution cost (the box is
        sized for the sparse cloud, not the object). Growing fixes the first, and one
        re-carve on the hull's own extent fixes the second.
        """
        lo, hi = self._grid_bounds(rec)
        attempts = self.config.max_expansions if self.config.auto_bounds else 0
        for attempt in range(attempts + 1):
            self._log_grid("carving", lo, hi)
            margin = self._carve(cams, masks, lo, hi, device)
            if attempt == attempts or not self._touches_boundary(margin):
                break
            centre, half = (lo + hi) / 2, (hi - lo) / 2 * 1.35
            lo, hi = centre - half, centre + half
            logger.info(f"[hull] hull reached the grid boundary; expanding and re-carving "
                        f"({attempt + 1}/{attempts})")

        if not self.config.refine_bounds:
            return margin, lo, hi
        tight = self._occupied_bounds(margin, lo, hi)
        if tight is None:
            return margin, lo, hi
        gain = float(np.prod(hi - lo) / max(np.prod(tight[1] - tight[0]), 1e-12)) ** (1 / 3)
        if gain < 1.05:
            logger.info(f"[hull] grid already tight (gain {gain:.2f}x); no refinement carve")
            return margin, lo, hi
        lo, hi = tight
        logger.info(f"[hull] tightening onto the hull: {gain:.2f}x finer voxels, re-carving")
        self._log_grid("refined", lo, hi)
        return self._carve(cams, masks, lo, hi, device), lo, hi

    def _log_grid(self, label, lo, hi):
        """Report the grid extent and the voxel size it implies."""
        voxel = (hi - lo) / (self.config.res - 1)
        logger.info(f"[hull] {label} bbox lo={lo.round(4)} hi={hi.round(4)} "
                    f"voxel={voxel.max():.5f}")

    def _occupied_bounds(self, margin, lo, hi):
        """Bounding box of the carved volume, padded, in world units."""
        occupied = margin >= 0
        if not occupied.any():
            return None
        spacing = (hi - lo) / (self.config.res - 1)
        idx = [np.nonzero(occupied.any(axis=tuple(j for j in range(3) if j != i)))[0]
               for i in range(3)]
        tight_lo = lo + np.array([i[0] for i in idx]) * spacing
        tight_hi = lo + np.array([i[-1] for i in idx]) * spacing
        centre, half = (tight_lo + tight_hi) / 2, (tight_hi - tight_lo) / 2 * (1 + self.config.pad)
        # A degenerate axis would collapse the grid, so never shrink below one voxel.
        half = np.maximum(half, spacing)
        return centre - half, centre + half

    @staticmethod
    def _touches_boundary(margin) -> bool:
        """Whether any occupied voxel lies on the outer shell, meaning the grid clipped the hull."""
        occupied = margin >= 0
        if not occupied.any():
            return False
        shell = [occupied[0], occupied[-1], occupied[:, 0], occupied[:, -1],
                 occupied[:, :, 0], occupied[:, :, -1]]
        return any(bool(face.any()) for face in shell)

    def _carve(self, cams, masks, lo, hi, device):
        """Intersect the silhouette cones by vote, returning a continuous margin field.

        The field is ratio-minus-tau rather than a 0/1 occupancy, so marching cubes can
        interpolate the crossing instead of always landing on a voxel-edge midpoint.
        """
        import torch

        n = self.config.res
        axes = [torch.linspace(float(lo[i]), float(hi[i]), n, device=device) for i in range(3)]
        grid = torch.meshgrid(*axes, indexing="ij")
        points = torch.stack([g.reshape(-1) for g in grid], 1).double()
        votes = torch.zeros(points.shape[0], dtype=torch.int16, device=device)
        seen = torch.zeros(points.shape[0], dtype=torch.int16, device=device)

        for cam, mask in zip(cams, masks):
            mh, mw = mask.shape
            u, v, z = _project(points, cam)
            ui, vi, in_view = _pixel_lookup(u, v, z, mw, mh, cam["W"] / mw)
            seen += in_view.to(torch.int16)
            sel = in_view.nonzero(as_tuple=True)[0]
            inside = torch.zeros_like(in_view)
            inside[sel] = mask[vi[sel], ui[sel]]
            votes += inside.to(torch.int16)

        ratio = votes.float() / seen.float().clamp(min=1)
        enough = seen.float() >= self.config.min_seen_frac * len(cams)
        margin = ratio - self.config.tau
        suppressed = int(((~enough) & (margin >= 0)).sum())
        margin = torch.where(enough, margin, torch.full_like(margin, -1.0))

        count = int((margin >= 0).sum())
        logger.info(f"[hull] occupied voxels {count}/{points.shape[0]}"
                    + (f"; {suppressed} suppressed by min_seen_frac" if suppressed else ""))
        if count < 8:
            raise SceneError(
                f"hull carve collapsed to {count} voxels; --tau {self.config.tau} is likely "
                f"too strict, or the masks do not correspond to these poses")
        return margin.reshape(n, n, n).cpu().numpy().astype(np.float32)

    def _to_mesh(self, margin, lo, hi):
        """Marching-cubes the carved volume and drop components too small to be real."""
        import open3d as o3d
        from skimage.measure import marching_cubes

        spacing = (hi - lo) / (self.config.res - 1)
        if self.config.soft_surface:
            volume, level = margin, 0.0
        else:
            volume, level = (margin >= 0).astype(np.float32), 0.5
        verts, faces, _, _ = marching_cubes(volume, level=level, spacing=tuple(spacing))
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts + lo),
            o3d.utility.Vector3iVector(faces))
        mesh.compute_vertex_normals()

        labels, counts, _ = mesh.cluster_connected_triangles()
        counts = np.asarray(counts)
        if len(counts) <= 1:
            return mesh
        if self.config.component_frac <= 0:
            # Exactly one, not every component tied for largest.
            keep = np.zeros(len(counts), bool)
            keep[counts.argmax()] = True
        else:
            keep = counts >= max(1, int(self.config.component_frac * counts.max()))
        labels = np.asarray(labels)
        mesh.remove_triangles_by_mask(~keep[labels])
        mesh.remove_unreferenced_vertices()
        logger.info(f"[hull] kept {int(keep.sum())}/{len(counts)} mesh components "
                    f"(largest {counts.max()} faces, threshold "
                    f"{max(1, int(self.config.component_frac * counts.max())) if self.config.component_frac > 0 else counts.max()})")
        return mesh

    def _visible_from(self, cam, mask, pt, device):
        """Z-buffer the sampled points in one camera, so occluded points cannot take its colour."""
        import torch

        mh, mw = mask.shape
        u, v, z = _project(pt, cam)
        ui, vi, in_view = _pixel_lookup(u, v, z, mw, mh, cam["W"] / mw)
        visible = in_view.clone()
        if not self.config.occlusion_aware:
            return visible
        # Foreground test at the sampled pixel: with tau < 1 a point may sit outside a silhouette.
        visible &= mask[vi, ui]
        flat = (vi * mw + ui).long()
        depth = z.float()
        zbuf = torch.full((mh * mw,), float("inf"), device=device, dtype=torch.float32)
        sel = visible.nonzero(as_tuple=True)[0]
        zbuf.scatter_reduce_(0, flat[sel], depth[sel], reduce="amin", include_self=True)
        # 2% slack absorbs the depth spread of several surface samples inside one pixel.
        return visible & (depth <= zbuf[flat] * 1.02)

    def _sample_and_colour(self, hull, scene, cams, masks, device):
        """Sample the hull surface and colour each point from the cameras that can see it."""
        import torch
        from PIL import Image

        pcd = hull.sample_points_uniformly(
            number_of_points=self.config.n_points, use_triangle_normal=True)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        pt = torch.from_numpy(points).to(device).double()
        nt = torch.from_numpy(normals).to(device).double()
        accum = torch.zeros(len(points), 3, device=device)
        weights = torch.zeros(len(points), device=device)
        step = self.config.img_downscale

        for cam, mask in zip(cams, masks):
            path = scene.images_dir / cam["name"]
            if not path.exists():
                continue
            image = np.asarray(Image.open(path).convert("RGB"))[::step, ::step]
            ih, iw, _ = image.shape
            tex = torch.from_numpy(image).to(device).float() / 255.0
            u, v, z = _project(pt, cam)
            ui, vi, _ = _pixel_lookup(u, v, z, iw, ih, cam["W"] / iw)
            visible = self._visible_from(cam, mask, pt, device)
            centre = -(cam["R_t"].T @ cam["t_t"])
            view = centre - pt
            view = view / (view.norm(dim=1, keepdim=True) + 1e-9)
            facing = (nt * view).sum(1).clamp(min=0.0) * visible.double()
            idx = (facing > 1e-3).nonzero(as_tuple=True)[0]
            accum[idx] += tex[vi[idx], ui[idx]] * facing[idx, None].float()
            weights[idx] += facing[idx].float()

        rgb = (accum / weights.clamp(min=1e-6)[:, None]).clamp(0, 1)
        rgb[weights < 1e-6] = 0.5
        coloured = int((weights > 1e-6).sum())
        return (points.astype(np.float32), normals.astype(np.float32),
                (rgb.cpu().numpy() * 255).astype(np.uint8), coloured)

    def _write_scene(self, scene, out_dir, points, normals, colours, hull):
        """Link the media, copy the model and split unchanged, and write the hull init cloud."""
        import open3d as o3d
        from plyfile import PlyData, PlyElement

        sparse_out = out_dir / "sparse" / "0"
        sparse_out.mkdir(parents=True, exist_ok=True)
        for name, source in (("images", scene.images_dir), ("masks", scene.masks_dir)):
            link = out_dir / name
            if source.is_dir() and not link.exists():
                os.symlink(source, link)
        for name in _MODEL_FILES:
            src = scene.sparse_dir / name
            if src.exists():
                shutil.copy2(src, sparse_out / name)
        # The reconstruction stage must hold out exactly the views the carve was denied.
        split_src = scene.root / "split.json"
        if split_src.is_file():
            shutil.copy2(split_src, out_dir / "split.json")

        if not self.config.write_normals:
            normals = np.zeros_like(points)
        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"),
                 ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                 ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        elements = np.empty(len(points), dtype=dtype)
        elements[:] = list(map(tuple, np.concatenate([points, normals, colours], 1)))
        PlyData([PlyElement.describe(elements, "vertex")]).write(str(sparse_out / "points3D.ply"))

        if self.config.save_mesh:
            o3d.io.write_triangle_mesh(str(out_dir / "visual_hull.ply"), hull)
