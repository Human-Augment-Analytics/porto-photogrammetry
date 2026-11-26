"""LOCAL PATCH (augenblick): first-party file, not upstream NeuS2.

Stage 3: marching cubes on the trained SDF's zero level, mapped back into the SfM frame,
then the same connected-component clean-up as 2DGS.
"""
import argparse
import logging
import time
from pathlib import Path

import numpy as np

import ngp_session
import scene_io
from augenblick.core.errors import SceneError

logger = logging.getLogger("neus2.extract_mesh")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("-m", "--model_path", type=Path, required=True, help="Output directory")
    parser.add_argument("--n_steps", type=int, default=30000, help="Snapshot to mesh")
    parser.add_argument("--mc_res", type=int, default=1024, help="Marching-cubes grid per axis")
    parser.add_argument("--num_cluster", type=int, default=1,
                        help="Connected components to keep; 0 keeps every component")
    args = parser.parse_args(argv)
    args.model_path = args.model_path.resolve()
    return args


def write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray, colors: np.ndarray):
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(faces.astype(np.int32)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0).astype(np.float64))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(path), mesh):
        raise SceneError(f"could not write {path}")
    return mesh


def post_process(mesh, n_cluster: int):
    """Keep the largest components and drop what they leave unreferenced."""
    import copy
    import open3d as o3d

    cleaned = copy.deepcopy(mesh)
    if n_cluster > 0 and len(cleaned.triangles):
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            ids, _, _ = cleaned.cluster_connected_triangles()
        keep = scene_io.keep_largest_clusters(np.asarray(cleaned.triangles), np.asarray(ids), n_cluster)
        cleaned.remove_triangles_by_mask(~keep)
        cleaned.remove_unreferenced_vertices()
        cleaned.remove_degenerate_triangles()
    return cleaned


def main(argv=None) -> None:
    import open3d as o3d

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    norm = scene_io.read_json(args.model_path / scene_io.NORMALIZATION)
    testbed, _ = ngp_session.open_testbed(scene_io.snapshot_path(args.model_path, args.n_steps))
    aabb = testbed.render_aabb
    logger.info(f"marching cubes at {args.mc_res}^3 over NGP box "
                f"{np.round(aabb.min, 4).tolist()}-{np.round(aabb.max, 4).tolist()}")

    start = time.monotonic()
    # thresh 0 is the SDF zero level (upstream's m_mesh.thresh); the default is a NeRF density.
    data = testbed.compute_marching_cubes_mesh(resolution=[args.mc_res] * 3, thresh=0.0)
    vertices = scene_io.ngp_to_world(data["V"], norm["centre"], norm["radius"])
    # Under from_na upstream's save_mesh writes faces in this order (marching_cubes.cu:912).
    faces = np.asarray(data["F"])
    if len(faces) == 0:
        raise SceneError("marching cubes found no surface; the SDF has no zero level in the box")
    logger.info(f"{len(vertices)} vertices, {len(faces)} faces in {time.monotonic() - start:.1f}s")

    mesh_dir = args.model_path / "mesh"
    raw = write_ply(mesh_dir / "neus2_raw.ply", vertices, faces, np.asarray(data["C"]))
    cleaned = post_process(raw, args.num_cluster)
    if not o3d.io.write_triangle_mesh(str(mesh_dir / "neus2_post.ply"), cleaned):
        raise SceneError(f"could not write {mesh_dir / 'neus2_post.ply'}")
    logger.info(f"kept {len(cleaned.triangles)} of {len(faces)} faces "
                f"(num_cluster={args.num_cluster}) -> {mesh_dir / 'neus2_post.ply'}")


if __name__ == "__main__":
    scene_io.cli(main)
