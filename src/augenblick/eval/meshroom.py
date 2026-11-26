"""Train-only Meshroom SfMData and dense graph preparation; no held-out observations."""
import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from augenblick.core.errors import SceneError
from augenblick.eval.cameras import camera_records
from augenblick.eval.split import check_split, read_split

SFM_NODES = {"CameraInit_1", "FeatureExtraction_1", "ImageMatching_1",
             "FeatureMatching_1", "StructureFromMotion_1", "Publish_1"}


def filter_sfmdata(data: dict, records: list[dict], train: set[str], images: Path) -> dict:
    """Keep shared poses/points, but expose only training views and observations to dense stages."""
    records = {record["img_name"]: record for record in records}
    views = {Path(view["path"]).stem: view for view in data["views"]}
    if len(views) != len(data["views"]) or set(views) != set(records):
        raise SceneError("SfMData must describe exactly the registered COLMAP images")
    poses = {str(pose["poseId"]): pose["pose"]["transform"] for pose in data["poses"]}
    intrinsics = {str(i["intrinsicId"]): i for i in data["intrinsics"]}
    for name, view in views.items():
        camera = records[name]
        pose = poses[str(view["poseId"])]
        if not np.allclose(pose["center"], camera["position"], rtol=0, atol=1e-8):
            raise SceneError(f"{name}: SfMData and COLMAP camera centers disagree")
        if not np.allclose(np.array(pose["rotation"]).reshape(3, 3), camera["rotation"], rtol=0, atol=1e-8):
            raise SceneError(f"{name}: SfMData and COLMAP camera rotations disagree")
        intrinsic = intrinsics[str(view["intrinsicId"])]
        if intrinsic["type"] != "pinhole" or intrinsic.get("distortionParams"):
            raise SceneError("Meshroom benchmark currently requires undistorted pinhole inputs")
        fx = float(intrinsic["focalLength"]) * int(view["width"]) / float(intrinsic["sensorWidth"])
        fy = fx * float(intrinsic.get("pixelRatio", 1))
        cx = int(view["width"]) / 2 + intrinsic["principalPoint"][0]
        cy = int(view["height"]) / 2 + intrinsic["principalPoint"][1]
        if not np.allclose([fx, fy, cx, cy], [camera[k] for k in ("fx", "fy", "cx", "cy")], rtol=1e-8, atol=1e-7):
            raise SceneError(f"{name}: SfMData and COLMAP calibration disagree")
    if not train or not train <= set(records):
        raise SceneError("Meshroom training selection contains unknown or no views")
    result = copy.deepcopy(data)
    result["views"] = [v for v in result["views"] if Path(v["path"]).stem in train]
    for view in result["views"]:
        view["path"] = str(Path(images) / Path(view["path"]).name)
    view_ids = {str(view["viewId"]) for view in result["views"]}
    pose_ids = {str(view["poseId"]) for view in result["views"]}
    intrinsic_ids = {str(view["intrinsicId"]) for view in result["views"]}
    result["poses"] = [p for p in result["poses"] if str(p["poseId"]) in pose_ids]
    result["intrinsics"] = [i for i in result["intrinsics"] if str(i["intrinsicId"]) in intrinsic_ids]
    result["structure"] = []
    for landmark in data.get("structure", []):
        observations = [o for o in landmark["observations"] if str(o["observationId"]) in view_ids]
        if len({str(o["observationId"]) for o in observations}) >= 2:
            result["structure"].append({**landmark, "observations": observations})
    # No feature/matching folders can reintroduce the all-view observation set.
    result["featuresFolders"], result["matchesFolders"] = [], []
    return result


def dense_graph(template: dict, sfmdata: Path, masks: Path) -> dict:
    graph = copy.deepcopy(template)
    for name in SFM_NODES:
        graph["graph"].pop(name, None)
        graph.get("header", {}).get("nodesVersions", {}).pop(name.rsplit("_", 1)[0], None)
    graph["graph"]["PrepareDenseScene_1"]["inputs"].update(
        input=str(sfmdata), masksFolders=[str(masks)], imagesFolders=[], evCorrection=False)
    graph["graph"]["Texturing_1"]["inputs"]["correctEV"] = False
    for node in graph["graph"].values():
        node.pop("uid", None)
        node.pop("uids", None)
        node.pop("parallelization", None)
    # This is a new graph template, not the computed all-view graph with its cache UIDs.
    graph.setdefault("header", {})["template"] = False
    expected = {
        "PrepareDenseScene_1", "DepthMap_1", "DepthMapFilter_1",
        "Meshing_1", "MeshFiltering_1", "Texturing_1"}
    if set(graph["graph"]) != expected:
        raise SceneError("unsupported Meshroom dense subgraph; unexpected/missing nodes")
    connections = {
        "DepthMap_1": {"input": "{PrepareDenseScene_1.input}", "imagesFolder": "{PrepareDenseScene_1.output}"},
        "DepthMapFilter_1": {"input": "{DepthMap_1.input}", "depthMapsFolder": "{DepthMap_1.output}"},
        "Meshing_1": {"input": "{DepthMapFilter_1.input}", "depthMapsFolder": "{DepthMapFilter_1.output}"},
        "MeshFiltering_1": {"inputMesh": "{Meshing_1.outputMesh}"},
        "Texturing_1": {"input": "{Meshing_1.output}", "imagesFolder": "{DepthMap_1.imagesFolder}",
                        "inputMesh": "{MeshFiltering_1.outputMesh}"}}
    for node, edges in connections.items():
        for key, value in edges.items():
            if graph["graph"][node]["inputs"].get(key) != value:
                raise SceneError(f"unsafe Meshroom edge: {node}.{key}")
    for key in ("inputRefMesh", "normalsFolder"):
        if graph["graph"]["Texturing_1"]["inputs"].get(key):
            raise SceneError(f"external Meshroom texturing input: {key}")
    text = json.dumps(graph["graph"])
    if any("{" + name + "." in text for name in SFM_NODES):
        raise SceneError("dense graph still references an excluded SfM/Publish node")
    return graph


def prepare(scene, output: Path, sfmdata: Path, template: Path, evaluate: bool) -> list[dict]:
    records = camera_records(scene.model_dir)
    split = read_split(scene.root) if evaluate else None
    if evaluate:
        if split is None:
            raise SceneError("Meshroom held-out training needs an explicit split.json")
        check_split(split, [r["img_name"] for r in records])
    train = set(split["train"]) if split else {r["img_name"] for r in records}
    source = json.loads(Path(sfmdata).read_text())
    filtered = filter_sfmdata(source, records, train, scene.images_dir)
    output = Path(output)
    (output / "sfm").mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(filtered) + "\n").encode()
    training_sfm = output / "sfm" / ("train_" + hashlib.sha256(payload).hexdigest()[:16] + ".sfm")
    training_sfm.write_bytes(payload)
    canonical = output / "sfm" / "train.sfm"
    temporary = canonical.with_suffix(".sfm.tmp")
    if temporary.is_symlink() or temporary.exists():
        temporary.unlink()
    temporary.symlink_to(training_sfm.name)
    temporary.replace(canonical)
    graph = dense_graph(json.loads(Path(template).read_text()), training_sfm, scene.masks_dir)
    (output / "pipeline.mg").write_text(json.dumps(graph, indent=2) + "\n")
    (output / "cameras.json").write_text(json.dumps(records, indent=2) + "\n")
    (output / "view_selection.json").write_text(json.dumps({
        "train": sorted(train), "test": sorted(split["test"]) if split else [],
        "source_sfmdata": str(sfmdata), "shared_colmap": str(scene.model_dir),
        "dense_views": len(filtered["views"]), "dense_landmarks": len(filtered["structure"]),
        "representation": "textured_mesh"}, indent=2) + "\n")
    return records


def compute(output: Path, meshroom_root: Path, meshroom_python: str,
            alicevision_root: Path, keep_cache: bool = False):
    output, meshroom_root, alicevision_root = map(Path, (output, meshroom_root, alicevision_root))
    env = dict(os.environ)
    env["MESHROOM_ROOT"] = str(meshroom_root)
    env["ALICEVISION_ROOT"] = str(alicevision_root)
    env["PYTHONPATH"] = str(meshroom_root) + os.pathsep + env.get("PYTHONPATH", "")
    for name in ("MESHROOM_NODES_PATH", "MESHROOM_PIPELINE_TEMPLATES_PATH"):
        env[name] = str(alicevision_root / "share" / "meshroom")
    env["PATH"] = str(alicevision_root / "bin") + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join([str(alicevision_root / "bin"),
        str(alicevision_root / "lib"), env.get("LD_LIBRARY_PATH", "")])
    cache = output / "MeshroomCache"
    command = [meshroom_python, str(meshroom_root / "bin" / "meshroom_compute"),
               str(output / "pipeline.mg"), "--cache", str(cache), "--toNode", "Texturing_1"]
    if output.is_symlink() or cache.is_symlink():
        raise SceneError("Meshroom output/cache must not be adopted symlinks")
    subprocess.run(command, env=env, check=True)
    # Publish only to our own output, never the source graph's Publish_1 destination.
    meshes = list((cache / "Texturing").glob("*/texturedMesh.obj"))
    if len(meshes) != 1:
        raise SceneError("expected exactly one completed textured mesh; refusing ambiguous cache")
    destination = output / "textured_mesh"
    if destination.is_symlink() or cache.is_symlink():
        raise SceneError("computed mesh/cache directories must not be adopted symlinks")
    destination.mkdir(exist_ok=True)
    for path in meshes[0].parent.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    if not keep_cache:
        shutil.rmtree(cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--meshroom-root", type=Path, required=True)
    parser.add_argument("--meshroom-python", required=True)
    parser.add_argument("--alicevision-root", type=Path, required=True)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    compute(args.output, args.meshroom_root, args.meshroom_python, args.alicevision_root, args.keep_cache)


if __name__ == "__main__":
    main()
