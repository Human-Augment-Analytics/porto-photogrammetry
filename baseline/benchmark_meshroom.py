#!/usr/bin/env python3
"""
Run the Meshroom (AliceVision) photogrammetry baseline via meshroom_batch.

Input: a directory of source images, or with --from_sfm a COLMAP scene whose cameras
and tracks replace Meshroom's own SfM (only PrepareDenseScene onward runs).
Output: Meshroom reconstruction artifacts in the output directory and an
optional save file capturing the pipeline graph.

Usage:
    python baseline/benchmark_meshroom.py <input_images> <output_dir> \
        [--pipeline {photogrammetry,turntable}] [--save_file <path>] \
        [--meshroom_root <path>]
    python baseline/benchmark_meshroom.py <colmap_scene> <output_dir> --from_sfm \
        [--sparse sparse/0] [--save_file <path>] [--meshroom_root <path>]
"""
import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# The Texturing node's declared outputs (see AliceVision share/meshroom/aliceVision/
# Texturing.py): the mesh, its material, and the texture atlases. Everything else in
# MeshroomCache is intermediate.
TEXTURED_MESH_GLOBS = ("texturedMesh.*", "texture_*")

PIPELINES = {
    "photogrammetry": "photogrammetry",
    "turntable": "photogrammetryObjectTurntable",
}

SEGMENTATION_NODES = ("ImageDetectionPrompt_1", "ImageSegmentationBox_1")

# What --from_sfm replaces with a converted COLMAP model.
SFM_NODES = ("CameraInit_1", "FeatureExtraction_1", "ImageMatching_1", "FeatureMatching_1",
             "StructureFromMotion_1")

# Native-mode values for the flags of nodes --from_sfm skips; None means "not given".
SFM_NODE_DEFAULTS = {
    "describer_preset": "normal",
    "describer_quality": "normal",
    "describer_types": "dspsift",
    "max_threads": 16,
}

RUN_GRAPH = Path(__file__).resolve().parent / "meshroom_run_graph.py"


def run(cmd, cwd=None, env=None):
    """Run a command, streaming output to the terminal."""
    logger.info(f"Running: {' '.join(cmd)}")
    logger.info(f"  cwd: {cwd or os.getcwd()}")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        logger.error(f"Command failed with return code {result.returncode}")
        sys.exit(result.returncode)


def _dir_size(path):
    """Total bytes under path, ignoring anything that vanishes mid-walk."""
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def prune_cache(output_dir):
    """Keep the textured mesh, delete the rest of MeshroomCache.

    A 663-image scene leaves ~50 GB in MeshroomCache, almost all of it
    PrepareDenseScene EXRs and DepthMap tiles, against a 300 GB quota. The mesh
    itself is a few hundred MB. This moves the Texturing outputs to
    <output_dir>/textured_mesh/ and removes the cache.

    Only called after meshroom_batch exits 0: on failure the cache is the only way
    to resume, so a failed run keeps everything.
    """
    cache = Path(output_dir) / "MeshroomCache"
    if not cache.is_dir():
        logger.warning(f"No MeshroomCache at {cache}; nothing to prune")
        return

    texturing = cache / "Texturing"
    # Texturing writes into a per-uid subfolder; there is normally exactly one.
    mesh_files = []
    for glob in TEXTURED_MESH_GLOBS:
        mesh_files.extend(texturing.glob(f"*/{glob}"))
    mesh_files = [f for f in mesh_files if f.is_file()]

    if not mesh_files:
        # Refuse to delete a cache we cannot replace the mesh from. This is the
        # safety net for a pipeline that reported success without a usable mesh.
        logger.error(f"No textured mesh found under {texturing}; keeping cache intact")
        return

    dest = Path(output_dir) / "textured_mesh"
    dest.mkdir(parents=True, exist_ok=True)
    for f in mesh_files:
        shutil.copy2(f, dest / f.name)
    logger.info(f"Kept {len(mesh_files)} mesh file(s) -> {dest}")

    freed = _dir_size(cache)
    shutil.rmtree(cache)
    logger.info(f"Pruned MeshroomCache, freed {freed / 2**30:.1f} GiB")


def find_template(name):
    """Locate <name>.mg on MESHROOM_PIPELINE_TEMPLATES_PATH, where meshroom_batch looks."""
    for folder in os.environ.get("MESHROOM_PIPELINE_TEMPLATES_PATH", "").split(os.pathsep):
        path = Path(folder) / f"{name}.mg"
        if folder and path.is_file():
            return path
    logger.error(f"{name}.mg not found on MESHROOM_PIPELINE_TEMPLATES_PATH")
    sys.exit(1)


def write_turntable_graph(dest):
    """Write the turntable template to dest without its segmentation nodes.

    FeatureExtraction reads CameraInit directly and both mask inputs are left at
    their defaults, for the masks/ overrides in main() to fill. What remains
    turntable-specific is FeatureMatching.minRequired2DMotion, which drops matches
    on the static background.
    """
    with open(find_template(PIPELINES["turntable"])) as f:
        template = json.load(f)
    graph = template["graph"]
    for name in SEGMENTATION_NODES:
        del graph[name]
        template["header"]["nodesVersions"].pop(name.rsplit("_", 1)[0], None)

    graph["FeatureExtraction_1"]["inputs"]["input"] = "{CameraInit_1.output}"
    graph["FeatureExtraction_1"]["inputs"].pop("masksFolder")
    graph["PrepareDenseScene_1"]["inputs"].pop("masksFolders")

    dangling = [n for n in SEGMENTATION_NODES if f"{{{n}." in json.dumps(graph)]
    if dangling:
        raise RuntimeError(f"turntable template still references {dangling}")

    with open(dest, "w") as f:
        json.dump(template, f, indent=4)
    return dest


def write_dense_graph(dest, sfm_path):
    """Write the photogrammetry template to dest with its SfM half removed.

    PrepareDenseScene reads sfm_path directly; DepthMap onward is untouched. The
    turntable template's dense nodes are linked identically, so it is not offered.
    """
    with open(find_template(PIPELINES["photogrammetry"])) as f:
        template = json.load(f)
    graph = template["graph"]
    for name in SFM_NODES:
        del graph[name]
        template["header"]["nodesVersions"].pop(name.rsplit("_", 1)[0], None)

    graph["PrepareDenseScene_1"]["inputs"]["input"] = str(sfm_path)

    dangling = [n for n in SFM_NODES if f"{{{n}." in json.dumps(graph)]
    if dangling:
        raise RuntimeError(f"photogrammetry template still references {dangling}")

    with open(dest, "w") as f:
        json.dump(template, f, indent=4)
    return dest


def convert_sfm(scene_dir, output_dir, sparse):
    """COLMAP scene -> <output_dir>/sfm/colmap.sfm, kept as the record of the cameras used."""
    from colmap_to_sfmdata import ConversionError, convert

    sfm_path = output_dir / "sfm" / "colmap.sfm"
    t0 = time.time()
    try:
        stats = convert(scene_dir, sfm_path, sparse)
    except ConversionError as e:
        logger.error(f"Cannot use {scene_dir / sparse}: {e}")
        sys.exit(1)
    logger.info(f"Converted COLMAP model in {time.time() - t0:.1f}s")
    return sfm_path, stats


def main():
    parser = argparse.ArgumentParser(
        description="Run the Meshroom batch photogrammetry pipeline.",
    )
    parser.add_argument("input_dir", type=Path, help="Directory of dataset")
    parser.add_argument("output_dir", type=Path, help="Meshroom output directory")
    parser.add_argument("--save_file", type=Path, default=None, help="Path to save the pipeline graph (.mg)")
    parser.add_argument(
        "--meshroom_root",
        type=Path,
        default=None,
        help="Path to MESHROOM_ROOT (defaults to $MESHROOM_ROOT env var)",
    )
    parser.add_argument(
        "--pipeline",
        choices=sorted(PIPELINES),
        default="photogrammetry",
        help="AliceVision template: 'photogrammetry' (standard) or 'turntable' "
             "(photogrammetryObjectTurntable, which filters static-background matches)",
    )

    parser.add_argument(
        "--from_sfm",
        action="store_true",
        help="input_dir is a COLMAP scene (images/, sparse/0/, optional masks/): convert its "
             "model to SfMData and run only PrepareDenseScene onward",
    )
    parser.add_argument("--sparse", default="sparse/0", help="COLMAP model under input_dir (--from_sfm)")

    parser.add_argument("--describer_preset", default=None, help="FeatureExtraction.describerPreset (default: normal)")
    parser.add_argument("--describer_quality", default=None, help="FeatureExtraction.describerQuality (default: normal)")
    parser.add_argument("--describer_types", default=None, help="FeatureExtraction.describerTypes (default: dspsift)")
    parser.add_argument("--depthmap_downscale", type=int, default=2, help="DepthMap.downscale")
    parser.add_argument("--max_input_points", type=int, default=50_000_000, help="Meshing.maxInputPoints")
    parser.add_argument("--max_points", type=int, default=5_000_000, help="Meshing.maxPoints")
    parser.add_argument(
        "--estimate_space_min_observations",
        type=int,
        default=3,
        help="Meshing.estimateSpaceMinObservations",
    )
    parser.add_argument("--smoothing_iterations", type=int, default=5, help="MeshFiltering.smoothingIterations")
    parser.add_argument(
        "--filter_large_triangles_factor",
        type=float,
        default=60.0,
        help="MeshFiltering.filterLargeTrianglesFactor",
    )
    parser.add_argument(
        "--filter_triangles_ratio",
        type=float,
        default=0.0,
        help="MeshFiltering.filterTrianglesRatio",
    )
    parser.add_argument("--texture_side", type=int, default=8192, help="Texturing.textureSide")
    parser.add_argument("--texturing_downscale", type=int, default=2, help="Texturing.downscale")
    parser.add_argument("--max_threads", type=int, default=None, help="FeatureExtraction:maxThreads (default: 16)")
    parser.add_argument(
        "--keep_cache",
        action="store_true",
        help="Keep MeshroomCache after a successful run (default: delete it, keeping only "
             "the textured mesh). The cache is ~50 GB for a 663-image scene.",
    )

    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    input_images = os.path.join(input_dir, "images")
    mask_images = os.path.join(input_dir, "masks")

    if args.from_sfm:
        if args.pipeline == "turntable":
            parser.error("--pipeline turntable only changes the SfM half, which --from_sfm skips")
        given = [f"--{k}" for k in SFM_NODE_DEFAULTS if getattr(args, k) is not None]
        if given:
            logger.warning(f"Ignoring {', '.join(given)}: --from_sfm skips the nodes they configure")
    else:
        for k, v in SFM_NODE_DEFAULTS.items():
            if getattr(args, k) is None:
                setattr(args, k, v)

    meshroom_root = args.meshroom_root or os.environ.get("MESHROOM_ROOT")
    if not meshroom_root:
        logger.error("MESHROOM_ROOT is not set. Pass --meshroom_root or export MESHROOM_ROOT.")
        sys.exit(1)
    meshroom_batch = Path(meshroom_root) / "bin" / "meshroom_batch"
    if not meshroom_batch.exists():
        logger.error(f"meshroom_batch not found at {meshroom_batch}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_sfm:
        sfm_path, stats = convert_sfm(input_dir, output_dir, args.sparse)
        pipeline = str(write_dense_graph(output_dir / "dense_template.mg", sfm_path))
    elif args.pipeline == "turntable":
        pipeline = str(write_turntable_graph(output_dir / "turntable_template.mg"))
    else:
        pipeline = PIPELINES[args.pipeline]

    logger.info("=" * 60)
    logger.info("Meshroom Baseline Pipeline")
    logger.info(f"  Input:    {input_images}")
    logger.info(f"  Mask dir: {mask_images} (will be ignored if not present)")
    if args.from_sfm:
        logger.info(f"  SfM:      {input_dir / args.sparse} -> {sfm_path} "
                    f"({stats.views} views, {stats.landmarks} landmarks)")
    logger.info(f"  Output:   {output_dir}")
    logger.info(f"  Pipeline: {pipeline}")
    logger.info("=" * 60)

    dense_overrides = [
        f"DepthMap:downscale={args.depthmap_downscale}",
        f"Meshing:maxInputPoints={args.max_input_points}",
        f"Meshing:maxPoints={args.max_points}",
        f"Meshing:estimateSpaceMinObservations={args.estimate_space_min_observations}",
        f"MeshFiltering:smoothingIterations={args.smoothing_iterations}",
        f"MeshFiltering:filterLargeTrianglesFactor={args.filter_large_triangles_factor}",
        f"MeshFiltering:filterTrianglesRatio={args.filter_triangles_ratio}",
        f"Texturing:textureSide={args.texture_side}",
        f"Texturing:downscale={args.texturing_downscale}",
        f"Texturing:colorMapping.colorMappingFileType=png",
    ]
    has_masks = os.path.exists(mask_images)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(meshroom_root), env.get("PYTHONPATH", "")]))

    if args.from_sfm:
        # No EXIF reaches the converted views, so there is no exposure to correct.
        param_overrides = dense_overrides + ["Texturing:correctEV=False"]
        if has_masks:
            param_overrides.append(f"PrepareDenseScene:masksFolders={[mask_images]!r}")
        # The cache goes beside the project, so keep it in output_dir for prune_cache.
        project = output_dir / "pipeline.mg"
        cmd = [
            sys.executable, str(RUN_GRAPH),
            pipeline, str(output_dir), str(project),
            "--paramOverrides", *param_overrides
        ]
    else:
        param_overrides = [
            f"FeatureExtraction:maxThreads={args.max_threads}",
            f"FeatureExtraction:describerPreset={args.describer_preset}",
            f"FeatureExtraction:describerQuality={args.describer_quality}",
            f"FeatureExtraction:describerTypes={args.describer_types}",
            *dense_overrides,
        ]
        if has_masks:
            param_overrides += [
                f"FeatureExtraction:masksFolder={mask_images}",
                f"PrepareDenseScene:masksFolders={[mask_images]!r}",
            ]

        cmd = [
            sys.executable, str(meshroom_batch),
            "-i", str(input_images),
            "-o", str(output_dir),
            "-p", pipeline,
            "--paramOverrides", *param_overrides
        ]
        if args.save_file is not None:
            cmd += ["-s", str(args.save_file.resolve())]

    t0 = time.time()
    # run() exits non-zero on failure, so reaching the next line means success and
    # the cache is safe to prune.
    run(cmd, env=env)
    elapsed = time.time() - t0

    if args.from_sfm and args.save_file is not None and args.save_file.resolve() != project:
        shutil.copy2(project, args.save_file)

    if args.keep_cache:
        logger.info("Keeping MeshroomCache (--keep_cache)")
    else:
        prune_cache(output_dir)

    logger.info("=" * 60)
    logger.info("Pipeline complete")
    logger.info(f"  Total time: {elapsed:.1f}s")
    logger.info(f"  Output:     {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
