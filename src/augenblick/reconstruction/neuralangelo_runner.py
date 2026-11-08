"""Bridge an Augenblick COLMAP scene to an unmodified Neuralangelo checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from augenblick.core.errors import SceneError
from augenblick.eval.split import read_split


def camera_matrix(camera):
    parameters = camera.params
    model = camera.model.name
    if model == "SIMPLE_PINHOLE":
        focal_x, center_x, center_y = parameters
        focal_y = focal_x
    elif model == "PINHOLE":
        focal_x, focal_y, center_x, center_y = parameters
    else:
        raise SceneError(f"Neuralangelo requires undistorted pinhole images, got {model}")
    return np.array([[focal_x, 0, center_x], [0, focal_y, center_y], [0, 0, 1.0]])


def partition_names(names, split):
    if len(names) != len(set(names)):
        raise SceneError("Registered image stems must be unique")
    for subset in ("train", "test"):
        if len(split[subset]) != len(set(split[subset])):
            raise SceneError("Duplicate image in saved split")
    if (not split["train"] or set(split["train"]) & set(split["test"])
            or set(split["train"] + split["test"]) != set(names)):
        raise SceneError("Saved split must partition the registered images exactly")


def normalization(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise SceneError("Invalid training sparse points for normalization")
    center = np.median(points, axis=0)
    radius = float(np.linalg.norm(points - center, axis=1).max() * 1.1)
    if not np.isfinite(radius) or radius <= 0:
        raise SceneError("Degenerate training sparse cloud")
    return center, radius


def image_size(width, height, resolution):
    longest = max(width, height)
    if longest <= resolution:
        return width, height
    return max(1, width * resolution // longest), max(1, height * resolution // longest)


def training_schedule(iterations, shortened=False):
    if not shortened:
        return {}
    if iterations < 1000 or iterations > 500000:
        raise SceneError("Shortened schedules require 1000 to 500000 iterations")
    scale = iterations / 500000
    return {
        "optim": {"sched": {"warm_up_end": max(1, round(5000 * scale)),
                            "two_steps": [round(300000 * scale), round(400000 * scale)]}},
        "model": {"object": {"sdf": {"encoding": {"coarse2fine": {
            "step": max(1, round(5000 * scale))}}}}},
    }


def apply_training_schedule(config, iterations, shortened=False):
    updates = training_schedule(iterations, shortened)
    if updates:
        config.setdefault("optim", {}).setdefault("sched", {}).update(updates["optim"]["sched"])
        config.setdefault("model", {}).setdefault("object", {}).setdefault("sdf", {}).setdefault(
            "encoding", {}).setdefault("coarse2fine", {}).update(
                updates["model"]["object"]["sdf"]["encoding"]["coarse2fine"])
    return config


def prepare(args):
    import pycolmap
    import yaml

    scene = args.scene.resolve()
    output = args.output.resolve()
    target = output / "scene"
    if target.exists():
        raise SceneError(f"Refusing to overwrite prepared scene: {target}")
    if min(args.iterations, args.resolution, args.rays, args.batch_size) < 1:
        raise SceneError("Iterations, resolution, rays and batch size must be positive")
    reconstruction = pycolmap.Reconstruction(str(scene / "sparse/0"))
    images = sorted(reconstruction.images.values(), key=lambda image: image.name)
    names = [Path(image.name).stem for image in images]
    split = read_split(scene) if args.eval else {"train": names, "test": []}
    if split is None or (args.eval and not split["test"]):
        raise SceneError("Evaluation requires a nonempty saved split.json")
    partition_names(names, split)
    train_ids = {image.image_id for image in images if Path(image.name).stem in split["train"]}
    points = [point.xyz for point in reconstruction.points3D.values()
              if point.error <= 2.0 and point.track.length() >= 3
              and any(element.image_id in train_ids for element in point.track.elements)]
    if not points:
        raise SceneError("No reliable training sparse points for reconstruction bounds")
    center, radius = normalization(points)
    target.mkdir(parents=True)
    (target / "images").mkdir()
    frames = []
    has_masks = (scene / "masks").is_dir()
    image_digest = hashlib.sha256()
    for image in images:
        name = Path(image.name).stem
        camera = reconstruction.cameras[image.camera_id]
        intrinsic = camera_matrix(camera)
        with Image.open(scene / "images" / image.name) as photograph:
            rgb = photograph.convert("RGB")
        if rgb.size != (camera.width, camera.height):
            raise SceneError(f"SfM dimensions disagree with photograph: {image.name}")
        if has_masks:
            mask_path = scene / "masks" / f"{name}.png"
            with Image.open(mask_path) as silhouette:
                if silhouette.size != rgb.size:
                    raise SceneError(f"Mask dimensions disagree: {name}")
                mask = silhouette.convert("L").point(lambda value: 255 if value > 127 else 0)
            if mask.getbbox() is None:
                raise SceneError(f"Empty specimen mask: {name}")
            rgb = Image.composite(rgb, Image.new("RGB", rgb.size), mask)
        width, height = image_size(rgb.width, rgb.height, args.resolution)
        intrinsic[0] *= width / rgb.width
        intrinsic[1] *= height / rgb.height
        rgb = rgb.resize((width, height), Image.Resampling.LANCZOS)
        rgb.save(target / "images" / f"{name}.png")
        image_digest.update(name.encode())
        image_digest.update(rgb.tobytes())
        world_to_camera = np.eye(4)
        world_to_camera[:3] = image.cam_from_world().matrix()
        camera_to_world = np.linalg.inv(world_to_camera)
        camera_to_world[:, 1:3] *= -1
        frames.append({"name": name, "file_path": f"images/{name}.png",
                       "transform_matrix": camera_to_world.tolist(),
                       "intrinsic_matrix": intrinsic.tolist(), "width": width, "height": height})
    dimensions = {(frame["height"], frame["width"]) for frame in frames}
    if len(dimensions) != 1:
        raise SceneError("This backend requires consistent prepared image dimensions")
    height, width = dimensions.pop()
    train_frames = [frame for frame in frames if frame["name"] in split["train"]]
    val_frames = [frame for frame in frames if frame["name"] in split["test"]]
    metadata = {"frames": train_frames, "val_frames": val_frames,
                "sphere_center": center.tolist(), "sphere_radius": radius,
                "aabb_range": [[center[axis] - radius, center[axis] + radius] for axis in range(3)]}
    (target / "transforms.json").write_text(json.dumps(metadata, indent=2))
    (target / "split.json").write_text(json.dumps(split, indent=2))
    config = {
        "_parent_": str(args.upstream.resolve() / "projects/neuralangelo/configs/base.yaml"),
        "max_iter": args.iterations,
        "wandb_scalar_iter": min(100, args.iterations),
        "checkpoint": {"save_iter": min(20000, args.iterations)},
        "validation_iter": min(5000, args.iterations),
        "model": {"render": {"rand_rays": args.rays}, "background": {"enabled": False},
                  "appear_embed": {"enabled": False}},
        "data": {"type": "augenblick.reconstruction.neuralangelo_data", "root": str(target),
                 "num_workers": 4, "preload": True, "num_images": len(train_frames),
                 "train": {"image_size": [height, width], "batch_size": args.batch_size},
                 "val": {"image_size": [max(1, round(height * min(1, 300 / height))),
                                        max(1, round(width * min(1, 300 / height)))],
                         "batch_size": 1, "subset": min(4, len(val_frames) or len(train_frames))}},
    }
    apply_training_schedule(config, args.iterations, args.shortened_schedule)
    (output / "neuralangelo.yaml").write_text(yaml.safe_dump(config))
    manifest = {"source_scene": str(scene), "upstream_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True).strip(),
        "seed": args.seed, "iterations": args.iterations, "train_views": len(train_frames),
        "test_views": len(val_frames), "resolution": [width, height],
        "mask_policy": "binary specimen mask composited to black" if has_masks else "unmasked RGB",
        "appearance_embeddings": False, "background_model": False,
        "batch_size": args.batch_size, "rays_per_image": args.rays,
        "shortened_schedule": args.shortened_schedule,
        "schedule_overrides": training_schedule(args.iterations, args.shortened_schedule),
        "split_sha256": hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
        "prepared_rgb_sha256": image_digest.hexdigest(),
        "normalization_points": len(points), "sphere_center": center.tolist(), "sphere_radius": radius}
    (output / "neuralangelo_manifest.json").write_text(json.dumps(manifest, indent=2))


def disabled_logging(trainer, *arguments, **keywords):
    import wandb

    return wandb.init(mode="disabled")


def configure_runtime(upstream):
    sys.path.insert(0, str(upstream.resolve()))
    os.chdir(upstream)
    import imaginaire.utils.gpu_affinity as affinity
    from imaginaire.trainers.base import BaseTrainer

    affinity.set_affinity = lambda *arguments, **keywords: os.sched_getaffinity(0)
    BaseTrainer.init_wandb = disabled_logging
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"


def checkpoint(output):
    candidates = list((output / "logs").glob("epoch_*_iteration_*_checkpoint.pt"))
    if not candidates:
        raise SceneError(f"No Neuralangelo checkpoint under {output / 'logs'}")
    return max(candidates, key=lambda path: int(path.name.split("_iteration_")[1].split("_")[0]))


def invoke(args, script, arguments):
    configure_runtime(args.upstream)
    sys.argv = [str(args.upstream / script), *arguments]
    runpy.run_path(str(args.upstream / script), run_name="__main__")


def training_arguments(output, manifest, resume_from=None):
    arguments = ["--config", str(output / "neuralangelo.yaml"),
                 "--logdir", str(output / "logs"), "--seed", str(manifest["seed"]), "--show_pbar"]
    if resume_from is not None:
        resume_from = resume_from.resolve()
        if not resume_from.is_file() or resume_from.parent != (output / "logs").resolve():
            raise SceneError("Resume checkpoint must exist in this training run's logs directory")
        arguments += ["--checkpoint", str(resume_from), "--resume"]
    return arguments


def render_request(args, manifest):
    selected = getattr(args, "checkpoint", None)
    destination = getattr(args, "evaluation_output", None)
    expected_iteration = getattr(args, "iteration", None)
    if any(value is not None for value in (selected, destination, expected_iteration)):
        if any(value is None for value in (selected, destination, expected_iteration)):
            raise SceneError("Checkpoint evaluation requires --checkpoint, --iteration and --evaluation-output")
        if expected_iteration < 1 or expected_iteration > manifest["iterations"]:
            raise SceneError("Evaluation iteration must be within the original training schedule")
        selected = selected.resolve()
        destination = destination.resolve()
        if not selected.is_file():
            raise SceneError(f"Checkpoint not found: {selected}")
        if destination == args.output.resolve() or args.output.resolve() in destination.parents:
            raise SceneError("Checkpoint evaluation must write outside the training output")
        if destination.exists():
            raise SceneError(f"Refusing to overwrite checkpoint evaluation: {destination}")
        return selected, destination, expected_iteration
    return checkpoint(args.output), args.output, manifest["iterations"]


def extraction_request(args):
    selected = getattr(args, "checkpoint", None)
    destination = getattr(args, "evaluation_output", None)
    if selected is None and destination is None:
        return checkpoint(args.output), args.output
    if selected is None or destination is None:
        raise SceneError("Evaluation mesh extraction requires --checkpoint and --evaluation-output")
    selected, destination = selected.resolve(), destination.resolve()
    provenance_path = destination / "checkpoint_evaluation.json"
    if not provenance_path.is_file():
        raise SceneError("Render and validate this evaluation checkpoint before extracting its mesh")
    provenance = json.loads(provenance_path.read_text())
    if (Path(provenance["training_output"]).resolve() != args.output.resolve()
            or Path(provenance["checkpoint"]).resolve() != selected
            or not selected.is_file()
            or provenance["checkpoint_bytes"] != selected.stat().st_size):
        raise SceneError("Mesh checkpoint does not match the validated NVS evaluation")
    if (destination == args.output.resolve() or args.output.resolve() in destination.parents
            or (destination / "mesh/neuralangelo.ply").exists()):
        raise SceneError("Refusing to overwrite training or evaluation mesh outputs")
    return selected, destination


def validate_resume_state(trainer, expected_iteration, target_iterations):
    iteration = trainer.checkpointer.resume_iteration
    if iteration != expected_iteration or not 0 < iteration < target_iterations:
        raise SceneError(f"Resume checkpoint iteration {iteration} differs from {expected_iteration}")
    if not trainer.optim.state or trainer.sched.last_epoch not in (iteration - 1, iteration):
        raise SceneError("Optimizer or scheduler state was not restored")
    steps = [int(state["step"].item()) for state in trainer.optim.state.values() if "step" in state]
    if not steps or min(steps) < 1:
        raise SceneError("Checkpoint has no populated optimizer steps")
    return iteration, steps


def resume_check(args):
    manifest = json.loads((args.output / "neuralangelo_manifest.json").read_text())
    training_arguments(args.output, manifest, args.resume_from)
    configure_runtime(args.upstream)
    import torch
    from imaginaire.config import Config
    from imaginaire.trainers.utils.get_trainer import get_trainer
    from augenblick.reconstruction.neuralangelo_data import Dataset

    config = Config(str(args.output / "neuralangelo.yaml"))
    config.data.preload = False
    config.logdir = str(args.output / "logs")
    if config.max_iter != manifest["iterations"]:
        raise SceneError("Training config and manifest iteration targets disagree")
    trainer = get_trainer(config, is_inference=False, seed=manifest["seed"])
    trainer.checkpointer.load(str(args.resume_from), resume=True, load_opt=True, load_sch=True)
    iteration, steps = validate_resume_state(trainer, args.iteration, config.max_iter)
    learning_rates = [group["lr"] for group in trainer.optim.param_groups]
    dataset = Dataset(config)
    batch = torch.utils.data.default_collate([dataset[index] for index in range(config.data.train.batch_size)])
    data = trainer.start_of_iteration(batch, iteration)
    trainer.train_step(data)
    torch.cuda.synchronize()
    if not all(torch.isfinite(value).all() for value in trainer.losses.values()):
        raise SceneError("Nonfinite optimizer step after restoring checkpoint")
    print("NEURALANGELO_RESUME_CHECK_OK", json.dumps({
        "iteration": iteration, "target_iterations": config.max_iter,
        "batch_size": config.data.train.batch_size,
        "scheduler_last_epoch": trainer.sched.last_epoch,
        "optimizer_step_range_before_check": [min(steps), max(steps)],
        "learning_rates": learning_rates,
        "checkpoint_written": False,
    }), flush=True)


def render(args):
    configure_runtime(args.upstream)
    import torch
    from imaginaire.config import Config
    from imaginaire.trainers.utils.get_trainer import get_trainer
    from augenblick.reconstruction.neuralangelo_data import Dataset

    config = Config(str(args.output / "neuralangelo.yaml"))
    manifest = json.loads((args.output / "neuralangelo_manifest.json").read_text())
    selected, destination, expected_iteration = render_request(args, manifest)
    config.data.val.image_size = list(reversed(manifest["resolution"]))
    config.data.val.subset = None
    config.data.preload = False
    config.logdir = ""
    trainer = get_trainer(config, is_inference=True, seed=manifest["seed"])
    trainer.checkpointer.load(str(selected), load_opt=False, load_sch=False)
    model = trainer.model_module
    model.eval()
    iteration = trainer.checkpointer.eval_iteration
    if iteration != expected_iteration:
        raise SceneError(f"Checkpoint is at iteration {iteration}, expected {expected_iteration}")
    model.progress = iteration / config.max_iter
    model.neural_sdf.warm_up_end = config.optim.sched.warm_up_end
    model.neural_sdf.set_active_levels(iteration)
    model.neural_sdf.set_normal_epsilon()
    dataset = Dataset(config, is_inference=True)
    split = json.loads((Path(config.data.root) / "split.json").read_text())
    names = [frame["name"] for frame in dataset.list]
    if (sorted(names) != sorted(split["test"]) or len(names) != len(set(names))
            or len(names) != manifest["test_views"]):
        raise SceneError("Render dataset does not match the exact saved held-out split")
    export = destination / "test" / f"ours_{iteration}"
    for directory in ("renders", "gt"):
        (export / directory).mkdir(parents=True, exist_ok=False)
    with torch.no_grad():
        for index in range(len(dataset)):
            sample = dataset[index]
            data = {key: value.unsqueeze(0).cuda() if torch.is_tensor(value) else torch.tensor([value]).cuda()
                    for key, value in sample.items()}
            prediction = model.inference(data)["rgb_map"][0]
            name = dataset.list[index]["name"]
            for directory, tensor in (("renders", prediction), ("gt", sample["image"])):
                pixels = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
                Image.fromarray(np.round(pixels * 255).astype(np.uint8)).save(export / directory / f"{name}.png")
    if len(dataset) != manifest["test_views"]:
        raise SceneError("Exported held-out view count differs from manifest")
    if destination != args.output:
        evaluation = {"training_output": str(args.output), "checkpoint": str(selected),
                      "checkpoint_bytes": selected.stat().st_size, "iteration": iteration,
                      "training_target_iterations": manifest["iterations"],
                      "resolution": manifest["resolution"], "n_test": len(dataset),
                      "split_sha256": manifest["split_sha256"],
                      "prepared_rgb_sha256": manifest["prepared_rgb_sha256"],
                      "metric_representation": "neural field RGB, not extracted mesh",
                      "intermediate_checkpoint": iteration < manifest["iterations"]}
        (destination / "checkpoint_evaluation.json").write_text(json.dumps(evaluation, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    commands = parser.add_subparsers(dest="stage", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--scene", type=Path, required=True)
    preparation.add_argument("--eval", action="store_true")
    preparation.add_argument("--iterations", type=int, default=500000)
    preparation.add_argument("--resolution", type=int, default=1600)
    preparation.add_argument("--rays", type=int, default=512)
    preparation.add_argument("--batch-size", type=int, default=2)
    preparation.add_argument("--seed", type=int, default=22)
    preparation.add_argument("--shortened-schedule", action="store_true")
    for command in ("train", "train-worker"):
        training = commands.add_parser(command)
        training.add_argument("--resume-from", type=Path)
    continuation_check = commands.add_parser("resume-check")
    continuation_check.add_argument("--resume-from", type=Path, required=True)
    continuation_check.add_argument("--iteration", type=int, required=True)
    extraction = commands.add_parser("extract")
    extraction.add_argument("--resolution", type=int, default=1024)
    extraction.add_argument("--block-resolution", type=int, default=64)
    extraction.add_argument("--checkpoint", type=Path)
    extraction.add_argument("--evaluation-output", type=Path)
    rendering = commands.add_parser("render")
    rendering.add_argument("--checkpoint", type=Path)
    rendering.add_argument("--iteration", type=int)
    rendering.add_argument("--evaluation-output", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.upstream = args.upstream.resolve()
    started = time.perf_counter()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "resume-check":
        args.resume_from = args.resume_from.resolve()
        resume_check(args)
    elif args.stage == "train":
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   "--nproc_per_node=1", "--module", __spec__.name,
                   "--upstream", str(args.upstream), "--output", str(args.output), "train-worker"]
        if args.resume_from is not None:
            command += ["--resume-from", str(args.resume_from.resolve())]
        subprocess.run(command, check=True)
    elif args.stage == "train-worker":
        manifest = json.loads((args.output / "neuralangelo_manifest.json").read_text())
        invoke(args, "train.py", training_arguments(args.output, manifest, args.resume_from))
    elif args.stage == "extract":
        selected, destination = extraction_request(args)
        extraction_arguments = [
            "--single_gpu", "--config", str(args.output / "neuralangelo.yaml"),
            "--checkpoint", str(selected),
            "--output_file", str(destination / "mesh/neuralangelo.ply"),
            "--resolution", str(args.resolution), "--block_res", str(args.block_resolution),
            "--textured",
        ]
        invoke(args, "projects/neuralangelo/scripts/extract_mesh.py", extraction_arguments)
    elif args.stage == "render":
        render(args)
    if args.stage not in ("train-worker", "resume-check"):
        timing_root = args.evaluation_output.resolve() if args.stage in ("render", "extract") and args.evaluation_output else args.output
        timing_path = timing_root / "stage_timings.json"
        timings = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        timing_name = f"train_resume_{args.resume_from.stem}" if args.stage == "train" and args.resume_from else args.stage
        timings[timing_name] = time.perf_counter() - started
        timing_path.write_text(json.dumps(timings, indent=2))


if __name__ == "__main__":
    main()