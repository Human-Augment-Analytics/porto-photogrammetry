"""Static masked NeuS2 reconstruction through its native CUDA Python bindings."""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

from augenblick.core.errors import SceneError
from augenblick.core.registry import register_reconstruction
from augenblick.core.scene import Scene
from augenblick.eval.split import read_split
from augenblick.reconstruction.base import LIBS_DIR, REPO_ROOT, EvalParams, Stage, SubprocessBackend


@dataclass(frozen=True)
class Neus2Config(EvalParams):
    upstream: Path = field(default=LIBS_DIR / "neus2")
    build_dir: str = field(default="", metadata={"help": "NeuS2 build directory; defaults to upstream/build"})
    iterations: int = 15000
    resolution: int = field(default=1600, metadata={"help": "Maximum input image dimension"})
    batch_size: int = field(default=262144, metadata={"help": "Target sampled points per training step, not rays"})
    mesh_resolution: int = 512
    render_spp: int = 8


@register_reconstruction
class Neus2Backend(SubprocessBackend):
    name: ClassVar[str] = "neus2"
    config_cls: ClassVar[type] = Neus2Config
    backend_dir: ClassVar[Path] = REPO_ROOT
    title: ClassVar[str] = "NeuS2 Reconstruction Pipeline"

    def validate(self, scene: Scene) -> None:
        super().validate(scene)
        if not (self.config.upstream / "configs/nerf/dtu.json").is_file():
            raise SceneError("Initialize NeuS2 with git submodule update --init --recursive src/libs/neus2")
        if not scene.masks_dir.is_dir():
            raise SceneError("The static NeuS2 backend requires foreground masks")
        if min(self.config.iterations, self.config.resolution, self.config.render_spp) < 1:
            raise SceneError("Iterations, resolution and render samples must be positive")
        if self.config.mesh_resolution < 2 or self.config.batch_size < 128 or self.config.batch_size % 128:
            raise SceneError("Mesh resolution must be >=2; batch size must be a positive multiple of 128")
        self.binding_path(self.build_dir)

    def stages(self, scene: Scene, output_dir: Path) -> list[Stage]:
        config = self.config
        common = [sys.executable, "-c", "from augenblick.reconstruction.neus2 import main; main()",
                  "--upstream", str(config.upstream), "--output", str(output_dir)]
        if config.build_dir:
            common += ["--build-dir", config.build_dir]
        preparation = common + ["prepare", "--scene", str(scene.root), "--iterations", str(config.iterations),
                                "--resolution", str(config.resolution), "--batch-size", str(config.batch_size)]
        if config.eval:
            preparation.append("--eval")
        return [Stage("Prepare posed RGBA images", preparation),
                Stage("NeuS2 CUDA training", common + ["train"]),
                Stage("Zero-level SDF mesh extraction", common + ["extract", "--resolution", str(config.mesh_resolution)]),
                *([Stage("Held-out neural rendering", common + ["render", "--spp", str(config.render_spp)])]
                  if config.eval else [])]

    def mesh_path(self, output_dir: Path) -> Path:
        return output_dir / "mesh/neus2.ply"

    @property
    def build_dir(self) -> Path:
        return Path(self.config.build_dir) if self.config.build_dir else self.config.upstream / "build"

    @staticmethod
    def camera_matrix(camera):
        import numpy as np

        if camera.model.name == "SIMPLE_PINHOLE":
            focal_x, center_x, center_y = camera.params
            focal_y = focal_x
        elif camera.model.name == "PINHOLE":
            focal_x, focal_y, center_x, center_y = camera.params
        else:
            raise SceneError(f"NeuS2 needs undistorted pinhole cameras, got {camera.model.name}")
        intrinsic = np.eye(4)
        intrinsic[0, :3] = [focal_x, 0, center_x]
        intrinsic[1, :3] = [0, focal_y, center_y]
        if not np.isfinite(intrinsic).all() or min(focal_x, focal_y) <= 0:
            raise SceneError("Invalid pinhole intrinsics")
        return intrinsic

    @staticmethod
    def partition_names(names, split):
        if len(names) != len(set(names)):
            raise SceneError("Registered image stems must be unique")
        for subset in ("train", "test"):
            if len(split[subset]) != len(set(split[subset])):
                raise SceneError("Duplicate image in saved split")
        if (not split["train"] or set(split["train"]) & set(split["test"])
                or set(split["train"] + split["test"]) != set(names)):
            raise SceneError("Saved split must partition registered images exactly")

    @staticmethod
    def image_size(width, height, resolution):
        longest = max(width, height)
        if longest <= resolution:
            return width, height
        return max(1, width * resolution // longest), max(1, height * resolution // longest)

    @staticmethod
    def normalization(points):
        import numpy as np

        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise SceneError("No finite reliable training sparse points for NeuS2 bounds")
        center = np.median(points, axis=0)
        radius = float(np.linalg.norm(points - center, axis=1).max() * 1.1)
        if not np.isfinite(radius) or radius <= 0:
            raise SceneError("Degenerate training sparse cloud")
        scale = 1 / (2 * radius)
        return center, radius, scale, np.full(3, 0.5) - center * scale

    def prepare_scene(self, scene: Scene, output_dir: Path) -> None:
        import numpy as np
        import pycolmap
        from PIL import Image

        config = self.config
        source = scene.root.resolve()
        output = output_dir.resolve()
        target = output / "scene"
        if target.exists():
            raise SceneError(f"Refusing to overwrite prepared scene: {target}")
        if min(config.iterations, config.resolution) < 1:
            raise SceneError("Iterations and resolution must be positive")
        if not (source / "masks").is_dir():
            raise SceneError("The static NeuS2 recipe requires foreground masks; NeuS2++ is not this backend")
        reconstruction = pycolmap.Reconstruction(str(source / "sparse/0"))
        images = sorted(reconstruction.images.values(), key=lambda image: image.name)
        names = [Path(image.name).stem for image in images]
        split = read_split(source) if config.eval else {"train": names, "test": []}
        if split is None or (config.eval and not split["test"]):
            raise SceneError("Evaluation requires a nonempty saved split.json")
        self.partition_names(names, split)
        train_ids = {image.image_id for image in images if Path(image.name).stem in split["train"]}
        points = [point.xyz for point in reconstruction.points3D.values()
                  if point.error <= 2 and point.track.length() >= 3
                  and any(element.image_id in train_ids for element in point.track.elements)]
        center, radius, scale, offset = self.normalization(points)
        (target / "images").mkdir(parents=True)
        frames = []
        digest = hashlib.sha256()
        for image in images:
            name = Path(image.name).stem
            camera = reconstruction.cameras[image.camera_id]
            intrinsic = self.camera_matrix(camera)
            with Image.open(source / "images" / image.name) as photograph:
                rgb = photograph.convert("RGB")
            mask_path = source / "masks" / f"{name}.png"
            if not mask_path.is_file():
                raise SceneError(f"Missing foreground mask: {mask_path}")
            with Image.open(mask_path) as silhouette:
                if rgb.size != (camera.width, camera.height) or silhouette.size != rgb.size:
                    raise SceneError(f"Image/mask dimensions disagree with SfM: {image.name}")
                mask = silhouette.convert("L").point(lambda value: 255 if value > 127 else 0)
            if mask.getbbox() is None:
                raise SceneError(f"Empty foreground mask: {name}")
            width, height = self.image_size(rgb.width, rgb.height, config.resolution)
            intrinsic[0] *= width / rgb.width
            intrinsic[1] *= height / rgb.height
            rgb = Image.composite(rgb, Image.new("RGB", rgb.size), mask)
            rgb = rgb.resize((width, height), Image.Resampling.LANCZOS)
            mask = mask.resize((width, height), Image.Resampling.NEAREST)
            rgb = Image.composite(rgb, Image.new("RGB", rgb.size), mask)
            rgb.putalpha(mask)
            rgb.save(target / "images" / f"{name}.png")
            digest.update(name.encode())
            digest.update(rgb.tobytes())
            world_to_camera = np.eye(4)
            world_to_camera[:3] = image.cam_from_world().matrix()
            frames.append({"name": name, "file_path": f"images/{name}.png", "w": width, "h": height,
                           "intrinsic_matrix": intrinsic.tolist(),
                           "transform_matrix": np.linalg.inv(world_to_camera).tolist()})
        dimensions = {(frame["w"], frame["h"]) for frame in frames}
        if len(dimensions) != 1:
            raise SceneError("NeuS2 requires consistent prepared image dimensions")
        width, height = dimensions.pop()
        shared = {"from_na": True, "w": width, "h": height, "aabb_scale": 1,
                  "scale": scale, "offset": offset.tolist()}
        for subset in ("train", "test"):
            selected = [frame for frame in frames if frame["name"] in split[subset]]
            (target / f"transforms_{subset}.json").write_text(json.dumps({**shared, "frames": selected}, indent=2))
        (target / "split.json").write_text(json.dumps(split, indent=2))
        network = json.loads((config.upstream / "configs/nerf/dtu.json").read_text())
        network["hyperparams"]["first_frame_max_training_step"] = config.iterations
        network["hyperparams"]["predict_global_movement"] = False
        (output / "neus2.json").write_text(json.dumps(network, indent=2))
        manifest = {"source_scene": str(source), "upstream_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.upstream, text=True).strip(),
            "iterations": config.iterations, "batch_size": config.batch_size, "resolution": [width, height],
            "train_views": len(split["train"]), "test_views": len(split["test"]),
            "normalization_points": len(points), "sphere_center": center.tolist(), "sphere_radius": radius,
            "scale": scale, "offset": offset.tolist(), "camera_convention": "COLMAP/OpenCV c2w, from_na=true",
            "mask_policy": "binary foreground in PNG alpha, black outside mask",
            "split_sha256": hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
            "prepared_rgba_sha256": digest.hexdigest(), "metric_representation": "neural field RGB, not mesh"}
        (output / "neus2_manifest.json").write_text(json.dumps(manifest, indent=2))

    @staticmethod
    def binding_path(build_dir):
        candidates = sorted(build_dir.glob("pyngp*.so"))
        if len(candidates) != 1:
            raise SceneError(f"Expected one NeuS2 pyngp extension in {build_dir}; run scripts/setup_neus2.sh")
        return candidates[0].resolve()

    @classmethod
    def load_bindings(cls, build_dir):
        path = cls.binding_path(build_dir)
        if "pyngp" in sys.modules:
            if Path(sys.modules["pyngp"].__file__).resolve() != path:
                raise SceneError("Another backend's pyngp extension is already loaded")
            return sys.modules["pyngp"]
        spec = importlib.util.spec_from_file_location("pyngp", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["pyngp"] = module
        return module

    @staticmethod
    def snapshot_path(output, manifest):
        selected = output / "checkpoints" / f"{manifest['iterations']}.msgpack"
        if not selected.is_file():
            raise SceneError(f"NeuS2 snapshot not found: {selected}")
        return selected

    def load_testbed(self, output: Path, subset: str, restore=False):
        import numpy as np

        manifest = json.loads((output / "neus2_manifest.json").read_text())
        ngp = self.load_bindings(self.build_dir)
        testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
        testbed.shall_train = False
        testbed.load_training_data(str(output / "scene" / f"transforms_{subset}.json"))
        if restore:
            testbed.load_snapshot(str(self.snapshot_path(output, manifest)))
            if testbed.training_step != manifest["iterations"]:
                raise SceneError("NeuS2 snapshot iteration does not match the manifest")
        else:
            testbed.reload_network_from_file(str(output / "neus2.json"))
        dataset = testbed.nerf.training.dataset
        expected_views = manifest["train_views" if subset == "train" else "test_views"]
        if dataset.n_images != expected_views or not dataset.from_na:
            raise SceneError("NeuS2 loaded the wrong view partition or camera convention")
        if not np.isclose(dataset.scale, manifest["scale"]) or not np.allclose(dataset.offset, manifest["offset"]):
            raise SceneError("NeuS2 normalization disagrees with the manifest")
        testbed.background_color = [0.0, 0.0, 0.0, 1.0]
        testbed.nerf.training.random_bg_color = False
        for setting in ("optimize_extrinsics", "optimize_focal_length", "optimize_distortion", "optimize_exposure"):
            setattr(testbed.nerf.training, setting, False)
        return testbed, manifest

    @staticmethod
    def run_training(testbed, iterations, batch_size):
        import numpy as np

        testbed.shall_train = True
        while testbed.training_step < iterations:
            previous = testbed.training_step
            testbed.train(batch_size)
            if testbed.training_step != previous + 1 or not testbed.shall_train:
                raise SceneError("NeuS2 training stalled or aborted with no valid samples")
            if not np.isfinite([testbed.loss, testbed.ek_loss, testbed.mask_loss]).all():
                raise SceneError("Nonfinite NeuS2 training loss")
            if testbed.training_step % 100 == 0 or testbed.training_step == iterations:
                print(f"NeuS2 step {testbed.training_step}/{iterations}, loss={testbed.loss:.6g}", flush=True)
        testbed.shall_train = False

    def train(self, output: Path) -> None:
        manifest = json.loads((output / "neus2_manifest.json").read_text())
        selected = output / "checkpoints" / f"{manifest['iterations']}.msgpack"
        if selected.exists():
            raise SceneError(f"Refusing to overwrite NeuS2 snapshot: {selected}")
        testbed, manifest = self.load_testbed(output, "train")
        self.run_training(testbed, manifest["iterations"], manifest["batch_size"])
        selected.parent.mkdir(exist_ok=True)
        testbed.save_snapshot(str(selected), False)
        if not selected.is_file() or selected.stat().st_size == 0:
            raise SceneError("NeuS2 failed to save a nonempty snapshot")

    def extract(self, output: Path) -> None:
        import numpy as np
        from plyfile import PlyData

        destination = self.mesh_path(output)
        if destination.exists():
            raise SceneError(f"Refusing to overwrite NeuS2 mesh: {destination}")
        testbed, manifest = self.load_testbed(output, "train", restore=True)
        destination.parent.mkdir(exist_ok=True)
        testbed.compute_and_save_marching_cubes_mesh(str(destination), [self.config.mesh_resolution] * 3, thresh=0.0)
        mesh = PlyData.read(destination)
        if len(mesh["vertex"]) == 0 or len(mesh["face"]) == 0:
            raise SceneError("NeuS2 extraction produced empty geometry")
        if not all(np.isfinite(mesh["vertex"][axis]).all() for axis in ("x", "y", "z")):
            raise SceneError("NeuS2 extraction produced nonfinite vertices")

    @staticmethod
    def srgb_from_linear(pixels):
        import numpy as np

        pixels = np.asarray(pixels, dtype=np.float64)
        if not np.isfinite(pixels).all():
            raise SceneError("Nonfinite NeuS2 render pixels")
        pixels = np.maximum(pixels, 0)
        return np.clip(np.where(pixels <= 0.0031308, pixels * 12.92,
                                1.055 * np.power(pixels, 1 / 2.4) - 0.055), 0, 1)

    def render(self, output: Path) -> None:
        import numpy as np
        from PIL import Image

        manifest = json.loads((output / "neus2_manifest.json").read_text())
        metadata = json.loads((output / "scene/transforms_test.json").read_text())
        split = json.loads((output / "scene/split.json").read_text())
        names = [frame["name"] for frame in metadata["frames"]]
        if (not names or sorted(names) != sorted(split["test"]) or len(set(names)) != len(names)
                or len(names) != manifest["test_views"]):
            raise SceneError("NeuS2 render cameras do not match the exact held-out split")
        export = output / "test" / f"ours_{manifest['iterations']}"
        if export.exists():
            raise SceneError(f"Refusing to overwrite NeuS2 renders: {export}")
        testbed, manifest = self.load_testbed(output, "test", restore=True)
        testbed.shall_train = False
        testbed.snap_to_pixel_centers = True
        testbed.nerf.rendering_min_transmittance = 1e-4
        testbed.nerf.render_with_camera_distortion = False
        for directory in ("renders", "gt"):
            (export / directory).mkdir(parents=True)
        for index, frame in enumerate(metadata["frames"]):
            testbed.reset_camera()
            testbed.set_camera_to_training_view(index)
            prediction = np.asarray(testbed.render(frame["w"], frame["h"], self.config.render_spp, True))
            if prediction.shape != (frame["h"], frame["w"], 4):
                raise SceneError(f"Unexpected NeuS2 render dimensions: {frame['name']}")
            pixels = np.round(self.srgb_from_linear(prediction[:, :, :3]) * 255).astype(np.uint8)
            Image.fromarray(pixels).save(export / "renders" / f"{frame['name']}.png")
            with Image.open(output / "scene" / frame["file_path"]) as reference:
                reference.convert("RGB").save(export / "gt" / f"{frame['name']}.png")
        if testbed.training_step != manifest["iterations"]:
            raise SceneError("Training state changed during held-out rendering")
        provenance = {"snapshot": str(self.snapshot_path(output, manifest)),
                      "iteration": manifest["iterations"], "n_test": len(names), "names": names,
                      "spp": self.config.render_spp, "split_sha256": manifest["split_sha256"],
                      "metric_representation": "neural field RGB, not extracted mesh"}
        (output / "render_provenance.json").write_text(json.dumps(provenance, indent=2))

    def execute_stage(self, stage: str, output: Path, scene: Scene | None = None) -> None:
        output = output.resolve()
        previous_cwd = Path.cwd()
        started = time.perf_counter()
        try:
            if stage == "prepare":
                self.prepare_scene(scene, output)
            else:
                os.chdir(self.config.upstream)
                {"train": self.train, "extract": self.extract, "render": self.render}[stage](output)
        finally:
            os.chdir(previous_cwd)
        timing_path = output / "stage_timings.json"
        timings = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        timings[stage] = time.perf_counter() - started
        timing_path.write_text(json.dumps(timings, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    commands = parser.add_subparsers(dest="stage", required=True)
    preparation = commands.add_parser("prepare")
    preparation.add_argument("--scene", type=Path, required=True)
    preparation.add_argument("--eval", action="store_true")
    preparation.add_argument("--iterations", type=int, default=15000)
    preparation.add_argument("--resolution", type=int, default=1600)
    preparation.add_argument("--batch-size", type=int, default=262144)
    commands.add_parser("train")
    extraction = commands.add_parser("extract")
    extraction.add_argument("--resolution", dest="mesh_resolution", type=int, default=512)
    rendering = commands.add_parser("render")
    rendering.add_argument("--spp", dest="render_spp", type=int, default=8)
    args = parser.parse_args()
    config = Neus2Config(
        upstream=args.upstream.resolve(),
        build_dir=str(args.build_dir.resolve()) if args.build_dir else "",
        eval=getattr(args, "eval", False), iterations=getattr(args, "iterations", 15000),
        resolution=getattr(args, "resolution", 1600), batch_size=getattr(args, "batch_size", 262144),
        mesh_resolution=getattr(args, "mesh_resolution", 512), render_spp=getattr(args, "render_spp", 8))
    scene = Scene(args.scene.resolve()) if args.stage == "prepare" else None
    Neus2Backend(config).execute_stage(args.stage, args.output, scene)