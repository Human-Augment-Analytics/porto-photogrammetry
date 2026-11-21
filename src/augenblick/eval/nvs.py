"""Score a backend's held-out renders against the photographs, masked to the specimen.

Reads the PNG pairs a reconstruction backend wrote to <output>/test/ours_<iter>/{renders,gt}
and scores them there, so one protocol covers every backend without importing any of their
mutually incompatible renderers. Background pixels are excluded because every scene here is a
masked turntable capture, where scoring them would mostly reward reproducing the black
surround.

Usage:  python -m augenblick.eval.nvs --scene <scene> --output <model_dir>
"""
import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from augenblick.core.errors import SceneError
from augenblick.core.scene import Scene
from augenblick.eval import metrics
from augenblick.eval.split import build_split, read_split, registered_stems, sparse_dir

logger = logging.getLogger(__name__)

RENDERS_SUBDIR = "renders"
GT_SUBDIR = "gt"
METRICS_FILENAME = "nvs_metrics.json"
METRIC_DOMAIN = "mask"


def render_iteration(name: str) -> int | None:
    """Iteration of an ours_<iter>[suffix] render dir; GeoSVR appends e.g. _r3.9."""
    match = re.match(r"ours_(\d+)", name)
    return int(match.group(1)) if match else None


def find_test_dir(test_root: Path, iteration: int | None = None,
                  render_variant: str | None = None) -> Path:
    """Select one render variant; never guess between same-iteration outputs."""
    if render_variant is not None:
        if Path(render_variant).name != render_variant or render_variant in (".", ".."):
            raise SceneError("render variant must name one direct subdirectory")
        selected = test_root / render_variant
        selected_iteration = render_iteration(render_variant)
        if selected_iteration is None or not selected.is_dir():
            raise SceneError(f"no supported render variant {render_variant} under {test_root}")
        if iteration is not None and selected_iteration != iteration:
            raise SceneError("render variant does not match the requested iteration")
        return selected
    if (test_root / "mesh").is_dir():
        if iteration is not None or any(test_root.glob("ours_*")):
            raise SceneError("ambiguous Gaussian checkpoints and textured-mesh renders")
        return test_root / "mesh"
    candidates = [(p, render_iteration(p.name)) for p in test_root.glob("ours_*") if p.is_dir()]
    candidates = [(p, i) for p, i in candidates if i is not None]
    if iteration is not None:
        candidates = [(p, i) for p, i in candidates if i == iteration]
    if not candidates:
        raise SceneError(f"no held-out renders under {test_root} for iteration {iteration}; "
                         "the backend may have been run without an evaluation split")
    wanted = iteration if iteration is not None else max(i for _, i in candidates)
    selected = [p for p, i in candidates if i == wanted]
    if len(selected) != 1:
        raise SceneError("ambiguous render variants at iteration " + str(wanted) + ": " +
                         ", ".join(sorted(p.name for p in selected)) + "; select --render-variant explicitly")
    return selected[0]



def resolve_test_stems(scene: Scene) -> list[str]:
    """Return the held-out image stems, from split.json when present, else from the model."""
    split = read_split(scene.root)
    if split is not None:
        # The readers sort their cameras by image name, and positional renders (00000.png) are
        # indexed against that order, so a hand-authored split.json in file order would
        # mislabel every view and mask it with the wrong silhouette.
        return sorted(split["test"])
    # Models trained before split.json existed fell back to the llffhold rule, which
    # build_split reproduces, so those runs stay scoreable.
    logger.info(f"No split.json in {scene.root}; deriving the held-out set from the model")
    return build_split(registered_stems(sparse_dir(scene.root)))["test"]


def _load_rgb(path: Path, device: str) -> torch.Tensor:
    """Read a PNG as a [3, H, W] float tensor in [0, 1]."""
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).to(device)


def _load_mask(masks_dir: Path, stem: str, height: int, width: int, device: str):
    """Return the view's object mask as [1, H, W] in {0, 1}, or None when absent."""
    path = masks_dir / f"{stem}.png"
    if not path.is_file():
        return None
    mask = Image.open(path).convert("L").resize((width, height), Image.NEAREST)
    binary = (np.asarray(mask) > 127).astype(np.float32)
    return torch.from_numpy(binary).to(device).unsqueeze(0)


def pair_views(test_dir: Path, test_stems: list[str]) -> list[tuple[str, Path, Path]]:
    """Match every render to its ground truth and to the photograph it was rendered from.

    2DGS and Gaussian Wrapping name their exports by position (00000.png) while PGSR names
    them by image name, so numeric names are resolved through the ordered held-out list.

    Args:
        test_dir: Directory holding the backend's renders/ and gt/ subdirectories.
        test_stems: Held-out image stems, ordered as the backend ordered its cameras.

    Returns:
        One (stem, render_path, gt_path) triple per held-out view.

    Raises:
        SceneError: If the directory holds no renders, or a render has no matching ground
            truth, or a positional name falls outside the held-out list.
    """
    renders = sorted((test_dir / RENDERS_SUBDIR).glob("*.png"))
    if not renders:
        raise SceneError(f"no renders in {test_dir / RENDERS_SUBDIR}")
    truths = {path.stem: path for path in (test_dir / GT_SUBDIR).glob("*.png")}

    pairs = []
    for render in renders:
        truth = truths.get(render.stem)
        if truth is None:
            raise SceneError(f"render {render.name} has no ground truth in {test_dir / GT_SUBDIR}")
        if render.stem.isdigit():
            position = int(render.stem)
            if position >= len(test_stems):
                raise SceneError(
                    f"render {render.name} is view {position} but the split holds only "
                    f"{len(test_stems)} views; the scene and the model disagree")
            stem = test_stems[position]
        else:
            stem = render.stem
        pairs.append((stem, render, truth))
    matched = [stem for stem, _, _ in pairs]
    if len(matched) != len(set(matched)) or set(matched) != set(test_stems):
        raise SceneError("held-out renders must cover the split exactly once")
    if set(truths) != {render.stem for _, render, _ in pairs}:
        raise SceneError("ground-truth exports and held-out renders disagree")
    return pairs


def score(test_dir: Path, masks_dir: Path, test_stems: list[str],
          out_path: Path, extra: dict | None = None, *,
          require_masks: bool = False, require_complete: bool = False,
          canonical_images: Path | None = None, resolution: int = -1,
          require_native_raster: bool = False) -> dict:
    """Score every held-out render, masked to the specimen, and write the metrics as JSON.

    Args:
        test_dir: Directory holding the backend's renders/ and gt/ subdirectories.
        masks_dir: Scene masks, used to restrict scoring to the specimen.
        test_stems: Held-out image stems, ordered as the backend ordered its cameras.
        out_path: File to write the metrics JSON to.
        extra: Additional key/value pairs recorded alongside the metrics, such as the backend.

    Returns:
        The metrics dictionary, including a per-view breakdown.

    Raises:
        SceneError: If a render and its ground truth disagree on resolution.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips = metrics.Lpips()
    dists = metrics.Dists()
    pairs = pair_views(test_dir, test_stems)

    originals = {}
    if canonical_images is not None:
        for path in Path(canonical_images).iterdir():
            if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff") and ".mask." not in path.name:
                if path.stem in originals: raise SceneError("ambiguous original photograph stem")
                originals[path.stem] = path
    views, skipped, masked = [], [], 0
    with torch.no_grad():
        for stem, render_path, gt_path in pairs:
            native_size = None
            source_hash = canonical_hash = None
            resized_render = False
            if canonical_images is not None:
                from augenblick.eval.mesh_render import image_size
                if stem not in originals: raise SceneError(f"{stem}: original photograph missing")
                source = originals[stem]
                source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
                with Image.open(source) as photo:
                    size = image_size(dict(width=photo.width, height=photo.height), resolution)
                    truth_image = photo.convert("RGB").resize(size, Image.Resampling.BICUBIC)
                canonical_hash = hashlib.sha256(truth_image.tobytes()).hexdigest()
                with Image.open(render_path) as render:
                    native_size = list(render.size)
                    resized_render = render.size != size
                    if resized_render and require_native_raster:
                        raise SceneError(f"{stem}: native render {render.size} must match canonical raster {size}")
                    pred_image = render.convert("RGB").resize(size, Image.Resampling.BICUBIC) if resized_render else render.convert("RGB")
                pred = torch.from_numpy(np.asarray(pred_image, dtype=np.float32).copy()/255).permute(2,0,1).to(device)
                truth = torch.from_numpy(np.asarray(truth_image, dtype=np.float32).copy()/255).permute(2,0,1).to(device)
            else:
                pred = _load_rgb(render_path, device)
                truth = _load_rgb(gt_path, device)
            if pred.shape != truth.shape:
                raise SceneError(
                    f"{render_path.name}: render is {tuple(pred.shape)} but ground truth is "
                    f"{tuple(truth.shape)}")

            mask = _load_mask(masks_dir, stem, pred.shape[1], pred.shape[2], device)
            fill = None
            if mask is None and require_masks:
                raise SceneError(f"{stem}: required scoring mask is missing")
            if mask is not None and float(mask.sum()) == 0.0:
                if require_complete:
                    raise SceneError(f"{stem}: empty scoring mask prevents complete evaluation")
                # No domain to score over. The masked reductions would divide by a clamped
                # denominator and report a perfect match, silently lifting the mean.
                logger.warning(f"{stem}: mask is empty, skipping this view")
                skipped.append(stem)
                del pred, truth, mask
                continue
            if mask is not None:
                # Composite both to a common background so windows straddling the silhouette
                # see the same thing, then hand the mask on to restrict each metric's domain.
                pred, truth = pred * mask, truth * mask
                fill = float(mask.mean())
                masked += 1

            views.append({
                "image": stem,
                "render_sha256": hashlib.sha256(render_path.read_bytes()).hexdigest(),
                "native_gt_sha256": hashlib.sha256(gt_path.read_bytes()).hexdigest(),
                "mask_sha256": hashlib.sha256((masks_dir / (stem + '.png')).read_bytes()).hexdigest() if (masks_dir / (stem + '.png')).is_file() else None,
                "canonical_photo_sha256": source_hash,
                "canonical_rgb_sha256": canonical_hash,
                "native_render_size": native_size,
                "scored_size": [int(pred.shape[2]), int(pred.shape[1])],
                "render_resampled": resized_render,
                "psnr": metrics.psnr(pred, truth, mask),
                "ssim": metrics.ssim(pred, truth, mask),
                "lpips": lpips(pred, truth, mask),
                "dists": dists(pred, truth, mask),
                "mask_fill": fill,
            })
            del pred, truth, mask

    if not views:
        raise SceneError(f"no scoreable views in {test_dir}; every mask was empty")

    result = {
        **(extra or {}),
        "test_dir": str(test_dir),
        "iteration": render_iteration(test_dir.name),
        "n_test": len(views),
        "n_masked": masked,
        "n_skipped": len(skipped),
        "skipped": skipped,
        # Records which convention produced this file; see the module docstring.
        "metric_domain": METRIC_DOMAIN,
        "scorer_source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), Path(metrics.__file__))},
        "evaluation_protocol": "canonical_raw_photo_v1" if canonical_images is not None else "native_exports",
        "resize_filter": "PIL BICUBIC",
        "render_resampling_requires_publication_sensitivity_check": canonical_images is not None and any(v["render_resampled"] for v in views),
        "psnr": float(np.mean([v["psnr"] for v in views])),
        "ssim": float(np.mean([v["ssim"] for v in views])),
        "lpips": float(np.mean([v["lpips"] for v in views])),
        "dists": float(np.mean([v["dists"] for v in views])),
        "views": views,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(result, handle, indent=2)
    return result


def score_output(scene: Scene, output_dir: Path, iteration: int | None = None,
                 extra: dict | None = None, *, require_masks: bool = False,
                 require_complete: bool = False, canonical_images: bool = False,
                 resolution: int = -1, require_native_raster: bool = False,
                 render_variant: str | None = None) -> dict:
    """Score the held-out renders a backend left under output_dir against the scene."""
    test_dir = find_test_dir(output_dir / "test", iteration, render_variant)
    ordered = resolve_test_stems(scene)
    manifests = list((output_dir / "native_selections").glob("render_*.json"))
    if not manifests:
        manifests = list((scene.root / "native_selections").glob("render_*.json"))
    if manifests:
        if len(manifests) != 1: raise SceneError("ambiguous native render selection manifests")
        manifest = json.loads(manifests[0].read_text())
        consumed = [r["name"] for r in manifest["test"]]
        if len(consumed) != len(set(consumed)) or set(consumed) != set(ordered):
            raise SceneError("native render camera partition disagrees with scene split")
        ordered = consumed
        extra = {**(extra or {}), "native_selection_manifest": str(manifests[0]),
                 "native_selection_sha256": hashlib.sha256(manifests[0].read_bytes()).hexdigest()}
    elif require_complete and test_dir.name != "mesh":
        raise SceneError("complete workflow scoring requires an actual native render selection manifest")
    return score(test_dir, scene.masks_dir, ordered,
                 output_dir / METRICS_FILENAME, extra,
                 require_masks=require_masks, require_complete=require_complete,
                 canonical_images=scene.images_dir if canonical_images else None, resolution=resolution,
                 require_native_raster=require_native_raster)


def main(argv: list[str] | None = None) -> int:
    """Score an already-trained model's held-out renders from the command line.

    Args:
        argv: Argument list, defaulting to sys.argv[1:].

    Returns:
        A process exit code: 2 when the scene or the renders are not where they should be.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(
        prog="augenblick.eval.nvs",
        description="Masked novel-view metrics for any backend's held-out renders.")
    parser.add_argument("--scene", type=Path, required=True, help="Scene the model was trained on")
    parser.add_argument("--output", type=Path, required=True, help="Model directory to score")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Checkpoint to score; defaults to the highest present")
    parser.add_argument("--render-variant", help="Exact ours_<iteration>[_suffix] directory name")
    parser.add_argument("--canonical-images", action="store_true")
    parser.add_argument("--require-native-raster", action="store_true")
    parser.add_argument("--canonical-resolution", type=int, default=-1)
    parser.add_argument("--require-masks", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = score_output(Scene(args.scene.resolve()), args.output.resolve(), args.iteration,
                              require_masks=args.require_masks, require_complete=args.require_complete,
                              canonical_images=args.canonical_images, resolution=args.canonical_resolution,
                              require_native_raster=args.require_native_raster, render_variant=args.render_variant)
    except SceneError as exc:
        logger.error(str(exc))
        return 2

    summary = {k: v for k, v in result.items() if k != "views"}
    print(json.dumps(summary, indent=2))
    print("NVS_EVAL_DONE", args.output / METRICS_FILENAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
