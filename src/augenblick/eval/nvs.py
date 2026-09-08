"""Score a backend's held-out renders against the photographs, masked to the specimen.

Reads the PNG pairs a reconstruction backend wrote to <output>/test/ours_<iter>/{renders,gt}
and scores them there, so one protocol covers every backend without importing any of their
mutually incompatible renderers. Background pixels are excluded because every scene here is a
masked turntable capture, where scoring them would mostly reward reproducing the black
surround.

Usage:  python -m augenblick.eval.nvs --scene <scene> --output <model_dir>
"""
import argparse
import json
import logging
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


def find_test_dir(test_root: Path, iteration: int | None = None) -> Path:
    """Locate the held-out render directory a backend wrote under its test root."""
    if iteration is not None:
        return test_root / f"ours_{iteration}"
    candidates = [p for p in test_root.glob("ours_*") if p.is_dir()]
    if not candidates:
        raise SceneError(
            f"no held-out renders under {test_root}; "
            "the backend was run without an evaluation split")
    # Highest iteration wins, so a 7k checkpoint never shadows the final one.
    return max(candidates, key=lambda p: int(p.name.split("_")[-1]))


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
    return pairs


def score(test_dir: Path, masks_dir: Path, test_stems: list[str],
          out_path: Path, extra: dict | None = None) -> dict:
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
    pairs = pair_views(test_dir, test_stems)

    views, skipped, masked = [], [], 0
    with torch.no_grad():
        for stem, render_path, gt_path in pairs:
            pred = _load_rgb(render_path, device)
            truth = _load_rgb(gt_path, device)
            if pred.shape != truth.shape:
                raise SceneError(
                    f"{render_path.name}: render is {tuple(pred.shape)} but ground truth is "
                    f"{tuple(truth.shape)}")

            mask = _load_mask(masks_dir, stem, pred.shape[1], pred.shape[2], device)
            fill = None
            if mask is not None and float(mask.sum()) == 0.0:
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
                "psnr": metrics.psnr(pred, truth, mask),
                "ssim": metrics.ssim(pred, truth, mask),
                "lpips": lpips(pred, truth, mask),
                "mask_fill": fill,
            })
            del pred, truth, mask

    if not views:
        raise SceneError(f"no scoreable views in {test_dir}; every mask was empty")

    result = {
        **(extra or {}),
        "test_dir": str(test_dir),
        "iteration": int(test_dir.name.split("_")[-1]) if "_" in test_dir.name else None,
        "n_test": len(views),
        "n_masked": masked,
        "n_skipped": len(skipped),
        "skipped": skipped,
        # Records which convention produced this file; see the module docstring.
        "metric_domain": METRIC_DOMAIN,
        "psnr": float(np.mean([v["psnr"] for v in views])),
        "ssim": float(np.mean([v["ssim"] for v in views])),
        "lpips": float(np.mean([v["lpips"] for v in views])),
        "views": views,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(result, handle, indent=2)
    return result


def score_output(scene: Scene, output_dir: Path, iteration: int | None = None,
                 extra: dict | None = None) -> dict:
    """Score the held-out renders a backend left under output_dir against the scene."""
    test_dir = find_test_dir(output_dir / "test", iteration)
    return score(test_dir, scene.masks_dir, resolve_test_stems(scene),
                 output_dir / METRICS_FILENAME, extra)


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
    args = parser.parse_args(argv)

    try:
        result = score_output(Scene(args.scene.resolve()), args.output.resolve(), args.iteration)
    except SceneError as exc:
        logger.error(str(exc))
        return 2

    summary = {k: v for k, v in result.items() if k != "views"}
    print(json.dumps(summary, indent=2))
    print("NVS_EVAL_DONE", args.output / METRICS_FILENAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
