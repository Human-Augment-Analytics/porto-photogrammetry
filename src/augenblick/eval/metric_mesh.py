"""Validate reconstruction lineage and export a mesh in millimetres."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

from augenblick.core.errors import SceneError
from augenblick.eval.scale import apply_scale

logger = logging.getLogger(__name__)

# Maximum camera-centre difference in model units.
LINEAGE_TOLERANCE = 1e-9
SCALE_KEY = "scale_mm_per_model_unit"


def image_token(name: str) -> str | None:
    """Match image names by frame token, ignoring camera-body prefixes."""
    found = re.search(r"IMG_\d+", name)
    return found.group(0) if found else None


def model_digests(model_dir: Path) -> dict[str, str] | None:
    """Per-file digests of a COLMAP model, the shape the scale estimator records."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return None
    found = {}
    for entry in sorted(model_dir.iterdir()):
        if entry.is_file() and entry.suffix in (".bin", ".txt"):
            h = hashlib.sha256()
            with entry.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    h.update(block)
            found[entry.name] = h.hexdigest()
    return found or None


def camera_centres_agree(cameras_json: Path, model_dir: Path) -> tuple[bool, str]:
    """Whether a splatting run and a sparse model are the same reconstruction."""
    import numpy as np
    import pycolmap

    try:
        reconstruction = pycolmap.Reconstruction(str(model_dir))
    except Exception as error:                                       # noqa: BLE001
        return False, f"cannot read model: {error}"
    model = {image_token(image.name): np.asarray(image.projection_center())
             for image in reconstruction.images.values() if image_token(image.name)}

    try:
        listed = json.loads(Path(cameras_json).read_text())
    except (OSError, ValueError) as error:
        return False, f"cannot read cameras.json: {error}"
    splat = {image_token(camera.get("img_name", "")): np.asarray(camera["position"])
             for camera in listed if image_token(camera.get("img_name", ""))}

    shared = sorted(set(model) & set(splat))
    if not shared:
        return False, f"no camera names in common ({len(model)} model, {len(splat)} splat)"
    worst = max(float(np.linalg.norm(model[k] - splat[k])) for k in shared)
    return worst <= LINEAGE_TOLERANCE, f"{len(shared)} cameras, worst {worst:.3e}"


def published_factor(scale_json: Path) -> float | None:
    """The scale this tree publishes, or None when the estimate was rejected."""
    try:
        record = json.loads(Path(scale_json).read_text())
    except (OSError, ValueError):
        return None
    value = record.get(SCALE_KEY)
    return float(value) if isinstance(value, (int, float)) else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="augenblick.eval.metric_mesh",
                                     description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True,
                        help="the splatting run's cameras.json, for the lineage check")
    parser.add_argument("--model", type=Path, required=True,
                        help="the sparse model the scale was measured in")
    parser.add_argument("--scale", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    agree, detail = camera_centres_agree(args.cameras, args.model)
    logger.info(f"lineage {'ok' if agree else 'MISMATCH'} ({detail})")
    if not agree:
        logger.error("the mesh was not reconstructed in the model the scale was "
                     "measured in")
        return 1

    recorded = json.loads(args.scale.read_text()).get("model_sha256")
    if recorded != model_digests(args.model):
        logger.error("the published scale was measured in a different model than the "
                     "one present now")
        return 1

    factor = published_factor(args.scale)
    if factor is None:
        logger.error(f"{args.scale} carries no usable scale")
        return 1

    if args.out.exists():
        args.out.unlink()
    try:
        apply_scale(args.mesh, args.out, factor)
    except (OSError, ValueError) as error:
        raise SceneError(f"cannot scale {args.mesh}: {error}") from error
    logger.info(f"scaled by {factor:.6f} mm per model unit -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
