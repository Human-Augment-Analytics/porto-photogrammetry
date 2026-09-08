"""The held-out view split, written once so every backend evaluates the same images."""
import json
import logging
import shutil
from pathlib import Path

from augenblick.core.errors import SceneError

logger = logging.getLogger(__name__)

SPLIT_FILENAME = "split.json"
DEFAULT_HOLDOUT = 8


def sparse_dir(scene_root: Path) -> Path:
    """Locate the COLMAP model; a function for callers holding only a path."""
    # Imported here so the module stays importable from core code without a cycle.
    from augenblick.core.scene import Scene

    return Scene(scene_root).model_dir


def registered_stems(model_dir: Path) -> list[str]:
    """Return the registered image stems in the order the backends' readers sort them."""
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model_dir))
    # The readers key on basename.split(".")[0] and sort by it; match that exactly.
    return sorted(image.name.split(".")[0] for image in reconstruction.images.values())


def build_split(stems: list[str], holdout: int = DEFAULT_HOLDOUT) -> dict[str, list[str]]:
    """Hold out every nth stem, reproducing the llffhold rule all four backends share."""
    return {
        "train": [s for i, s in enumerate(stems) if i % holdout != 0],
        "test": [s for i, s in enumerate(stems) if i % holdout == 0],
    }


def read_split(scene_root: Path) -> dict[str, list[str]] | None:
    """Return the scene's split.json, or None when it has not been written."""
    path = scene_root / SPLIT_FILENAME
    if not path.is_file():
        return None
    with path.open() as handle:
        return json.load(handle)


def write_split(scene_root: Path, holdout: int = DEFAULT_HOLDOUT) -> dict[str, list[str]]:
    """Derive the held-out split from the scene's COLMAP model and persist it.

    An existing split.json is reused rather than regenerated, so a hand-authored split
    survives and every backend pointed at the scene trains on the same images.

    Args:
        scene_root: Scene directory holding images/ and the COLMAP model.
        holdout: Hold out one view in every `holdout`, matching the backends' llffhold.

    Returns:
        The split, as {"train": [stem, ...], "test": [stem, ...]}.

    Raises:
        SceneError: If the scene carries no COLMAP model to derive a split from.
    """
    existing = read_split(scene_root)
    if existing is not None:
        logger.info(f"Reusing {scene_root / SPLIT_FILENAME}: "
                    f"{len(existing['train'])} train, {len(existing['test'])} test")
        return existing

    model_dir = sparse_dir(scene_root)
    if not model_dir.is_dir() or not any(model_dir.iterdir()):
        raise SceneError(f"no SfM model at {model_dir}; cannot derive a held-out split")

    split = build_split(registered_stems(model_dir), holdout)
    if not split["test"]:
        raise SceneError(f"holdout {holdout} left no held-out views for {scene_root}")

    path = scene_root / SPLIT_FILENAME
    with path.open("w") as handle:
        json.dump(split, handle, indent=2)
    logger.info(f"Wrote {path}: {len(split['train'])} train, {len(split['test'])} test")
    return split


def copy_split(src_root: Path, dest_root: Path) -> Path | None:
    """Carry an existing split.json into a prepared copy of a scene."""
    src = src_root / SPLIT_FILENAME
    if not src.is_file():
        return None
    dest = dest_root / SPLIT_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return dest
