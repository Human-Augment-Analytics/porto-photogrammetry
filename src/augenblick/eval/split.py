"""Build and validate shared training and test view splits."""
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
    """Reuse an existing split.json or derive and save one from the COLMAP model."""
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


MANUAL = "manual"


def parse_rule(rule) -> tuple[str, int | None]:
    """Read a split rule into its kind and its parameter."""
    if rule == MANUAL:
        return MANUAL, None
    if isinstance(rule, str) and ":" in rule:
        kind, value = rule.split(":", 1)
    else:
        raise SceneError(f"unreadable split rule: {rule!r}")
    if kind != "holdout":
        raise SceneError(f"unknown split rule {kind!r}; expected holdout:n or {MANUAL}")
    try:
        value = int(value)
    except (TypeError, ValueError):
        raise SceneError(f"holdout needs an integer, got {value!r}")
    if value < 2:
        raise SceneError("holdout:n needs n of at least two, or nothing is trained on")
    return kind, value


def check_split(split: dict, stems) -> dict:
    """Reject empty, overlapping or unknown view sets."""
    for side in ("train", "test"):
        if not isinstance(split.get(side), list) or not split[side]:
            raise SceneError(f"split has no {side} views")
    train, test = set(split["train"]), set(split["test"])
    overlap = train & test
    if overlap:
        raise SceneError("views in both train and test: "
                         + ", ".join(sorted(overlap)[:4]))
    unknown = (train | test) - set(stems)
    if unknown:
        raise SceneError("split names views the reconstruction does not hold: "
                         + ", ".join(sorted(unknown)[:4]))
    return split


def main(argv: list[str] | None = None) -> int:
    """Write a scene's split, derived from a rule or adopted from a hand-written file."""
    import argparse

    parser = argparse.ArgumentParser(prog="augenblick.eval.split",
                                     description=main.__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="the COLMAP model whose registered views are split")
    parser.add_argument("--rule", required=True,
                        help=f"holdout:n, or {MANUAL} to adopt --manual")
    parser.add_argument("--manual", type=Path,
                        help=f"a hand-written split, required when --rule is {MANUAL}")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    kind, value = parse_rule(args.rule)
    stems = registered_stems(args.model)
    if kind == MANUAL:
        if not args.manual or not args.manual.is_file():
            raise SceneError(f"--rule {MANUAL} needs a split at {args.manual}")
        split = json.loads(args.manual.read_text())
        split.setdefault("rule", MANUAL)
        split["source"] = str(args.manual)
    else:
        split = {"rule": f"{kind}:{value}", **build_split(stems, value)}
    check_split(split, stems)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
    logger.info(f"{len(split['train'])} train, {len(split['test'])} test "
                f"({split['rule']}) -> {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
