"""Reference detector for circular coded scale-bar targets, via PGT-Toolkit.

Produces the detections JSON consumed by ``augenblick.eval.scale``: a list of
``{"image": <file name>, "raw_detections": [{"code": <id>, "x": <px>, "y": <px>}]}``
records, one per photograph, with centres in OpenCV pixel convention. Any decoder of
circular coded targets can produce this file; this module is one working recipe, not
a dependency of the estimator.

PGT-Toolkit has no PyPI release; clone it and point ``--toolkit`` or ``PGT_TOOLKIT``
at the checkout. Adaptive thresholding is enabled by default because a global
threshold misses bars that occupy a small fraction of the frame: the background
dominates the intensity histogram and the bar's own contrast never forms a mode.
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODES = "adaptive+otsu"
DEFAULT_CIRCULARITY = 0.6
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

_WORKER: dict = {}


def _worker_init(workdir: str, toolkit: str, modes: str) -> None:
    # The toolkit reads settings.yaml from the working directory, so every worker
    # runs inside the prepared one.
    os.chdir(workdir)
    sys.path.insert(0, str(Path(toolkit) / "src/pgt_toolkit"))
    _WORKER["modes"] = modes.split("+")


def _detect_one(photo: str) -> dict | None:
    import cv2
    from targets.targets_engine import PGTEngine

    image = cv2.imread(photo)
    if image is None:
        return None
    found = PGTEngine.find_targets_in_image(
        "CCT_20bit_black", image, Path(photo).name, _WORKER["modes"], None, [0])
    return {"image": Path(photo).name,
            "raw_detections": [{"code": target.id, "x": target.center_pt_x,
                                "y": target.center_pt_y} for target in found]}


def prepare_workdir(toolkit: Path, workdir: Path, circularity: float) -> None:
    """Write the toolkit settings this run uses, so the output is reproducible."""
    import yaml

    settings = yaml.safe_load((toolkit / "settings.yaml").read_text())
    settings["logging"]["log_file"] = str(workdir / "detector.log")
    settings["logging"]["level"] = "WARNING"
    settings["cct_decoder"]["target_center_circularity_threshold"] = circularity
    (workdir / "settings.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))


def detect_images(images: list[Path], toolkit: Path, *, modes: str = DEFAULT_MODES,
                  circularity: float = DEFAULT_CIRCULARITY,
                  workers: int | None = None, workdir: Path | None = None,
                  skip_unreadable: bool = False) -> list[dict]:
    """Run the decoder over the photographs; order follows the input list."""
    # Workers chdir into the prepared directory, so every path they touch must
    # survive that: resolve everything in the parent first.
    toolkit = Path(toolkit).resolve()
    images = [Path(p).resolve() for p in images]
    if not (toolkit / "src/pgt_toolkit/targets/targets_engine.py").is_file():
        raise FileNotFoundError(f"No PGT-Toolkit checkout at {toolkit}")
    if workers is None:
        try:
            workers = max(1, len(os.sched_getaffinity(0)))
        except AttributeError:
            workers = max(1, os.cpu_count() or 1)
    with tempfile.TemporaryDirectory() as fallback:
        prepared = Path(workdir).resolve() if workdir else Path(fallback)
        prepared.mkdir(parents=True, exist_ok=True)
        prepare_workdir(toolkit, prepared, circularity)
        # The decoder is single-threaded OpenCV; parallelise across photographs.
        # Spawn, not fork: workers must chdir and import inside the prepared dir.
        with multiprocessing.get_context("spawn").Pool(
                workers, initializer=_worker_init,
                initargs=(str(prepared), str(toolkit), modes)) as pool:
            records = pool.map(_detect_one, [str(p) for p in images])
    unreadable = [img.name for img, r in zip(images, records) if r is None]
    if unreadable:
        # A silently shrinking examined set would undermine the estimator's
        # one-record-per-photograph accounting, so dropping is opt-in.
        if not skip_unreadable:
            raise OSError(f"{len(unreadable)} unreadable images: {unreadable[:5]}")
        logger.warning("Skipping %d unreadable images: %s",
                       len(unreadable), unreadable[:5])
    return [r for r in records if r is not None]


def main(argv: list[str] | None = None) -> int:
    """Detect coded targets in a directory of photographs and write the JSON."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m augenblick.eval.detect",
        description="Decode circular coded scale-bar targets with PGT-Toolkit, "
                    "writing the detections JSON that augenblick.eval.scale consumes.")
    parser.add_argument("--images", required=True, type=Path,
                        help="Directory of unmasked photographs showing the bars")
    parser.add_argument("--output", required=True, type=Path,
                        help="Detections JSON to write; refuses to overwrite")
    parser.add_argument("--toolkit", type=Path,
                        default=os.environ.get("PGT_TOOLKIT"),
                        help="PGT-Toolkit checkout (default: $PGT_TOOLKIT)")
    parser.add_argument("--modes", default=DEFAULT_MODES,
                        help="Thresholding modes, '+'-separated (default: %(default)s)")
    parser.add_argument("--circularity", type=float, default=DEFAULT_CIRCULARITY,
                        help="Target-centre circularity threshold (default: %(default)s)")
    parser.add_argument("--workers", type=int,
                        help="Worker processes (default: the CPUs available)")
    parser.add_argument("--skip-unreadable", action="store_true",
                        help="Skip unreadable image files instead of failing")
    args = parser.parse_args(argv)

    if args.toolkit is None:
        parser.error("--toolkit or $PGT_TOOLKIT must point at a PGT-Toolkit checkout")
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    images = sorted(p for p in args.images.iterdir()
                    if p.suffix.lower() in IMAGE_SUFFIXES
                    and ".mask." not in p.name.lower())
    if not images:
        raise SystemExit(f"No photographs under {args.images}")
    records = detect_images(images, args.toolkit, modes=args.modes,
                            circularity=args.circularity, workers=args.workers,
                            skip_unreadable=args.skip_unreadable)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records))
    detections = sum(len(r["raw_detections"]) for r in records)
    print(f"{len(records)} images, {detections} detections -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
