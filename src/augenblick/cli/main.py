"""Command-line entry point for preparation, masking, SfM, and reconstruction stages."""
import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

from augenblick.core.config import add_dataclass_arguments
from augenblick.core.errors import BackendError, MethodNotFound, SceneError
from augenblick.core.registry import (
    MASK_REGISTRY,
    RECONSTRUCTION_REGISTRY,
    SFM_REGISTRY,
    get_method,
)
from augenblick.preparation.color import (
    ColorCalibrationError,
    calibrate_directory,
    load_config,
)

logger = logging.getLogger(__name__)

class Stage(NamedTuple):
    """What the CLI needs to build and resolve one stage."""

    registry: dict[str, type]
    kind: str
    help: str
    input_flag: str
    input_help: str


# One entry per stage; adding a stage is one edit here.
STAGES: dict[str, Stage] = {
    "mask": Stage(MASK_REGISTRY, "mask", "Run a masking method",
                  "--images", "Input directory of .jpg/.JPG/.jpeg photographs"),
    "sfm": Stage(SFM_REGISTRY, "SfM", "Run an SfM method",
                 "--scene", "Input scene directory"),
    "recon": Stage(RECONSTRUCTION_REGISTRY, "reconstruction",
                   "Run a reconstruction backend", "--scene", "Input scene directory"),
}


def _add_stage_parser(subparsers, stage_name: str, stage: Stage):
    """Build the parser for one stage, with a per-method subparser drawn from the registry."""
    parser = subparsers.add_parser(stage_name, help=stage.help)
    parser.set_defaults(stage_parser=parser)
    parser.add_argument("--list", action="store_true", help="List available methods and exit")
    method_subs = parser.add_subparsers(dest="method")
    for name, cls in sorted(stage.registry.items()):
        sub = method_subs.add_parser(name, help=cls.__doc__)
        sub.add_argument(stage.input_flag, dest="input_dir", type=Path, required=True,
                         help=stage.input_help)
        sub.add_argument("--output", type=Path, required=True, help="Output directory")
        add_dataclass_arguments(sub, cls.config_cls)
    return parser


def build_parser(include_backends: bool = True) -> argparse.ArgumentParser:
    """Build the CLI parser, optionally loading GPU/backend dependencies."""
    if include_backends:
        # Importing these packages populates the method registries.
        importlib.import_module("augenblick.masking")
        importlib.import_module("augenblick.reconstruction")
        importlib.import_module("augenblick.sfm")
    parser = argparse.ArgumentParser(
        prog="augenblick",
        description="Masking, SfM initialisation, and Gaussian-primitive surface reconstruction.",
    )
    subparsers = parser.add_subparsers(dest="stage")
    if include_backends:
        for name, stage in STAGES.items():
            _add_stage_parser(subparsers, name, stage)
    color = subparsers.add_parser("color", help="Calibrate image colours by camera")
    color.add_argument("--input", type=Path, required=True, help="Input image tree")
    color.add_argument("--output", type=Path, required=True, help="Calibrated output tree")
    color.add_argument("--config", type=Path, required=True, help="Chart configuration JSON")
    color.add_argument("--overwrite", action="store_true", help="Allow writes into a non-empty output")
    return parser


def _list_methods(registry: dict[str, type]) -> int:
    """Print the registered method names with their one-line descriptions."""
    for name, cls in sorted(registry.items()):
        summary = (cls.__doc__ or "").strip().splitlines()[0] if cls.__doc__ else ""
        print(f"{name:<12} {summary}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, resolve the method, and run it.

    Args:
        argv: Argument list, defaulting to sys.argv[1:].

    Returns:
        A process exit code: 2 for scene/lookup errors, a backend's own code on failure.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser(include_backends=raw_argv[:1] != ["color"])
    # GW forwards unknown flags to its training step, so parse leniently and gate below.
    args, extras = parser.parse_known_args(raw_argv)

    if args.stage is None:
        parser.print_help()
        return 2

    if args.stage == "color":
        if extras:
            logger.error(f"unrecognised arguments: {' '.join(extras)}")
            return 2
        try:
            config = load_config(args.config.resolve())
            report = calibrate_directory(
                args.input.resolve(),
                args.output.resolve(),
                config,
                overwrite=args.overwrite,
            )
        except (ColorCalibrationError, OSError, json.JSONDecodeError) as exc:
            logger.error(str(exc))
            return 2
        logger.info(
            "Colour calibration complete: %s",
            args.output.resolve() / "color_calibration_report.json",
        )
        logger.info("Processed images by camera: %s", report["processed_images"])
        return 0

    stage = STAGES[args.stage]
    if getattr(args, "list", False):
        return _list_methods(stage.registry)
    if args.method is None:
        args.stage_parser.print_help()
        return 2

    try:
        cls = get_method(stage.registry, args.method, stage.kind)
    except MethodNotFound as exc:
        logger.error(str(exc))
        return 2

    if extras and not cls.accepts_passthrough:
        logger.error(f"unrecognised arguments: {' '.join(extras)}")
        return 2

    method = cls.from_namespace(args, extras) if cls.accepts_passthrough else cls.from_namespace(args)

    try:
        method.run(cls.build_input(args.input_dir.resolve()), args.output.resolve())
    except SceneError as exc:
        logger.error(str(exc))
        return 2
    except BackendError as exc:
        logger.error(str(exc))
        return exc.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
