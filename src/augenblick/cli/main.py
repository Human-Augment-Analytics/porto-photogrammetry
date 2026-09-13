"""Command-line entry point: `augenblick {mask,sfm,recon} <method> --<input> <dir> --output <dir>`."""
import argparse
import logging
import sys
from pathlib import Path

from augenblick.core.config import add_dataclass_arguments
from augenblick.core.errors import BackendError, MethodNotFound, SceneError
from augenblick.core.registry import (
    MASK_REGISTRY,
    RECONSTRUCTION_REGISTRY,
    SFM_REGISTRY,
    get_mask,
    get_reconstruction,
    get_sfm,
)

logger = logging.getLogger(__name__)

# Importing the packages populates the registries the subparsers are built from.
import augenblick.masking  # noqa: E402,F401
import augenblick.reconstruction  # noqa: E402,F401
import augenblick.sfm  # noqa: E402,F401


# One entry per stage; adding a stage is one edit here.
#     stage -> (registry, getter, help, input_flag, input_help)
STAGES: dict[str, tuple[dict[str, type], object, str, str, str]] = {
    "mask": (MASK_REGISTRY, get_mask, "Run a masking method",
             "--images", "Input directory of .jpg/.JPG/.jpeg photographs"),
    "sfm": (SFM_REGISTRY, get_sfm, "Run an SfM method",
            "--scene", "Input scene directory"),
    "recon": (RECONSTRUCTION_REGISTRY, get_reconstruction,
              "Run a reconstruction backend", "--scene", "Input scene directory"),
}


def _add_stage_parser(subparsers, stage: str, registry: dict[str, type],
                      help_text: str, input_flag: str, input_help: str):
    """Build the parser for one stage, with a per-method subparser drawn from the registry."""
    parser = subparsers.add_parser(stage, help=help_text)
    parser.set_defaults(stage_parser=parser)
    parser.add_argument("--list", action="store_true", help="List available methods and exit")
    method_subs = parser.add_subparsers(dest="method")
    for name, cls in sorted(registry.items()):
        sub = method_subs.add_parser(name, help=cls.__doc__)
        sub.add_argument(input_flag, dest="input_dir", type=Path, required=True,
                         help=input_help)
        sub.add_argument("--output", type=Path, required=True, help="Output directory")
        add_dataclass_arguments(sub, cls.config_cls)
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Build the full CLI parser, one subparser per registered method."""
    parser = argparse.ArgumentParser(
        prog="augenblick",
        description="Masking, SfM initialisation, and Gaussian-primitive surface reconstruction.",
    )
    subparsers = parser.add_subparsers(dest="stage")
    for stage, (reg, _, help_text, flag, help_flag) in STAGES.items():
        _add_stage_parser(subparsers, stage, reg, help_text, flag, help_flag)
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
    parser = build_parser()
    # GW forwards unknown flags to its training step, so parse leniently and gate below.
    args, extras = parser.parse_known_args(argv)

    if args.stage is None:
        parser.print_help()
        return 2

    registry, getter, _, _, _ = STAGES[args.stage]
    if getattr(args, "list", False):
        return _list_methods(registry)
    if args.method is None:
        args.stage_parser.print_help()
        return 2

    try:
        cls = getter(args.method)
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
