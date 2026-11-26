"""LOCAL PATCH (augenblick): first-party file, not upstream NeuS2.

Stage 2: train NeuS2 on prepare.py's transforms_train.json and save a snapshot. The loop
mirrors upstream scripts/run.py:207-216; scripts/ itself is not vendored.
"""
import argparse
import copy
import logging
import time
from pathlib import Path

import ngp_session
import scene_io
from augenblick.core.errors import SceneError

logger = logging.getLogger("neus2.train")
LOG_EVERY = 500


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[2])
    parser.add_argument("-m", "--model_path", type=Path, required=True, help="Output directory")
    parser.add_argument("--n_steps", type=int, default=15000, help="Training steps")
    parser.add_argument("--ek_loss_weight", type=float, default=0.01, help="Eikonal loss weight")
    parser.add_argument("--mask_loss_weight", type=float, default=0.0, help="Mask BCE loss weight")
    args = parser.parse_args(argv)
    args.model_path = args.model_path.resolve()
    return args


def network_config(n_steps: int, ek_loss_weight: float, mask_loss_weight: float) -> dict:
    """configs/nerf/dtu.json with the loss weights and step limit this run uses."""
    config = copy.deepcopy(scene_io.read_json(scene_io.NETWORK_CONFIG))
    config["hyperparams"].update({"ek_loss_weight": ek_loss_weight,
                                  "mask_loss_weight": mask_loss_weight,
                                  "first_frame_max_training_step": n_steps})
    return config


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    if args.n_steps <= 0:
        raise SceneError("--n_steps must be positive")
    train_json = args.model_path / scene_io.SCENE_DIR / "transforms_train.json"
    if not train_json.is_file():
        raise SceneError(f"no {train_json}; run prepare.py first")
    network = args.model_path / scene_io.STAGE_DIR / "network.json"
    scene_io.write_json(network, network_config(args.n_steps, args.ek_loss_weight, args.mask_loss_weight))

    testbed, _ = ngp_session.open_testbed()
    # A file path, never the directory: a directory loads every JSON in it as a frame.
    testbed.load_training_data(str(train_json))
    testbed.reload_network_from_file(str(network))
    testbed.shall_train = True
    logger.info(f"training {testbed.nerf.training.dataset.n_images} views for {args.n_steps} steps")

    start = time.monotonic()
    logged = 0
    while testbed.frame():
        step = testbed.training_step
        if step >= args.n_steps:
            break
        if step - logged >= LOG_EVERY:
            logged = step
            logger.info(f"step {step}: loss {testbed.loss:.5f} ek {testbed.ek_loss:.5f} "
                        f"mask {testbed.mask_loss:.5f} ({time.monotonic() - start:.0f}s)")
    logger.info(f"trained {testbed.training_step} steps in {time.monotonic() - start:.1f}s, "
                f"final loss {testbed.loss:.5f}")

    snapshot = scene_io.snapshot_path(args.model_path, args.n_steps)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    testbed.save_snapshot(str(snapshot), False)
    logger.info(f"saved {snapshot}")


if __name__ == "__main__":
    scene_io.cli(main)
