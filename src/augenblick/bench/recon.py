"""Submit reconstruction jobs and publish outputs into a benchmark tree."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from augenblick.core.errors import SceneError

logger = logging.getLogger(__name__)

# stage -> (launcher, output subdirectory, completion marker, extra environment)
LAUNCHERS = {
    "sfm": ("hpg_slurm/colmap_sfm.sbatch", "colmap", "sparse/0/cameras.bin", {}),
    "2dgs": ("hpg_slurm/recon.sbatch", "colmap-2dgs",
             "train/ours_30000/fuse_post.ply", {"BACKEND": "2dgs", "SFM": "colmap"}),
}
CONCURRENCY = 4


def add_arguments(parser) -> None:
    parser.add_argument("--stage", choices=sorted(LAUNCHERS),
                        help="which reconstruction step to submit")
    parser.add_argument("--status", action="store_true",
                        help="report what exists and what is pending, submit nothing")


def scenes_dir(tree) -> Path:
    return tree.work / "scenes"


def output_dir(tree, specimen: str, stage: str) -> Path:
    return tree.work / "replica" / specimen / LAUNCHERS[stage][1]


def done(tree, specimen: str, stage: str) -> bool:
    return (output_dir(tree, specimen, stage) / LAUNCHERS[stage][2]).exists()


def stage_scenes(tree, specimens) -> None:
    """Stage pending specimens in the launchers' array-index order."""
    root = scenes_dir(tree)
    root.mkdir(parents=True, exist_ok=True)
    wanted = set(specimens)
    for scene in sorted(root.iterdir()):
        if scene.name in wanted:
            continue
        for sub in ("images", "masks"):
            link = scene / sub
            if link.is_symlink():
                link.unlink()
        scene.rmdir()
    for specimen in specimens:
        scene = root / specimen
        scene.mkdir(exist_ok=True)
        for sub in ("images", "masks"):
            link, target = scene / sub, tree.path(sub, specimen)
            if link.is_symlink():
                if os.readlink(link) == str(target):
                    continue
                link.unlink()
            elif link.exists():
                raise SceneError(f"{link} exists and is not a symlink")
            link.symlink_to(target)


def publish(tree, specimens, stage: str) -> int:
    """Publish output links; replace adopted stages only via the config's produce list."""
    stage_dir = tree.stage(stage)
    wanted = set(tree.config.get("produce") or ())
    stage_dir.mkdir(parents=True, exist_ok=True)
    published, withheld = [], []
    for specimen in specimens:
        if not done(tree, specimen, stage):
            continue
        target, link = output_dir(tree, specimen, stage), stage_dir / specimen
        adopted = link.is_symlink() and tree.root not in link.resolve().parents
        if adopted and specimen not in wanted:
            withheld.append(specimen)
            continue
        if link.is_symlink():
            if os.readlink(link) != str(target):
                link.unlink()
                link.symlink_to(target)
        elif link.exists():
            raise SceneError(f"{link} exists and is not a symlink")
        else:
            link.symlink_to(target)
        published.append(specimen)
    if withheld:
        print(f"  {stage}: {len(withheld)} replica(s) kept out of the stage, which "
              "adopts; add them to the config's produce list to publish")
    if published:
        record = stage_dir / "provenance.json"
        payload = json.loads(record.read_text()) if record.is_file() else {}
        payload.update({s: {"origin": "produced", "path": str(output_dir(tree, s, stage))}
                        for s in published})
        record.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return len(published)


def submit(tree, stage: str, specimens) -> int:
    launcher, _, _, extra = LAUNCHERS[stage]
    env = dict(os.environ)
    env.update(extra)
    env.update({"DATA_ROOT": str(scenes_dir(tree)),
                "RESULT_ROOT": str(tree.work / "replica"),
                "SCENE_LAYOUT": "flat"})
    command = [
        "sbatch", f"--array=0-{len(specimens) - 1}%{CONCURRENCY}",
        # Override launcher email settings.
        "--mail-type=NONE",
        f"--job-name=bench-recon-{stage}",
        launcher,
    ]
    print("submitting: " + " ".join(command))
    print("  scenes: " + ", ".join(specimens))
    result = subprocess.run(command, cwd=str(tree.repo), env=env,
                            capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


def run(tree, args) -> int:
    """Reconstruct the specimens the tree's config asks us to produce ourselves."""
    specimens = tree.select(getattr(args, "only", None), getattr(args, "index", None),
                            getattr(args, "listing", None))
    produce = tree.config.get("produce")
    if produce is not None:
        specimens = [s for s in specimens if s in set(produce)]
    elif getattr(args, "only", None) is None:
        # Default to the specimens with a reference: those are the ones worth the GPU.
        specimens = [s for s in specimens if s in tree.config.get("references", {})]
    if not specimens:
        raise SceneError("config names nothing to reconstruct; set produce or references")

    if args.status or not args.stage:
        for stage in sorted(LAUNCHERS):
            pending = [s for s in specimens if not done(tree, s, stage)]
            complete = len(specimens) - len(pending)
            published = publish(tree, specimens, stage)
            print(f"{stage:<5}  replica {complete}/{len(specimens)}  "
                  f"published {published}  pending: "
                  + (", ".join(pending) if pending else "none"))
        return 0

    pending = [s for s in specimens if not done(tree, s, args.stage)]
    publish(tree, specimens, args.stage)
    if not pending:
        print(f"{args.stage}: nothing pending")
        return 0
    stage_scenes(tree, pending)
    return submit(tree, args.stage, pending)
