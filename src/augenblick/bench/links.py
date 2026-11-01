"""Create input symlinks for the configured specimen roster."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from augenblick.core.errors import SceneError

logger = logging.getLogger(__name__)

# stage -> (which configured source, path template under it, required)
LINKED_STAGES = [
    ("images", "received", "prepared/{spec}/images", True),
    ("masks", "received", "prepared/{spec}/masks", True),
    ("gt_artec", "received", "subset/gt/{spec}", False),
    ("realityscan", "received", "subset/realityscan/{spec}", False),
    ("sfm", "reconstructions", "{spec}/colmap", True),
    ("2dgs", "reconstructions", "{spec}/colmap-2dgs", True),
    ("pgsr", "reconstructions", "{spec}/colmap-pgsr", False),
    ("gw", "reconstructions", "{spec}/colmap-gw", False),
    ("geosvr", "reconstructions", "{spec}/colmap-geosvr", False),
]


def add_arguments(parser) -> None:
    parser.add_argument("--check", action="store_true",
                        help="verify the links without writing anything")


def templates(tree) -> list[tuple[str, Path, str, bool]]:
    """Resolve the configured sources into (stage, root, template, required)."""
    sources = tree.config.get("sources", {})
    configured = tree.config.get("link_stages")
    table = [(s, k, t, bool(r)) for s, k, t, r in configured] if configured \
        else LINKED_STAGES
    resolved = []
    for stage, key, template, required in table:
        root = sources.get(key)
        if not root:
            raise SceneError(f"config names no source {key!r} for stage {stage}")
        resolved.append((stage, Path(root), template, required))
    return resolved


def discover(tree, root: Path) -> list[str]:
    """Specimens present in a source tree, so the roster can be checked against it."""
    if not root.is_dir():
        raise SceneError(f"no source tree at {root}")
    return sorted(p.name for p in root.iterdir()
                  if p.is_dir() and not p.name.startswith((".", "_"))
                  and p.name != "subset")


def link_one(stage_dir: Path, name: str, target: Path, check: bool) -> str:
    """Create one symlink, or say what is wrong with the entry already there."""
    path = stage_dir / name
    if path.is_symlink():
        if os.readlink(path) == str(target):
            return "ok"
        if check:
            return f"wrong target: {os.readlink(path)}"
        path.unlink()
    elif path.exists():
        # A real directory here would mean computed output landed in a linked stage.
        return "not a symlink, refusing to touch it"
    if check:
        return "missing"
    path.symlink_to(target)
    return "created"


def run(tree, args) -> int:
    """Build or verify the tree's linked stages."""
    if tree.config.get("adopted_from"):
        raise SceneError(
            f"{tree.name} adopts its data from {tree.config['adopted_from']}; linking "
            "it afresh from the sources would overwrite that adoption")
    check = bool(getattr(args, "check", False))
    keep, discard = tree.specimens, set(tree.discarded)
    problems, created, absent = [], 0, {}

    overlap = discard & set(keep)
    if overlap:
        raise SceneError("roster both keeps and discards: " + ", ".join(sorted(overlap)))

    stages = templates(tree)
    # Validate the roster against the reconstruction source.
    recon_root = next((root for stage, root, _, _ in stages if stage == "sfm"), None)
    if recon_root is not None:
        found = set(discover(tree, recon_root))
        for name, label in ((set(keep) - found, "kept"), (discard - found, "discarded")):
            if name:
                problems.append(f"roster {label} specimens absent from {recon_root}: "
                                + ", ".join(sorted(name)[:4]))

    for stage, root, template, required in stages:
        stage_dir = tree.stage(stage)
        if not check:
            stage_dir.mkdir(parents=True, exist_ok=True)
        elif not stage_dir.is_dir():
            problems.append(f"{stage}: stage directory missing")
            continue

        for specimen in keep:
            target = root / template.format(spec=specimen)
            if not target.exists():
                if required:
                    problems.append(f"{stage}/{specimen}: source missing at {target}")
                absent.setdefault(stage, []).append(specimen)
                continue
            status = link_one(stage_dir, specimen, target, check)
            if status == "created":
                created += 1
            elif status != "ok":
                problems.append(f"{stage}/{specimen}: {status}")

        # A discarded specimen appearing here would quietly put it back in the results.
        leaked = sorted({p.name for p in stage_dir.iterdir()} & discard)
        if leaked:
            problems.append(f"{stage}: discarded specimens present: " + ", ".join(leaked))

    print(f"specimens: {len(keep)} kept, {len(discard)} discarded")
    for stage in sorted(absent):
        names = absent[stage]
        print(f"  {stage:<12} no source for {len(names)}: "
              + ", ".join(names[:4]) + (" ..." if len(names) > 4 else ""))
    if not check:
        print(f"links created: {created}")
    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for problem in problems:
            print("  " + problem)
        return 1
    print("\nOK" if check else "\ndone")
    return 0
