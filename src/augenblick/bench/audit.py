"""Check output paths and optionally audit source-tree ownership."""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from augenblick.bench.tree import COMPUTED_STAGES, INPUT_STAGES, RECONSTRUCTION_STAGES

logger = logging.getLogger(__name__)


def add_arguments(parser) -> None:
    parser.add_argument("--shared-trees", action="store_true",
                        help="also walk the source trees; takes minutes")


def shared_roots(tree) -> list[Path]:
    return [Path(p) for p in sorted(set(tree.config.get("sources", {}).values()))]


def computed_stay_inside(tree) -> list[str]:
    """Every computed output path resolves within the tree."""
    problems = []
    for stage in (*COMPUTED_STAGES, "work", "analysis"):
        root = tree.stage(stage) if stage in COMPUTED_STAGES else tree.root / stage
        if not root.is_dir():
            continue
        if root.is_symlink():
            problems.append(f"{stage} is a symlink; computed output must not follow one")
            continue
        for child in root.iterdir():
            target = child.resolve()
            if tree.root not in target.parents and target != tree.root:
                problems.append(f"{stage}/{child.name} resolves outside the tree, "
                                f"to {target}")
    return problems


def adopted_are_read_only_links(tree) -> list[str]:
    """Adopted stages point into shared data and are not written through."""
    shared = shared_roots(tree)
    problems = []
    for stage in (*INPUT_STAGES, *RECONSTRUCTION_STAGES):
        root = tree.stage(stage)
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if child.name == "provenance.json" or not child.is_symlink():
                continue
            target = Path(os.readlink(child))
            if any(root_path in target.parents for root_path in shared):
                # Detect files written through an adopted link.
                owned = [p for p in target.glob("*")
                         if p.exists() and p.stat().st_uid == os.getuid()]
                if owned:
                    problems.append(
                        f"{stage}/{child.name}: {len(owned)} file(s) under the shared "
                        f"target are owned by us, e.g. {owned[0]}")
    return problems


def shared_trees_hold_nothing_of_ours(tree) -> list[str]:
    """Walk the source trees for anything this account owns."""
    problems = []
    for root in shared_roots(tree):
        if not root.is_dir():
            continue
        result = subprocess.run(["find", str(root), "-user", str(os.getuid()), "-print"],
                                capture_output=True, text=True)
        found = [line for line in result.stdout.splitlines() if line.strip()]
        print(f"  {root}: {len(found)} file(s) owned by us")
        problems += [f"owned by us in a shared tree: {line}" for line in found[:10]]
        if len(found) > 10:
            problems.append(f"... and {len(found) - 10} more under {root}")
    return problems


def run(tree, args) -> int:
    problems = computed_stay_inside(tree) + adopted_are_read_only_links(tree)
    print(f"structural check: computed stages stay inside {tree.root}")
    if getattr(args, "shared_trees", False):
        print("walking the source trees:")
        problems += shared_trees_hold_nothing_of_ours(tree)

    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for problem in problems:
            print("  " + problem)
        return 1
    print("\nOK: nothing written outside the tree")
    return 0
