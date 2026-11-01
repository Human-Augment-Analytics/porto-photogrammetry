#!/usr/bin/env python3
"""Create an experiment that reuses another tree's inputs and SfM."""
import argparse
import json
import os
import sys
from pathlib import Path

ADOPTED = ["inputs/images", "inputs/masks", "inputs/gt_artec", "inputs/realityscan",
           "sfm"]


def adopt(source_stage: Path, dest_stage: Path, specimens, check: bool) -> int:
    linked = 0
    for specimen in specimens:
        target = source_stage / specimen
        if not target.exists():
            continue
        if not check:
            dest_stage.mkdir(parents=True, exist_ok=True)
        link = dest_stage / specimen
        if link.is_symlink():
            if os.readlink(link) == str(target.resolve()):
                continue
            if check:
                print(f"  {link}: wrong target")
                continue
            link.unlink()
        elif link.exists():
            sys.exit(f"ERROR: {link} exists and is not a symlink")
        if check:
            print(f"  would link {link}")
            continue
        # Resolved, so the new tree does not depend on the old one's internal layout.
        link.symlink_to(target.resolve())
        linked += 1
    return linked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="source", type=Path, required=True)
    parser.add_argument("--to", dest="dest", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", required=True,
                        help="holdout:n, or manual for a hand-written split.json")
    parser.add_argument("--question", default="")
    parser.add_argument("--only", action="append",
                        help="limit the instance to these specimens")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    source_config = json.loads((args.source / "config.json").read_text())
    specimens = args.only or sorted(source_config.get("references", {}))
    if not specimens:
        sys.exit("ERROR: nothing to adopt; pass --only or give the source references")

    adopted = {}
    for stage in ADOPTED:
        source_stage = args.source / stage
        if not source_stage.is_dir():
            continue
        count = adopt(source_stage, args.dest / stage, specimens, args.check)
        adopted[stage.split("/")[-1]] = {
            "origin": "adopted", "from": str(args.source / stage), "linked": count}

    config = {
        "name": args.name,
        "question": args.question or f"novel view synthesis under {args.split}",
        "repo": source_config.get("repo"),
        "toolkit": source_config.get("toolkit"),
        "sources": source_config.get("sources", {}),
        "roster": {"keep": specimens, "discard": [],
                   "rule": f"adopted from {args.source.name}"},
        "references": {s: r for s, r in source_config.get("references", {}).items()
                       if s in specimens},
        "split": args.split,
        "split_model_stage": "sfm",
        "scale_model_stage": source_config.get("scale_model_stage", "sfm"),
        "mesh_stage": source_config.get("mesh_stage", "2dgs"),
        "mesh_path": source_config.get("mesh_path"),
        "adopted_from": str(args.source),
    }

    if args.check:
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n")
    (args.dest / "provenance.json").write_text(
        json.dumps(adopted, indent=2, sort_keys=True) + "\n")
    for directory in ("work/logs", "analysis"):
        (args.dest / directory).mkdir(parents=True, exist_ok=True)

    print(f"{args.name}: {len(specimens)} specimens, split {args.split}")
    for stage, record in sorted(adopted.items()):
        print(f"  adopted {stage:<12} {record['linked']} newly linked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
