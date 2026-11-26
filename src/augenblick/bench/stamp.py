"""Write output digest stamps and input provenance records."""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

from augenblick.bench.digest import (code_version, digest_file, digest_listing,
                                     model_file_digests, write_record)

logger = logging.getLogger(__name__)


def digest_of(path: Path) -> str | dict | None:
    """Digest a path as whatever it is: a COLMAP model, a directory, or one file."""
    path = Path(path)
    if not path.exists():
        return None
    if path.is_dir():
        if any(p.suffix in (".bin", ".txt") for p in path.iterdir() if p.is_file()):
            return model_file_digests(path)
        return digest_listing(path)
    return digest_file(path)


def combined_digest(artefacts) -> str:
    """One digest over several artefacts, so a stage with two outputs has one stamp."""
    h = hashlib.sha256()
    for path in artefacts:
        h.update(Path(path).name.encode())
        h.update((digest_file(path) or "").encode())
    return h.hexdigest()


def stamp(artefacts, out: Path) -> bool:
    """Update the output digest and mtime; return whether the digest changed."""
    out = Path(out)
    wanted = combined_digest(artefacts) + "\n"
    newest = max(Path(a).stat().st_mtime for a in artefacts)
    changed = not (out.is_file() and out.read_text() == wanted)
    if changed:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(wanted)
    os.utime(out, (newest, newest))
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="augenblick.bench.stamp", description=__doc__)
    parser.add_argument("--artefact", type=Path, action="append", required=True,
                        help="a file this stage produced; repeat for several")
    parser.add_argument("--out", type=Path, required=True, help="the stamp to write")
    parser.add_argument("--repo", type=Path, help="checkout to attribute the result to")
    parser.add_argument("--input", action="append", default=[], metavar="NAME=PATH",
                        help="an input to digest into the provenance record")
    parser.add_argument("--record", type=Path,
                        help="where to write provenance; defaults to beside the result")
    args = parser.parse_args(argv)

    missing = [p for p in args.artefact if not Path(p).is_file()]
    if missing:
        raise SystemExit(f"nothing to stamp at {missing[0]}")

    changed = stamp(args.artefact, args.out)

    record: dict = {}
    for pair in args.input:
        name, _, path = pair.partition("=")
        record[name] = str(path)
        record[f"{name}_digest"] = digest_of(path)
    if args.repo:
        record["code"] = code_version(args.repo)
    if record:
        write_record(args.record or args.artefact[0].parent / "inputs.json", record)

    print(f"{args.out.name}: {'changed' if changed else 'unchanged'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
