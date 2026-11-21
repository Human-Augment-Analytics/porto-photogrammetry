"""Digests and provenance records: what an output was derived from."""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def digest_file(path: Path) -> str | None:
    """Content digest of one file, or None when it is not there."""
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_listing(directory: Path, deep: bool = False) -> str | None:
    """Hash entry names and sizes; deep=True also hashes file contents."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    h = hashlib.sha256()
    for entry in sorted(directory.iterdir()):
        h.update(entry.name.encode())
        h.update((digest_file(entry) or "").encode() if deep
                 else ("%d" % entry.stat().st_size).encode())
    return h.hexdigest()


def model_file_digests(model_dir: Path) -> dict[str, str] | None:
    """Return per-file COLMAP digests for scale validation."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return None
    found = {entry.name: digest_file(entry) for entry in sorted(model_dir.iterdir())
             if entry.is_file() and entry.suffix in (".bin", ".txt")}
    return found or None


def code_version(repo: Path) -> dict:
    """Record the repository commit and working-tree state."""
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"],
                                capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, OSError):
        return {"repo": str(repo), "commit": None, "modified": None}
    tracked = [line for line in status.splitlines() if not line.startswith("??")]
    return {"repo": str(repo), "commit": head, "modified": bool(tracked)}


def read_record(path: Path) -> dict | None:
    """Read a JSON record, treating absent or malformed as absent."""
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def write_record(path: Path, payload: dict) -> None:
    """Write a JSON record, sorted so two runs produce identical bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
