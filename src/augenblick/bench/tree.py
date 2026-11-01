"""Configuration and paths for a benchmark experiment."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from augenblick.core.errors import SceneError

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "config.json"

# Received data is read-only and kept apart from what the pipeline computes.
INPUT_STAGES = ("images", "masks", "gt_artec", "realityscan")
RECONSTRUCTION_STAGES = ("sfm", "2dgs", "pgsr", "gw", "geosvr")
COMPUTED_STAGES = ("markers", "scales", "eval", "nvs", "splits")


class Tree:
    """The directory an experiment lives in, plus the config that defines it."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._config = None

    # --- config ----------------------------------------------------------
    @property
    def config(self) -> dict:
        """The tree's config, read once from config.json."""
        if self._config is None:
            path = self.root / CONFIG_FILENAME
            if not path.is_file():
                raise SceneError(f"no {CONFIG_FILENAME} at {self.root}")
            self._config = json.loads(path.read_text())
        return self._config

    @property
    def name(self) -> str:
        return self.config.get("name", self.root.name)

    @property
    def repo(self) -> Path:
        """The checkout whose code this tree's results are attributed to."""
        configured = self.config.get("repo")
        return Path(configured) if configured else Path(
            os.environ.get("AUGENBLICK", Path(__file__).resolve().parents[3]))

    @property
    def toolkit(self) -> Path | None:
        configured = self.config.get("toolkit") or os.environ.get("PGT_TOOLKIT")
        return Path(configured) if configured else None

    @property
    def recipes(self) -> dict:
        """Load marker codes and known lengths; codes alone do not determine bar size."""
        path = self.root / self.config.get("recipes", "recipes.json")
        if not path.is_file():
            raise SceneError(f"no recipes at {path}")
        return json.loads(path.read_text())

    @property
    def split_rule(self) -> dict | None:
        """How views are held out, or None when the tree trains on everything."""
        return self.config.get("split")

    # --- roster ----------------------------------------------------------
    @property
    def specimens(self) -> list[str]:
        """Return the configured specimen roster in order."""
        return list(self.config.get("roster", {}).get("keep", []))

    @property
    def discarded(self) -> list[str]:
        """Specimens excluded from this tree, kept by name so the exclusion is checkable."""
        return list(self.config.get("roster", {}).get("discard", []))

    def select(self, only=None, index=None, listing=None) -> list[str]:
        """Resolve a request into specimens, refusing anything off the roster."""
        if index is not None and only:
            # Require an unambiguous specimen selector.
            raise SceneError("--only and --index select different things; pass one")
        names = ([line.strip() for line in Path(listing).read_text().splitlines()
                  if line.strip()] if listing else self.specimens)
        if index is not None:
            if index >= len(names):
                raise SceneError(f"index {index} exceeds {len(names)} specimens")
            return [names[index]]
        if only:
            # Reject specimens outside the selected roster.
            unknown = sorted(set(only) - set(names))
            if unknown:
                raise SceneError("not on this tree's roster: " + ", ".join(unknown))
            return list(only)
        return names

    # --- paths -----------------------------------------------------------
    def stage(self, name: str) -> Path:
        """Where a stage lives, with received data kept under inputs/."""
        if name in INPUT_STAGES:
            return self.root / "inputs" / name
        return self.root / name

    def path(self, stage: str, specimen: str) -> Path:
        return self.stage(stage) / specimen

    @property
    def work(self) -> Path:
        return self.root / "work"

    def claim(self, out_dir: Path) -> Path:
        """Validate an output directory and remove its previous completion record."""
        out_dir = Path(out_dir)
        if out_dir.is_symlink():
            raise SceneError(f"{out_dir} is a symlink; computed output must not follow one")
        if out_dir.exists() and self.root not in out_dir.resolve().parents:
            raise SceneError(f"{out_dir} resolves outside the tree, to {out_dir.resolve()}")
        out_dir.mkdir(parents=True, exist_ok=True)
        record = out_dir / "inputs.json"
        if record.exists():
            record.unlink()
        return out_dir
