"""Workflow configuration checks without submitting jobs."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SNAKEMAKE = shutil.which("snakemake")
pytestmark = pytest.mark.skipif(SNAKEMAKE is None, reason="Snakemake is not installed")


def dry_run(tmp_path, backend, mesh_paths=None):
    config = {"repo": str(REPO), "roster": {"keep": []}, "backends": [backend]}
    if mesh_paths is not None:
        config["mesh_paths"] = mesh_paths
    (tmp_path / "config.json").write_text(json.dumps(config))
    return subprocess.run(
        [SNAKEMAKE, "-s", str(REPO / "workflow/Snakefile"),
         "--config", f"tree={tmp_path}", "-n"],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("backend", ["geosvr", "milo", "typo"])
def test_unconfigured_backend_does_not_fall_back_to_a_2dgs_mesh(tmp_path, backend):
    result = dry_run(tmp_path, backend)
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert backend in output
    assert "Configure mesh_paths" in output
    assert "cameras.json" in output


def test_explicit_mesh_path_accepts_an_additional_backend(tmp_path):
    result = dry_run(tmp_path, "milo", {"milo": "mesh_regular_tsdf_res4096_post.ply"})
    assert result.returncode == 0, result.stdout + result.stderr
