import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("backends,has_source", [
    ("neuralangelo", True), ("neuralangelo", False), ("unused", False),
])
def test_shared_setup_selects_neuralangelo_only_when_requested(tmp_path, backends, has_source):
    root = Path(__file__).resolve().parents[1]
    commands = tmp_path / "bin"
    commands.mkdir()
    trace = tmp_path / "commands.log"
    upstream = tmp_path / "upstream"
    if has_source:
        upstream.mkdir()
        (upstream / "train.py").touch()
    stub = """#!/bin/bash
printf '%s %s\\n' "${0##*/}" "$*" >> "$SETUP_TRACE"
case "${0##*/}" in
    python)
        case "$*" in
            *sys.version_info*) echo 3.10 ;;
            --version) echo 'Python 3.10' ;;
        esac ;;
    nvcc) echo 'Cuda compilation tools, release 12.9' ;;
esac
"""
    for name in ("python", "git", "nvcc", "module"):
        executable = commands / name
        executable.write_text(stub)
        executable.chmod(0o755)
    result = subprocess.run(
        ["bash", str(root / "scripts/setup_common.sh")],
        env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}",
             "SETUP_TRACE": str(trace), "BACKENDS": backends,
             "GPU_LABEL": "test", "GPU_ARCH": "8.9", "CUDA_MODULE": "test",
             "TORCH_SPEC": "torch", "TORCH_INDEX_URL": "https://example.invalid",
             "NUMPY_GENERATION": "2", "PYTORCH3D_WHEEL": "https://example.invalid",
             "NEURALANGELO_ROOT": str(upstream)},
        capture_output=True, text=True,
    )
    if backends == "neuralangelo":
        assert result.returncode == (0 if has_source else 1), result.stdout + result.stderr
        assert "git submodule update --init --recursive src/libs/neuralangelo" in trace.read_text()
        assert "Neuralangelo dependencies and CUDA preflight" in result.stdout
        if has_source:
            assert "python -m pip freeze --exclude-editable" in trace.read_text()
            assert "wandb==0.31.0" in trace.read_text()
        else:
            assert "Neuralangelo source not found" in result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert "src/libs/neuralangelo" not in trace.read_text()
        assert "Neuralangelo dependencies and CUDA preflight" not in result.stdout