import os
from pathlib import Path
import shutil
import subprocess

import pytest


def test_build_wrapper_uses_modern_headers_without_modifying_upstream(tmp_path):
    if not shutil.which("cmake"):
        pytest.skip("CMake is unavailable")
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    config = source / "CMakeLists.txt"
    config.write_text(
        'add_library(pybind11_headers INTERFACE)\n'
        'add_library(pyngp MODULE dummy.cpp)\n')
    (source / "dummy.cpp").write_text("int neus2_test() { return 0; }\n")
    headers = tmp_path / "headers"
    (headers / "pybind11").mkdir(parents=True)
    (headers / "pybind11/numpy.h").touch()
    original = config.read_bytes()
    subprocess.run(["cmake", "-S", str(root / "scripts/neus2_build"), "-B", str(tmp_path / "build"),
                    f"-DNEUS2_SOURCE_DIR={source}", f"-DNEUS2_PYBIND11_INCLUDE={headers}",
                    "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"],
                   check=True, capture_output=True, text=True)
    assert str(headers) in (tmp_path / "build/compile_commands.json").read_text()
    assert config.read_bytes() == original


def test_setup_requires_initialized_submodule_before_installing(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(["bash", str(root / "scripts/setup_neus2.sh")],
                            env={**os.environ, "NEUS2_ROOT": str(tmp_path / "missing")},
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "git submodule update --init --recursive src/libs/neus2" in result.stderr


@pytest.mark.parametrize("backends", ["neus2", "unused"])
def test_shared_installer_only_builds_neus2_when_selected(tmp_path, backends):
    root = Path(__file__).resolve().parents[1]
    commands = tmp_path / "bin"
    commands.mkdir()
    trace = tmp_path / "commands.log"
    upstream = tmp_path / "upstream"
    (upstream / "dependencies/neus2_tcnn").mkdir(parents=True)
    (upstream / "dependencies/neus2_tcnn/CMakeLists.txt").touch()
    stub = """#!/bin/bash
printf '%s %s\\n' "${0##*/}" "$*" >> "$SETUP_TRACE"
case "${0##*/}" in
    python)
        case "$*" in
            *sys.version_info*) echo 3.10 ;;
            *sys.executable*) echo "$0" ;;
            --version) echo 'Python 3.10' ;;
        esac ;;
    nvcc) echo 'Cuda compilation tools, release 12.9' ;;
esac
"""
    for name in ("python", "git", "nvcc", "module", "cmake"):
        executable = commands / name
        executable.write_text(stub)
        executable.chmod(0o755)
    result = subprocess.run(["bash", str(root / "scripts/setup_common.sh")],
                            env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}",
                                 "SETUP_TRACE": str(trace), "BACKENDS": backends,
                                 "GPU_LABEL": "test", "GPU_ARCH": "8.9", "CUDA_MODULE": "test",
                                 "TORCH_SPEC": "torch", "TORCH_INDEX_URL": "https://example.invalid",
                                 "NUMPY_GENERATION": "2", "PYTORCH3D_WHEEL": "https://example.invalid",
                                 "NEUS2_ROOT": str(upstream), "NEUS2_BUILD_DIR": str(tmp_path / "build")},
                            check=True, capture_output=True, text=True)
    calls = trace.read_text()
    if backends == "neus2":
        assert "git submodule update --init --recursive src/libs/neus2" in calls
        assert "pybind11==2.13.6" in calls
        assert "-DNGP_BUILD_WITH_GUI=OFF" in calls
        assert "--target pyngp" in calls
        assert "NeuS2 headless CUDA bindings" in result.stdout
    else:
        assert "pybind11==2.13.6" not in calls
        assert "src/libs/neus2" not in calls