#!/bin/bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
UPSTREAM="${NEUS2_ROOT:-$REPO/src/libs/neus2}"
BUILD_DIR="${NEUS2_BUILD_DIR:-$UPSTREAM/build}"
if [[ ! -f "$UPSTREAM/dependencies/neus2_tcnn/CMakeLists.txt" ]]; then
    echo "Initialize NeuS2 with git submodule update --init --recursive src/libs/neus2" >&2
    exit 1
fi
for tool in cmake nvcc; do
    if ! command -v "$tool" >/dev/null; then
        echo "Missing $tool; run setup in a GPU allocation with a CUDA toolkit loaded." >&2
        exit 1
    fi
done
PYTHON="$("$PYTHON" -c 'import sys; print(sys.executable)')"
if [[ -z "${TCNN_CUDA_ARCHITECTURES:-}" ]]; then
    if [[ -n "${GPU_ARCH:-}" ]]; then
        export TCNN_CUDA_ARCHITECTURES="${GPU_ARCH//./}"
    elif command -v nvidia-smi >/dev/null; then
        CAPABILITIES="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)"
        read -r CAPABILITY <<< "$CAPABILITIES"
        export TCNN_CUDA_ARCHITECTURES="${CAPABILITY//./}"
    else
        echo "Set TCNN_CUDA_ARCHITECTURES to the allocated GPU architecture (for example 89 or 80)." >&2
        exit 1
    fi
fi
mkdir -p "$BUILD_DIR"
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
UPSTREAM="$(cd "$UPSTREAM" && pwd)"
"$PYTHON" -m pip install --no-deps --upgrade --target "$BUILD_DIR/build-python" pybind11==2.13.6
cmake -S "$REPO/scripts/neus2_build" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DNEUS2_SOURCE_DIR="$UPSTREAM" \
    -DNEUS2_PYBIND11_INCLUDE="$BUILD_DIR/build-python/pybind11/include" \
    -DPython_EXECUTABLE="$PYTHON" -DPYTHON_EXECUTABLE="$PYTHON" \
    -DCMAKE_CUDA_ARCHITECTURES="$TCNN_CUDA_ARCHITECTURES" \
    -DNGP_BUILD_WITH_GUI=OFF -DNGP_BUILD_WITH_VULKAN=OFF -DNGP_BUILD_WITH_OPTIX=OFF
cmake --build "$BUILD_DIR" --config Release --target pyngp --parallel "${BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-4}}"
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" NEUS2_BUILD_DIR="$BUILD_DIR" "$PYTHON" - <<'PY'
import os
from pathlib import Path
import numpy as np
from augenblick.reconstruction.neus2 import Neus2Backend

ngp = Neus2Backend.load_bindings(Path(os.environ["NEUS2_BUILD_DIR"]))
testbed = ngp.Testbed(ngp.TestbedMode.Nerf)
camera = np.eye(4, dtype=np.float32)[:3]
testbed.camera_matrix = camera
np.testing.assert_allclose(np.asarray(testbed.camera_matrix), camera)
print("NEUS2_NATIVE_IMPORT_AND_NUMPY_OK")
PY