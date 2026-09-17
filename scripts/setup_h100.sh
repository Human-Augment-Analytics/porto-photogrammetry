#!/bin/bash
# Setup for NVIDIA H100 / H200 (compute capability 9.0).
# Same cu129 / torch 2.9.1 toolchain as every other wrapper: onnxruntime-gpu links CUDA 12
# sonames, so torch must be on CUDA 12 too or its GPU provider cannot load.
set -euo pipefail
export GPU_LABEL="H100"
export GPU_ARCH="9.0"
export CUDA_MODULE="${CUDA_MODULE:-cuda/12.9.1}"
export TORCH_SPEC="torch==2.9.1 torchvision==0.24.1"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu129"
export NUMPY_GENERATION="2"
exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_common.sh"
