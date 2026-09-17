#!/bin/bash
# Setup for NVIDIA A40 (compute capability 8.6, 48 GB).
# Same Ampere toolchain as the A100: CUDA 12.9 / torch 2.9.1 (cu129 wheels).
set -euo pipefail
export GPU_LABEL="A40"
export GPU_ARCH="8.6"
export CUDA_MODULE="${CUDA_MODULE:-cuda/12.9.1}"
export TORCH_SPEC="torch==2.9.1 torchvision==0.24.1"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu129"
export NUMPY_GENERATION="2"
exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_common.sh"
