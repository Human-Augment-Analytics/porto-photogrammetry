#!/bin/bash
# Setup for NVIDIA A40 (compute capability 8.6, 48 GB).
# Same Ampere toolchain as the A100: CUDA 13.0 / torch 2.9.1 (cu130 wheels).
set -euo pipefail
export GPU_LABEL="A40"
export GPU_ARCH="8.6"
export CUDA_MODULE="${CUDA_MODULE:-cuda/13.0.1}"
export TORCH_SPEC="torch==2.9.1 torchvision==0.24.1"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
export NUMPY_GENERATION="2"
exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_common.sh"
