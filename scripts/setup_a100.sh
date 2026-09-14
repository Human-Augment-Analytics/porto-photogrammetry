#!/bin/bash
# Setup for NVIDIA A100 (compute capability 8.0), both 40 GB and 80 GB.
# CUDA 12.9 toolkit / torch 2.9.1 (cu129 wheels) — one CUDA generation across every wrapper,
# so onnxruntime-gpu's CUDA 12 sonames are satisfied by torch's own bundled libs.
set -euo pipefail
export GPU_LABEL="A100"
export GPU_ARCH="8.0"
export CUDA_MODULE="${CUDA_MODULE:-cuda/12.9.1}"
export TORCH_SPEC="torch==2.9.1 torchvision==0.24.1"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu129"
export NUMPY_GENERATION="2"
exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_common.sh"
