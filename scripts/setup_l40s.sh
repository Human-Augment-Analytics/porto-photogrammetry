#!/bin/bash
# Setup for NVIDIA Ada, compute capability 8.9: L40S, L40, RTX 6000 Ada.
# One wrapper serves all three - they share an arch, so the toolchain is identical.
set -euo pipefail
export GPU_LABEL="L40S"
export GPU_ARCH="8.9"
export CUDA_MODULE="${CUDA_MODULE:-cuda/13.0.1}"
export TORCH_SPEC="torch==2.9.1 torchvision==0.24.1"
export TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"
export NUMPY_GENERATION="2"
exec bash "$(dirname "${BASH_SOURCE[0]}")/setup_common.sh"
