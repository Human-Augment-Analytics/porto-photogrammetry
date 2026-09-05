#!/bin/bash
# Detect the GPU (via nvidia-smi compute capability) and run the matching setup wrapper.
# Pass-through env vars (BACKENDS, SKIP_TETRA, ...) are forwarded.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Run this on a GPU node (or call a specific setup_<gpu>.sh)."
    exit 1
fi

CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Detected GPU: $NAME  (compute capability $CAP)"

case "$CAP" in
    8.9)  TARGET="setup_l40s.sh" ;;   # L40S / L40 / RTX 6000 Ada
    8.0)  TARGET="setup_a100.sh" ;;   # A100 (40 or 80 GB)
    8.6)  TARGET="setup_a40.sh"  ;;   # A40 (48 GB)
    9.0)  TARGET="setup_h100.sh" ;;   # H100 / H200
    10.0) TARGET="setup_b200.sh" ;;   # B200
    12.0) TARGET="setup_rtx_pro_6000.sh" ;;   # RTX Pro 6000 Blackwell
    *)
        cat <<EOF
No dedicated wrapper for compute capability '$CAP'.
Supported out of the box: 8.0 (A100), 8.6 (A40), 8.9 (L40S/L40), 9.0 (H100/H200), 10.0 (B200), 12.0 (RTX Pro 6000 Blackwell).

To set up this GPU, copy one of scripts/setup_<gpu>.sh and set:
  GPU_ARCH=$CAP   plus a matching CUDA_MODULE / TORCH_SPEC / TORCH_INDEX_URL.
Quick reference: V100=7.0.
Or run setup_common.sh directly with those variables exported.
EOF
        exit 2
        ;;
esac

echo "Dispatching to scripts/$TARGET"
exec bash "$SCRIPT_DIR/$TARGET"
