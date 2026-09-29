# Sourced by every sbatch script except the Meshroom baseline: selects GPU, loads
# modules, exports the scratch model caches, activates conda, then sources scene_common.sh.
# Not executable on its own.
# Set GPU=a100|l40s|a40 before submitting to switch targets.

set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/scratch/conda}"
GPU="${GPU:-a100}"

# GPU_ARCH mirrors scripts/setup_<gpu>.sh; GRES_NAME is what --gres=gpu:<name>:1 wants.
case "$GPU" in
    a100)
        CONDA_ENV="$CONDA_ROOT/augenblick_a100"
        CUDA_MODULE="cuda/12.9.1"
        GPU_ARCH="8.0"
        GRES_NAME="a100"
        ;;
    l40s)
        CONDA_ENV="$CONDA_ROOT/augenblick_l40s"
        CUDA_MODULE="cuda/12.9.1"
        GPU_ARCH="8.9"
        GRES_NAME="l40s"
        ;;
    a40)
        CONDA_ENV="$CONDA_ROOT/augenblick_a40"
        CUDA_MODULE="cuda/12.9.1"
        GPU_ARCH="8.6"
        GRES_NAME="a40"
        ;;
    h200)
        CONDA_ENV="$CONDA_ROOT/augenblick_h200"
        CUDA_MODULE="cuda/12.9.1"
        GPU_ARCH="9.0"
        GRES_NAME="h200"
        ;;
    h100)
        CONDA_ENV="$CONDA_ROOT/augenblick_h100"
        CUDA_MODULE="cuda/12.9.1"
        GPU_ARCH="9.0"
        GRES_NAME="h100"
        ;;
    # Blackwell; the only card not on ice-gpu - add --partition=ice-bw-gpu.
    rtx_pro_6000)
        CONDA_ENV="$CONDA_ROOT/augenblick_rtx_pro_6000"
        CUDA_MODULE="cuda/12.9.1"
        GPU_ARCH="12.0"
        GRES_NAME="rtx_pro_6000_blackwell"
        ;;
    *)
        echo "ERROR: unknown GPU='$GPU' (valid: a100, l40s, a40, h100, h200, rtx_pro_6000)" >&2
        exit 2
        ;;
esac

# A batch shell does not source ~/.bashrc, so load the toolchain explicitly.
module purge
module load "$CUDA_MODULE"

# Same reason: the model caches set in ~/.bashrc are not inherited either, and home is capped
# at 30 GB. Without these, every job re-downloads its multi-GB checkpoint into $HOME/.cache.
export HF_HOME="${HF_HOME:-$HOME/scratch/huggingface/}"
export TORCH_HOME="${TORCH_HOME:-$HOME/scratch/torch/}"
CONDA_SH="${CONDA_SH:-/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh}"

if [ ! -d "$CONDA_ENV" ]; then
    echo "ERROR: no conda env at $CONDA_ENV" >&2
    echo "       build it with: bash scripts/setup_${GPU}.sh" >&2
    exit 2
fi

if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: no conda.sh at $CONDA_SH (override with CONDA_SH=...)" >&2
    exit 2
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"

NVIDIA_LIB_ROOT="$(python -c 'import site;print(site.getsitepackages()[0])' 2>/dev/null)/nvidia"
if [ -d "$NVIDIA_LIB_ROOT" ]; then
    for _libdir in "$NVIDIA_LIB_ROOT"/*/lib; do
        [ -d "$_libdir" ] && LD_LIBRARY_PATH="$_libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
    export LD_LIBRARY_PATH
    unset _libdir
fi

# scene_common.sh reads GPU and CONDA_ENV; BANNER_EXTRA adds the CUDA target.
BANNER_EXTRA="gpu target : $GPU (sm_$GPU_ARCH, $CUDA_MODULE)"

source "$(dirname "${BASH_SOURCE[0]}")/scene_common.sh"
