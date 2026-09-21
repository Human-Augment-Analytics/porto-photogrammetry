# Sourced by every sbatch script except the Meshroom baseline: GPU switch, modules, model
# caches, conda activate, then scene_common.sh. Not executable on its own.
# Set GPU=rtx6000|b200 before submitting to switch targets.

set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/blue/arthur.porto/srizvi63.gatech/conda}"
GPU="${GPU:-rtx6000}"

# GPU_ARCH mirrors scripts/setup_<gpu>.sh; GPU_PARTITION is the partition carrying the
# card, since HPG selects with --gpus=1 plus a partition rather than a gres name.
case "$GPU" in
    rtx6000)
        CONDA_ENV="$CONDA_ROOT/augenblick_rtx_pro_6000"
        CUDA_MODULE="cuda/13.0.2"
        GPU_ARCH="12.0"
        GPU_PARTITION="hpg-rtx6000"
        ;;
    b200)
        CONDA_ENV="$CONDA_ROOT/augenblick_b200"
        CUDA_MODULE="cuda/12.8"
        GPU_ARCH="10.0"
        GPU_PARTITION="hpg-b200"
        ;;
    *)
        echo "ERROR: unknown GPU='$GPU' (valid: rtx6000, b200)" >&2
        exit 2
        ;;
esac

# A batch shell does not source ~/.bashrc, so load the toolchain explicitly.
module purge
module load "$CUDA_MODULE"
module load conda/25.7.0
module load cmake/3.30.5
module load xerces/3.1.4
module load gcc/12.2.0 yasm/1.3.0

# Same reason: ~/.bashrc's model caches are not inherited. Without these, every
# checkpoint-fetching job (VGGT ~4 GB, SAM 3 ~2 GB) re-downloads into $HOME/.cache,
# a small quota rather than the /blue allocation.
export HF_HOME="${HF_HOME:-/blue/arthur.porto/srizvi63.gatech/cache/huggingface/}"
export TORCH_HOME="${TORCH_HOME:-/blue/arthur.porto/srizvi63.gatech/cache/torch/}"

if [ ! -d "$CONDA_ENV" ]; then
    echo "ERROR: no conda env at $CONDA_ENV" >&2
    echo "       build it with: sbatch hpg_slurm/setup_env.sbatch  (GPU=$GPU)" >&2
    exit 2
fi

# A bare `conda activate` fails in a non-interactive shell without this.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"

# torch preloads its own CUDA libs, but rembg imports onnxruntime without torch: without
# these dirs ORT finds no libcudnn/libcufft and silently drops to CPU at ~4x the cost.
NVIDIA_LIB_ROOT="$(python -c 'import site;print(site.getsitepackages()[0])' 2>/dev/null)/nvidia"
if [ -d "$NVIDIA_LIB_ROOT" ]; then
    for _libdir in "$NVIDIA_LIB_ROOT"/*/lib; do
        [ -d "$_libdir" ] && LD_LIBRARY_PATH="$_libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
    export LD_LIBRARY_PATH
    unset _libdir
fi

# scene_common.sh reads GPU, CONDA_ENV, GPU_PARTITION; BANNER_EXTRA adds the CUDA target.
BANNER_EXTRA="gpu target : $GPU (sm_$GPU_ARCH, $CUDA_MODULE, $GPU_PARTITION)"

source "$(dirname "${BASH_SOURCE[0]}")/scene_common.sh"
