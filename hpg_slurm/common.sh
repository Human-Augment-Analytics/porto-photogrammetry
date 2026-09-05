# Sourced by every sbatch script here: selects GPU, loads modules, activates conda.
# Not executable on its own. Set GPU=rtx6000|b200 before submitting to switch targets.

set -euo pipefail

CONDA_ROOT="/blue/arthur.porto/srizvi63.gatech/conda"
GPU="${GPU:-rtx6000}"

# Arch strings mirror the GPU_ARCH each scripts/setup_<gpu>.sh exports.
case "$GPU" in
    rtx6000)
        CONDA_ENV="$CONDA_ROOT/augenblick_rtx_pro_6000"
        CUDA_MODULE="cuda/13.0.2"
        GPU_ARCH="12.0"
        ;;
    b200)
        CONDA_ENV="$CONDA_ROOT/augenblick_b200"
        CUDA_MODULE="cuda/12.8"
        GPU_ARCH="10.0"
        ;;
    *)
        echo "ERROR: unknown GPU='$GPU' (valid: rtx6000, b200)" >&2
        exit 2
        ;;
esac

# A batch shell does not source ~/.bashrc, so load the toolchain explicitly.
# colmap is deliberately not loaded: the Python SfM path uses pycolmap from the env.
module purge
module load "$CUDA_MODULE"
module load conda/25.7.0
module load cmake/3.30.5
module load xerces/3.1.4
module load gcc/12.2.0 yasm/1.3.0

# A bare `conda activate` fails in a non-interactive shell without this.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Provenance banner so a log explains itself when revisited later.
echo "=========================================================="
echo "job        : ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-no-jobid})"
echo "node       : $(hostname)"
echo "started    : $(date -Is)"
echo "gpu target : $GPU (sm_$GPU_ARCH, $CUDA_MODULE)"
echo "conda env  : $CONDA_ENV"
echo "python     : $(python --version 2>&1) @ $(command -v python)"
echo "repo       : $REPO_ROOT ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
nvidia-smi -L 2>/dev/null || echo "gpus       : none visible"
echo "=========================================================="
