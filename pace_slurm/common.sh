# Sourced by every sbatch script here: selects GPU, loads modules, activates conda.
# Not executable on its own. Set GPU=a100|l40s|a40 before submitting to switch targets.
#
# PACE ICE variant. Differences from hpg_slurm/common.sh: conda lives in ~/scratch (home is
# capped at 30 GB), the account is coc-ice, and PACE has no xerces/yasm modules.

set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/scratch/conda}"
GPU="${GPU:-a100}"

# Arch strings mirror the GPU_ARCH each scripts/setup_<gpu>.sh exports.
# gres names come from `sinfo -p ice-gpu -o %G` and are what --gres=gpu:<name>:1 expects.
case "$GPU" in
    a100)
        CONDA_ENV="$CONDA_ROOT/augenblick_a100"
        CUDA_MODULE="cuda/13.0.1"
        GPU_ARCH="8.0"
        GRES_NAME="a100"
        ;;
    l40s)
        CONDA_ENV="$CONDA_ROOT/augenblick_l40s"
        CUDA_MODULE="cuda/13.0.1"
        GPU_ARCH="8.9"
        GRES_NAME="l40s"
        ;;
    a40)
        CONDA_ENV="$CONDA_ROOT/augenblick_a40"
        CUDA_MODULE="cuda/13.0.1"
        GPU_ARCH="8.6"
        GRES_NAME="a40"
        ;;
    *)
        echo "ERROR: unknown GPU='$GPU' (valid: a100, l40s, a40)" >&2
        exit 2
        ;;
esac

# A batch shell does not source ~/.bashrc, so load the toolchain explicitly.
# colmap is deliberately not loaded: the Python SfM path uses pycolmap from the env.
module purge
module load "$CUDA_MODULE"
CONDA_SH="${CONDA_SH:-/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh}"

if [ ! -d "$CONDA_ENV" ]; then
    echo "ERROR: no conda env at $CONDA_ENV" >&2
    echo "       build it with: bash scripts/setup_${GPU}.sh" >&2
    exit 2
fi

# A bare `conda activate` fails in a non-interactive shell without this.
if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: no conda.sh at $CONDA_SH (override with CONDA_SH=...)" >&2
    exit 2
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# On ICE both roots live inside the repo checkout on scratch, so they follow REPO_ROOT
# rather than being absolute like the HPG /blue/... paths. Override either to relocate.
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/main}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/output}"

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
