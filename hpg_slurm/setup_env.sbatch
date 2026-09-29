#!/bin/bash
# Builds augenblick_<card> on a node of that GPU, then runs scripts/setup_<gpu>.sh.
# GPU=rtx6000|b200; the partition must carry the same card (see README.md).
# RECREATE=1 deletes an existing env first; otherwise it is reused and reinstalled over.
#SBATCH --job-name=augenblick-setup
#SBATCH --account=arthur.porto
#SBATCH --partition=hpg-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64gb
#SBATCH --time=4:00:00
#SBATCH --output=hpg_slurm/logs/%x-%j.out
#SBATCH --error=hpg_slurm/logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=srizvi63@gatech.edu

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ ! -f "$REPO_ROOT/scripts/setup_common.sh" ]; then
    echo "ERROR: no scripts/setup_common.sh under $REPO_ROOT; submit from the repo root" >&2
    exit 2
fi
cd "$REPO_ROOT"

GPU="${GPU:-rtx6000}"
# The env name follows the card, not the GPU= alias: rtx6000 -> augenblick_rtx_pro_6000.
case "$GPU" in
    rtx6000) WANT_CAP="12.0"; SETUP_SCRIPT="setup_rtx_pro_6000.sh"; ENV_SUFFIX="rtx_pro_6000"; WANT_PARTITION="hpg-rtx6000" ;;
    b200)    WANT_CAP="10.0"; SETUP_SCRIPT="setup_b200.sh";         ENV_SUFFIX="b200";         WANT_PARTITION="hpg-b200" ;;
    *) echo "ERROR: unknown GPU='$GPU' (valid: rtx6000, b200)" >&2; exit 2 ;;
esac
if [ ! -f "$REPO_ROOT/scripts/$SETUP_SCRIPT" ]; then
    echo "ERROR: no scripts/$SETUP_SCRIPT for GPU=$GPU" >&2
    exit 2
fi

CONDA_ROOT="${CONDA_ROOT:-/blue/arthur.porto/srizvi63.gatech/conda}"
CONDA_ENV="$CONDA_ROOT/augenblick_$ENV_SUFFIX"

# Does NOT source common.sh: that exits 2 on a missing env, the case this script fixes.
module purge
module load conda/25.7.0
module load cmake/3.30.5
module load xerces/3.1.4
module load gcc/12.2.0 yasm/1.3.0
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=========================================================="
echo "job        : ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-no-jobid})"
echo "node       : $(hostname)"
echo "started    : $(date -Is)"
echo "gpu target : $GPU (sm_$WANT_CAP, $WANT_PARTITION)"
echo "env        : $CONDA_ENV"
echo "repo       : $REPO_ROOT ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
nvidia-smi -L 2>/dev/null || echo "gpus       : none visible"
echo "=========================================================="

# Building on the wrong card survives setup and fails only at kernel launch. Refuse it.
DETECTED_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
if [ -z "$DETECTED_CAP" ]; then
    echo "ERROR: no GPU visible - submit with --partition=$WANT_PARTITION --gpus=1" >&2
    exit 2
fi
if [ "$DETECTED_CAP" != "$WANT_CAP" ]; then
    echo "ERROR: GPU=$GPU expects compute capability $WANT_CAP but this node reports $DETECTED_CAP" >&2
    echo "       submit with --partition=$WANT_PARTITION so the CUDA builds match the card" >&2
    exit 2
fi
echo "arch check : sm_$DETECTED_CAP matches GPU=$GPU"

if [ "${RECREATE:-0}" = "1" ] && [ -d "$CONDA_ENV" ]; then
    echo "RECREATE=1 - removing $CONDA_ENV"
    conda env remove -p "$CONDA_ENV" -y || rm -rf "$CONDA_ENV"
fi

# setup_common.sh requires an already-active py3.10 env; it does not create one.
if [ -d "$CONDA_ENV" ]; then
    echo "env        : exists, reusing (RECREATE=1 to rebuild from scratch)"
else
    echo "env        : creating python=3.10"
    conda create -p "$CONDA_ENV" python=3.10 -y
fi
conda activate "$CONDA_ENV"
echo "python     : $(python --version 2>&1) @ $(command -v python)"

git submodule update --init --recursive

# Never run two of these against one checkout concurrently (see README.md).
bash "scripts/$SETUP_SCRIPT"

echo "=========================================================="
echo "setup done : $CONDA_ENV"
echo "finished   : $(date -Is)"
echo "next       : GPU=$GPU sbatch --partition=$WANT_PARTITION hpg_slurm/colmap_sfm.sbatch"
echo "=========================================================="
