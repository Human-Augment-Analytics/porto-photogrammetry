#!/bin/bash
# Builds the augenblick_<gpu> conda env on a node of that GPU, then runs scripts/setup_<gpu>.sh.
# Valid GPU: l40s | a100 | h200 | a40; the gres must name the same card (see README.md).
# RECREATE=1 deletes an existing env first; otherwise it is reused and re-installed over.
#SBATCH --job-name=augenblick-setup
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=10gb
#SBATCH --time=2:00:00
#SBATCH --output=pace_slurm/logs/%x-%j.out
#SBATCH --error=pace_slurm/logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=srizvi63@gatech.edu

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
if [ ! -f "$REPO_ROOT/scripts/setup_common.sh" ]; then
    echo "ERROR: no scripts/setup_common.sh under $REPO_ROOT; submit from the repo root" >&2
    exit 2
fi
cd "$REPO_ROOT"

GPU="${GPU:-l40s}"
# Wrapper names do not always match the GPU (H200 uses setup_h100.sh, same sm_9.0).
case "$GPU" in
    l40s) WANT_CAP="8.9"; SETUP_SCRIPT="setup_l40s.sh" ;;
    a100) WANT_CAP="8.0"; SETUP_SCRIPT="setup_a100.sh" ;;
    a40)  WANT_CAP="8.6"; SETUP_SCRIPT="setup_a40.sh"  ;;
    h200) WANT_CAP="9.0"; SETUP_SCRIPT="setup_h100.sh" ;;
    h100) WANT_CAP="9.0"; SETUP_SCRIPT="setup_h100.sh" ;;
    # Blackwell: needs --partition=ice-bw-gpu at submit time.
    rtx_pro_6000) WANT_CAP="12.0"; SETUP_SCRIPT="setup_rtx_pro_6000.sh" ;;
    *) echo "ERROR: unknown GPU='$GPU' (valid: l40s, a100, h100, h200, a40, rtx_pro_6000)" >&2; exit 2 ;;
esac
if [ ! -f "$REPO_ROOT/scripts/$SETUP_SCRIPT" ]; then
    echo "ERROR: no scripts/$SETUP_SCRIPT for GPU=$GPU" >&2
    exit 2
fi

CONDA_ROOT="${CONDA_ROOT:-$HOME/scratch/conda}"
CONDA_ENV="$CONDA_ROOT/augenblick_$GPU"
CONDA_SH="${CONDA_SH:-/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh}"

# Does NOT source common.sh: that exits 2 on a missing env, the case this script fixes.
module purge
if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: no conda.sh at $CONDA_SH (override with CONDA_SH=...)" >&2
    exit 2
fi
source "$CONDA_SH"

echo "=========================================================="
echo "job        : ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-no-jobid})"
echo "node       : $(hostname)"
echo "started    : $(date -Is)"
echo "gpu target : $GPU"
echo "env        : $CONDA_ENV"
echo "repo       : $REPO_ROOT ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
nvidia-smi -L 2>/dev/null || echo "gpus       : none visible"
echo "=========================================================="

# Building on the wrong card survives setup and only fails at kernel launch. Refuse it.
DETECTED_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
if [ -z "$DETECTED_CAP" ]; then
    echo "ERROR: no GPU visible - submit with --gres=gpu:$GPU:1" >&2
    exit 2
fi
if [ "$DETECTED_CAP" != "$WANT_CAP" ]; then
    echo "ERROR: GPU=$GPU expects compute capability $WANT_CAP but this node reports $DETECTED_CAP" >&2
    echo "       submit with --gres=gpu:$GPU:1 so the CUDA builds match the card" >&2
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
echo "next       : GPU=$GPU sbatch --gres=gpu:$GPU:1 pace_slurm/colmap_sfm.sbatch"
echo "=========================================================="
