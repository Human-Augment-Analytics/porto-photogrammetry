#!/bin/bash
# Copy this file, edit the command block at the bottom, then: sbatch hpg_slurm/<yours>.sbatch
# NOTE: GPU=<name> picks the conda env and arch but NOT the partition; change both together.
#SBATCH --job-name=augenblick
#SBATCH --account=arthur.porto
#SBATCH --partition=hpg-rtx6000
##SBATCH --partition=hpg-b200          # alternative; pair with GPU=b200
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

# Own set -e: common.sh's is moot if the source itself fails.
set -euo pipefail

# Slurm runs this from /var/spool, so $0 cannot find the repo; use the submit dir.
SLURM_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/hpg_slurm"
if [ ! -f "$SLURM_DIR/common.sh" ]; then
    echo "ERROR: cannot find common.sh at $SLURM_DIR; submit from the repo root" >&2
    exit 2
fi
source "$SLURM_DIR/common.sh"

# DATA_ROOT/RESULT_ROOT come from common.sh, as do discover_scenes/select_scene if you
# turn this into an array job (see vggt_sfm.sbatch).
SCENE="$DATA_ROOT/CHANGE_ME"
OUT="$RESULT_ROOT/CHANGE_ME"

# ================= EDIT BELOW: your command =================
augenblick sfm vggt --scene "$SCENE" --output "$OUT"
# ============================================================
