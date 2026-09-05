#!/bin/bash
# Copy this file, edit the command block at the bottom, then: sbatch pace_slurm/<yours>.sbatch
# NOTE: no --account line, ICE assigns the account (coc) itself.
#SBATCH --job-name=augenblick
#SBATCH --partition=ice-gpu
##SBATCH --partition=ice-bw-gpu        # alternative; 18h limit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64gb
#SBATCH --time=4:00:00
#SBATCH --output=pace_slurm/logs/%x-%j.out
#SBATCH --error=pace_slurm/logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=srizvi63@gatech.edu

# Abort on error here too: common.sh's own set -e is moot if sourcing fails.
set -euo pipefail

# Slurm copies this script to /var/spool, so $0 cannot locate the repo; use the submit dir.
SLURM_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/pace_slurm"
if [ ! -f "$SLURM_DIR/common.sh" ]; then
    echo "ERROR: cannot find common.sh at $SLURM_DIR; submit from the repo root" >&2
    exit 2
fi
source "$SLURM_DIR/common.sh"

SCENE="data/CHANGE_ME"
OUT="output/CHANGE_ME"

# ================= EDIT BELOW: your command =================
augenblick sfm vggt --scene "$SCENE" --output "$OUT"
# ============================================================
