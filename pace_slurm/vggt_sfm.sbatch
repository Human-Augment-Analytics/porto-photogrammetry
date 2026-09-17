#!/bin/bash
# VGGT -> COLMAP for every scene under DATA_ROOT, one array task per scene.
# Usage: sbatch pace_slurm/vggt_sfm.sbatch [extra flags]
# Single scene: sbatch --array=N pace_slurm/vggt_sfm.sbatch   (N = index in the sorted scene list)
# Large scenes want >=80GB VRAM; the a100 gres here is the 80GB PCIe model.
#SBATCH --job-name=vggt-sfm
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=96gb
#SBATCH --time=4:00:00
#SBATCH --array=0-46%8
#SBATCH --output=pace_slurm/logs/%x-%A_%a.out
#SBATCH --error=pace_slurm/logs/%x-%A_%a.err
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


discover_scenes
select_scene
OUT="$RESULT_ROOT/$SCENE_NAME/vggt"
mkdir -p "$OUT"

# Records elapsed time (and any OOM) for this scene, even if the job dies.
start_timing "$SCENE_NAME" "$(ls -1 "$SCENE/images" 2>/dev/null | wc -l)"

echo "scene      : $SCENE_NAME ($SCENE_INDEX of ${#SCENES[@]}) -> $OUT"

augenblick sfm vggt --scene "$SCENE" --output "$OUT" "$@"
