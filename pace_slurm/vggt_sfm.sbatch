#!/bin/bash
# VGGT -> COLMAP for every scene under DATA_ROOT, one array task per scene.
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

set -euo pipefail

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

start_timing "$SCENE_NAME" "$(ls -1 "$SCENE/images" 2>/dev/null | wc -l)"

echo "scene      : $SCENE_NAME ($SCENE_INDEX of ${#SCENES[@]}) -> $OUT"

augenblick sfm vggt --scene "$SCENE" --output "$OUT" "$@"
