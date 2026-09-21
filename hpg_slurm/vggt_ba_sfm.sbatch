#!/bin/bash
# VGGT + bundle adjustment -> COLMAP for every scene under DATA_ROOT, one array task per scene.
# Writes <scene>/vggt_ba, so it does not collide with the non-BA job's <scene>/vggt.
# Usage: sbatch hpg_slurm/vggt_ba_sfm.sbatch [extra flags]
# One scene: sbatch --array=N ...   (N indexes the sorted scene list)
# BA needs at least as much VRAM again as plain VGGT; prefer GPU=b200.
#SBATCH --job-name=vggt-ba-sfm
#SBATCH --account=arthur.porto
#SBATCH --partition=hpg-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=96gb
#SBATCH --time=8:00:00
#SBATCH --array=0-75%8
#SBATCH --output=hpg_slurm/logs/%x-%A_%a.out
#SBATCH --error=hpg_slurm/logs/%x-%A_%a.err
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

discover_scenes
select_scene
OUT="$RESULT_ROOT/$SCENE_NAME/vggt_ba"
mkdir -p "$OUT"

N_IMAGES=$(ls -1 "$SCENE/images" 2>/dev/null | wc -l)
N_MASKS=$(ls -1 "$SCENE/masks" 2>/dev/null | wc -l)

echo "scene      : $SCENE_NAME ($SCENE_INDEX of ${#SCENES[@]}) -> $OUT"
echo "images     : $N_IMAGES"

# Machine-readable marker for log scraping; keep the prefix and fields stable.
echo "SCENE_INFO scene=$SCENE_NAME sfm=vggt_ba images=$N_IMAGES masks=$N_MASKS"

# One CSV row per scene, written even if the job dies.
start_timing "$SCENE_NAME" "$N_IMAGES" vggt_ba

augenblick sfm vggt --scene "$SCENE" --output "$OUT" --use_ba "$@"
