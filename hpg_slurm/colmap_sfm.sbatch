#!/bin/bash
# Mask-aware COLMAP SfM for every scene under DATA_ROOT, one array task per scene.
# Usage: sbatch hpg_slurm/colmap_sfm.sbatch [extra flags]
# One scene: sbatch --array=N ...   (N indexes the sorted scene list)
# Drives pycolmap from the conda env; no colmap binary is needed.
#SBATCH --job-name=colmap-sfm
#SBATCH --account=arthur.porto
#SBATCH --partition=hpg-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64gb
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
OUT="$RESULT_ROOT/$SCENE_NAME/colmap"
mkdir -p "$OUT"

N_IMAGES=$(ls -1 "$SCENE/images" 2>/dev/null | wc -l)
N_MASKS=$(ls -1 "$SCENE/masks" 2>/dev/null | wc -l)

echo "scene      : $SCENE_NAME ($SCENE_INDEX of ${#SCENES[@]}) -> $OUT"
# The colmap SfM method falls back to unmasked SfM when masks/ is absent.
if [ "$N_MASKS" -gt 0 ]; then
    echo "masks      : $N_MASKS file(s)"
else
    echo "masks      : none - running unmasked"
fi

# Machine-readable marker for log scraping; keep the prefix and fields stable.
echo "SCENE_INFO scene=$SCENE_NAME sfm=colmap images=$N_IMAGES masks=$N_MASKS"

# One CSV row per scene, written even if the job dies.
start_timing "$SCENE_NAME" "$N_IMAGES" colmap

augenblick sfm colmap --scene "$SCENE" --output "$OUT" "$@"
