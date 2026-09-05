#!/bin/bash
# VGGT -> COLMAP for every scene under DATA_ROOT, one array task per scene.
# Usage: sbatch hpg_slurm/vggt_sfm.sbatch [extra flags]
# Single scene: sbatch --array=N hpg_slurm/vggt_sfm.sbatch   (N = index in the sorted scene list)
# Large scenes want >=80GB VRAM; prefer GPU=b200 --partition=hpg-b200 for those.
#SBATCH --job-name=vggt-sfm
#SBATCH --account=arthur.porto
#SBATCH --partition=hpg-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=24gb
#SBATCH --time=1:00:00
#SBATCH --array=0-5
#SBATCH --output=hpg_slurm/logs/%x-%A_%a.out
#SBATCH --error=hpg_slurm/logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=srizvi63@gatech.edu

# Abort on error here too: common.sh's own set -e is moot if sourcing fails.
set -euo pipefail

# Slurm copies this script to /var/spool, so $0 cannot locate the repo; use the submit dir.
SLURM_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/hpg_slurm"
if [ ! -f "$SLURM_DIR/common.sh" ]; then
    echo "ERROR: cannot find common.sh at $SLURM_DIR; submit from the repo root" >&2
    exit 2
fi
source "$SLURM_DIR/common.sh"

DATA_ROOT="${DATA_ROOT:-/blue/arthur.porto/data/datasets/photogrammetry/main}"
RESULT_ROOT="${RESULT_ROOT:-/blue/arthur.porto/srizvi63.gatech/results}"

# Sorted so an array index maps to the same scene across submissions.
mapfile -t SCENES < <(find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -type d -name prepared \
    -printf '%h\n' | xargs -r -n1 basename | sort)

if [ "${#SCENES[@]}" -eq 0 ]; then
    echo "ERROR: no <scene>/prepared dirs under $DATA_ROOT" >&2
    exit 2
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#SCENES[@]}" ]; then
    echo "ERROR: array index $IDX exceeds ${#SCENES[@]} scene(s): ${SCENES[*]}" >&2
    exit 2
fi

SCENE_NAME="${SCENES[$IDX]}"
SCENE="$DATA_ROOT/$SCENE_NAME/prepared"
OUT="$RESULT_ROOT/$SCENE_NAME/all/vggt"
mkdir -p "$OUT"

echo "scene      : $SCENE_NAME ($IDX of ${#SCENES[@]}) -> $OUT"

augenblick sfm vggt --scene "$SCENE" --output "$OUT" "$@"
