#!/bin/bash
# Meshroom (AliceVision) baseline, one array task per scene.
# %1 is deliberate: see "Meshroom baseline" in README.md.
#SBATCH --job-name=meshroom
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64gb
#SBATCH --time=6:00:00
#SBATCH --array=0-46%1
#SBATCH --output=pace_slurm/logs/%x-%A_%a.out
#SBATCH --error=pace_slurm/logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=srizvi63@gatech.edu

set -euo pipefail

SLURM_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/pace_slurm"
if [ ! -f "$SLURM_DIR/meshroom_common.sh" ]; then
    echo "ERROR: cannot find meshroom_common.sh at $SLURM_DIR; submit from the repo root" >&2
    exit 2
fi
source "$SLURM_DIR/meshroom_common.sh"

discover_scenes
select_scene
OUT="$RESULT_ROOT/$SCENE_NAME/meshroom"
mkdir -p "$OUT"

start_timing "$SCENE_NAME" "$(ls -1 "$SCENE/images" 2>/dev/null | wc -l)"

echo "scene      : $SCENE_NAME ($SCENE_INDEX of ${#SCENES[@]}) -> $OUT"
if [ -d "$SCENE/masks" ]; then
    echo "masks      : $(ls -1 "$SCENE/masks" | wc -l) file(s)"
else
    echo "masks      : none - running unmasked"
fi

# CPU-bound outside DepthMap, so match threads to the allocation.
MAX_THREADS="${MAX_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"

python baseline/benchmark_meshroom.py \
    "$SCENE" \
    "$OUT" \
    --save_file "$OUT/pipeline.mg" \
    --max_threads "$MAX_THREADS" \
    "$@"
