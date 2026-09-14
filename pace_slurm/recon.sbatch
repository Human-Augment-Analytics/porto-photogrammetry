#!/bin/bash
# Gaussian reconstruction for every scene under DATA_ROOT, one array task per scene.
# Usage: BACKEND=2dgs SFM=vggt sbatch pace_slurm/recon.sbatch [extra flags]
# Single scene: sbatch --array=N pace_slurm/recon.sbatch   (N = index in the sorted scene list)
# BACKEND selects the registered reconstruction backend (2dgs|sugar|pgsr|gw).
# SFM selects which SfM result feeds it: results/<scene>/all/<SFM>/.
#SBATCH --job-name=recon
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64gb
#SBATCH --time=3:00:00
#SBATCH --array=0-5
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

BACKEND="${BACKEND:-2dgs}"
case "$BACKEND" in
    2dgs|sugar|pgsr|gw) ;;
    *) echo "ERROR: unknown BACKEND='$BACKEND' (valid: 2dgs, sugar, pgsr, gw)" >&2; exit 2 ;;
esac

SFM="${SFM:-vggt}"


discover_scenes
select_scene
# Reconstruction consumes an SfM result, not the prepared dataset.
SCENE="$RESULT_ROOT/$SCENE_NAME/all/$SFM"
OUT="$RESULT_ROOT/$SCENE_NAME/all/$SFM-$BACKEND"

echo "scene      : $SCENE_NAME ($SCENE_INDEX of ${#SCENES[@]})"
echo "backend    : $BACKEND  <- sfm: $SFM"
echo "input      : $SCENE"
echo "output     : $OUT"

# A missing or empty sparse/0 means that scene's SfM never produced a model.
if [ ! -d "$SCENE/sparse/0" ] || [ -z "$(ls -A "$SCENE/sparse/0" 2>/dev/null)" ]; then
    echo "ERROR: no SfM model at $SCENE/sparse/0 - run the $SFM SfM job for this scene first" >&2
    exit 2
fi

mkdir -p "$OUT"

# The registry resolves BACKEND; an unknown name fails listing the valid set.
augenblick recon "$BACKEND" --scene "$SCENE" --output "$OUT" "$@"
