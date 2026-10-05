#!/bin/bash
# Turntable rig refinement, one array task per scene. It refines an existing COLMAP
# scene, so the input SfM runs first when absent.
# Usage: sbatch hpg_slurm/turntable_sfm.sbatch [extra turntable flags]
# One scene: sbatch --array=N ...   (N indexes the sorted scene list)
# SFM= picks the input: <scene>/<SFM>/ (default colmap).
#SBATCH --job-name=turntable-sfm
#SBATCH --account=arthur.porto
#SBATCH --partition=hpg-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=24gb
#SBATCH --time=2:00:00
#SBATCH --array=0-75
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

SFM="${SFM:-colmap}"
case "$SFM" in
    colmap|vggt|vggt_ba) ;;
    *) echo "ERROR: unknown SFM='$SFM' (valid: colmap, vggt, vggt_ba)" >&2; exit 2 ;;
esac

discover_scenes
select_scene
PREPARED="$SCENE"
SFM_OUT="$RESULT_ROOT/$SCENE_NAME/$SFM"
OUT="$RESULT_ROOT/$SCENE_NAME/$SFM-turntable"

echo "scene      : $SCENE_NAME ($SCENE_INDEX of ${#SCENES[@]})"
echo "input sfm  : $SFM -> $SFM_OUT"
echo "output     : $OUT"

# Turntable refines an existing reconstruction; build the input SfM first if it is missing.
# An empty sparse/0 counts as missing: it means that SfM run failed partway.
if [ -d "$SFM_OUT/sparse/0" ] && [ -n "$(ls -A "$SFM_OUT/sparse/0" 2>/dev/null)" ]; then
    echo "input sfm  : found, reusing"
else
    echo "input sfm  : absent - running $SFM first"
    if [ ! -d "$PREPARED/images" ]; then
        echo "ERROR: no images/ at $PREPARED - prepare this scene first" >&2
        exit 2
    fi
    mkdir -p "$SFM_OUT"
    case "$SFM" in
        colmap)  augenblick sfm colmap --scene "$PREPARED" --output "$SFM_OUT" ;;
        vggt)    augenblick sfm vggt   --scene "$PREPARED" --output "$SFM_OUT" ;;
        vggt_ba) augenblick sfm vggt   --scene "$PREPARED" --output "$SFM_OUT" --use_ba ;;
    esac
    echo "input sfm  : $SFM completed"
fi

mkdir -p "$OUT"

# Extra flags land on turntable, not on the input SfM above.
augenblick sfm turntable --scene "$SFM_OUT" --output "$OUT" "$@"
