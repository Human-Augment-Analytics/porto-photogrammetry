#!/bin/bash
# COLMAP -> turntable rig refinement for every scene under DATA_ROOT, one array task per scene.
# Turntable refines an existing COLMAP scene, so the input SfM is run first if it is absent.
# Usage: sbatch hpg_slurm/turntable_sfm.sbatch [extra turntable flags]
# Single scene: sbatch --array=N hpg_slurm/turntable_sfm.sbatch   (N = index in the sorted scene list)
# SFM selects which SfM result is refined: results/<scene>/all/<SFM>/ (default colmap).
#SBATCH --job-name=turntable-sfm
#SBATCH --account=arthur.porto
#SBATCH --partition=hpg-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=24gb
#SBATCH --time=2:00:00
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

SFM="${SFM:-colmap}"
case "$SFM" in
    colmap|vggt|vggt_ba) ;;
    *) echo "ERROR: unknown SFM='$SFM' (valid: colmap, vggt, vggt_ba)" >&2; exit 2 ;;
esac

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
PREPARED="$DATA_ROOT/$SCENE_NAME/prepared"
SFM_OUT="$RESULT_ROOT/$SCENE_NAME/all/$SFM"
OUT="$RESULT_ROOT/$SCENE_NAME/all/$SFM-turntable"

echo "scene      : $SCENE_NAME ($IDX of ${#SCENES[@]})"
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
