#!/bin/bash
# Flatten every raw NeurIPS object into a COLMAP-ready <scene>/images + <scene>/masks pair.
# Usage: sbatch hpg_slurm/prepare_neurips.sbatch [extra prepare_uf_dataset.py flags]
# Single object: sbatch --array=N hpg_slurm/prepare_neurips.sbatch  (N = index in the sorted object list)
# CPU-only: symlinking touches no GPU, so this asks for none.
#SBATCH --job-name=prepare-neurips
#SBATCH --account=arthur.porto
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
#SBATCH --time=0:30:00
#SBATCH --array=0-75
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

NEURIPS_ROOT="${NEURIPS_ROOT:-/blue/arthur.porto/data/datasets/photogrammetry/neurips}"
RAW_ROOT="${RAW_ROOT:-$NEURIPS_ROOT/raw}"
PREPARED_ROOT="${PREPARED_ROOT:-$NEURIPS_ROOT/prepared}"

if [ ! -d "$RAW_ROOT" ]; then
    echo "ERROR: raw root does not exist: $RAW_ROOT" >&2
    exit 2
fi

# Sorted so an array index maps to the same object across submissions.
mapfile -t OBJECTS < <(find "$RAW_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

if [ "${#OBJECTS[@]}" -eq 0 ]; then
    echo "ERROR: no object directories under $RAW_ROOT" >&2
    exit 2
fi

IDX="${SLURM_ARRAY_TASK_ID:-0}"
if [ "$IDX" -ge "${#OBJECTS[@]}" ]; then
    echo "ERROR: array index $IDX exceeds ${#OBJECTS[@]} object(s); resize --array" >&2
    exit 2
fi

OBJECT="${OBJECTS[$IDX]}"
SRC="$RAW_ROOT/$OBJECT"
OUT="$PREPARED_ROOT/$OBJECT"

echo "object     : $OBJECT ($((IDX + 1)) of ${#OBJECTS[@]})"
echo "source     : $SRC"
echo "output     : $OUT"

# Symlinks point at the raw tree, so a stale run leaves dangling links behind a
# re-flatten; clear the scene first to keep images/ and masks/ consistent.
rm -rf "$OUT"
mkdir -p "$OUT"

python pipeline/preparation/prepare_uf_dataset.py "$SRC" \
    --out "$OUT" \
    --mode symlink \
    --include-unmatched \
    "$@"

# Masks are optional downstream, but an empty images/ means the scene is unusable.
N_IMAGES=$(find "$OUT/images" -type l 2>/dev/null | wc -l)
N_MASKS=$(find "$OUT/masks" -type l 2>/dev/null | wc -l)
echo "result     : $N_IMAGES image(s), $N_MASKS mask(s) in $OUT"

if [ "$N_IMAGES" -eq 0 ]; then
    echo "ERROR: no images written for $OBJECT" >&2
    exit 1
fi

if [ "$N_MASKS" -eq 0 ]; then
    echo "WARNING: $OBJECT has no masks; downstream SfM runs unmasked"
fi
