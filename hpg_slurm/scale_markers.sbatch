#!/bin/bash
# Marker detection + metric scale for one scene. CPU-only.
#   SCENE=.../scene MODEL=.../colmap/sparse/0 OUT=.../scale PRIMARY="1675 1419" \
#     LENGTH_MM=100 [CHECK="1181 1949"] [MESH=.../fuse_post.ply] \
#     PGT_TOOLKIT=.../pgt-toolkit sbatch hpg_slurm/scale_markers.sbatch
# Codes and printed lengths must be verified on a photograph first: the detector codes
# do not determine the bar length (see docs/bar-codebook.json). Without CHECK the
# best-effort estimate applies: pooled spans, dispersion instead of a verdict.
#SBATCH --job-name=scale-markers
#SBATCH --account=arthur.porto
#SBATCH --qos=arthur.porto
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32gb
#SBATCH --time=2:00:00
#SBATCH --output=hpg_slurm/logs/%x-%j.out
#SBATCH --error=hpg_slurm/logs/%x-%j.err

set -euo pipefail

SLURM_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/hpg_slurm"
if [ ! -f "$SLURM_DIR/common.sh" ]; then
    echo "ERROR: cannot find common.sh at $SLURM_DIR; submit from the repo root" >&2
    exit 2
fi
source "$SLURM_DIR/common.sh"

# Run the code of the repo this job was submitted from, not whatever augenblick
# version the conda env happens to have installed.
export PYTHONPATH="${SLURM_DIR%/hpg_slurm}/src${PYTHONPATH:+:$PYTHONPATH}"

: "${SCENE:?scene directory with images/ (and capture_manifest.json when positions matter)}"
: "${MODEL:?COLMAP sparse model directory (the gauge the scale is for)}"
: "${OUT:?output directory for detections.json and scale_result.json}"
: "${PRIMARY:?primary bar detector codes, space-separated, e.g. 1675 1419}"
: "${LENGTH_MM:?printed bar length in mm, verified on a photograph (see docs/bar-codebook.json)}"
: "${PGT_TOOLKIT:?path to a PGT-Toolkit checkout (no PyPI release exists)}"

# The toolkit imports svgwrite/cairosvg, which the training env does not carry.
python -c "import svgwrite, cairosvg" 2>/dev/null || {
    echo "ERROR: PGT-Toolkit needs svgwrite and cairosvg in this env:" >&2
    echo "       pip install svgwrite cairosvg" >&2
    exit 2
}

mkdir -p "$OUT"
DETECTIONS="$OUT/detections.json"
IMAGES="${IMAGES:-$SCENE/images}"

if [ -f "$DETECTIONS" ]; then
    echo "detections exist, skipping: $DETECTIONS"
else
    # Detection reads the unmasked photographs; masks hide the bars by design.
    # Point IMAGES at the bar-bearing position's directory to skip bar-free views.
    python -m augenblick.eval.detect \
        --images "$IMAGES" \
        --output "$DETECTIONS" \
        --toolkit "$PGT_TOOLKIT"
fi

SCALE_ARGS=(--model "$MODEL" --scene "$SCENE" --detections "$DETECTIONS"
            --primary $PRIMARY --known-length-mm "$LENGTH_MM"
            --json-out "$OUT/scale_result.json")
[ -n "${CHECK:-}" ] && SCALE_ARGS+=(--check $CHECK)
[ -n "${MESH:-}" ] && SCALE_ARGS+=(--apply-to "$MESH" --output "$OUT/mesh_mm.ply")
[ -n "${ALLOW_UNREGISTERED:-}" ] && SCALE_ARGS+=(--allow-unregistered)

# A rejected scale is a recorded outcome (exit 1 with the JSON written); anything
# that fails to produce the JSON is a real failure. Remove any stale result first so
# an old file can never pass for this run's evidence.
rm -f "$OUT/scale_result.json"
set +e
python -m augenblick.eval.scale "${SCALE_ARGS[@]}"
RC=$?
set -e
if [ ! -f "$OUT/scale_result.json" ]; then
    echo "ERROR: estimator exited $RC without writing scale_result.json" >&2
    exit 1
fi
echo "scale result: $OUT/scale_result.json (estimator exit $RC)"
