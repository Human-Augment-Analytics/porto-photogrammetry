#!/bin/bash
# Geometric evaluation of one mesh against a metric reference. CPU-only.
#   OURS=.../fuse_post.ply REF=.../gt.obj OUT=.../eval \
#     [SCALE_JSON=.../scale_result.json] sbatch hpg_slurm/geometry_eval.sbatch
# Always scores shape-only (similarity alignment). With SCALE_JSON from an accepted
# scale_markers run it additionally scores rigidly at that metric scale, so any
# residual size error stays in the metrics instead of being fitted away.
#SBATCH --job-name=geometry-eval
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

: "${OURS:?mesh to evaluate, in the gauge of its own reconstruction}"
: "${REF:?metric reference mesh (millimetres)}"
: "${OUT:?output directory for the score JSONs}"

mkdir -p "$OUT"

if [ -f "$OUT/similarity_scores.json" ]; then
    echo "similarity scores exist, skipping"
else
    python -m augenblick.eval.mesh \
        --ours "$OURS" --reference "$REF" \
        --samples 100000 --seeds 0 1 2 \
        --label similarity \
        --json-out "$OUT/similarity_scores.json"
fi

if [ -n "${SCALE_JSON:-}" ]; then
    SCALE=$(python -c "import json, sys; r = json.load(open(sys.argv[1])); sys.exit('scale was not accepted; rigid scoring would be meaningless') if not r.get('accepted') else print(r['scale_mm_per_model_unit'])" "$SCALE_JSON")
    if [ -f "$OUT/rigid_scores.json" ]; then
        echo "rigid scores exist, skipping"
    else
        python -m augenblick.eval.mesh \
            --ours "$OURS" --reference "$REF" \
            --scale-ours "$SCALE" --rigid \
            --samples 100000 --seeds 0 1 2 \
            --absolute-tolerances 0.25 0.5 1.0 \
            --label rigid \
            --json-out "$OUT/rigid_scores.json"
    fi
fi

echo "scores in: $OUT"
