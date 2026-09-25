#!/bin/bash
# Full-resolution photo UV-atlas bake vs native texture vs ground truth, for ONE method's mesh.
# Renders the mesh's native texture and a full-res photo re-bake at the held-out cameras and
# scores both against the ground-truth photos, into:
#   <OUTPUT>/nvs_metrics_mesh_native.json     (skipped if the mesh has no native texture)
#   <OUTPUT>/nvs_metrics_mesh_photofull.json
# Cameras come from the shared COLMAP scene, so the split matches the NVS eval for every method.
#
# Usage (defaults target the SuGaR vanilla checkpoint as the shared camera provider):
#   SCENE=output/UF_Fish_245298/all/colmap \
#   MESH=output/UF_Fish_245298/all/colmap-2dgs/train/ours_30000/fuse_post.ply \
#   OUTPUT=output/UF_Fish_245298/all/colmap-2dgs \
#   sbatch pace_slurm/photobake_eval.sbatch
#SBATCH --job-name=photobake-eval
#SBATCH --partition=coc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96gb
#SBATCH --time=2:00:00
#SBATCH --output=pace_slurm/logs/%x-%j.out
#SBATCH --error=pace_slurm/logs/%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=okshatriya3@gatech.edu

set -euo pipefail

SLURM_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/pace_slurm"
if [ ! -f "$SLURM_DIR/common.sh" ]; then
    echo "ERROR: cannot find common.sh at $SLURM_DIR; submit from the repo root" >&2
    exit 2
fi
source "$SLURM_DIR/common.sh"

: "${SCENE:?set SCENE to the COLMAP scene (images/, masks/, split.json)}"
: "${MESH:?set MESH to the backend's extracted mesh (.obj or .ply)}"
: "${OUTPUT:?set OUTPUT to the backend's model directory}"

# The vanilla 3DGS checkpoint is only a camera provider; all backends share this scene/split,
# so the SuGaR vanilla model is a convenient default source of the held-out cameras.
VANILLA_CHECKPOINT="${VANILLA_CHECKPOINT:-output/UF_Fish_245298/all/colmap-sugar/gs_model}"
ITERATION_TO_LOAD="${ITERATION_TO_LOAD:-7000}"
PHOTO_MAX_SIZE="${PHOTO_MAX_SIZE:--1}"
SQUARE_SIZE="${SQUARE_SIZE:-10}"

PYTHONPATH=src python src/libs/sugar/photobake_mesh.py \
    --scene "$SCENE" \
    --vanilla-checkpoint "$VANILLA_CHECKPOINT" \
    --iteration-to-load "$ITERATION_TO_LOAD" \
    --mesh "$MESH" \
    --output "$OUTPUT" \
    --square-size "$SQUARE_SIZE" \
    --photo-max-size "$PHOTO_MAX_SIZE"

echo "PHOTOBAKE_EVAL_DONE native -> $OUTPUT/nvs_metrics_mesh_native.json  photofull -> $OUTPUT/nvs_metrics_mesh_photofull.json"
