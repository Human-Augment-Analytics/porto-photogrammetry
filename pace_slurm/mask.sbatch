#!/bin/bash
# Masking stage for every scene under DATA_ROOT, one array task per scene.
# Produces a scene-shaped output (images/ symlink + masks/) that sfm/recon consume directly.
# Usage: MASK_METHOD=rembg sbatch pace_slurm/mask.sbatch [extra flags]
# Single scene: sbatch --array=N pace_slurm/mask.sbatch   (N = index in the sorted scene list)
# MASK_METHOD selects the registered masking method (rembg|threshold).
# Output goes to $RESULT_ROOT/<method>/<scene>/, so the two methods can coexist.
# Timing rows take the method from the job name, so to separate two methods in one CSV:
#   MASK_METHOD=threshold sbatch --job-name=mask-threshold pace_slurm/mask.sbatch
# U^2-Net is small and the stage is I/O-bound; any of a100/a40/l40s works.
#SBATCH --job-name=mask
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32gb
#SBATCH --time=2:00:00
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

MASK_METHOD="${MASK_METHOD:-rembg}"
case "$MASK_METHOD" in
    rembg|threshold) ;;
    *) echo "ERROR: unknown MASK_METHOD='$MASK_METHOD' (valid: rembg, threshold)" >&2; exit 2 ;;
esac

# A provider that is merely *advertised* is not a provider that loads: ORT lists
# CUDAExecutionProvider even when the .so fails to dlopen, then silently runs on CPU at ~4x
# the cost. Fail the job instead of burning hours on a silent fallback — and note that
# --only_missing would otherwise reuse those slow masks on every later run.
if [ "$MASK_METHOD" = "rembg" ]; then
    python - <<'PY'
from rembg import new_session
providers = new_session("isnet-general-use").inner_session.get_providers()
print("rembg session providers:", providers)
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("ERROR: rembg fell back to CPU; the env's CUDA-12 activate hook is missing")
PY
fi

discover_scenes
select_scene

# Masking reads images/ and writes its own scene, so unlike the SfM jobs it needs no masks/
# in the input — it is the stage that produces them.
#   nested (default) : <method>/<scene>/  — rembg and threshold coexist per scene
#   flat             : <scene>/                  — one method per scene; a second OVERWRITES
OUT_LAYOUT="${OUT_LAYOUT:-nested}"
case "$OUT_LAYOUT" in
    nested) OUT="$RESULT_ROOT/$MASK_METHOD/$SCENE_NAME" ;;
    flat)   OUT="$RESULT_ROOT/$SCENE_NAME" ;;
    *) echo "ERROR: unknown OUT_LAYOUT='$OUT_LAYOUT' (valid: nested, flat)" >&2; exit 2 ;;
esac
mkdir -p "$OUT"

if [ ! -d "$SCENE/images" ]; then
    echo "ERROR: no images/ at $SCENE - prepare this scene first" >&2
    exit 2
fi
N_IMAGES="$(ls -1 "$SCENE/images" 2>/dev/null | wc -l)"

# Records elapsed time (and any OOM) for this scene, even if the job dies.
start_timing "$SCENE_NAME" "$N_IMAGES"

echo "scene      : $SCENE_NAME ($SCENE_INDEX of ${#SCENES[@]})"
echo "input      : $SCENE"
echo "output     : $OUT"
echo "method     : $MASK_METHOD"
echo "images     : $N_IMAGES"

# rembg downloads its model to ~/.u2net/ on first use; the compute nodes have outbound
# network, so no login-node warm-up is needed. --only_missing makes a requeued job resume.
augenblick mask "$MASK_METHOD" --images "$SCENE/images" --output "$OUT" "$@"

if [ -z "$(ls -A "$OUT/masks" 2>/dev/null)" ]; then
    echo "ERROR: masking wrote no masks to $OUT/masks" >&2
    exit 2
fi
echo "masks      : $(ls -1 "$OUT/masks" | wc -l) of $N_IMAGES images"
