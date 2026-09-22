#!/bin/bash
# Masking stage, one array task per scene. Emits a scene-shaped dir (images/ symlink +
# masks/) that sfm/recon consume directly.
# Usage: MASK_METHOD=rembg|threshold|sam3 sbatch hpg_slurm/mask.sbatch [extra flags]
# One scene: sbatch --array=N ...   (N indexes the sorted scene list)
# Timing rows take the method from the JOB NAME, not MASK_METHOD, so to keep two methods
# apart in one CSV: MASK_METHOD=threshold sbatch --job-name=mask-threshold ...
# Small model, I/O-bound: the default partition is plenty.
#SBATCH --job-name=mask
#SBATCH --account=arthur.porto
#SBATCH --partition=hpg-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32gb
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

MASK_METHOD="${MASK_METHOD:-rembg}"
case "$MASK_METHOD" in
    rembg|threshold|sam3) ;;
    *) echo "ERROR: unknown MASK_METHOD='$MASK_METHOD' (valid: rembg, threshold, sam3)" >&2; exit 2 ;;
esac

# ORT lists CUDAExecutionProvider even when the .so fails to dlopen, then runs on CPU at
# ~4x the cost — and --only_missing would cache those slow masks for every later run.
# Advertised != loadable, so fail the job rather than burn hours on a silent fallback.
if [ "$MASK_METHOD" = "rembg" ]; then
    python - <<'PY'
from rembg import new_session
providers = new_session("isnet-general-use").inner_session.get_providers()
print("rembg session providers:", providers)
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("ERROR: rembg fell back to CPU; the env's CUDA activate hook is missing")
PY
fi

if [ "$MASK_METHOD" = "sam3" ]; then
    python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: no CUDA device; SAM 3 masking would run on CPU")
print("sam3 preflight: CUDA OK,", torch.cuda.get_device_name(0))
PY
fi

discover_scenes
select_scene

# This is the stage that produces masks, so it is the one job needing none in its input.
#   nested (default) : <method>/<scene>/  — the three methods coexist per scene
#   flat             : <scene>/           — one method per scene; a second OVERWRITES
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

# One CSV row per scene, written even if the job dies.
start_timing "$SCENE_NAME" "$N_IMAGES"

echo "scene      : $SCENE_NAME ($SCENE_INDEX of ${#SCENES[@]})"
echo "input      : $SCENE"
echo "output     : $OUT"
echo "method     : $MASK_METHOD"
echo "images     : $N_IMAGES"

# rembg fetches its model on first use; compute nodes have outbound network, so no
# login-node warm-up. --only_missing lets a requeued job resume where it stopped.
augenblick mask "$MASK_METHOD" --images "$SCENE/images" --output "$OUT" "$@"

if [ -z "$(ls -A "$OUT/masks" 2>/dev/null)" ]; then
    echo "ERROR: masking wrote no masks to $OUT/masks" >&2
    exit 2
fi
echo "masks      : $(ls -1 "$OUT/masks" | wc -l) of $N_IMAGES images"
