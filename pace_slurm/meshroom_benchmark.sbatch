#!/bin/bash
# Meshroom (AliceVision) photogrammetry baseline for every scene under DATA_ROOT,
# one array task per scene.
#
# Usage:        sbatch pace_slurm/meshroom_benchmark.sbatch [extra flags]
# Single scene: sbatch --array=N pace_slurm/meshroom_benchmark.sbatch
# Other GPU:    GPU=h200 sbatch --gres=gpu:h200:1 pace_slurm/meshroom_benchmark.sbatch
#
# Extra flags are forwarded to baseline/benchmark_meshroom.py (--depthmap_downscale,
# --texture_side, --describer_types, ...).
#
# Runs ONE scene at a time (--array=...%1). Meshroom's MeshroomCache reaches ~50 GB for a
# 663-image scene, against a 300 GB Lustre quota; running 8-wide exhausted it and killed
# the whole array. benchmark_meshroom.py prunes the cache to just the textured mesh after
# each successful scene, so only one scene's intermediates exist at a time.
#
# Sources meshroom_common.sh, not common.sh: this baseline uses the standalone
# $CONDA_ROOT/meshroom env and the prebuilt AliceVision tarball, loads no CUDA module,
# and needs the ALICEVISION_* exports that ~/.bashrc holds but a batch job never sources.
#
# Supported GPUs: v100, rtx_6000, a100, a40, l40s, h100, h200. AliceVision 3.3.0's
# prebuilt kernels are SASS-only up to sm_90 with no PTX, so rtx_pro_6000_blackwell
# (sm_120) cannot JIT and mi210 is AMD; meshroom_common.sh rejects both up front.
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

# Abort on error here too: meshroom_common.sh's own set -e is moot if sourcing fails.
set -euo pipefail

# Slurm copies this script to /var/spool, so $0 cannot locate the repo; use the submit dir.
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

# Records elapsed time (and any OOM or unsupported-arch death) for this scene, even if
# the job is killed. Same CSV as the SfM jobs, so the rows compare directly.
start_timing "$SCENE_NAME" "$(ls -1 "$SCENE/images" 2>/dev/null | wc -l)"

echo "scene      : $SCENE_NAME ($SCENE_INDEX of ${#SCENES[@]}) -> $OUT"
# benchmark_meshroom.py passes masks/ to FeatureExtraction only when the dir exists.
if [ -d "$SCENE/masks" ]; then
    echo "masks      : $(ls -1 "$SCENE/masks" | wc -l) file(s)"
else
    echo "masks      : none - running unmasked"
fi

# Meshroom is CPU-bound outside DepthMap; match its thread count to the allocation
# rather than leaving the script's default of 16.
MAX_THREADS="${MAX_THREADS:-${SLURM_CPUS_PER_TASK:-16}}"

# meshroom_batch resumes from an existing cache, so a requeued job reuses finished nodes.
python baseline/benchmark_meshroom.py \
    "$SCENE" \
    "$OUT" \
    --save_file "$OUT/pipeline.mg" \
    --max_threads "$MAX_THREADS" \
    "$@"
