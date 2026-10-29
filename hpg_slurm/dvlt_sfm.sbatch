#!/bin/bash
# DVLT -> COLMAP (feed-forward, no BA) for every scene under DATA_ROOT, one array task per scene.
# Usage: sbatch hpg_slurm/dvlt_sfm.sbatch [extra flags]
# One scene: sbatch --array=N ...   (N indexes the sorted scene list)
# RTX only: dvlt is installed in augenblick_rtx_pro_6000, not the B200 env.
#SBATCH --job-name=dvlt-sfm
#SBATCH --account=arthur.porto
#SBATCH --partition=hpg-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=48gb
#SBATCH --time=2:00:00
#SBATCH --array=0-75%8
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

# The B200 env has drifted from its setup script and does not carry dvlt.
if [ "$GPU" != rtx6000 ]; then
    echo "ERROR: dvlt is installed only in augenblick_rtx_pro_6000 (GPU=$GPU)" >&2
    exit 2
fi

# SFM_TAG names the output dir and the timing/RESULT method, so runs with different flags
# (e.g. SFM_TAG=dvlt-24 with --inference_steps 24) sit side by side instead of overwriting.
SFM_TAG="${SFM_TAG:-dvlt}"

discover_scenes
select_scene
OUT="$RESULT_ROOT/$SCENE_NAME/$SFM_TAG"
mkdir -p "$OUT"

N_IMAGES=$(ls -1 "$SCENE/images" 2>/dev/null | wc -l)
N_MASKS=$( (ls -1 "$SCENE/masks" 2>/dev/null || true) | wc -l)

echo "scene      : $SCENE_NAME ($SCENE_INDEX of ${#SCENES[@]}) -> $OUT"
echo "images     : $N_IMAGES"

# Machine-readable marker for log scraping; keep the prefix and fields stable.
echo "SCENE_INFO scene=$SCENE_NAME sfm=$SFM_TAG images=$N_IMAGES masks=$N_MASKS"

# One CSV row per scene, written even if the job dies.
start_timing "$SCENE_NAME" "$N_IMAGES" "$SFM_TAG"

augenblick sfm dvlt --scene "$SCENE" --output "$OUT" "$@"
