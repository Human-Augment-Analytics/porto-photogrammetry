#!/bin/bash
# Gaussian reconstruction, one array task per scene.
# Usage: BACKEND=2dgs SFM=vggt sbatch hpg_slurm/recon.sbatch [extra flags]
# One scene: sbatch --array=N ...   (N indexes the sorted scene list)
# BACKEND=2dgs|sugar|pgsr|gw; SFM= picks the input SfM at <scene>/<SFM>/.
# 64gb is NOT enough for 2DGS TSDF at --mesh_res 4096 on ~660 images (peaked 97 GiB RSS,
# oom_killed); raise --mem there - nodes carry 515-2063 gb, so headroom is free.
#SBATCH --job-name=recon
#SBATCH --account=arthur.porto
#SBATCH --partition=hpg-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64gb
#SBATCH --time=3:00:00
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

BACKEND="${BACKEND:-2dgs}"
case "$BACKEND" in
    2dgs|sugar|pgsr|gw) ;;
    *) echo "ERROR: unknown BACKEND='$BACKEND' (valid: 2dgs, sugar, pgsr, gw)" >&2; exit 2 ;;
esac

SFM="${SFM:-vggt}"

discover_scenes
select_scene
# Reconstruction consumes an SfM result, not the prepared dataset.
SCENE="$RESULT_ROOT/$SCENE_NAME/$SFM"
OUT="$RESULT_ROOT/$SCENE_NAME/$SFM-$BACKEND"

echo "scene      : $SCENE_NAME ($SCENE_INDEX of ${#SCENES[@]})"
echo "backend    : $BACKEND  <- sfm: $SFM"
echo "input      : $SCENE"
echo "output     : $OUT"

# A missing or empty sparse/0 means that scene's SfM never produced a model.
if [ ! -d "$SCENE/sparse/0" ] || [ -z "$(ls -A "$SCENE/sparse/0" 2>/dev/null)" ]; then
    echo "ERROR: no SfM model at $SCENE/sparse/0 - run the $SFM SfM job for this scene first" >&2
    exit 2
fi

mkdir -p "$OUT"

# One CSV row per scene, written even if the job dies.
start_timing "$SCENE_NAME" "$(ls -1 "$SCENE/images" 2>/dev/null | wc -l)"

# The registry resolves BACKEND; an unknown name fails listing the valid set.
augenblick recon "$BACKEND" --scene "$SCENE" --output "$OUT" "$@"
