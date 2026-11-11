#!/bin/bash
# Meshroom (AliceVision) baseline, one array task per scene.
# %1 is deliberate: MeshroomCache hits ~50 GB/scene, pruned after each. See README.md.
# SFM=<dir> runs only the dense half on that SfM result's cameras and writes <dir>-meshroom;
# submit it with -J <dir>-meshroom so the timing CSV names the method.
# SCENE_ROOT=<dir> (with SFM) reads <dir>/<scene> instead of <scene>/<SFM>/.
#SBATCH --job-name=meshroom
#SBATCH --account=arthur.porto
#SBATCH --partition=hpg-rtx6000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus=1
#SBATCH --mem=64gb
#SBATCH --time=6:00:00
#SBATCH --array=0-75%1
#SBATCH --output=hpg_slurm/logs/%x-%A_%a.out
#SBATCH --error=hpg_slurm/logs/%x-%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=srizvi63@gatech.edu

# Own set -e: meshroom_common.sh's is moot if the source itself fails.
set -euo pipefail

# Slurm runs this from /var/spool, so $0 cannot find the repo; use the submit dir.
SLURM_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}/hpg_slurm"
if [ ! -f "$SLURM_DIR/meshroom_common.sh" ]; then
    echo "ERROR: cannot find meshroom_common.sh at $SLURM_DIR; submit from the repo root" >&2
    exit 2
fi
source "$SLURM_DIR/meshroom_common.sh"

SFM="${SFM:-}"

# Native mode reads the prepared dataset already, and only SFM names a from-SfM output.
if [ -n "${SCENE_ROOT:-}" ] && [ -z "$SFM" ]; then
    echo "ERROR: SCENE_ROOT needs SFM=<name>, which names the output <name>-meshroom" >&2
    exit 2
fi

discover_scenes
select_scene

if [ -n "$SFM" ]; then
    # Consume an SfM result, not the prepared dataset; the native meshroom/ is untouched.
    # SCENE_ROOT=<dir> reads <dir>/<scene> instead (e.g. DATA_ROOT once a prepared scene
    # carries sparse/); SFM still names the output.
    if [ -n "${SCENE_ROOT:-}" ]; then
        SCENE="$SCENE_ROOT/$SCENE_NAME"
    else
        SCENE="$RESULT_ROOT/$SCENE_NAME/$SFM"
    fi
    OUT="$RESULT_ROOT/$SCENE_NAME/$SFM-meshroom"
    if [ ! -d "$SCENE/sparse/0" ] || [ -z "$(ls -A "$SCENE/sparse/0" 2>/dev/null)" ]; then
        echo "ERROR: no SfM model at $SCENE/sparse/0 - run the $SFM SfM job for this scene first" >&2
        exit 2
    fi
    if [ ! -e "$SCENE/images/." ]; then
        echo "ERROR: $SCENE/images does not resolve - its symlink target has moved" >&2
        exit 2
    fi
    if [ "${SLURM_JOB_NAME:-}" != "$SFM-meshroom" ]; then
        echo "WARNING: job name '${SLURM_JOB_NAME:-}' goes into the timing CSV; submit with -J $SFM-meshroom" >&2
    fi
else
    OUT="$RESULT_ROOT/$SCENE_NAME/meshroom"
fi
mkdir -p "$OUT"

# The third argument adds the RESULT sfm=<sfm>-meshroom log marker; native mode has none.
start_timing "$SCENE_NAME" "$(ls -1 "$SCENE/images" 2>/dev/null | wc -l)" "${SFM:+$SFM-meshroom}"

echo "scene      : $SCENE_NAME ($SCENE_INDEX of ${#SCENES[@]}) -> $OUT"
if [ -n "$SFM" ]; then
    echo "input sfm  : $SFM -> $SCENE"
fi
if [ -d "$SCENE/masks" ]; then
    echo "masks      : $(ls -1 "$SCENE/masks" | wc -l) file(s)"
else
    echo "masks      : none - running unmasked"
fi

if [ -n "$SFM" ]; then
    # --max_threads only reaches FeatureExtraction, which --from_sfm skips.
    MODE_ARGS=(--from_sfm)
else
    # CPU-bound outside DepthMap; match threads to the allocation.
    MODE_ARGS=(--max_threads "${MAX_THREADS:-${SLURM_CPUS_PER_TASK:-16}}")
fi

python baseline/benchmark_meshroom.py \
    "$SCENE" \
    "$OUT" \
    --save_file "$OUT/pipeline.mg" \
    "${MODE_ARGS[@]}" \
    "$@"
