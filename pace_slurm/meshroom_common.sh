# Sourced by meshroom_benchmark.sbatch. The Meshroom/AliceVision baseline cannot use
# common.sh, for three reasons:
#
#   1. common.sh maps GPU -> $CONDA_ROOT/augenblick_<gpu> and rejects anything else. The
#      Meshroom frontend is a single env ($CONDA_ROOT/meshroom, pure Python: PySide6 +
#      numpy, no torch) that is GPU-independent, so there is nothing per-GPU to select.
#   2. common.sh does `module load cuda/12.9.1`. The AliceVision 3.3.0 release tarball
#      bundles its own libcudart.so.12.1.105 and does not link libcuda directly; putting any
#      CUDA runtime ahead of it on LD_LIBRARY_PATH is at best pointless and at worst shadows
#      the bundled one. This file loads no CUDA module at all.
#   3. The ALICEVISION_* variables live only in ~/.bashrc, which a batch job never sources.
#      They are re-exported here.
#
# common.sh is deliberately left untouched; the scene-discovery and timing helpers that
# this baseline does need are duplicated below rather than refactored out of it.

set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/scratch/conda}"
CONDA_ENV="${CONDA_ENV:-$CONDA_ROOT/meshroom}"

# --- AliceVision / Meshroom -------------------------------------------------
# Mirrors the block meshroom-setup.md tells you to append to ~/.bashrc.
export ALICEVISION_ROOT="${ALICEVISION_ROOT:-$HOME/scratch/alicevision}"
export MESHROOM_ROOT="${MESHROOM_ROOT:-$HOME/scratch/Meshroom}"

export PATH="$ALICEVISION_ROOT/bin:$ALICEVISION_ROOT/lib:$PATH"
export LD_LIBRARY_PATH="$ALICEVISION_ROOT/bin:$ALICEVISION_ROOT/lib:${LD_LIBRARY_PATH:-}"

export MESHROOM_NODES_PATH="$ALICEVISION_ROOT/share/meshroom"
export MESHROOM_PIPELINE_TEMPLATES_PATH="$ALICEVISION_ROOT/share/meshroom"

# CameraInit fails without the sensor DB; ImageMatching fails without the vocabulary tree.
export ALICEVISION_SENSOR_DB="$ALICEVISION_ROOT/share/aliceVision/cameraSensors.db"
export ALICEVISION_VOCTREE="$ALICEVISION_ROOT/share/aliceVision/vlfeat_K80L3.SIFT.tree"
export ALICEVISION_SPHERE_DETECTION_MODEL="$ALICEVISION_ROOT/share/aliceVision/sphereDetection_Mask-RCNN.onnx"
export ALICEVISION_SEMANTIC_SEGMENTATION_MODEL="$ALICEVISION_ROOT/share/aliceVision/fcn_resnet50.onnx"
export ALICEVISION_COLORCHARTDETECTION_MODEL_FOLDER="$ALICEVISION_ROOT/share/aliceVision/ColorChartDetectionModel"
export ALICEVISION_LENS_PROFILE_INFO=""

export ALICEVISION_USE_OPENCV=ON
export ALICEVISION_USE_POPSIFT=ON
export ALICEVISION_USE_CCTAG=ON
export ALICEVISION_BUILD_SWIG_BINDING=ON
export ALICEVISION_INSTALL_MESHROOM_PLUGIN=ON

# Fail early and specifically: a missing sensor DB otherwise surfaces as an opaque
# CameraInit crash several minutes into the graph.
for _v in ALICEVISION_ROOT MESHROOM_ROOT ALICEVISION_SENSOR_DB ALICEVISION_VOCTREE; do
    if [ ! -e "${!_v}" ]; then
        echo "ERROR: $_v points at a missing path: ${!_v}" >&2
        echo "       see meshroom-setup.md to install AliceVision 3.3.0 + Meshroom" >&2
        exit 2
    fi
done
unset _v

if [ ! -f "$MESHROOM_ROOT/bin/meshroom_batch" ]; then
    echo "ERROR: no meshroom_batch at $MESHROOM_ROOT/bin/meshroom_batch" >&2
    exit 2
fi

# --- GPU support gate -------------------------------------------------------
# AliceVision 3.3.0 ships prebuilt: its CUDA kernels are SASS-only, with no PTX in
# libaliceVision_depthMap.so, libpopsift.so, or libCCTag.so (verified with cuobjdump).
# No PTX means no JIT, so a device newer than the newest compiled SASS (sm_90) cannot
# run DepthMap at all - it dies with "no kernel image is available for execution on the
# device" rather than falling back. Unlike the torch backends, which usually carry PTX
# and can JIT onto Blackwell, this has a hard ceiling.
MESHROOM_MAX_SM="${MESHROOM_MAX_SM:-90}"

# Every ice-gpu model and its compute capability, as sm_XX*10. mi210 is AMD/ROCm and has
# no CUDA capability at all, so it is rejected outright.
gpu_supported() {
    case "$1" in
        v100)                   echo 70 ;;
        rtx_6000)               echo 75 ;;
        a100)                   echo 80 ;;
        a40)                    echo 86 ;;
        l40s)                   echo 89 ;;
        h100|h200)              echo 90 ;;
        rtx_pro_6000_blackwell) echo 120 ;;
        mi210)                  echo -1 ;;
        *)                      echo 0 ;;
    esac
}

# GPU names the model this job was allocated. It is only a label for the check below:
# unlike common.sh, nothing here varies per GPU, so it does not select an env or a module.
GPU="${GPU:-}"
if [ -z "$GPU" ]; then
    # Infer from the gres Slurm actually gave us, so the check works without GPU= being set.
    GPU="$(echo "${SLURM_JOB_GRES:-${SBATCH_GRES:-}}" | grep -oE 'gpu:[a-z0-9_]+' \
        | head -1 | cut -d: -f2)"
    GPU="${GPU:-unknown}"
fi

_sm="$(gpu_supported "$GPU")"
if [ "$_sm" -lt 0 ]; then
    echo "ERROR: GPU='$GPU' is AMD/ROCm; AliceVision is CUDA-only" >&2
    exit 2
elif [ "$_sm" -eq 0 ]; then
    echo "WARNING: unrecognised GPU='$GPU'; cannot verify AliceVision supports it" >&2
elif [ "$_sm" -gt "$MESHROOM_MAX_SM" ]; then
    echo "ERROR: GPU='$GPU' is sm_$_sm, above AliceVision 3.3.0's sm_$MESHROOM_MAX_SM ceiling" >&2
    echo "       Its kernels are SASS-only (no PTX), so DepthMap cannot JIT onto this device." >&2
    echo "       Use v100, rtx_6000, a100, a40, l40s, h100, or h200." >&2
    exit 2
fi
unset _sm

# --- conda ------------------------------------------------------------------
CONDA_SH="${CONDA_SH:-/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh}"

if [ ! -d "$CONDA_ENV" ]; then
    echo "ERROR: no conda env at $CONDA_ENV" >&2
    echo "       build it per meshroom-setup.md step 4" >&2
    exit 2
fi
if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: no conda.sh at $CONDA_SH (override with CONDA_SH=...)" >&2
    exit 2
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV"

# meshroom_batch is a script inside the checkout, not an installed console entry point.
export PYTHONPATH="$MESHROOM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/main}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/output}"

# --- scene discovery --------------------------------------------------------
# Same two on-disk layouts, and the same sorted ordering, as common.sh: an array index
# maps to the same scene across every submission, Meshroom's included.
#   nested : <DATA_ROOT>/<scene>/prepared/{images,masks}
#   flat   : <DATA_ROOT>/<scene>/{images,masks}
SCENE_LAYOUT="${SCENE_LAYOUT:-auto}"

if [ "$SCENE_LAYOUT" = "auto" ]; then
    if find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -type d -name prepared -print -quit 2>/dev/null | grep -q .; then
        SCENE_LAYOUT="nested"
    else
        SCENE_LAYOUT="flat"
    fi
fi

discover_scenes() {
    case "$SCENE_LAYOUT" in
        nested)
            mapfile -t SCENES < <(find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -type d -name prepared \
                -printf '%h\n' | xargs -r -n1 basename | sort)
            ;;
        flat)
            mapfile -t SCENES < <(find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -type d -name images \
                -printf '%h\n' | xargs -r -n1 basename | sort)
            ;;
        *)
            echo "ERROR: unknown SCENE_LAYOUT='$SCENE_LAYOUT' (valid: auto, nested, flat)" >&2
            exit 2
            ;;
    esac

    if [ "${#SCENES[@]}" -eq 0 ]; then
        echo "ERROR: no $SCENE_LAYOUT-layout scenes under $DATA_ROOT" >&2
        exit 2
    fi
}

scene_path() {
    if [ "$SCENE_LAYOUT" = "nested" ]; then
        echo "$DATA_ROOT/$1/prepared"
    else
        echo "$DATA_ROOT/$1"
    fi
}

select_scene() {
    local idx="${SLURM_ARRAY_TASK_ID:-0}"
    if [ "$idx" -ge "${#SCENES[@]}" ]; then
        echo "ERROR: array index $idx exceeds ${#SCENES[@]} scene(s) under $DATA_ROOT" >&2
        exit 2
    fi
    SCENE_NAME="${SCENES[$idx]}"
    SCENE="$(scene_path "$SCENE_NAME")"
    SCENE_INDEX="$idx"
}

# Provenance banner, matching common.sh's but reporting AliceVision rather than a
# torch/CUDA target, since no CUDA module is loaded.
echo "=========================================================="
echo "job        : ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-no-jobid})"
echo "node       : $(hostname)"
echo "started    : $(date -Is)"
echo "gpu target : $GPU (AliceVision ceiling sm_$MESHROOM_MAX_SM, bundled CUDA 12.1)"
echo "conda env  : $CONDA_ENV"
echo "python     : $(python --version 2>&1) @ $(command -v python)"
echo "alicevision: $ALICEVISION_ROOT"
echo "meshroom   : $MESHROOM_ROOT ($(git -C "$MESHROOM_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
echo "repo       : $REPO_ROOT ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
nvidia-smi -L 2>/dev/null || echo "gpus       : none visible"
echo "=========================================================="

# --- per-scene timing -------------------------------------------------------
# Mirrors common.sh so Meshroom rows land in the same CSV as the SfM methods and can be
# compared directly. Rows are appended, never rewritten, so array tasks do not clobber.
TIMING_DIR="${TIMING_DIR:-$RESULT_ROOT/_timing}"
TIMING_CSV="${TIMING_CSV:-$TIMING_DIR/sfm_timings.csv}"

_timing_finish() {
    local status=$?
    local elapsed=$(( $(date +%s) - _TIMING_START ))
    local note=""
    local log="pace_slurm/logs/${SLURM_JOB_NAME:-interactive}-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}_${SLURM_ARRAY_TASK_ID:-0}.err"

    # Slurm SIGTERMs the job at its time limit; bash's EXIT trap then sees status 0,
    # so a timeout is indistinguishable from success by exit code alone. Detect it
    # from Slurm's own message instead, before the status check below.
    if [ -f "$log" ] && grep -qiE 'DUE TO TIME LIMIT|CANCELLED AT .* DUE TO' "$log"; then
        note="KILLED_OR_TIMEOUT"
        [ "$status" -eq 0 ] && status=124
    fi

    if [ -z "$note" ] && [ "$status" -ne 0 ]; then
        # "no kernel image" is the signature of running past the sm_90 ceiling; it is
        # called out separately so it is not misread as a plain crash.
        if [ -f "$log" ] && grep -qiE "no kernel image is available" "$log"; then
            note="UNSUPPORTED_ARCH"
        elif [ -f "$log" ] && grep -qiE "CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|torch\.OutOfMemoryError" "$log"; then
            note="GPU_OOM"
        elif [ -f "$log" ] && grep -qiE "Out of memory|oom-kill|Killed process|MemoryError|std::bad_alloc" "$log"; then
            note="HOST_OOM"
        elif [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
            note="KILLED_OR_TIMEOUT"
        else
            note="FAILED"
        fi
    fi

    mkdir -p "$TIMING_DIR"
    echo "${SLURM_JOB_NAME:-interactive},${_TIMING_SCENE:-unknown},${_TIMING_NIMG:-0},${elapsed},${status},${note},${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}_${SLURM_ARRAY_TASK_ID:-0},$(hostname),$(date -Is)" >> "$TIMING_CSV"
}

start_timing() {
    _TIMING_SCENE="$1"
    _TIMING_NIMG="${2:-0}"
    _TIMING_START=$(date +%s)
    mkdir -p "$TIMING_DIR"
    if [ ! -f "$TIMING_CSV" ]; then
        echo "method,scene,n_images,seconds,exit_code,note,jobid,node,finished" > "$TIMING_CSV"
    fi
    trap _timing_finish EXIT
}
