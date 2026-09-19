# AliceVision counterpart of common.sh: sets up the Meshroom env, then sources
# scene_common.sh. See "Meshroom baseline" in README.md for why it cannot use common.sh.

set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/scratch/conda}"
CONDA_ENV="${CONDA_ENV:-$CONDA_ROOT/meshroom}"

# --- AliceVision / Meshroom (mirrors the ~/.bashrc block in meshroom-setup.md) ---
export ALICEVISION_ROOT="${ALICEVISION_ROOT:-$HOME/scratch/alicevision}"
export MESHROOM_ROOT="${MESHROOM_ROOT:-$HOME/scratch/Meshroom}"

export PATH="$ALICEVISION_ROOT/bin:$ALICEVISION_ROOT/lib:$PATH"
export LD_LIBRARY_PATH="$ALICEVISION_ROOT/bin:$ALICEVISION_ROOT/lib:${LD_LIBRARY_PATH:-}"

export MESHROOM_NODES_PATH="$ALICEVISION_ROOT/share/meshroom"
export MESHROOM_PIPELINE_TEMPLATES_PATH="$ALICEVISION_ROOT/share/meshroom"

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

# Fail early: a missing sensor DB otherwise surfaces as an opaque CameraInit crash.
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

# --- GPU support gate (see "Supported GPUs" in README.md) -------------------
MESHROOM_MAX_SM="${MESHROOM_MAX_SM:-120}"

# Compute capability as sm_XX*10; -1 = not CUDA at all.
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

# A label for the gate below only; nothing here varies per GPU.
GPU="${GPU:-}"
if [ -z "$GPU" ]; then
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
    echo "ERROR: GPU='$GPU' is sm_$_sm, above the tested sm_$MESHROOM_MAX_SM" >&2
    echo "       Untested, not known-broken: set MESHROOM_MAX_SM=$_sm to try it." >&2
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

export PYTHONPATH="$MESHROOM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# scene_common.sh reads GPU and CONDA_ENV; BANNER_EXTRA adds AliceVision provenance.
BANNER_EXTRA="gpu target : $GPU (AliceVision max sm_$MESHROOM_MAX_SM, bundled CUDA 12.1 + nvrtc)
alicevision: $ALICEVISION_ROOT
meshroom   : $MESHROOM_ROOT ($(git -C "$MESHROOM_ROOT" rev-parse --short HEAD 2>/dev/null || echo 'no git'))"

source "$(dirname "${BASH_SOURCE[0]}")/scene_common.sh"
