# Sourced by every sbatch script here: selects GPU, loads modules, activates conda.
# Not executable on its own. Set GPU=a100|l40s|a40 before submitting to switch targets.
#
# PACE ICE variant, the account is coc-ice, and PACE has no xerces/yasm modules.

set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-$HOME/scratch/conda}"
GPU="${GPU:-a100}"

# Arch strings mirror the GPU_ARCH each scripts/setup_<gpu>.sh exports.
# gres names come from `sinfo -p ice-gpu -o %G` and are what --gres=gpu:<name>:1 expects.
case "$GPU" in
    a100)
        CONDA_ENV="$CONDA_ROOT/augenblick_a100"
        CUDA_MODULE="cuda/12.9.1"
        GPU_ARCH="8.0"
        GRES_NAME="a100"
        ;;
    l40s)
        CONDA_ENV="$CONDA_ROOT/augenblick_l40s"
        CUDA_MODULE="cuda/12.9.1"
        GPU_ARCH="8.9"
        GRES_NAME="l40s"
        ;;
    a40)
        CONDA_ENV="$CONDA_ROOT/augenblick_a40"
        CUDA_MODULE="cuda/12.9.1"
        GPU_ARCH="8.6"
        GRES_NAME="a40"
        ;;
    h200)
        CONDA_ENV="$CONDA_ROOT/augenblick_h200"
        CUDA_MODULE="cuda/12.9.1"
        GPU_ARCH="9.0"
        GRES_NAME="h200"
        ;;
    *)
        echo "ERROR: unknown GPU='$GPU' (valid: a100, l40s, a40, h200)" >&2
        exit 2
        ;;
esac

# A batch shell does not source ~/.bashrc, so load the toolchain explicitly.
# colmap is deliberately not loaded: the Python SfM path uses pycolmap from the env.
module purge
module load "$CUDA_MODULE"
CONDA_SH="${CONDA_SH:-/usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh}"

if [ ! -d "$CONDA_ENV" ]; then
    echo "ERROR: no conda env at $CONDA_ENV" >&2
    echo "       build it with: bash scripts/setup_${GPU}.sh" >&2
    exit 2
fi

# A bare `conda activate` fails in a non-interactive shell without this.
if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: no conda.sh at $CONDA_SH (override with CONDA_SH=...)" >&2
    exit 2
fi
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"

NVIDIA_LIB_ROOT="$(python -c 'import site;print(site.getsitepackages()[0])' 2>/dev/null)/nvidia"
if [ -d "$NVIDIA_LIB_ROOT" ]; then
    for _libdir in "$NVIDIA_LIB_ROOT"/*/lib; do
        [ -d "$_libdir" ] && LD_LIBRARY_PATH="$_libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
    export LD_LIBRARY_PATH
    unset _libdir
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# On ICE both roots live inside the repo checkout on scratch, so they follow REPO_ROOT
# rather than being absolute like the HPG /blue/... paths. Override either to relocate.
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/main}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/output}"

# Two on-disk layouts exist.
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

# Populates SCENES (sorted, so an array index maps to the same scene across submissions)
# and defines scene_path() to turn a name into the COLMAP scene dir.
discover_scenes() {
    case "$SCENE_LAYOUT" in
        nested)
            mapfile -t SCENES < <(find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -type d -name prepared \
                -printf '%h\n' | xargs -r -n1 basename | sort)
            ;;
        flat)
            # A scene is any dir holding images/; that skips stray files and manifests.
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

# Resolves SLURM_ARRAY_TASK_ID against SCENES; sets SCENE_NAME and SCENE.
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

# Provenance banner so a log explains itself when revisited later.
echo "=========================================================="
echo "job        : ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-no-jobid})"
echo "node       : $(hostname)"
echo "started    : $(date -Is)"
echo "gpu target : $GPU (sm_$GPU_ARCH, $CUDA_MODULE)"
echo "conda env  : $CONDA_ENV"
echo "python     : $(python --version 2>&1) @ $(command -v python)"
echo "repo       : $REPO_ROOT ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
nvidia-smi -L 2>/dev/null || echo "gpus       : none visible"
echo "=========================================================="

# --- per-scene timing -------------------------------------------------------
# Each job appends one CSV row to $TIMING_CSV on exit.
TIMING_DIR="${TIMING_DIR:-$RESULT_ROOT/_timing}"
TIMING_CSV="${TIMING_CSV:-$TIMING_DIR/sfm_timings.csv}"

_timing_finish() {
    local status=$?
    local elapsed=$(( $(date +%s) - _TIMING_START ))
    local note=""

    if [ "$status" -ne 0 ]; then
        local log="pace_slurm/logs/${SLURM_JOB_NAME:-interactive}-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}_${SLURM_ARRAY_TASK_ID:-0}.err"
        if [ -f "$log" ] && grep -qiE "CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|torch\.OutOfMemoryError" "$log"; then
            note="GPU_OOM"
        elif [ -f "$log" ] && grep -qiE "Out of memory|oom-kill|Killed process|MemoryError|std::bad_alloc" "$log"; then
            note="HOST_OOM"
        elif [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
            # 137 = SIGKILL, which is also how Slurm ends a job that hits its time limit.
            note="KILLED_OR_TIMEOUT"
        else
            note="FAILED"
        fi
    fi

    mkdir -p "$TIMING_DIR"
    echo "${SLURM_JOB_NAME:-interactive},${_TIMING_SCENE:-unknown},${_TIMING_NIMG:-0},${elapsed},${status},${note},${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}_${SLURM_ARRAY_TASK_ID:-0},$(hostname),$(date -Is)" >> "$TIMING_CSV"
}

# Armed by each job once it knows its scene, so the row carries the scene name.
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
