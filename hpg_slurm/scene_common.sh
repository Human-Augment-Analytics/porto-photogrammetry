# Shared by common.sh and meshroom_common.sh: roots, scene discovery, banner, timing CSV.
# These must stay identical across methods for results to compare; env setup is the
# caller's job. Callers set GPU and CONDA_ENV first, and may set BANNER_EXTRA (newline-
# separated "label: value" lines). See "Shared job setup" in README.md.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Absolute /blue/... paths, outside the checkout (PACE derives its own from REPO_ROOT).
# Override either to relocate.
DATA_ROOT="${DATA_ROOT:-/blue/arthur.porto/data/datasets/photogrammetry/neurips/prepared}"
RESULT_ROOT="${RESULT_ROOT:-/blue/arthur.porto/srizvi63.gatech/results/neurips}"

# --- scene discovery --------------------------------------------------------
#   nested : <DATA_ROOT>/<scene>/prepared/{images,masks}   (the main dataset)
#   flat   : <DATA_ROOT>/<scene>/{images,masks}            (the neurips dataset)
SCENE_LAYOUT="${SCENE_LAYOUT:-auto}"

if [ "$SCENE_LAYOUT" = "auto" ]; then
    if find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -type d -name prepared -print -quit 2>/dev/null | grep -q .; then
        SCENE_LAYOUT="nested"
    else
        SCENE_LAYOUT="flat"
    fi
fi

# Populates SCENES, sorted: array index N means the same scene across submissions.
discover_scenes() {
    case "$SCENE_LAYOUT" in
        nested)
            mapfile -t SCENES < <(find "$DATA_ROOT" -mindepth 2 -maxdepth 2 -type d -name prepared \
                -printf '%h\n' | xargs -r -n1 basename | sort)
            ;;
        flat)
            # -xtype d too: images/ is often a symlink, and dropping those scenes
            # would shift every later array index.
            mapfile -t SCENES < <(find "$DATA_ROOT" -mindepth 2 -maxdepth 2 \
                \( -type d -o -xtype d \) -name images \
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

# --- GPU label --------------------------------------------------------------
# The CSV's gpu column: GPU, or GPU_<profile> on a MIG slice. See README.md.
GPU_SLICE="$GPU"
if _mig="$(nvidia-smi -L 2>/dev/null | sed -n 's/.*MIG *\([0-9]\+g\.[0-9]\+gb\).*/\1/p' | head -1)" \
   && [ -n "$_mig" ]; then
    GPU_SLICE="${GPU}_${_mig}"
fi
unset _mig
export GPU_SLICE

# --- banner -----------------------------------------------------------------
echo "=========================================================="
echo "job        : ${SLURM_JOB_NAME:-interactive} (${SLURM_JOB_ID:-no-jobid})"
echo "node       : $(hostname)"
echo "started    : $(date -Is)"
echo "gpu actual : ${GPU_SLICE:-$GPU}"
echo "conda env  : $CONDA_ENV"
echo "python     : $(python --version 2>&1) @ $(command -v python)"
[ -n "${BANNER_EXTRA:-}" ] && printf '%s\n' "$BANNER_EXTRA"
echo "repo       : $REPO_ROOT ($(git rev-parse --short HEAD 2>/dev/null || echo 'no git'))"
nvidia-smi -L 2>/dev/null || echo "gpus       : none visible"
echo "=========================================================="

# GPU= picks the env and arch but NOT the partition, so a mismatch silently runs one card's
# env on another's hardware. Warn, don't exit: an interactive shell has no partition.
if [ -n "${GPU_PARTITION:-}" ] && [ -n "${SLURM_JOB_PARTITION:-}" ] \
   && [ "$SLURM_JOB_PARTITION" != "$GPU_PARTITION" ]; then
    echo "WARNING: GPU=$GPU expects --partition=$GPU_PARTITION but this job is on $SLURM_JOB_PARTITION" >&2
fi

# --- per-scene timing -------------------------------------------------------
# One appended row per job. Schema and columns are documented in README.md.
TIMING_DIR="${TIMING_DIR:-$RESULT_ROOT/_timing}"
TIMING_CSV="${TIMING_CSV:-$TIMING_DIR/sfm_timings.csv}"

_timing_finish() {
    local status=$?
    local elapsed=$(( $(date +%s) - _TIMING_START ))
    local note=""
    local log="hpg_slurm/logs/${SLURM_JOB_NAME:-interactive}-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}_${SLURM_ARRAY_TASK_ID:-0}.err"

    # Slurm's SIGTERM at the time limit leaves status 0, making a timeout look like
    # success; read Slurm's own message instead. Must stay before the status check.
    if [ -f "$log" ] && grep -qiE 'DUE TO TIME LIMIT|CANCELLED AT .* DUE TO' "$log"; then
        note="KILLED_OR_TIMEOUT"
        [ "$status" -eq 0 ] && status=124
    fi

    if [ -z "$note" ] && [ "$status" -ne 0 ]; then
        # No SASS for this device, no runtime compile to fall back on (Meshroom only).
        if [ -f "$log" ] && grep -qiE "no kernel image is available" "$log"; then
            note="UNSUPPORTED_ARCH"
        elif [ -f "$log" ] && grep -qiE "CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|torch\.OutOfMemoryError" "$log"; then
            note="GPU_OOM"
        elif [ -f "$log" ] && grep -qiE "Out of memory|oom-kill|Killed process|MemoryError|std::bad_alloc" "$log"; then
            note="HOST_OOM"
        elif [ "$status" -eq 124 ] || [ "$status" -eq 137 ]; then
            # 137 = SIGKILL, also how Slurm ends a job at its time limit.
            note="KILLED_OR_TIMEOUT"
        else
            note="FAILED"
        fi
    fi

    # Machine-readable marker for log scraping; keep the prefix and fields stable.
    if [ -n "${_TIMING_METHOD:-}" ]; then
        echo "RESULT scene=${_TIMING_SCENE:-unknown} sfm=${_TIMING_METHOD} exit=${status} seconds=${elapsed}"
    fi

    mkdir -p "$TIMING_DIR"
    echo "${SLURM_JOB_NAME:-interactive},${_TIMING_SCENE:-unknown},${GPU_SLICE:-$GPU},${_TIMING_NIMG:-0},${elapsed},${status},${note},${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-0}}_${SLURM_ARRAY_TASK_ID:-0},$(hostname),$(date -Is)" >> "$TIMING_CSV"
}

# A third argument names the method for the RESULT marker; omit it to skip that line.
start_timing() {
    _TIMING_SCENE="$1"
    _TIMING_NIMG="${2:-0}"
    _TIMING_METHOD="${3:-}"
    _TIMING_START=$(date +%s)
    mkdir -p "$TIMING_DIR"
    if [ ! -f "$TIMING_CSV" ]; then
        echo "method,scene,gpu,n_images,seconds,exit_code,note,jobid,node,finished" > "$TIMING_CSV"
    fi
    trap _timing_finish EXIT
}
