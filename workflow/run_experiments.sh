#!/usr/bin/env bash
# Run configured geometry and held-out-view experiments.
# Example: BACKENDS="2dgs pgsr gw" workflow/run_experiments.sh
# DRY=1 skips jobs but still updates tree configs.
# SuGaR is unsupported because it cannot honor held-out splits.
set -euo pipefail

BACKENDS="${BACKENDS:-2dgs}"
EXPERIMENTS="${EXPERIMENTS:-geom_gt8 nvs_ring1}"
JOBS="${JOBS:-3}"
DRY="${DRY:-}"

REPO="${AUGENBLICK_REPO:-/blue/arthur.porto/ivilkhovyi3.gatech/augenblick}"
ROOT="${BENCHMARK_ROOT:-/blue/arthur.porto/ivilkhovyi3.gatech/benchmark}"
PYBIN="${AUGENBLICK_PYTHON:-/blue/arthur.porto/srizvi63.gatech/conda/augenblick_rtx_pro_6000/bin/python}"
SNAKEMAKE_MODULE="${SNAKEMAKE_MODULE:-snakemake/7.32.4}"

for backend in $BACKENDS; do
    case "$backend" in
        2dgs|pgsr|gw) ;;
        sugar) echo "ERROR: sugar cannot honour a split; see the note above" >&2; exit 2 ;;
        *) echo "ERROR: unknown backend '$backend' (valid: 2dgs, pgsr, gw)" >&2; exit 2 ;;
    esac
done

if ! command -v snakemake >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh 2>/dev/null || true
    module load "$SNAKEMAKE_MODULE"
fi

for tree in $EXPERIMENTS; do
    target="$ROOT/$tree"
    [ -f "$target/config.json" ] || { echo "ERROR: no tree at $target" >&2; exit 2; }

    # Persist the backend selection in the tree config.
    "$PYBIN" - "$target/config.json" $BACKENDS <<'PY'
import json, sys
from pathlib import Path
path, backends = Path(sys.argv[1]), sys.argv[2:]
config = json.loads(path.read_text())
config["backends"] = sorted(set(config.get("backends", [])) | set(backends))
config["train"] = True
config.pop("backend", None)
path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
print(f"{path.parent.name}: backends {' '.join(config['backends'])}, "
      f"split {config.get('split')}")
PY

    mkdir -p "$target/work/logs"
    cd "$target"
    if [ -n "$DRY" ]; then
        snakemake -s "$REPO/workflow/Snakefile" \
            --config tree="$target" python="$PYBIN" -n
    else
        # Keep the driver alive after SSH disconnection.
        setsid nohup snakemake -s "$REPO/workflow/Snakefile" \
            --profile "$REPO/workflow/profiles/hipergator" \
            --config tree="$target" python="$PYBIN" --jobs "$JOBS" \
            >> "$target/work/logs/run_${tree}.log" 2>&1 < /dev/null &
        echo "  started, following $target/work/logs/run_${tree}.log"
        sleep 5
    fi
done

if [ -z "$DRY" ]; then
    cat <<EOS

Running. To watch:
    squeue -u \$USER
    tail -f $ROOT/<tree>/work/logs/run_<tree>.log

A driver lives on the login node it was started from, and HiPerGator round-robins those,
so check it by the freshness of that log rather than with ps.
EOS
fi
