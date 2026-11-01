#!/usr/bin/env bash
# Translate Slurm accounting states to Snakemake job states.
set -euo pipefail

jobid="$1"
state="$(sacct -j "$jobid" --format=State --noheader --parsable2 2>/dev/null \
         | head -1 | cut -d' ' -f1)"

case "$state" in
    COMPLETED)                                   echo "success" ;;
    RUNNING|PENDING|COMPLETING|CONFIGURING|SUSPENDED|REQUEUED|RESIZING)
                                                 echo "running" ;;
    "")  # Not yet visible in the accounting database; assume it is still queued.
                                                 echo "running" ;;
    *)                                           echo "failed"  ;;
esac
