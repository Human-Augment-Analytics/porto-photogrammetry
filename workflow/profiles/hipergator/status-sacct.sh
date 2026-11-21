#!/usr/bin/env bash
# Slurm accounting can briefly be unavailable or lag the scheduler.
# Unknown status must not fail a healthy Snakemake workflow.
set -uo pipefail
jobid="${1:?Slurm job id required}"
if [[ ! "$jobid" =~ ^[0-9]+(_[0-9]+)?$ ]]; then
    echo 'Invalid Slurm job id' >&2
    exit 2
fi
if queued=$(timeout 20 squeue -h -j "$jobid" -o '%T' 2>/dev/null); then
    if [[ -n "$queued" ]]; then
        echo running
        exit 0
    fi
fi
if accounting=$(timeout 20 sacct -X -j "$jobid" --format=JobIDRaw,State --noheader --parsable2 2>/dev/null); then
    state=$(awk -F'|' -v id="$jobid" '$1 == id {split($2,s," "); print s[1]; exit}' <<< "$accounting")
    state="${state%+}"
    case "$state" in
        COMPLETED) echo success; exit 0 ;;
        FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED|REVOKED)
            echo failed; exit 0 ;;
        RUNNING|PENDING|COMPLETING|CONFIGURING|SUSPENDED|REQUEUED|RESIZING|REQUEUE_FED|REQUEUE_HOLD|SIGNALING|STAGE_OUT)
            echo running; exit 0 ;;
    esac
fi
printf 'Slurm status unavailable for job %s; waiting for a confirmed state\n' "$jobid" >&2
echo running
