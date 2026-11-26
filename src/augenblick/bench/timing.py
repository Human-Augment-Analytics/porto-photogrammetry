"""Time one complete command, independently of backend subprocess-stage timers."""
import argparse
import csv
import fcntl
import socket
import datetime
import json
import os
import subprocess
import time
from pathlib import Path


CSV_FIELDS = ("method", "scene", "gpu", "n_images", "seconds", "exit_code", "note", "jobid", "node", "finished")


def append_csv(path: Path, record: dict, metadata: dict) -> None:
    """Append one full-command row using Syed's schema; serialize concurrent jobs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**metadata, "seconds": round(record["seconds"], 3),
           "exit_code": record["exit_code"], "jobid": record["slurm_job_id"] or "",
           "node": socket.gethostname(), "finished": record["finished_utc"]}
    with path.open("a+", newline="") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        first = handle.readline().strip()
        if first and next(csv.reader([first])) != list(CSV_FIELDS):
            raise ValueError("refusing incompatible timing CSV header")
        handle.seek(0, 2)
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not first:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def run(command: list[str], output: Path, csv_path: Path | None = None,
        metadata: dict | None = None) -> int:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    begin = time.perf_counter()
    code = 1
    try:
        code = subprocess.run(command).returncode
        return code
    finally:
        record = {"scope": "command wall time; excludes queue and caller setup",
                  "command": command, "started_utc": started,
                  "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "seconds": time.perf_counter() - begin, "exit_code": code,
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK")}
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(output)
        if csv_path is not None:
            append_csv(csv_path, record, metadata or {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--method")
    parser.add_argument("--scene-name")
    parser.add_argument("--images", type=Path)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--gpu", default="rtx6000")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    metadata = None
    if args.csv:
        if not (args.method and args.scene_name and args.images and args.split):
            parser.error("CSV timing requires method, scene-name, images and split")
        split = json.loads(args.split.read_text())
        n_images = sum(p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")
                       for p in args.images.iterdir())
        metadata = {"method": "colmap-" + args.method, "scene": args.scene_name,
                    "gpu": args.gpu, "n_images": n_images,
                    "note": f"nvs_ring; train={len(split['train'])}; test={len(split['test'])}; "
                            "scope=full_recon_cli; excludes_shared_scoring"}
    return run(command, args.out, args.csv, metadata)


if __name__ == "__main__":
    raise SystemExit(main())
