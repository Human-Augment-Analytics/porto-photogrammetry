"""Timing export must preserve Syed's CSV schema and failed attempts."""
import csv
import sys
from augenblick.bench.timing import run, CSV_FIELDS


def test_full_command_csv_records_success_and_failure(tmp_path):
    path = tmp_path / "sfm_timings.csv"
    meta = dict(method="colmap-pgsr", scene="specimen", gpu="rtx6000", n_images=432,
                note="nvs_ring; train=288; test=144")
    for code in (0, 7):
        assert run([sys.executable, "-c", f"raise SystemExit({code})"],
                   tmp_path / f"timing_{code}.json", path, meta) == code
    with path.open() as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == list(CSV_FIELDS)
        rows = list(reader)
    assert len(rows) == 2 and [r["exit_code"] for r in rows] == ["0", "7"]
    assert all(float(r["seconds"]) > 0 and r["n_images"] == "432" for r in rows)
    assert all(r["node"] and r["finished"] and r["note"] == meta["note"] for r in rows)
