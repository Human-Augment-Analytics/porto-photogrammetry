"""COLMAP SfM configuration invariants that must hold on any machine."""
import os

import pytest

pycolmap = pytest.importorskip("pycolmap")

from augenblick.sfm.colmap import ColmapConfig, ColmapSfM  # noqa: E402


class _StopBeforeExtraction(Exception):
    """Raised from the patched extractor so the test never runs real SfM."""


@pytest.fixture
def extraction_options(tmp_path, monkeypatch):
    """Run the method until it calls extract_features, and return the options it built."""
    captured = {}

    def fake_extract(*args, **kwargs):
        captured["options"] = kwargs.get("extraction_options")
        raise _StopBeforeExtraction

    monkeypatch.setattr(pycolmap, "extract_features", fake_extract)

    def run_with(affinity: int):
        monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(affinity)))
        scene_dir = tmp_path / f"scene_{affinity}"
        (scene_dir / "images").mkdir(parents=True)
        (scene_dir / "images" / "a.jpg").touch()

        method = ColmapSfM(ColmapConfig())
        # build_input is how the CLI turns a path into the method's input type; going through
        # it keeps the test on the real entry path rather than guessing at the signature.
        scene = ColmapSfM.build_input(scene_dir)
        with pytest.raises(_StopBeforeExtraction):
            method.run(scene, tmp_path / f"out_{affinity}")
        assert "options" in captured, "extract_features was never reached"
        return captured["options"]

    return run_with


def test_thread_count_follows_the_cpu_allocation(extraction_options):
    """Threads track the CPUs the job was given, not the cores the node happens to have."""
    assert extraction_options(8).num_threads == 8


def test_thread_count_tracks_a_different_allocation(extraction_options):
    """A second width, so the test cannot pass by coincidence against a fixed default."""
    assert extraction_options(4).num_threads == 4


def test_thread_count_is_never_minus_one(extraction_options):
    """`-1` is the specific value COLMAP resolves to the machine's core count."""
    threads = extraction_options(2).num_threads
    assert threads > 0
    assert threads != -1
