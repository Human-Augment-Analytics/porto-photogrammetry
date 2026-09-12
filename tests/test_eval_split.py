"""The held-out split is the contract that makes a cross-backend comparison meaningful."""
import json

import pytest

from augenblick.core.errors import SceneError
from augenblick.eval.split import (
    DEFAULT_HOLDOUT,
    SPLIT_FILENAME,
    build_split,
    copy_split,
    read_split,
    sparse_dir,
    write_split,
)

STEMS = [f"camera1_IMG_{i:04d}" for i in range(24)]


def test_build_split_reproduces_the_llffhold_rule():
    split = build_split(STEMS)
    assert split["test"] == STEMS[::DEFAULT_HOLDOUT]
    assert len(split["train"]) == len(STEMS) - len(split["test"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert sorted(split["train"] + split["test"]) == sorted(STEMS)


def test_build_split_honours_a_custom_holdout():
    assert build_split(STEMS, holdout=4)["test"] == STEMS[::4]


def test_sparse_dir_prefers_the_nested_model(tmp_path):
    (tmp_path / "sparse" / "0").mkdir(parents=True)
    assert sparse_dir(tmp_path) == tmp_path / "sparse" / "0"


def test_sparse_dir_falls_back_to_the_flattened_model(tmp_path):
    # PGSR flattens sparse/0/ into sparse/ when it prepares its copy of the scene.
    (tmp_path / "sparse").mkdir()
    assert sparse_dir(tmp_path) == tmp_path / "sparse"


def test_read_split_returns_none_when_absent(tmp_path):
    assert read_split(tmp_path) is None


def test_write_split_reuses_an_existing_file(tmp_path):
    # A hand-authored split must survive, and reuse must not need a COLMAP model present.
    authored = {"train": ["a", "b"], "test": ["c"]}
    (tmp_path / SPLIT_FILENAME).write_text(json.dumps(authored))
    assert write_split(tmp_path) == authored


def test_write_split_requires_a_model(tmp_path):
    with pytest.raises(SceneError):
        write_split(tmp_path)


def test_copy_split_carries_the_file_into_a_prepared_scene(tmp_path):
    src, dest = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    authored = {"train": ["a"], "test": ["b"]}
    (src / SPLIT_FILENAME).write_text(json.dumps(authored))

    assert copy_split(src, dest) == dest / SPLIT_FILENAME
    assert read_split(dest) == authored


def test_copy_split_is_a_no_op_without_a_split(tmp_path):
    src, dest = tmp_path / "in", tmp_path / "out"
    src.mkdir()
    assert copy_split(src, dest) is None
    assert not dest.exists()
