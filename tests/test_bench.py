"""Tests for benchmark tree operations and provenance."""
import json
import os
import time

import pytest

from augenblick.bench.digest import digest_listing
from augenblick.bench.stamp import combined_digest, stamp
from augenblick.bench.tree import Tree
from augenblick.core.errors import SceneError


SPECIMENS = ["spec_a", "spec_b"]


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def make_tree(root, **overrides) -> Tree:
    config = {
        "name": "test",
        "repo": str(root / "repo"),
        "roster": {"keep": list(SPECIMENS), "discard": ["spec_dropped"]},
    }
    config.update(overrides)
    (root / "repo").mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(config))
    for specimen in SPECIMENS:
        images = root / "inputs" / "images" / specimen
        images.mkdir(parents=True)
        (images / "IMG_0001.JPG").write_bytes(b"photograph")
    return Tree(root)


@pytest.fixture
def tree(tmp_path):
    return make_tree(tmp_path / "tree")


# --- stamps ----------------------------------------------------------------

def test_an_identical_rerun_leaves_the_stamp_alone(tmp_path):
    """This is what stops a deterministic rerun cascading through the collection."""
    artefact = tmp_path / "detections.json"
    artefact.write_text('{"detections": 1374}')
    mark = tmp_path / "detections.stamp"

    assert stamp([artefact], mark) is True
    before = mark.read_text(), mark.stat().st_mtime

    time.sleep(0.01)
    artefact.write_text('{"detections": 1374}')  # same content, new timestamp
    assert stamp([artefact], mark) is False
    assert mark.read_text() == before[0]


def test_a_changed_artefact_changes_the_stamp(tmp_path):
    artefact = tmp_path / "scale.json"
    artefact.write_text('{"scale_mm_per_model_unit": 118.97}')
    mark = tmp_path / "scale.stamp"
    stamp([artefact], mark)

    artefact.write_text('{"scale_mm_per_model_unit": 120.01}')
    assert stamp([artefact], mark) is True


def test_a_stamp_carries_the_time_of_what_it_describes(tmp_path):
    """Dating a stamp to now would tell the engine everything below it is stale."""
    artefact = tmp_path / "result.json"
    artefact.write_text("{}")
    old = time.time() - 86_400
    os.utime(artefact, (old, old))
    mark = tmp_path / "result.stamp"

    stamp([artefact], mark)
    assert mark.stat().st_mtime == pytest.approx(old, abs=1)


def test_one_stamp_covers_a_stage_with_several_outputs(tmp_path):
    rigid = tmp_path / "geometry_rigid_scores.json"
    similarity = tmp_path / "geometry_similarity_scores.json"
    rigid.write_text('{"fscore": 0.933}')
    similarity.write_text('{"fscore": 0.941}')
    mark = tmp_path / "geometry.stamp"

    stamp([similarity, rigid], mark)
    unchanged = mark.read_text()

    rigid.write_text('{"fscore": 0.812}')
    assert stamp([similarity, rigid], mark) is True
    assert mark.read_text() != unchanged


def test_the_combined_digest_depends_on_which_file_holds_what(tmp_path):
    """Two outputs with swapped contents are not the same result."""
    one, two = tmp_path / "a.json", tmp_path / "b.json"
    one.write_text("1")
    two.write_text("2")
    before = combined_digest([one, two])
    one.write_text("2")
    two.write_text("1")
    assert combined_digest([one, two]) != before


# --- the roster and the write boundary --------------------------------------

def test_select_refuses_a_specimen_that_is_not_on_the_roster(tree):
    assert tree.select() == SPECIMENS
    assert tree.select(only=["spec_b"]) == ["spec_b"]
    with pytest.raises(SceneError):
        tree.select(only=["spec_dropped"])


def test_the_config_is_the_only_roster(tree):
    """A listing beside the config must not be able to override it."""
    (tree.root / "specimens.txt").write_text("spec_dropped\n")
    assert tree.select() == SPECIMENS


def test_only_and_index_together_are_refused_rather_than_silently_ranked(tree):
    with pytest.raises(SceneError):
        tree.select(only=["spec_b"], index=0)


def test_an_index_addresses_the_roster_in_order(tree):
    assert tree.select(index=0) == ["spec_a"]
    with pytest.raises(SceneError):
        tree.select(index=len(SPECIMENS))


def test_a_computed_output_may_not_be_a_symlink_into_another_tree(tree, tmp_path):
    elsewhere = tmp_path / "someone_elses_tree"
    elsewhere.mkdir()
    link = tree.path("markers", "spec_a")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(SceneError):
        tree.claim(link)


def test_a_deep_digest_sees_a_replacement_that_a_manifest_digest_misses(tree):
    images = tree.path("images", "spec_a")
    shallow = digest_listing(images)
    deep = digest_listing(images, deep=True)
    (images / "IMG_0001.JPG").write_bytes(b"different!")  # same name, same size
    assert digest_listing(images) == shallow
    assert digest_listing(images, deep=True) != deep


# --- tree-level operations -------------------------------------------------

def make_source_tree(root):
    """A stand-in for the group's data: one directory per specimen, plus a discard."""
    for specimen in SPECIMENS + ["spec_dropped"]:
        (root / "prepared" / specimen / "images").mkdir(parents=True)
        (root / "prepared" / specimen / "masks").mkdir(parents=True)
        (root / specimen / "colmap").mkdir(parents=True)
        (root / specimen / "colmap-2dgs").mkdir(parents=True)
    return root


def linked_tree(tmp_path):
    source = make_source_tree(tmp_path / "source")
    root = tmp_path / "linked"
    (root / "repo").mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({
        "name": "test",
        "repo": str(root / "repo"),
        "roster": {"keep": list(SPECIMENS), "discard": ["spec_dropped"]},
        "sources": {"received": str(source), "reconstructions": str(source)},
    }))
    return Tree(root), source


def test_links_builds_only_the_roster_and_leaves_the_source_untouched(tmp_path):
    from augenblick.bench import links

    tree, source = linked_tree(tmp_path)
    assert links.run(tree, Args(check=False)) == 0
    assert sorted(p.name for p in tree.stage("images").iterdir()) == SPECIMENS
    assert (tree.stage("sfm") / "spec_a").is_symlink()
    assert not (tree.stage("images") / "spec_dropped").exists()
    assert (source / "prepared" / "spec_dropped").is_dir()
    assert links.run(tree, Args(check=True)) == 0


def test_links_refuses_a_discarded_specimen_that_reappeared(tmp_path):
    from augenblick.bench import links

    tree, source = linked_tree(tmp_path)
    links.run(tree, Args(check=False))
    (tree.stage("images") / "spec_dropped").symlink_to(source / "prepared" / "spec_dropped")
    assert links.run(tree, Args(check=True)) == 1


def test_linking_refuses_a_tree_that_adopts_its_data(tmp_path):
    """Relinking from the sources would silently replace the adopted reconstruction."""
    from augenblick.bench import links

    tree, _ = linked_tree(tmp_path)
    tree.config["adopted_from"] = str(tmp_path / "another_tree")
    with pytest.raises(SceneError):
        links.run(tree, Args(check=True))


def test_the_audit_catches_a_computed_output_pointing_out_of_the_tree(tmp_path):
    from augenblick.bench import audit

    tree, source = linked_tree(tmp_path)
    escape = tree.stage("markers")
    escape.mkdir(parents=True)
    (escape / "spec_a").symlink_to(source, target_is_directory=True)
    assert audit.run(tree, Args(shared_trees=False)) == 1


def test_the_audit_passes_on_a_tree_that_only_links_inwards(tmp_path):
    from augenblick.bench import audit, links

    tree, _ = linked_tree(tmp_path)
    links.run(tree, Args(check=False))
    assert audit.run(tree, Args(shared_trees=False)) == 0


def test_a_replica_is_not_published_over_an_adopted_reconstruction(tmp_path):
    from augenblick.bench import links, recon

    tree, source = linked_tree(tmp_path)
    links.run(tree, Args(check=False))
    adopted = (tree.stage("sfm") / "spec_a").resolve()
    replica = tree.work / "replica" / "spec_a" / "colmap" / "sparse" / "0"
    replica.mkdir(parents=True)
    (replica / "cameras.bin").write_bytes(b"")
    tree.config["references"] = {"spec_a": "ref.obj"}

    assert recon.run(tree, Args(stage=None, status=True, only=None, index=None,
                                listing=None)) == 0
    assert (tree.stage("sfm") / "spec_a").resolve() == adopted

    tree.config["produce"] = ["spec_a"]
    assert recon.run(tree, Args(stage=None, status=True, only=None, index=None,
                                listing=None)) == 0
    assert (tree.stage("sfm") / "spec_a").resolve() == replica.parents[1]


# --- splits, now that the rules live in augenblick.eval.split ----------------

def test_a_holdout_rule_reproduces_the_convention_the_backends_already_read():
    from augenblick.eval.split import build_split, parse_rule

    assert parse_rule("holdout:8") == ("holdout", 8)
    stems = [f"IMG_{n}" for n in range(1000, 1032)]
    assert len(build_split(stems, 8)["test"]) == 4
    assert len(build_split(stems, 5)["test"]) == 7


def test_an_unusable_rule_is_refused_rather_than_approximated():
    from augenblick.eval.split import parse_rule

    for rule in ("holdout:1", "every-other", "holdout:x", "ring:1"):
        with pytest.raises(SceneError):
            parse_rule(rule)
    assert parse_rule("manual") == ("manual", None)


def test_a_hand_written_split_is_checked_against_the_reconstruction():
    from augenblick.eval.split import check_split

    stems = [f"IMG_{n}" for n in range(1000, 1010)]
    good = {"train": stems[:7], "test": stems[7:]}
    assert check_split(dict(good), stems) == good

    for broken in ({"train": stems[:7], "test": ["IMG_9999"]},
                   {"train": stems, "test": stems[:2]},
                   {"train": stems, "test": []},
                   {"train": stems}):
        with pytest.raises(SceneError):
            check_split(broken, stems)
