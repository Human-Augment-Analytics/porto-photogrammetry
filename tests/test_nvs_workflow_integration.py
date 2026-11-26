"""NVS boundaries: no duplicate scoring, full view coverage and explicit mask staging."""
import argparse
import json
import sys
from pathlib import Path

import pytest

from augenblick.bench.scene import stage
from augenblick.bench.timing import run
from augenblick.cli.main import build_parser
from augenblick.core.errors import SceneError
from augenblick.eval.nvs import pair_views
from augenblick.eval.split import check_split


def test_workflow_can_disable_only_scoring():
    for backend in ('2dgs', 'pgsr', 'gw', 'milo', 'geosvr', 'meshroom'):
        parsed = build_parser().parse_args(['recon', backend, '--scene', '/scene',
                                            '--output', '/out', '--eval', '--no-score'])
        assert parsed.eval and not parsed.score


def test_partial_and_foreign_named_renders_are_rejected(tmp_path):
    for sub in ('renders', 'gt'):
        (tmp_path / sub).mkdir()
        (tmp_path / sub / 'train_view.png').touch()
    with pytest.raises(SceneError, match='cover the split'):
        pair_views(tmp_path, ['held_out'])


def test_split_must_be_a_partition_with_no_duplicates():
    for split in ({'train': ['a', 'a'], 'test': ['b']},
                  {'train': ['a'], 'test': ['b']}):
        with pytest.raises(SceneError):
            check_split(split, ['a', 'b', 'c'])


def test_command_failure_keeps_a_timing_record(tmp_path):
    path = tmp_path / 'timing.json'
    assert run([sys.executable, '-c', 'raise SystemExit(7)'], path) == 7
    record = json.loads(path.read_text())
    assert record['exit_code'] == 7 and record['seconds'] > 0


def test_staging_uses_selected_masks_without_editing_adopted_scene(tmp_path):
    sfm, selected, images, old = [tmp_path / n for n in ('sfm', 'new', 'photos', 'old')]
    for path in (sfm / 'sparse' / '0', selected, images, old):
        path.mkdir(parents=True)
    for name in ('cameras', 'images', 'points3D'):
        (sfm / 'sparse' / '0' / (name + '.bin')).write_bytes(b'model')
    (sfm / 'masks').symlink_to(old)
    out = tmp_path / 'private'
    stage(sfm, images, selected, out)
    assert (out / 'masks').resolve() == selected
    assert (sfm / 'masks').resolve() == old


def test_backend_pointcloud_cache_is_private(tmp_path):
    sfm = tmp_path / 'sfm'
    (sfm / 'sparse' / '0').mkdir(parents=True)
    for name in ('cameras', 'images', 'points3D'):
        (sfm / 'sparse' / '0' / (name + '.bin')).write_bytes(b'model')
    (sfm / 'sparse' / '0' / 'points3D.ply').write_bytes(b'unverified source cache')
    for sub in ('images', 'masks'):
        (sfm / sub).mkdir()
    out = tmp_path / 'private'
    stage(sfm, sfm / 'images', sfm / 'masks', out)
    assert not (out / 'sparse' / '0' / 'points3D.ply').exists()
    (out / 'sparse' / '0' / 'points3D.ply').write_bytes(b'backend cache')
    assert (sfm / 'sparse' / '0' / 'points3D.ply').read_bytes() == b'unverified source cache'


@pytest.mark.parametrize("score_enabled", [False, True])
def test_no_score_still_runs_backend_stages(tmp_path, monkeypatch, score_enabled):
    from types import SimpleNamespace
    from augenblick.reconstruction import base
    calls = []

    class Backend(base.SubprocessBackend):
        name = "dummy"
        title = "test"
        backend_dir = tmp_path
        config_cls = base.EvalParams
        def validate(self, scene):
            pass
        def stages(self, scene, output):
            return [base.Stage("render", ["render"])]
        def mesh_path(self, output):
            return output / "mesh.ply"
        def evaluate(self, scene, output):
            calls.append("score")
            return None

    monkeypatch.setattr(base, "write_split", lambda root: None)
    monkeypatch.setattr(base.process, "run", lambda *a, **kw: calls.append("render"))
    Backend(base.EvalParams(eval=True, score=score_enabled)).run(
        SimpleNamespace(root=tmp_path / "scene"), tmp_path / "out")
    assert calls == (["render", "score"] if score_enabled else ["render"])


def test_pgsr_refreshes_targets_and_invalidates_old_ply(tmp_path):
    from augenblick.reconstruction.pgsr import PgsrBackend,PgsrConfig
    from augenblick.core.scene import Scene
    source=tmp_path/'source';(source/'images').mkdir(parents=True)
    (source/'sparse/0').mkdir(parents=True)
    for n in ('cameras','images','points3D'):(source/'sparse/0'/(n+'.bin')).write_bytes(b'a')
    first=tmp_path/'masks_a';second=tmp_path/'masks_b';first.mkdir();second.mkdir()
    (source/'masks').symlink_to(first)
    out=tmp_path/'model';backend=PgsrBackend(PgsrConfig())
    backend.prepare(Scene(source),out)
    cache=out/'scene/sparse/points3D.ply';cache.write_bytes(b'old')
    (source/'masks').unlink();(source/'masks').symlink_to(second)
    (source/'sparse/0/points3D.bin').write_bytes(b'b')
    backend.prepare(Scene(source),out)
    assert (out/'scene/masks').resolve()==second
    assert not cache.exists()


def test_native_selection_manifest_rejects_heldout_training(tmp_path):
    from types import SimpleNamespace
    from augenblick.eval.selection import record_selection
    (tmp_path/'split.json').write_text(json.dumps({'train':['a'],'test':['b']}))
    cam=lambda n:SimpleNamespace(image_name=n,image_width=64,image_height=48)
    with pytest.raises(SceneError):record_selection(tmp_path,tmp_path,[cam('a'),cam('b')],[cam('b')])
    record_selection(tmp_path,tmp_path,[cam('a')],[cam('b')])
    assert list((tmp_path/'native_selections').glob('*.json'))
