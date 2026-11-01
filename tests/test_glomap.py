"""Global mapping reuses the masked front end and preserves the scene contract."""
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from augenblick.cli.main import build_parser
from augenblick.core.errors import SceneError
from augenblick.core.scene import Scene
from augenblick.sfm.colmap import ColmapConfig, ColmapSfM
from augenblick.sfm.glomap import GlomapSfM


@pytest.fixture
def api(monkeypatch):
    fake = SimpleNamespace(
        ImageReaderOptions=SimpleNamespace,
        FeatureExtractionOptions=SimpleNamespace,
        FeatureMatchingOptions=SimpleNamespace,
        IncrementalPipelineOptions=SimpleNamespace,
        GlobalPipelineOptions=lambda: SimpleNamespace(
            num_threads=-1, mapper=SimpleNamespace(
                num_threads=-1, bundle_adjustment=SimpleNamespace(
                    ceres=SimpleNamespace(solver_options=SimpleNamespace(num_threads=-1))))),
        CameraMode=SimpleNamespace(PER_IMAGE="per_image"),
        extract_features=Mock(), match_exhaustive=Mock(),
        global_mapping=Mock(return_value={}), incremental_mapping=Mock(return_value={}),
    )
    monkeypatch.setitem(sys.modules, "pycolmap", fake)
    monkeypatch.setattr("os.sched_getaffinity", lambda process: {2, 3})
    return fake


@pytest.fixture
def scene(tmp_path):
    root = tmp_path / "input"
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    (root / "images/specimen.jpg").touch()
    (root / "masks/specimen.png").touch()
    return Scene(root)


def test_global_dispatch_and_thread_limits(api):
    assert GlomapSfM(ColmapConfig())._map(api, "db", "images", "sparse") == {}
    options = api.global_mapping.call_args.kwargs["options"]
    api.global_mapping.assert_called_once_with("db", "images", "sparse", options=options)
    assert options.num_threads == options.mapper.num_threads == 2
    assert options.mapper.bundle_adjustment.ceres.solver_options.num_threads == 2
    api.incremental_mapping.assert_not_called()


def test_optional_thread_fields(api):
    api.GlobalPipelineOptions = SimpleNamespace
    GlomapSfM(ColmapConfig())._map(api, "db", "images", "sparse")
    assert vars(api.global_mapping.call_args.kwargs["options"]) == {}


def test_incremental_colmap_dispatch_unchanged(api):
    ColmapSfM(ColmapConfig())._map(api, "db", "images", "sparse")
    options = api.incremental_mapping.call_args.kwargs["options"]
    api.incremental_mapping.assert_called_once_with("db", "images", "sparse", options=options)
    assert options.num_threads == 2
    api.global_mapping.assert_not_called()


def test_glomap_cli_uses_existing_feature_options():
    args = build_parser().parse_args([
        "sfm", "glomap", "--scene", "input", "--output", "output",
        "--max_image_size", "1200", "--camera_model", "PINHOLE",
    ])
    assert args.method == "glomap"
    assert args.max_image_size == 1200
    assert args.camera_model == "PINHOLE"


@pytest.mark.parametrize("field", ["global_mapping", "GlobalPipelineOptions"])
@pytest.mark.parametrize("value", [None, False])
def test_unavailable_api_fails_before_extraction(scene, api, tmp_path, field, value):
    setattr(api, field, value)
    with pytest.raises(SceneError, match="GLOMAP_UNAVAILABLE"):
        GlomapSfM(ColmapConfig()).run(scene, tmp_path / "output")
    api.extract_features.assert_not_called()
    api.incremental_mapping.assert_not_called()


def test_missing_api_fails_before_extraction(scene, api, tmp_path):
    del api.global_mapping
    with pytest.raises(SceneError, match="GLOMAP_UNAVAILABLE"):
        GlomapSfM(ColmapConfig()).run(scene, tmp_path / "output")
    api.extract_features.assert_not_called()


def test_empty_global_result_does_not_fall_back(scene, api, tmp_path):
    with pytest.raises(SceneError, match="GLOMAP_FAIL"):
        GlomapSfM(ColmapConfig()).run(scene, tmp_path / "output")
    api.incremental_mapping.assert_not_called()


def test_global_result_preserves_masked_scene_contract(scene, api, tmp_path):
    small, best = Mock(), Mock()
    small.num_reg_images.return_value = 1
    best.num_reg_images.return_value = 2
    best.num_points3D.return_value = 100
    api.global_mapping.return_value = {0: small, 1: best}
    output = tmp_path / "output"
    result = GlomapSfM(ColmapConfig(max_image_size=1200)).run(scene, output)
    best.write.assert_called_once_with(str(output / "sparse/0"))
    assert result.scene.root == output
    assert result.num_images == 2 and result.num_points == 100
    assert (output / "images").resolve() == scene.images_dir
    assert (output / "masks").resolve() == scene.masks_dir
    reader = api.extract_features.call_args.kwargs["reader_options"]
    assert reader.camera_model == "SIMPLE_PINHOLE"
    assert reader.mask_path == str(output / "masks_colmap")
    assert (output / "masks_colmap/specimen.jpg.png").is_symlink()
    extraction = api.extract_features.call_args.kwargs["extraction_options"]
    assert extraction.num_threads == 2 and extraction.max_image_size == 1200
    assert api.match_exhaustive.call_args.kwargs["matching_options"].num_threads == 2
    api.incremental_mapping.assert_not_called()