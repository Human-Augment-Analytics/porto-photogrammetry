"""Backend argv is dictated by upstream train.py/render.py and must not drift.

Expected argv is derived by reading the pre-refactor wrapper scripts, not by running them.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from augenblick.core.errors import SceneError
from augenblick.core.scene import Scene
from augenblick.reconstruction.base import LIBS_DIR, REPO_ROOT, require_registered_masks
from augenblick.reconstruction.geosvr import GeosvrBackend, GeosvrConfig
from augenblick.reconstruction.gw import GWBackend, GWConfig
from augenblick.reconstruction.milo import (
    RENDER_SCRIPT as MILO_RENDER_SCRIPT, SDF_SCRIPT as MILO_SDF_SCRIPT,
    TRAIN_FAST_SCRIPT as MILO_TRAIN_FAST_SCRIPT,
    TRAIN_REGULAR_SCRIPT as MILO_TRAIN_REGULAR_SCRIPT, TSDF_SCRIPT as MILO_TSDF_SCRIPT,
    MiloBackend, MiloConfig)
from augenblick.reconstruction.pgsr import PgsrBackend, PgsrConfig
from augenblick.reconstruction.sugar import SugarBackend, SugarConfig
from augenblick.reconstruction.twodgs import TwoDGSBackend, TwoDGSConfig

SCENE = Scene(Path("/scene"))
OUT = Path("/out")


def test_repo_root_resolves_to_checkout():
    assert (REPO_ROOT / "pyproject.toml").is_file()
    assert LIBS_DIR == REPO_ROOT / "src" / "libs"


def test_2dgs_argv():
    backend = TwoDGSBackend(TwoDGSConfig(unbounded=True))
    train, render = backend.stages(SCENE, OUT)
    assert train.cmd == [
        sys.executable, str(LIBS_DIR / "2dgs" / "train.py"),
        "-s", "/scene", "-m", "/out",
        "--iterations", "30000",
        "--test_iterations", "7000", "30000",
        "--save_iterations", "7000", "30000",
        "--lambda_dist", "0.0",
        "--lambda_normal", "0.05",
        "--depth_ratio", "0.0",
        "--densify_grad_threshold", "0.0002",
        "--densify_until_iter", "15000",
        "--opacity_cull", "0.05",
        "--lambda_mask_hull", "0.0",
        "--mask_dilate_kernel", "9",
        "--lambda_hull", "0.0",
        "--hull_anneal_start_iter", "500",
        "--hull_anneal_end_iter", "15000",
    ]
    assert render.cmd == [
        sys.executable, str(LIBS_DIR / "2dgs" / "render.py"),
        "-s", "/scene", "-m", "/out",
        "--voxel_size", "-1.0",
        "--depth_trunc", "-1.0",
        "--sdf_trunc", "-1.0",
        "--num_cluster", "50",
        "--mesh_res", "4096",
        "--skip_test",
        "--unbounded",
    ]
    assert backend.mesh_path(OUT) == OUT / "train" / "ours_30000" / "fuse_post.ply"


def test_sugar_argv_uses_string_booleans():
    backend = SugarBackend(SugarConfig(high_poly=True))
    gs, sugar = backend.stages(SCENE, OUT)
    assert gs.cmd == [
        sys.executable, str(LIBS_DIR / "sugar" / "gaussian_splatting" / "train.py"),
        "-s", "/scene", "-m", "/out/gs_model",
        "--iterations", "20000",
        "--densify_grad_threshold", "0.0002",
        "--densify_until_iter", "15000",
        "--lambda_dssim", "0.2",
        "--sh_degree", "3",
    ]
    assert sugar.cmd == [
        sys.executable, str(LIBS_DIR / "sugar" / "train.py"),
        "-s", "/scene", "-c", "/out/gs_model", "-o", "/out/sugar",
        "-i", "7000",
        "-r", "dn_consistency",
        "-l", "0.1",
        "-v", "1000000",
        "-g", "1",
        "-f", "15000",
        "--square_size", "4",
        "--eval", "False",
        "--gpu", "0",
        "--high_poly", "True",
    ]
    assert backend.mesh_path(OUT) == OUT / "sugar" / "refined_mesh" / "scene"


def test_pgsr_argv_render_has_no_scene_flag():
    backend = PgsrBackend(PgsrConfig(skip_mesh=True))
    train, render = backend.stages(Scene(Path("/out/scene")), OUT)
    assert train.cmd == [
        sys.executable, str(LIBS_DIR / "pgsr" / "train.py"),
        "-s", "/out/scene", "-m", "/out",
        "--iterations", "30000",
        "--test_iterations", "7000", "30000",
        "--save_iterations", "7000", "30000",
        "--max_abs_split_points", "0",
        "--opacity_cull_threshold", "0.05",
        "--lambda_dssim", "0.2",
        "--single_view_weight", "0.015",
        "--multi_view_ncc_weight", "0.15",
        "--multi_view_geo_weight", "0.03",
        "--multi_view_num", "8",
        "--densify_grad_threshold", "0.0002",
        "--densify_until_iter", "15000",
    ]
    # skip_mesh maps to --skip_train on render, matching upstream's spelling.
    assert render.cmd == [
        sys.executable, str(LIBS_DIR / "pgsr" / "render.py"),
        "-m", "/out",
        "--max_depth", "10.0",
        "--voxel_size", "0.001",
        "--num_cluster", "1",
        "--skip_test",
        "--skip_train",
    ]
    assert backend.mesh_path(OUT) == OUT / "mesh" / "tsdf_fusion_post.ply"


def test_pgsr_prepare_flattens_sparse(tmp_path):
    scene_root = tmp_path / "in"
    (scene_root / "images").mkdir(parents=True)
    (scene_root / "masks").mkdir()
    (scene_root / "sparse" / "0").mkdir(parents=True)
    (scene_root / "images" / "a.jpg").touch()
    (scene_root / "sparse" / "0" / "cameras.bin").touch()

    out = tmp_path / "out"
    backend = PgsrBackend(PgsrConfig())
    prepared = backend.prepare(Scene(scene_root), out)

    assert prepared.root == out / "scene"
    assert (out / "scene" / "sparse" / "cameras.bin").is_file()
    assert not (out / "scene" / "sparse" / "0").exists()

    # A second call reuses the prepared copy rather than re-copying.
    assert backend.prepare(Scene(scene_root), out).root == out / "scene"


def test_gw_argv_and_passthrough():
    backend = GWBackend(GWConfig(resolution=2), passthrough=["--foo", "1"])
    train, extract, texture = backend.stages(SCENE, OUT)
    assert train.cmd == [
        sys.executable, str(LIBS_DIR / "gaussian_wrapping" / "train.py"),
        "--rasterizer", "ours",
        "-s", "/scene", "-m", "/out",
        "--feature_dc_lr", "0.0013",
        "--feature_rest_lr", "0.00011",
        "--position_lr_init", "0.00016",
        "--position_lr_final", "1.6e-06",
        "--position_lr_delay_mult", "0.01",
        "--position_lr_max_steps", "30000",
        "--opacity_lr", "0.05",
        "--scaling_lr", "0.005",
        "--rotation_lr", "0.001",
        "--appearance_embeddings_lr", "0.001",
        "--appearance_network_lr", "0.001",
        "--gaussian_features_lr", "0.025",
        "--pgsr_appearance_lr", "0.001",
        "--exposure_compensation",
        "--data_device", "cpu",
        "--iterations", "30000",
        "--sh_degree", "3",
        "--N_max_gaussians", "6000000",
        "--densify_until_iter", "15000",
        "--densify_grad_threshold", "0.0002",
        "--lambda_depth_normal", "0.05",
        "--multiview_factor", "1.0",
        "-r", "2",
        "--foo", "1",
    ]
    assert extract.cmd == [
        sys.executable, str(LIBS_DIR / "gaussian_wrapping" / "pivot_based_mesh_extraction.py"),
        "--sdf_mode", "ours",
        "--rasterizer", "ours",
        "--dtype", "int32",
        "-s", "/scene", "-m", "/out",
        "--n_pivots", "2",
        "--std_factor", "3.0",
        "--n_binary_steps", "10",
        "--isosurface_value", "0.0",
        "--iteration", "30000",
        "--use_valid_mask",
        "--data_device", "cpu",
        "--postprocess",
        "--filter_large_edges",
        "-r", "2",
    ]
    assert texture.cmd == [
        sys.executable, str(LIBS_DIR / "gaussian_wrapping" / "texture_mesh.py"),
        "--rasterizer", "ours",
        "-s", "/scene", "-m", "/out",
        "--mesh", "/out/mesh_ours_2pivots_post.ply",
        "--iteration", "30000",
        "--n_iter", "1000",
        "--lambda_dssim", "0.2",
        "--lr", "0.0025",
        "--sh_degree_for_texturing", "0",
        "-r", "2",
    ]
    # Passthrough reaches only the training stage.
    assert "--foo" not in extract.cmd and "--foo" not in texture.cmd


def test_gw_mesh_paths_track_flags():
    assert GWBackend(GWConfig()).mesh_path(OUT) == OUT / "mesh_ours_2pivots_post.ply"
    assert GWBackend(GWConfig(postprocess=False)).mesh_path(OUT) == OUT / "mesh_ours_2pivots.ply"
    assert GWBackend(GWConfig(n_pivots=4)).mesh_path(OUT) == OUT / "mesh_ours_4pivots_post.ply"
    assert (GWBackend(GWConfig()).textured_mesh_path(OUT)
            == OUT / "mesh_ours_2pivots_post_texture_refined_999.ply")


def test_gw_extract_iteration_defaults_to_iterations():
    assert GWBackend(GWConfig(iterations=7000)).extract_iteration == 7000
    assert GWBackend(GWConfig(iterations=7000, extract_iteration=3000)).extract_iteration == 3000


def test_gw_uses_no_cwd():
    assert GWBackend.use_cwd is False
    assert TwoDGSBackend.use_cwd is True


def test_2dgs_eval_renders_held_out_views():
    train, render = TwoDGSBackend(TwoDGSConfig(eval=True)).stages(SCENE, OUT)
    assert "--eval" in train.cmd
    assert "--eval" in render.cmd
    assert "--skip_test" not in render.cmd


def test_pgsr_eval_renders_held_out_views():
    train, render = PgsrBackend(PgsrConfig(eval=True)).stages(SCENE, OUT)
    assert "--eval" in train.cmd
    assert "--skip_test" not in render.cmd


def test_pgsr_resolution_reaches_training_only():
    train, render = PgsrBackend(PgsrConfig(resolution=2)).stages(SCENE, OUT)
    assert train.cmd[-2:] == ["-r", "2"]
    # render.py reads the resolution back from the model's cfg_args, as it does the scene path.
    assert "-r" not in render.cmd


def test_gw_eval_disables_exposure_and_renders_before_extraction():
    stages = GWBackend(GWConfig(eval=True)).stages(SCENE, OUT)
    assert [s.name for s in stages] == [
        "Training", "Held-out rendering", "Mesh extraction", "Texture refinement"]

    train = stages[0].cmd
    # Exposure compensation fits one exposure per training camera, so it cannot be evaluated.
    assert "--no-exposure_compensation" in train
    assert "--exposure_compensation" not in train
    assert "--eval" in train

    assert stages[1].cmd == [
        sys.executable, str(LIBS_DIR / "gaussian_wrapping" / "render.py"),
        "--rasterizer", "ours",
        "-s", "/scene", "-m", "/out",
        "--iteration", "30000",
        "--eval",
        "--skip_train",
    ]


def test_gw_keeps_exposure_compensation_when_not_evaluating():
    train, _, _ = GWBackend(GWConfig()).stages(SCENE, OUT)
    assert "--exposure_compensation" in train.cmd
    assert "--no-exposure_compensation" not in train.cmd


def test_eval_enabled_tracks_the_config_field():
    assert TwoDGSBackend(TwoDGSConfig(eval=True)).eval_enabled is True
    assert TwoDGSBackend(TwoDGSConfig()).eval_enabled is False
    # SuGaR declares no eval field, so it can never be asked for held-out metrics.
    assert SugarBackend(SugarConfig()).eval_enabled is False


def test_test_renders_root_is_the_shared_3dgs_layout():
    for backend in (TwoDGSBackend(TwoDGSConfig()), PgsrBackend(PgsrConfig()),
                    GWBackend(GWConfig())):
        assert backend.test_renders_root(OUT) == OUT / "test"


GEOSVR = LIBS_DIR / "geosvr"


def _geosvr_scene(root: Path, size=(800, 600), masked=("a", "b")) -> Scene:
    """A scene with two images, masks for the given stems, and a non-empty sparse/0."""
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir()
    (root / "sparse" / "0").mkdir(parents=True)
    (root / "sparse" / "0" / "cameras.bin").write_bytes(b"")
    for stem in ("a", "b"):
        Image.new("RGB", size).save(root / "images" / f"{stem}.jpg")
    for stem in masked:
        Image.new("L", size).save(root / "masks" / f"{stem}.png")
    return Scene(root)


def test_geosvr_argv(tmp_path):
    scene = _geosvr_scene(tmp_path / "scene")
    backend = GeosvrBackend(GeosvrConfig())
    train, render, mesh = backend.stages(scene, OUT)
    assert train.cmd == [
        sys.executable, str(GEOSVR / "train.py"),
        "-s", str(scene.root), "-m", "/out",
        "--cfg_files", str(GEOSVR / "cfg" / "dtu_mesh.yaml"),
        "--sche_mult", "1.0",
        "--lambda_depthanythingv2", "0.1",
        "--enable_da2_uncertainty",
        "--power_level_uncertainty", "1.0",
        "--voxel_dropout_min", "0.5",
        "--multi_view_ncc_weight", "0.05",
        "--multi_view_geo_weight", "0.01",
        "--lambda_rectify", "1e-05",
        "--lambda_scaling_penalty", "1e-06",
        "--lambda_mask", "0.5",
        "--no-white_background",
        "--black_background",
        "--blend_mask",
        "--no-eval",
        "--res_downscale", "0.0",
    ]
    assert render.cmd == [sys.executable, str(GEOSVR / "render.py"), "-m", "/out", "--skip_test"]
    assert mesh.cmd == [
        sys.executable, str(GEOSVR / "mesh_extract" / "tsdf_mesh.py"),
        "-m", "/out",
        "--voxel_size", "0.001",
        "--max_depth", "5.0",
        "--sdf_trunc_scale", "4.0",
        "--num_cluster", "1",
    ]
    assert backend.mesh_path(OUT) == OUT / "mesh" / "tsdf" / "tsdf_fusion_post.ply"
    assert backend.test_renders_root(OUT) == OUT / "test"


def test_geosvr_bool_flags(tmp_path):
    scene = _geosvr_scene(tmp_path / "scene")
    config = GeosvrConfig(da2_uncertainty=False, white_background=True, eval=True)
    train = GeosvrBackend(config).stages(scene, OUT)[0].cmd
    assert "--no-enable_da2_uncertainty" in train
    assert "--white_background" in train and "--no-black_background" in train
    assert "--eval" in train and "--no-eval" not in train


def test_geosvr_no_prior_also_drops_the_mono_normal_loss(tmp_path):
    scene = _geosvr_scene(tmp_path / "scene")
    train = GeosvrBackend(GeosvrConfig(lambda_depthanythingv2=0.0)).stages(scene, OUT)[0].cmd
    assert train[-2:] == ["--lambda_normal_dmono", "0.0"]
    assert "--lambda_normal_dmono" not in GeosvrBackend(GeosvrConfig()).stages(scene, OUT)[0].cmd


def test_geosvr_config_cli_patch():
    """Patch 3d against the vendored config: the CLI always beats the preset yaml."""
    pytest.importorskip("yacs")
    code = """
import json, sys
from src.config import cfg, update_config
update_config(["cfg/dtu_mesh.yaml"], sys.argv[1:])
print(json.dumps({"blend_mask": cfg.data.blend_mask, "eval": cfg.data.eval,
                  "da2u": cfg.regularizer.enable_da2_uncertainty,
                  "photo": cfg.regularizer.lambda_photo,
                  "source_path": cfg.data.source_path, "model_path": cfg.model.model_path}))
"""

    def parse(*flags):
        # cwd=GEOSVR so upstream's top-level `src` package cannot resolve to this repo's src/.
        out = subprocess.run([sys.executable, "-c", code, *flags], cwd=GEOSVR,
                             capture_output=True, text=True)
        return out.returncode, (json.loads(out.stdout.splitlines()[-1]) if out.returncode == 0 else None)

    _, got = parse("--no-blend_mask", "--no-enable_da2_uncertainty")
    assert got["blend_mask"] is False and got["da2u"] is False
    _, got = parse("--blend_mask", "--eval", "-s", "/scene", "-m", "/out")
    # DTU sets blend_mask False; the flag must still win.
    assert got["blend_mask"] is True and got["eval"] is True
    assert got["source_path"] == "/scene" and got["model_path"] == "/out"
    _, got = parse("--no-eval")
    assert got["eval"] is False
    # DTU sets lambda_photo 0.1; an explicit value equal to the src/config.py default (1.0) wins.
    _, got = parse("--lambda_photo", "1.0")
    assert got["photo"] == 1.0
    code_, _ = parse("--no_such_flag", "1")
    assert code_ != 0


def test_geosvr_eval_also_renders_held_out_views(tmp_path):
    scene = _geosvr_scene(tmp_path / "scene")
    stages = GeosvrBackend(GeosvrConfig(eval=True)).stages(scene, OUT)
    assert [s.name for s in stages] == ["GeoSVR training", "Rendering", "TSDF mesh extraction"]
    assert stages[0].cmd[-2:] == ["--test_iterations", "-1"]
    # One call renders both splits: held-out views for scoring, train views for TSDF colour.
    assert stages[1].cmd == [sys.executable, str(GEOSVR / "render.py"), "-m", "/out"]


def test_geosvr_always_renders_train_views_for_tsdf(tmp_path):
    scene = _geosvr_scene(tmp_path / "scene")
    stages = GeosvrBackend(GeosvrConfig()).stages(scene, OUT)
    assert [s.name for s in stages] == ["GeoSVR training", "Rendering", "TSDF mesh extraction"]
    assert stages[1].cmd[-1] == "--skip_test"


def test_geosvr_tsdf_auto_switches_script(tmp_path):
    scene = _geosvr_scene(tmp_path / "scene")
    config = GeosvrConfig(tsdf="auto", use_depth_filter=True)
    mesh = GeosvrBackend(config).stages(scene, OUT)[-1].cmd
    assert mesh == [
        sys.executable, str(GEOSVR / "mesh_extract" / "tsdf_mesh_360.py"),
        "-m", "/out",
        "--radius_scale", "2.0",
        "--num_cluster", "1",
        "--use_depth_filter",
    ]


def test_geosvr_reads_scene_in_place(tmp_path):
    scene = _geosvr_scene(tmp_path / "scene")
    out = tmp_path / "out"
    backend = GeosvrBackend(GeosvrConfig())
    before = sorted(p for p in tmp_path.rglob("*"))
    prepared = backend.prepare(scene, out)
    stages = backend.stages(prepared, out)
    assert prepared is scene
    assert stages[0].cmd[2:4] == ["-s", str(scene.root)]
    assert sorted(p for p in tmp_path.rglob("*")) == before


def test_geosvr_patches_present():
    """Guards the input-scene, mask and script-interface patches against a careless resync."""
    train = (GEOSVR / "train.py").read_text()
    render = (GEOSVR / "render.py").read_text()
    assert "source_path=cfg.data.source_path" not in train
    assert "source_path=data_pack.source_path" not in render
    assert '"multi_view.json"), \'w\')' not in (GEOSVR / "src" / "dataloader" / "data_pack.py").read_text()
    loader = (GEOSVR / "src" / "dataloader" / "colmap_loader.py").read_text()
    reader = loader[loader.index("def read_colmap_ply"):]
    assert "storePly(" not in reader and "open(cor_path, 'w')" not in reader
    # The non-RGBA branch must not copy cam_info.mask, or the dedicated mask block raises.
    assert "        mask = cam_info.mask\n" not in (GEOSVR / "src" / "dataloader" / "data_pack.py").read_text()
    for script in ("tsdf_mesh.py", "tsdf_mesh_360.py"):
        text = (GEOSVR / "mesh_extract" / script).read_text()
        assert text.index("sys.path.insert(0,") < text.index("from src")
    for text in (render, *((GEOSVR / "mesh_extract" / s).read_text()
                           for s in ("tsdf_mesh.py", "tsdf_mesh_360.py"))):
        assert "add_argument('model_path')" not in text
        assert "add_argument('-m', '--model_path', required=True)" in text
    assert 'if hasattr(view, "depthanythingv2"):' in train
    # The depth prior must ignore the composited background, or it builds floaters there.
    losses = (GEOSVR / "src" / "utils" / "loss_utils.py").read_text()
    assert losses.count("outside_specimen(cam, alpha.shape)") == 2
    # A compact masked cloud never thins to the target density; upstream then raised IndexError.
    bounding = (GEOSVR / "src" / "utils" / "bounding_utils.py").read_text()
    assert "if len(below) else len(dist) - 1" in bounding
    # Mixed image sizes: no stacked depths, and each view fused at its own size.
    for script in ("tsdf_mesh.py", "tsdf_mesh_360.py"):
        text = (GEOSVR / "mesh_extract" / script).read_text()
        assert "torch.stack(depths_tsdf_fusion" not in text
        assert "            H, W = ref_depth.shape\n" in text


def test_geosvr_mask_loss_requires_every_mask(tmp_path, monkeypatch):
    scene = _geosvr_scene(tmp_path / "scene", masked=("a",))
    monkeypatch.setattr("augenblick.reconstruction.base.registered_stems", lambda _: ["a", "b"])
    with pytest.raises(SceneError, match="lambda_mask"):
        GeosvrBackend(GeosvrConfig()).validate(scene)
    GeosvrBackend(GeosvrConfig(lambda_mask=0.0)).validate(scene)


def test_geosvr_mask_loss_ignores_unregistered_images(tmp_path, monkeypatch):
    # Masked COLMAP drops unmasked images from the model, and GeoSVR never loads them.
    scene = _geosvr_scene(tmp_path / "scene", masked=("a",))
    monkeypatch.setattr("augenblick.reconstruction.base.registered_stems", lambda _: ["a"])
    GeosvrBackend(GeosvrConfig()).validate(scene)


def test_geosvr_resolution_maps_long_side(tmp_path):
    scene = _geosvr_scene(tmp_path / "scene", size=(4000, 3000))
    assert GeosvrBackend(GeosvrConfig()).res_downscale(scene) == 2.5
    assert GeosvrBackend(GeosvrConfig(resolution=2)).res_downscale(scene) == 2.0


MILO = LIBS_DIR / "milo" / "milo"


@pytest.fixture
def milo_tetra(monkeypatch):
    """Pretend GW's tetranerf build is installed, so validate() runs on any env."""
    monkeypatch.setattr("augenblick.reconstruction.milo.tetranerf_installed", lambda: True)


def test_milo_argv():
    backend = MiloBackend(MiloConfig())
    train, extract = (s.cmd for s in backend.stages(SCENE, OUT))
    assert train == [
        sys.executable, str(MILO_TRAIN_REGULAR_SCRIPT),
        "-s", "/scene", "-m", "/out",
        "--imp_metric", "indoor", "--rasterizer", "radegs",
        "--mesh_config", "default_dtu", "--iterations", "30000",
        "--lambda_mask", "0.1", "--data_device", "cpu",
        "--test_iterations", "-1", "--checkpoint_iterations", "-1",
        "-r", "-1",
    ]
    assert extract == [
        sys.executable, str(MILO_SDF_SCRIPT), "-m", "/out",
        "--iteration", "30000", "--rasterizer", "radegs", "--config", "default_dtu",
        "--imp_metric", "indoor", "--refine_iter", "1000",
    ]
    assert backend.mesh_path(OUT) == OUT / "mesh_learnable_sdf.ply"
    assert backend.backend_dir == MILO and backend.use_cwd


def test_milo_fast_recipe_defaults():
    train, extract = (s.cmd for s in MiloBackend(MiloConfig(densification="fast")).stages(SCENE, OUT))
    assert train[1] == str(MILO_TRAIN_FAST_SCRIPT)
    assert train[train.index("--iterations") + 1] == "18000"
    assert train[train.index("--mesh_config") + 1] == "default"
    # Extraction must load what training saved, with the same mesh config.
    assert extract[extract.index("--iteration") + 1] == "18000"
    assert extract[extract.index("--config") + 1] == "default"


def test_milo_overrides_reach_train_and_extract():
    cfg = MiloConfig(iterations=5000, densify_until_iter=3000, regularization_from_iter=2000,
                     eval=True, white_background=True, dense_gaussians=True,
                     decoupled_appearance=True, remove_oof_vertices=True, resolution=2)
    train, _, extract = (s.cmd for s in MiloBackend(cfg).stages(SCENE, OUT))
    for flag, value in (("--iterations", "5000"), ("--densify_until_iter", "3000"),
                        ("--regularization_from_iter", "2000"), ("-r", "2")):
        assert train[train.index(flag) + 1] == value
    assert {"--eval", "--white_background", "--dense_gaussians", "--decoupled_appearance"} <= set(train)
    assert extract[extract.index("--iteration") + 1] == "5000" and "--remove_oof_vertices" in extract


def test_milo_mesh_config_path_reaches_both_stages(tmp_path, milo_tetra):
    scene = _geosvr_scene(tmp_path / "scene")
    yaml = tmp_path / "smoke.yaml"
    yaml.write_text("start_iter: 3501\n")
    train, extract = (s.cmd for s in MiloBackend(MiloConfig(mesh_config=str(yaml))).stages(scene, OUT))
    assert train[train.index("--mesh_config") + 1] == str(yaml)
    assert extract[extract.index("--config") + 1] == str(yaml)
    with pytest.raises(ValueError, match="does not exist"):
        MiloBackend(MiloConfig(mesh_config=str(tmp_path / "missing.yaml"))).validate(scene)


def test_milo_eval_renders_before_extraction():
    stages = MiloBackend(MiloConfig(eval=True)).stages(SCENE, OUT)
    assert [s.name for s in stages] == ["MILo training", "Held-out rendering", "Mesh extraction"]
    assert stages[1].cmd == [sys.executable, str(MILO_RENDER_SCRIPT), "-m", "/out",
                             "--rasterizer", "radegs", "--skip_train"]
    assert "--eval" in stages[0].cmd


def test_milo_without_eval_has_two_stages():
    stages = MiloBackend(MiloConfig()).stages(SCENE, OUT)
    assert [s.name for s in stages] == ["MILo training", "Mesh extraction"]
    assert "--eval" not in stages[0].cmd


def test_milo_tsdf_switches_script_and_mesh_path():
    backend = MiloBackend(MiloConfig(extract="tsdf", mesh_res=512))
    assert backend.stages(SCENE, OUT)[-1].cmd == [
        sys.executable, str(MILO_TSDF_SCRIPT), "-m", "/out",
        "--rasterizer", "radegs", "--mesh_res", "512", "--num_cluster", "50",
    ]
    assert backend.mesh_path(OUT) == OUT / "mesh_regular_tsdf_res512_post.ply"


def test_milo_mask_loss_requires_every_registered_mask(tmp_path, monkeypatch, milo_tetra):
    scene = _geosvr_scene(tmp_path / "scene", masked=("a",))
    monkeypatch.setattr("augenblick.reconstruction.base.registered_stems", lambda _: ["a", "b"])
    with pytest.raises(SceneError, match="lambda_mask"):
        MiloBackend(MiloConfig()).validate(scene)
    MiloBackend(MiloConfig(lambda_mask=0.0)).validate(scene)
    monkeypatch.setattr("augenblick.reconstruction.base.registered_stems", lambda _: ["a"])
    MiloBackend(MiloConfig()).validate(scene)


def test_milo_gof_refuses_mask_loss(tmp_path, milo_tetra):
    # GOF renders no alpha, so the mask loss would silently do nothing.
    scene = _geosvr_scene(tmp_path / "scene")
    with pytest.raises(ValueError, match="radegs"):
        MiloBackend(MiloConfig(rasterizer="gof")).validate(scene)
    MiloBackend(MiloConfig(rasterizer="gof", lambda_mask=0.0)).validate(scene)


def test_milo_requires_tetranerf(tmp_path, monkeypatch):
    # Upstream only warns without it, then dies at the first Delaunay step, hours in.
    monkeypatch.setattr("augenblick.reconstruction.milo.tetranerf_installed", lambda: False)
    with pytest.raises(SceneError, match="tetranerf"):
        MiloBackend(MiloConfig(lambda_mask=0.0)).validate(_geosvr_scene(tmp_path / "scene"))


def test_milo_reads_scene_in_place(tmp_path):
    scene = _geosvr_scene(tmp_path / "scene")
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    backend = MiloBackend(MiloConfig(eval=True))
    assert backend.prepare(scene, tmp_path / "out") is scene
    train = backend.stages(scene, tmp_path / "out")[0].cmd
    assert train[train.index("-s") + 1] == str(scene.root)
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before


def test_milo_read_config_cli_patch(tmp_path):
    """Patch 3e against the vendored arguments module: a CLI flag beats configs/fast."""
    cfg = tmp_path / "fast"
    cfg.write_text("Namespace(iterations=18_000, densify_until_iter=3_000)")
    code = f"""
import json, sys
from argparse import ArgumentParser
from arguments import read_config
p = ArgumentParser()
p.add_argument("--iterations", type=int, default=30000)
p.add_argument("--densify_until_iter", type=int, default=15000)
p.add_argument("--resolution", "-r", type=int, default=-1)
p.add_argument("--config_path", type=str, default={str(cfg)!r})
ns = read_config(p)
print(json.dumps([ns.iterations, ns.densify_until_iter, ns.resolution]))
"""

    def parse(*flags):
        # cwd=MILO so upstream's top-level `arguments` package is the one imported.
        out = subprocess.run([sys.executable, "-c", code, *flags], cwd=MILO,
                             capture_output=True, text=True, check=True)
        return json.loads(out.stdout.splitlines()[-1])

    assert parse() == [18000, 3000, -1]
    assert parse("--iterations", "2000") == [2000, 3000, -1]
    # Explicit values equal to the config, and the --x=v spelling, still count as given.
    assert parse("--iterations", "18000", "--densify_until_iter=500") == [18000, 500, -1]
    assert parse("-r", "2") == [18000, 3000, 2]


def test_milo_patches_present():
    """Guards the input-scene, socket, rename and mesh-config patches against a careless resync."""
    files = {p: p.read_text() for p in MILO.rglob("*.py")}
    readers = (MILO / "scene" / "dataset_readers.py").read_text()
    assert "    storePly(ply_path, xyz, rgb)\n" not in readers
    assert "split.json" in readers and 'masks_folder = os.path.join' in readers
    assert "mask_path" in (MILO / "utils" / "camera_utils.py").read_text()
    for trainer in ("train.py", "train_regular_densification.py"):
        text = (MILO / trainer).read_text()
        assert "network_gui.init(" not in text
        assert 'args.mesh_config.endswith(".yaml")' in text
        assert "opt.lambda_mask" in text
    assert 'args.config.endswith(".yaml")' in (MILO / "mesh_extract_sdf.py").read_text()
    assert "self.lambda_mask = 0.0" in (MILO / "arguments" / "__init__.py").read_text()
    assert "min(cluster_to_keep, len(cluster_n_triangles))" in (MILO / "utils" / "regular_tsdf_utils.py").read_text()
    # SuGaR owns diff_gaussian_rasterization; nothing in MILo may import that name.
    for path, text in files.items():
        assert "from diff_gaussian_rasterization import" not in text, path
        assert "import diff_gaussian_rasterization\n" not in text, path
    setup = (MILO.parent / "submodules" / "diff-gaussian-rasterization-radegs" / "setup.py").read_text()
    assert 'name="diff_gaussian_rasterization_radegs._C"' in setup
    assert (MILO.parent / "submodules" / "diff-gaussian-rasterization-radegs"
            / "diff_gaussian_rasterization_radegs" / "__init__.py").is_file()
    patched = [p for p, t in files.items() if "LOCAL PATCH (augenblick)" in t]
    assert len(patched) == 10


def test_require_registered_masks(tmp_path, monkeypatch):
    scene = _geosvr_scene(tmp_path / "scene", masked=("a",))
    monkeypatch.setattr("augenblick.reconstruction.base.registered_stems", lambda _: ["a", "b"])
    require_registered_masks(scene, 0.0)
    with pytest.raises(SceneError, match=r"1 of 2 lack masks/<stem>.png \(e.g. \['b'\]\)"):
        require_registered_masks(scene, 0.5)


def test_milo_fast_refuses_regularising_during_densification(tmp_path, milo_tetra):
    # train.py's regularised renderer returns no area_max, which densification still reads.
    scene = _geosvr_scene(tmp_path / "scene")
    with pytest.raises(ValueError, match="regularization_from_iter"):
        MiloBackend(MiloConfig(densification="fast", regularization_from_iter=2000)).validate(scene)
    with pytest.raises(ValueError, match="regularization_from_iter"):
        MiloBackend(MiloConfig(densification="fast", densify_until_iter=4000)).validate(scene)
    # lambda_mask 0 keeps these off the mask check, which reads a real COLMAP model.
    MiloBackend(MiloConfig(densification="fast", lambda_mask=0.0)).validate(scene)
    MiloBackend(MiloConfig(densification="fast", densify_until_iter=2000,
                           regularization_from_iter=2000, lambda_mask=0.0)).validate(scene)
    # The regular trainer densifies with RaDe-GS throughout, so it has no such coupling.
    MiloBackend(MiloConfig(densify_until_iter=3000, regularization_from_iter=2000,
                           lambda_mask=0.0)).validate(scene)


def test_milo_tsdf_default_resolution():
    # Upstream's 1024 spans the camera distance, leaving a turntable specimen ~150 voxels wide.
    backend = MiloBackend(MiloConfig(extract="tsdf"))
    cmd = backend.stages(SCENE, OUT)[-1].cmd
    assert cmd[cmd.index("--mesh_res") + 1] == "4096"
    assert backend.mesh_path(OUT) == OUT / "mesh_regular_tsdf_res4096_post.ply"
