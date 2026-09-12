"""Backend argv is dictated by upstream train.py/render.py and must not drift.

Expected argv is derived by reading the pre-refactor wrapper scripts, not by running them.
"""
import sys
from pathlib import Path

from augenblick.core.scene import Scene
from augenblick.reconstruction.base import LIBS_DIR, REPO_ROOT
from augenblick.reconstruction.gw import GWBackend, GWConfig
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
