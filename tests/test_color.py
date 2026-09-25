"""Colour calibration math, grid recovery, and output safety."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from augenblick.preparation.color import (
    CalibrationConfig,
    CameraChart,
    ColorCalibrationError,
    apply_color_matrix,
    calibrate_directory,
    fit_color_matrix,
    linear_to_srgb,
    load_config,
    locate_patch_centers,
    sample_patches,
    srgb_to_linear,
)


def synthetic_chart() -> np.ndarray:
    chart = np.full((400, 600, 3), 20, dtype=np.uint8)
    colors = np.linspace(35, 235, 24, dtype=np.uint8)
    for index, value in enumerate(colors):
        row, column = divmod(index, 6)
        center_x = 75 + 90 * column - 3 * row
        center_y = 70 + 80 * row + 2 * column
        color = (
            int(value),
            int((int(value) * 3) % 220 + 25),
            int((int(value) * 7) % 220 + 25),
        )
        cv2.rectangle(
            chart,
            (center_x - 30, center_y - 24),
            (center_x + 30, center_y + 24),
            color,
            -1,
        )
    return chart


def test_srgb_round_trip():
    values = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(values)), values, atol=1e-6)


def test_fit_recovers_known_linear_matrix():
    rng = np.random.default_rng(4)
    source = rng.uniform(0.1, 0.8, size=(24, 3)).astype(np.float32)
    expected = np.array(
        [[1.05, -0.02, 0.01], [0.01, 0.96, 0.02], [-0.01, 0.03, 1.02]],
        dtype=np.float32,
    )
    target = linear_to_srgb(srgb_to_linear(source) @ expected)
    actual = fit_color_matrix(source, target, ridge=0.0)
    assert np.allclose(actual, expected, atol=1e-5)


def test_patch_grid_is_recovered_from_chart():
    chart = synthetic_chart()
    centers = locate_patch_centers(chart)
    samples = sample_patches(chart, centers)
    assert centers.shape == (24, 2)
    assert samples.shape == (24, 3)
    assert np.all(np.diff(centers[:6, 0]) > 60)


def test_apply_identity_is_exact():
    image = np.arange(12 * 8 * 3, dtype=np.uint8).reshape(12, 8, 3)
    assert np.array_equal(apply_color_matrix(image, np.eye(3, dtype=np.float32)), image)


def test_nonempty_output_requires_overwrite(tmp_path):
    output = tmp_path.parent / f"{tmp_path.name}_existing"
    output.mkdir()
    (output / "keep.txt").write_text("user data")
    config = CalibrationConfig(
        reference_camera="camera1",
        camera_regex=r"camera\d+",
        cameras={
            "camera1": CameraChart(
                tmp_path / "missing.jpg",
                np.zeros((4, 2), np.float32),
            )
        },
    )
    with pytest.raises(ColorCalibrationError, match="not empty"):
        calibrate_directory(tmp_path, output, config)


def test_config_normalizes_camera_names_and_rejects_parent_paths(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "reference_camera": "Camera1",
        "cameras": {
            "Camera1": {
                "reference_image": "camera1_ref.png",
                "corners": [[0, 0], [10, 0], [10, 10], [0, 10]],
            }
        },
    }))
    assert load_config(path).reference_camera == "camera1"

    data = json.loads(path.read_text())
    data["cameras"]["Camera1"]["reference_image"] = "../outside.png"
    path.write_text(json.dumps(data))
    with pytest.raises(ColorCalibrationError, match="inside input"):
        load_config(path)


def test_absolute_target_requires_named_24_patch_srgb_values(tmp_path):
    path = tmp_path / "config.json"
    data = {
        "reference_camera": "camera1",
        "cameras": {
            "camera1": {
                "reference_image": "camera1_ref.png",
                "corners": [[0, 0], [10, 0], [10, 10], [0, 10]],
            }
        },
        "target_srgb_d65": [[0.5, 0.5, 0.5]] * 24,
    }
    path.write_text(json.dumps(data))
    with pytest.raises(ColorCalibrationError, match="target_name"):
        load_config(path)

    data["target_name"] = "Verified chart values, edition X"
    path.write_text(json.dumps(data))
    config = load_config(path)
    assert config.target_srgb_d65.shape == (24, 3)
    assert config.target_name == "Verified chart values, edition X"

    data["target_srgb_d65"] = [[1.1, 0.5, 0.5]] * 24
    path.write_text(json.dumps(data))
    with pytest.raises(ColorCalibrationError, match="normalized"):
        load_config(path)


def test_output_cannot_be_inside_input(tmp_path):
    config = CalibrationConfig(
        reference_camera="camera1",
        camera_regex=r"camera\d+",
        cameras={
            "camera1": CameraChart(
                Path("camera1_ref.png"),
                np.zeros((4, 2), np.float32),
            )
        },
    )
    with pytest.raises(ColorCalibrationError, match="inside"):
        calibrate_directory(tmp_path, tmp_path / "output", config)


def test_reference_camera_images_are_copied_exactly(tmp_path):
    chart = synthetic_chart()
    reference = tmp_path / "camera1_ref.png"
    capture = tmp_path / "camera1_capture.png"
    cv2.imwrite(str(reference), chart)
    cv2.imwrite(str(capture), chart)
    config = CalibrationConfig(
        reference_camera="camera1",
        camera_regex=r"camera\d+",
        cameras={
            "camera1": CameraChart(
                Path(reference.name),
                np.float32([[0, 0], [599, 0], [599, 399], [0, 399]]),
            )
        },
    )

    output = tmp_path.parent / f"{tmp_path.name}_calibrated"
    report = calibrate_directory(tmp_path, output, config)

    assert report["processed_images"]["camera1"] == 1
    assert (output / capture.name).read_bytes() == capture.read_bytes()
    assert not (output / reference.name).exists()


def test_absolute_target_calibrates_reference_camera(tmp_path):
    chart = synthetic_chart()
    reference = tmp_path / "camera1_ref.png"
    capture = tmp_path / "camera1_capture.png"
    cv2.imwrite(str(reference), chart)
    cv2.imwrite(str(capture), chart)
    centers = locate_patch_centers(chart)
    measured = sample_patches(chart, centers)
    target = np.clip(measured * np.array([0.92, 1.03, 1.08]), 0.0, 1.0)
    config = CalibrationConfig(
        reference_camera="camera1",
        camera_regex=r"camera\d+",
        cameras={
            "camera1": CameraChart(
                Path(reference.name),
                np.float32([[0, 0], [599, 0], [599, 399], [0, 399]]),
            )
        },
        target_srgb_d65=target.astype(np.float32),
        target_name="Synthetic sRGB-D65 target",
    )

    output = tmp_path.parent / f"{tmp_path.name}_absolute"
    report = calibrate_directory(tmp_path, output, config)

    assert report["calibration_mode"] == "absolute_srgb_d65"
    assert report["target_name"] == "Synthetic sRGB-D65 target"
    assert report["metrics"]["camera1"]["mean_delta_e76_after"] < report["metrics"]["camera1"]["mean_delta_e76_before"]
    assert not np.allclose(report["matrices"]["camera1"], np.eye(3))


def test_report_values_are_json_serializable():
    matrix = fit_color_matrix(
        np.full((24, 3), 0.4, np.float32),
        np.full((24, 3), 0.4, np.float32),
    )
    json.dumps({"matrix": matrix.tolist()})
