"""Generate a redistributable three-camera input for `augenblick color`."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

WIDTH = 600
HEIGHT = 400
PATCH_COLORS_BGR = [
    (68, 82, 115), (130, 150, 194), (157, 122, 98), (67, 108, 87),
    (177, 128, 133), (170, 189, 103), (44, 126, 214), (166, 91, 80),
    (99, 90, 193), (108, 60, 94), (64, 188, 157), (46, 163, 224),
    (150, 61, 56), (73, 148, 70), (60, 54, 175), (31, 199, 231),
    (149, 86, 187), (161, 133, 8), (243, 243, 243), (200, 200, 200),
    (160, 160, 160), (122, 122, 121), (85, 85, 85), (52, 52, 52),
]


def chart() -> np.ndarray:
    """Create a high-contrast 4 by 6 colour chart."""
    image = np.full((HEIGHT, WIDTH, 3), 20, dtype=np.uint8)
    for index, color in enumerate(PATCH_COLORS_BGR):
        row, column = divmod(index, 6)
        center_x = 75 + 90 * column - 3 * row
        center_y = 70 + 80 * row + 2 * column
        cv2.rectangle(
            image,
            (center_x - 30, center_y - 24),
            (center_x + 30, center_y + 24),
            color,
            -1,
        )
    return image


def cast(image: np.ndarray, gains: tuple[float, float, float]) -> np.ndarray:
    """Apply a deterministic BGR camera cast for demonstration purposes."""
    scaled = image.astype(np.float32) * np.asarray(gains, dtype=np.float32)
    output = np.clip(np.round(scaled), 0, 255).astype(np.uint8)
    chart_background = np.all(image == 20, axis=2)
    output[chart_background] = image[chart_background]
    return output


def capture() -> np.ndarray:
    """Create a simple synthetic specimen-like capture."""
    image = np.full((HEIGHT, WIDTH, 3), 180, dtype=np.uint8)
    cv2.ellipse(image, (300, 210), (155, 105), -12, 0, 360, (72, 126, 176), -1)
    cv2.circle(image, (250, 180), 28, (45, 65, 95), -1)
    cv2.circle(image, (345, 225), 38, (105, 165, 205), -1)
    cv2.line(image, (170, 280), (430, 115), (38, 78, 115), 12)
    return image


def main() -> None:
    root = Path(__file__).resolve().parent
    input_dir = root / "generated" / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    gains = {
        "camera1": (1.0, 1.0, 1.0),
        "camera2": (0.82, 1.04, 1.16),
        "camera3": (1.12, 0.91, 0.86),
    }
    reference = chart()
    specimen = capture()
    config = {
        "reference_camera": "camera1",
        "camera_regex": "camera[0-9]+",
        "ridge": 1e-6,
        "cameras": {},
    }
    for name, camera_gains in gains.items():
        reference_name = f"{name}_chart.png"
        reference_canvas = np.full((600, 800, 3), 235, dtype=np.uint8)
        reference_canvas[100:500, 100:700] = cast(reference, camera_gains)
        cv2.imwrite(str(input_dir / reference_name), reference_canvas)
        cv2.imwrite(str(input_dir / f"{name}_capture.png"), cast(specimen, camera_gains))
        mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        cv2.ellipse(mask, (300, 210), (170, 120), -12, 0, 360, 255, -1)
        cv2.imwrite(str(input_dir / f"{name}_capture.mask.png"), mask)
        config["cameras"][name] = {
            "reference_image": reference_name,
            "corners": "auto",
        }

    config_path = root / "generated" / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"Wrote synthetic input to {input_dir}")
    print(f"Wrote configuration to {config_path}")


if __name__ == "__main__":
    main()
