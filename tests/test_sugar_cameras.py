"""SuGaR camera conversion must preserve heterogeneous COLMAP intrinsics."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SUGAR_DIR = Path(__file__).resolve().parents[1] / "src" / "libs" / "sugar"
sys.path.insert(0, str(SUGAR_DIR))
sys.path.insert(0, str(SUGAR_DIR / "gaussian_splatting"))

from sugar_scene.cameras import convert_camera_from_gs_to_pytorch3d


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SuGaR cameras construct CUDA transforms")
def test_pytorch3d_conversion_preserves_per_view_focal_lengths():
    cameras = [
        SimpleNamespace(
            R=torch.eye(3).numpy(), T=torch.zeros(3).numpy(),
            FoVx=0.5, FoVy=0.4, image_width=1600, image_height=1066),
        SimpleNamespace(
            R=torch.eye(3).numpy(), T=torch.zeros(3).numpy(),
            FoVx=0.8, FoVy=0.7, image_width=1600, image_height=1066),
    ]

    converted = convert_camera_from_gs_to_pytorch3d(cameras)

    assert converted.K.shape[0] == 2
    assert converted.K[0, 0, 0] != converted.K[1, 0, 0]
    assert converted.K[0, 1, 1] != converted.K[1, 1, 1]