"""Exercise training reporting on CPU without importing GPU-only rasterizers."""
import ast
from pathlib import Path
import re
from types import SimpleNamespace

import pytest
import torch


TRAIN_PATH = Path(__file__).resolve().parents[1] / "src/libs/2dgs/train.py"


@pytest.mark.parametrize("background_value", [0.0, 1.0])
@pytest.mark.parametrize("masked", [False, True])
def test_training_report_uses_training_background(monkeypatch, capsys, background_value, masked):
    tree = ast.parse(TRAIN_PATH.read_text())
    report_node = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "training_report")
    namespace = {"torch": torch, "Scene": object}
    exec(compile(ast.Module(body=[report_node], type_ignores=[]), str(TRAIN_PATH), "exec"),
         namespace)

    tensor_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if args and args[0] == "cuda":
            args = ("cpu", *args[1:])
        return tensor_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    cameras = []
    expected_psnrs = []
    expected_losses = []
    for height, width in [(2, 4), (3, 5)]:
        truth = torch.full((3, height, width), 1.0 - background_value)
        mask = torch.zeros(1, height, width)
        mask[:, :, 0] = 1.0
        mask[:, :, 1] = 0.5
        truth[:, :, :2] = 0.4
        target = torch.where(mask == 0, background_value, truth) if masked else truth
        prediction = (target + 0.01).clamp(0.0, 1.0)
        cameras.append(SimpleNamespace(original_image=truth, gt_alpha_mask=mask if masked else None,
                                       prediction=prediction))
        expected_psnrs.append(-10 * torch.log10(((prediction - target) ** 2).mean()))
        expected_losses.append((prediction - target).abs().mean())

    scene = SimpleNamespace(gaussians=None, getTestCameras=lambda: cameras,
                            getTrainCameras=lambda: cameras)

    def render(camera, gaussians, pipeline, background):
        return {"render": camera.prediction}

    def psnr(prediction, target):
        return -10 * torch.log10(((prediction - target) ** 2).flatten(1).mean(1))

    namespace["psnr"] = psnr
    namespace["training_report"](
        None, 30000, None, None, lambda prediction, target: (prediction - target).abs().mean(),
        0, [30000], scene, render, (None, torch.full((3,), background_value)))

    output = capsys.readouterr().out
    match = re.search(r"Evaluating test: L1 (\S+) PSNR (\S+)", output)
    assert match is not None
    assert float(match[1]) == pytest.approx(torch.stack(expected_losses).mean().item())
    assert float(match[2]) == pytest.approx(torch.stack(expected_psnrs).mean().item())