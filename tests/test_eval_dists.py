"""Focused tests for DISTS cropping, input range, and CPU fallback."""
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from augenblick.eval.metrics import Dists


@pytest.mark.parametrize("mask_kind", ["specimen", "empty", "none"])
def test_dists_uses_mask_bounding_box_and_unit_range(monkeypatch, mask_kind):
    observed = {}

    class FakeDISTS:
        def __call__(self, first, second):
            observed["first"] = first
            observed["second"] = second
            observed["grad_enabled"] = torch.is_grad_enabled()
            return torch.tensor(0.25)

    fake_module = ModuleType("DISTS_pytorch")
    fake_module.DISTS = FakeDISTS
    monkeypatch.setitem(sys.modules, "DISTS_pytorch", fake_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    scorer = Dists()
    pred = torch.rand(3, 8, 9)
    truth = torch.rand(3, 8, 9)
    mask = None if mask_kind == "none" else torch.zeros(1, 8, 9)
    if mask_kind == "specimen":
        mask[:, 2:6, 3:8] = 1
        expected_pred, expected_truth = pred[:, 2:6, 3:8], truth[:, 2:6, 3:8]
    else:
        expected_pred, expected_truth = pred, truth

    assert scorer(pred, truth, mask) == 0.25
    torch.testing.assert_close(observed["first"], expected_pred.unsqueeze(0))
    torch.testing.assert_close(observed["second"], expected_truth.unsqueeze(0))
    assert observed["grad_enabled"] is False
    assert scorer._cpu_model() is scorer._cpu_model()


def test_dists_cuda_oom_retries_on_cached_cpu(monkeypatch):
    fake_module = ModuleType("DISTS_pytorch")
    gpu = Mock(side_effect=torch.cuda.OutOfMemoryError("view too large"))
    cpu = Mock(return_value=torch.tensor(0.125))
    initial_model = Mock()
    initial_model.cuda.return_value = gpu
    factory = Mock(side_effect=[initial_model, cpu])
    fake_module.DISTS = factory
    monkeypatch.setitem(sys.modules, "DISTS_pytorch", fake_module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda tensor: tensor)
    empty_cache = Mock()
    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    scorer = Dists()
    pred, truth = torch.rand(3, 8, 9), torch.rand(3, 8, 9)
    for _ in range(2):
        assert scorer(pred, truth) == 0.125
    assert factory.call_count == 2
    assert cpu.call_count == empty_cache.call_count == 2
    assert cpu.call_args.args[0].device.type == "cpu"
    torch.testing.assert_close(cpu.call_args.args[0][0], pred)