#!/bin/bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
UPSTREAM="${NEURALANGELO_ROOT:-$REPO/../neuralangelo}"
[[ -f "$UPSTREAM/train.py" ]]
if [[ "${1:-}" != --python-only ]]; then
    "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "Run setup in a GPU allocation"; print(torch.__version__, torch.cuda.get_device_name())'
    command -v nvcc
fi
CONSTRAINTS="$(mktemp)"
BUILD_ENV=""
trap 'rm -f "$CONSTRAINTS"; if [[ -n "$BUILD_ENV" ]]; then rm -rf "$BUILD_ENV"; fi' EXIT
"$PYTHON" -m pip freeze --exclude-editable > "$CONSTRAINTS"
"$PYTHON" -m pip install --constraint "$CONSTRAINTS" \
    wandb==0.31.0 diskcache==5.6.3 termcolor==2.5.0 nvidia-ml-py==12.560.30
if [[ "${1:-}" == --python-only ]]; then
    exit 0
fi
if ! "$PYTHON" -c 'from importlib.metadata import distribution; import torch; package = distribution("tinycudann"); architecture = "".join(map(str, torch.cuda.get_device_capability())); assert package.version == "2.0"; assert any(f"_{architecture}_C." in str(path) for path in package.files); import tinycudann'; then
    BUILD_ENV="$(mktemp -d)"
    "$PYTHON" -m venv --system-site-packages "$BUILD_ENV/venv"
    "$BUILD_ENV/venv/bin/python" -m pip install --no-deps setuptools==80.9.0
    "$BUILD_ENV/venv/bin/python" -m pip wheel --no-build-isolation --no-deps --wheel-dir "$BUILD_ENV/wheels" \
        'git+https://github.com/NVlabs/tiny-cuda-nn.git@3c7931caab7b691c6f30dffe20952ed727d935b9#subdirectory=bindings/torch'
    "$PYTHON" -m pip install --no-deps --ignore-installed "$BUILD_ENV"/wheels/tinycudann-*.whl
fi
PYTHONPATH="$REPO/src:$UPSTREAM" "$PYTHON" - <<'PY'
import torch
import tinycudann as tcnn
from imaginaire.config import Config
from imaginaire.trainers.utils.get_trainer import get_trainer
from augenblick.reconstruction.neuralangelo_data import Dataset
encoder = tcnn.Encoding(3, {"otype": "HashGrid", "n_levels": 2,
                           "n_features_per_level": 2, "log2_hashmap_size": 12,
                           "base_resolution": 16, "per_level_scale": 2.0})
points = torch.rand(32, 3, device="cuda", requires_grad=True)
values = encoder(points)
values.square().mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(values).all() and torch.isfinite(points.grad).all()
print("NEURALANGELO_CUDA_BACKWARD_OK")
PY