# Environment, Install, and GPU Notes

## Install (canonical)

There is **no `environment.yml`** any more (it was removed; older docs referencing it are stale).
Create the env yourself, then run the per-GPU installer:

```bash
conda create --name augenblick python=3.10
conda activate augenblick
git submodule update --init --recursive
bash scripts/auto_setup.sh          # detects GPU via nvidia-smi, dispatches to a wrapper
```

`scripts/` layout (added in `0ed266e`, which also deleted the old sbatch/`setup_sugar_b200.sh` scripts in `b1fccc1`):

| Script | GPU | sm | CUDA module | Torch |
|--------|-----|----|-------------|-------|
| `auto_setup.sh` | detect + dispatch | — | — | — |
| `setup_l40s.sh` | L40S / L40 / RTX 6000 Ada | 8.9 | `cuda/13.0.1` | 2.9.1 / cu130 |
| `setup_a100.sh` | A100 | 8.0 | `cuda/13.0.1` | 2.9.1 / cu130 |
| `setup_a40.sh` | A40 (48 GB) | 8.6 | `cuda/13.0.1` | 2.9.1 / cu130 |
| `setup_h100.sh` | H100 / H200 | 9.0 | `cuda/12.1.1` | 2.3.1 / cu121 |
| `setup_b200.sh` | B200 (original target) | 10.0 | `cuda/12.8` | 2.9.1 / cu130 |
| `setup_rtx_pro_6000.sh` | RTX Pro 6000 Blackwell | 12.0 | `cuda/13.0.2` | 2.9.1 / cu130 |

Wrappers only export `GPU_LABEL / GPU_ARCH / CUDA_MODULE / TORCH_SPEC / TORCH_INDEX_URL /
NUMPY_GENERATION` and `exec` into `scripts/setup_common.sh`, which does all the work (7 stages +
an import verification block). `auto_setup.sh` exits 2 with a copy-this-wrapper hint on an
unknown compute capability.

### `setup_common.sh` knobs

- `BACKENDS` (default `"sugar 2dgs pgsr gw"`) — subset of CUDA rasterizers to build.
- `SKIP_TETRA=1` — skip the fragile CGAL `tetra_triangulation` build (GW pivot extraction only).
- `PYTORCH3D_WHEEL=<url>` — install a prebuilt pytorch3d instead of the source build.

### Gotchas encoded in `setup_common.sh`

- **numpy generation follows the torch wheel** — see [numpy generations](#numpy-generations)
  below. `setup_common.sh` exports `PIP_CONSTRAINT` for the whole install so no transitive dep
  can drift numpy.
- **Stale `build/` dirs**: each rasterizer build does `rm -rf <pkg>/build <pkg>/*.egg-info`
  first — leftover objects from another GPU arch are silently reused and ignore
  `TORCH_CUDA_ARCH_LIST`.
- **No concurrent setups against one checkout** — they race on those `build/` dirs. Use a
  separate checkout per parallel build.
- `TORCH_CUDA_ARCH_LIST` is set to the wrapper's `GPU_ARCH`; import checks only validate the
  arch of the node the script ran on.
- nvdiffrast and tetra_triangulation failures are `WARN`-only (non-fatal).

### numpy generations

PyTorch wheels are compiled against a specific numpy C-ABI, and `scipy` / `scikit-learn` /
`scikit-image` (plus `imageio`, dragged along by scikit-image's floor) must match it. These
versions therefore live in `constraints/`, **not** `requirements.txt`, which leaves them
unpinned; each wrapper picks a file via `NUMPY_GENERATION` and `setup_common.sh` resolves it to
`constraints/numpy<N>.txt`:

| File | numpy | scipy | sklearn | skimage | imageio | Wrappers | torch |
|------|-------|-------|---------|---------|---------|----------|-------|
| `numpy1.txt` | 1.26.4 | 1.10.1 | 1.3.0 | 0.20.0 | 2.16.2 | l40s, h100 | 2.3.1 (cu121) |
| `numpy2.txt` | 2.2.6 | 1.15.3 | 1.6.1 | 0.25.2 | 2.37.0 | a100, b200, rtx_pro_6000 | 2.9.1 (cu130) |

A mismatch surfaces two ways, both at import, never at resolve time (the pins are lower bounds,
so pip accepts either):

- numpy 2 + numpy-1-built scipy/skimage → `ValueError: numpy.dtype size changed` or
  `AttributeError: _ARRAY_API not found`
- numpy-2-built torch + numpy 1 → `TypeError: expected np.ndarray (got numpy.ndarray)` from
  `torch.from_numpy`

`torch` is the immovable constraint: it cannot be rebuilt, so numpy follows it and the
scipy-family versions follow numpy. The verification block checks the installed numpy major
against `NUMPY_GENERATION` and exercises `torch.from_numpy`, which plain imports do not catch.

When adding a GPU wrapper, set `NUMPY_GENERATION` to match its torch wheel.

### Dependencies declared inside `src/libs/`

The editable installs carry their own dependency lists, which the constraint file must override:

- `src/libs/vggt/requirements.txt` (via `pyproject.toml` `dynamic = ["dependencies"]`) —
  upstream hard-pinned `numpy==1.26.4` here, which made the numpy-2 wrappers abort at stage 3
  with `ResolutionImpossible`. It is now unpinned locally; **keep it unpinned when syncing from
  upstream VGGT.**
- `src/libs/light_glue/requirements.txt` — `kornia>=0.6.11`, `opencv-python`, unpinned `torch`.
- `src/libs/pytorch3d` — `install_requires=["iopath"]`.
- `tetra_triangulation` — `trimesh>=3.20.2`.
- The eight CUDA rasterizers declare **no** Python deps; they are unaffected by numpy pins
  (though they must be rebuilt against the installed torch).

`torch`/`torchvision` appear unpinned in the vggt and lightglue lists, but stage 1 installs the
arch-specific `+cuXXX` wheel first and it satisfies those lower bounds, so it is not replaced.

`opencv-python` is pinned in `constraints/` because it otherwise floats to a 5.x major.
`requirements.txt` asks for plain `opencv-python`, not `opencv-contrib-python`: both install the
same `cv2` namespace and silently overwrite each other, and no code here uses contrib-only APIs
(`SIFT_create` moved into the base build in OpenCV 4.4).

### Manual setup

The manual pip sequence the scripts wrap is in [README.md](../../README.md) ("Manual setup").
VGGT is installed editable from `src/libs/vggt` (the package root with `pyproject.toml`; the
importable package is the inner `src/libs/vggt/vggt/`). An old editable install made from `src/`
must be redone from `src/libs/vggt`.

## Submodules

`.gitmodules` lists exactly three: `src/libs/light_glue`, `src/libs/pytorch3d`,
`src/libs/gaussian_wrapping/submodules/Depth-Anything-V2`.
**`src/libs/sugar` is not a submodule** (older docs claimed it was); it is vendored in-tree, as are
`src/libs/2dgs`, `src/libs/pgsr`, `src/libs/gaussian_wrapping`, `src/libs/vggt`.

## GPU notes

- Mixed precision: bfloat16 on Ampere+ (SM >= 8.0), float16 otherwise.
- Blackwell (B200, SM >= 10.0): `torch.compile(mode="max-autotune")` applied automatically.
- VGGT wants >= 80 GB VRAM for large scenes; COLMAP and the reconstruction backends run on
  32–40 GB (validated on an A100 PCIe 40 GB).
- Crash dumps (`core.colmap-*.ufhpc.*`) in the repo root are HPC artifacts — safe to delete.
