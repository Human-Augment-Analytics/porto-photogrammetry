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

Every wrapper builds the **same toolchain** — `cuda/12.9.1` + torch 2.9.1 (cu129) + numpy 2 —
and differs only in `GPU_ARCH`. One CUDA generation is deliberate: `onnxruntime-gpu` links CUDA
12 sonames, so a cu130 torch leaves its GPU provider unloadable (see the masking section below).
`nvcc` 12.9 compiles sm_80 through sm_120, so no arch needs a different module.

| Script | GPU | sm |
|--------|-----|----|
| `auto_setup.sh` | detect + dispatch | — |
| `setup_l40s.sh` | L40S / L40 / RTX 6000 Ada | 8.9 |
| `setup_a100.sh` | A100 | 8.0 |
| `setup_a40.sh` | A40 (48 GB) | 8.6 |
| `setup_h100.sh` | H100 / H200 | 9.0 |
| `setup_b200.sh` | B200 (original target) | 10.0 |
| `setup_rtx_pro_6000.sh` | RTX Pro 6000 Blackwell | 12.0 |

Wrappers only export `GPU_LABEL / GPU_ARCH / CUDA_MODULE / TORCH_SPEC / TORCH_INDEX_URL /
NUMPY_GENERATION` and `exec` into `scripts/setup_common.sh`, which does all the work (7 stages +
an import verification block). `auto_setup.sh` exits 2 with a copy-this-wrapper hint on an
unknown compute capability.

### `setup_common.sh` knobs

- `BACKENDS` (default `"sugar 2dgs pgsr gw"`) — subset of CUDA rasterizers to build.
- `SKIP_TETRA=1` — skip the fragile CGAL `tetra_triangulation` build (GW pivot extraction only).
- `PYTORCH3D_WHEEL=<url>` — install a prebuilt pytorch3d instead of the source build.

### Masking dependencies live in `requirements.txt`

`rembg==2.0.68` + `onnxruntime-gpu==1.23.2` + `opencv-python-headless==4.11.0.86` are ordinary
entries in `requirements.txt`, so **every** env any wrapper builds carries them. Two pins there
are load-bearing and must not be "tidied" back to their obvious names:

- **`onnxruntime-gpu`, not `onnxruntime`.** The two are different distributions that unpack into
  the same `onnxruntime/` directory. Listing both installs both, with no pip error, and whichever
  lands last wins — which is how the GPU provider silently disappears.
- **`opencv-python-headless`, not `opencv-python`.** Same story for `cv2/`; and the version must
  stay pinned to 4.11.0.86, since `rembg`'s unpinned floor otherwise pulls 5.0.0.93, the 5.x
  major this repo avoids.
- **`onnxruntime-gpu` 1.23.2 links CUDA 12 and the cu130 toolchain does not satisfy it.** It
  needs `libcufft.so.11` and `libcudnn.so.9`; `cuda/13.0.1` (and the `nvidia-*` wheels torch
  drags in) ship `.so.12`, so the CUDA provider fails to `dlopen` and ORT silently runs on CPU.
  This is why the wrappers pin **cu129, not cu130**: torch 2.9.1+cu129 bundles
  `nvidia-cudnn-cu12` 9.10.2.21 and `nvidia-cufft-cu12` 11.4.1.4, which *are* the `.so.9` and
  `.so.11` ORT links. Same CUDA generation for torch and ORT means no extra wheels, no
  `LD_LIBRARY_PATH` hook and no second CUDA module — keep them aligned when bumping either.
- **Do not "just bump" `onnxruntime-gpu`: 1.23.2 is the last release with a cp310 wheel.**
  Every release from 1.24.1 onward ships cp311–cp314 only (`requires_python >=3.11`), so on this
  repo's Python 3.10 pip silently resolves no further than 1.23.2. Those newer releases are also
  **CUDA 13** (`nvidia-cuda-runtime~=13.0`, `nvidia-cudnn-cu13~=9.0`), so the cu129 pin above is
  a consequence of the Python floor, not a preference — upstream has moved on. **If this repo
  ever moves to Python 3.11+, the alignment flips**: bump ORT past 1.24 and move every wrapper
  back to `cuda/13.x` + a cu130 torch index. Verified against PyPI wheel tags and
  `requires_dist` on 2026-09-13.
- **`get_available_providers()` is not proof.** It lists providers compiled into the wheel even
  when the shared library cannot load, so it reports `CUDAExecutionProvider` on a broken install
  and rembg then runs 4x slower with no error. Check the *session* instead:
  `python -c "from rembg import new_session; print(new_session('isnet-general-use').inner_session.get_providers())"`
  — and run it on a **GPU node**, since a login node fails with `CUDA failure 35 / GPU=-1`
  regardless. Worth running as a pre-flight guard in any masking job, so a CPU fallback shows up
  in the log instead of the run quietly taking 4x longer. `masking/rembg.py` logs exactly this
  check at session creation (a warning, not a hard failure).
- `rembg` is an ordinary `requirements.txt` entry, so every env any wrapper builds has the
  masking stage available — there is no special-purpose masking env to build or keep in sync.

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
| `numpy2.txt` | 2.2.6 | 1.15.3 | 1.6.1 | 0.25.2 | 2.37.0 | **all six** | 2.9.1 (cu129) |
| `numpy1.txt` | 1.26.4 | 1.10.1 | 1.3.0 | 0.20.0 | 2.16.2 | *none* | 2.3.1 (cu121) |

`numpy1.txt` currently has **no consumers** — every wrapper is `NUMPY_GENERATION=2` since h100
moved off torch 2.3.1. It is kept for anyone who needs to pin an older torch back; delete it
only once that is ruled out.

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
  upstream VGGT.** Upstream also listed plain `onnxruntime`, which was **removed locally** —
  stage 3/8 installs VGGT *after* stage 2/8's `requirements.txt`, so that entry silently
  reinstalled the CPU onnxruntime over `onnxruntime-gpu` and cost `CUDAExecutionProvider` (both
  share one `onnxruntime/` directory, so uninstalling the CPU package alone breaks the import
  entirely). Nothing under `src/libs/vggt/` imports it, and the top-level `requirements.txt`
  pin still serves `visual_util.py`'s skyseg path. **Keep it out when syncing from upstream.**
- `src/libs/light_glue/requirements.txt` — `kornia>=0.6.11`, `opencv-python`, unpinned `torch`.
- `src/libs/pytorch3d` — `install_requires=["iopath"]`.
- `tetra_triangulation` — `trimesh>=3.20.2`.
- The eight CUDA rasterizers declare **no** Python deps; they are unaffected by numpy pins
  (though they must be rebuilt against the installed torch).

`torch`/`torchvision` appear unpinned in the vggt and lightglue lists, but stage 1 installs the
arch-specific `+cuXXX` wheel first and it satisfies those lower bounds, so it is not replaced.

**OpenCV: `requirements.txt` and the vendored libs ask for `opencv-python-headless`; the two
submodules still ask for plain `opencv-python`.** `constraints/` therefore pins **both** at
4.11.0.86. That is not redundancy:

- `opencv-python`, `opencv-python-headless` and `opencv-contrib-python` all install the same
  `cv2/` namespace and silently overwrite each other — only the version cap makes the outcome
  predictable whichever lands last.
- `light_glue` declares `opencv-python` **unpinned**, so without its own cap it resolves to
  5.0.0.93 (a real 5.x wheel exists for cp310 — it is `cp37-abi3`, so "no 5.x wheel" is not the
  reason to avoid it; the reason is that the vendored backends are unaudited against a major
  bump). `gaussian_wrapping` pins 4.11.0.86 itself.
- Both are **submodules**, so their requirements cannot be changed from this repo — the cap in
  `constraints/` is the only lever. Drop either pin and a backend install drags in 5.x.

No code here uses contrib-only APIs (`SIFT_create` moved into the base build in OpenCV 4.4).
Headless is safe: the only `imshow`/`waitKey` calls in the tree are already commented out or sit
in `src/libs/pytorch3d/docs/examples/`, which nothing in the pipeline imports.

### Manual setup

The manual pip sequence the scripts wrap is in [README.md](../../README.md) ("Manual setup").
VGGT is installed editable from `src/libs/vggt` (the package root with `pyproject.toml`; the
importable package is the inner `src/libs/vggt/vggt/`). An old editable install made from `src/`
must be redone from `src/libs/vggt`.

## Submodules

`.gitmodules` lists exactly three: `src/libs/light_glue`, `src/libs/pytorch3d`,
`src/libs/gaussian_wrapping/submodules/Depth-Anything-V2`. Note that
`src/libs/gaussian_wrapping` **itself** is also a gitlink in the index (`git ls-files --stage`
shows mode 160000), so edits inside it do not commit with this repo either — check with
`git ls-files --stage src/libs/<name>` before editing a backend, not against this list alone.
**`src/libs/sugar` is not a submodule** (older docs claimed it was); it is vendored in-tree, as are
`src/libs/2dgs`, `src/libs/pgsr`, `src/libs/gaussian_wrapping`, `src/libs/vggt`.

## GPU notes

- Mixed precision: bfloat16 on Ampere+ (SM >= 8.0), float16 otherwise.
- Blackwell (B200, SM >= 10.0): `torch.compile(mode="max-autotune")` applied automatically.
- VGGT wants >= 80 GB VRAM for large scenes; COLMAP and the reconstruction backends run on
  32–40 GB (validated on an A100 PCIe 40 GB).
- Crash dumps (`core.colmap-*.ufhpc.*`) in the repo root are HPC artifacts — safe to delete.

### A faulty GPU imitates a broken install

An ECC-faulted card makes ORT's CUDA init fail **silently, with no dlopen error**, so rembg
falls back to CPU exactly as a CUDA-generation mismatch would — but nothing is wrong with the
env, and rebuilding it fixes nothing. Observed on PACE node `atl1-1-03-004-21-0`, one bad card
among several good ones on the same node, 311,614 `uncorrectable ECC error encountered` lines
in a single job.

Distinguish them before touching the env, since the fixes are opposite:

| Symptom | Bad card | Env/CUDA mismatch |
|---|---|---|
| dlopen error in the log | **absent** | names the missing `.so` |
| `nvidia-smi -q -d ECC` / job log | ECC errors | clean |
| Same env on another card | works | fails identically |

The same bad card also aborts COLMAP outright — `colmap::SetBestCudaDevice(int)` SIGABRT,
exit 134, with host RAM barely touched. **A CUDA fault that exits 134 with low MaxRSS is a
hardware suspicion, not an OOM.** Recover with `--exclude=<node>` and report the card; a node
with one bad GPU is not drained and keeps accepting jobs.
