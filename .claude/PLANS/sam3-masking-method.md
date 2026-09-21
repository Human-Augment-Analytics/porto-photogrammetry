# Plan: a `sam3` masking method

Implementation spec. Historical once it lands — where this and the code disagree, the code
wins; the durable description belongs in
[pipeline-masking.md](../MEMORY/pipeline-masking.md).

**Status: verified end-to-end (2026-09-20, L40S, job 5871290, 35m10s).** 432/432 masks written,
0 failed, 690 s (**1.60 s/image**); contract check passed; 432/432 `masks_colmap/` links with no
unmatched masks; COLMAP logged `Mask: Yes` per image and reconstructed **432/432 images,
53,612 points, 1 model**. Overlays confirm `skeleton` isolates the specimen. Every "verified"
claim below was checked against a real run — the gotchas in §2 and §5 each cost a failed job
first, and both hid behind login-node stub tests that passed.

**Outstanding:** the tree was pruned further after that run (§1 pass 2). The current tree has a
clean `compileall`, a dangling-import scan and a passing GPU run on the immediately preceding
version, but the smoke test has not yet been placed on a card — jobs 5882582 (`ice-bw-gpu`) and
5881138 (`ice-gpu`) are queued. Documentation (§"Documentation to update") is written.

## Goal

A third masking method, `augenblick mask sam3`, alongside `rembg` and `threshold`:

```bash
augenblick mask sam3 --images <scene>/images --output <out>
```

SAM 3 ([facebookresearch/sam3](https://github.com/facebookresearch/sam3)) is promptable with a
text *concept*, unlike `rembg` which segments whatever is most salient. **Default prompt:
`skeleton`** (`--prompt`).

Non-goals: video/tracking mode, the point/box API, fine-tuning, any change to the mask contract
or to `rembg`/`threshold`.

## Verified state of the tree

Verified against branch `syed/dvlt-inclusion`.

- `MaskMethod` ([base.py](../../src/augenblick/masking/base.py)) implements `run()` once. A new
  method implements **only** `mask_for(path)` + a config dataclass.
- `mask_for` returns a **bool numpy array `[H, W]`**, `True` on the specimen, shape equal to
  `Image.open(path).size[::-1]`. A wrong shape is rejected and counted `num_failed`, not raised.
- Registration = `@register_mask` + an import in
  [masking/\_\_init\_\_.py](../../src/augenblick/masking/__init__.py). The CLI subparser,
  `--list` entry and all flags are generated from `config_cls` — **no CLI edit needed**.
- `MaskCommonConfig` already supplies `--only_missing`, `--min_foreground`, `--max_foreground`,
  `--keep_largest`, `--fill_holes`. Subclass it; do not redefine these.
- Heavy imports are function-local in both existing methods so `mask --list` works on a login
  node. Follow this.
- Envs are **Python 3.10.21, torch 2.9.1+cu129, numpy 2.2.6** (verified by running
  `augenblick_rtx_pro_6000/bin/python`). All six envs surveyed identical on py/torch/numpy and
  all have the `augenblick` CLI installed, except that `augenblick_a40` is on **cu130** where
  the other five are cu129 — irrelevant to SAM 3, which builds nothing, but it means a40 is not
  a like-for-like substitute for a `pip freeze` comparison. All six `scripts/setup_*.sh` set
  `NUMPY_GENERATION=2`.
- `src/libs/vggt` is **vendored in-tree, not a submodule** (`.gitmodules` has no `vggt` entry).
- `common.sh` does `module purge` and a batch shell does not source `~/.bashrc`. It now
  **exports `HF_HOME`/`TORCH_HOME` itself**, defaulting to `$HOME/scratch/huggingface/` and
  `$HOME/scratch/torch/` (the `~/.bashrc:13-14` values), so no job sets them. At the time this
  plan was written it did not, which is why §6 below originally put the exports in
  `mask.sbatch`.
- `data/neurips/subset/prepared/` holds 10 scenes, each with `images/` + `masks/`.

### Verified SAM 3 facts

From the upstream README and `pyproject.toml` (fetched 2026-09-19):

| Fact | Value | Consequence |
|------|-------|-------------|
| `requires-python` | `>=3.8` | **Python 3.10 is fine.** The README's "3.12 or higher" is advisory, not enforced. Do not create a new env. |
| torch | README ">=2.7"; unpinned in `pyproject.toml` | 2.9.1 satisfies it. |
| numpy | `numpy>=1.26,<2` | **Conflicts with the env's numpy 2.2.6.** See below. |
| Other deps | `timm>=1.0.17`, `tqdm`, `ftfy==6.1.1`, `regex`, `iopath>=0.1.10`, `typing_extensions`, `huggingface_hub` | `timm`, `ftfy`, `regex` are new. **`pycocotools`/`psutil` are not** — they were needed only while `sam3/train/` and the video path were still vendored; the deeper prune in §1 removed both callers, so neither is installed. |
| Checkpoints | Gated on HF (`facebook/sam3`), needs `hf auth login` | Warm on the login node; compute nodes have no interactive TTY. |
| License | SAM License (not Apache/MIT) | Note in the README entry. |

## Dependency rule: nothing gets reinstalled or upgraded

**No existing library may be reinstalled, upgraded, or downgraded by this change.** SAM 3
declares `numpy>=1.26,<2`; a plain `pip install -e src/libs/sam3` silently downgrades numpy
2.2.6 → 1.26.x and breaks the torch↔numpy C-ABI for every other stage — at runtime, not at
install time. The ceiling is a declared bound, not a used one; nothing in the image path needs
numpy 1.x semantics.

Rules for the install step:

1. Install `sam3` with **`--no-deps`**. Never let pip resolve its dependency tree.
2. Add only the genuinely-new packages (`timm`, `ftfy`, `regex`), each with **`--no-deps`**
   too, so their own transitive pins cannot touch torch, numpy or Pillow. (`pycocotools` and
   `psutil` were in this list until the §1 prune removed the modules that imported them.)
3. `PIP_CONSTRAINT` (already exported by `setup_common.sh`) stays in force as a second guard.
4. Record `pip freeze` before and after; **any line that changes other than the additions
   is a bug in this step**, not an acceptable side effect. Verified outcome at the time: 6 lines
   added (`sam3`, `timm`, `ftfy`, `regex`, `pycocotools`, `psutil`), 0 removed, 0 changed; the
   final tree drops the last two, leaving 4. Note `ftfy` lands at 6.3.1 rather than upstream's
   `==6.1.1`, since `--no-deps` bypasses that pin.

## Changes

### 1. Vendor the upstream repo — `src/libs/sam3/`

Clone in-tree and commit the source, exactly as `src/libs/vggt` is handled. **Not a submodule** —
`.gitmodules` is unchanged.

```bash
git clone https://github.com/facebookresearch/sam3.git src/libs/sam3
rm -rf src/libs/sam3/.git
git add src/libs/sam3
```

Then prune to the inference subset. This was done in two passes; the **final tree is 31 `.py`
files / 1.9 MB**, down from upstream's 284 files / 11 MB, and holds only
`sam3/{model,perflib,sam,assets}`:

```bash
# pass 1 — the obviously-unused trees (223 files / 9.1 MB -> 136 files / 5.0 MB)
rm -rf src/libs/sam3/{scripts,sam3/eval,sam3/agent,test,.github}
# pass 2 — the video/tracking and training paths
rm -rf src/libs/sam3/sam3/train src/libs/sam3/sam3/logger.py
```

`scripts/` alone is 4.3 MB, almost all of it one iNaturalist eval JSON.

The second pass needed the §1b patches first. `sam3/train/` looks load-bearing — `sam3_image.py`
imports `BatchedDatapoint` from the collator for a type annotation, and `model_builder.py`
builds a training matcher — but both are reachable only on paths the image-masking method never
takes, so each was converted to a local patch (see §1b) and the tree removed. Dropping
`sam3/train/` is what removed the `pycocotools` dependency.

**Keep `sam3/assets/`** — it holds the BPE vocab the model builder loads. **Keep
`sam3/perflib/fa3.py`**: static analysis says nothing imports it, but it is imported lazily on
Hopper cards and deleting it breaks H100/H200.

There are no `notebooks/` or `training/` directories in the upstream tree; the top-level
`assets/` (55 MB of docs media, distinct from `sam3/assets/`) and `examples/` are dropped at
clone time.

Because a deleted module can break an import that only fires inside a function body — which
neither `grep` nor `compileall` catches — every prune pass is followed by a GPU smoke test:
`pace_slurm/sam3_smoke.sbatch` masks 2 images with the real checkpoint and prints
`SMOKE TEST PASSED`.

### 1b. Local patches to the vendored source

Every deviation from upstream carries a `# LOCAL PATCH (augenblick)` comment, so `grep -rn
"LOCAL PATCH (augenblick)" src/libs/sam3` lists them all. Vendoring in-tree rather than as a
submodule is what makes these edits maintainable. There are four:

| File | Patch |
|---|---|
| `sam3/model_builder.py:8` | `pkg_resources` shim. setuptools removed it in 81 and these envs ship 83, so the upstream import raises `ModuleNotFoundError` at model-build time. Downgrading setuptools would violate the no-reinstall rule, so the single API used (`resource_filename`, which only locates `sam3/assets/bpe_simple_vocab_16e6.txt.gz`) is shimmed onto stdlib `importlib.resources`. |
| `sam3/model_builder.py:471` | The tracker/video/multiplex builders raise `NotImplementedError` instead of importing modules the prune removed. |
| `sam3/model_builder.py:~329` | The `eval_mode=False` branch raises `NotImplementedError` rather than constructing a training matcher from `sam3.train.matcher`. Inference always passes `eval_mode=True`. |
| `sam3/model/sam3_image.py:14` | `BatchedDatapoint` is taken from `sam3.model.data_misc` instead of the collator in `sam3/train/`. |

`README.md` inside the vendored tree also carries the marker, noting that the copy is trimmed.

### 2. `src/augenblick/masking/sam3.py`

Module docstring in the house style: what it is, where it runs, and that heavy imports are
function-local.

```python
@dataclass(frozen=True)
class Sam3Config(MaskCommonConfig):
    """SAM 3 concept-prompted segmentation parameters."""

    prompt: str = field(default="skeleton", metadata={
        "help": "Text concept to segment, e.g. 'skeleton', 'fish skull', 'bone'"})
    score_threshold: float = field(default=0.5, metadata={
        "help": "Drop detections scoring below this before the masks are unioned"})
    max_detections: int = field(default=0, metadata={
        "help": "Keep only the N highest-scoring detections (0 = keep all above threshold)"})
    checkpoint_path: Optional[str] = field(default=None, metadata={
        "help": "Local SAM 3 checkpoint; default downloads from the gated HF repo"})
    device: str = field(default="cuda", metadata={"help": "Torch device for the model"})
    batch_log_every: int = field(default=25, metadata={
        "help": "Log progress every N images"})


@register_mask
class Sam3Mask(MaskMethod):
    """SAM 3 concept-prompted segmentation (text prompt, default 'skeleton')."""

    name: ClassVar[str] = "sam3"
    title: ClassVar[str] = "Masking (SAM 3)"
    config_cls: ClassVar[type] = Sam3Config
```

Implementation notes, in order of importance:

- **Lazy, cached processor.** Mirror `RembgMask._get_session`: `__init__` sets
  `self._processor = None` and `self._count = 0`; `_get_processor()` builds once. Building per
  image would reload multi-GB weights 400+ times.

  ```python
  from sam3.model_builder import build_sam3_image_model
  from sam3.model.sam3_image_processor import Sam3Processor

  model = build_sam3_image_model(
      device=self.config.device,
      checkpoint_path=self.config.checkpoint_path,  # None → HF download
  )
  self._processor = Sam3Processor(model)
  ```

- **Fail loudly on CPU.** If `device` starts with `cuda` and `torch.cuda.is_available()` is
  False, raise `SceneError`. A silent CPU fallback plus `--only_missing` bakes the slow result
  into every later rerun (the lesson `rembg` already encodes).

- **Per-image inference under bfloat16 autocast** in `mask_for`. The autocast is not optional:
  the checkpoint's weights are bfloat16, so float32 input dies with `RuntimeError: mat1 and
  mat2 must have the same dtype, but got BFloat16 and Float` on *every* image. Upstream never
  documents this — only the scripts in `sam3/scripts/` show it. `set_text_prompt` takes the
  prompt **first** and returns the mutated `state`, not a separate dict.

  ```python
  device_type = "cuda" if self.config.device.startswith("cuda") else "cpu"
  with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
      with Image.open(image_path) as im:
          im = im.convert("RGB")
          state = processor.set_image(im)
      state = processor.set_text_prompt(self.config.prompt, state)
  masks, scores = state["masks"], state["scores"]
  ```

- **Reduce N detections to one bool `[H, W]`.** SAM 3 is an instance segmenter; the contract
  wants one silhouette. Filter by `score_threshold`, optionally truncate to `max_detections` by
  descending score, then **union** (`np.any(..., axis=0)`). Union, not argmax, so a specimen
  split across detections stays whole — consistent with `keep_largest` defaulting to False.

- **Normalise defensively before returning — casting on the torch side first.** Under autocast
  the masks come back **bfloat16**, which numpy cannot represent at all: converting first and
  casting after raises `TypeError: Got unsupported ScalarType BFloat16`. So `.float()` the
  tensor (unless it is already `bool`/`uint8`) *before* `.detach().cpu().numpy()`, then `> 0.5`
  if not bool, then collapse any channel axis to `[N, H, W]` before the union. Verified shape
  from upstream is `[N, 1, H, W]`, already interpolated to the source dims.

- **Check for zero detections before reshaping.** An empty `[0, 1, H, W]` cannot be reshaped,
  and the `ValueError` surfaces as an opaque `num_failed` instead of the real reason.

- **Empty result is a failure, not an empty mask.** If nothing survives the threshold, raise
  (the `mask_for` exception path logs and counts `num_failed`) or return all-False, which
  `min_foreground` then rejects. Either way **no file is written** — a missing mask means "skip
  this view" to consumers, whereas an all-foreground mask silently corrupts downstream stages.

- **Progress logging** every `batch_log_every` images, as `rembg` does.

- Override `header()` to add `Prompt` and `Score threshold` to the banner.

### 3. Register it

```python
from augenblick.masking import rembg, sam3, threshold  # noqa: F401
```

### 4. Install step — `scripts/setup_common.sh`

A new step after step 3 (VGGT + LightGlue), renumbering the `banner` labels through 9. No
`.gitmodules` or `git submodule` edit — `sam3` is vendored.

```bash
banner "4/9 SAM 3 (editable, --no-deps)"
# --no-deps throughout is load-bearing: sam3 declares numpy>=1.26,<2 and would downgrade the
# env's numpy 2.x, breaking the torch<->numpy ABI for every other stage at runtime. Nothing
# already installed may be touched by this step.
$PIP install timm ftfy regex --no-deps
$PIP install -e src/libs/sam3 --no-deps --no-build-isolation
```

Add `timm>=1.0.17`, `ftfy>=6.1.1` and `regex` to `requirements.txt` beside the `rembg` entry so
the manual pip path in the README matches.

**Do not** add `sam3` to `BACKENDS` — that selects *reconstruction* backends and gates CUDA
rasterizer builds; SAM 3 has no compiled extension.

### 5. Checkpoint access

`facebook/sam3` is gated. Once, on the **login node**:

```bash
hf auth login   # token from an account with access granted
hf download facebook/sam3 sam3.pt   # ~2 GB into $HF_HOME
```

This populates `$HF_HOME`, which compute nodes then read. Document both lines in the README — a
gated 401 on a compute node is otherwise opaque.

Two traps here, both hit during verification:

- **Do not warm up with `build_sam3_image_model(device='cpu')`.** SAM 3 cannot be built
  CPU-only: `position_encoding.py:55` and `decoder.py:301` hardcode `device="cuda"` when
  precomputing caches, ignoring the `device` argument, so it dies with "Found no NVIDIA
  driver" on a login node. Use `hf download`, which only needs the network.
- **Do not fetch the checkpoint on the login node at all.** The node's cgroup memory cap killed
  `download_ckpt_from_hf()` three times at the identical byte offset, and `hf download` then
  stalled dead partway. Each attempt left `.incomplete` blobs that `du` still counts toward
  ~2 GB, **so the cache looks full while `snapshots/` holds only `config.json`** — verify with
  `ls -lL $HF_HOME/hub/models--facebook--sam3/snapshots/*/` (expect `sam3.pt`, ~3.45 GB) and
  delete any `blobs/*.incomplete` before retrying.

  The fix that works: fetch it **from inside the batch job**, where there is no cap and compute
  nodes have outbound network. It is a no-op once the blob is cached, so it costs nothing on
  reruns:

  ```bash
  python - <<'PYDL'
  from sam3.model_builder import download_ckpt_from_hf
  print("checkpoint ->", download_ckpt_from_hf("sam3"), flush=True)
  PYDL
  ```

### 6. SLURM — `pace_slurm/mask.sbatch`

1. Extend the method guard: `case "$MASK_METHOD" in rembg|threshold|sam3) ;;`
2. **Export the cache roots.** `common.sh` runs `module purge` and a batch shell does not source
   `~/.bashrc`, so `HF_HOME`/`TORCH_HOME` are **not** inherited — without them SAM 3 re-downloads
   its checkpoint into `$HOME/.cache` on every task. This was originally two `export` lines in
   `mask.sbatch`; they have since moved into `common.sh`, which applies the same defaults for
   every job, so **`mask.sbatch` no longer sets them**.
3. The ORT preflight is already wrapped in `if [ "$MASK_METHOD" = "rembg" ]`, so it skips
   `sam3`. Add the parallel guard:
   ```bash
   if [ "$MASK_METHOD" = "sam3" ]; then
       python - <<'PY'
   import torch
   if not torch.cuda.is_available():
       raise SystemExit("ERROR: no CUDA device; SAM 3 masking would run on CPU")
   PY
   fi
   ```

Nothing else changes: `OUT_LAYOUT=nested` already writes
`$RESULT_ROOT/$MASK_METHOD/$SCENE_NAME`, so `sam3` coexists per scene. Extra flags forward via
the trailing `"$@"`.

**Submitting on the RTX Pro 6000 (the verification target).** `mask.sbatch` hardcodes
`#SBATCH --partition=ice-gpu` and `--gres=gpu:l40s:1`, and the RTX Pro 6000 Blackwell is the
only card **not** on `ice-gpu` — `sinfo` places it on `ice-bw-gpu`, and `common.sh` carries the
same note in its `rtx_pro_6000` case. The gres name is `rtx_pro_6000_blackwell`, not the env
suffix `rtx_pro_6000`; `common.sh` keeps `GRES_NAME` separate from the env name for exactly
this reason, but the `#SBATCH` lines do not read it. So both must be overridden at submit time:

```bash
GPU=rtx_pro_6000 sbatch --partition=ice-bw-gpu \
    --gres=gpu:rtx_pro_6000_blackwell:1 \
    --job-name=mask-sam3 pace_slurm/mask.sbatch --prompt "fish skull"
```

On any `ice-gpu` card the plain form still applies, e.g.
`MASK_METHOD=sam3 sbatch --job-name=mask-sam3 pace_slurm/mask.sbatch`.

The timing CSV's `method` column is the **Slurm job name**, so a distinct `--job-name` is what
keeps the three methods separable in one CSV.

**Resources — measured, not estimated.** On an L40S, 432 images at 6240x4160 took **690 s
(1.60 s/image)** with `--cpus-per-task=4 --mem=32gb`: *faster* than `rembg`'s ~2.2 s, so no
walltime increase is needed for the mask stage. SAM 3 runs one image at a time at 1008², so
VRAM is roughly constant in scene size; it did not come close to OOM on an L40S. If it does
OOM on some other card, that is a finding: record it and stop, do not retune.

**COLMAP, not masking, is the long pole.** Exhaustive matching over 432 images dominates the
job; size `--time` for that, not for the mask stage.

### 7. Verification

**Environment: `augenblick_l40s`** (`GPU=l40s`, partition `ice-gpu`, `--gres=gpu:l40s:1` —
all three are `mask.sbatch`'s own defaults, so no overrides). Python 3.10.21 / torch
2.9.1+cu129 / numpy 2.2.6, matching what step (b) asserts.

`augenblick_rtx_pro_6000` works identically but **`ice-bw-gpu` is only 4 nodes and was
unschedulable for 9+ hours** during this work, against ~1 minute to place on `ice-gpu`. Prefer
L40S for verification. Note each env needs its own `--no-deps` install of the six packages;
installing into one does not reach the others.

Steps (a) and (b) are import-level and run on the **login node**, which is also where §5's
`hf auth login` warm-up must happen. Only (c) and the closing COLMAP run need the card.

Pick one scene from `data/neurips/subset/prepared/` (10 available, each with `images/` +
`masks/`; `UF_Herp_87980_cranium` works). Every output **and every log** lands under
`output/neurips_verify/sam3/<scene>/`.

```bash
conda activate "$HOME/scratch/conda/augenblick_rtx_pro_6000"
SCENE=UF_Herp_87980_cranium
IMAGES=data/neurips/subset/prepared/$SCENE/images
OUT=output/neurips_verify/sam3/$SCENE
mkdir -p "$OUT"

# a. Registration + generated CLI, no model load required.
augenblick mask --list      > "$OUT/cli_list.log" 2>&1     # expects rembg, sam3, threshold
augenblick mask sam3 --help > "$OUT/cli_help.log" 2>&1     # expects --prompt (default: skeleton)

# b. Nothing was reinstalled or upgraded; numpy survived.
pip freeze | sort > "$OUT/pip_freeze_after.txt"
diff "$OUT/pip_freeze_before.txt" "$OUT/pip_freeze_after.txt" > "$OUT/pip_diff.txt" || true
#    expect ONLY: +sam3, +timm, +ftfy, +regex
python - > "$OUT/abi_check.log" 2>&1 <<'PY'
import numpy, torch
print("numpy", numpy.__version__, "| torch", torch.__version__)
assert numpy.__version__.startswith("2."), "numpy was downgraded"
torch.from_numpy(numpy.zeros((2, 2), numpy.uint8)); print("ABI OK")
PY

# c. The run itself, on a GPU node.
augenblick mask sam3 --images "$IMAGES" --output "$OUT" 2>&1 | tee "$OUT/mask_sam3.log"

# d. Contract check on the output.
python - > "$OUT/contract_check.log" 2>&1 <<PY
from pathlib import Path
from PIL import Image
import numpy as np
out = Path("$OUT")
for m in sorted((out / "masks").glob("*.png"))[:5]:
    src = next((out / "images").glob(m.stem + ".*"))
    im = Image.open(m); a = np.array(im)
    assert im.mode == "L", m
    assert set(np.unique(a)) <= {0, 255}, (m, np.unique(a))
    assert a.shape == np.array(Image.open(src)).shape[:2], (m, a.shape)
    print(m.name, "fg=%.3f" % ((a > 127).mean()))
PY
```

Capture `pip_freeze_before.txt` into the same directory *before* step 4 runs.

Then the check that actually matters: **open a few masks next to their images.** Dims and value
checks pass just as happily for a confidently wrong concept — if `skeleton` latches onto the
turntable or a scale bar, only looking catches it.

**Do not rely on the scene's shipped `masks/` as a reference:** on `UF_Herp_87980_cranium` that
directory exists but is **empty**, so there is no IoU baseline. Render overlays instead — tint
the background and check the silhouette by eye. A mask bounding box spanning most of the frame
is the tell for a wrong latch; a compact box that moves between views is the specimen.

After any further prune of `src/libs/sam3`, re-run the GPU smoke test rather than repeating the
full sequence — it is the cheapest thing that catches a function-local import broken by a
deleted module:

```bash
sbatch pace_slurm/sam3_smoke.sbatch          # ~30 s of work, 5-minute wall clock
```

Expect `SMOKE TEST PASSED` and `fg=0.0518`/`fg=0.0516` on the two images, which every run so far
has reproduced. The Blackwell env needs the override pair:
`GPU=rtx_pro_6000 sbatch --partition=ice-bw-gpu --gres=gpu:rtx_pro_6000_blackwell:1 pace_slurm/sam3_smoke.sbatch`.
Only `augenblick_l40s` and `augenblick_rtx_pro_6000` have `sam3` installed, so the GPU cannot be
left to the scheduler's choice: `common.sh` derives the conda env from `GPU`.

Finally, confirm a consumer accepts them — **COLMAP**:

```bash
augenblick sfm colmap --scene "$OUT" --output "$OUT/sfm" 2>&1 | tee "$OUT/sfm_colmap.log"
```

Two things to know about this check, both verified in
[sfm/colmap.py](../../src/augenblick/sfm/colmap.py):

- **There is no `--use_masks` flag.** Unlike VGGT, `ColmapSfM.run()` picks masks up
  automatically via `scene.has_masks()` and sets `pycolmap.ImageReaderOptions.mask_path`.
  Passing `--use_masks` is an unrecognised-argument error, not a no-op.
- **Masks are consumed through the `masks_colmap/` symlink dir**, built by
  `Scene.link_colmap_masks()`, because COLMAP wants `<image_name>.png` (`foo.jpg.png`) while
  the stage writes `<stem>.png`. Confirm `$OUT/sfm/masks_colmap/` is populated and that the log
  reports no unmatched masks — an empty or partial link dir means COLMAP silently ran
  *unmasked*, which still produces a plausible-looking reconstruction.

Verify the run produced `$OUT/sfm/sparse/0/` and that the registered-image count in the log is
close to the input image count.

## Documentation — written

- **[pipeline-masking.md](../MEMORY/pipeline-masking.md)** — the `sam3` method bullet with its
  flags, the union-not-argmax rationale, a `### sam3 traps` subsection (bfloat16 autocast; the
  bfloat16→numpy cast; the zero-detection check; CPU model-build failure; gated repo +
  login-node OOM), the trimmed-tree paragraph, and the measured 1.60 s/image as the exception to
  the cost table.
- **[environment-and-gpu.md](../MEMORY/environment-and-gpu.md)** — `src/libs/sam3` installed at
  stage 4/9 with `--no-deps`, and its three real dependencies.
- **[cluster-slurm.md](../MEMORY/cluster-slurm.md)** — `MASK_METHOD=rembg|threshold|sam3` and the
  CUDA preflight; `common.sh`'s `HF_HOME`/`TORCH_HOME` exports as a deliberate choice.
- **CLAUDE.md** — the stage row, the MEMORY index row, `sam3` in the `src/libs/` list, and two
  gotchas (the trimmed vendored copy; GPU-only + gated checkpoint + text `--prompt`).
- **Root README** — a Stage 0 masking table plus a Step 0.5 usage section covering all three
  methods, `masking/` and `libs/sam3/` in the structure tree, and the checkpoint cache note.

Still to do: **[pace_slurm/README.md](../../pace_slurm/README.md)** — `sam3` in the
`MASK_METHOD` list, the `--prompt` forwarding example, the `ice-bw-gpu` +
`rtx_pro_6000_blackwell` override pair. And add this file to the landed-plans list in
[repo-conventions.md](../MEMORY/repo-conventions.md) in the same commit as the feature.

## Open questions

- **Is `skeleton` right across all 47 scenes?** **Answered for `UF_Herp_87980_cranium`: yes.**
  All 432 views gave a single detection at 0.949-0.965, and overlays show a clean crocodilian
  cranium — teeth resolved, fenestrae correctly background, both scale bars excluded. Because
  detections were single-instance throughout, the union-vs-argmax reduction was never exercised
  here; a specimen that fragments across detections still needs checking. Untested on soft
  tissue. Resolve per material by looking, not by reasoning; if it turns out scene-dependent,
  the honest fix is a per-scene prompt in the SLURM job, not a cleverer default.
- **Fall back to `rembg` on a low-score scene?** Deliberately not planned — it would make the
  timing CSV's `method` column lie about what produced a mask. Prefer a visible failure and a
  rerun with a different `--prompt`.
