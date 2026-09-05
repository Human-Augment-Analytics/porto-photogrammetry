# The `augenblick` package

`src/augenblick/` replaces the six standalone scripts that used to live under `pipeline/sfm/`
and `pipeline/reconstruction/`. Adding a method now means writing one subclass, not copying a
150-line wrapper.

## Install

The package is installed editable, with **no dependencies of its own**:

```bash
pip install -e . --no-deps --no-build-isolation
```

`--no-deps` and `--no-build-isolation` are mandatory. `pyproject.toml` deliberately declares no
`dependencies`: numpy/scipy pins live in `constraints/numpy{1,2}.txt` and must match the GPU's
torch wheel, so letting pip resolve them here would break imports at runtime, not at install.
`scripts/auto_setup.sh` remains the only installer for the environment itself.

It must be installed into **each** per-GPU conda env, since the SLURM jobs call the bare
`augenblick` console script:

| Env | Path | numpy |
|---|---|---|
| rtx6000 (HPG) | `$CONDA_ROOT/augenblick_rtx_pro_6000` | 2.x |
| b200 (HPG) | `$CONDA_ROOT/augenblick_b200` | 1.x |
| a100 (PACE) | `$CONDA_ROOT/augenblick_a100` | 2.x |

> The b200 env's `bin/pip` has a stale shebang pointing at a renamed directory; use
> `<env>/bin/python -m pip` there instead. This predates the package and is unrelated to it.

## CLI

```
augenblick sfm   <method> --scene <dir> --output <dir> [method flags]
augenblick recon <method> --scene <dir> --output <dir> [method flags]
augenblick sfm --list
augenblick recon --list
```

Uniform `--scene` / `--output` replaces the old split between reconstruction positionals and
SfM `--input_dir` / `--output_dir`. Methods: SfM `vggt`, `colmap`, `turntable`; reconstruction
`2dgs`, `sugar`, `pgsr`, `gw`.

Exit codes, which SLURM guards depend on:

| Condition | Code |
|---|---|
| `SceneError` (missing/empty `images/` or `sparse/0/`, COLMAP produced no model) | 2 |
| `MethodNotFound` / no method given | 2 |
| `BackendError` | the backend subprocess's own return code |

## Architecture

```
core/     errors, Scene, config bridge, registry, process.run, StageTimer, Method ABC
sfm/      base (SfMMethod, SceneRefiner) + vggt, colmap, turntable
reconstruction/  base (ReconstructionMethod, SubprocessBackend, Stage) + the four backends
cli/      main.py — the only place logging is configured
```

### The two ABCs

`Method` (`core/method.py`) is the root: a `name`, a `config_cls`, `validate(scene)`, and
`run(scene, output_dir) -> StageResult`.

- **`SfMMethod`** requires only `images/`. **`SceneRefiner`** additionally requires an existing
  non-empty `sparse/0/` — this is what encodes in the type system that `turntable` is a post-SfM
  refinement step, not a standalone SfM.
- **`ReconstructionMethod`** requires both `images/` and `sparse/0/`, and adds `stages()` and
  `mesh_path()`. **`SubprocessBackend`** implements `run()` once for all four backends:
  validate → `prepare()` → run each `Stage` under a `StageTimer`.

### Registry

`@register_sfm` / `@register_reconstruction` key on `cls.name`; duplicates raise `ValueError`.
`get_sfm` / `get_reconstruction` raise `MethodNotFound` **listing the available names** — that
message is the discoverability surface when a SLURM job passes a bad backend string. Registration
fires on import, which is why `sfm/__init__.py` and `reconstruction/__init__.py` import their
modules with `# noqa: F401`; removing those imports silently empties the registry.

### Config ↔ argparse bridge

Each method declares a frozen dataclass; `add_dataclass_arguments` derives the parser from field
types, and `config_from_namespace` builds the config back. Help text rides in
`field(metadata={"help": ...})`.

| Field type | Default | argparse |
|---|---|---|
| `bool` | `False` | `store_true` |
| `bool` | `True` | `BooleanOptionalAction` → `--no-<name>` |
| `int` / `float` / `str` / `Path` | any | `type=<t>` |
| `Optional[T]` | `None` | `type=T, default=None` |
| `Literal["a","b"]` | any | `choices=[...]` |
| `list[int]` | `default_factory` | `nargs="+"` |

The bool split is load-bearing: it reproduces GW's `--no-postprocess` and everyone else's plain
switches **from the defaults alone**. `metadata["cli_name"]` renames a flag and
`metadata["short"]` adds a short form (GW's `-r`); `dest` is always pinned to the field name so
either override still maps back. Because the bridge reads annotations at runtime, the type
annotations are load-bearing, not decoration.

This dataclass-first shape is what makes each method usable as a library —
`VGGTSfM(VGGTConfig(use_ba=True)).run(scene, out)` — which the old argparse-only scripts could not
do, and which a future N×M sweep driver needs.

### Scene

`Scene(root)` owns the COLMAP layout: `images_dir`, `masks_dir`, `sparse_dir` (`sparse/0`),
`has_masks()`, `has_reconstruction()`, `require_images()`, `require_reconstruction()`.
`has_reconstruction()` checks the directory is **non-empty**, mirroring the shell guard in
each `recon.sbatch` — an empty `sparse/0/` means SfM failed.

`link_colmap_masks(dest)` centralises the symlink trick both pycolmap paths used: COLMAP wants
`<image_name>.png`, so `foo.png` is linked as `foo.jpg.png`. It is idempotent.

## Per-backend divergences (preserved deliberately)

- **PGSR** overrides `prepare()`: it copies the scene and flattens `sparse/0/` → `sparse/`, and
  **returns early if the prepared copy already exists**. Its render stage takes only `-m`, no
  `-s`, and `--skip_mesh` maps to upstream's `--skip_train`.
- **GW** sets `use_cwd = False` — its scripts import from their own directory, so cwd would not
  help. It is the only backend with `accepts_passthrough = True`: unknown CLI flags are forwarded
  to the **training stage only**. Its mesh filename is computed from `n_pivots` + `postprocess`,
  and the textured name from `texture_n_iter - 1`.
- **SuGaR** runs a nested vanilla-3DGS train first and passes booleans as `--flag True` **string
  pairs**, not `store_true`. Its `refined_mesh/<scene_name>/` directory is named from the *scene*
  path, not the output dir.
- **2DGS** always passes `--skip_test` on render; `mesh_path()` is
  `train/ours_<iterations>/fuse_post.ply`.

Passthrough is gated on `accepts_passthrough` rather than applied globally, so a typo in any other
backend is rejected instead of silently forwarded.

## Adding a new backend

1. Create `reconstruction/<name>.py` with a frozen `<Name>Config` dataclass (one field per
   upstream flag, with `metadata={"help": ...}`).
2. Subclass `SubprocessBackend`, setting `name`, `config_cls`, `backend_dir`, `title`, and
   `use_cwd` if the backend must not run from its own directory.
3. Implement `stages()` returning `[Stage(label, argv), ...]` and `mesh_path()`.
4. Override `prepare()` only if the backend needs a rewritten scene.
5. Add it to the `reconstruction/__init__.py` import line.
6. Add an argv fixture to `tests/test_recon_argv.py`.

It then appears in `augenblick recon --list` and `--help` with no CLI edit. An SfM method is the
same, subclassing `SfMMethod` (or `SceneRefiner`) and implementing `run()` directly.

## Testing

`tests/` is CPU-only and runs on the login node:

```bash
pytest tests/
```

`test_recon_argv.py` is the important one: the outer CLI may be reshaped freely, but the **argv
handed to each backend script** is dictated by upstream `train.py` / `render.py` and must not
drift — a wrong flag there surfaces only as a bad reconstruction hours later. Expected argv is
derived by reading the pre-refactor wrappers, and asserted on the full list.

> `pytest` is not part of the pinned environment spec; it was installed into the rtx6000 env
> (pure-Python, `--no-deps`-safe, does not perturb the numpy/torch pins).

## Port validation (Phase 6 smoke run)

The port was validated end-to-end on `UF_birds_ivory2` (array index 2 — the smallest scene that
still has masks: 184 images, 184 masks), GPU target `rtx6000`:

| Step | Job ID | State | Elapsed | Result |
|---|---|---|---|---|
| VGGT SfM (`hpg_slurm/vggt_sfm.sbatch`) | 40649294 | COMPLETED | 2:02 | 184/184 images registered, 100,000 points |
| 2DGS recon (`hpg_slurm/recon.sbatch`, `--iterations 2000`) | 40649370 | COMPLETED | 12:01 | mesh at the predicted path, 263 MB |

Stage timings, useful as the baseline for spotting a future regression:

- VGGT: model load 9.3s, inference 26.0s, total 99.9s.
- 2DGS: training 152.4s, rendering + TSDF fusion 560.9s, total 713.3s.

> `--iterations 2000` is a **smoke value only**, far below the 30000 default. It proves plumbing,
> not reconstruction quality — never quote a mesh from this run as a result.

**Proven:** the package imports under the batch env; the CLI parses; the registry resolves;
`Scene` validation passes on real data; argv reaches the backends correctly; VGGT and 2DGS run to
completion; `mesh_path()` matches where the backend actually wrote.

**Not proven:** SuGaR, PGSR, and GW have never been run through the package. PGSR's `prepare()`
scene-flattening and GW's no-`cwd` invocation plus passthrough are the two highest-risk
unexercised paths. Non-default flags are likewise unexercised.

## Known follow-up work

- **`.jpg`-hardcoded mask symlinks.** `Scene.link_colmap_masks` builds `<stem>.jpg.png`, so a
  scene whose images are not `.jpg` gets mask names COLMAP will never match. Behaviour was carried
  over verbatim from the old scripts; all current scenes are `.jpg` because
  `prepare_uf_dataset.py` normalises to it. Fix separately, with a test, rather than as part of a
  move.
