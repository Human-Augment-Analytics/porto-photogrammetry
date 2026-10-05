# Evaluation (`augenblick.eval`)

Scores a finished reconstruction two ways, kept separate because on this data they disagree:
novel-view metrics (`nvs` + `metrics`) against held-out photographs, and geometric metrics
(`mesh`) against a reference mesh. Metric scale lives in `scale` — see pipeline-scale.md.

## Not on the `augenblick` CLI

The CLI's subparsers are built from the stage registries (`mask`/`sfm`/`recon`) only. Each eval
module carries its own argparse main instead: `python -m augenblick.eval.{nvs,mesh,scale}`.
`split` has no main — a `--eval` recon run writes it.

## Held-out split (`eval/split.py`)

- `--eval` on a recon config comes from `EvalParams` in `reconstruction/base.py`, inherited by
  `2dgs`, `pgsr` and `gw`. **`sugar` does not inherit it** and hardcodes `--eval False`, so it
  has no `--eval` flag and produces no held-out metrics.
- The first `--eval` run calls `write_split(scene.root)` → `<scene>/split.json`, holding out
  every 8th registered stem (`DEFAULT_HOLDOUT = 8`, the backends' shared `llffhold` rule). An
  existing file is **reused, never overwritten**, so a hand-authored split survives.
- It is written *before* `prepare()`, so a backend that copies the scene carries the split with
  it; `pgsr` additionally calls `copy_split()` into its own copy, and `hull` reads it under
  `--use_split` to carve from training views only.
- Stems come from `sorted(image.name.split(".")[0] ...)` — the order the backends' COLMAP
  readers use. Filesystem order would pair each render with the wrong photograph and still
  produce plausible numbers.

## Novel-view metrics (`eval/nvs.py`, `eval/metrics.py`)

Reads the PNG pairs under `<output>/test/ours_<iter>/{renders,gt}` (highest iteration unless
`--iteration`) and writes `<output>/nvs_metrics.json`, recording `"metric_domain": "mask"`.
Rescore without retraining:

```bash
python -m augenblick.eval.nvs --scene <sfm> --output <model_dir> [--iteration N]
```

- Formulas are transcribed from the Inria 3DGS reference, not imported from a vendored backend.
  PSNR is the 3DGS per-channel average (not one PSNR over a global MSE); SSIM is 11x11 Gaussian,
  sigma 1.5; LPIPS is the pip `lpips` VGG backbone — **absolute LPIPS is not comparable with
  published 3DGS numbers**, which use a differently normalising vendored copy.
- Each metric is restricted to the specimen differently: PSNR by a weighted mean over mask
  pixels, SSIM by averaging the full-frame map over mask pixels (so straddling windows see the
  same background in both inputs), LPIPS by cropping both images to the mask bounding box.
- The masking must stay a **weighted mean**. Multiplying both images and leaving the background
  in each denominator scores it as a perfect match: measured inflation was +6.5 to +12.0 dB
  PSNR, varying with mask fill, which also destroys cross-specimen comparability. The invariance
  tests exist to prevent its return.
- An all-zero mask skips the view (the clamped denominator would report a perfect match).
- Comparing backends: pass the **same `-r`**, since metrics are computed at whatever resolution
  the backend trained at. `gw --eval` trains without exposure compensation on purpose — that
  stage fits one exposure per training camera, and a held-out camera has none.

## Geometric metrics (`eval/mesh.py`)

```bash
python -m augenblick.eval.mesh --ours <mesh.ply> --reference <ref.ply> \
    [--rigid] [-n 200000] [--seeds 0 1 2] [--absolute-tolerances 0.5 1.0] \
    [--point-to-point] [--no-align] [--scale-ours 1000] [--label L] [--json-out f.json]
```

Tanks and Temples definition: bidirectional nearest-neighbour distances, precision/recall within
a tolerance, F-score as their harmonic mean. Accuracy and completeness are always reported
separately — a visual hull should score well on completeness and badly on accuracy, and a
symmetric Chamfer hides exactly that.

- **`score_surface` (exact point-to-triangle) is the default and the preferred scorer.** It is
  converged at 100k samples. `--point-to-point` selects the TnT point-to-point form, kept for
  comparability with published numbers; it is **biased low** by -0.0031 to -0.0110 F@0.5% at the
  200k default, and the bias varies 3.5x across specimens, so it does not cancel in
  cross-specimen comparisons of absolute F.
- Verified against the official TnT `get_f1_score_histo2`: exactly equal on identical point
  sets, 45 comparisons. That check covers the distance/threshold/harmonic-mean arithmetic only —
  not the transform, reference choice, diagonal, region or units.
- `DEFAULT_SAMPLES = 200_000`, sampled uniformly over **surface area** (vertex sampling would
  weight dense tessellation). `DEFAULT_TOLERANCES = (0.1%, 0.25%, 0.5%, 1%)` of the **reference**
  axis-aligned bounding-box diagonal. The tolerance must sit well above the sample spacing: at
  200k, 0.25% is unsafe, 0.5% marginal, 1% comfortable.
- The diagonal comes from the reference, so **changing the reference silently redefines every
  relative tolerance**, and it is set by extremes — a mount or support surface inflates it.
  Freeze and publish the per-specimen diagonal with any score. Scoring several candidates
  against one reference, pass `diag` explicitly (API) so the tolerance is the same distance for
  all of them.
- Surface sampling is random: `--seeds` averages the runs and reports
  `fscore_seed_spread@0.5pct`. Resampling spread alone can exceed a reported two-run retraining
  spread, so tie/win/loss calls below it are not meaningful.
- Registration: `fit_transform` initialises from principal axes over every sign combination
  (near-symmetric specimens give ICP several basins) and refines with trimmed ICP, keeping the
  lowest symmetric Chamfer. `--rigid` leaves scale at exactly 1.0 and **requires the
  reconstruction to already carry metric scale**; a COLMAP gauge does not — ours sit at roughly
  0.15-0.20x a millimetre reference, and registering rigidly scores **F = 0.000 at every
  tolerance**, reporting honestly that the objects are different sizes. Without `--rigid` the
  similarity fit *solves for* scale, which is a legitimate shape-modulo-similarity protocol but
  forbids any metric-scale claim. For metric scale, use pipeline-scale.md then `--rigid`.
- The CLI refits per invocation. Scoring many candidates of one specimen through the API, fit
  once and reuse: all candidates share the COLMAP frame, and refitting only adds ICP
  local-minimum variance that looks like a candidate collapsing for non-geometric reasons.
- Not handled yet: **observability** (no DTU-style masks or TnT cropping volume, so completeness
  against a contact/structured-light reference is a lower bound) and reference-side defects
  (mounting material, support surfaces, scanner holes).

## Tests

`tests/test_eval_{metrics,mesh,nvs,nvs_scoring,split,scale}.py` assert closed-form answers and
invariances rather than values the code happens to produce: constant-offset PSNR, SSIM of 1.0,
background randomisation leaving masked PSNR bit-identical, concentric spheres giving exactly
`d`, scale equivariance, and a rigid fit *refusing* to absorb a 5% size error.
