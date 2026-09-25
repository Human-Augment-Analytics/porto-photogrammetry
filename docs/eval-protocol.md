Generated with the help of an AI agent.

# Evaluation protocol

This document states what the evaluation code computes, and why each choice was made. It
exists because most of the quantities below are *conventions* rather than consequences:
there is no single correct way to score a masked, object-centric capture, and a number is
only meaningful once the convention that produced it is declared.

Two metric families are reported and they are kept separate throughout. Novel-view
metrics score rendered images against held-out photographs. Geometric metrics score a
reconstructed mesh against a reference mesh. They answer different questions and, on this
data, they do not agree.

## 1. Held-out split

`augenblick.eval.split` holds out every 8th registered image, which reproduces the
`llffhold` rule shared by the Gaussian-splatting backends. The split is written once to
`split.json` in the scene directory and every backend reads it, so all of them train on
the same images and are scored on the same images.

Stems are sorted by name, because the backends' COLMAP readers sort their cameras that
way and positional exports (`00000.png`) are indexed against that order. A split written
in filesystem order would pair each render with the wrong photograph and mask it with the
wrong silhouette, which is a failure that produces plausible-looking numbers.

## 2. Novel-view metrics

`augenblick.eval.metrics` implements PSNR, SSIM and LPIPS. The formulas are transcribed
from the Inria 3DGS reference implementation (`utils.image_utils.psnr`,
`utils.loss_utils.ssim`) rather than imported from a backend's vendored copy, so scoring
does not depend on which backend happens to be installed.

**PSNR follows the 3DGS definition**, which averages three per-channel PSNRs. This is not
the same as one PSNR over a global MSE, and the two differ whenever channel errors are
unequal. The 3DGS definition is used because the numbers it produces are the ones the
literature reports. `tests/test_eval_metrics.py` pins this explicitly.

**SSIM** uses an 11x11 Gaussian window with sigma 1.5 and the standard stabilisers
`C1 = 0.01^2`, `C2 = 0.03^2`.

**LPIPS** uses the pip `lpips` package with the VGG backbone. Note that 3DGS uses its own
vendored `lpipsPyTorch`, which normalises its inputs differently. **Absolute LPIPS values
here are therefore not directly comparable with published 3DGS numbers**, though
within-scene differences are.

### 2.1 Scoring domain

Every scene is a specimen photographed against a dark surround. Scoring the whole frame
mostly rewards reproducing the background, so all three metrics are restricted to the
object. The three are restricted differently, and the differences are real:

| Metric | Restriction | Guarantee |
|---|---|---|
| PSNR | squared error averaged over mask pixels only | exactly invariant to anything outside the mask |
| SSIM | map computed on the full composited frame, then averaged over mask pixels | invariant to the background except within one window radius (5 px) of the silhouette |
| LPIPS | both images cropped to the mask bounding box | invariant only to changes outside that box |

SSIM's map is deliberately computed before masking so that windows straddling the
silhouette see the same background in both inputs, exactly as the rendered and
photographed views do. LPIPS has no per-pixel domain to restrict, and scoring a specimen
that fills 6% of the frame without cropping measures mostly the surround, at the wrong
feature scale.

Masking is applied as a **weighted mean over mask pixels**. It must never be applied by
multiplying both images: that leaves the background inside each metric's denominator,
where it scores as a perfect match and inflates every value. Measured inflation from that
bug was +6.5 to +12.0 dB PSNR, varying with how much of the frame the specimen filled,
which also destroyed cross-specimen comparability. The invariance tests exist to prevent
its return.

Metrics files record `"metric_domain": "mask"` so that a file cannot be misread later.

## 3. Geometric metrics

`augenblick.eval.mesh` implements the scoring. The definition is the Tanks and Temples
one: nearest-neighbour distances in both directions, precision and recall as the fraction
of each side within a tolerance, F-score as their harmonic mean.

**This has been verified against the official toolbox.** Scoring identical point sets with
our implementation and with the official `get_f1_score_histo2` from the Tanks and Temples
Website Toolbox returns exactly equal values across 45 comparisons (5 specimens x 3
reconstruction arms x 3 thresholds).

The scope of that check is worth stating precisely, because it is narrower than "our
evaluation is correct". It establishes that, given identical point clouds and thresholds,
our nearest-neighbour distances, thresholding and harmonic mean agree with the official
scorer. It does **not** check what produces those point clouds: the transform applied, the
choice of reference file, the diagonal, the evaluated region, or the units. Those are
covered, where they are covered at all, by the tests in section 5.

Accuracy and completeness are reported separately and never collapsed into a symmetric
Chamfer alone. An outer bound such as a visual hull should score well on completeness,
because it contains the object, and badly on accuracy, because it fills concavities; a
symmetric mean hides precisely that asymmetry.

### 3.1 Sampling convention and sampling density

Points are sampled **uniformly over surface area**, 200,000 per mesh by default. Sampling
vertices instead would weight densely tessellated regions more heavily, so two meshes
describing the same surface with different triangulations would score differently.

Tanks and Temples uses a different convention: it voxel-downsamples the point clouds at
`tau/2`. The two effects were separated by a factorial over {200k, 2M} samples x
{direct, voxelised}, because comparing 200k-direct against 2M-voxelised confounds them.

**Density dominates at the tight threshold; at 0.5% the two are comparable.** At a 0.25%
tolerance, going from 200k to 2M samples moves the F-score by +0.0096 to +0.0307, while
voxelising at fixed density moves it by only -0.0041 to +0.0021. At 0.5% neither dominates and
which is larger depends on the case: density +0.0015 / voxel -0.0005 on one, +0.0019 / +0.0038
on another, +0.0048 / -0.0051 on a third. Do not treat voxelisation as negligible in general.

The mechanism is sample spacing. At 200k the median within-cloud nearest-neighbour spacing is
about `1e-3` of the diagonal, so a 0.25% tolerance is only about **2.5x the spacing** and the
score is partly measuring the evaluation's own discretisation rather than the surface. At 2M
the spacing falls to about `3e-4` of the diagonal. This is circumstantial rather than
demonstrated: a proper convergence curve over several sample counts and seeds, using upper
quantiles of the spacing distribution rather than the median, has not been run.

**Rule: the tolerance must be well above the sample spacing.** At 200k samples that makes
0.25% unsafe, 0.5% marginal and 1% comfortable. At 200k, F@0.5% still carries density-induced
error as large as 0.0048, which exceeds the precision needed for small between-method
comparisons. Raise the sample count, or report the sensitivity, before quoting a tight
tolerance.

Across every specimen and every reconstruction arm, switching between direct and voxelised
scoring at fixed density flipped the sign of a contrast 34 times out of 1,107 (all pairs, three
thresholds). At 0.5% specifically, 34 of 369 pairwise win/tie/loss verdicts changed. Every
affected contrast was below 0.0013 and they concentrate on specimens with the smallest claimed
spread — but see section 3.1.1: that verdict rule is itself unsound, so most of those labels
were never meaningful. Ranking is broadly stable: Spearman between the two conventions is 1.00
on three specimens at 0.25%, but 0.811 on one specimen at 1%, so nearly tied arms do reorder.

### 3.1.1 The evaluation has its own repeat spread

Surface sampling is random, so scoring the same mesh twice gives different answers. Over
five seeds at 200k samples, the spread of F@0.5% is:

| case | resampling spread | reported retraining spread | ratio |
|---|---|---|---|
| skull, vanilla | 3.0e-4 | 2.5e-3 | 0.12 |
| herp3998, hull300k | 1.9e-3 | 2.7e-3 | 0.68 |
| birds59449, vanilla | 1.0e-3 | 2.0e-4 | **5.0** |

**This matters for how results are judged.** A per-specimen spread derived from repeated
*training* runs also contains the evaluation's own resampling noise, and on the third case
the resampling range alone exceeds the whole reported figure. A threshold set at twice that
figure is then below the resolution of the measurement, and any tie/win/loss call made
against it is not meaningful.

Two cautions on reading the table. A five-seed range is expected to exceed a two-run range
even under an identical distribution, and the two-run figure already contains sampling noise,
so these are not cleanly separated variance components; the defensible statement is that
*resampling alone can exceed the reported two-run discrepancy*, not that one dominates the
other by a measured factor. Separating them properly needs a hierarchical design: several
sampling seeds for each of several retrained meshes.

The practical fix is to reuse one fixed high-density reference cloud per specimen, average
each score over several declared sampling seeds, and raise the sample count until the Monte
Carlo spread is below the smallest effect the benchmark intends to resolve. Reusing a single
seed across arms is **not** an adequate substitute: meshes differ in tessellation and triangle
order, so the same seed does not produce matched surface locations, and it bakes one arbitrary
realisation into the benchmark.

A stronger alternative is now implemented and is the **preferred scorer**: `score_surface`
uses exact bidirectional **point-to-triangle** distance. Sampling only the source surface and
querying the distance to the target *mesh* removes the target cloud's discretisation error
entirely. `score` is retained because it is the point-to-point form Tanks and Temples uses and
is what published numbers are comparable with.

### 3.1.2 Point-to-point is biased low, and by how much depends on the specimen

Point-to-point nearest-neighbour distance systematically **overestimates** the distance to a
surface, because the nearest *sample* of the target is not the nearest *point* of it. The
overestimate is of order the target's sample spacing, and since a larger distance falls outside
the tolerance more often, it **depresses the F-score**. Measured over a sample ladder with three
seeds per point, holding registration fixed:

F@0.5 %, exact point-to-triangle versus point-to-point:

| samples | skull exact | skull p2p | herp3998 exact | herp3998 p2p | birds exact | birds p2p |
|---|---|---|---|---|---|---|
| 100k | 0.94248 | 0.93470 | 0.72366 | 0.70897 | 0.49219 | 0.47186 |
| 200k | 0.94254 | 0.93946 | 0.72368 | 0.71763 | 0.49132 | 0.48030 |
| 500k | 0.94245 | 0.94125 | 0.72360 | 0.72078 | 0.49116 | 0.48597 |
| 1M | 0.94220 | 0.94156 | 0.72358 | 0.72193 | 0.49155 | 0.48851 |
| 2M | 0.94229 | 0.94192 | 0.72371 | 0.72271 | 0.49133 | 0.48951 |

**The exact scorer is flat.** Across a twentyfold change in sample count it varies by 3.4e-4,
1.3e-4 and 1.0e-3 respectively — within its own seed-to-seed spread. It is converged at 100k.

**The point-to-point scorer is not.** It climbs monotonically toward the exact value, its bias
roughly halving each time the sample count doubles, which is the expected behaviour for a term
proportional to point spacing. At 2M it is still short by 0.0004 to 0.0018.

At the 200k default the bias is **-0.0031 (skull), -0.0061 (herp3998), -0.0110 (birds59449)**.
Two consequences:

- Every F-score computed point-to-point at 200k is **too low**, so those numbers are
  conservative rather than flattering.
- The bias **varies by a factor of 3.5 across specimens**, so it does not cancel in
  cross-specimen comparisons of absolute F. It largely cancels in between-arm contrasts within
  one specimen, which is why the comparative conclusions stand.

Exact scoring at 100k is both more accurate and cheaper than point-to-point at 2M (about 20 s
against 56 s per mesh), so there is no trade-off to weigh: prefer `score_surface`.

Seed-to-seed spread is essentially identical between the two scorers at equal sample count, so
the improvement is the removal of a bias, not a reduction in Monte Carlo noise. That noise still
needs averaging over seeds when small effects are at stake.

### 3.2 Tolerances

Tolerances may be expressed two ways and both are reported when both are requested.

**Relative** tolerances are fractions of the reference bounding-box diagonal. They allow
specimens of very different physical sizes to be pooled.

**Absolute** tolerances are distances in the reference's own coordinate unit. This is what
Tanks and Temples and DTU do, and it is only meaningful once the reference carries true
scale.

Because the diagonal comes from the **reference**, changing the reference silently
redefines every relative tolerance. When the reference changes, relative numbers from
before and after the change are not comparable.

The diagonal is that of an **axis-aligned** bounding box of the reference samples, which
has two consequences worth knowing. It is not rotation invariant, so the same geometry
expressed in a different frame yields a different tolerance; and it is set by the extremes,
so one stray point, a mount or a support surface inflates it and makes every tolerance more
permissive. It is used because it is simple and reproducible, not because it is robust.
Freeze and publish the per-specimen diagonal alongside any score.

When scoring several candidates against one reference, pass `diag` explicitly so that the
tolerance denotes the same physical distance for every candidate.

### 3.3 Registration

Reconstructions live in an arbitrary COLMAP frame, so they must be registered to the
reference before scoring. `fit_transform` initialises from principal axes, trying every
sign combination because near-symmetric specimens give ICP several plausible basins, and
refines with trimmed ICP, keeping the candidate with the lowest symmetric Chamfer.

Two modes exist and the choice matters:

- `rigid=True` fits rotation and translation only, leaving scale at exactly 1.0. **Use
  this whenever the reference carries true scale**, because any size error then remains in
  the residual where it can be measured.
- `rigid=False` fits a similarity, which *solves for* scale. This is necessary when the
  reference has no meaningful scale, but it **absorbs metric-scale error**, so scale error
  cannot be reported from a run that used it.

⚠️ `rigid=True` requires the reconstruction to **already** carry metric scale. A COLMAP
reconstruction does not: its gauge is arbitrary, and measured against a millimetre
reference our reconstructions sit at roughly 0.15-0.20x its size. Registering those rigidly
scores F = 0.000 at every tolerance, which is the mode reporting honestly that the two
objects are not the same size. That ratio is a **gauge difference, not a scale error**: the
pipeline never attempted to recover metric scale, so there is no estimate to be wrong.

Similarity alignment therefore remains a legitimate protocol, evaluating **shape modulo global
similarity**, provided that is stated and no metric-scale claim is attached to it. It becomes
circular only if the fitted scale is presented as evidence of scale accuracy.

Reporting metric-scale error requires a scale constraint independent of the reference mesh:

1. Scale the reconstruction using something external to the ground truth - a scale bar or
   known object dimension, a calibrated camera baseline, or known turntable geometry.
2. Register to the reference **rigidly**.
3. Report the geometric error and the residual scale error separately.

When scoring many candidates of one specimen against one reference, fit the transform once
and reuse it. All candidates for a specimen inherit the same COLMAP frame, so refitting
per candidate adds no accuracy and introduces ICP local-minimum variance, which appears as
a candidate whose score collapses for reasons that are not geometric.

## 4. What is not yet handled

**Observability.** A reference acquired by contact or structured-light scanning covers
surfaces the cameras never saw. Scored naively, completeness penalises the reconstruction
for failing to recover geometry that was never photographed. DTU addresses this with
observability masks and Tanks and Temples with a cropping volume; this protocol has
neither yet. Until it does, completeness against such a reference is a lower bound.

**Reference-side defects.** Mounting material, support surfaces and scanner holes are part
of the reference mesh unless removed, and each biases a different metric.

## 5. Tests

`tests/test_eval_metrics.py` and `tests/test_eval_mesh.py` assert closed-form answers and
invariances rather than values the code happens to produce:

- PSNR of images differing by a constant `d` is exactly `20*log10(1/d)`
- SSIM of identical images is exactly 1.0
- randomising the background leaves masked PSNR bit-identical, and leaves SSIM unchanged
  on a mask eroded by one window radius but *changed* on the full mask
- an all-ones mask reproduces the unmasked path
- two concentric spheres separated by `d` give accuracy, completeness and Chamfer of
  exactly `d`, and an F-score that steps from 0 to 1 as the tolerance crosses `d`
- scaling both meshes leaves every relative metric unchanged and scales every absolute one
- a rigid fit recovers a known pose with scale exactly 1.0, and **refuses** to absorb a 5%
  size error, which a similarity fit removes
- sampling is proportional to triangle area, checked on two triangles differing 100x
