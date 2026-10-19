Generated with the help of an AI agent.

# Independent metric scale from printed scale bars

A COLMAP reconstruction has an arbitrary gauge, so its meshes carry no unit. Similarity
registration against a metric reference cannot report scale error, because it solves for
scale. `augenblick.eval.scale` recovers the millimetre scale from the printed bars
photographed alongside the specimen and refuses to hand over a number it cannot check.

## Inputs

- A COLMAP sparse model. Its camera fit must not have consumed marker observations, or
  the held-out reprojection checks below stop being independent.
- The original photographs. Detection needs the bars visible, so it always reads the
  unmasked images, regardless of how strict the object masks used for reconstruction are.
- A detections JSON: per image, the decoded target code and OpenCV pixel centre of every
  candidate. The detector is out of scope; any decoder of circular coded targets works.
- The four target codes forming the two bars, split into a primary and a check pair.

The scene directory may carry a `capture_manifest.json`: a list of
`{"image": <path or name>, "position": <label>}` entries assigning each photograph to a
turntable position. When present, ring grouping keys on (camera body, lens, focal length,
position); without it, on the camera hardware alone. A capture whose images resolve to
fewer usable rings than the two the acceptance gate requires is reported as such up front:
its windows are still evaluated for diagnostics, but it cannot produce an accepted scale.
Rings shorter than one window are recorded and skipped, because a wrapped window would
hold repeated views and the holdout would overlap the training set.

Two conventions matter at the input boundary. OpenCV puts the first pixel centre at
(0, 0) and COLMAP at (0.5, 0.5); the module adds the half pixel itself. And the detector's
canonical code for a bit pattern does not equal the label printed on the bar; the mapping
must be verified once per capture by inspecting a photograph, because a wrong pairing
associates observations of different physical targets and produces silent garbage.

## Why windows instead of one global triangulation

A single static 3D position per marker over a full turntable capture is an assumption, not
a given. On real captures it can fail held-out reprojection by several pixels while the
centre localisation itself is stable to a fraction of a pixel, and while short spans of
consecutive views pass the same gates with bar lengths agreeing to a fraction of a
percent. Candidate explanations are perspective eccentricity of circular targets
(Luhmann 2014, ISPRS Archives XL-5, doi:10.5194/isprsarchives-XL-5-363-2014), spatially
varying camera or pose error, and physical movement during capture; the protocol does not
need to decide between them, because all three respect local rigidity.

Scale is therefore estimated per window: every span of `window_size` consecutive views at
a fixed stride, wrapping around each camera ring. All windows are enumerated and reported;
none is selected by its outcome.

## Gates

All thresholds are declared before results are seen and are recorded in the output.

Per window and marker: at least 16 accepted observations, of which 8 evenly spaced ones
are withheld; the remainder must triangulate with at least 8 inliers within 2 px, 60%
consensus and 2 degrees of ray separation; and at least 60% of the withheld observations
must reproject within 2 px. The last check is the important one: it asks the fitted point
to predict measurements it never saw.

Acceptance across windows: at least 6 passing windows spanning at least 2 rings and 3
offsets, primary-bar length spread at most 0.5%, and the check bar within 1% of its
nominal length in every passing window and in the median. The two-ring requirement is
support breadth: windows of a single ring can share a view-angle-dependent systematic and
still agree with each other, so their internal consistency is not evidence of correctness.

The check bar never participates in fitting the scale. One ruler sets it, a second ruler
the fit never touched verifies it. The roles are fixed in advance and must not be swapped
after seeing which bar behaves better.

## What acceptance does and does not establish

An accepted scale means the bars, as reconstructed by this camera model, tell a consistent
metric story with independent verification. It does not bound three residual unknowns: the
physical bars' dimensional accuracy (the printed length is nominal), a systematic camera
error shared by both bars, and any global surface bias of the mesh being scaled. The
passing-window range is a spread, not a standard error, because overlapping windows share
observations.

The scale is a property of one bundle adjustment's gauge. The output records the model's
file hashes; a mesh may only be scaled by a result whose hashes match the model it was
trained on. Applying the scale happens after mesh extraction, preserving topology and
vertex colours, verified by reloading; the exporter refuses to overwrite and rejects
non-finite scales.

Metric evaluation of the scaled mesh then uses rigid registration
(`python -m augenblick.eval.mesh --rigid`), so any remaining size error stays in the
residuals instead of being fitted away. The similarity fit remains useful as a separately
labelled shape-only comparison, and the ratio of the bar-derived scale to the
similarity-fitted one is itself a measurement of metric-scale error.

## Best-effort estimation for single-bar captures

Most captures carry one bar, not two, so no in-capture check bar exists, and a gated
protocol would refuse much of the collection over failure modes the downstream metric
comparison catches anyway. Omitting `--check` therefore selects a best-effort estimate
instead of a verdict: the bar is measured over every short span of consecutive views
observing both targets (drift-tolerant, because marker positions drift smoothly over a
capture while short spans stay rigid), each span triangulated robustly under a
reprojection gate proportional to the bar's apparent pixel size (defocus-tolerant,
because the decoded centre error of large, out-of-focus targets grows with apparent
size), and the span lengths pool into a median scale. The output always carries the
estimate with its support (spans, rings) and dispersion (median absolute deviation,
interquartile range, spread); nothing is accepted or rejected, and a consumer must read
the support and dispersion columns, not just the number. On captures where a gated
windowed variant could also produce a scale, the two were measured to agree to about a
tenth of a percent. Prefer the two-bar protocol whenever a check bar exists: an
independent withheld ruler is strictly stronger evidence.

## The codebook

The detector's canonical code for a bit pattern does not determine the printed bar
length: the same code pair has been photographed on bars of different lengths in this
collection. Treating codes as identifying a bar produces a silent factor-level scale
error, and when both bars of a two-bar capture share a wrong assumed length, the check
bar cannot catch it. `--known-length-mm` is therefore always explicit, with no default.
`docs/bar-codebook.json` records every code pair observed so far with its
photo-verified printed length; before scaling a new capture, open one photograph,
read the printed labels and length, and extend the codebook.

## End-to-end walkthrough for a new specimen

1. Reconstruct with masked COLMAP as usual; the sparse model defines the gauge.
2. Detect the targets on the unmasked photographs of the bar-bearing position
   (requires a PGT-Toolkit checkout; it has no PyPI release):

       python -m augenblick.eval.detect \
           --images data/<scene>/images \
           --output detections.json \
           --toolkit <pgt-toolkit checkout>

3. Verify the codes and printed length on a photograph; consult and extend
   `docs/bar-codebook.json`.
4. Estimate, check and apply the scale (omit `--check` for single-bar captures):

       python -m augenblick.eval.scale \
           --model results/<scene>/colmap/sparse/0 \
           --scene data/<scene> \
           --detections detections.json \
           --primary 1675 1419 --check 1181 1949 \
           --known-length-mm 100 \
           --apply-to results/<scene>/2dgs/train/ours_30000/fuse_post.ply \
           --json-out scale_result.json

5. Score the scaled mesh rigidly against the metric reference:

       python -m augenblick.eval.mesh --ours mesh_mm.ply --reference gt.obj \
           --rigid --absolute-tolerances 0.25 0.5 1.0 --json-out rigid_scores.json

On HiPerGator, `hpg_slurm/scale_markers.sbatch` runs steps 2 and 4 and
`hpg_slurm/geometry_eval.sbatch` runs step 5.

## Usage

    python -m augenblick.eval.scale \
        --model results/<scene>/colmap/sparse/0 \
        --scene data/<scene> \
        --detections detections.json \
        --primary 1675 1419 --check 1181 1949 \
        --known-length-mm 100 \
        --apply-to results/<scene>/2dgs/train/ours_30000/fuse_post.ply \
        --json-out scale_result.json

Exit status is 0 only when the scale is accepted. The JSON records every window, the
acceptance checks, the model hashes and, when a mesh was exported, its provenance.
