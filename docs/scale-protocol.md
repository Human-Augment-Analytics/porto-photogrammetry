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

## Usage

    python -m augenblick.eval.scale \
        --model results/<scene>/colmap/sparse/0 \
        --scene data/<scene> \
        --detections detections.json \
        --primary 1675 1419 --check 1181 1949 \
        --apply-to results/<scene>/2dgs/train/ours_30000/fuse_post.ply \
        --json-out scale_result.json

Exit status is 0 only when the scale is accepted. The JSON records every window, the
acceptance checks, the model hashes and, when a mesh was exported, its provenance.
