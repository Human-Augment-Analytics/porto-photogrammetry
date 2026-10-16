# Metric scale from scale bars (`augenblick.eval.scale`)

A COLMAP reconstruction's gauge is arbitrary, so its meshes carry no unit, and a similarity fit
against a metric reference cannot report scale error because it *solves for* scale. This module
recovers millimetres from coded targets on printed bars photographed with the specimen, verifies
the result against a bar the fit never touched, and **refuses to hand over a number it cannot
check**.

## Usage

```bash
python -m augenblick.eval.scale \
    --model results/<scene>/colmap/sparse/0 \
    --scene data/<scene> \
    --detections detections.json \
    --primary 1675 1419 --check 1181 1949 \
    [--known-length-mm 100] \
    [--apply-to <mesh.ply> [--output <mesh_mm.ply>]] \
    --json-out scale_result.json
```

**Exit status is 0 only when the scale is accepted** (1 otherwise) — usable as a gate in a job
script. `--apply-to` writes nothing unless accepted; the default output is
`<apply-to stem>_mm.ply`, and the exporter refuses to overwrite and rejects non-finite scales.

## Inputs and their traps

- **The COLMAP model.** Its camera fit must not have consumed the marker observations, or the
  held-out reprojection checks stop being independent. The scale belongs to *that* bundle
  adjustment's gauge: the JSON records the model's file hashes, and a mesh may only be scaled by
  a result whose hashes match the model it was trained on.
- **The unmasked photographs.** Detection needs the bars visible, so it always reads
  `images/`, however strict the object masks used for reconstruction were.
- **`detections.json`** — per image, the decoded target code and OpenCV pixel centre of every
  candidate. The detector itself is out of scope (any circular-coded-target decoder works).
- Two boundary conventions: OpenCV's first pixel centre is (0, 0) and COLMAP's is (0.5, 0.5) —
  `colmap_pixel()` adds the half pixel itself. And **the detector's canonical code for a bit
  pattern is not the label printed on the bar**; verify the mapping once per capture by
  inspecting a photograph, because a wrong pairing associates different physical targets and
  produces silent garbage.
- Optional `<scene>/capture_manifest.json`: `{"image": ..., "position": ...}` entries assigning
  each photograph to a turntable position. Present, ring grouping keys on (camera body, lens,
  focal length, position); absent, on the camera hardware alone. A capture resolving to fewer
  than the two rings acceptance requires is reported as such up front — its windows are still
  evaluated for diagnostics but cannot produce an accepted scale. Rings shorter than one window
  are recorded and skipped, since a wrapped window would put the holdout in the training set.

## Windows, not one triangulation

One static 3D position per marker over a full capture is an assumption, and on real captures it
fails held-out reprojection by several pixels while short spans of consecutive views pass the
same gates with bar lengths agreeing to a fraction of a percent. Scale is therefore estimated
per **window**: every span of `window_size` consecutive views at a fixed stride, wrapping around
each ring. All windows are enumerated and reported; none is selected by its outcome.

## Gates (`Gates`, all predeclared and recorded in the output)

Per window and marker: `WINDOW_SIZE = 24`, `WINDOW_STRIDE = 8`, ≥16 accepted observations of
which 8 evenly spaced are **withheld**; the rest must triangulate with ≥8 inliers within 2 px,
60% consensus and 2° of ray separation; and ≥60% of the withheld observations must reproject
within 2 px — the check that asks the fitted point to predict measurements it never saw.

Acceptance across windows: ≥6 passing windows spanning ≥2 rings and ≥3 offsets, primary-bar
length spread ≤0.5%, and the check bar within 1% of nominal in every passing window and in the
median. `KNOWN_LENGTH_MM = 100.0` is the nominal centre-to-centre length.

Two rings is **support breadth**: windows from one ring can share a view-angle-dependent
systematic and still agree with each other. The check bar never participates in fitting, and the
primary/check roles are fixed in advance — do not swap them after seeing which behaves better.

## What acceptance does not establish

Not the printed bars' dimensional accuracy (the length is nominal), not a systematic camera error
shared by both bars, and not a global surface bias of the mesh being scaled. The passing-window
range is a spread, not a standard error: overlapping windows share observations.

## Downstream

Score the scaled mesh with `python -m augenblick.eval.mesh --rigid`, so any remaining size error
stays in the residual instead of being fitted away (see pipeline-evaluation.md). The similarity
fit remains useful as a separately labelled shape-only comparison, and the **ratio of the
bar-derived scale to the similarity-fitted one is itself a measurement of metric-scale error**.
