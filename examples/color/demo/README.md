# Colour calibration demo

This self-contained demo exercises `augenblick color` without private museum data. It creates
three synthetic camera groups with known colour casts, one automatically detected 24-patch
reference image per camera, one capture per camera, and unchanged binary masks.

## Run

From the repository root, in an environment containing NumPy, OpenCV, and Pillow:

```bash
python examples/color/demo/generate_demo.py

augenblick color \
  --input examples/color/demo/generated/input \
  --output examples/color/demo/generated/output \
  --config examples/color/demo/generated/config.json
```

Generated files stay under `examples/color/demo/generated/`, which is ignored by Git.

## Inspect

Open `generated/output/color_calibration_report.json` and confirm:

- all three camera groups processed one capture;
- `skipped_images` is empty;
- camera 1 uses the identity transform;
- mean patch Delta E76 decreases for cameras 2 and 3;
- the three masks in `generated/output/` are byte-identical to their inputs; and
- `mask_clipping` contains one foreground-only measurement per camera with no warnings; and
- rectified chart previews exist under `generated/output/calibration_previews/`.

The demo establishes command behavior, report structure, mask preservation, and measurable
relative alignment. It is not evidence for absolute chart calibration or reconstruction quality.

## Absolute-reference mode

Once a physical chart and its value edition are verified, add these fields to the configuration:

```json
{
  "target_name": "Manufacturer, model, value edition and source",
  "target_srgb_d65": [
    [0.0, 0.0, 0.0]
  ]
}
```

`target_srgb_d65` must contain exactly 24 normalized `[R, G, B]` entries in rectified
row-major order; the single entry above only illustrates the schema and is intentionally not a
valid target. In absolute mode every camera, including `reference_camera`, is fitted to the
verified target. The report records `calibration_mode` and `target_name` so results retain their
reference provenance.

## Museum-data validation already completed

The private validation used `UF_Herp_3998` and is summarized here without redistributing source
photographs:

| Metric | Original | Calibrated | Change |
|---|---:|---:|---:|
| Camera 2 mean patch Delta E76 | 9.218 | 3.131 | -66.0% |
| Camera 3 mean patch Delta E76 | 3.698 | 1.432 | -61.3% |
| Registered COLMAP images | 36/36 | 36/36 | unchanged |
| Verified feature inliers | 28,084 | 28,658 | +2.0% |
| Sparse points | 3,845 | 3,929 | +2.2% |
| Mean reprojection error | 0.670 px | 0.676 px | +0.006 px |

Interpret the sparse reconstruction as preserved, not improved. The 0.006 px reprojection-error
change is operationally neutral, and the result currently covers one specimen.
