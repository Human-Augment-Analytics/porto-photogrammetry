# SuGaR Evaluation

SuGaR accepts the shared `--resolution` (`-r`) setting and forwards it through
vanilla training, coarse/refined training, mesh extraction, and held-out rendering.
The combined vendored training script keeps `-r` for regularization and receives
resolution through its long flag.

As in vanilla 3DGS, `-1` caps image width at 1600, `1/2/4/8` divide both dimensions,
and other positive values specify target width. Portrait images do not have a
separate long-side cap. Use the same resolution across backend comparisons.

New run manifests record resolution and texture square size. The SuGaR atlas
default is now 8; configuration rejects sizes below 5 because 4 collapses each
UV triangle. Existing exports and metrics are not repaired by changing defaults.

## Rescoring

```bash
PYTHONPATH=src python -m augenblick.eval.nvs --scene <scene> --output <run>
```

When the normal `test/` root is absent, this command discovers SuGaR's
`test_splat/` and `test_mesh/`. It uses the same scorer as inline backend evaluation
and writes separate splat/mesh metrics plus the combined `nvs_metrics.json`.
`--iteration` selects that iteration in both products. Both render sets must exist;
missing targets are an error. Manifest settings are included in per-product
metadata, or `null` when unknown.

The changed default resolution does not make historical 1920-pixel renders
equivalent to new renders. Preserve historical measurements and rerender explicitly
at a declared common resolution before making new cross-backend comparisons.