# Benchmark workflow

The Snakemake workflow runs reconstruction and evaluation across a specimen collection.
Each experiment has its own directory (a benchmark tree), with configuration, inputs,
reconstructions and scores. Experiments can reuse the same SfM models while training
separate models on different view sets.

## Layout

```text
config.json                         specimen roster and experiment settings
recipes.json                        scale-bar codes and known lengths
inputs/{images,masks,gt_artec,realityscan}/
sfm/                                COLMAP scenes
2dgs/  pgsr/  gw/                    reconstruction outputs
markers/  scales/  splits/           preprocessing outputs
eval/<backend>/<specimen>/           metric meshes and geometry scores
nvs/<backend>/<specimen>/            held-out image scores
work/logs/                          job logs
```

## Configuration

The workflow reads `config.json` from the experiment directory.

| Setting | Purpose |
|---|---|
| `roster.keep` | Specimens to process |
| `sources` | Locations of input data and existing reconstructions |
| `repo`, `toolkit` | Augenblick checkout and marker-detection toolkit paths |
| `recipes` | Scale-bar recipe file; defaults to `recipes.json` |
| `references` | Specimen names mapped to reference mesh filenames |
| `backends` | Reconstruction methods: `2dgs`, `pgsr`, `gw` |
| `train` | Set to `true` to train models; otherwise evaluate existing meshes |
| `split` | `null` for all views, `holdout:n` for every nth view, or `manual` |
| `mesh_paths` | Optional overrides for each backend's mesh output path |

The workflow has default mesh paths for 2DGS, PGSR and GW. Other registered methods
need an explicit `mesh_paths` entry and a compatible `cameras.json` export for metric
lineage checks. In particular, the GeoSVR wrapper does not currently export that file;
its CLI support alone does not make it ready for this metric benchmark. The convenience
launcher accepts only the three backends with built-in paths.

For a manual split, provide `splits_manual/<specimen>/split.json` with `train` and
`test` lists of image stems. Validation rejects empty sets, overlapping views and
names absent from the sparse reconstruction. Training uses the split only when
evaluation is enabled; the workflow passes the required `--eval` flag.

## Running

Install Augenblick, the required reconstruction backends and Snakemake 7. Run from
inside the experiment directory so logs and workflow metadata stay with its outputs.

```bash
REPO=/path/to/augenblick
PYTHON_BIN=/path/to/environment/bin/python
cd /path/to/experiment

# Inspect the jobs without executing them.
snakemake -s "$REPO/workflow/Snakefile" \
  --config tree="$PWD" python="$PYTHON_BIN" -n

# Run locally.
snakemake -s "$REPO/workflow/Snakefile" \
  --config tree="$PWD" python="$PYTHON_BIN" --cores 1
```

Stages cover marker detection, scale estimation, split preparation, training, metric
mesh export, geometry evaluation and novel-view evaluation. Geometry requires a
reference mesh; novel-view evaluation requires a held-out split and a trained model.
The shared novel-view scorer reports PSNR, SSIM, LPIPS and DISTS; install the evaluation
dependencies in `requirements.txt`.
See [scale-protocol.md](scale-protocol.md) for the scale-bar setup.

For SLURM, add `--profile "$REPO/workflow/profiles/hipergator"` to the command.
That profile and the rule resource requests use HiPerGator settings; adapt the account,
QoS, partitions and resource limits for another cluster.

## Reusing inputs

To create an experiment that links another tree's inputs and SfM:

```bash
"$PYTHON_BIN" "$REPO/tools/instantiate_tree.py" \
  --from /path/to/source --to /path/to/new_experiment \
  --name ring_holdout --split manual
```

The tool selects reference-paired specimens by default; use `--only` to select others.
Provide the new tree's scale-bar recipes and manual splits, then set `train` and
`backends` in its config. Adopted stages are symlinks with a provenance record.

## Validation and provenance

Before metric scoring, the workflow checks that the mesh cameras and scale estimate
match the sparse model. Stages record output hashes in stamps and input/code provenance
in `inputs.json`. Moving a tree can trigger rebuilding because records contain absolute
paths.

Check that computed outputs remain inside the experiment directory with:

```bash
PYTHONPATH="$REPO/src" "$PYTHON_BIN" -m augenblick.bench audit --tree /path/to/experiment
```

`workflow/run_experiments.sh` is a convenience launcher for multiple configured trees.
Set its repository, interpreter and experiment paths for your installation. It updates
tree configs, including when `DRY=1`; use Snakemake's `-n` directly for inspection.
