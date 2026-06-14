# CROP-3D: A Metric 3D Reconstruction Paradigm for Mobile Crop Monitoring in Plant Factory

CROP-3D is the paper artifact and project homepage for real-scene crop reconstruction and phenotyping. It packages a compact, reproducible workflow for multi-view COLMAP/2D Gaussian Splatting reconstruction, metric alignment, plant-height measurement, and canopy light-interception estimation.

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#repository-at-a-glance">Repository at a Glance</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#reproduction-scope">Reproduction Scope</a> ·
  <a href="#citation">Citation</a>
</p>

<div align="center">
  <img src="media/teaser.gif" alt="CROP-3D teaser" width="85%" />
</div>

<p align="center">
  <img src="media/pipeline.png" alt="CROP-3D pipeline" width="95%" />
</p>

## Overview

CROP-3D follows a three-stage workflow:

1. Reconstruction: multi-view crop reconstruction with COLMAP, metric alignment, optional DA3 depth priors, and 2DGS training.
2. Plant-height phenotyping: height estimation from reconstructed canopy point clouds and x-z side-view inspection.
3. Canopy light interception: PPFD light-field fitting and oriented disk surface completion for interception estimation.

This repository is a paper-friendly release, not a full raw-data dump. The large datasets and heavyweight training assets are kept external so the homepage stays easy to browse and easy to clone.

## Repository at a Glance

| Area | What it covers | Key paths |
| --- | --- | --- |
| Reconstruction | COLMAP, metric alignment, DA3 priors, and 2DGS training | [`quick_start.py`](quick_start.py), [`crop3d/reconstruction/`](crop3d/reconstruction/), [`examples/reconstruction_demo/`](examples/reconstruction_demo/) |
| Plant height | Metric height evaluation and side-view figures | [`scripts/run_plant_height_demo.py`](scripts/run_plant_height_demo.py), [`examples/plant_height_demo/`](examples/plant_height_demo/), [`docs/plant_height.md`](docs/plant_height.md) |
| Light interception | ROI cropping, PPFD light-field model, and interception results | [`scripts/run_light_interception_demo.py`](scripts/run_light_interception_demo.py), [`crop3d/phenotyping/light_interception/`](crop3d/phenotyping/light_interception/), [`docs/light_interception.md`](docs/light_interception.md) |
| Documentation | Setup, dataset boundary, and reproduction notes | [`docs/`](docs/), [`examples/`](examples/) |

## Quick Start

Create the environment:

```bash
conda env create -f environment.yml
conda activate crop3d-paper
pip install -e .
pip install crop3d/reconstruction/submodules/diff-surfel-rasterization
pip install crop3d/reconstruction/submodules/simple-knn
```

Inspect the reconstruction plan:

```bash
python quick_start.py \
  --input-images examples/reconstruction_demo/images \
  --scene work_dirs/reconstruction_demo_scene \
  --run-name demo_reconstruction \
  --dry-run \
  --skip-train
```

Run the compact demos:

```bash
python scripts/run_plant_height_demo.py
python scripts/run_light_interception_demo.py
```

Full usage notes live in [docs/quick_start.md](docs/quick_start.md).

## Reproduction Scope

This artifact supports two levels of reproduction:

- Included: compact plant-height outputs, one light-interception ROI, five-region summary results, and the cleaned reconstruction / phenotyping scripts.
- Not included: full raw camera datasets, complete COLMAP databases, large checkpoints, and full-size training outputs.

See [docs/reproduction.md](docs/reproduction.md) for the exact boundary.

## Documentation

- [Installation](docs/installation.md)
- [Dataset](docs/dataset.md)
- [Model Weights](docs/model_weights.md)
- [Reproduction](docs/reproduction.md)
- [Plant Height](docs/plant_height.md)
- [Light Interception](docs/light_interception.md)

## Citation

If you use this artifact, please cite the accompanying paper. The citation metadata lives in [CITATION.cff](CITATION.cff) and will be updated with the final publication details before release.

## License

The license is still a placeholder. Before public release, replace [LICENSE](LICENSE) with the final license and verify compatibility for third-party components.
