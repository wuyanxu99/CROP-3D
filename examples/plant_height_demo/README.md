# Plant-Height Demo Assets

This folder contains compact assets for regenerating the plant-height figures in the paper artifact.

- `reference_output/heights.csv`: packaged prediction table.
- `reference_output/errors.csv`: ranked error table.
- `reference_output/canopy.ply`: compact merged canopy PLY used by the side-view plotter.
- `reference_output/qc_all/*_surface.ply`: small per-plant surface PLY files used to mark top-k points.
- `input/plant_height_GT_demo.csv`: lightweight demo table derived from the packaged result columns.

Run from the repository root:

```bash
python scripts/run_plant_height_demo.py
```

The full raw point cloud is not included in this Git repository.

