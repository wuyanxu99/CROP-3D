# Plant-Height Demo

The plant-height demo exposes the final downstream workflow used in the research workspace, but with compact packaged assets.

## Method Summary

- Segmentation mode: `adaptive_soft_growth`
- Ground reference: fixed `ground_z = 0.46 m`
- Evaluation target: manual highest point (`gt_highest_cm`)
- Height definition: `height_cm = (top_z - ground_z) * 100`
- Top estimate: median of the highest top-k canopy surface cells

## Included Assets

- `examples/plant_height_demo/reference_output/heights.csv`
- `examples/plant_height_demo/reference_output/errors.csv`
- `examples/plant_height_demo/reference_output/canopy.ply`
- `examples/plant_height_demo/reference_output/qc_all/*_surface.ply`
- `media/plant_height_workflow.png`
- `media/plant_height_xz_side.png`

The full raw point cloud is not included because it is large. The copied scripts retain the full measurement implementation, but the packaged demo focuses on regenerating visual summaries from curated outputs.

## Run

```bash
python scripts/run_plant_height_demo.py
```

