# Light-Interception Demo Assets

This folder contains compact assets for recomputing one canopy light-interception ROI.

- `input/plant_local.ply`: cropped plant points in local light-field coordinates.
- `input/roi_summary.json`: ROI origin and crop metadata.
- `input/ppfd_light_field_model.pkl`: fitted PPFD light-field model.
- `input/standardized_ppfd.csv`: compact measured PPFD table used by the model-fitting script.
- `reference_output/`: reference outputs from the research workspace.
- `five_regions/`: packaged five-region comparison summaries.

Run from the repository root:

```bash
python scripts/run_light_interception_demo.py
```

The full rack-scale point cloud and large top-voxel CSV files are intentionally not included.

