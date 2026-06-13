# Canopy Light-Interception Demo

The light-interception demo packages a single cropped canopy region and a small fitted PPFD light-field model.

## Method Summary

- Light-field model: measured PPFD model stored as `ppfd_light_field_model.pkl`
- Default surface mode: `oriented_disk_splat`
- Voxel size: `0.002 m`
- Splat radius: `0.006 m`
- Disk thickness: `0.002 m`
- Normal radius: `0.018 m`
- Default setting for five-region comparison: `240`

The estimate should be interpreted as an apparent direct-light interception estimate under the captured geometry and measured light-field model. It does not model diffuse light, leaf optical properties, or repeated lower-canopy interception.

## Included Assets

- `examples/light_interception_demo/input/plant_local.ply`
- `examples/light_interception_demo/input/roi_summary.json`
- `examples/light_interception_demo/input/ppfd_light_field_model.pkl`
- `examples/light_interception_demo/five_regions/region_comparison_setting_240.csv`
- `examples/light_interception_demo/five_regions/region_comparison_setting_240.json`
- `media/light_interception_overview.png`

## Run

```bash
python scripts/run_light_interception_demo.py
```

