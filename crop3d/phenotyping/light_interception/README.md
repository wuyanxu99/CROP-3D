# Canopy Light Interception

Estimate **apparent direct-light interception** for a canopy under a specified LED setting, using a reconstructed point cloud and a measured PPFD light-field model. This directory contains the full workflow: ROI cropping, light-field fitting, top-voxel interception, and five-region batch comparison.

For the paper-level overview, see [`docs/light_interception.md`](../../../docs/light_interception.md) at the repository root.

## File Overview

| File | Role |
| --- | --- |
| `light_field_model.py` | Light-field library: read measured PPFD, RBF fitting, `predict_ppfd()` queries |
| `fit_ppfd_light_field.py` | **Offline**: fit `ppfd_light_field_model.pkl` from measured PPFD tables |
| `prepare_canopy_roi.py` | **Stage 1**: crop the light-field ROI, filter plant points, export QC |
| `calculate_light_interception.py` | **Core computation**: top voxels + PPFD field → interception metrics |
| `run_region_batch.py` | **Batch entry**: five consecutive 1.5 m regions, interception, comparison table |

## Pipeline

```text
Measured PPFD table (xlsx/csv)
        |
        v
fit_ppfd_light_field.py  -->  ppfd_light_field_model.pkl
                                        |
Full-scene point cloud PLY              |
        |                               |
        v                               |
prepare_canopy_roi.py                   |
  -> plant_local.ply                    |
  -> roi_summary.json                   |
        |                               |
        +----------> calculate_light_interception.py
                           -> interception_summary.csv/json
                           -> top-voxel / block PPFD maps

run_region_batch.py chains the two stages for regions 1-1 ... 1-5 and writes region_comparison_setting_*.csv
```

The **core algorithm lives in `calculate_light_interception.py`**. `run_region_batch.py` only orchestrates the calls.

## Default Method and Parameters

These defaults match the research workspace:

| Parameter | Default |
| --- | --- |
| Surface completion mode | `oriented_disk_splat` |
| Voxel size `voxel_size` | `0.002 m` |
| Splat radius `splat_radius` | `0.006 m` |
| Disk thickness `disk_thickness` | `0.002 m` |
| Normal neighborhood radius `normal_radius` | `0.018 m` |
| Light-field domain `x x y x z` | `1.5 x 0.56 x 0.45 m` |
| Five-region comparison setting | `240` |
| Five-region `origin_x` | `0, 1.5, 3.0, 4.5, 6.0` (`origin_z=0.275`) |

The estimate is an **apparent direct-light interception** value: PPFD is integrated over the reconstructed top surface under the measured light-field model. It does not model diffuse light, leaf optical properties, or repeated lower-canopy interception.

## How to Run

Run from the repository root `CROP-3D/` after `pip install -e .`:

### 1. Fit the light-field model (one-time / when measured data changes)

```bash
python -m crop3d.phenotyping.light_interception.fit_ppfd_light_field \
  --input examples/light_interception_demo/input/standardized_ppfd.csv \
  --output-dir outputs/light_interception/ppfd_model
```

Outputs: `ppfd_light_field_model.pkl`, fitting reports, and optional prediction grids / heatmaps.

### 2. Prepare a single-region ROI

```bash
python -m crop3d.phenotyping.light_interception.prepare_canopy_roi \
  --input-ply data/full_point_cloud_unavailable.ply \
  --output-dir outputs/light_interception/roi/1-1 \
  --origin-x 0 --origin-y 0 --origin-z 0.275
```

Main outputs:

- `plant_local.ply` — filtered plant points in local light-field coordinates (**interception input**)
- `roi_summary.json` — ROI origin and crop metadata
- `roi_qc_3d.png` — 3D QC figure

### 3. Compute light interception for one region (core step)

```bash
python -m crop3d.phenotyping.light_interception.calculate_light_interception \
  --plant-ply outputs/light_interception/roi/1-1/plant_local.ply \
  --roi-summary outputs/light_interception/roi/1-1/roi_summary.json \
  --model outputs/light_interception/ppfd_model/ppfd_light_field_model.pkl \
  --output-dir outputs/light_interception/regions/1-1 \
  --settings 240
```

Main outputs:

- `interception_summary.csv` / `interception_summary.json`
- `top_voxels_setting_*.csv`
- `ppfd_top_voxels_3d_setting_*.png` and related visualizations

### 4. Five-region batch comparison (recommended entry point)

```bash
python -m crop3d.phenotyping.light_interception.run_region_batch \
  --input-ply data/full_point_cloud_unavailable.ply \
  --model outputs/light_interception/ppfd_model/ppfd_light_field_model.pkl \
  --setting 240
```

Summary output: `outputs/light_interception/regions/region_comparison_setting_240.csv`

### 5. Demo package (single ROI with bundled sample assets)

```bash
python scripts/run_light_interception_demo.py
```

Demo assets live in [`examples/light_interception_demo/`](../../../examples/light_interception_demo/).

## Core Metrics

For each `setting`, PPFD is queried on canopy top voxels and integrated:

| Metric | Meaning |
| --- | --- |
| `I_region` | Regional intercepted photon flux (umol s^-1) |
| `R_region` | `I_region / A_region`, mean regional PPFD (umol m^-2 s^-1) |
| `L_model` | `I_region / I_empty_region`, interception fraction relative to an empty canopy (0-1) |
| `canopy_cover` | Canopy projected area / regional floor area |

`I_empty_region` is computed by querying PPFD on an empty plant grid at reference height `z_ref` (default 0.30 m).

## Coordinate Convention

- **Local light-field coordinates**: `prepare_canopy_roi` maps rack coordinates into a light-field box anchored at `(origin_x, origin_y, origin_z)`; `plant_local.ply` uses this frame.
- **PPFD queries**: `calculate_light_interception` computes in local coordinates; the origin stored in `roi_summary.json` is used only for global height conversion and optional global PLY export.
- Legacy `pointcloud-y-up` axis mapping can be selected with `prepare_canopy_roi --coord-mode`.

## Dependencies

- Python 3.10+
- `numpy`, `plyfile` (PLY I/O)
- `scipy` (RBF light-field fitting in `light_field_model.py`)
- `matplotlib` (QC and interception plots)

See `environment.yml` at the repository root for the full environment.
