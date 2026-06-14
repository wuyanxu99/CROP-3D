# Quick Start

Run from the repository root.

## One-Command Reconstruction

The root-level script orchestrates the reconstruction pipeline. The packaged
`examples/reconstruction_demo/images` directory follows the required
`images/<camera_name>/` layout and can be used for a small smoke test. Use the
same layout for larger full-scene runs.

Inspect the plan:

```bash
python quick_start.py \
  --input-images examples/reconstruction_demo/images \
  --scene work_dirs/reconstruction_demo_scene \
  --run-name demo_reconstruction \
  --dry-run \
  --skip-train
```

With real images and dependencies installed:

```bash
python quick_start.py \
  --input-images path/to/scene/images \
  --scene work_dirs/reconstruction_demo_scene \
  --run-name demo_reconstruction \
  --iterations 30000 \
  --resolution 2
```

Short smoke test with the packaged example images:

```bash
python quick_start.py \
  --input-images examples/reconstruction_demo/images \
  --scene work_dirs/reconstruction_demo_scene \
  --run-name smoke_reconstruction \
  --iterations 10 \
  --resolution 4 \
  --lambda-depth 0 \
  --no-eval \
  --test-iterations 10 \
  --save-iterations 10
```

The script prepares `scene/images/<camera>/`, runs COLMAP with fixed shared intrinsics, optionally applies AprilTag metric alignment, optionally generates DA3 depth priors, converts COLMAP `cameras.bin` to `PINHOLE`, and then calls the packaged 2DGS training entry `crop3d/reconstruction/twodgs/train.py`.

Images are copied into the scene by default. Avoid symlinked images for COLMAP
runs because COLMAP may resolve the real path and break
`single_camera_per_folder` camera grouping.

DA3 also receives the same fixed camera parameters by default so the later
steps stay on the same intrinsics set.

AprilTag alignment and DA3 depth priors are disabled by default so that small
smoke-test datasets without tags or local DA3 weights can run through COLMAP,
PINHOLE conversion, and 2DGS training. Pass `--with-apriltag` and `--with-da3`
to enable those optional stages.

The PINHOLE conversion is needed because the packaged 2DGS loader only accepts `PINHOLE` and `SIMPLE_PINHOLE` camera models. If your COLMAP model is already compatible, pass `--skip-pinhole-conversion`.

If you do not want AprilTag or DA3 for a first full-data smoke test:

```bash
python quick_start.py \
  --input-images path/to/scene/images \
  --scene work_dirs/reconstruction_demo_scene \
  --iterations 30000
```

## Plant Height Demo

```bash
python scripts/run_plant_height_demo.py
```

This copies packaged reference outputs and regenerates:

- `outputs/plant_height/heights.csv`
- `outputs/plant_height/errors.csv`
- `outputs/plant_height/figures/metrics/`
- `outputs/plant_height/figures/xz_side/`

The demo uses the curated canopy PLY and summary CSV under `examples/plant_height_demo/reference_output/`.

## Light Interception Demo

```bash
python scripts/run_light_interception_demo.py
```

This recomputes the single-region canopy light-interception result from:

- `examples/light_interception_demo/input/plant_local.ply`
- `examples/light_interception_demo/input/roi_summary.json`
- `examples/light_interception_demo/input/ppfd_light_field_model.pkl`

It writes results under `outputs/light_interception/` and copies the packaged five-region comparison summary into `outputs/light_interception/regions/`.
