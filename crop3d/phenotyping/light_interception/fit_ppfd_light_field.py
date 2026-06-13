#!/usr/bin/env python3
"""Fit the measured LED PPFD light-field model."""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .light_field_model import (
        LightFieldConfig,
        config_to_dict,
        fit_light_field,
        fit_layered_light_field,
        fit_scaled_spatial_light_field,
        read_ppfd_records,
        regression_metrics,
        validate_records,
        write_json,
        write_records_csv,
    )
except ImportError:  # pragma: no cover - direct script execution
    from crop3d.phenotyping.light_interception.light_field_model import (
        LightFieldConfig,
        config_to_dict,
        fit_light_field,
        fit_layered_light_field,
        fit_scaled_spatial_light_field,
        read_ppfd_records,
        regression_metrics,
        validate_records,
        write_json,
        write_records_csv,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit PPFD = f(setting, x, y, z) from measured LED light-field samples."
    )
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "examples" / "light_interception_demo" / "input" / "standardized_ppfd.csv"),
        help="Input PPFD table: xlsx or csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "light_interception" / "ppfd_model"),
        help="Directory for model and reports.",
    )
    parser.add_argument(
        "--smoothing-candidates",
        default="0,0.1,1,5,10,25,50,100",
        help="Comma-separated RBF smoothing candidates selected by leave-one-position-out CV.",
    )
    parser.add_argument("--kernel", default="thin_plate_spline", help="RBF kernel name.")
    parser.add_argument("--edge-decay", type=float, default=0.75, help="Boundary pseudo-point PPFD multiplier.")
    parser.add_argument("--grid-step-x", type=float, default=0.025, help="Prediction grid x step in meters.")
    parser.add_argument("--grid-step-y", type=float, default=0.02, help="Prediction grid y step in meters.")
    parser.add_argument("--skip-grid", action="store_true", help="Do not export a dense prediction grid CSV.")
    parser.add_argument("--skip-plots", action="store_true", help="Do not export PPFD heatmap PNGs.")
    parser.add_argument("--no-edge-prior", action="store_true", help="Disable boundary pseudo-observations.")
    parser.add_argument(
        "--height-prior",
        action="store_true",
        help="Enable soft non-decreasing-z pseudo-observations for filtered/missing height knots.",
    )
    parser.add_argument(
        "--height-prior-gain-fraction",
        type=float,
        default=0.35,
        help="Fraction of the median measured vertical PPFD gain used to generate height-prior pseudo points.",
    )
    parser.add_argument(
        "--height-prior-min-gain",
        type=float,
        default=0.0,
        help="Minimum PPFD gain per meter used by height-prior pseudo points.",
    )
    parser.add_argument(
        "--exclude-z-values",
        default="",
        help="Comma-separated measured z values to exclude before fitting, e.g. 0.45.",
    )
    parser.add_argument(
        "--filter-nondecreasing-z",
        action="store_true",
        help=(
            "Remove low outliers that violate the physical prior that PPFD should "
            "not decrease with z for the same setting,x,y profile."
        ),
    )
    parser.add_argument(
        "--monotonic-filter-tolerance",
        type=float,
        default=0.0,
        help="Allowed PPFD decrease before a higher-z point is filtered as too low.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Hold out this fraction of usable records as test data.")
    parser.add_argument("--test-seed", type=int, default=42, help="Random seed for the holdout test split.")
    parser.add_argument(
        "--model-type",
        choices=["global4d", "layered", "scaled_spatial"],
        default="layered",
        help=(
            "global4d fits one RBF over (setting,x,y,z); layered fits one RBF per measured z over "
            "(setting,x,y); scaled_spatial fits P(setting,x,y,z)=S(setting)*F(x,y,z)."
        ),
    )
    parser.add_argument(
        "--setting-scale-mode",
        choices=["setting_value", "train_mean"],
        default="setting_value",
        help="Setting scale source for --model-type scaled_spatial.",
    )
    parser.add_argument(
        "--force-monotonic-z",
        action="store_true",
        help="Always apply non-decreasing z post-processing at query time.",
    )
    parser.add_argument(
        "--no-monotonic-z",
        action="store_true",
        help="Never apply non-decreasing z post-processing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.force_monotonic_z and args.no_monotonic_z:
        raise SystemExit("Choose only one of --force-monotonic-z or --no-monotonic-z.")

    smoothing = tuple(float(x.strip()) for x in args.smoothing_candidates.split(",") if x.strip())
    monotonic_flag = None
    if args.force_monotonic_z:
        monotonic_flag = True
    elif args.no_monotonic_z:
        monotonic_flag = False

    config = LightFieldConfig(
        kernel=args.kernel,
        smoothing_candidates=smoothing,
        edge_decay=args.edge_decay,
        use_edge_prior=not args.no_edge_prior,
        use_height_prior=args.height_prior,
        height_prior_gain_fraction=args.height_prior_gain_fraction,
        height_prior_min_gain=args.height_prior_min_gain,
        setting_scale_mode=args.setting_scale_mode,
        enforce_monotonic_z=monotonic_flag,
    )

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_records = read_ppfd_records(input_path)
    excluded_z_values = parse_float_list(args.exclude_z_values)
    records_after_z_filter = filter_records(raw_records, excluded_z_values)
    excluded_records = [r for r in raw_records if any(abs(r.z - z) <= 1e-9 for z in excluded_z_values)]
    if args.filter_nondecreasing_z:
        records, monotonic_removed_records = filter_nondecreasing_z_profiles(
            records_after_z_filter,
            tolerance=args.monotonic_filter_tolerance,
        )
    else:
        records = records_after_z_filter
        monotonic_removed_records = []
    train_records, test_records = split_train_test(records, args.test_ratio, args.test_seed)
    data_summary = validate_records(records, config.domain)
    raw_data_summary = validate_records(raw_records, config.domain)
    train_summary = validate_records(train_records, config.domain)
    test_summary = validate_records(test_records, config.domain) if test_records else None
    if args.model_type == "global4d":
        model = fit_light_field(train_records, config)
    elif args.model_type == "scaled_spatial":
        model = fit_scaled_spatial_light_field(train_records, config)
    else:
        model = fit_layered_light_field(train_records, config)

    standardized_path = output_dir / "standardized_ppfd.csv"
    raw_standardized_path = output_dir / "standardized_all_ppfd.csv"
    train_path = output_dir / "train_ppfd.csv"
    test_path = output_dir / "test_ppfd.csv"
    excluded_path = output_dir / "excluded_ppfd.csv"
    monotonic_removed_path = output_dir / "monotonic_removed_ppfd.csv"
    model_path = output_dir / "ppfd_light_field_model.pkl"
    summary_path = output_dir / "fit_summary.json"
    grid_path = None
    heatmap_paths = []

    write_records_csv(records, standardized_path)
    write_records_csv(raw_records, raw_standardized_path)
    write_records_csv(train_records, train_path)
    if test_records:
        write_records_csv(test_records, test_path)
    if excluded_records:
        write_records_csv(excluded_records, excluded_path)
    if monotonic_removed_records:
        write_records_csv(monotonic_removed_records, monotonic_removed_path)
    model.save(model_path)

    if not args.skip_grid:
        grid_path = output_dir / "predicted_grid.csv"
        export_prediction_grid(model, grid_path, x_step=args.grid_step_x, y_step=args.grid_step_y)

    if not args.skip_plots:
        heatmap_paths = export_heatmaps(model, output_dir / "heatmaps", x_step=args.grid_step_x, y_step=args.grid_step_y)

    write_json(
        {
            "input": str(input_path),
            "standardized_columns": ["setting", "x", "y", "z", "ppfd"],
            "domain_m": {
                "x": [config.domain.x_min, config.domain.x_max],
                "y": [config.domain.y_min, config.domain.y_max],
                "z": [config.domain.z_min, config.domain.z_max],
            },
            "raw_data_summary": raw_data_summary,
            "usable_data_summary": data_summary,
            "train_summary": train_summary,
            "test_summary": test_summary,
            "excluded_z_values": excluded_z_values,
            "excluded_record_count": len(excluded_records),
            "monotonic_filter": {
                "enabled": bool(args.filter_nondecreasing_z),
                "tolerance": args.monotonic_filter_tolerance,
                "removed_record_count": len(monotonic_removed_records),
                "removed_by_z": count_by(monotonic_removed_records, lambda r: r.z),
                "removed_by_setting_z": count_by(
                    monotonic_removed_records,
                    lambda r: f"{r.setting:g}@z={r.z:.2f}",
                ),
                "removed_records": str(monotonic_removed_path) if monotonic_removed_records else None,
            },
            "test_ratio": args.test_ratio,
            "test_seed": args.test_seed,
            "model_summary": model.training_summary,
            "model_type": args.model_type,
            "fit_metrics_train": evaluate_records(model, train_records),
            "fit_metrics_usable_all": evaluate_records(model, records),
            "test_metrics": evaluate_records(model, test_records) if test_records else None,
            "metrics_by_z_usable_all": metrics_by_group(
                model,
                records,
                key_fn=lambda r: r.z,
                key_name="z",
            ),
            "metrics_by_z_test": metrics_by_group(
                model,
                test_records,
                key_fn=lambda r: r.z,
                key_name="z",
            )
            if test_records
            else None,
            "metrics_by_setting_usable_all": metrics_by_group(
                model,
                records,
                key_fn=lambda r: r.setting,
                key_name="setting",
            ),
            "metrics_by_setting_test": metrics_by_group(
                model,
                test_records,
                key_fn=lambda r: r.setting,
                key_name="setting",
            )
            if test_records
            else None,
            "config": config_to_dict(config),
            "kernel": config.kernel,
            "edge_decay": config.edge_decay,
            "use_edge_prior": config.use_edge_prior,
            "prediction_grid": str(grid_path) if grid_path else None,
            "heatmaps": [str(path) for path in heatmap_paths],
        },
        summary_path,
    )

    selected = model.training_summary["smoothing_cv"]["selected_metrics"]
    smoothing_display = getattr(model, "smoothing", getattr(model, "smoothing_by_z", None))
    print("Fitted PPFD light-field model")
    print(f"  model type: {args.model_type}")
    print(f"  raw records: {raw_data_summary['count']} measured")
    print(f"  usable records: {data_summary['count']} measured")
    print(f"  excluded records: {len(excluded_records)}")
    print(f"  monotonic-filtered records: {len(monotonic_removed_records)}")
    print(f"  train/test records: {len(train_records)}/{len(test_records)}")
    print(f"  plant positions: {data_summary['plant_positions']}")
    print(f"  selected smoothing: {smoothing_display}")
    print(
        "  CV metrics: "
        f"MAE={selected['mae']:.3f}, RMSE={selected['rmse']:.3f}, "
        f"R2={selected['r2']:.3f}, r={selected['pearson_r']:.3f}"
    )
    print(f"  monotonic z postprocess: {model.training_summary['enforce_monotonic_z']}")
    print(f"  model: {model_path}")
    print(f"  summary: {summary_path}")
    print(f"  standardized data: {standardized_path}")
    if test_records:
        test_metrics = evaluate_records(model, test_records)
        print(
            "  test metrics: "
            f"MAE={test_metrics['mae']:.3f}, RMSE={test_metrics['rmse']:.3f}, "
            f"R2={test_metrics['r2']:.3f}, r={test_metrics['pearson_r']:.3f}"
        )
    if grid_path:
        print(f"  prediction grid: {grid_path}")
    if heatmap_paths:
        print(f"  heatmaps: {output_dir / 'heatmaps'}")


def export_prediction_grid(model, path: Path, x_step: float, y_step: float) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    d = model.config.domain
    settings = model.training_summary["smoothing_cv"].get("settings") or [170.0, 205.0, 240.0]
    z_levels = list(model.config.height_knots)
    xs = _grid_axis(d.x_min, d.x_max, x_step)
    ys = _grid_axis(d.y_min, d.y_max, y_step)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["setting", "x", "y", "z", "ppfd_pred"])
        writer.writeheader()
        for setting in settings:
            for z in z_levels:
                xx, yy = np.meshgrid(xs, ys, indexing="xy")
                pred = model.predict_ppfd(setting, xx, yy, z)
                for x, y, value in zip(xx.reshape(-1), yy.reshape(-1), pred.reshape(-1)):
                    writer.writerow(
                        {
                            "setting": float(setting),
                            "x": round(float(x), 6),
                            "y": round(float(y), 6),
                            "z": round(float(z), 6),
                            "ppfd_pred": round(float(value), 6),
                        }
                    )


def export_heatmaps(model, output_dir: Path, x_step: float, y_step: float) -> list[Path]:
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir.parent / "matplotlib_cache"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"  heatmaps skipped: matplotlib unavailable ({exc})")
        return []

    d = model.config.domain
    settings = model.training_summary["smoothing_cv"].get("settings") or [170.0, 205.0, 240.0]
    z_levels = list(model.config.height_knots)
    xs = _grid_axis(d.x_min, d.x_max, x_step)
    ys = _grid_axis(d.y_min, d.y_max, y_step)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    paths: list[Path] = []
    for setting in settings:
        for z in z_levels:
            pred = model.predict_ppfd(setting, xx, yy, z)
            fig, ax = plt.subplots(figsize=(8, 3.4), constrained_layout=True)
            image = ax.imshow(
                pred,
                origin="lower",
                extent=[d.x_min, d.x_max, d.y_min, d.y_max],
                aspect="auto",
                cmap="viridis",
            )
            ax.set_title(f"PPFD prediction, setting={setting:g}, z={z:.2f} m")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            cbar = fig.colorbar(image, ax=ax)
            cbar.set_label("PPFD (umol m^-2 s^-1)")
            path = output_dir / f"ppfd_heatmap_setting_{setting:g}_z_{z:.2f}m.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(path)
    return paths


def _grid_axis(start: float, stop: float, step: float):
    import numpy as np

    if step <= 0:
        raise ValueError("Grid step must be positive.")
    count = int(round((stop - start) / step))
    values = start + np.arange(count + 1, dtype=float) * step
    if values[-1] < stop:
        values = np.append(values, stop)
    values[-1] = stop
    return values


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def filter_records(records, excluded_z_values: list[float]):
    if not excluded_z_values:
        return list(records)
    return [r for r in records if not any(abs(r.z - z) <= 1e-9 for z in excluded_z_values)]


def filter_nondecreasing_z_profiles(records, tolerance: float = 0.0):
    """Keep each setting/x/y vertical profile non-decreasing by removing low points."""
    if tolerance < 0:
        raise ValueError("--monotonic-filter-tolerance must be non-negative.")
    groups = {}
    for record in records:
        groups.setdefault((record.setting, record.x, record.y), []).append(record)

    kept = []
    removed = []
    for _, profile in sorted(groups.items(), key=lambda item: item[0]):
        running_max = None
        for record in sorted(profile, key=lambda r: r.z):
            if running_max is None or record.ppfd + tolerance >= running_max:
                kept.append(record)
                running_max = record.ppfd if running_max is None else max(running_max, record.ppfd)
            else:
                removed.append(record)
    return kept, removed


def split_train_test(records, test_ratio: float, seed: int):
    records = list(records)
    if test_ratio <= 0:
        return records, []
    if not 0 < test_ratio < 1:
        raise ValueError("--test-ratio must be in [0, 1).")
    rng = random.Random(seed)
    groups = {}
    for idx, record in enumerate(records):
        groups.setdefault((record.setting, record.z), []).append(idx)
    test_indices = set()
    for _, indices in sorted(groups.items()):
        group_test_count = max(1, round(len(indices) * test_ratio))
        group_test_count = min(group_test_count, len(indices) - 1)
        if group_test_count > 0:
            test_indices.update(rng.sample(indices, group_test_count))
    train = [record for idx, record in enumerate(records) if idx not in test_indices]
    test = [record for idx, record in enumerate(records) if idx in test_indices]
    return train, test


def evaluate_records(model, records):
    records = list(records)
    if not records:
        return None
    return regression_metrics(
        [r.ppfd for r in records],
        model.predict_ppfd([r.setting for r in records], [r.x for r in records], [r.y for r in records], [r.z for r in records]),
    )


def metrics_by_group(model, records, key_fn, key_name: str):
    groups = {}
    for record in records:
        key = key_fn(record)
        groups.setdefault(key, []).append(record)
    return {
        str(key): {
            key_name: key,
            **evaluate_records(model, group),
            "bias": mean_prediction_error(model, group),
        }
        for key, group in sorted(groups.items(), key=lambda item: item[0])
    }


def mean_prediction_error(model, records) -> float:
    records = list(records)
    pred = model.predict_ppfd(
        [r.setting for r in records],
        [r.x for r in records],
        [r.y for r in records],
        [r.z for r in records],
    )
    return sum(float(p) - r.ppfd for p, r in zip(pred, records)) / len(records)


def count_by(records, key_fn):
    counts = {}
    for record in records:
        key = key_fn(record)
        if isinstance(key, float):
            key = f"{key:g}"
        else:
            key = str(key)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


if __name__ == "__main__":
    main()
