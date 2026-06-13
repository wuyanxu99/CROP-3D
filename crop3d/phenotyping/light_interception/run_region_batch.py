#!/usr/bin/env python3
"""Batch crop consecutive canopy regions and compare light interception."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = str(REPO_ROOT / "examples" / "light_interception_demo" / "input" / "ppfd_light_field_model.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop consecutive 1.5 m canopy regions and compare setting-240 light interception."
    )
    parser.add_argument("--input-ply", default=str(REPO_ROOT / "data" / "full_point_cloud_unavailable.ply"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--roi-root", default=str(REPO_ROOT / "outputs" / "light_interception" / "roi"))
    parser.add_argument("--interception-root", default=str(REPO_ROOT / "outputs" / "light_interception" / "regions"))
    parser.add_argument("--row-prefix", default="1")
    parser.add_argument("--region-count", type=int, default=5)
    parser.add_argument("--region-length", type=float, default=1.5)
    parser.add_argument("--origin-x-start", type=float, default=0.0)
    parser.add_argument("--origin-y", type=float, default=0.0)
    parser.add_argument("--origin-z", type=float, default=0.275)
    parser.add_argument("--setting", type=float, default=240.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roi_root = Path(args.roi_root)
    interception_root = Path(args.interception_root)
    roi_root.mkdir(parents=True, exist_ok=True)
    interception_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx in range(args.region_count):
        region = f"{args.row_prefix}-{idx + 1}"
        origin_x = args.origin_x_start + idx * args.region_length
        roi_dir = roi_root / region
        interception_dir = interception_root / region

        if not args.skip_existing or not (roi_dir / "plant_local.ply").exists():
            run_command(
                [
                    args.python,
                    "-B",
                    "-m",
                    "crop3d.phenotyping.light_interception.prepare_canopy_roi",
                    "--input-ply",
                    args.input_ply,
                    "--output-dir",
                    str(roi_dir),
                    "--origin-x",
                    f"{origin_x:g}",
                    "--origin-y",
                    f"{args.origin_y:g}",
                    "--origin-z",
                    f"{args.origin_z:g}",
                ],
                cwd=REPO_ROOT,
            )

        if should_run_interception(args, interception_dir):
            run_command(
                [
                    args.python,
                    "-B",
                    "-m",
                    "crop3d.phenotyping.light_interception.calculate_light_interception",
                    "--plant-ply",
                    str(roi_dir / "plant_local.ply"),
                    "--roi-summary",
                    str(roi_dir / "roi_summary.json"),
                    "--model",
                    args.model,
                    "--output-dir",
                    str(interception_dir),
                    "--settings",
                    f"{args.setting:g}",
                ],
                cwd=REPO_ROOT,
            )

        rows.append(read_region_summary(region, origin_x, args.origin_y, args.origin_z, roi_dir, interception_dir))

    csv_path = interception_root / f"region_comparison_setting_{format_setting(args.setting)}.csv"
    json_path = interception_root / f"region_comparison_setting_{format_setting(args.setting)}.json"
    write_comparison_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "input_ply": args.input_ply,
            "model": args.model,
            "setting": args.setting,
            "region_count": args.region_count,
            "region_length": args.region_length,
            "origin": {"x_start": args.origin_x_start, "y": args.origin_y, "z": args.origin_z},
            "roi_root": str(roi_root),
            "interception_root": str(interception_root),
            "regions": rows,
        },
    )

    print("Region comparison complete")
    print(f"  regions: {args.region_count}")
    print(f"  setting: {args.setting:g}")
    print(f"  comparison csv: {csv_path}")
    print(f"  comparison json: {json_path}")
    for row in rows:
        print(
            f"  {row['region']}: plant={row['plant_filtered_point_count']}, "
            f"cover={row['canopy_cover']:.4f}, "
            f"R={row['R_region']:.3f}, L={row['L_model']:.4f}"
        )


def run_command(command: list[str], cwd: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def should_run_interception(args: argparse.Namespace, interception_dir: Path) -> bool:
    summary_path = interception_dir / "interception_summary.json"
    if not args.skip_existing:
        return True
    if not (interception_dir / "interception_summary.csv").exists() or not summary_path.exists():
        return True
    summary = read_json(summary_path)
    top_summary = summary.get("top_voxel_summary", {})
    return "filled_component_cleanup" not in top_summary


def read_region_summary(
    region: str,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    roi_dir: Path,
    interception_dir: Path,
) -> dict:
    roi_summary = read_json(roi_dir / "roi_summary.json")
    interception_summary = read_json(interception_dir / "interception_summary.json")
    row = dict(interception_summary["per_setting"][0])
    top_summary = interception_summary["top_voxel_summary"]
    return {
        "region": region,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "origin_z": origin_z,
        "roi_raw_point_count": roi_summary["roi_raw_point_count"],
        "plant_filtered_point_count": roi_summary["plant_filtered_point_count"],
        "N_top_voxels": row["N_top_voxels"],
        "N_source_voxels": top_summary.get("N_source_voxels"),
        "N_filled_voxels_raw": top_summary.get("N_filled_voxels_raw"),
        "N_filled_voxels": top_summary.get("N_filled_voxels"),
        "min_filled_component_voxels": row.get("min_filled_component_voxels"),
        "filled_component_count": row.get("filled_component_count"),
        "filled_component_kept_count": row.get("filled_component_kept_count"),
        "filled_component_removed_count": row.get("filled_component_removed_count"),
        "filled_component_removed_voxels": row.get("filled_component_removed_voxels"),
        "canopy_projected_area": row["canopy_projected_area"],
        "canopy_cover": row["canopy_cover"],
        "z_top_mean": top_summary["z_top_mean"],
        "z_top_min": top_summary["z_top_min"],
        "z_top_max": top_summary["z_top_max"],
        "z_top_mean_global_m": top_summary.get("z_top_mean_global_m"),
        "z_top_min_global_m": top_summary.get("z_top_min_global_m"),
        "z_top_max_global_m": top_summary.get("z_top_max_global_m"),
        "z_top_mean_above_ground_cm": top_summary.get("z_top_mean_above_ground_cm"),
        "z_top_min_above_ground_cm": top_summary.get("z_top_min_above_ground_cm"),
        "z_top_max_above_ground_cm": top_summary.get("z_top_max_above_ground_cm"),
        "I_region": row["I_region"],
        "R_region": row["R_region"],
        "I_empty_region": row["I_empty_region"],
        "L_model": row["L_model"],
        "ppfd_top_mean": row["ppfd_top_mean"],
        "ppfd_top_min": row["ppfd_top_min"],
        "ppfd_top_max": row["ppfd_top_max"],
        "roi_dir": str(roi_dir),
        "interception_dir": str(interception_dir),
    }


def write_comparison_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "region",
        "origin_x",
        "origin_y",
        "origin_z",
        "roi_raw_point_count",
        "plant_filtered_point_count",
        "N_top_voxels",
        "N_source_voxels",
        "N_filled_voxels_raw",
        "N_filled_voxels",
        "min_filled_component_voxels",
        "filled_component_count",
        "filled_component_kept_count",
        "filled_component_removed_count",
        "filled_component_removed_voxels",
        "canopy_projected_area",
        "canopy_cover",
        "z_top_mean",
        "z_top_min",
        "z_top_max",
        "z_top_mean_global_m",
        "z_top_min_global_m",
        "z_top_max_global_m",
        "z_top_mean_above_ground_cm",
        "z_top_min_above_ground_cm",
        "z_top_max_above_ground_cm",
        "I_region",
        "R_region",
        "I_empty_region",
        "L_model",
        "ppfd_top_mean",
        "ppfd_top_min",
        "ppfd_top_max",
        "roi_dir",
        "interception_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def format_setting(setting: float) -> str:
    return f"{setting:g}".replace(".", "p")


if __name__ == "__main__":
    main()
