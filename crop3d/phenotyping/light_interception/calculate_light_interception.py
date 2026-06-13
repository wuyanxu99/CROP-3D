#!/usr/bin/env python3
"""Calculate canopy light interception from top voxels and a fitted PPFD field."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from .light_field_model import load_light_field
except ImportError:  # pragma: no cover - direct script execution
    from crop3d.phenotyping.light_interception.light_field_model import load_light_field

try:
    from plyfile import PlyData, PlyElement
except ImportError as exc:  # pragma: no cover - import guard
    PlyData = None
    PlyElement = None
    PLY_IMPORT_ERROR = exc
else:
    PLY_IMPORT_ERROR = None


DOMAIN_BOX_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute instantaneous canopy light interception from 2 mm top voxels."
    )
    demo_root = REPO_ROOT / "examples" / "light_interception_demo"
    parser.add_argument("--plant-ply", default=str(demo_root / "input" / "plant_local.ply"))
    parser.add_argument("--model", default=str(demo_root / "input" / "ppfd_light_field_model.pkl"))
    parser.add_argument("--roi-summary", default=str(demo_root / "input" / "roi_summary.json"))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs" / "light_interception"))
    parser.add_argument("--settings", default="170,205,240", help="Comma-separated light settings.")
    parser.add_argument("--voxel-size", type=float, default=0.002)
    parser.add_argument(
        "--surface-mode",
        choices=("raw", "sphere_splat", "oriented_disk_splat"),
        default="oriented_disk_splat",
        help=(
            "raw uses observed point voxels only; sphere_splat fills nearby 3D voxels; "
            "oriented_disk_splat estimates local normals and fills thin leaf-like disks."
        ),
    )
    parser.add_argument(
        "--splat-radius",
        type=float,
        default=0.006,
        help="Sphere/disk radius in meters used by splat surface modes.",
    )
    parser.add_argument(
        "--disk-thickness",
        type=float,
        default=0.002,
        help="Full disk thickness in meters used when --surface-mode oriented_disk_splat.",
    )
    parser.add_argument(
        "--normal-radius",
        type=float,
        default=0.018,
        help="Neighbor search radius in meters for oriented disk normal estimation.",
    )
    parser.add_argument(
        "--normal-min-neighbors",
        type=int,
        default=8,
        help="Minimum neighbors required for local PCA normal estimation.",
    )
    parser.add_argument(
        "--min-filled-component-voxels",
        type=int,
        default=2500,
        help=(
            "Remove disconnected completed voxel components smaller than this size after surface completion. "
            "Set <=1 to disable."
        ),
    )
    parser.add_argument("--z-ref", type=float, default=0.30, help="Reference height for empty-region flux.")
    parser.add_argument("--x-min", type=float, default=0.0)
    parser.add_argument("--x-max", type=float, default=1.5)
    parser.add_argument("--y-min", type=float, default=0.0)
    parser.add_argument("--y-max", type=float, default=0.56)
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=0.45)
    parser.add_argument("--block-sizes", default="0.01,0.05,0.10", help="Comma-separated block sizes in meters.")
    parser.add_argument("--origin-x", type=float, default=None, help="Global origin x; defaults to roi_summary.")
    parser.add_argument("--origin-y", type=float, default=None, help="Global origin y; defaults to roi_summary.")
    parser.add_argument("--origin-z", type=float, default=None, help="Global origin z; defaults to roi_summary.")
    parser.add_argument(
        "--ground-z",
        type=float,
        default=0.46,
        help="Global cultivation plane z used only for display/summary height conversion.",
    )
    parser.add_argument("--save-top-global-ply", action="store_true")
    parser.add_argument("--ppfd-vmin", type=float, default=0.0, help="Minimum PPFD value for visualization color scale.")
    parser.add_argument("--ppfd-vmax", type=float, default=400.0, help="Maximum PPFD value for visualization color scale.")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if PLY_IMPORT_ERROR is not None:
        raise RuntimeError("plyfile is required to read/write PLY point clouds.") from PLY_IMPORT_ERROR
    if args.voxel_size <= 0:
        raise ValueError("voxel_size must be positive.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib_cache"))

    settings = parse_float_list(args.settings)
    block_sizes = parse_float_list(args.block_sizes)
    roi_summary = read_optional_json(Path(args.roi_summary))
    origin = resolve_origin(args, roi_summary)

    points, rgb = read_ply_points(Path(args.plant_ply))
    in_domain_mask = domain_mask(points, args)
    points_in_domain = points[in_domain_mask]
    rgb_in_domain = rgb[in_domain_mask]

    top = extract_top_voxels(points_in_domain, rgb_in_domain, args)
    model = load_light_field(args.model)
    empty_grid = build_empty_grid(args)
    top_height_summary = converted_height_stats(top["z_top"], origin[2], args.ground_z, "z_top")
    source_height_summary = converted_height_stats(
        top["source_top"]["z_top"],
        origin[2],
        args.ground_z,
        "source_z_top",
    )

    summary_rows = []
    summary_json = {
        "input": {
            "plant_ply": args.plant_ply,
            "model": args.model,
            "roi_summary": args.roi_summary if Path(args.roi_summary).exists() else None,
        },
        "domain": {
            "x": [args.x_min, args.x_max],
            "y": [args.y_min, args.y_max],
            "z": [args.z_min, args.z_max],
        },
        "origin_local_to_global": {"x": origin[0], "y": origin[1], "z": origin[2]},
        "height_reference": {
            "ground_z": args.ground_z,
            "light_plane_global_z": origin[2] + args.z_max,
            "light_plane_above_ground_cm": (origin[2] + args.z_max - args.ground_z) * 100.0,
            "note": "PPFD calculation keeps local light-field z; *_above_ground_cm fields are for reporting and plotting.",
        },
        "settings": settings,
        "voxel_size": args.voxel_size,
        "surface_mode": args.surface_mode,
        "splat_radius": args.splat_radius if args.surface_mode in {"sphere_splat", "oriented_disk_splat"} else 0.0,
        "disk_thickness": args.disk_thickness if args.surface_mode == "oriented_disk_splat" else 0.0,
        "normal_radius": args.normal_radius if args.surface_mode == "oriented_disk_splat" else 0.0,
        "normal_min_neighbors": args.normal_min_neighbors if args.surface_mode == "oriented_disk_splat" else 0,
        "min_filled_component_voxels": (
            args.min_filled_component_voxels if args.surface_mode in {"sphere_splat", "oriented_disk_splat"} else 0
        ),
        "voxel_area": top["voxel_area"],
        "z_ref": args.z_ref,
        "A_region": top["A_region"],
        "point_counts": {
            "input": int(len(points)),
            "in_domain": int(len(points_in_domain)),
            "out_of_domain": int(len(points) - len(points_in_domain)),
        },
        "top_voxel_summary": {
            "N_top_voxels": int(len(top["x"])),
            "N_source_voxels": int(top["N_source_voxels"]),
            "N_filled_voxels_raw": int(top["N_filled_voxels_raw"]),
            "N_filled_voxels": int(top["N_filled_voxels"]),
            "N_source_top_voxels": int(len(top["source_top"]["x"])),
            "normal_estimation": top.get("normal_estimation", {}),
            "filled_component_cleanup": top.get("filled_component_cleanup", {}),
            "canopy_projected_area": float(len(top["x"]) * top["voxel_area"]),
            "canopy_cover": safe_divide(len(top["x"]) * top["voxel_area"], top["A_region"]),
            "source_canopy_projected_area": float(len(top["source_top"]["x"]) * top["voxel_area"]),
            "source_canopy_cover": safe_divide(len(top["source_top"]["x"]) * top["voxel_area"], top["A_region"]),
            "z_top_min": float(np.nanmin(top["z_top"])) if len(top["z_top"]) else None,
            "z_top_max": float(np.nanmax(top["z_top"])) if len(top["z_top"]) else None,
            "z_top_mean": float(np.nanmean(top["z_top"])) if len(top["z_top"]) else None,
            **top_height_summary,
            "source_z_top_min": float(np.nanmin(top["source_top"]["z_top"])) if len(top["source_top"]["z_top"]) else None,
            "source_z_top_max": float(np.nanmax(top["source_top"]["z_top"])) if len(top["source_top"]["z_top"]) else None,
            "source_z_top_mean": float(np.nanmean(top["source_top"]["z_top"])) if len(top["source_top"]["z_top"]) else None,
            **source_height_summary,
        },
        "warnings": [],
        "per_setting": [],
        "outputs": {
            "summary_csv": "interception_summary.csv",
            "summary_json": "interception_summary.json",
            "top_voxel_csv": [],
            "maps": [],
            "block_outputs": [],
            "top_global_ply": [],
        },
    }

    if len(points) != len(points_in_domain):
        summary_json["warnings"].append(
            f"{len(points) - len(points_in_domain)} plant points were outside the calculation domain and ignored."
        )
    if not len(top["x"]):
        raise RuntimeError("No top voxels were extracted from plant_local.ply.")

    setting_results = []
    for setting in settings:
        setting_label = format_setting(setting)
        ppfd_top = np.asarray(model.predict_ppfd(setting, top["x"], top["y"], top["z_top"]), dtype=float)
        intercepted_flux = ppfd_top * top["voxel_area"]
        empty_ppfd = np.asarray(model.predict_ppfd(setting, empty_grid["x"], empty_grid["y"], args.z_ref), dtype=float)
        I_region = float(np.nansum(intercepted_flux))
        I_empty_region = float(np.nansum(empty_ppfd * top["voxel_area"]))
        R_region = safe_divide(I_region, top["A_region"])
        L_model = safe_divide(I_region, I_empty_region)
        warnings = []
        if I_region <= 0:
            warnings.append("I_region <= 0")
        if I_empty_region <= 0:
            warnings.append("I_empty_region <= 0")
        if L_model is not None and not (0.0 <= L_model <= 1.0):
            warnings.append("L_model outside [0, 1]")

        top_csv = output_dir / f"top_voxels_setting_{setting_label}.csv"
        write_top_voxels_csv(top_csv, setting, top, ppfd_top, intercepted_flux)
        summary_json["outputs"]["top_voxel_csv"].append(top_csv.name)

        if args.save_top_global_ply:
            ply_path = output_dir / f"top_voxels_global_setting_{setting_label}.ply"
            colors = colorize_values(ppfd_top)
            top_global = np.column_stack([top["x"] + origin[0], top["y"] + origin[1], top["z_top"] + origin[2]])
            write_ply(ply_path, top_global, colors)
            summary_json["outputs"]["top_global_ply"].append(ply_path.name)

        blocks = {}
        for block_size in block_sizes:
            block = aggregate_blocks(top, intercepted_flux, empty_ppfd, empty_grid, block_size, args)
            block_label = block_size_label(block_size)
            csv_path = output_dir / f"block_{block_label}_setting_{setting_label}.csv"
            write_block_csv(csv_path, setting, block_size, block)
            blocks[block_label] = block
            summary_json["outputs"]["block_outputs"].append(csv_path.name)

        row = {
            "setting": setting,
            "N_top_voxels": int(len(top["x"])),
            "surface_mode": args.surface_mode,
            "splat_radius": args.splat_radius if args.surface_mode in {"sphere_splat", "oriented_disk_splat"} else 0.0,
            "disk_thickness": args.disk_thickness if args.surface_mode == "oriented_disk_splat" else 0.0,
            "N_source_voxels": int(top["N_source_voxels"]),
            "N_filled_voxels_raw": int(top["N_filled_voxels_raw"]),
            "N_filled_voxels": int(top["N_filled_voxels"]),
            "min_filled_component_voxels": (
                args.min_filled_component_voxels if args.surface_mode in {"sphere_splat", "oriented_disk_splat"} else 0
            ),
            "filled_component_count": int(top.get("filled_component_cleanup", {}).get("component_count", 0)),
            "filled_component_kept_count": int(top.get("filled_component_cleanup", {}).get("kept_components", 0)),
            "filled_component_removed_count": int(top.get("filled_component_cleanup", {}).get("removed_components", 0)),
            "filled_component_removed_voxels": int(top.get("filled_component_cleanup", {}).get("removed_voxels", 0)),
            "voxel_size": args.voxel_size,
            "voxel_area": top["voxel_area"],
            "A_region": top["A_region"],
            "canopy_projected_area": float(len(top["x"]) * top["voxel_area"]),
            "canopy_cover": safe_divide(len(top["x"]) * top["voxel_area"], top["A_region"]),
            "z_ref": args.z_ref,
            "I_region": I_region,
            "R_region": R_region,
            "I_empty_region": I_empty_region,
            "L_model": L_model,
            "ppfd_top_min": float(np.nanmin(ppfd_top)),
            "ppfd_top_max": float(np.nanmax(ppfd_top)),
            "ppfd_top_mean": float(np.nanmean(ppfd_top)),
            "warnings": "; ".join(warnings),
        }
        summary_rows.append(row)
        summary_json["per_setting"].append(row)
        setting_results.append(
            {
                "setting": setting,
                "setting_label": setting_label,
                "ppfd_top": ppfd_top,
                "intercepted_flux": intercepted_flux,
                "blocks": blocks,
            }
        )

    visual_scales = build_visual_scales(top, setting_results, block_sizes, args)
    summary_json["visualization_scales"] = visual_scales
    summary_json["outputs"]["top_voxel_3d_maps"] = []

    for result in setting_results:
        setting_label = result["setting_label"]
        map_paths = plot_setting_maps(
            output_dir,
            setting_label,
            top,
            result["ppfd_top"],
            result["intercepted_flux"],
            args,
            visual_scales,
        )
        summary_json["outputs"]["maps"].extend([p.name for p in map_paths])

        top_3d_path = output_dir / f"ppfd_top_voxels_3d_setting_{setting_label}.png"
        plot_top_voxels_3d(
            top_3d_path,
            top,
            result["ppfd_top"],
            "PPFD (umol m^-2 s^-1)",
            "hot",
            visual_scales["ppfd_on_canopy"][0],
            visual_scales["ppfd_on_canopy"][1],
            args,
        )
        summary_json["outputs"]["top_voxel_3d_maps"].append(top_3d_path.name)

        for block_size in block_sizes:
            block_label = block_size_label(block_size)
            block = result["blocks"][block_label]
            png_path = output_dir / f"block_{block_label}_setting_{setting_label}.png"
            vmin, vmax = visual_scales["block_R"][block_label]
            plot_block_map(png_path, block, setting_label, block_size, args, vmin=vmin, vmax=vmax)
            summary_json["outputs"]["block_outputs"].append(png_path.name)

    write_summary_csv(output_dir / "interception_summary.csv", summary_rows)
    write_json(output_dir / "interception_summary.json", summary_json)

    print("Calculated canopy light interception")
    print(f"  plant points in domain: {len(points_in_domain)} / {len(points)}")
    print(f"  top voxels: {len(top['x'])}")
    print(f"  canopy cover: {summary_json['top_voxel_summary']['canopy_cover']:.4f}")
    for row in summary_rows:
        print(
            f"  setting {row['setting']:g}: "
            f"I_region={row['I_region']:.6f} umol s^-1, "
            f"R_region={row['R_region']:.3f} umol m^-2 s^-1, "
            f"L_model={row['L_model']:.4f}"
        )
    print(f"  output: {output_dir}")


def parse_float_list(text: str) -> list[float]:
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError("At least one value is required.")
    return values


def read_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_origin(args: argparse.Namespace, summary: dict) -> tuple[float, float, float]:
    stored = summary.get("origin_offset_rack_to_light") or summary.get("origin_local_to_global") or {}
    return (
        float(args.origin_x if args.origin_x is not None else stored.get("x", 0.0)),
        float(args.origin_y if args.origin_y is not None else stored.get("y", 0.0)),
        float(args.origin_z if args.origin_z is not None else stored.get("z", 0.275)),
    )


def converted_height_stats(values: np.ndarray, origin_z: float, ground_z: float, prefix: str) -> dict:
    if len(values) == 0:
        return {
            f"{prefix}_min_global_m": None,
            f"{prefix}_max_global_m": None,
            f"{prefix}_mean_global_m": None,
            f"{prefix}_min_above_ground_cm": None,
            f"{prefix}_max_above_ground_cm": None,
            f"{prefix}_mean_above_ground_cm": None,
        }
    global_z = np.asarray(values, dtype=np.float64) + float(origin_z)
    above_ground_cm = (global_z - float(ground_z)) * 100.0
    return {
        f"{prefix}_min_global_m": float(np.nanmin(global_z)),
        f"{prefix}_max_global_m": float(np.nanmax(global_z)),
        f"{prefix}_mean_global_m": float(np.nanmean(global_z)),
        f"{prefix}_min_above_ground_cm": float(np.nanmin(above_ground_cm)),
        f"{prefix}_max_above_ground_cm": float(np.nanmax(above_ground_cm)),
        f"{prefix}_mean_above_ground_cm": float(np.nanmean(above_ground_cm)),
    }


def read_ply_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    points = np.column_stack(
        [
            np.asarray(vertex["x"], dtype=np.float64),
            np.asarray(vertex["y"], dtype=np.float64),
            np.asarray(vertex["z"], dtype=np.float64),
        ]
    )
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.float64)
        if np.nanmax(rgb) > 1.0:
            rgb /= 255.0
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        rgb = np.full((len(points), 3), 0.55, dtype=np.float64)
    return points, rgb


def domain_mask(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    return (
        (points[:, 0] >= args.x_min)
        & (points[:, 0] <= args.x_max)
        & (points[:, 1] >= args.y_min)
        & (points[:, 1] <= args.y_max)
        & (points[:, 2] >= args.z_min)
        & (points[:, 2] <= args.z_max)
    )


def extract_top_voxels(points: np.ndarray, rgb: np.ndarray, args: argparse.Namespace) -> dict:
    voxel_size = args.voxel_size
    nx = int(np.ceil((args.x_max - args.x_min) / voxel_size))
    ny = int(np.ceil((args.y_max - args.y_min) / voxel_size))
    nz = int(np.ceil((args.z_max - args.z_min) / voxel_size))
    A_region = (args.x_max - args.x_min) * (args.y_max - args.y_min)
    voxel_area = voxel_size * voxel_size

    ix = np.floor((points[:, 0] - args.x_min) / voxel_size).astype(np.int64)
    iy = np.floor((points[:, 1] - args.y_min) / voxel_size).astype(np.int64)
    iz = np.floor((points[:, 2] - args.z_min) / voxel_size).astype(np.int64)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    iz = np.clip(iz, 0, nz - 1)

    source_voxels = np.unique(np.column_stack([ix, iy, iz]), axis=0)
    source_top = top_surface_from_voxels(source_voxels, args, nx, ny)
    filled_voxels = source_voxels
    normal_summary = {}
    component_summary = component_cleanup_summary(0, 0, 0, 0)
    if args.surface_mode == "sphere_splat":
        filled_voxels = splat_voxels(source_voxels, args.splat_radius, voxel_size, nx, ny, nz)
    elif args.surface_mode == "oriented_disk_splat":
        filled_voxels, normal_summary = oriented_disk_splat_voxels(points, args, nx, ny, nz)
    raw_filled_count = int(len(filled_voxels))
    if args.surface_mode in {"sphere_splat", "oriented_disk_splat"}:
        filled_voxels, component_summary = filter_filled_components(
            filled_voxels,
            int(args.min_filled_component_voxels),
            nx,
            ny,
            nz,
        )

    ix = filled_voxels[:, 0]
    iy = filled_voxels[:, 1]
    iz = filled_voxels[:, 2]
    xy_key = ix * ny + iy
    unique_key, inverse = np.unique(xy_key, return_inverse=True)
    iz_top = np.full(len(unique_key), -1, dtype=np.int64)
    np.maximum.at(iz_top, inverse, iz)
    point_count = np.bincount(inverse, minlength=len(unique_key)).astype(np.int64)

    top_ix = unique_key // ny
    top_iy = unique_key % ny
    top_x = args.x_min + (top_ix.astype(np.float64) + 0.5) * voxel_size
    top_y = args.y_min + (top_iy.astype(np.float64) + 0.5) * voxel_size
    top_z = args.z_min + (iz_top.astype(np.float64) + 0.5) * voxel_size

    return {
        "ix": top_ix.astype(np.int64),
        "iy": top_iy.astype(np.int64),
        "iz_top": iz_top.astype(np.int64),
        "x": top_x,
        "y": top_y,
        "z_top": top_z,
        "point_count": point_count,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "voxel_area": float(voxel_area),
        "A_region": float(A_region),
        "N_source_voxels": int(len(source_voxels)),
        "N_filled_voxels_raw": raw_filled_count,
        "N_filled_voxels": int(len(filled_voxels)),
        "normal_estimation": normal_summary,
        "filled_component_cleanup": component_summary,
        "source_top": source_top,
    }


def top_surface_from_voxels(voxels: np.ndarray, args: argparse.Namespace, nx: int, ny: int) -> dict:
    ix = voxels[:, 0]
    iy = voxels[:, 1]
    iz = voxels[:, 2]
    xy_key = ix * ny + iy
    unique_key, inverse = np.unique(xy_key, return_inverse=True)
    iz_top = np.full(len(unique_key), -1, dtype=np.int64)
    np.maximum.at(iz_top, inverse, iz)
    point_count = np.bincount(inverse, minlength=len(unique_key)).astype(np.int64)

    top_ix = unique_key // ny
    top_iy = unique_key % ny
    top_x = args.x_min + (top_ix.astype(np.float64) + 0.5) * args.voxel_size
    top_y = args.y_min + (top_iy.astype(np.float64) + 0.5) * args.voxel_size
    top_z = args.z_min + (iz_top.astype(np.float64) + 0.5) * args.voxel_size
    return {
        "ix": top_ix.astype(np.int64),
        "iy": top_iy.astype(np.int64),
        "iz_top": iz_top.astype(np.int64),
        "x": top_x,
        "y": top_y,
        "z_top": top_z,
        "point_count": point_count,
        "nx": nx,
        "ny": ny,
    }


def splat_voxels(
    source_voxels: np.ndarray,
    radius: float,
    voxel_size: float,
    nx: int,
    ny: int,
    nz: int,
) -> np.ndarray:
    if radius <= 0:
        return source_voxels
    offsets = sphere_offsets(radius, voxel_size)
    expanded = source_voxels[:, None, :] + offsets[None, :, :]
    expanded = expanded.reshape(-1, 3)
    in_domain = (
        (expanded[:, 0] >= 0)
        & (expanded[:, 0] < nx)
        & (expanded[:, 1] >= 0)
        & (expanded[:, 1] < ny)
        & (expanded[:, 2] >= 0)
        & (expanded[:, 2] < nz)
    )
    return np.unique(expanded[in_domain], axis=0)


def sphere_offsets(radius: float, voxel_size: float) -> np.ndarray:
    r_vox = int(np.ceil(radius / voxel_size))
    grid = np.arange(-r_vox, r_vox + 1, dtype=np.int64)
    dx, dy, dz = np.meshgrid(grid, grid, grid, indexing="ij")
    dist = np.sqrt(dx.astype(float) ** 2 + dy.astype(float) ** 2 + dz.astype(float) ** 2) * voxel_size
    offsets = np.column_stack([dx[dist <= radius], dy[dist <= radius], dz[dist <= radius]]).astype(np.int64)
    return offsets


def oriented_disk_splat_voxels(points: np.ndarray, args: argparse.Namespace, nx: int, ny: int, nz: int):
    try:
        from scipy.spatial import cKDTree
    except ModuleNotFoundError as exc:
        raise RuntimeError("oriented_disk_splat requires scipy.spatial.cKDTree.") from exc

    tree = cKDTree(points)
    neighbor_lists = tree.query_ball_point(points, r=args.normal_radius)
    offsets = disk_candidate_offsets(args.splat_radius, args.disk_thickness, args.voxel_size)
    fallback_normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    chunks = []
    normal_count = 0
    fallback_count = 0
    normal_z_abs = []

    for point, neighbors in zip(points, neighbor_lists):
        if len(neighbors) >= args.normal_min_neighbors:
            normal = estimate_normal(points[np.asarray(neighbors, dtype=np.int64)])
            normal_count += 1
        else:
            normal = fallback_normal
            fallback_count += 1
        if normal[2] < 0:
            normal = -normal
        normal_z_abs.append(abs(float(normal[2])))

        local_offsets = offsets[np.abs(offsets @ normal) <= args.disk_thickness / 2.0 + args.voxel_size * 0.25]
        if len(local_offsets) == 0:
            local_offsets = np.zeros((1, 3), dtype=np.float64)
        coords = point[None, :] + local_offsets
        voxels = points_to_voxels(coords, args, nx, ny, nz)
        chunks.append(voxels)

    if chunks:
        filled = np.unique(np.vstack(chunks), axis=0)
    else:
        filled = np.empty((0, 3), dtype=np.int64)
    summary = {
        "normal_radius": args.normal_radius,
        "normal_min_neighbors": args.normal_min_neighbors,
        "normal_estimated_points": int(normal_count),
        "normal_fallback_points": int(fallback_count),
        "mean_abs_normal_z": float(np.mean(normal_z_abs)) if normal_z_abs else None,
        "disk_radius": args.splat_radius,
        "disk_thickness": args.disk_thickness,
        "candidate_offsets": int(len(offsets)),
    }
    return filled, summary


def estimate_normal(neighborhood: np.ndarray) -> np.ndarray:
    centered = neighborhood - np.mean(neighborhood, axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(neighborhood) - 1, 1)
    _, eigenvectors = np.linalg.eigh(cov)
    normal = eigenvectors[:, 0]
    norm = np.linalg.norm(normal)
    if not np.isfinite(norm) or norm <= 1e-9:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return normal / norm


def disk_candidate_offsets(radius: float, thickness: float, voxel_size: float) -> np.ndarray:
    r = max(radius, voxel_size / 2.0)
    half_extent = r + max(thickness, voxel_size) / 2.0 + voxel_size
    steps = np.arange(-np.ceil(half_extent / voxel_size), np.ceil(half_extent / voxel_size) + 1, dtype=np.float64)
    dx, dy, dz = np.meshgrid(steps, steps, steps, indexing="ij")
    offsets = np.column_stack([dx.reshape(-1), dy.reshape(-1), dz.reshape(-1)]) * voxel_size
    keep = np.linalg.norm(offsets, axis=1) <= half_extent
    return offsets[keep]


def points_to_voxels(points: np.ndarray, args: argparse.Namespace, nx: int, ny: int, nz: int) -> np.ndarray:
    ix = np.floor((points[:, 0] - args.x_min) / args.voxel_size).astype(np.int64)
    iy = np.floor((points[:, 1] - args.y_min) / args.voxel_size).astype(np.int64)
    iz = np.floor((points[:, 2] - args.z_min) / args.voxel_size).astype(np.int64)
    voxels = np.column_stack([ix, iy, iz])
    in_domain = (
        (voxels[:, 0] >= 0)
        & (voxels[:, 0] < nx)
        & (voxels[:, 1] >= 0)
        & (voxels[:, 1] < ny)
        & (voxels[:, 2] >= 0)
        & (voxels[:, 2] < nz)
    )
    return voxels[in_domain]


def filter_filled_components(
    voxels: np.ndarray,
    min_component_voxels: int,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, dict]:
    if len(voxels) == 0 or min_component_voxels <= 1:
        return voxels, component_cleanup_summary(0, 0, 0, 0)
    try:
        from scipy import ndimage
    except ModuleNotFoundError:
        return voxels, component_cleanup_summary(0, 0, 0, 0, skipped=True)

    grid = np.zeros((nx, ny, nz), dtype=bool)
    grid[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, component_count = ndimage.label(grid, structure=structure)
    component_sizes = np.bincount(labels[grid])
    keep_labels = np.flatnonzero(component_sizes >= int(min_component_voxels))
    keep_labels = keep_labels[keep_labels != 0]
    voxel_labels = labels[voxels[:, 0], voxels[:, 1], voxels[:, 2]]
    keep_mask = np.isin(voxel_labels, keep_labels)
    filtered = voxels[keep_mask]
    return filtered, component_cleanup_summary(
        component_count=int(component_count),
        kept_components=int(len(keep_labels)),
        removed_components=int(component_count - len(keep_labels)),
        removed_voxels=int(len(voxels) - len(filtered)),
    )


def component_cleanup_summary(
    component_count: int,
    kept_components: int,
    removed_components: int,
    removed_voxels: int,
    skipped: bool = False,
) -> dict:
    return {
        "enabled": not skipped,
        "component_count": int(component_count),
        "kept_components": int(kept_components),
        "removed_components": int(removed_components),
        "removed_voxels": int(removed_voxels),
    }


def build_empty_grid(args: argparse.Namespace) -> dict:
    voxel_size = args.voxel_size
    nx = int(np.ceil((args.x_max - args.x_min) / voxel_size))
    ny = int(np.ceil((args.y_max - args.y_min) / voxel_size))
    xs = args.x_min + (np.arange(nx, dtype=np.float64) + 0.5) * voxel_size
    ys = args.y_min + (np.arange(ny, dtype=np.float64) + 0.5) * voxel_size
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return {"x": xx.reshape(-1), "y": yy.reshape(-1), "nx": nx, "ny": ny}


def write_top_voxels_csv(path: Path, setting: float, top: dict, ppfd: np.ndarray, intercepted_flux: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["setting", "ix", "iy", "iz_top", "x", "y", "z_top", "ppfd", "area", "intercepted_flux"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(top["x"])):
            writer.writerow(
                {
                    "setting": f"{setting:g}",
                    "ix": int(top["ix"][i]),
                    "iy": int(top["iy"][i]),
                    "iz_top": int(top["iz_top"][i]),
                    "x": f"{top['x'][i]:.6f}",
                    "y": f"{top['y'][i]:.6f}",
                    "z_top": f"{top['z_top'][i]:.6f}",
                    "ppfd": f"{ppfd[i]:.6f}",
                    "area": f"{top['voxel_area']:.9f}",
                    "intercepted_flux": f"{intercepted_flux[i]:.9f}",
                }
            )


def aggregate_blocks(
    top: dict,
    intercepted_flux: np.ndarray,
    empty_ppfd: np.ndarray,
    empty_grid: dict,
    block_size: float,
    args: argparse.Namespace,
) -> dict:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    nbx = int(np.ceil((args.x_max - args.x_min) / block_size))
    nby = int(np.ceil((args.y_max - args.y_min) / block_size))
    n_blocks = nbx * nby
    voxel_area = top["voxel_area"]

    bix = np.clip(np.floor((top["x"] - args.x_min) / block_size).astype(np.int64), 0, nbx - 1)
    biy = np.clip(np.floor((top["y"] - args.y_min) / block_size).astype(np.int64), 0, nby - 1)
    block_key = bix * nby + biy
    I_block = np.bincount(block_key, weights=intercepted_flux, minlength=n_blocks).astype(np.float64)
    n_top = np.bincount(block_key, minlength=n_blocks).astype(np.int64)

    empty_bix = np.clip(np.floor((empty_grid["x"] - args.x_min) / block_size).astype(np.int64), 0, nbx - 1)
    empty_biy = np.clip(np.floor((empty_grid["y"] - args.y_min) / block_size).astype(np.int64), 0, nby - 1)
    empty_key = empty_bix * nby + empty_biy
    I_empty = np.bincount(empty_key, weights=empty_ppfd * voxel_area, minlength=n_blocks).astype(np.float64)

    rows = []
    for block_ix in range(nbx):
        for block_iy in range(nby):
            key = block_ix * nby + block_iy
            x0 = args.x_min + block_ix * block_size
            y0 = args.y_min + block_iy * block_size
            x1 = min(x0 + block_size, args.x_max)
            y1 = min(y0 + block_size, args.y_max)
            area = max(x1 - x0, 0.0) * max(y1 - y0, 0.0)
            occupied_area = float(n_top[key] * voxel_area)
            rows.append(
                {
                    "block_ix": block_ix,
                    "block_iy": block_iy,
                    "x_min": x0,
                    "x_max": x1,
                    "y_min": y0,
                    "y_max": y1,
                    "A_block": area,
                    "N_top_voxels": int(n_top[key]),
                    "occupied_area": occupied_area,
                    "canopy_cover_block": safe_divide(occupied_area, area),
                    "I_block": float(I_block[key]),
                    "R_block": safe_divide(float(I_block[key]), area),
                    "I_empty_block": float(I_empty[key]),
                    "L_model_block": safe_divide(float(I_block[key]), float(I_empty[key])),
                }
            )
    return {"rows": rows, "nbx": nbx, "nby": nby, "block_size": block_size}


def write_block_csv(path: Path, setting: float, block_size: float, block: dict) -> None:
    fieldnames = [
        "setting",
        "block_size",
        "block_ix",
        "block_iy",
        "x_min",
        "x_max",
        "y_min",
        "y_max",
        "A_block",
        "N_top_voxels",
        "occupied_area",
        "canopy_cover_block",
        "I_block",
        "R_block",
        "I_empty_block",
        "L_model_block",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in block["rows"]:
            out = {"setting": f"{setting:g}", "block_size": f"{block_size:.6f}"}
            out.update(row)
            writer.writerow(out)


def plot_setting_maps(
    output_dir: Path,
    setting_label: str,
    top: dict,
    ppfd: np.ndarray,
    intercepted_flux: np.ndarray,
    args: argparse.Namespace,
    visual_scales: dict,
) -> list[Path]:
    paths = []
    source = top["source_top"]
    source_grid = np.full((top["ny"], top["nx"]), np.nan, dtype=np.float64)
    source_grid[source["iy"], source["ix"]] = source["z_top"]
    source_path = output_dir / f"source_z_top_map_setting_{setting_label}.png"
    plot_grid_map(
        source_path,
        source_grid,
        "z_top (m)",
        "viridis",
        args,
        vmin=visual_scales["z_top_map"][0],
        vmax=visual_scales["z_top_map"][1],
    )
    paths.append(source_path)

    maps = [
        ("z_top_map", top["z_top"], "z_top (m)", "viridis", visual_scales["z_top_map"]),
        ("ppfd_on_canopy", ppfd, "PPFD (umol m^-2 s^-1)", "hot", visual_scales["ppfd_on_canopy"]),
        (
            "intercepted_flux_map",
            intercepted_flux,
            "intercepted flux (umol s^-1)",
            "magma",
            visual_scales["intercepted_flux_map"],
        ),
    ]
    for name, values, cbar_label, cmap, scale in maps:
        grid = np.full((top["ny"], top["nx"]), np.nan, dtype=np.float64)
        grid[top["iy"], top["ix"]] = values
        path = output_dir / f"{name}_setting_{setting_label}.png"
        plot_grid_map(path, grid, cbar_label, cmap, args, vmin=scale[0], vmax=scale[1])
        paths.append(path)
    return paths


def plot_grid_map(
    path: Path,
    grid: np.ndarray,
    cbar_label: str,
    cmap_name: str,
    args: argparse.Namespace,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    im = ax.imshow(
        grid,
        origin="lower",
        extent=[args.x_min, args.x_max, args.y_min, args.y_max],
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("x local (m)")
    ax.set_ylabel("y local (m)")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)


def plot_block_map(
    path: Path,
    block: dict,
    setting_label: str,
    block_size: float,
    args: argparse.Namespace,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = np.full((block["nby"], block["nbx"]), np.nan, dtype=np.float64)
    for row in block["rows"]:
        grid[row["block_iy"], row["block_ix"]] = row["R_block"]

    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    im = ax.imshow(
        grid,
        origin="lower",
        extent=[args.x_min, args.x_max, args.y_min, args.y_max],
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("x local (m)")
    ax.set_ylabel("y local (m)")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("R_block (umol m^-2 s^-1)")
    ax.text(
        0.99,
        0.98,
        f"setting={setting_label}, block={block_size_label(block_size)}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)


def build_visual_scales(top: dict, setting_results: list[dict], block_sizes: list[float], args: argparse.Namespace) -> dict:
    ppfd = np.concatenate([r["ppfd_top"] for r in setting_results])
    flux = np.concatenate([r["intercepted_flux"] for r in setting_results])
    if float(args.ppfd_vmax) <= float(args.ppfd_vmin):
        raise ValueError("--ppfd-vmax must be greater than --ppfd-vmin.")
    if float(np.nanmax(ppfd)) > float(args.ppfd_vmax):
        print(
            f"[Warn] max PPFD {float(np.nanmax(ppfd)):.2f} is above color scale "
            f"{float(args.ppfd_vmax):.2f}; values are clipped in the display."
        )
    scales = {
        "z_top_map": [float(np.nanmin(top["z_top"])), float(np.nanmax(top["z_top"]))],
        "ppfd_on_canopy": [float(args.ppfd_vmin), float(args.ppfd_vmax)],
        "intercepted_flux_map": [0.0, float(np.nanmax(flux))],
        "block_R": {},
    }
    for block_size in block_sizes:
        label = block_size_label(block_size)
        values = []
        for result in setting_results:
            values.extend(row["R_block"] for row in result["blocks"][label]["rows"] if row["R_block"] is not None)
        scales["block_R"][label] = [0.0, float(np.nanmax(values)) if values else 1.0]
    return scales


def plot_top_voxels_3d(
    path: Path,
    top: dict,
    values: np.ndarray,
    cbar_label: str,
    cmap_name: str,
    vmin: float,
    vmax: float,
    args: argparse.Namespace,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(9.8, 6.2))
    ax = fig.add_axes([0.02, 0.06, 0.78, 0.90], projection="3d")
    cmap = plt.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    origin_z = resolve_origin(args, read_optional_json(Path(args.roi_summary)))[2]
    z_display = top["z_top"] + origin_z
    z_min = float(args.ground_z)
    z_light_plane = origin_z + args.z_max
    z_max = max(z_light_plane, float(np.nanmax(z_display)) if len(z_display) else z_light_plane)
    ax.scatter(
        top["x"],
        top["y"],
        z_display,
        s=1.8,
        c=cmap(norm(values)),
        alpha=0.96,
        linewidths=0,
        depthshade=False,
        label="top voxels",
    )
    draw_display_domain_box(ax, args, z_min, z_max)
    draw_reference_plane(
        ax,
        args,
        z=args.ground_z,
        facecolor="#c6ad7c",
        edgecolor="#8a6f32",
        alpha=0.16,
        label=f"substrate surface z={args.ground_z:.2f} m",
    )
    draw_reference_plane(
        ax,
        args,
        z=z_light_plane,
        facecolor="#7fb3d5",
        edgecolor="#2b6f9b",
        alpha=0.12,
        label=f"simulated light plane z={z_light_plane:.3f} m",
    )
    ax.set_xlim(args.x_min, args.x_max)
    ax.set_ylim(args.y_min, args.y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("x local (m)", labelpad=18)
    ax.set_ylabel("y local (m)", labelpad=8)
    ax.set_zlabel("z (m)", labelpad=8)
    ax.set_zticks([0.46, 0.50, 0.55, 0.60, 0.65, 0.70])
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=3)
    ax.tick_params(axis="z", pad=3)
    ax.xaxis.labelpad = 18
    ax.yaxis.labelpad = 8
    ax.zaxis.labelpad = 8
    ax.view_init(elev=25, azim=-58)
    ax.legend(loc="upper right", bbox_to_anchor=(0.82, 0.99), markerscale=6, frameon=True)
    try:
        ax.set_box_aspect((args.x_max - args.x_min, args.y_max - args.y_min, z_max - z_min))
    except AttributeError:
        pass
    cax = fig.add_axes([0.90, 0.18, 0.022, 0.64])
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(cbar_label)
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)


def draw_display_domain_box(ax, args: argparse.Namespace, z0: float, z1: float) -> None:
    corners = np.asarray(
        [
            [args.x_min, args.y_min, z0],
            [args.x_max, args.y_min, z0],
            [args.x_max, args.y_max, z0],
            [args.x_min, args.y_max, z0],
            [args.x_min, args.y_min, z1],
            [args.x_max, args.y_min, z1],
            [args.x_max, args.y_max, z1],
            [args.x_min, args.y_max, z1],
        ],
        dtype=float,
    )
    for start, end in DOMAIN_BOX_EDGES:
        xs, ys, zs = zip(corners[start], corners[end])
        ax.plot(xs, ys, zs, color="black", linewidth=0.7, alpha=0.42)


def draw_reference_plane(
    ax,
    args: argparse.Namespace,
    z: float,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    label: str,
) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    verts = [
        [
            (args.x_min, args.y_min, z),
            (args.x_max, args.y_min, z),
            (args.x_max, args.y_max, z),
            (args.x_min, args.y_max, z),
        ]
    ]
    plane = Poly3DCollection(verts, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.8, alpha=alpha, label=label)
    ax.add_collection3d(plane)


def draw_domain_box(ax, args: argparse.Namespace) -> None:
    x0, x1 = args.x_min, args.x_max
    y0, y1 = args.y_min, args.y_max
    z0, z1 = args.z_min, args.z_max
    corners = np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=float,
    )
    for start, end in DOMAIN_BOX_EDGES:
        xs, ys, zs = zip(corners[start], corners[end])
        ax.plot(xs, ys, zs, color="black", linewidth=0.7, alpha=0.42)


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_ply(path: Path, points: np.ndarray, rgb: np.ndarray) -> None:
    vertex = np.empty(
        len(points),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    colors = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    vertex["x"] = points[:, 0].astype(np.float32)
    vertex["y"] = points[:, 1].astype(np.float32)
    vertex["z"] = points[:, 2].astype(np.float32)
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def colorize_values(values: np.ndarray) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=float)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if vmax <= vmin:
        normed = np.zeros_like(values)
    else:
        normed = (values - vmin) / (vmax - vmin)
    return plt.get_cmap("hot")(normed)[:, :3]


def format_setting(setting: float) -> str:
    return f"{setting:g}".replace(".", "p")


def block_size_label(block_size: float) -> str:
    cm = block_size * 100.0
    if abs(cm - round(cm)) < 1e-9:
        return f"{int(round(cm))}cm"
    return f"{cm:g}cm".replace(".", "p")


def safe_divide(value: float, total: float):
    return float(value / total) if total else None


if __name__ == "__main__":
    main()
