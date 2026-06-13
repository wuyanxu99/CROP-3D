#!/usr/bin/env python3
"""Prepare and inspect a canopy ROI before light interception analysis.

Stage 1 only: crop the point cloud into the light-field coordinate domain,
filter likely plant points, and export QC files. No light interception metrics
are computed here.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

try:
    from plyfile import PlyData, PlyElement
except ImportError as exc:  # pragma: no cover - import guard
    PlyData = None
    PlyElement = None
    PLY_IMPORT_ERROR = exc
else:
    PLY_IMPORT_ERROR = None


SH_C0 = 0.28209479177387814


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crop a light-field-sized canopy ROI and export plant-filter QC outputs."
    )
    parser.add_argument(
        "--input-ply",
        default=str(REPO_ROOT / "data" / "full_point_cloud_unavailable.ply"),
        help="Input 2DGS/3DGS PLY.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "outputs" / "light_interception" / "roi" / "1-1"),
        help="QC output directory.",
    )
    parser.add_argument("--origin-x", type=float, default=0.0, help="Light-field origin x in rack coordinates.")
    parser.add_argument("--origin-y", type=float, default=0.0, help="Light-field origin y in rack coordinates.")
    parser.add_argument("--origin-z", type=float, default=0.275, help="Light-field origin z in rack coordinates.")
    parser.add_argument(
        "--coord-mode",
        choices=("xyz", "pointcloud-y-up"),
        default="xyz",
        help=(
            "Coordinate mapping. xyz keeps input axes as light x/y/z. "
            "pointcloud-y-up maps old x->light x, old z->light y, old y->light z."
        ),
    )
    parser.add_argument(
        "--flip-light-y",
        action="store_true",
        help="For pointcloud-y-up mode, use light y = origin_z - old z instead of old z - origin_z.",
    )
    parser.add_argument("--x-min", type=float, default=0.0, help="Local light-field x min.")
    parser.add_argument("--x-max", type=float, default=1.5, help="Local light-field x max.")
    parser.add_argument("--y-min", type=float, default=0.0, help="Local light-field y min.")
    parser.add_argument("--y-max", type=float, default=0.56, help="Local light-field y max.")
    parser.add_argument("--z-min", type=float, default=0.0, help="Local light-field z min.")
    parser.add_argument("--z-max", type=float, default=0.45, help="Local light-field z max.")
    parser.add_argument("--z-filter-min", type=float, default=0.02, help="Minimum local z for plant candidates.")
    parser.add_argument("--green-min", type=float, default=0.18, help="Minimum green channel after RGB recovery.")
    parser.add_argument("--exg-min", type=float, default=0.05, help="Minimum excess-green value.")
    parser.add_argument(
        "--green-dominance-min",
        type=float,
        default=0.02,
        help="Minimum G - max(R,B) value.",
    )
    parser.add_argument("--saturation-min", type=float, default=0.08, help="Minimum RGB saturation.")
    parser.add_argument("--voxel-size", type=float, default=0.01, help="Voxel size for density filtering.")
    parser.add_argument("--min-voxel-points", type=int, default=3, help="Minimum points in a candidate voxel.")
    parser.add_argument(
        "--component-voxel-size",
        type=float,
        default=0.03,
        help="Voxel size for post-filter connected-component cleanup.",
    )
    parser.add_argument(
        "--min-component-points",
        type=int,
        default=80,
        help="Minimum points in a connected component kept after color/density filtering.",
    )
    parser.add_argument(
        "--min-component-voxels",
        type=int,
        default=3,
        help="Minimum voxels in a connected component kept after color/density filtering.",
    )
    parser.add_argument("--height-grid-size", type=float, default=0.01, help="Grid size for height map preview.")
    parser.add_argument("--height-percentile", type=float, default=95.0, help="Height percentile for preview map.")
    parser.add_argument("--max-plot-points", type=int, default=200_000, help="Maximum points drawn per projection layer.")
    parser.add_argument(
        "--save-removed",
        action="store_true",
        help="Also save removed_global.ply for debugging the plant filter.",
    )
    parser.add_argument(
        "--save-height-map",
        action="store_true",
        help="Also save a separate canopy_top_height_map.png preview.",
    )
    parser.add_argument(
        "--default-ply-frame",
        choices=("global", "local"),
        default="global",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if PLY_IMPORT_ERROR is not None:
        raise RuntimeError("plyfile is required to read/write PLY point clouds.") from PLY_IMPORT_ERROR

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib_cache"))

    rack_points, rgb, rgb_source, input_count, raw_bounds = read_point_cloud(Path(args.input_ply))
    light_points = to_light_coordinates(rack_points, args)

    roi_mask = (
        (light_points[:, 0] >= args.x_min)
        & (light_points[:, 0] <= args.x_max)
        & (light_points[:, 1] >= args.y_min)
        & (light_points[:, 1] <= args.y_max)
        & (light_points[:, 2] >= args.z_min)
        & (light_points[:, 2] <= args.z_max)
    )
    roi_points = light_points[roi_mask]
    roi_rgb = rgb[roi_mask]

    candidate_mask = plant_color_mask(roi_points, roi_rgb, args)
    dense_mask = density_mask(roi_points[candidate_mask], args.voxel_size, args.min_voxel_points)
    color_indices = np.flatnonzero(candidate_mask)
    dense_indices = color_indices[dense_mask]
    component_keep_mask = component_cleanup_mask(
        roi_points[dense_indices],
        args.component_voxel_size,
        args.min_component_points,
        args.min_component_voxels,
    )
    plant_mask = np.zeros(len(roi_points), dtype=bool)
    plant_mask[dense_indices[component_keep_mask]] = True

    plant_points = roi_points[plant_mask]
    plant_rgb = roi_rgb[plant_mask]
    removed_points = roi_points[~plant_mask]
    removed_rgb = roi_rgb[~plant_mask]

    roi_points_global = to_rack_coordinates(roi_points, args)
    plant_points_global = to_rack_coordinates(plant_points, args)
    removed_points_global = to_rack_coordinates(removed_points, args)

    write_ply(output_dir / "roi_global.ply", roi_points_global, roi_rgb)
    write_ply(output_dir / "plant_global.ply", plant_points_global, plant_rgb)
    write_ply(output_dir / "plant_local.ply", plant_points, plant_rgb)
    if args.save_removed:
        write_ply(output_dir / "removed_global.ply", removed_points_global, removed_rgb)

    plot_qc_3d(output_dir / "roi_qc_3d.png", plant_points, plant_rgb, args)
    if args.save_height_map:
        plot_height_map(output_dir / "canopy_top_height_map.png", plant_points, args)

    summary = build_summary(
        args=args,
        input_count=input_count,
        raw_bounds=raw_bounds,
        rgb_source=rgb_source,
        roi_points=roi_points,
        candidate_count=int(candidate_mask.sum()),
        dense_candidate_count=int(dense_mask.sum()),
        component_removed_count=int(len(dense_indices) - component_keep_mask.sum()),
        plant_points=plant_points,
        removed_points=removed_points,
        roi_points_global=roi_points_global,
        plant_points_global=plant_points_global,
        removed_points_global=removed_points_global,
    )
    write_json(output_dir / "roi_summary.json", summary)

    print("Prepared canopy ROI QC outputs")
    print(f"  input points: {input_count}")
    print(f"  ROI raw points: {len(roi_points)}")
    print(f"  green candidates: {int(candidate_mask.sum())}")
    print(f"  dense candidates: {int(dense_mask.sum())}")
    print(f"  component-removed candidates: {int(len(dense_indices) - component_keep_mask.sum())}")
    print(f"  plant filtered points: {len(plant_points)}")
    print(f"  removed points: {len(removed_points)}")
    print(f"  output: {output_dir}")


def read_point_cloud(path: Path):
    ply = PlyData.read(path)
    vertex = ply["vertex"].data
    names = vertex.dtype.names or ()
    points = np.column_stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ]
    )
    rgb, source = recover_rgb(vertex, names)
    return points, rgb, source, len(vertex), bounds_dict(points)


def recover_rgb(vertex, names: tuple[str, ...]):
    if {"red", "green", "blue"}.issubset(names):
        rgb = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.float32)
        if np.nanmax(rgb) > 1.0:
            rgb /= 255.0
        return np.clip(rgb, 0.0, 1.0), "rgb"

    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(names):
        sh = np.column_stack([vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]]).astype(np.float32)
        rgb = sh * SH_C0 + 0.5
        return np.clip(rgb, 0.0, 1.0), "sh_dc"

    rgb = np.full((len(vertex), 3), 0.65, dtype=np.float32)
    return rgb, "constant_gray"


def plant_color_mask(points: np.ndarray, rgb: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    exg = 2.0 * g - r - b
    dominance = g - np.maximum(r, b)
    rgb_max = np.maximum.reduce([r, g, b])
    rgb_min = np.minimum.reduce([r, g, b])
    saturation = (rgb_max - rgb_min) / np.maximum(rgb_max, 1e-6)
    return (
        (points[:, 2] >= args.z_filter_min)
        & (g >= args.green_min)
        & (g > r)
        & (g > b)
        & (exg >= args.exg_min)
        & (dominance >= args.green_dominance_min)
        & (saturation >= args.saturation_min)
    )


def density_mask(points: np.ndarray, voxel_size: float, min_voxel_points: int) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive.")
    if min_voxel_points <= 1:
        return np.ones(len(points), dtype=bool)
    voxels = np.floor(points / voxel_size).astype(np.int64)
    _, inverse, counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
    return counts[inverse] >= min_voxel_points


def component_cleanup_mask(
    points: np.ndarray,
    voxel_size: float,
    min_component_points: int,
    min_component_voxels: int,
) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    if voxel_size <= 0:
        raise ValueError("component_voxel_size must be positive.")
    if min_component_points <= 1 and min_component_voxels <= 1:
        return np.ones(len(points), dtype=bool)

    voxels = np.floor(points / voxel_size).astype(np.int64)
    unique_voxels, inverse, voxel_counts = np.unique(
        voxels,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    voxel_lookup = {tuple(v.tolist()): idx for idx, v in enumerate(unique_voxels)}
    visited = np.zeros(len(unique_voxels), dtype=bool)
    keep_voxels = np.zeros(len(unique_voxels), dtype=bool)

    for start_idx in range(len(unique_voxels)):
        if visited[start_idx]:
            continue
        queue: deque[int] = deque([start_idx])
        visited[start_idx] = True
        component: list[int] = []
        point_count = 0

        while queue:
            voxel_idx = queue.popleft()
            component.append(voxel_idx)
            point_count += int(voxel_counts[voxel_idx])
            center = unique_voxels[voxel_idx]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        neighbor = (int(center[0] + dx), int(center[1] + dy), int(center[2] + dz))
                        neighbor_idx = voxel_lookup.get(neighbor)
                        if neighbor_idx is not None and not visited[neighbor_idx]:
                            visited[neighbor_idx] = True
                            queue.append(neighbor_idx)

        if point_count >= min_component_points and len(component) >= min_component_voxels:
            keep_voxels[component] = True

    return keep_voxels[inverse]


def write_ply(path: Path, points: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertex = np.empty(
        len(points),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    if len(points):
        colors = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
        vertex["x"] = points[:, 0].astype(np.float32)
        vertex["y"] = points[:, 1].astype(np.float32)
        vertex["z"] = points[:, 2].astype(np.float32)
        vertex["red"] = colors[:, 0]
        vertex["green"] = colors[:, 1]
        vertex["blue"] = colors[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def plot_projection(
    path: Path,
    roi_points: np.ndarray,
    roi_rgb: np.ndarray,
    plant_points: np.ndarray,
    plant_rgb: np.ndarray,
    x_name: str,
    y_name: str,
    args: argparse.Namespace,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis = {"x": 0, "y": 1, "z": 2}
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    raw_idx = sample_indices(len(roi_points), args.max_plot_points)
    plant_idx = sample_indices(len(plant_points), args.max_plot_points)
    if len(raw_idx):
        ax.scatter(
            roi_points[raw_idx, axis[x_name]],
            roi_points[raw_idx, axis[y_name]],
            s=0.25,
            c=np.clip(roi_rgb[raw_idx], 0.0, 1.0),
            alpha=0.25,
            linewidths=0,
            label="ROI raw",
        )
    if len(plant_idx):
        ax.scatter(
            plant_points[plant_idx, axis[x_name]],
            plant_points[plant_idx, axis[y_name]],
            s=0.5,
            c=np.clip(plant_rgb[plant_idx], 0.0, 1.0),
            alpha=0.9,
            linewidths=0,
            label="plant filtered",
        )
    ax.set_xlabel(f"{x_name} (m)")
    ax.set_ylabel(f"{y_name} (m)")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(loc="upper right", markerscale=8, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_qc_3d(
    path: Path,
    plant_points: np.ndarray,
    plant_rgb: np.ndarray,
    args: argparse.Namespace,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.2, 6.2))
    ax = fig.add_subplot(111, projection="3d")

    plant_idx = sample_indices(len(plant_points), args.max_plot_points)
    if len(plant_idx):
        ax.scatter(
            plant_points[plant_idx, 0],
            plant_points[plant_idx, 1],
            plant_points[plant_idx, 2],
            s=0.6,
            c=np.clip(plant_rgb[plant_idx], 0.0, 1.0),
            alpha=0.92,
            linewidths=0,
            label="plant local",
            depthshade=False,
        )

    draw_roi_box(ax, args)
    ax.set_xlim(args.x_min, args.x_max)
    ax.set_ylim(args.y_min, args.y_max)
    ax.set_zlim(args.z_min, args.z_max)
    ax.set_xlabel("x local (m)")
    ax.set_ylabel("y local (m)")
    ax.set_zlabel("z local (m)")
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=3)
    ax.tick_params(axis="z", pad=3)
    ax.xaxis.labelpad = 9
    ax.yaxis.labelpad = 8
    ax.zaxis.labelpad = 8
    ax.view_init(elev=25, azim=-58)
    ax.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88), markerscale=8, frameon=True)
    try:
        ax.set_box_aspect((args.x_max - args.x_min, args.y_max - args.y_min, args.z_max - args.z_min))
    except AttributeError:
        pass
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.98)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def draw_roi_box(ax, args: argparse.Namespace) -> None:
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
    edges = [
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
    for start, end in edges:
        xs, ys, zs = zip(corners[start], corners[end])
        ax.plot(xs, ys, zs, color="black", linewidth=0.7, alpha=0.45)


def plot_height_map(path: Path, plant_points: np.ndarray, args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_edges = np.arange(args.x_min, args.x_max + args.height_grid_size, args.height_grid_size)
    y_edges = np.arange(args.y_min, args.y_max + args.height_grid_size, args.height_grid_size)
    height = np.full((len(y_edges) - 1, len(x_edges) - 1), np.nan, dtype=np.float32)
    if len(plant_points):
        xi = np.searchsorted(x_edges, plant_points[:, 0], side="right") - 1
        yi = np.searchsorted(y_edges, plant_points[:, 1], side="right") - 1
        valid = (xi >= 0) & (xi < height.shape[1]) & (yi >= 0) & (yi < height.shape[0])
        groups: dict[tuple[int, int], list[float]] = {}
        for x_cell, y_cell, z in zip(xi[valid], yi[valid], plant_points[valid, 2]):
            groups.setdefault((int(y_cell), int(x_cell)), []).append(float(z))
        for (y_cell, x_cell), z_values in groups.items():
            height[y_cell, x_cell] = np.percentile(z_values, args.height_percentile)

    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    im = ax.imshow(
        height,
        origin="lower",
        extent=[args.x_min, args.x_max, args.y_min, args.y_max],
        aspect="auto",
        cmap="viridis",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(f"z top Q{args.height_percentile:g} (m)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def sample_indices(n: int, max_count: int) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=np.int64)
    if n <= max_count:
        return np.arange(n, dtype=np.int64)
    step = int(np.ceil(n / max_count))
    return np.arange(0, n, step, dtype=np.int64)[:max_count]


def build_summary(
    args: argparse.Namespace,
    input_count: int,
    raw_bounds: dict,
    rgb_source: str,
    roi_points: np.ndarray,
    candidate_count: int,
    dense_candidate_count: int,
    component_removed_count: int,
    plant_points: np.ndarray,
    removed_points: np.ndarray,
    roi_points_global: np.ndarray,
    plant_points_global: np.ndarray,
    removed_points_global: np.ndarray,
) -> dict:
    roi_count = len(roi_points)
    plant_count = len(plant_points)
    removed_count = len(removed_points)
    return {
        "input_ply": args.input_ply,
        "input_point_count": input_count,
        "roi_raw_point_count": roi_count,
        "green_candidate_count": candidate_count,
        "dense_candidate_count": dense_candidate_count,
        "component_removed_candidate_count": component_removed_count,
        "plant_filtered_point_count": plant_count,
        "removed_point_count": removed_count,
        "filter_ratios": {
            "roi_over_input": safe_ratio(roi_count, input_count),
            "green_candidate_over_roi": safe_ratio(candidate_count, roi_count),
            "dense_candidate_over_roi": safe_ratio(dense_candidate_count, roi_count),
            "component_removed_over_dense_candidate": safe_ratio(component_removed_count, dense_candidate_count),
            "plant_filtered_over_roi": safe_ratio(plant_count, roi_count),
            "removed_over_roi": safe_ratio(removed_count, roi_count),
        },
        "raw_rack_coordinate_bounds": raw_bounds,
        "local_coordinate_bounds_roi": bounds_dict(roi_points),
        "local_coordinate_bounds_plant_filtered": bounds_dict(plant_points),
        "local_coordinate_bounds_removed": bounds_dict(removed_points),
        "global_coordinate_bounds_roi": bounds_dict(roi_points_global),
        "global_coordinate_bounds_plant_filtered": bounds_dict(plant_points_global),
        "global_coordinate_bounds_removed": bounds_dict(removed_points_global),
        "ply_outputs": {
            "global_overlay": ["roi_global.ply", "plant_global.ply"],
            "local_calculation": ["plant_local.ply"],
            "optional_debug": ["removed_global.ply"] if args.save_removed else [],
        },
        "figure_outputs": ["roi_qc_3d.png"] + (["canopy_top_height_map.png"] if args.save_height_map else []),
        "origin_offset_rack_to_light": {"x": args.origin_x, "y": args.origin_y, "z": args.origin_z},
        "coord_mode": args.coord_mode,
        "flip_light_y": args.flip_light_y,
        "domain_light_coordinates": {
            "x": [args.x_min, args.x_max],
            "y": [args.y_min, args.y_max],
            "z": [args.z_min, args.z_max],
        },
        "thresholds": {
            "z_filter_min": args.z_filter_min,
            "green_min": args.green_min,
            "exg_min": args.exg_min,
            "green_dominance_min": args.green_dominance_min,
            "saturation_min": args.saturation_min,
            "voxel_size": args.voxel_size,
            "min_voxel_points": args.min_voxel_points,
            "component_voxel_size": args.component_voxel_size,
            "min_component_points": args.min_component_points,
            "min_component_voxels": args.min_component_voxels,
        },
        "rgb_source": rgb_source,
    }


def bounds_dict(points: np.ndarray) -> dict:
    if len(points) == 0:
        return {"x": [None, None], "y": [None, None], "z": [None, None]}
    return {
        "x": [float(np.nanmin(points[:, 0])), float(np.nanmax(points[:, 0]))],
        "y": [float(np.nanmin(points[:, 1])), float(np.nanmax(points[:, 1]))],
        "z": [float(np.nanmin(points[:, 2])), float(np.nanmax(points[:, 2]))],
    }


def to_light_coordinates(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.coord_mode == "xyz":
        origin = np.asarray([args.origin_x, args.origin_y, args.origin_z], dtype=np.float32)
        return points - origin[None, :]
    if args.coord_mode == "pointcloud-y-up":
        light = np.empty_like(points, dtype=np.float32)
        light[:, 0] = points[:, 0] - args.origin_x
        if args.flip_light_y:
            light[:, 1] = args.origin_z - points[:, 2]
        else:
            light[:, 1] = points[:, 2] - args.origin_z
        light[:, 2] = points[:, 1] - args.origin_y
        return light
    raise ValueError(f"Unknown coord mode: {args.coord_mode}")


def to_rack_coordinates(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.coord_mode == "xyz":
        origin = np.asarray([args.origin_x, args.origin_y, args.origin_z], dtype=np.float32)
        return points + origin[None, :]
    if args.coord_mode == "pointcloud-y-up":
        rack = np.empty_like(points, dtype=np.float32)
        rack[:, 0] = points[:, 0] + args.origin_x
        if args.flip_light_y:
            rack[:, 2] = args.origin_z - points[:, 1]
        else:
            rack[:, 2] = points[:, 1] + args.origin_z
        rack[:, 1] = points[:, 2] + args.origin_y
        return rack
    raise ValueError(f"Unknown coord mode: {args.coord_mode}")


def safe_ratio(value: int, total: int):
    return float(value / total) if total else None


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
