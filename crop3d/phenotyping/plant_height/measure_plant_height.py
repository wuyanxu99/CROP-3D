#!/usr/bin/env python3
"""Measure plant height from a trained point cloud (adaptive_soft_growth + fixed ground).

Usage: edit CONFIG below, then run:
  python measure_plant_height.py
"""

from __future__ import annotations

import csv
from csv import Error as CsvError
import heapq
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np

try:
    from plyfile import PlyData, PlyElement
except ImportError as exc:
    PlyData = None
    PlyElement = None
    PLY_IMPORT_ERROR = exc
else:
    PLY_IMPORT_ERROR = None

SH_C0 = 0.28209479177387814
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]

# --- CONFIG ---
INPUT_PATH = str(REPO_ROOT / "examples" / "plant_height_demo" / "input" / "point_cloud.ply")
POSITIONS_FILE = str(REPO_ROOT / "examples" / "plant_height_demo" / "input" / "plant_height_GT_demo.csv")
FIXED_GROUND_Z = 0.46
EXPORT_Z_SCALE = 100.0
EXPORT_Z_DIGITS = 1
OUT_CSV = str(REPO_ROOT / "outputs" / "plant_height" / "heights.csv")
OUT_HEIGHTS_TXT = str(REPO_ROOT / "outputs" / "plant_height" / "ordered_heights.txt")
OUT_CANOPY_PLY = str(REPO_ROOT / "outputs" / "plant_height" / "canopy.ply")  # empty = skip
EXPORT_CANOPY_FRAME = "ply"
EXPORT_CANOPY_MODE = "volume"
EXPORT_CANOPY_SWAP_YZ = False
INPUT_X_SCALE = 1.0
INPUT_Y_SCALE = 1.0
INPUT_Z_SCALE = 1.0
SOIL_HEIGHT_SCALE = 1.0
GLOBAL_X_MIN: Optional[float] = None
GLOBAL_X_MAX: Optional[float] = None
GLOBAL_Y_MIN: Optional[float] = 0.03
GLOBAL_Y_MAX: Optional[float] = 0.28
GLOBAL_Z_MIN: Optional[float] = 0.445
GLOBAL_Z_MAX: Optional[float] = 0.76
USE_PLANT_ENVELOPE = True
GROUND_PEAK_WINDOW = 0.025
X_STRIP_HALF_WIDTH = 0.08
Y_SEARCH_HALF_WIDTH = 0.12
ENVELOPE_X_MARGIN = 0.005
ENVELOPE_Y_HALF_WIDTH = 0.22
ENVELOPE_EDGE_HALF_WIDTH = 0.10
SEARCH_ROI_X_EXPAND_MAX = 0.06
SEARCH_ROI_X_EXPAND_SPACING_FRAC = 0.40
CLUSTER_CELL_SIZE = 0.01
DENSE_CELL_MIN_POINTS = 12
DENSE_NEIGHBOR_MIN_POINTS = 18
SURFACE_CELL_SIZE = 0.008
SURFACE_CELL_MIN_POINTS = 3
SURFACE_COMPONENT_MIN_CELLS = 10
SURFACE_MIN_POINTS = 24
SURFACE_Z_PERCENTILE = 99.0
SURFACE_FOOTPRINT_DILATION = 0
CANOPY_OUTLIER_K = 6
CANOPY_OUTLIER_SIGMA = 3.0
CANOPY_OUTLIER_MIN_POINTS = 16
MIN_CANOPY_POINTS = 120
TOP_K = 10
MIN_CANOPY_CLEARANCE = 0.01
MAX_CANOPY_HEIGHT = 0.30
VOXEL_SIZE_3D = 0.01
MIN_VOXEL_POINTS_3D = 2
COMPONENT_VOXEL_SIZE = 0.03
MIN_COMPONENT_POINTS = 80
MIN_COMPONENT_VOXELS = 3
GREEN_MIN = 0.18
GREEN_EXG_MIN = 0.05
GREEN_DOMINANCE_MIN = 0.02
GREEN_SATURATION_MIN = 0.08
ADAPTIVE_GREEN_ENABLED = True
ADAPTIVE_GREEN_MIN_SEEDS = 30
ADAPTIVE_GREEN_MIN_G = 0.07
ADAPTIVE_GREEN_MIN_SATURATION = 0.055
ADAPTIVE_GREEN_NEXG_FLOOR = 0.015
ADAPTIVE_GREEN_G_RATIO_FLOOR = 0.30
ADAPTIVE_GREEN_QUANTILE = 0.10
ADAPTIVE_GREEN_NEXG_MARGIN = 0.035
ADAPTIVE_GREEN_G_RATIO_MARGIN = 0.035
ADAPTIVE_GREEN_MAX_THRESHOLDS = True
BODY_SEED_X_HALF_WIDTH = 0.075
BODY_SEED_Y_HALF_WIDTH = 0.11
BODY_SEED_Z_MIN_OFFSET = 0.015
BODY_SEED_Z_MAX_OFFSET = 0.16
MIN_BODY_SEED_POINTS = 30
GROWTH_VOXEL_SIZE = 0.015
GROWTH_NEIGHBOR_RADIUS = 1
GROWTH_MAX_COST = 5.0
GROWTH_STEP_COST = 1.0
GROWTH_OUTSIDE_CORE_COST = 0.55
GROWTH_OUTSIDE_SEARCH_COST = 2.0
GROWTH_HEIGHT_COST = 0.18
AMBIGUOUS_COST_MARGIN = 0.75
POST_GROWTH_COMPONENT_VOXEL_SIZE = 0.018
POST_GROWTH_MIN_COMPONENT_POINTS = 35
POST_GROWTH_MAX_ANCHOR_DIST = 0.12
POST_GROWTH_MAX_BODY_DIST = 0.09
POST_GROWTH_ANCHOR_Z_MAX_OFFSET = 0.08
POST_GROWTH_MAX_COMPONENT_GAP = 0.04
OPACITY_MIN = 0.05
SCALE_QUANTILE_MAX = 0.99


@dataclass
class PlantTarget:
    plant_id: str
    x: float
    y: Optional[float] = None
    z: Optional[float] = None
    soil_height: Optional[float] = None
    source_index: Optional[int] = None
    gt_highest_cm: Optional[float] = None
    gt_canopy_cm: Optional[float] = None
    envelope_x_min: Optional[float] = None
    envelope_x_max: Optional[float] = None
    envelope_y_min: Optional[float] = None
    envelope_y_max: Optional[float] = None


@dataclass
class CloudData:
    points: np.ndarray
    rgb: np.ndarray
    opacity: Optional[np.ndarray]
    max_scale: Optional[np.ndarray]
    source_kind: str
    source_path: Path


@dataclass
class PlantResult:
    row: Dict[str, object]
    canopy_xyz: Optional[np.ndarray] = None
    canopy_rgb: Optional[np.ndarray] = None


def load_targets(path: Path) -> List[PlantTarget]:
    if not path.exists():
        raise FileNotFoundError(f"Positions file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return load_targets_from_xlsx(path)
    if suffix == ".json":
        raise ValueError("Use CSV or XLSX for plant positions.")
    return load_targets_from_csv(path)



def load_targets_from_csv(path: Path) -> List[PlantTarget]:
    sample = path.read_text(encoding="utf-8").splitlines()
    if not sample:
        raise ValueError(f"Empty positions file: {path}")
    try:
        dialect = csv.Sniffer().sniff("\n".join(sample[:5]), delimiters=",\t;")
    except CsvError:
        dialect = csv.get_dialect("excel")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, dialect=dialect)
        if reader.fieldnames is None or "x" not in reader.fieldnames:
            raise ValueError("CSV header must include x; optional id, y, z, soil_height, gt_highest_cm.")
        targets = []
        for idx, row in enumerate(reader, start=1):
            plant_id = str(row.get("id") or row.get("plant_id") or f"plant_{idx:02d}")
            y_val = row.get("y")
            y = None if y_val in (None, "") else float(y_val) * INPUT_Y_SCALE
            z_val = row.get("z")
            z = None if z_val in (None, "") else float(z_val) * INPUT_Z_SCALE
            soil_val = row.get("soil_height")
            soil = None if soil_val in (None, "") else float(soil_val) * SOIL_HEIGHT_SCALE
            targets.append(
                PlantTarget(
                    plant_id=plant_id,
                    x=float(row["x"]) * INPUT_X_SCALE,
                    y=y,
                    z=z,
                    soil_height=soil,
                    source_index=idx,
                    gt_highest_cm=first_optional_float(row, ("gt_highest_cm", "highest_cm", "height_cm")),
                    gt_canopy_cm=first_optional_float(row, ("gt_canopy_cm", "canopy_cm")),
                )
            )
    return targets


def load_targets_from_xlsx(path: Path) -> List[PlantTarget]:
    rows = read_first_xlsx_sheet(path)
    if not rows:
        raise ValueError(f"Excel sheet is empty: {path}")

    header_raw = rows[0]
    header = [normalize_header(v) for v in header_raw]
    targets: List[PlantTarget] = []

    col_num = find_header_index(header, ("num", "index", "source_index"))
    col_id = find_header_index(header, ("id", "plant_id", "name"))
    if col_id is None and len(header_raw) > 1 and normalize_header(header_raw[1]) == "":
        col_id = 1
    col_highest = find_header_index(header, ("gt_highest_cm", "highest_cm", "height_cm"))
    col_canopy = find_header_index(header, ("gt_canopy_cm", "canopy_cm"))
    col_x = find_header_index(header, ("x", "input_x", "anchor_x"))
    col_y = find_header_index(header, ("y", "input_y", "anchor_y"))
    col_z = find_header_index(header, ("z", "input_z", "anchor_z"))
    col_soil = find_header_index(header, ("soil_height", "ground_z"))

    if col_x is None or col_y is None:
        raise ValueError("Excel Sheet1 must include x and y columns.")

    for row_idx, row in enumerate(rows[1:], start=1):
        x_raw = get_cell(row, col_x)
        y_raw = get_cell(row, col_y)
        if is_blank(x_raw) or is_blank(y_raw):
            continue
        source_index = parse_optional_int(get_cell(row, col_num)) or row_idx
        plant_id = str(get_cell(row, col_id) or f"plant_{source_index:02d}")
        z_value = parse_optional_float(get_cell(row, col_z))
        soil_value = parse_optional_float(get_cell(row, col_soil))
        targets.append(
            PlantTarget(
                plant_id=plant_id,
                x=float(x_raw) * INPUT_X_SCALE,
                y=float(y_raw) * INPUT_Y_SCALE,
                z=None if z_value is None else float(z_value) * INPUT_Z_SCALE,
                soil_height=None if soil_value is None else float(soil_value) * SOIL_HEIGHT_SCALE,
                source_index=source_index,
                gt_highest_cm=parse_optional_float(get_cell(row, col_highest)),
                gt_canopy_cm=parse_optional_float(get_cell(row, col_canopy)),
            )
        )
    if not targets:
        raise ValueError(f"No usable plant rows in Excel: {path}")
    return targets


def read_first_xlsx_sheet(path: Path) -> List[List[str]]:
    ns = {
        "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with ZipFile(path) as zf:
        names = set(zf.namelist())
        shared = read_xlsx_shared_strings(zf, ns) if "xl/sharedStrings.xml" in names else []
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        sheet = workbook.find(".//a:sheet", ns)
        if sheet is None:
            return []
        rel_id = sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        rel_root = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rel_root.findall("rel:Relationship", ns)}
        sheet_target = rel_map.get(rel_id or "")
        if not sheet_target:
            raise ValueError(f"Sheet1 not found in Excel: {path}")
        sheet_path = sheet_target.lstrip("/")
        if not sheet_path.startswith("xl/"):
            sheet_path = "xl/" + sheet_path
        root = ET.fromstring(zf.read(sheet_path))

    parsed_rows: List[List[str]] = []
    for row in root.findall(".//a:sheetData/a:row", ns):
        values: Dict[int, str] = {}
        max_col = -1
        for cell in row.findall("a:c", ns):
            ref = cell.attrib.get("r", "")
            col_idx = column_index_from_ref(ref)
            if col_idx is None:
                continue
            values[col_idx] = read_xlsx_cell_text(cell, shared, ns)
            max_col = max(max_col, col_idx)
        if max_col >= 0:
            parsed_rows.append([values.get(i, "") for i in range(max_col + 1)])
    return parsed_rows


def read_xlsx_shared_strings(zf: ZipFile, ns: Dict[str, str]) -> List[str]:
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    out = []
    for item in root.findall("a:si", ns):
        out.append("".join(t.text or "" for t in item.findall(".//a:t", ns)))
    return out


def read_xlsx_cell_text(cell: ET.Element, shared: Sequence[str], ns: Dict[str, str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(t.text or "" for t in cell.findall(".//a:t", ns)).strip()
    value = cell.find("a:v", ns)
    if value is None or value.text is None:
        return ""
    text = value.text.strip()
    if cell_type == "s":
        idx = int(text)
        return shared[idx] if 0 <= idx < len(shared) else ""
    return text


def column_index_from_ref(ref: str) -> Optional[int]:
    letters = "".join(ch for ch in ref if ch.isalpha())
    if not letters:
        return None
    out = 0
    for ch in letters.upper():
        out = out * 26 + (ord(ch) - ord("A") + 1)
    return out - 1


def normalize_header(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "").replace("_", "")


def find_header_index(header: Sequence[str], candidates: Sequence[str]) -> Optional[int]:
    normalized = {normalize_header(c) for c in candidates}
    for idx, value in enumerate(header):
        if value in normalized:
            return idx
    return None


def get_cell(row: Sequence[str], idx: Optional[int]) -> str:
    if idx is None or idx < 0 or idx >= len(row):
        return ""
    return str(row[idx]).strip()


def is_blank(value: object) -> bool:
    return value is None or str(value).strip() == ""


def parse_optional_float(value: object) -> Optional[float]:
    if is_blank(value):
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def parse_optional_int(value: object) -> Optional[int]:
    parsed = parse_optional_float(value)
    if parsed is None:
        return None
    return int(parsed)


def first_optional_float(row: Dict[str, str], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        if key in row:
            value = parse_optional_float(row.get(key))
            if value is not None:
                return value
    return None


def load_cloud(path: Path) -> CloudData:
    if PlyData is None:
        raise ImportError(
            "plyfile is required to read PLY. Install requirements-da3.txt in the project environment."
        ) from PLY_IMPORT_ERROR

    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        raise ValueError(f"PLY has no vertex element: {path}")

    vertex = ply["vertex"]
    names = [prop.name for prop in vertex.properties]

    for key in ("x", "y", "z"):
        if key not in names:
            raise ValueError(f"PLY vertex missing coordinate field {key}: {path}")
    points = np.column_stack([np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])]).astype(np.float32)

    rgb = load_rgb_from_vertex(vertex, names)

    opacity = None
    if "opacity" in names:
        opacity = sigmoid(np.asarray(vertex["opacity"]).astype(np.float32))

    max_scale = None
    scale_names = sorted([n for n in names if n.startswith("scale_")], key=lambda n: int(n.split("_")[-1]))
    if scale_names:
        raw_scale = np.column_stack([np.asarray(vertex[name]) for name in scale_names]).astype(np.float32)
        max_scale = np.exp(raw_scale.max(axis=1))

    source_kind = "gaussian_point_cloud" if "f_dc_0" in names else "rgb_ply"
    return CloudData(
        points=points,
        rgb=rgb,
        opacity=opacity,
        max_scale=max_scale,
        source_kind=source_kind,
        source_path=path,
    )


def load_rgb_from_vertex(vertex, names: Sequence[str]) -> np.ndarray:
    if {"red", "green", "blue"}.issubset(set(names)):
        rgb = np.column_stack(
            [
                np.asarray(vertex["red"]),
                np.asarray(vertex["green"]),
                np.asarray(vertex["blue"]),
            ]
        ).astype(np.float32)
        return np.clip(rgb / 255.0, 0.0, 1.0)

    if {"f_dc_0", "f_dc_1", "f_dc_2"}.issubset(set(names)):
        sh0 = np.column_stack(
            [
                np.asarray(vertex["f_dc_0"]),
                np.asarray(vertex["f_dc_1"]),
                np.asarray(vertex["f_dc_2"]),
            ]
        ).astype(np.float32)
        return np.clip(sh0 * SH_C0 + 0.5, 0.0, 1.0)

    raise ValueError("PLY has neither RGB nor Gaussian f_dc_0~2; cannot segment green.")


def filter_cloud(cloud: CloudData) -> CloudData:
    mask = np.all(np.isfinite(cloud.points), axis=1) & np.all(np.isfinite(cloud.rgb), axis=1)
    if cloud.opacity is not None:
        mask &= np.isfinite(cloud.opacity) & (cloud.opacity >= OPACITY_MIN)
    if cloud.max_scale is not None:
        valid_scale = np.isfinite(cloud.max_scale)
        mask &= valid_scale
        if valid_scale.any() and 0.0 < SCALE_QUANTILE_MAX < 1.0:
            cap = float(np.quantile(cloud.max_scale[valid_scale], SCALE_QUANTILE_MAX))
            mask &= cloud.max_scale <= cap

    return CloudData(
        points=cloud.points[mask],
        rgb=cloud.rgb[mask],
        opacity=cloud.opacity[mask] if cloud.opacity is not None else None,
        max_scale=cloud.max_scale[mask] if cloud.max_scale is not None else None,
        source_kind=cloud.source_kind,
        source_path=cloud.source_path,
    )


def crop_cloud_global_bounds(cloud: CloudData) -> CloudData:
    mask = np.ones(cloud.points.shape[0], dtype=bool)
    if GLOBAL_X_MIN is not None:
        mask &= cloud.points[:, 0] >= float(GLOBAL_X_MIN)
    if GLOBAL_X_MAX is not None:
        mask &= cloud.points[:, 0] <= float(GLOBAL_X_MAX)
    if GLOBAL_Y_MIN is not None:
        mask &= cloud.points[:, 1] >= float(GLOBAL_Y_MIN)
    if GLOBAL_Y_MAX is not None:
        mask &= cloud.points[:, 1] <= float(GLOBAL_Y_MAX)
    if GLOBAL_Z_MIN is not None:
        mask &= cloud.points[:, 2] >= float(GLOBAL_Z_MIN)
    if GLOBAL_Z_MAX is not None:
        mask &= cloud.points[:, 2] <= float(GLOBAL_Z_MAX)
    if mask.all():
        return cloud
    return CloudData(
        points=cloud.points[mask],
        rgb=cloud.rgb[mask],
        opacity=cloud.opacity[mask] if cloud.opacity is not None else None,
        max_scale=cloud.max_scale[mask] if cloud.max_scale is not None else None,
        source_kind=cloud.source_kind,
        source_path=cloud.source_path,
    )


def compute_green_seed_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    exg = 2.0 * g - r - b
    vmax = rgb.max(axis=1)
    vmin = rgb.min(axis=1)
    saturation = np.divide(vmax - vmin, vmax + 1e-8)
    return (
        (g >= GREEN_MIN)
        & ((g - r) >= GREEN_DOMINANCE_MIN)
        & ((g - b) >= GREEN_DOMINANCE_MIN)
        & (exg >= GREEN_EXG_MIN)
        & (saturation >= GREEN_SATURATION_MIN)
    )



def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def compute_color_features(rgb: np.ndarray) -> Dict[str, np.ndarray]:
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    sum_rgb = r + g + b + 1e-8
    exg = 2.0 * g - r - b
    vmax = rgb.max(axis=1)
    vmin = rgb.min(axis=1)
    return {
        "g": g,
        "nexg": exg / sum_rgb,
        "g_ratio": g / sum_rgb,
        "saturation": np.divide(vmax - vmin, vmax + 1e-8),
    }
def compute_adaptive_green_mask(
    rgb: np.ndarray,
    strict_seed_mask: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    if rgb.shape[0] == 0:
        return np.zeros(0, dtype=bool), {}
    features = compute_color_features(rgb)
    q = min(0.5, max(0.0, float(ADAPTIVE_GREEN_QUANTILE)))

    if ADAPTIVE_GREEN_ENABLED and int(strict_seed_mask.sum()) >= int(ADAPTIVE_GREEN_MIN_SEEDS):
        seed_nexg = features["nexg"][strict_seed_mask]
        seed_g_ratio = features["g_ratio"][strict_seed_mask]
        seed_sat = features["saturation"][strict_seed_mask]
        seed_g = features["g"][strict_seed_mask]
        nexg_from_seed = float(np.quantile(seed_nexg, q) - float(ADAPTIVE_GREEN_NEXG_MARGIN))
        g_ratio_from_seed = float(np.quantile(seed_g_ratio, q) - float(ADAPTIVE_GREEN_G_RATIO_MARGIN))
        sat_from_seed = float(np.quantile(seed_sat, q) * 0.55)
        g_from_seed = float(np.quantile(seed_g, q) * 0.45)
        if ADAPTIVE_GREEN_MAX_THRESHOLDS:
            nexg_thr = max(float(ADAPTIVE_GREEN_NEXG_FLOOR), nexg_from_seed)
            g_ratio_thr = max(float(ADAPTIVE_GREEN_G_RATIO_FLOOR), g_ratio_from_seed)
            sat_thr = max(float(ADAPTIVE_GREEN_MIN_SATURATION), sat_from_seed)
            g_thr = max(float(ADAPTIVE_GREEN_MIN_G), g_from_seed)
        else:
            nexg_thr = min(float(ADAPTIVE_GREEN_NEXG_FLOOR), nexg_from_seed)
            g_ratio_thr = min(float(ADAPTIVE_GREEN_G_RATIO_FLOOR), g_ratio_from_seed)
            sat_thr = min(float(ADAPTIVE_GREEN_MIN_SATURATION), sat_from_seed)
            g_thr = min(float(ADAPTIVE_GREEN_MIN_G), g_from_seed)
        source = "seed_calibrated"
    else:
        nexg_thr = float(ADAPTIVE_GREEN_NEXG_FLOOR)
        g_ratio_thr = float(ADAPTIVE_GREEN_G_RATIO_FLOOR)
        sat_thr = float(ADAPTIVE_GREEN_MIN_SATURATION)
        g_thr = float(ADAPTIVE_GREEN_MIN_G)
        source = "fallback_floor"

    adaptive = (
        (features["g"] >= g_thr)
        & (features["saturation"] >= sat_thr)
        & (features["nexg"] >= nexg_thr)
        & (features["g_ratio"] >= g_ratio_thr)
    )
    thresholds = {
        "g_min": float(g_thr),
        "saturation_min": float(sat_thr),
        "nexg_min": float(nexg_thr),
        "g_ratio_min": float(g_ratio_thr),
        "strict_seed_count": float(strict_seed_mask.sum()),
        "source": source,
    }
    return adaptive | strict_seed_mask, thresholds


def assign_target_envelopes(targets: Sequence[PlantTarget]) -> None:
    if not USE_PLANT_ENVELOPE or not targets:
        return
    order = sorted(range(len(targets)), key=lambda i: (float(targets[i].x), float(targets[i].y or 0.0)))
    margin = max(0.0, float(ENVELOPE_X_MARGIN))
    edge_half = max(float(X_STRIP_HALF_WIDTH), float(ENVELOPE_EDGE_HALF_WIDTH))
    for pos, idx in enumerate(order):
        target = targets[idx]
        if pos == 0:
            x_min = float(target.x) - edge_half
        else:
            left = targets[order[pos - 1]]
            x_min = (float(left.x) + float(target.x)) * 0.5 + margin
        if pos == len(order) - 1:
            x_max = float(target.x) + edge_half
        else:
            right = targets[order[pos + 1]]
            x_max = (float(target.x) + float(right.x)) * 0.5 - margin
        if x_max <= x_min:
            half = max(float(X_STRIP_HALF_WIDTH), 1e-3)
            x_min = float(target.x) - half
            x_max = float(target.x) + half

        y_center = float(target.y) if target.y is not None else 0.0
        y_half = max(float(Y_SEARCH_HALF_WIDTH), float(ENVELOPE_Y_HALF_WIDTH))
        target.envelope_x_min = x_min
        target.envelope_x_max = x_max
        target.envelope_y_min = y_center - y_half
        target.envelope_y_max = y_center + y_half


def resolve_input_ply(input_path: Path) -> Path:
    if input_path.is_file():
        if input_path.suffix.lower() != ".ply":
            raise FileNotFoundError(f"Input file is not a PLY: {input_path}")
        return input_path
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    candidates = sorted(
        input_path.glob("point_cloud/iteration_*/point_cloud.ply"),
        key=lambda p: int(re.search(r"(\d+)", p.parent.name).group(1)) if re.search(r"(\d+)", p.parent.name) else 0,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    if (input_path / "input.ply").exists():
        return input_path / "input.ply"
    raise FileNotFoundError(f"No point_cloud.ply under {input_path}")


def topk_median_z(z_values: np.ndarray, top_k: int) -> float:
    if z_values.size == 0:
        return float("nan")
    k = max(1, min(int(top_k), int(z_values.size)))
    return float(np.median(np.sort(z_values)[-k:]))


def measure_single_plant(
    cloud: CloudData,
    green_mask_global: np.ndarray,
    target: PlantTarget,
) -> PlantResult:
    pts = cloud.points
    if target.envelope_x_min is None or target.envelope_x_max is None:
        raise ValueError(f"Missing envelope for plant {target.plant_id}")

    core_x_min = float(target.envelope_x_min)
    core_x_max = float(target.envelope_x_max)
    core_y_min = float(target.envelope_y_min)
    core_y_max = float(target.envelope_y_max)
    spacing_hint = max(float(target.x) - core_x_min, core_x_max - float(target.x), 1e-6)
    x_expand = min(max(0.0, SEARCH_ROI_X_EXPAND_MAX), SEARCH_ROI_X_EXPAND_SPACING_FRAC * spacing_hint)
    strip_mask = (
        (pts[:, 0] >= core_x_min - x_expand)
        & (pts[:, 0] <= core_x_max + x_expand)
        & (pts[:, 1] >= core_y_min)
        & (pts[:, 1] <= core_y_max)
    )
    strip_points = pts[strip_mask]
    strip_rgb = cloud.rgb[strip_mask]
    strip_green_mask = green_mask_global[strip_mask]

    row: Dict[str, object] = {
        "source_index": target.source_index or "",
        "plant_id": target.plant_id,
        "input_x": float(target.x),
        "input_y": float(target.y) if target.y is not None else "",
        "gt_highest_cm": target.gt_highest_cm if target.gt_highest_cm is not None else "",
        "status": "",
        "note": "",
        "ground_z": "",
        "top_z": "",
        "pred_height_cm": "",
        "error_cm": "",
        "abs_error_cm": "",
    }
    if strip_points.shape[0] == 0:
        row["status"] = "failed_empty_strip"
        row["note"] = "No points in x/y neighborhood."
        return PlantResult(row, None, None)

    if FIXED_GROUND_Z is not None:
        ground_z = float(FIXED_GROUND_Z)
        soil_method = "fixed_ground_z"
    elif target.soil_height is not None:
        ground_z = float(target.soil_height)
        soil_method = "manual_soil_height"
    else:
        raise ValueError("Set FIXED_GROUND_Z in CONFIG or provide soil_height per plant.")
    row["ground_z"] = ground_z

    search_start_z = max(
        float(target.z) if target.z is not None else -float("inf"),
        ground_z + float(MIN_CANOPY_CLEARANCE),
    )
    search_end_z = ground_z + float(MAX_CANOPY_HEIGHT) if MAX_CANOPY_HEIGHT > 0 else float("inf")

    component_points, component_rgb, component_mode = segment_adaptive_soft_growth(
        strip_points=strip_points,
        strip_rgb=strip_rgb,
        strict_seed_mask=strip_green_mask,
        target=target,
        core_bounds=(core_x_min, core_x_max, core_y_min, core_y_max),
        search_bounds=(core_x_min - x_expand, core_x_max + x_expand, core_y_min, core_y_max),
        search_start_z=search_start_z,
        search_end_z=search_end_z,
    )
    if component_points.shape[0] < MIN_CANOPY_POINTS:
        row["status"] = "failed_not_enough_component"
        row["note"] = "Too few points assigned to plant."
        return PlantResult(row, None, None)

    surface_points, surface_rgb, surface_cells = build_top_surface_xy(
        component_points,
        component_rgb,
        SURFACE_CELL_SIZE,
        SURFACE_CELL_MIN_POINTS,
        SURFACE_Z_PERCENTILE,
    )
    if surface_points.shape[0] < SURFACE_COMPONENT_MIN_CELLS:
        row["status"] = "failed_not_enough_canopy"
        row["note"] = "Too few canopy surface cells."
        return PlantResult(row, None, None)

    canopy_surface_points = surface_points
    canopy_surface_rgb = surface_rgb
    canopy_surface_cells = surface_cells
    if canopy_surface_points.shape[0] < SURFACE_MIN_POINTS:
        row["status"] = "failed_not_enough_canopy"
        row["note"] = "Too few canopy points."
        return PlantResult(row, None, None)

    surface_keep = filter_isolated_points_knn(
        canopy_surface_points,
        k=CANOPY_OUTLIER_K,
        sigma=CANOPY_OUTLIER_SIGMA,
        min_points=CANOPY_OUTLIER_MIN_POINTS,
        dims=(0, 1, 2),
    )
    if surface_keep.any():
        canopy_surface_points = canopy_surface_points[surface_keep]
        canopy_surface_rgb = canopy_surface_rgb[surface_keep]
        canopy_surface_cells = canopy_surface_cells[surface_keep]
    if canopy_surface_points.shape[0] < SURFACE_MIN_POINTS:
        row["status"] = "failed_not_enough_canopy"
        row["note"] = "Too few canopy points after outlier removal."
        return PlantResult(row, None, None)

    top_z = topk_median_z(canopy_surface_points[:, 2], TOP_K)
    row["top_z"] = top_z

    if EXPORT_CANOPY_MODE == "volume":
        canopy_points, canopy_rgb = select_points_in_cells_xy(
            component_points,
            component_rgb,
            SURFACE_CELL_SIZE,
            canopy_surface_cells,
            SURFACE_FOOTPRINT_DILATION,
        )
        if canopy_points.shape[0] == 0:
            canopy_points = canopy_surface_points
            canopy_rgb = canopy_surface_rgb
        else:
            canopy_keep = filter_isolated_points_knn(
                canopy_points,
                k=CANOPY_OUTLIER_K,
                sigma=CANOPY_OUTLIER_SIGMA,
                min_points=CANOPY_OUTLIER_MIN_POINTS,
                dims=(0, 1, 2),
            )
            if canopy_keep.any():
                canopy_points = canopy_points[canopy_keep]
                canopy_rgb = canopy_rgb[canopy_keep]
    else:
        canopy_points = canopy_surface_points
        canopy_rgb = canopy_surface_rgb

    row["status"] = "ok"
    row["note"] = ",".join([m for m in (component_mode, soil_method) if m])
    return PlantResult(
        row,
        np.asarray(canopy_points, dtype=np.float32),
        np.asarray(canopy_rgb, dtype=np.float32),
    )



def segment_adaptive_soft_growth(
    strip_points: np.ndarray,
    strip_rgb: np.ndarray,
    strict_seed_mask: np.ndarray,
    target: PlantTarget,
    core_bounds: Tuple[float, float, float, float],
    search_bounds: Tuple[float, float, float, float],
    search_start_z: float,
    search_end_z: float,
) -> Tuple[np.ndarray, np.ndarray, str]:
    height_mask = (strip_points[:, 2] >= search_start_z) & (strip_points[:, 2] <= search_end_z)
    strict_height_mask = strict_seed_mask & height_mask
    adaptive_mask, _ = compute_adaptive_green_mask(strip_rgb, strict_height_mask)
    adaptive_mask &= height_mask

    adaptive_points = strip_points[adaptive_mask]
    adaptive_rgb = strip_rgb[adaptive_mask]
    if adaptive_points.shape[0] < MIN_CANOPY_POINTS:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), "adaptive_not_enough_green"

    dense_3d_mask = density_mask_3d(adaptive_points, VOXEL_SIZE_3D, MIN_VOXEL_POINTS_3D)
    adaptive_points = adaptive_points[dense_3d_mask]
    adaptive_rgb = adaptive_rgb[dense_3d_mask]

    dense_mask = filter_sparse_points_xy(
        adaptive_points[:, :2],
        CLUSTER_CELL_SIZE,
        DENSE_CELL_MIN_POINTS,
        DENSE_NEIGHBOR_MIN_POINTS,
    )
    adaptive_points = adaptive_points[dense_mask]
    adaptive_rgb = adaptive_rgb[dense_mask]
    if adaptive_points.shape[0] < MIN_CANOPY_POINTS:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), "adaptive_not_enough_dense_green"

    strict_on_adaptive = compute_green_seed_mask(adaptive_rgb)
    body_seed_mask = select_body_seed_mask(
        adaptive_points,
        strict_on_adaptive,
        target,
        core_bounds,
        search_start_z,
    )
    if int(body_seed_mask.sum()) < int(MIN_BODY_SEED_POINTS):
        body_seed_mask = fallback_body_seed_from_components(adaptive_points, target)
    if not body_seed_mask.any():
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32), "adaptive_no_body_seed"

    body_seed_points = adaptive_points[body_seed_mask]
    grown_mask, ambiguous_mask, n_voxels = grow_component_from_body_seed(
        adaptive_points,
        body_seed_mask,
        target,
        core_bounds,
        search_bounds,
    )
    grown_points = adaptive_points[grown_mask]
    grown_rgb = adaptive_rgb[grown_mask]
    if grown_points.shape[0] > 0:
        component_keep, detached_points = filter_detached_growth_components(
            grown_points,
            body_seed_points,
            target,
            search_start_z,
        )
        if detached_points.shape[0] > 0:
            grown_mask_indices = np.flatnonzero(grown_mask)
            detached_indices = grown_mask_indices[~component_keep]
            grown_mask[detached_indices] = False
            ambiguous_mask[detached_indices] = True
            grown_points = adaptive_points[grown_mask]
            grown_rgb = adaptive_rgb[grown_mask]
    return grown_points, grown_rgb, f"adaptive_soft_growth_voxels={n_voxels}"


def select_body_seed_mask(
    points: np.ndarray,
    strict_mask: np.ndarray,
    target: PlantTarget,
    core_bounds: Tuple[float, float, float, float],
    search_start_z: float,
) -> np.ndarray:
    core_x_min, core_x_max, core_y_min, core_y_max = core_bounds
    x_half = min(float(BODY_SEED_X_HALF_WIDTH), max(float(target.x) - core_x_min, core_x_max - float(target.x)))
    x_half = max(x_half, 0.025)
    y_center = float(target.y) if target.y is not None else (core_y_min + core_y_max) * 0.5
    anchor_mask = (
        (np.abs(points[:, 0] - float(target.x)) <= x_half)
        & (np.abs(points[:, 1] - y_center) <= float(BODY_SEED_Y_HALF_WIDTH))
        & (points[:, 2] >= search_start_z + float(BODY_SEED_Z_MIN_OFFSET) - float(MIN_CANOPY_CLEARANCE))
        & (points[:, 2] <= search_start_z + float(BODY_SEED_Z_MAX_OFFSET) - float(MIN_CANOPY_CLEARANCE))
    )
    seed = anchor_mask & strict_mask
    if int(seed.sum()) >= int(MIN_BODY_SEED_POINTS):
        return seed

    relaxed = anchor_mask & (
        (points[:, 0] >= core_x_min)
        & (points[:, 0] <= core_x_max)
        & (points[:, 1] >= core_y_min)
        & (points[:, 1] <= core_y_max)
    )
    if int(relaxed.sum()) >= int(MIN_BODY_SEED_POINTS):
        return relaxed
    return seed


def fallback_body_seed_from_components(
    points: np.ndarray,
    target: PlantTarget,
) -> np.ndarray:
    labels = label_components_3d(
        points,
        COMPONENT_VOXEL_SIZE,
        max(10, int(MIN_BODY_SEED_POINTS)),
        1,
    )
    valid = np.unique(labels[labels >= 0])
    out = np.zeros(points.shape[0], dtype=bool)
    if valid.size == 0:
        return out
    target_y = float(target.y) if target.y is not None else float(np.median(points[:, 1]))
    best_label = None
    best_score = None
    for label in valid:
        comp = points[labels == label]
        cx = float(np.median(comp[:, 0]))
        cy = float(np.median(comp[:, 1]))
        dist = math.hypot(cx - float(target.x), cy - target_y)
        score = (dist, -float(comp.shape[0]), -float(np.percentile(comp[:, 2], 90)))
        if best_score is None or score < best_score:
            best_score = score
            best_label = int(label)
    if best_label is not None:
        out = labels == best_label
    return out


def grow_component_from_body_seed(
    points: np.ndarray,
    body_seed_mask: np.ndarray,
    target: PlantTarget,
    core_bounds: Tuple[float, float, float, float],
    search_bounds: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray, int]:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool), 0
    voxel_size = float(GROWTH_VOXEL_SIZE)
    if voxel_size <= 0:
        raise ValueError("growth_voxel_size must be > 0.")

    voxels = np.floor(points / voxel_size).astype(np.int64)
    unique_voxels, inverse, counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
    voxel_centers = np.zeros((unique_voxels.shape[0], 3), dtype=np.float32)
    for idx in range(unique_voxels.shape[0]):
        voxel_centers[idx] = np.mean(points[inverse == idx], axis=0)

    seed_voxels = np.unique(inverse[body_seed_mask])
    if seed_voxels.size == 0:
        return np.zeros(points.shape[0], dtype=bool), np.zeros(points.shape[0], dtype=bool), int(unique_voxels.shape[0])

    cost = dijkstra_voxel_costs(unique_voxels, voxel_centers, seed_voxels, core_bounds, search_bounds)
    finite = np.isfinite(cost)
    max_cost = float(GROWTH_MAX_COST)
    target_cost = cost[inverse]
    grown_mask = finite[inverse] & (target_cost <= max_cost)

    competitor_cost = estimate_competitor_cost(points, target)
    ambiguous_mask = np.zeros(points.shape[0], dtype=bool)
    ambiguous_possible = grown_mask & np.isfinite(target_cost) & np.isfinite(competitor_cost)
    ambiguous_mask[ambiguous_possible] = (
        (target_cost[ambiguous_possible] - competitor_cost[ambiguous_possible])
        >= -float(AMBIGUOUS_COST_MARGIN)
    )
    grown_mask &= ~ambiguous_mask

    if grown_mask.any():
        keep = filter_isolated_points_knn(
            points[grown_mask],
            k=CANOPY_OUTLIER_K,
            sigma=CANOPY_OUTLIER_SIGMA,
            min_points=CANOPY_OUTLIER_MIN_POINTS,
            dims=(0, 1, 2),
        )
        original_idx = np.flatnonzero(grown_mask)
        dropped = original_idx[~keep]
        grown_mask[dropped] = False
    return grown_mask, ambiguous_mask, int(unique_voxels.shape[0])


def filter_detached_growth_components(
    grown_points: np.ndarray,
    body_seed_points: np.ndarray,
    target: PlantTarget,
    search_start_z: float,
) -> Tuple[np.ndarray, np.ndarray]:
    keep = np.ones(grown_points.shape[0], dtype=bool)
    if grown_points.shape[0] == 0 or body_seed_points.shape[0] == 0:
        return keep, np.empty((0, 3), dtype=np.float32)

    labels = label_components_3d(
        grown_points,
        float(POST_GROWTH_COMPONENT_VOXEL_SIZE),
        max(1, int(POST_GROWTH_MIN_COMPONENT_POINTS)),
        1,
    )
    valid_labels = np.unique(labels[labels >= 0])
    if valid_labels.size == 0:
        return keep, np.empty((0, 3), dtype=np.float32)
    if valid_labels.size == 1:
        keep = labels >= 0
        detached = grown_points[~keep]
        return keep, detached

    target_y = float(target.y) if target.y is not None else float(np.median(grown_points[:, 1]))
    anchor_z_max = (
        float(search_start_z)
        + float(POST_GROWTH_ANCHOR_Z_MAX_OFFSET)
        - float(MIN_CANOPY_CLEARANCE)
    )
    low_anchor_mask = (
        (body_seed_points[:, 2] <= anchor_z_max)
        & (np.abs(body_seed_points[:, 0] - float(target.x)) <= float(BODY_SEED_X_HALF_WIDTH))
        & (np.abs(body_seed_points[:, 1] - target_y) <= float(BODY_SEED_Y_HALF_WIDTH))
    )
    anchor_points = body_seed_points[low_anchor_mask]
    if anchor_points.shape[0] < int(MIN_BODY_SEED_POINTS):
        anchor_points = body_seed_points

    anchor_indices = nearest_indices_3d(grown_points, anchor_points, max_points=200)
    seed_component_labels = labels[anchor_indices]
    seed_component_labels = set(int(label) for label in seed_component_labels if label >= 0)
    if not seed_component_labels:
        largest_label = max(valid_labels, key=lambda label: int((labels == label).sum()))
        seed_component_labels = {int(largest_label)}

    main_component_mask = np.isin(labels, np.fromiter(seed_component_labels, dtype=np.int32))
    main_component_points = grown_points[main_component_mask]

    keep[:] = False
    for label in valid_labels:
        comp_mask = labels == label
        comp = grown_points[comp_mask]
        if int(label) in seed_component_labels:
            keep[comp_mask] = True
            continue
        centroid_xy = np.median(comp[:, :2], axis=0)
        anchor_dist = math.hypot(float(centroid_xy[0]) - float(target.x), float(centroid_xy[1]) - target_y)
        body_dist = min_xy_distance(comp[:, :2], anchor_points[:, :2])
        component_gap = min_3d_distance(comp, main_component_points)
        if (
            anchor_dist <= float(POST_GROWTH_MAX_ANCHOR_DIST)
            and body_dist <= float(POST_GROWTH_MAX_BODY_DIST)
            and component_gap <= float(POST_GROWTH_MAX_COMPONENT_GAP)
        ):
            keep[comp_mask] = True

    detached = grown_points[~keep]
    return keep, detached


def nearest_indices_xy(points_xy: np.ndarray, query_xy: np.ndarray, max_points: int = 200) -> np.ndarray:
    if points_xy.shape[0] == 0 or query_xy.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    if query_xy.shape[0] > max_points:
        idx = np.linspace(0, query_xy.shape[0] - 1, max_points).astype(np.int64)
        query_xy = query_xy[idx]
    diff = points_xy[None, :, :] - query_xy[:, None, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.argmin(dist2, axis=1).astype(np.int64)


def nearest_indices_3d(points: np.ndarray, query_points: np.ndarray, max_points: int = 200) -> np.ndarray:
    if points.shape[0] == 0 or query_points.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    if query_points.shape[0] > max_points:
        idx = np.linspace(0, query_points.shape[0] - 1, max_points).astype(np.int64)
        query_points = query_points[idx]
    diff = points[None, :, :] - query_points[:, None, :]
    dist2 = np.sum(diff * diff, axis=2)
    return np.argmin(dist2, axis=1).astype(np.int64)


def min_xy_distance(points_xy: np.ndarray, query_xy: np.ndarray) -> float:
    if points_xy.shape[0] == 0 or query_xy.shape[0] == 0:
        return float("inf")
    if points_xy.shape[0] * query_xy.shape[0] > 400_000:
        p_idx = np.linspace(0, points_xy.shape[0] - 1, min(points_xy.shape[0], 400)).astype(np.int64)
        q_idx = np.linspace(0, query_xy.shape[0] - 1, min(query_xy.shape[0], 400)).astype(np.int64)
        points_xy = points_xy[p_idx]
        query_xy = query_xy[q_idx]
    diff = points_xy[:, None, :] - query_xy[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return float(np.sqrt(np.min(dist2)))


def min_3d_distance(points: np.ndarray, query_points: np.ndarray) -> float:
    if points.shape[0] == 0 or query_points.shape[0] == 0:
        return float("inf")
    if points.shape[0] * query_points.shape[0] > 400_000:
        p_idx = np.linspace(0, points.shape[0] - 1, min(points.shape[0], 400)).astype(np.int64)
        q_idx = np.linspace(0, query_points.shape[0] - 1, min(query_points.shape[0], 400)).astype(np.int64)
        points = points[p_idx]
        query_points = query_points[q_idx]
    diff = points[:, None, :] - query_points[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return float(np.sqrt(np.min(dist2)))


def dijkstra_voxel_costs(
    unique_voxels: np.ndarray,
    voxel_centers: np.ndarray,
    seed_voxels: np.ndarray,
    core_bounds: Tuple[float, float, float, float],
    search_bounds: Tuple[float, float, float, float],
) -> np.ndarray:
    n = int(unique_voxels.shape[0])
    lookup = {tuple(v.tolist()): i for i, v in enumerate(unique_voxels)}
    costs = np.full(n, np.inf, dtype=np.float32)
    heap: List[Tuple[float, int]] = []
    for idx in seed_voxels:
        costs[int(idx)] = 0.0
        heapq.heappush(heap, (0.0, int(idx)))

    radius = max(1, int(GROWTH_NEIGHBOR_RADIUS))
    offsets = [
        (dx, dy, dz)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dz in range(-radius, radius + 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]
    max_cost = float(GROWTH_MAX_COST)
    while heap:
        current_cost, idx = heapq.heappop(heap)
        if current_cost > float(costs[idx]) or current_cost > max_cost:
            continue
        vx, vy, vz = unique_voxels[idx]
        z0 = float(voxel_centers[idx, 2])
        for dx, dy, dz in offsets:
            nb = lookup.get((int(vx + dx), int(vy + dy), int(vz + dz)))
            if nb is None:
                continue
            step = float(GROWTH_STEP_COST) * math.sqrt(float(dx * dx + dy * dy + dz * dz))
            step += soft_roi_penalty(voxel_centers[nb], core_bounds, search_bounds)
            step += max(0.0, float(voxel_centers[nb, 2]) - z0) / max(float(GROWTH_VOXEL_SIZE), 1e-8) * float(GROWTH_HEIGHT_COST)
            new_cost = current_cost + step
            if new_cost < float(costs[nb]) and new_cost <= max_cost:
                costs[nb] = new_cost
                heapq.heappush(heap, (new_cost, nb))
    return costs


def soft_roi_penalty(
    xyz: np.ndarray,
    core_bounds: Tuple[float, float, float, float],
    search_bounds: Tuple[float, float, float, float],
) -> float:
    x, y = float(xyz[0]), float(xyz[1])
    core_x_min, core_x_max, core_y_min, core_y_max = core_bounds
    search_x_min, search_x_max, search_y_min, search_y_max = search_bounds
    if not (search_x_min <= x <= search_x_max and search_y_min <= y <= search_y_max):
        return float(GROWTH_OUTSIDE_SEARCH_COST)
    if core_x_min <= x <= core_x_max and core_y_min <= y <= core_y_max:
        return 0.0
    dx = max(core_x_min - x, 0.0, x - core_x_max)
    dy = max(core_y_min - y, 0.0, y - core_y_max)
    dist = math.hypot(dx, dy)
    return float(GROWTH_OUTSIDE_CORE_COST) * (1.0 + dist / max(float(GROWTH_VOXEL_SIZE), 1e-8))


def estimate_competitor_cost(points: np.ndarray, target: PlantTarget) -> np.ndarray:
    if target.envelope_x_min is None or target.envelope_x_max is None:
        return np.full(points.shape[0], np.inf, dtype=np.float32)
    target_x = float(target.x)
    core_width = max(float(target.envelope_x_max) - float(target.envelope_x_min), 1e-6)
    left_dist = np.maximum(0.0, target_x - points[:, 0])
    right_dist = np.maximum(0.0, points[:, 0] - target_x)
    # A nearby neighbor core is only considered a competitor once the point is closer to a side
    # boundary than to the cultivation anchor. This intentionally ignores ordinary intra-plant spread.
    side_distance = np.minimum(left_dist, right_dist)
    competitor = side_distance / max(core_width * 0.5, 1e-6)
    in_own_core = (points[:, 0] >= float(target.envelope_x_min)) & (points[:, 0] <= float(target.envelope_x_max))
    competitor = np.where(in_own_core, np.inf, competitor)
    return competitor.astype(np.float32)


def density_mask_3d(points: np.ndarray, voxel_size: float, min_voxel_points: int) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if voxel_size <= 0:
        raise ValueError("voxel_size_3d must be > 0.")
    if min_voxel_points <= 1:
        return np.ones(points.shape[0], dtype=bool)
    voxels = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inverse, counts = np.unique(voxels, axis=0, return_inverse=True, return_counts=True)
    return counts[inverse] >= int(min_voxel_points)


def label_components_3d(
    points: np.ndarray,
    voxel_size: float,
    min_component_points: int,
    min_component_voxels: int,
) -> np.ndarray:
    labels = -np.ones(points.shape[0], dtype=np.int32)
    if points.shape[0] == 0:
        return labels
    if voxel_size <= 0:
        raise ValueError("component_voxel_size must be > 0.")

    voxels = np.floor(points / float(voxel_size)).astype(np.int64)
    unique_voxels, inverse, voxel_counts = np.unique(
        voxels,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    voxel_lookup = {tuple(v.tolist()): idx for idx, v in enumerate(unique_voxels)}
    comp_of_voxel = -np.ones(unique_voxels.shape[0], dtype=np.int32)
    comp_point_counts: List[int] = []
    comp_voxel_counts: List[int] = []
    next_label = 0

    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if not (dx == 0 and dy == 0 and dz == 0)
    ]

    for start_idx in range(unique_voxels.shape[0]):
        if comp_of_voxel[start_idx] >= 0:
            continue
        stack = [start_idx]
        comp_of_voxel[start_idx] = next_label
        point_count = 0
        voxel_count = 0
        while stack:
            idx = stack.pop()
            point_count += int(voxel_counts[idx])
            voxel_count += 1
            cx, cy, cz = unique_voxels[idx]
            for dx, dy, dz in offsets:
                nb = voxel_lookup.get((int(cx + dx), int(cy + dy), int(cz + dz)))
                if nb is None or comp_of_voxel[nb] >= 0:
                    continue
                comp_of_voxel[nb] = next_label
                stack.append(nb)
        comp_point_counts.append(point_count)
        comp_voxel_counts.append(voxel_count)
        next_label += 1

    point_labels = comp_of_voxel[inverse]
    point_counts = np.asarray(comp_point_counts, dtype=np.int32)
    voxel_counts_arr = np.asarray(comp_voxel_counts, dtype=np.int32)
    valid = (point_counts >= int(min_component_points)) & (voxel_counts_arr >= int(min_component_voxels))
    labels[:] = np.where(valid[point_labels], point_labels, -1)
    return labels


def filter_isolated_points_knn(
    points: np.ndarray,
    k: int,
    sigma: float,
    min_points: int,
    dims: Tuple[int, ...] = (0, 1, 2),
) -> np.ndarray:
    n = int(points.shape[0])
    keep = np.ones(n, dtype=bool)
    if n == 0 or n < max(int(min_points), int(k) + 1):
        return keep

    q = np.asarray(points[:, dims], dtype=np.float32)
    diff = q[:, None, :] - q[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, np.inf)

    kth = max(1, min(int(k), n - 1))
    knn_dist = np.sqrt(np.partition(dist2, kth - 1, axis=1)[:, kth - 1])
    med = float(np.median(knn_dist))
    mad = float(np.median(np.abs(knn_dist - med)))
    if mad < 1e-8:
        return keep

    robust_sigma = 1.4826 * mad
    thresh = med + float(sigma) * robust_sigma
    return knn_dist <= thresh


def filter_sparse_points_xy(
    points_xy: np.ndarray,
    cell_size: float,
    min_cell_points: int,
    min_neighbor_points: int,
) -> np.ndarray:
    keep = np.ones(points_xy.shape[0], dtype=bool)
    if points_xy.shape[0] == 0:
        return keep
    if cell_size <= 0:
        raise ValueError("cluster_cell_size must be > 0.")

    cells = np.floor(points_xy / float(cell_size)).astype(np.int64)
    unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    cell_index = {tuple(cell.tolist()): idx for idx, cell in enumerate(unique_cells)}

    keep_cell = np.ones(unique_cells.shape[0], dtype=bool)
    if min_cell_points > 1:
        keep_cell &= counts >= int(min_cell_points)

    if min_neighbor_points > 0:
        neigh_counts = np.zeros(unique_cells.shape[0], dtype=np.int32)
        for idx, (cx, cy) in enumerate(unique_cells):
            total = 0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = cell_index.get((int(cx + dx), int(cy + dy)))
                    if nb is not None:
                        total += int(counts[nb])
            neigh_counts[idx] = total
        keep_cell &= neigh_counts >= int(min_neighbor_points)

    return keep_cell[inverse]


def build_top_surface_xy(
    points: np.ndarray,
    rgb: np.ndarray,
    cell_size: float,
    min_cell_points: int,
    z_percentile: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.int64),
        )
    if cell_size <= 0:
        raise ValueError("surface_cell_size must be > 0.")

    cells = np.floor(points[:, :2] / float(cell_size)).astype(np.int64)
    unique_cells, inverse = np.unique(cells, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    keep_cell = counts >= int(max(1, min_cell_points))
    kept_idx = np.flatnonzero(keep_cell)
    if kept_idx.size == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 2), dtype=np.int64),
        )

    surface_xyz = np.empty((kept_idx.size, 3), dtype=np.float32)
    surface_rgb = np.empty((kept_idx.size, 3), dtype=np.float32)
    surface_cells = np.empty((kept_idx.size, 2), dtype=np.int64)
    out_i = 0
    for cell_idx in kept_idx:
        sel = inverse == cell_idx
        cell_pts = points[sel]
        cell_rgb = rgb[sel]
        xy_center = np.mean(cell_pts[:, :2], axis=0)
        z_top = float(np.percentile(cell_pts[:, 2], z_percentile))
        rgb_mean = np.mean(cell_rgb, axis=0)

        surface_xyz[out_i, 0] = float(xy_center[0])
        surface_xyz[out_i, 1] = float(xy_center[1])
        surface_xyz[out_i, 2] = z_top
        surface_rgb[out_i] = np.asarray(rgb_mean, dtype=np.float32)
        surface_cells[out_i] = unique_cells[cell_idx]
        out_i += 1

    return surface_xyz, surface_rgb, surface_cells


def select_points_in_cells_xy(
    points: np.ndarray,
    rgb: np.ndarray,
    cell_size: float,
    selected_cells: np.ndarray,
    dilation: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0 or selected_cells.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    if cell_size <= 0:
        raise ValueError("surface_cell_size must be > 0.")

    cell_set = set()
    for cx, cy in np.asarray(selected_cells, dtype=np.int64):
        for dx in range(-int(dilation), int(dilation) + 1):
            for dy in range(-int(dilation), int(dilation) + 1):
                cell_set.add((int(cx + dx), int(cy + dy)))

    cells = np.floor(points[:, :2] / float(cell_size)).astype(np.int64)
    mask = np.fromiter(
        ((int(cx), int(cy)) in cell_set for cx, cy in cells),
        dtype=bool,
        count=cells.shape[0],
    )
    return np.asarray(points[mask], dtype=np.float32), np.asarray(rgb[mask], dtype=np.float32)

def finalize_heights(rows: Sequence[Dict[str, object]]) -> None:
    for row in rows:
        if row.get("status") != "ok":
            continue
        top_z = numeric_or_none(row.get("top_z"))
        ground = numeric_or_none(row.get("ground_z"))
        if top_z is None or ground is None:
            row["status"] = "failed_missing_height"
            continue
        row["pred_height_cm"] = (top_z - ground) * EXPORT_Z_SCALE
        gt = numeric_or_none(row.get("gt_highest_cm"))
        if gt is not None:
            row["error_cm"] = row["pred_height_cm"] - gt
            row["abs_error_cm"] = abs(row["error_cm"])


def numeric_or_none(value: object) -> Optional[float]:
    if value in ("", None):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "plant_id", "input_x", "input_y", "gt_highest_cm",
        "pred_height_cm", "error_cm", "abs_error_cm",
        "ground_z", "top_z", "status", "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def write_ordered_heights_txt(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            h = row.get("pred_height_cm", "")
            if h != "":
                f.write(f"{i:02d},{row['plant_id']},{float(h):.{EXPORT_Z_DIGITS}f}\n")


def print_evaluation_summary(rows: Sequence[Dict[str, object]]) -> None:
    errs = [float(r["error_cm"]) for r in rows if numeric_or_none(r.get("error_cm")) is not None]
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"[Evaluation] ok: {ok}/{len(rows)}")
    if not errs:
        return
    a = np.asarray(errs)
    print(f"[Evaluation] MAE={float(np.mean(np.abs(a))):.2f} cm  bias={float(np.mean(a)):.2f} cm")


def write_rgb_ply(path: Path, points: np.ndarray, rgb: np.ndarray) -> None:
    if PlyData is None:
        raise ImportError("plyfile required") from PLY_IMPORT_ERROR
    verts = np.empty(
        points.shape[0],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    verts["x"], verts["y"], verts["z"] = points[:, 0], points[:, 1], points[:, 2]
    c = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    verts["red"], verts["green"], verts["blue"] = c[:, 0], c[:, 1], c[:, 2]
    PlyData([PlyElement.describe(verts, "vertex")], text=True).write(str(path))


def write_merged_canopy_ply(path: Path, parts: Sequence[Tuple[str, np.ndarray, np.ndarray]]) -> None:
    xyzs, rgbs = [], []
    for _, xyz, rgb in parts:
        if xyz.shape[0]:
            xyzs.append(xyz)
            rgbs.append(rgb)
    if not xyzs:
        return
    merged = np.vstack(xyzs)
    rgb_m = np.vstack(rgbs)
    if EXPORT_CANOPY_FRAME == "input_layout":
        merged = merged.copy()
        merged[:, 0] /= INPUT_X_SCALE
        merged[:, 1] /= INPUT_Y_SCALE
    if EXPORT_CANOPY_SWAP_YZ:
        merged = merged.copy()
        merged[:, [1, 2]] = merged[:, [2, 1]]
    write_rgb_ply(path, merged, rgb_m)


def main() -> int:
    ply_path = resolve_input_ply(Path(INPUT_PATH).expanduser().resolve())
    targets = load_targets(Path(POSITIONS_FILE).expanduser().resolve())
    assign_target_envelopes(targets)
    cloud = load_cloud(ply_path)
    n0 = cloud.points.shape[0]
    cloud = filter_cloud(cloud)
    cloud = crop_cloud_global_bounds(cloud)
    green = compute_green_seed_mask(cloud.rgb)
    print(f"[Info] ply: {ply_path}  points: {cloud.points.shape[0]}/{n0}  plants: {len(targets)}")
    results = [measure_single_plant(cloud, green, t) for t in targets]
    rows = [r.row for r in results]
    finalize_heights(rows)
    out_csv = Path(OUT_CSV).expanduser().resolve()
    write_csv(out_csv, rows)
    write_ordered_heights_txt(Path(OUT_HEIGHTS_TXT).expanduser().resolve(), rows)
    print_evaluation_summary(rows)
    print(f"[Done] {out_csv}")
    if OUT_CANOPY_PLY.strip():
        parts = [
            (str(r.row["plant_id"]), r.canopy_xyz, r.canopy_rgb)
            for r in results
            if r.canopy_xyz is not None and r.canopy_rgb is not None
        ]
        if parts:
            p = Path(OUT_CANOPY_PLY).expanduser().resolve()
            write_merged_canopy_ply(p, parts)
            print(f"[Done] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
