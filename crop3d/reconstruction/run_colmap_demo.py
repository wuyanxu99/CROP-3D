#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run COLMAP sparse reconstruction for a multi-camera scene.

Expected layout:
    scene/
      images/
        camera_1/
          frame_000001.jpg
        camera_2/
          frame_000001.jpg

Each direct subfolder under ``images/`` is treated as one physical camera.
Outputs are written to the scene folder: ``database.db``, ``sparse/``,
``image_matches.txt``, and ``colmap.log``.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

# Required: scene root. The script reads SCENE/images/.
SCENE = "./examples/reconstruction_demo/scene"
# Required if COLMAP is not on PATH.
COLMAP_BIN = "colmap"

REBUILD_DATABASE = True
CLEAN_SPARSE_OUTPUTS = True
SKIP_FEATURE_EXTRACTION = False
SKIP_SEQUENTIAL_MATCHING = False
SKIP_CROSS_CAMERA_MATCHING = False

USE_CAMERA_INTRINSICS = False
CAMERA_MODEL = "OPENCV"
CAMERA_PARAMS = "1407.0,1408.3,986.7009,518.1291,-0.0331,0.0209,0,0"
USE_RIG_EXTRINSICS = False

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
SEQUENTIAL_OVERLAP = 28

# Cross-camera matching: "exact_only", "exact_or_nearby", or "nearby_only".
CROSS_CAMERA_MATCH_MODE = "nearby_only"
CROSS_CAMERA_FRAME_REGEX = r"(\d+)"
CROSS_CAMERA_MAX_FRAME_DELTA = 4
CROSS_CAMERA_MAX_NEIGHBORS_PER_IMAGE = 4
CROSS_CAMERA_BIDIRECTIONAL = True

MAX_FEATURES = "65536"
SIFT_PEAK_THRESHOLD = "0.0020"
SIFT_ESTIMATE_AFFINE_SHAPE = 1
SIFT_DOMAIN_SIZE_POOLING = 1
SIFT_GUIDED_MATCHING = 1
SIFT_NUM_THREADS = 6
SIFT_MAX_IMAGE_SIZE = None

MAPPER_MULTIPLE_MODELS = False
MAPPER_MIN_MODEL_SIZE = 10
MAPPER_TRI_IGNORE_TWO_VIEW_TRACKS = 0
MAPPER_MIN_DEPTH = None
MAPPER_MAX_DEPTH = None
MAPPER_FILTER_MAX_REPROJ_ERROR = "2.5"
MAPPER_TRI_MIN_RAY_ANGLE = "1.0"
MAPPER_BA_REFINE_FOCAL_LENGTH = None
MAPPER_BA_REFINE_PRINCIPAL_POINT = None
MAPPER_BA_REFINE_EXTRA_PARAMS = None
MAPPER_BA_LOCAL_MAX_NUM_ITERATIONS = None
MAPPER_BA_GLOBAL_MAX_NUM_ITERATIONS = None
MAPPER_BA_GLOBAL_MAX_REFINEMENTS = None
MAPPER_BA_LOCAL_MAX_REFINEMENTS = None
MAPPER_BA_GLOBAL_FRAMES_RATIO = None
MAPPER_BA_GLOBAL_POINTS_RATIO = None
MAPPER_BA_USE_GPU = None
MAPPER_BA_GPU_INDEX = None

RUN_POINT_FILTERING = True
POINT_FILTER_MIN_TRACK_LEN = 2
POINT_FILTER_MAX_REPROJ_ERROR = "2.0"

PROMOTE_LARGEST_COMPONENT_TO_ZERO = True
MERGE_ADDITIONAL_COMPONENTS = False
MERGE_MIN_COMPONENT_IMAGES = 10

# Set to a custom path to override scene/colmap.log.
LOG_PATH = None
NO_FILTER_LOG = False


# -----------------------------------------------------------------------------
# Runtime constants
# -----------------------------------------------------------------------------

KEY_FILTER_PATTERN = re.compile(
    r"Registering|Registered|Mean|reprojection|Error|Elapsed|Writing|Cameras|Images|Points|"
    r"Extracted|Matching block|Residuals|Parameters|Iteration|Initial cost|Final cost|"
    r"Termination|observations|Filtered|Changed|Convergence|Time\s*:",
    re.IGNORECASE,
)

RE_ELAPSED = re.compile(r"Elapsed time:\s*([\d.]+)\s*\[minutes\]")
RE_FILTERED_OBS = re.compile(r"^Filtered observations:\s*(\d+)", re.MULTILINE)
RE_FINAL_COST = re.compile(r"Final cost\s*:\s*([\d.]+)\s*\[px\]")
RE_INITIAL_COST = re.compile(r"Initial cost\s*:\s*([\d.]+)\s*\[px\]")
RE_RESIDUALS = re.compile(r"Residuals\s*:\s*(\d+)")
RE_ITERATIONS = re.compile(r"Iterations\s*:\s*(\d+)")
RE_REGISTERING_IMAGE = re.compile(r"Registering image #(\d+) \(num_reg_frames=(\d+)\)")


@dataclass(frozen=True)
class ScenePaths:
    scene: Path
    image_path: Path
    db_path: Path
    sparse_path: Path
    match_list_path: Path
    colmap_log: Path


@dataclass(frozen=True)
class FeatureConfig:
    use_camera_intrinsics: bool
    camera_model: str
    camera_params: str | None
    max_features: str
    peak_threshold: str
    estimate_affine_shape: int
    domain_size_pooling: int
    guided_matching: int
    num_threads: int
    max_image_size: int | None


@dataclass(frozen=True)
class MatchConfig:
    image_extensions: tuple[str, ...]
    sequential_overlap: int
    frame_regex: str
    cross_camera_match_mode: str
    cross_camera_max_frame_delta: int
    cross_camera_max_neighbors: int
    cross_camera_bidirectional: bool


@dataclass(frozen=True)
class MapperConfig:
    multiple_models: bool
    min_model_size: int
    tri_ignore_two_view_tracks: int
    min_depth: str | None
    max_depth: str | None
    filter_max_reproj_error: str
    tri_min_ray_angle: str
    ba_refine_focal_length: int
    ba_refine_principal_point: int
    ba_refine_extra_params: int
    ba_local_max_num_iterations: str | None
    ba_global_max_num_iterations: str | None
    ba_global_max_refinements: str | None
    ba_local_max_refinements: str | None
    ba_global_frames_ratio: str | None
    ba_global_points_ratio: str | None
    ba_use_gpu: str | None
    ba_gpu_index: str | None


@dataclass(frozen=True)
class OutputConfig:
    rebuild_database: bool
    clean_sparse_outputs: bool
    run_point_filtering: bool
    point_filter_min_track_len: int
    point_filter_max_reproj_error: str
    promote_largest_component_to_zero: bool
    merge_additional_components: bool
    merge_min_component_images: int


@dataclass(frozen=True)
class MatchLookup:
    by_idx: dict[int, list[str]]
    exact_name: dict[str, str]
    nearby: list[tuple[int, str]]


def run_cmd(
    cmd: list[str],
    log_path: Path | None,
    quiet_filter: bool = True,
    mapper_total_images: int | None = None,
) -> int:
    if log_path is None:
        return subprocess.run(cmd, check=False).returncode

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            log_file.write(line + "\n")
            log_file.flush()

            reg = RE_REGISTERING_IMAGE.search(line)
            if reg and mapper_total_images:
                image_id = int(reg.group(1))
                num_reg_frames = int(reg.group(2))
                pct = 100.0 * num_reg_frames / max(1, mapper_total_images)
                print(
                    f"Registering image #{image_id} "
                    f"({num_reg_frames}/{mapper_total_images}, {pct:.1f}%)"
                )
                continue

            if not quiet_filter or KEY_FILTER_PATTERN.search(line):
                print(line)

        rc = proc.wait()

    if rc != 0 and log_path.is_file():
        print(f"\n[Error] Command exited with code {rc}. Full log: {log_path}", file=sys.stderr)
        try:
            tail = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()[-40:]
            print("\n--- Log tail ---\n" + "\n".join(tail), file=sys.stderr)
        except OSError:
            pass

    return rc


def discover_camera_folders(image_path: Path, extensions: tuple[str, ...]) -> list[str]:
    camera_folders: list[str] = []
    for child in sorted(image_path.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if any(f.is_file() and f.suffix.lower() in extensions for f in child.iterdir()):
            camera_folders.append(child.name)
    return camera_folders


def count_images(image_path: Path, extensions: tuple[str, ...]) -> int:
    return sum(1 for p in image_path.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


def extract_frame_index(name: str, frame_regex: str) -> int | None:
    matches = re.findall(frame_regex, Path(name).stem)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = next((item for item in reversed(value) if item), "")
    try:
        return int(value)
    except ValueError:
        return None


def collect_camera_images(
    image_path: Path,
    camera_folders: list[str],
    extensions: tuple[str, ...],
    frame_regex: str,
) -> dict[str, list[tuple[str, int | None]]]:
    rows: dict[str, list[tuple[str, int | None]]] = {}
    for camera_name in camera_folders:
        camera_rows: list[tuple[str, int | None]] = []
        for file_path in sorted((image_path / camera_name).iterdir()):
            if not file_path.is_file() or file_path.suffix.lower() not in extensions:
                continue
            rel_path = f"{camera_name}/{file_path.name}"
            camera_rows.append((rel_path, extract_frame_index(file_path.name, frame_regex)))
        rows[camera_name] = camera_rows
    return rows


def build_match_lookup(rows: list[tuple[str, int | None]]) -> MatchLookup:
    by_idx: dict[int, list[str]] = defaultdict(list)
    exact_name: dict[str, str] = {}
    nearby: list[tuple[int, str]] = []

    for rel_path, frame_idx in rows:
        exact_name[Path(rel_path).name] = rel_path
        if frame_idx is None:
            continue
        by_idx[frame_idx].append(rel_path)
        nearby.append((frame_idx, rel_path))

    nearby.sort(key=lambda item: (item[0], item[1]))
    return MatchLookup(dict(by_idx), exact_name, nearby)


def nearest_candidates(
    target_idx: int,
    candidates_sorted: list[tuple[int, str]],
    max_frame_delta: int,
    max_neighbors: int,
) -> list[str]:
    if not candidates_sorted or max_neighbors <= 0:
        return []

    frame_indices = [item[0] for item in candidates_sorted]
    pos = bisect_left(frame_indices, target_idx)
    left = pos - 1
    right = pos
    found: list[str] = []
    seen: set[str] = set()

    while len(found) < max_neighbors and (left >= 0 or right < len(candidates_sorted)):
        left_candidate = None
        right_candidate = None

        if left >= 0:
            delta = abs(candidates_sorted[left][0] - target_idx)
            if delta <= max_frame_delta:
                left_candidate = (delta, candidates_sorted[left][1])

        if right < len(candidates_sorted):
            delta = abs(candidates_sorted[right][0] - target_idx)
            if delta <= max_frame_delta:
                right_candidate = (delta, candidates_sorted[right][1])

        chosen = None
        if left_candidate is not None and right_candidate is not None:
            chosen = left_candidate if left_candidate[0] <= right_candidate[0] else right_candidate
            if chosen is left_candidate:
                left -= 1
            else:
                right += 1
        elif left_candidate is not None:
            chosen = left_candidate
            left -= 1
        elif right_candidate is not None:
            chosen = right_candidate
            right += 1
        else:
            break

        rel_path = chosen[1]
        if rel_path not in seen:
            seen.add(rel_path)
            found.append(rel_path)

    return found


def add_cross_camera_pairs(
    source_rows: list[tuple[str, int | None]],
    target_lookup: MatchLookup,
    match_mode: str,
    max_frame_delta: int,
    max_neighbors: int,
    pair_set: set[tuple[str, str]],
) -> None:
    for rel_source, frame_idx in source_rows:
        name = Path(rel_source).name
        matched_exact = False

        if match_mode in {"exact_only", "exact_or_nearby"}:
            if frame_idx is not None and frame_idx in target_lookup.by_idx:
                for rel_target in target_lookup.by_idx[frame_idx]:
                    pair_set.add(
                        (rel_source, rel_target)
                        if rel_source < rel_target
                        else (rel_target, rel_source)
                    )
                    matched_exact = True
            elif name in target_lookup.exact_name:
                rel_target = target_lookup.exact_name[name]
                pair_set.add(
                    (rel_source, rel_target)
                    if rel_source < rel_target
                    else (rel_target, rel_source)
                )
                matched_exact = True

        if match_mode == "exact_only" or frame_idx is None:
            continue
        if matched_exact and match_mode == "exact_or_nearby":
            continue

        for rel_target in nearest_candidates(
            frame_idx,
            target_lookup.nearby,
            max_frame_delta,
            max_neighbors,
        ):
            pair_set.add(
                (rel_source, rel_target)
                if rel_source < rel_target
                else (rel_target, rel_source)
            )


def build_cross_camera_pairs(
    camera_rows: dict[str, list[tuple[str, int | None]]],
    match_mode: str,
    max_frame_delta: int,
    max_neighbors: int,
    bidirectional: bool,
) -> list[tuple[str, str]]:
    pair_set: set[tuple[str, str]] = set()
    cameras = sorted(camera_rows)

    for cam_a, cam_b in combinations(cameras, 2):
        rows_a = camera_rows[cam_a]
        rows_b = camera_rows[cam_b]
        if not rows_a or not rows_b:
            continue

        add_cross_camera_pairs(
            rows_a,
            build_match_lookup(rows_b),
            match_mode,
            max_frame_delta,
            max_neighbors,
            pair_set,
        )
        if bidirectional:
            add_cross_camera_pairs(
                rows_b,
                build_match_lookup(rows_a),
                match_mode,
                max_frame_delta,
                max_neighbors,
                pair_set,
            )

    return sorted(pair_set)


def write_match_list(match_list_path: Path, pairs: list[tuple[str, str]]) -> None:
    content = "".join(f"{left} {right}\n" for left, right in pairs)
    match_list_path.write_text(content, encoding="utf-8")


def mapper_args(config: MapperConfig) -> list[str]:
    args: list[str] = []
    if config.min_depth and config.min_depth.strip():
        args.extend(["--Mapper.min_depth", config.min_depth.strip()])
    if config.max_depth and config.max_depth.strip():
        args.extend(["--Mapper.max_depth", config.max_depth.strip()])
    args.extend(
        [
            "--Mapper.multiple_models",
            str(int(config.multiple_models)),
            "--Mapper.min_model_size",
            str(config.min_model_size),
            "--Mapper.filter_max_reproj_error",
            config.filter_max_reproj_error,
            "--Mapper.tri_min_angle",
            config.tri_min_ray_angle,
            "--Mapper.tri_ignore_two_view_tracks",
            str(config.tri_ignore_two_view_tracks),
            "--Mapper.ba_refine_focal_length",
            str(config.ba_refine_focal_length),
            "--Mapper.ba_refine_principal_point",
            str(config.ba_refine_principal_point),
            "--Mapper.ba_refine_extra_params",
            str(config.ba_refine_extra_params),
        ]
    )
    optional_args = [
        ("--Mapper.ba_local_max_num_iterations", config.ba_local_max_num_iterations),
        ("--Mapper.ba_global_max_num_iterations", config.ba_global_max_num_iterations),
        ("--Mapper.ba_global_max_refinements", config.ba_global_max_refinements),
        ("--Mapper.ba_local_max_refinements", config.ba_local_max_refinements),
        ("--Mapper.ba_global_frames_ratio", config.ba_global_frames_ratio),
        ("--Mapper.ba_global_points_ratio", config.ba_global_points_ratio),
        ("--Mapper.ba_use_gpu", config.ba_use_gpu),
        ("--Mapper.ba_gpu_index", config.ba_gpu_index),
    ]
    for flag, value in optional_args:
        if value is not None and str(value).strip():
            args.extend([flag, str(value).strip()])
    return args


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def cleanup_database_files(db_path: Path) -> None:
    for suffix in ("", "-shm", "-wal"):
        remove_path(Path(str(db_path) + suffix))


def cleanup_sparse_outputs(sparse_path: Path) -> None:
    if not sparse_path.exists():
        return
    for child in sparse_path.iterdir():
        if child.name.isdigit() or child.name in {"merged_tmp", "_tmp_old_zero"}:
            remove_path(child)


def read_binary_count(path: Path) -> int | None:
    try:
        with open(path, "rb") as file:
            return struct.unpack("<Q", file.read(8))[0]
    except (OSError, struct.error):
        return None


def collect_sparse_components(sparse_path: Path) -> list[dict[str, int | Path]]:
    components: list[dict[str, int | Path]] = []
    if not sparse_path.exists():
        return components

    for child in sorted(sparse_path.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or not child.name.isdigit():
            continue
        num_images = read_binary_count(child / "images.bin")
        num_points = read_binary_count(child / "points3D.bin")
        if num_images is None or num_points is None:
            continue
        components.append(
            {
                "index": int(child.name),
                "path": child,
                "num_images": int(num_images),
                "num_points": int(num_points),
            }
        )

    components.sort(
        key=lambda item: (int(item["num_images"]), int(item["num_points"]), -int(item["index"])),
        reverse=True,
    )
    return components


def print_component_table(components: list[dict[str, int | Path]], title: str) -> None:
    if not components:
        return
    print(title)
    for info in components:
        print(
            "      sparse/{} -> images={}, points={}".format(
                int(info["index"]),
                int(info["num_images"]),
                int(info["num_points"]),
            )
        )


def promote_largest_component_to_zero(
    sparse_path: Path,
    components: list[dict[str, int | Path]],
) -> Path:
    sparse_0 = sparse_path / "0"
    if not components:
        return sparse_0

    primary = Path(components[0]["path"])
    if primary == sparse_0:
        return sparse_0

    tmp_old_zero = sparse_path / "_tmp_old_zero"
    if tmp_old_zero.exists():
        remove_path(tmp_old_zero)
    if sparse_0.exists():
        shutil.move(str(sparse_0), str(tmp_old_zero))

    shutil.move(str(primary), str(sparse_0))
    if tmp_old_zero.exists():
        dest = sparse_path / str(int(components[0]["index"]))
        if dest.exists():
            remove_path(dest)
        shutil.move(str(tmp_old_zero), str(dest))

    return sparse_0


def run_point_filtering(
    sparse_0: Path,
    min_track_len: int,
    max_reproj_error: str,
    colmap_log: Path,
    quiet_filter: bool,
) -> bool:
    cmd = [
        COLMAP_BIN,
        "point_filtering",
        "--input_path",
        str(sparse_0),
        "--output_path",
        str(sparse_0),
        "--min_track_len",
        str(min_track_len),
        "--max_reproj_error",
        max_reproj_error,
    ]
    if run_cmd(cmd, colmap_log, quiet_filter) != 0:
        print("Warning: point_filtering failed or not available, skipping.")
        return False
    return True


def parse_log_metrics(log_path: Path) -> dict[str, float | int | list[float] | None]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    elapsed_minutes = [float(item) for item in RE_ELAPSED.findall(text)]
    filtered_observations = RE_FILTERED_OBS.findall(text)
    final_costs = RE_FINAL_COST.findall(text)
    initial_costs = RE_INITIAL_COST.findall(text)
    residuals = RE_RESIDUALS.findall(text)
    iterations = RE_ITERATIONS.findall(text)

    return {
        "elapsed_minutes": elapsed_minutes,
        "total_minutes": sum(elapsed_minutes),
        "filtered_observations": int(filtered_observations[-1]) if filtered_observations else None,
        "final_cost_px": float(final_costs[-1]) if final_costs else None,
        "initial_cost_px": float(initial_costs[-1]) if initial_costs else None,
        "residuals": int(residuals[-1]) if residuals else None,
        "ba_iterations": int(iterations[-1]) if iterations else None,
    }


def run_model_analyzer(sparse_path: Path) -> str | None:
    try:
        result = subprocess.run(
            [COLMAP_BIN, "model_analyzer", "--path", str(sparse_path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception:
        return None

    if result.returncode != 0:
        return None

    chunks = []
    if result.stdout.strip():
        chunks.append(result.stdout.strip())
    if result.stderr.strip():
        chunks.append(result.stderr.strip())
    return "\n".join(chunks) if chunks else None


def strip_colmap_glog_line(line: str) -> str:
    line = line.strip()
    if "] " in line:
        return line.rsplit("] ", 1)[-1].strip()
    return line


def print_run_summary(
    scene: Path,
    sparse_0: Path,
    colmap_log: Path,
    ran_point_filtering: bool,
) -> None:
    lines = ["", "=" * 56, "  COLMAP run summary", "=" * 56]

    if sparse_0.is_dir():
        analyzer_output = run_model_analyzer(sparse_0)
        if analyzer_output:
            lines.extend(["", "[Sparse reconstruction]"])
            for raw_line in analyzer_output.splitlines():
                line = strip_colmap_glog_line(raw_line)
                if not line:
                    continue
                if "Rigs:" in line:
                    lines.append(f"  Rigs:          {line.split(':', 1)[1].strip()}")
                elif "Frames:" in line and "Registered" not in line:
                    lines.append(f"  Frames:        {line.split(':', 1)[1].strip()}")
                elif "Registered frames:" in line:
                    lines.append(f"  Registered frames: {line.split(':', 1)[1].strip()}")
                elif "Cameras:" in line:
                    lines.append(f"  Cameras:       {line.split(':', 1)[1].strip()}")
                elif "Images:" in line and "Registered" not in line:
                    lines.append(f"  Images:        {line.split(':', 1)[1].strip()}")
                elif "Registered images:" in line:
                    lines.append(f"  Registered images: {line.split(':', 1)[1].strip()}")
                elif "Points:" in line:
                    lines.append(f"  Points:        {line.split(':', 1)[1].strip()}")
                elif "Observations:" in line:
                    lines.append(f"  Observations:  {line.split(':', 1)[1].strip()}")
                elif "Mean track length:" in line:
                    lines.append(f"  Mean track length: {line.split(':', 1)[1].strip()}")
                elif "Mean observations per image:" in line:
                    lines.append(f"  Mean observations/image: {line.split(':', 1)[1].strip()}")
                elif "Mean reprojection error:" in line:
                    lines.append(f"  Mean reprojection error: {line.split(':', 1)[1].strip()}")

    if colmap_log.is_file():
        metrics = parse_log_metrics(colmap_log)
        lines.extend(["", "[Timing]"])
        for index, label in enumerate(["Feature extraction", "Feature matching", "Mapper", "Point filtering"]):
            elapsed = metrics["elapsed_minutes"]
            assert isinstance(elapsed, list)
            if index < len(elapsed):
                lines.append(f"  {label}: {elapsed[index]:.3f} min")
        lines.append(f"  Total: {float(metrics['total_minutes']):.3f} min")

        if ran_point_filtering and metrics["filtered_observations"] is not None:
            lines.extend(["", "[Point filtering]", f"  Filtered observations: {metrics['filtered_observations']}"])

        if metrics["final_cost_px"] is not None:
            lines.extend(["", "[Bundle adjustment]"])
            lines.append(f"  Final cost: {float(metrics['final_cost_px']):.4f} px")
            if metrics["initial_cost_px"] is not None:
                lines.append(f"  Initial cost: {float(metrics['initial_cost_px']):.4f} px")
            if metrics["residuals"] is not None:
                lines.append(f"  Residuals: {int(metrics['residuals'])}")
            if metrics["ba_iterations"] is not None:
                lines.append(f"  Iterations: {int(metrics['ba_iterations'])}")

    lines.extend(
        [
            "",
            "[Outputs]",
            f"  Scene:  {scene}",
            f"  sparse: {sparse_0}",
            f"  Log:    {colmap_log}",
            "=" * 56,
            "",
        ]
    )
    print("\n".join(lines))


def build_feature_extractor_cmd(
    paths: ScenePaths,
    config: FeatureConfig,
) -> list[str]:
    cmd = [
        COLMAP_BIN,
        "feature_extractor",
        "--database_path",
        str(paths.db_path),
        "--image_path",
        str(paths.image_path),
        "--ImageReader.camera_model",
        config.camera_model,
        "--SiftExtraction.max_num_features",
        config.max_features,
        "--SiftExtraction.peak_threshold",
        config.peak_threshold,
        "--SiftExtraction.estimate_affine_shape",
        str(config.estimate_affine_shape),
        "--SiftExtraction.domain_size_pooling",
        str(config.domain_size_pooling),
    ]
    if config.use_camera_intrinsics:
        if not config.camera_params:
            raise ValueError("use_camera_intrinsics=True requires camera_params")
        cmd.extend(["--ImageReader.camera_params", config.camera_params])
    cmd.extend(["--ImageReader.single_camera_per_folder", "1"])
    if config.num_threads > 0:
        cmd.extend(["--FeatureExtraction.num_threads", str(config.num_threads)])
    if config.max_image_size is not None and config.max_image_size > 0:
        cmd.extend(["--FeatureExtraction.max_image_size", str(config.max_image_size)])
    return cmd


def build_sequential_matcher_cmd(paths: ScenePaths, feature: FeatureConfig, overlap: int) -> list[str]:
    return [
        COLMAP_BIN,
        "sequential_matcher",
        "--database_path",
        str(paths.db_path),
        "--SequentialMatching.overlap",
        str(overlap),
        "--SequentialMatching.quadratic_overlap",
        "1",
        "--FeatureMatching.guided_matching",
        str(feature.guided_matching),
    ]


def build_matches_importer_cmd(paths: ScenePaths) -> list[str]:
    return [
        COLMAP_BIN,
        "matches_importer",
        "--database_path",
        str(paths.db_path),
        "--match_list_path",
        str(paths.match_list_path),
    ]


def build_mapper_cmd(paths: ScenePaths, config: MapperConfig) -> list[str]:
    return [
        COLMAP_BIN,
        "mapper",
        "--database_path",
        str(paths.db_path),
        "--image_path",
        str(paths.image_path),
        "--output_path",
        str(paths.sparse_path),
        *mapper_args(config),
    ]


def prepare_workspace(paths: ScenePaths, config: OutputConfig) -> None:
    paths.sparse_path.mkdir(parents=True, exist_ok=True)
    if config.rebuild_database:
        cleanup_database_files(paths.db_path)
    if config.clean_sparse_outputs:
        cleanup_sparse_outputs(paths.sparse_path)
    paths.colmap_log.parent.mkdir(parents=True, exist_ok=True)
    paths.colmap_log.write_text("", encoding="utf-8")


def require_camera_folders(
    image_path: Path,
    extensions: tuple[str, ...],
    mode: str,
) -> list[str]:
    if not image_path.is_dir():
        raise FileNotFoundError(f"image dir not found: {image_path}")

    camera_folders = discover_camera_folders(image_path, extensions)
    if len(camera_folders) < 2:
        raise ValueError(
            f"{mode} mode requires at least 2 image subfolders under {image_path}, "
            f"found: {camera_folders}"
        )
    return camera_folders


def select_primary_component(
    sparse_path: Path,
    config: OutputConfig,
    title: str,
) -> Path:
    components = collect_sparse_components(sparse_path)
    print_component_table(components, title)
    sparse_0 = sparse_path / "0"

    if config.promote_largest_component_to_zero and components:
        sparse_0 = promote_largest_component_to_zero(sparse_path, components)
        components = collect_sparse_components(sparse_path)
        print_component_table(components, "      Promoted primary component:")

    return sparse_0


def merge_remaining_components(
    sparse_path: Path,
    sparse_0: Path,
    config: OutputConfig,
    colmap_log: Path,
    quiet_filter: bool,
    step_label: str,
) -> None:
    remaining = [info for info in collect_sparse_components(sparse_path) if Path(info["path"]) != sparse_0]
    if not remaining:
        print(f"{step_label} No extra sparse components.")
        return

    if not config.merge_additional_components:
        print(f"{step_label} Extra sparse components detected, merge disabled.")
        print_component_table(remaining, "      Left as extra components:")
        return

    merged_any = False
    for info in remaining:
        component_path = Path(info["path"])
        component_index = int(info["index"])
        component_images = int(info["num_images"])

        if component_images < config.merge_min_component_images:
            print(
                f"{step_label} Skip sparse/{component_index}: "
                f"only {component_images} images (< {config.merge_min_component_images})."
            )
            continue

        print(f"{step_label} Merging sparse/{component_index} into sparse/0...")
        merged_tmp = sparse_path / "merged_tmp"
        if merged_tmp.exists():
            remove_path(merged_tmp)
        merged_tmp.mkdir(parents=True, exist_ok=True)

        cmd = [
            COLMAP_BIN,
            "model_merger",
            "--input_path1",
            str(sparse_0),
            "--input_path2",
            str(component_path),
            "--output_path",
            str(merged_tmp),
        ]
        if run_cmd(cmd, colmap_log, quiet_filter) != 0:
            print(f"Warning: model_merger failed for sparse/{component_index}, skipping.", file=sys.stderr)
            remove_path(merged_tmp)
            continue

        if (merged_tmp / "cameras.bin").exists() or (merged_tmp / "cameras.txt").exists():
            remove_path(sparse_0)
            shutil.move(str(merged_tmp), str(sparse_0))
            merged_any = True
        else:
            remove_path(merged_tmp)

    if not merged_any:
        print(f"{step_label} No additional sparse components were merged.")


def run_rig_bundle_adjustment(
    sparse_0: Path,
    rig_config: Path,
    colmap_log: Path,
    quiet_filter: bool,
) -> int:
    cmd = [
        COLMAP_BIN,
        "rig_bundle_adjuster",
        "--input_path",
        str(sparse_0),
        "--output_path",
        str(sparse_0),
        "--rig_config_path",
        str(rig_config),
        "--RigBundleAdjustment.refine_relative_poses",
        "0",
    ]
    return run_cmd(cmd, colmap_log, quiet_filter)


def run_multi(
    paths: ScenePaths,
    feature: FeatureConfig,
    match: MatchConfig,
    mapper: MapperConfig,
    output: OutputConfig,
    quiet_filter: bool,
    rig_config: Path | None,
) -> int:
    try:
        camera_folders = require_camera_folders(paths.image_path, match.image_extensions, "multi")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    total_images = count_images(paths.image_path, match.image_extensions)
    prepare_workspace(paths, output)

    print(f"\n====> Multi-cam mode: {len(camera_folders)} cameras {camera_folders}\n")

    print("[1/6] Extracting features...")
    if SKIP_FEATURE_EXTRACTION:
        print("      skipped, reusing existing database features")
    else:
        if run_cmd(build_feature_extractor_cmd(paths, feature), paths.colmap_log, quiet_filter) != 0:
            return 1

    print("[2/6] Sequential matching...")
    if SKIP_SEQUENTIAL_MATCHING:
        print("      skipped, reusing existing database matches")
    else:
        if run_cmd(
            build_sequential_matcher_cmd(paths, feature, match.sequential_overlap),
            paths.colmap_log,
            quiet_filter,
        ) != 0:
            return 1

    camera_rows = collect_camera_images(
        paths.image_path,
        camera_folders,
        match.image_extensions,
        match.frame_regex,
    )
    if SKIP_CROSS_CAMERA_MATCHING:
        print("      cross-camera matching skipped, reusing existing imported matches")
    else:
        pairs = build_cross_camera_pairs(
            camera_rows,
            match.cross_camera_match_mode,
            match.cross_camera_max_frame_delta,
            match.cross_camera_max_neighbors,
            match.cross_camera_bidirectional,
        )
        write_match_list(paths.match_list_path, pairs)
        total_with_index = sum(1 for rows in camera_rows.values() for _, idx in rows if idx is not None)
        total_all = sum(len(rows) for rows in camera_rows.values())
        print(
            "      cross_pairs={}, mode={}, max_dt={}, max_neighbors={}, bidirectional={}, frame_idx={}/{}".format(
                len(pairs),
                match.cross_camera_match_mode,
                match.cross_camera_max_frame_delta,
                match.cross_camera_max_neighbors,
                int(match.cross_camera_bidirectional),
                total_with_index,
                total_all,
            )
        )
        if run_cmd(build_matches_importer_cmd(paths), paths.colmap_log, quiet_filter) != 0:
            return 1

    print("[3/6] Mapper...")
    if run_cmd(
        build_mapper_cmd(paths, mapper),
        paths.colmap_log,
        quiet_filter,
        mapper_total_images=total_images,
    ) != 0:
        return 1

    sparse_0 = select_primary_component(paths.sparse_path, output, "[3/6] Mapper outputs:")

    if output.run_point_filtering and sparse_0.is_dir():
        print("[4/6] Filtering high-error points...")
        run_point_filtering(
            sparse_0,
            output.point_filter_min_track_len,
            output.point_filter_max_reproj_error,
            paths.colmap_log,
            quiet_filter,
        )
    else:
        print("[4/6] Skip point filtering.")

    if sparse_0.is_dir() and rig_config is not None:
        print("[5/6] Running rig bundle adjustment...")
        if run_rig_bundle_adjustment(sparse_0, rig_config, paths.colmap_log, quiet_filter) != 0:
            return 1
    else:
        print("[5/6] Skip rig bundle adjustment.")

    merge_remaining_components(
        paths.sparse_path,
        sparse_0,
        output,
        paths.colmap_log,
        quiet_filter,
        "[6/6]",
    )

    print_run_summary(paths.scene, sparse_0, paths.colmap_log, output.run_point_filtering)
    print("====> Multi-cam COLMAP process complete!")
    return 0


def build_paths(scene: Path) -> ScenePaths:
    if LOG_PATH is not None:
        colmap_log = Path(LOG_PATH).resolve()
    else:
        colmap_log = scene / "colmap.log"
    return ScenePaths(
        scene=scene,
        image_path=scene / "images",
        db_path=scene / "database.db",
        sparse_path=scene / "sparse",
        match_list_path=scene / "image_matches.txt",
        colmap_log=colmap_log,
    )


def build_feature_config() -> FeatureConfig:
    use_intrinsics = bool(USE_CAMERA_INTRINSICS)
    params = CAMERA_PARAMS.strip() if use_intrinsics else None
    if use_intrinsics and not params:
        raise ValueError("USE_CAMERA_INTRINSICS=True requires non-empty CAMERA_PARAMS")
    return FeatureConfig(
        use_camera_intrinsics=use_intrinsics,
        camera_model=CAMERA_MODEL,
        camera_params=params,
        max_features=MAX_FEATURES,
        peak_threshold=SIFT_PEAK_THRESHOLD,
        estimate_affine_shape=SIFT_ESTIMATE_AFFINE_SHAPE,
        domain_size_pooling=SIFT_DOMAIN_SIZE_POOLING,
        guided_matching=SIFT_GUIDED_MATCHING,
        num_threads=SIFT_NUM_THREADS,
        max_image_size=SIFT_MAX_IMAGE_SIZE,
    )


def build_match_config() -> MatchConfig:
    return MatchConfig(
        image_extensions=tuple(IMAGE_EXTENSIONS),
        sequential_overlap=SEQUENTIAL_OVERLAP,
        frame_regex=CROSS_CAMERA_FRAME_REGEX,
        cross_camera_match_mode=CROSS_CAMERA_MATCH_MODE,
        cross_camera_max_frame_delta=CROSS_CAMERA_MAX_FRAME_DELTA,
        cross_camera_max_neighbors=CROSS_CAMERA_MAX_NEIGHBORS_PER_IMAGE,
        cross_camera_bidirectional=CROSS_CAMERA_BIDIRECTIONAL,
    )


def build_mapper_config() -> MapperConfig:
    use_intrinsics = bool(USE_CAMERA_INTRINSICS)
    return MapperConfig(
        multiple_models=MAPPER_MULTIPLE_MODELS,
        min_model_size=MAPPER_MIN_MODEL_SIZE,
        tri_ignore_two_view_tracks=MAPPER_TRI_IGNORE_TWO_VIEW_TRACKS,
        min_depth=MAPPER_MIN_DEPTH,
        max_depth=MAPPER_MAX_DEPTH,
        filter_max_reproj_error=MAPPER_FILTER_MAX_REPROJ_ERROR,
        tri_min_ray_angle=MAPPER_TRI_MIN_RAY_ANGLE,
        ba_refine_focal_length=(
            int(MAPPER_BA_REFINE_FOCAL_LENGTH)
            if MAPPER_BA_REFINE_FOCAL_LENGTH is not None
            else 1
        ),
        ba_refine_principal_point=(
            int(MAPPER_BA_REFINE_PRINCIPAL_POINT)
            if MAPPER_BA_REFINE_PRINCIPAL_POINT is not None
            else 1
        ),
        ba_refine_extra_params=(
            int(MAPPER_BA_REFINE_EXTRA_PARAMS)
            if MAPPER_BA_REFINE_EXTRA_PARAMS is not None
            else (0 if use_intrinsics else 1)
        ),
        ba_local_max_num_iterations=MAPPER_BA_LOCAL_MAX_NUM_ITERATIONS,
        ba_global_max_num_iterations=MAPPER_BA_GLOBAL_MAX_NUM_ITERATIONS,
        ba_global_max_refinements=MAPPER_BA_GLOBAL_MAX_REFINEMENTS,
        ba_local_max_refinements=MAPPER_BA_LOCAL_MAX_REFINEMENTS,
        ba_global_frames_ratio=MAPPER_BA_GLOBAL_FRAMES_RATIO,
        ba_global_points_ratio=MAPPER_BA_GLOBAL_POINTS_RATIO,
        ba_use_gpu=MAPPER_BA_USE_GPU,
        ba_gpu_index=MAPPER_BA_GPU_INDEX,
    )


def build_output_config() -> OutputConfig:
    return OutputConfig(
        rebuild_database=REBUILD_DATABASE,
        clean_sparse_outputs=CLEAN_SPARSE_OUTPUTS,
        run_point_filtering=RUN_POINT_FILTERING,
        point_filter_min_track_len=POINT_FILTER_MIN_TRACK_LEN,
        point_filter_max_reproj_error=POINT_FILTER_MAX_REPROJ_ERROR,
        promote_largest_component_to_zero=PROMOTE_LARGEST_COMPONENT_TO_ZERO,
        merge_additional_components=MERGE_ADDITIONAL_COMPONENTS,
        merge_min_component_images=MERGE_MIN_COMPONENT_IMAGES,
    )


def resolve_rig_config(scene: Path) -> Path | None:
    if not USE_RIG_EXTRINSICS:
        return None
    candidate = scene / "rig_config.json"
    if not candidate.is_file():
        print(f"Error: USE_RIG_EXTRINSICS=True but rig config not found: {candidate}", file=sys.stderr)
        return None
    return candidate


def main() -> int:
    scene = Path(SCENE).resolve()
    quiet_filter = not NO_FILTER_LOG
    paths = build_paths(scene)
    try:
        feature = build_feature_config()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    intrinsics_mode = "fixed intrinsics" if feature.use_camera_intrinsics else "estimated intrinsics"
    print(f"\n====> Camera intrinsics: {intrinsics_mode} (model={feature.camera_model})\n")
    match = build_match_config()
    mapper = build_mapper_config()
    output = build_output_config()

    rig_config = resolve_rig_config(scene)
    if USE_RIG_EXTRINSICS and rig_config is None:
        return 1

    return run_multi(paths, feature, match, mapper, output, quiet_filter, rig_config)


if __name__ == "__main__":
    sys.exit(main())
