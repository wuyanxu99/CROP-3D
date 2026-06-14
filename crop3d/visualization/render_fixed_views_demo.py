from __future__ import annotations

import json
import math
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys

import mediapy as media
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TWODGS_ROOT = PROJECT_ROOT / "crop3d" / "reconstruction" / "twodgs"
if str(TWODGS_ROOT) not in sys.path:
    sys.path.insert(0, str(TWODGS_ROOT))

from arguments import PipelineParams
from gaussian_renderer import render
from scene.colmap_loader import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)
from scene.gaussian_model import GaussianModel
from utils.colmap_paths import image_name_from_colmap_rel
from utils.graphics_utils import focal2fov, getProjectionMatrix
from utils.render_utils import save_img_u8
from utils.system_utils import searchForMaxIteration


# ============================================================================
# User configuration
# For day-to-day use, only edit this block.
# Run with: python render_fixed_views.py
# ============================================================================

# Adjust this section as needed.
# Each entry should look like "<scene_name>/<run_name>", for example
# "4.13RLU/04-20_13:12".
MODEL_RUNS = [
    "demo_scene/demo_run",
]

# Load the most recently saved iteration by default; set a concrete value to
# render a specific checkpoint.
MODEL_DEFAULT_ITERATION = -1

# Export mode:
#   "images" exports fixed-view images only
#   "videos" exports trajectory videos only
#   "both"   exports both images and videos
EXPORT_MODE = "images"


def _normalize_model_run(model_run: str) -> str:
    normalized = str(model_run).replace("\\", "/").strip().strip("/")
    if "/" not in normalized:
        raise ValueError(
            f"MODEL_RUNS paths should look like '4.13RLU/04-20_13:12'; got: {model_run}"
        )
    return normalized


def _build_model_spec_from_run(model_run: str) -> dict:
    normalized = _normalize_model_run(model_run)
    return {
        "name": normalized.replace("/", "_"),
        "model_path": f"./outputs/data/{normalized}",
        "iteration": MODEL_DEFAULT_ITERATION,
    }


def _default_output_root_name(model_runs: list[str]) -> str:
    normalized = [_normalize_model_run(model_run) for model_run in model_runs]
    if len(normalized) == 1:
        return normalized[0].replace("/", "_")
    return "__vs__".join(run.replace("/", "_") for run in normalized)


MODEL_SPECS = [_build_model_spec_from_run(model_run) for model_run in MODEL_RUNS]

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "fixed_view_renders" / _default_output_root_name(MODEL_RUNS)

# If None, read source_path from MODEL_SPECS[0]/cfg_args.
# The default is inferred as "./data/<scene_name>".
REFERENCE_SOURCE_PATH = f"./examples/reconstruction_demo/{_normalize_model_run(MODEL_RUNS[0]).split('/', 1)[0]}"

# If None, use the background color from the training configuration.
BACKGROUND_COLOR_OVERRIDE = None  # Example: [0.0, 0.0, 0.0]

# Pipeline settings used by gaussian_renderer.render(...)
PIPELINE_DEPTH_RATIO = 0.0

# If not None, all COLMAP-selected views are resized to this output
# resolution. Keep the aspect ratio consistent with the original cameras.
OUTPUT_RESOLUTION_OVERRIDE = None  # Example: (1920, 1080)

# Default COLMAP-recommended views for multi-model comparison.
# Disabled for now; we render only the three manually configured images below.
USE_COLMAP_RECOMMENDED_VIEWS = False
RECOMMENDED_VIEW_LABELS = ["start", "quarter", "middle", "three_quarter", "end"]

# Default intrinsics used for manual look-at views.
# Alternative form:
# DEFAULT_MANUAL_INTRINSICS = {
#     "intrinsics_from": {"kind": "recommended", "label": "middle"}
# }
DEFAULT_MANUAL_INTRINSICS = {
    "width": 2560,
    "height": 1440,
    "fov_y_deg": 40.0,
}

# Scene overlay markers.
# These annotations are composited after rendering and do not affect the
# Gaussian rendering itself.
# The coordinate semantics match the AprilTag alignment script:
#   +X points right
#   +Y points inward
#   +Z points upward
SCENE_OVERLAY = {
    "enabled": False,
    "apply_to_images": False,
    "apply_to_videos": False,
    "origin_cross": {
        "enabled": False,
        "half_length": 0.10,
        "color": [255, 230, 0],
        "line_width": 4,
    },
    "rgb_axes": {
        "enabled": False,
        "line_width": 5,
        "x_length": 0.40,
        "y_length": 0.40,
        "z_length": 0.40,
        "x_color": [255, 0, 0],
        "y_color": [0, 255, 0],
        "z_color": [64, 160, 255],
    },
    "ground_grid": {
        "enabled": False,
        "z": 0.0,
        "x_min": -0.5,
        "x_max": 8.5,
        "y_min": -1.5,
        "y_max": 1.5,
        "spacing": 0.5,
        "major_every": 4,
        "minor_color": [150, 150, 150],
        "major_color": [255, 255, 255],
        "minor_width": 1,
        "major_width": 2,
    },
}

# Fixed image view configuration. Supported kinds:
# 1) {"name": "...", "kind": "colmap_index", "index": 10}
# 2) {"name": "...", "kind": "colmap_name", "image_name": "cam_a/frame_0001"}
# 3) {"name": "...", "kind": "recommended", "label": "middle"}
# 4) {"name": "...", "kind": "look_at", "position": [...], "look_at": [...], "up": [...]}
#
# For kind="look_at", you can additionally override intrinsics with:
# width / height / fov_y_deg, or fx / fy, or intrinsics_from.
IMAGE_VIEW_SPECS = [
    # 0. Helper example: look toward the origin from negative y to verify the
    # origin placement.
    # {
    #     "name": "00_origin_from_neg_y",
    #     "kind": "look_at",
    #     "position": [0.0000, -2.0000, 0.0000],
    #     "look_at": [0.0000, 0.0000, 0.0000],
    #     "up": [0.0, 0.0, 1.0],
    #     "width": 1920,
    #     "height": 1080,
    #     "fov_y_deg": 60.0,
    # },
    # 1. Fixed x=3.84, updated for the latest calibration frame
    # (+X right / +Y inward / +Z upward), medium-far view.
    {
        "name": "01_overall_far",
        "kind": "look_at",
        "position": [3.700, -5.0000, 2.0000],
        "look_at": [3.700, 0.1500, 0.5000],
        "up": [0.0, 0.0, 1.0],
        "width": 2560,
        "height": 1440,
        "fov_y_deg": 45.0,
    },
    # 2. Keep the same orientation and move closer to a medium view.
    {
        "name": "02_middle_section",
        "kind": "look_at",
        "position": [3.700, -1.0000, 1.000],
        "look_at": [3.700, 0.1500, 0.5000],
        "up": [0.0, 0.0, 1.0],
        "width": 2560,
        "height": 1440,
        "fov_y_deg": 40.0,
    },
    # 3. Keep the same orientation and move into a close view.
    {
        "name": "03_close_detail",
        "kind": "look_at",
        "position": [3.700, -0.5000, 0.8500],
        "look_at": [3.700, 0.1500, 0.5000],
        "up": [0.0, 0.0, 1.0],
        "width": 2560,
        "height": 1440,
        "fov_y_deg": 35.0,
    },
    # 4. Continue moving closer to cover an ultra-close view around one pot.
    {
        "name": "04_ultra_close_pot",
        "kind": "look_at",
        "position": [3.700, -0.2200, 0.7000],
        "look_at": [3.700, 0.1500, 0.5400],
        "up": [0.0, 0.0, 1.0],
        "width": 2560,
        "height": 1440,
        "fov_y_deg": 35.0,
    },
]

# Linear video trajectory configuration.
# 1) If only look_at is specified, the camera moves linearly from
#    start_position to end_position while always looking at the fixed target.
# 2) If start_look_at / end_look_at are specified, both camera position and
#    target are interpolated linearly, which better preserves the viewing angle.
VIDEO_TRAJECTORIES = [
    # {
    #     "enabled": True,
    #     "name": "x_sweep_0_to_8",
    #     "start_position": [0.0, -0.5000, 0.8500],
    #     "end_position": [8.0, -0.5000, 0.8500],
    #     "start_look_at": [0.0, 0.1500, 0.5000],
    #     "end_look_at": [8.0, 0.1500, 0.5000],
    #     "up": [0.0, 0.0, 1.0],
    #     "width": 1920,
    #     "height": 1080,
    #     "fov_y_deg": 35.0,
    #     "num_frames": 180,
    #     "fps": 30,
    #     "save_frames": True,
    # },
]


@dataclass
class ReferenceCamera:
    order_index: int
    image_name: str
    width: int
    height: int
    fx: float
    fy: float
    fov_x: float
    fov_y: float
    c2w: np.ndarray
    position: np.ndarray
    right: np.ndarray
    down: np.ndarray
    forward: np.ndarray


def ensure_models_configured() -> None:
    if MODEL_SPECS:
        return
    raise ValueError("MODEL_SPECS is empty. Please fill at least one trained model path at the top of render_fixed_views.py.")


def as_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()


def load_cfg_args(model_path: Path) -> Namespace:
    cfg_path = model_path / "cfg_args"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing cfg_args: {cfg_path}")
    text = cfg_path.read_text(encoding="utf-8")
    cfg = eval(text, {"Namespace": Namespace})
    if not isinstance(cfg, Namespace):
        raise TypeError(f"cfg_args in {cfg_path} did not evaluate to argparse.Namespace")
    cfg.model_path = str(model_path)
    return cfg


def sanitize_name(name: str) -> str:
    safe = name.strip().replace("\\", "_").replace("/", "_").replace(" ", "_")
    return safe or "unnamed"


def normalize_export_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    allowed_modes = {"images", "videos", "both"}
    if normalized not in allowed_modes:
        raise ValueError(
            f"Unsupported EXPORT_MODE: {mode}. Expected one of {sorted(allowed_modes)}."
        )
    return normalized


def load_pipeline() -> object:
    parser = ArgumentParser()
    params = PipelineParams(parser)
    pipe = params.extract(parser.parse_args([]))
    pipe.depth_ratio = PIPELINE_DEPTH_RATIO
    return pipe


def resolve_iteration(model_path: Path, model_spec: dict) -> int:
    requested = int(model_spec.get("iteration", -1))
    point_cloud_root = model_path / "point_cloud"
    if not point_cloud_root.is_dir():
        raise FileNotFoundError(f"Missing point_cloud directory: {point_cloud_root}")
    if requested == -1:
        return int(searchForMaxIteration(str(point_cloud_root)))
    return requested


def point_cloud_ply_path(model_path: Path, iteration: int) -> Path:
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply_path.is_file():
        raise FileNotFoundError(f"Missing trained point cloud: {ply_path}")
    return ply_path


def load_gaussians(model_path: Path, cfg: Namespace, iteration: int) -> GaussianModel:
    gaussians = GaussianModel(int(getattr(cfg, "sh_degree", 3)))
    gaussians.load_ply(str(point_cloud_ply_path(model_path, iteration)))
    return gaussians


def get_background_tensor(cfg: Namespace, device: torch.device) -> torch.Tensor:
    if BACKGROUND_COLOR_OVERRIDE is not None:
        bg = BACKGROUND_COLOR_OVERRIDE
    else:
        use_white = bool(getattr(cfg, "white_background", False))
        bg = [1.0, 1.0, 1.0] if use_white else [0.0, 0.0, 0.0]
    return torch.tensor(bg, dtype=torch.float32, device=device)


def camera_model_to_focals(model_name: str, params: np.ndarray) -> tuple[float, float]:
    if model_name in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "FOV"}:
        focal = float(params[0])
        return focal, focal
    if model_name in {"PINHOLE", "OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE", "THIN_PRISM_FISHEYE"}:
        return float(params[0]), float(params[1])
    raise ValueError(f"Unsupported COLMAP camera model for render-fixed-views: {model_name}")


def natural_sort_key(text: str) -> tuple:
    parts = []
    token = ""
    is_digit = None
    for ch in text:
        if ch.isdigit():
            if is_digit is False:
                parts.append((0, token.lower()))
                token = ""
            token += ch
            is_digit = True
        else:
            if is_digit is True:
                parts.append((1, int(token)))
                token = ""
            token += ch
            is_digit = False
    if token:
        if is_digit:
            parts.append((1, int(token)))
        else:
            parts.append((0, token.lower()))
    return tuple(parts)


def sort_image_names_frame_major(image_names: list[str]) -> list[str]:
    if not image_names:
        return image_names

    parsed = []
    for image_name in image_names:
        normalized = image_name.replace("\\", "/").strip().lstrip("/")
        if "/" not in normalized:
            return sorted(image_names, key=natural_sort_key)
        parent, stem = normalized.rsplit("/", 1)
        parsed.append((image_name, parent, stem))

    parents = {parent for _, parent, _ in parsed}
    if len(parents) < 2:
        return sorted(image_names, key=natural_sort_key)

    cameras_sorted = sorted(parents, key=natural_sort_key)
    stems_sorted = sorted({stem for _, _, stem in parsed}, key=natural_sort_key)
    bucket = {(parent, stem): image_name for image_name, parent, stem in parsed}

    ordered = []
    for stem in stems_sorted:
        for parent in cameras_sorted:
            key = (parent, stem)
            if key in bucket:
                ordered.append(bucket[key])

    if len(ordered) != len(image_names):
        return sorted(image_names, key=natural_sort_key)
    return ordered


def build_reference_camera(
    order_index: int,
    image_name: str,
    width: int,
    height: int,
    fx: float,
    fy: float,
    c2w: np.ndarray,
) -> ReferenceCamera:
    c2w = np.asarray(c2w, dtype=np.float32)
    position = c2w[:3, 3].astype(np.float32)
    right = c2w[:3, 0].astype(np.float32)
    down = c2w[:3, 1].astype(np.float32)
    forward = c2w[:3, 2].astype(np.float32)
    return ReferenceCamera(
        order_index=order_index,
        image_name=image_name,
        width=int(width),
        height=int(height),
        fx=float(fx),
        fy=float(fy),
        fov_x=float(focal2fov(float(fx), int(width))),
        fov_y=float(focal2fov(float(fy), int(height))),
        c2w=c2w,
        position=position,
        right=right,
        down=down,
        forward=forward,
    )


def load_reference_cameras_from_colmap(scene_root: Path) -> list[ReferenceCamera]:
    sparse_root = scene_root / "sparse" / "0"
    if not sparse_root.is_dir():
        raise FileNotFoundError(f"Missing COLMAP sparse model: {sparse_root}")

    images_bin = sparse_root / "images.bin"
    cameras_bin = sparse_root / "cameras.bin"
    images_txt = sparse_root / "images.txt"
    cameras_txt = sparse_root / "cameras.txt"

    if images_bin.is_file() and cameras_bin.is_file():
        extrinsics = read_extrinsics_binary(str(images_bin))
        intrinsics = read_intrinsics_binary(str(cameras_bin))
    elif images_txt.is_file() and cameras_txt.is_file():
        extrinsics = read_extrinsics_text(str(images_txt))
        intrinsics = read_intrinsics_text(str(cameras_txt))
    else:
        raise FileNotFoundError(f"Could not find COLMAP images/cameras files under {sparse_root}")

    unsorted_records = {}
    for _, extr in extrinsics.items():
        intr = intrinsics[extr.camera_id]
        fx, fy = camera_model_to_focals(intr.model, intr.params)

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = qvec2rotmat(extr.qvec).astype(np.float32)
        w2c[:3, 3] = np.asarray(extr.tvec, dtype=np.float32)
        c2w = np.linalg.inv(w2c).astype(np.float32)

        image_name = image_name_from_colmap_rel(extr.name)
        unsorted_records[image_name] = build_reference_camera(
            order_index=-1,
            image_name=image_name,
            width=int(intr.width),
            height=int(intr.height),
            fx=fx,
            fy=fy,
            c2w=c2w,
        )

    ordered_names = sort_image_names_frame_major(list(unsorted_records.keys()))
    ordered = []
    for idx, image_name in enumerate(ordered_names):
        camera = unsorted_records[image_name]
        ordered.append(
            build_reference_camera(
                order_index=idx,
                image_name=camera.image_name,
                width=camera.width,
                height=camera.height,
                fx=camera.fx,
                fy=camera.fy,
                c2w=camera.c2w,
            )
        )
    return ordered


def load_reference_cameras_from_json(cameras_json_path: Path) -> list[ReferenceCamera]:
    if not cameras_json_path.is_file():
        raise FileNotFoundError(f"Missing cameras.json fallback: {cameras_json_path}")
    payload = json.loads(cameras_json_path.read_text(encoding="utf-8"))

    unsorted_records = {}
    for entry in payload:
        image_name = str(entry["img_name"]).replace("\\", "/")
        rotation = np.asarray(entry["rotation"], dtype=np.float32)
        position = np.asarray(entry["position"], dtype=np.float32)
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = rotation
        c2w[:3, 3] = position
        unsorted_records[image_name] = build_reference_camera(
            order_index=-1,
            image_name=image_name,
            width=int(entry["width"]),
            height=int(entry["height"]),
            fx=float(entry["fx"]),
            fy=float(entry["fy"]),
            c2w=c2w,
        )

    ordered_names = sort_image_names_frame_major(list(unsorted_records.keys()))
    ordered = []
    for idx, image_name in enumerate(ordered_names):
        camera = unsorted_records[image_name]
        ordered.append(
            build_reference_camera(
                order_index=idx,
                image_name=camera.image_name,
                width=camera.width,
                height=camera.height,
                fx=camera.fx,
                fy=camera.fy,
                c2w=camera.c2w,
            )
        )
    return ordered


def load_reference_cameras(reference_cfg: Namespace, reference_model_path: Path) -> list[ReferenceCamera]:
    source_override = REFERENCE_SOURCE_PATH
    if source_override is not None:
        source_path = as_path(source_override)
    else:
        source_path = as_path(getattr(reference_cfg, "source_path"))

    try:
        cameras = load_reference_cameras_from_colmap(source_path)
        print(f"Loaded {len(cameras)} reference cameras from COLMAP: {source_path}")
        return cameras
    except Exception as colmap_error:
        fallback_path = reference_model_path / "cameras.json"
        try:
            cameras = load_reference_cameras_from_json(fallback_path)
            print(
                "COLMAP camera load failed, used cameras.json fallback instead:\n"
                f"  COLMAP error: {colmap_error}\n"
                f"  Fallback: {fallback_path}"
            )
            return cameras
        except Exception as json_error:
            raise RuntimeError(
                "Failed to load reference cameras from both COLMAP and cameras.json.\n"
                f"COLMAP source: {source_path}\n"
                f"COLMAP error: {colmap_error}\n"
                f"cameras.json error: {json_error}"
            ) from json_error


def build_recommended_views(reference_cameras: list[ReferenceCamera]) -> dict[str, ReferenceCamera]:
    if not reference_cameras:
        return {}
    max_index = len(reference_cameras) - 1
    requested = {
        "start": 0,
        "quarter": round(max_index * 0.25),
        "middle": round(max_index * 0.50),
        "three_quarter": round(max_index * 0.75),
        "end": max_index,
    }

    ordered = {}
    used_indices = set()
    for label in RECOMMENDED_VIEW_LABELS:
        if label not in requested:
            raise ValueError(f"Unsupported recommended view label: {label}")
        idx = int(requested[label])
        if idx in used_indices:
            continue
        ordered[label] = reference_cameras[idx]
        used_indices.add(idx)
    return ordered


def reference_camera_to_dict(camera: ReferenceCamera) -> dict:
    return {
        "order_index": camera.order_index,
        "image_name": camera.image_name,
        "width": camera.width,
        "height": camera.height,
        "fx": camera.fx,
        "fy": camera.fy,
        "fov_x_deg": float(np.rad2deg(camera.fov_x)),
        "fov_y_deg": float(np.rad2deg(camera.fov_y)),
        "position": camera.position.tolist(),
        "right": camera.right.tolist(),
        "down": camera.down.tolist(),
        "forward": camera.forward.tolist(),
    }


def write_reference_report(
    output_root: Path,
    reference_cfg: Namespace,
    reference_cameras: list[ReferenceCamera],
    recommended: dict[str, ReferenceCamera],
) -> None:
    report = {
        "reference_source_path": str(REFERENCE_SOURCE_PATH or getattr(reference_cfg, "source_path")),
        "camera_count": len(reference_cameras),
        "recommended_views": {
            label: reference_camera_to_dict(camera) for label, camera in recommended.items()
        },
        "all_cameras": [reference_camera_to_dict(camera) for camera in reference_cameras],
    }
    out_path = output_root / "reference_camera_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved reference camera report: {out_path}")


def normalize_vec(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError("Encountered a near-zero vector while building a camera pose.")
    return vec / norm


def look_at_to_c2w(position: list[float], look_at: list[float], up: list[float]) -> np.ndarray:
    position_np = np.asarray(position, dtype=np.float32)
    look_at_np = np.asarray(look_at, dtype=np.float32)
    up_np = np.asarray(up, dtype=np.float32)

    forward = normalize_vec(look_at_np - position_np)
    right = normalize_vec(np.cross(forward, up_np))
    down = normalize_vec(np.cross(forward, right))

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = down
    c2w[:3, 2] = forward
    c2w[:3, 3] = position_np
    return c2w


def camera_from_c2w(
    view_name: str,
    c2w: np.ndarray,
    width: int,
    height: int,
    fov_x: float,
    fov_y: float,
    device: torch.device,
    znear: float = 0.01,
    zfar: float = 100.0,
) -> SimpleNamespace:
    w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float32))
    world_view_transform = torch.tensor(w2c, dtype=torch.float32, device=device).transpose(0, 1)
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fov_x, fovY=fov_y).transpose(0, 1).to(device)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    return SimpleNamespace(
        uid=-1,
        image_name=view_name,
        image_width=int(width),
        image_height=int(height),
        FoVx=float(fov_x),
        FoVy=float(fov_y),
        znear=float(znear),
        zfar=float(zfar),
        world_view_transform=world_view_transform,
        projection_matrix=projection_matrix,
        full_proj_transform=full_proj_transform,
        camera_center=camera_center,
    )


def find_reference_by_index(reference_cameras: list[ReferenceCamera], index: int) -> ReferenceCamera:
    if index < 0 or index >= len(reference_cameras):
        raise IndexError(f"COLMAP camera index out of range: {index} (total {len(reference_cameras)})")
    return reference_cameras[index]


def normalize_image_name_key(image_name: str) -> str:
    normalized = str(image_name).replace("\\", "/").strip().lstrip("/")
    return str(Path(normalized).with_suffix("").as_posix())


def build_reference_camera_lookup(reference_cameras: list[ReferenceCamera]) -> dict[str, ReferenceCamera]:
    return {normalize_image_name_key(camera.image_name): camera for camera in reference_cameras}


def resolve_reference_camera_selector(
    selector: dict,
    reference_cameras: list[ReferenceCamera],
    recommended: dict[str, ReferenceCamera],
    reference_lookup: dict[str, ReferenceCamera],
) -> ReferenceCamera:
    kind = selector.get("kind")
    if kind == "recommended":
        label = selector["label"]
        if label not in recommended:
            raise KeyError(f"Unknown recommended label: {label}")
        return recommended[label]
    if kind == "colmap_index":
        return find_reference_by_index(reference_cameras, int(selector["index"]))
    if kind == "colmap_name":
        key = normalize_image_name_key(selector["image_name"])
        if key not in reference_lookup:
            raise KeyError(f"COLMAP image_name not found in reference cameras: {selector['image_name']}")
        return reference_lookup[key]
    raise ValueError(f"Unsupported selector kind: {kind}")


def resolve_intrinsics(
    spec: dict,
    manual_defaults: dict,
    reference_cameras: list[ReferenceCamera],
    recommended: dict[str, ReferenceCamera],
    reference_lookup: dict[str, ReferenceCamera],
) -> tuple[int, int, float, float]:
    merged = dict(manual_defaults)
    merged.update({k: v for k, v in spec.items() if v is not None})

    intrinsics_selector = merged.get("intrinsics_from")
    base_camera = None
    if intrinsics_selector is not None:
        base_camera = resolve_reference_camera_selector(
            intrinsics_selector,
            reference_cameras,
            recommended,
            reference_lookup,
        )

    width = merged.get("width")
    height = merged.get("height")

    if width is None and base_camera is not None:
        width = base_camera.width
    if height is None and base_camera is not None:
        height = base_camera.height

    if width is None or height is None:
        raise ValueError(f"Missing width/height for manual view spec: {spec}")

    width = int(width)
    height = int(height)

    if "fx" in merged or "fy" in merged:
        fx = float(merged.get("fx", merged.get("fy")))
        fy = float(merged.get("fy", merged.get("fx")))
        if fx is None or fy is None:
            raise ValueError(f"fx/fy must both be resolvable in view spec: {spec}")
        return width, height, float(focal2fov(fx, width)), float(focal2fov(fy, height))

    if base_camera is not None:
        scale_x = width / float(base_camera.width)
        scale_y = height / float(base_camera.height)
        fx = base_camera.fx * scale_x
        fy = base_camera.fy * scale_y
        return width, height, float(focal2fov(fx, width)), float(focal2fov(fy, height))

    fov_y_deg = merged.get("fov_y_deg")
    if fov_y_deg is None:
        raise ValueError(f"Missing fov_y_deg or intrinsics_from in view spec: {spec}")
    fov_y = math.radians(float(fov_y_deg))
    fov_x = 2.0 * math.atan(math.tan(fov_y * 0.5) * (float(width) / float(height)))
    return width, height, fov_x, fov_y


def maybe_rescale_reference_intrinsics(camera: ReferenceCamera) -> tuple[int, int, float, float]:
    if OUTPUT_RESOLUTION_OVERRIDE is None:
        return camera.width, camera.height, camera.fov_x, camera.fov_y

    width, height = OUTPUT_RESOLUTION_OVERRIDE
    width = int(width)
    height = int(height)
    scale_x = width / float(camera.width)
    scale_y = height / float(camera.height)
    fx = camera.fx * scale_x
    fy = camera.fy * scale_y
    return width, height, float(focal2fov(fx, width)), float(focal2fov(fy, height))


def build_image_render_jobs(
    reference_cameras: list[ReferenceCamera],
    recommended: dict[str, ReferenceCamera],
) -> list[dict]:
    jobs = []
    seen_names = set()
    reference_lookup = build_reference_camera_lookup(reference_cameras)

    def append_job(name: str, camera: SimpleNamespace, metadata: dict) -> None:
        clean_name = sanitize_name(name)
        if clean_name in seen_names:
            raise ValueError(f"Duplicate image view name detected: {clean_name}")
        seen_names.add(clean_name)
        jobs.append({"name": clean_name, "camera": camera, "metadata": metadata})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if USE_COLMAP_RECOMMENDED_VIEWS:
        for label, ref_camera in recommended.items():
            width, height, fov_x, fov_y = maybe_rescale_reference_intrinsics(ref_camera)
            camera = camera_from_c2w(
                view_name=f"colmap_{label}",
                c2w=ref_camera.c2w,
                width=width,
                height=height,
                fov_x=fov_x,
                fov_y=fov_y,
                device=device,
            )
            append_job(
                name=f"colmap_{label}",
                camera=camera,
                metadata={
                    "kind": "recommended",
                    "label": label,
                    "reference_image_name": ref_camera.image_name,
                    "reference_order_index": ref_camera.order_index,
                },
            )

    for spec in IMAGE_VIEW_SPECS:
        kind = spec["kind"]
        if kind == "recommended":
            ref_camera = resolve_reference_camera_selector(
                {"kind": "recommended", "label": spec["label"]},
                reference_cameras,
                recommended,
                reference_lookup,
            )
            width, height, fov_x, fov_y = maybe_rescale_reference_intrinsics(ref_camera)
            camera = camera_from_c2w(
                view_name=spec["name"],
                c2w=ref_camera.c2w,
                width=width,
                height=height,
                fov_x=fov_x,
                fov_y=fov_y,
                device=device,
            )
            append_job(
                name=spec["name"],
                camera=camera,
                metadata={
                    "kind": "recommended",
                    "label": spec["label"],
                    "reference_image_name": ref_camera.image_name,
                    "reference_order_index": ref_camera.order_index,
                },
            )
            continue

        if kind in {"colmap_index", "colmap_name"}:
            ref_camera = resolve_reference_camera_selector(
                spec,
                reference_cameras,
                recommended,
                reference_lookup,
            )
            width, height, fov_x, fov_y = maybe_rescale_reference_intrinsics(ref_camera)
            camera = camera_from_c2w(
                view_name=spec["name"],
                c2w=ref_camera.c2w,
                width=width,
                height=height,
                fov_x=fov_x,
                fov_y=fov_y,
                device=device,
            )
            append_job(
                name=spec["name"],
                camera=camera,
                metadata={
                    "kind": kind,
                    "reference_image_name": ref_camera.image_name,
                    "reference_order_index": ref_camera.order_index,
                },
            )
            continue

        if kind == "look_at":
            width, height, fov_x, fov_y = resolve_intrinsics(
                spec,
                DEFAULT_MANUAL_INTRINSICS,
                reference_cameras,
                recommended,
                reference_lookup,
            )
            c2w = look_at_to_c2w(
                position=spec["position"],
                look_at=spec["look_at"],
                up=spec["up"],
            )
            camera = camera_from_c2w(
                view_name=spec["name"],
                c2w=c2w,
                width=width,
                height=height,
                fov_x=fov_x,
                fov_y=fov_y,
                device=device,
            )
            append_job(
                name=spec["name"],
                camera=camera,
                metadata={
                    "kind": "look_at",
                    "position": list(map(float, spec["position"])),
                    "look_at": list(map(float, spec["look_at"])),
                    "up": list(map(float, spec["up"])),
                    "width": width,
                    "height": height,
                    "fov_x_deg": float(np.rad2deg(fov_x)),
                    "fov_y_deg": float(np.rad2deg(fov_y)),
                },
            )
            continue

        raise ValueError(f"Unsupported IMAGE_VIEW_SPECS kind: {kind}")

    return jobs


def build_video_render_jobs() -> list[dict]:
    jobs = []
    for spec in VIDEO_TRAJECTORIES:
        if not spec.get("enabled", True):
            continue
        width = int(spec["width"])
        height = int(spec["height"])
        fov_y = math.radians(float(spec["fov_y_deg"]))
        fov_x = 2.0 * math.atan(math.tan(fov_y * 0.5) * (float(width) / float(height)))
        jobs.append(
            {
                "name": sanitize_name(spec["name"]),
                "start_position": np.asarray(spec["start_position"], dtype=np.float32),
                "end_position": np.asarray(spec["end_position"], dtype=np.float32),
                "look_at": np.asarray(spec["look_at"], dtype=np.float32) if "look_at" in spec else None,
                "start_look_at": np.asarray(spec["start_look_at"], dtype=np.float32) if "start_look_at" in spec else None,
                "end_look_at": np.asarray(spec["end_look_at"], dtype=np.float32) if "end_look_at" in spec else None,
                "up": np.asarray(spec["up"], dtype=np.float32),
                "width": width,
                "height": height,
                "fov_x": fov_x,
                "fov_y": fov_y,
                "num_frames": int(spec["num_frames"]),
                "fps": int(spec["fps"]),
                "save_frames": bool(spec.get("save_frames", True)),
            }
        )
    return jobs


def _normalize_color_uint8(color: list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    arr = [int(round(float(c))) for c in color]
    return tuple(max(0, min(255, c)) for c in arr[:3])


def project_world_points(camera: SimpleNamespace, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float32)
    if points.ndim == 1:
        points = points[None, :]

    full_proj = camera.full_proj_transform.detach().cpu()
    pts = torch.from_numpy(points)
    ones = torch.ones((pts.shape[0], 1), dtype=pts.dtype)
    clip = torch.cat([pts, ones], dim=1) @ full_proj

    w = clip[:, 3].numpy()
    valid = np.abs(w) > 1e-8
    ndc = np.zeros((points.shape[0], 2), dtype=np.float32)
    ndc[valid] = (clip[valid, :2] / clip[valid, 3:4]).numpy()

    x_pix = ((ndc[:, 0] + 1.0) * 0.5) * float(camera.image_width - 1)
    y_pix = ((ndc[:, 1] + 1.0) * 0.5) * float(camera.image_height - 1)
    pixels = np.stack([x_pix, y_pix], axis=1)

    return pixels, w, valid


def _draw_segment_3d(
    draw: ImageDraw.ImageDraw,
    camera: SimpleNamespace,
    p0: np.ndarray,
    p1: np.ndarray,
    color: tuple[int, int, int],
    width: int,
) -> None:
    pixels, w, valid = project_world_points(camera, np.stack([p0, p1], axis=0))
    if not bool(valid[0] and valid[1]):
        return
    if float(w[0]) <= 0.0 or float(w[1]) <= 0.0:
        return
    draw.line(
        [(float(pixels[0, 0]), float(pixels[0, 1])), (float(pixels[1, 0]), float(pixels[1, 1]))],
        fill=color,
        width=max(1, int(width)),
    )


def _iter_ground_grid_segments(grid_cfg: dict) -> list[tuple[np.ndarray, np.ndarray, bool]]:
    spacing = float(grid_cfg["spacing"])
    if spacing <= 0:
        raise ValueError("SCENE_OVERLAY['ground_grid']['spacing'] must be > 0")

    x_min = float(grid_cfg["x_min"])
    x_max = float(grid_cfg["x_max"])
    y_min = float(grid_cfg["y_min"])
    y_max = float(grid_cfg["y_max"])
    z = float(grid_cfg["z"])
    major_every = max(1, int(grid_cfg.get("major_every", 4)))
    eps = spacing * 0.1

    xs = np.arange(x_min, x_max + eps, spacing, dtype=np.float32)
    ys = np.arange(y_min, y_max + eps, spacing, dtype=np.float32)
    segments: list[tuple[np.ndarray, np.ndarray, bool]] = []

    for x in xs:
        grid_idx = int(round(x / spacing))
        is_major = (grid_idx % major_every) == 0
        segments.append(
            (
                np.array([x, y_min, z], dtype=np.float32),
                np.array([x, y_max, z], dtype=np.float32),
                is_major,
            )
        )

    for y in ys:
        grid_idx = int(round(y / spacing))
        is_major = (grid_idx % major_every) == 0
        segments.append(
            (
                np.array([x_min, y, z], dtype=np.float32),
                np.array([x_max, y, z], dtype=np.float32),
                is_major,
            )
        )

    return segments


def apply_scene_overlay(image: np.ndarray, camera: SimpleNamespace) -> np.ndarray:
    if not SCENE_OVERLAY.get("enabled", False):
        return image

    img_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_image = Image.fromarray(img_u8)
    draw = ImageDraw.Draw(pil_image)

    grid_cfg = SCENE_OVERLAY.get("ground_grid", {})
    if grid_cfg.get("enabled", False):
        minor_color = _normalize_color_uint8(grid_cfg.get("minor_color", [160, 160, 160]))
        major_color = _normalize_color_uint8(grid_cfg.get("major_color", [255, 255, 255]))
        minor_width = int(grid_cfg.get("minor_width", 1))
        major_width = int(grid_cfg.get("major_width", 2))
        for p0, p1, is_major in _iter_ground_grid_segments(grid_cfg):
            _draw_segment_3d(
                draw,
                camera,
                p0,
                p1,
                major_color if is_major else minor_color,
                major_width if is_major else minor_width,
            )

    cross_cfg = SCENE_OVERLAY.get("origin_cross", {})
    if cross_cfg.get("enabled", False):
        half = float(cross_cfg.get("half_length", 0.08))
        color = _normalize_color_uint8(cross_cfg.get("color", [255, 230, 0]))
        width = int(cross_cfg.get("line_width", 4))
        segments = [
            (np.array([-half, 0.0, 0.0], dtype=np.float32), np.array([half, 0.0, 0.0], dtype=np.float32)),
            (np.array([0.0, -half, 0.0], dtype=np.float32), np.array([0.0, half, 0.0], dtype=np.float32)),
            (np.array([0.0, 0.0, -half], dtype=np.float32), np.array([0.0, 0.0, half], dtype=np.float32)),
        ]
        for p0, p1 in segments:
            _draw_segment_3d(draw, camera, p0, p1, color, width)

    axes_cfg = SCENE_OVERLAY.get("rgb_axes", {})
    if axes_cfg.get("enabled", False):
        width = int(axes_cfg.get("line_width", 5))
        axis_specs = [
            (np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([float(axes_cfg.get("x_length", 0.4)), 0.0, 0.0], dtype=np.float32), _normalize_color_uint8(axes_cfg.get("x_color", [255, 0, 0]))),
            (np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, float(axes_cfg.get("y_length", 0.4)), 0.0], dtype=np.float32), _normalize_color_uint8(axes_cfg.get("y_color", [0, 255, 0]))),
            (np.array([0.0, 0.0, 0.0], dtype=np.float32), np.array([0.0, 0.0, float(axes_cfg.get("z_length", 0.4))], dtype=np.float32), _normalize_color_uint8(axes_cfg.get("z_color", [64, 160, 255]))),
        ]
        for p0, p1, color in axis_specs:
            _draw_segment_3d(draw, camera, p0, p1, color, width)

    return np.asarray(pil_image).astype(np.float32) / 255.0


@torch.no_grad()
def render_rgb(
    gaussians: GaussianModel,
    pipe: object,
    background: torch.Tensor,
    camera: SimpleNamespace,
) -> np.ndarray:
    render_pkg = render(camera, gaussians, pipe, background)
    rgb = torch.clamp(render_pkg["render"], min=0.0, max=1.0)
    return rgb.permute(1, 2, 0).detach().cpu().numpy()


def save_image_jobs_for_model(
    model_name: str,
    gaussians: GaussianModel,
    pipe: object,
    background: torch.Tensor,
    image_jobs: list[dict],
    output_root: Path,
) -> None:
    if not image_jobs:
        return

    images_root = output_root / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    for job in tqdm(image_jobs, desc=f"Images for {model_name}"):
        image = render_rgb(gaussians, pipe, background, job["camera"])
        if SCENE_OVERLAY.get("enabled", False) and SCENE_OVERLAY.get("apply_to_images", False):
            image = apply_scene_overlay(image, job["camera"])
        output_name = f"{sanitize_name(job['name'])}__{sanitize_name(model_name)}.png"
        save_img_u8(image, str(images_root / output_name))


def save_video_jobs_for_model(
    model_name: str,
    gaussians: GaussianModel,
    pipe: object,
    background: torch.Tensor,
    video_jobs: list[dict],
    output_root: Path,
    device: torch.device,
) -> None:
    if not video_jobs:
        return

    model_video_root = output_root / "videos" / sanitize_name(model_name)
    model_video_root.mkdir(parents=True, exist_ok=True)

    for job in video_jobs:
        video_path = model_video_root / f"{job['name']}.mp4"
        frames_root = model_video_root / f"{job['name']}_frames"
        if job["save_frames"]:
            frames_root.mkdir(parents=True, exist_ok=True)

        print(f"Rendering video '{job['name']}' for model '{model_name}' -> {video_path}")

        with media.VideoWriter(
            str(video_path),
            shape=(job["height"], job["width"]),
            codec="h264",
            fps=job["fps"],
            crf=18,
            input_format="rgb",
        ) as writer:
            for frame_idx in tqdm(range(job["num_frames"]), desc=f"Video {model_name}/{job['name']}"):
                alpha = 0.0 if job["num_frames"] == 1 else frame_idx / float(job["num_frames"] - 1)
                position = (1.0 - alpha) * job["start_position"] + alpha * job["end_position"]
                if job["start_look_at"] is not None and job["end_look_at"] is not None:
                    look_at = (1.0 - alpha) * job["start_look_at"] + alpha * job["end_look_at"]
                elif job["look_at"] is not None:
                    look_at = job["look_at"]
                else:
                    raise ValueError(
                        f"Video trajectory '{job['name']}' must define either look_at or both start_look_at/end_look_at."
                    )
                c2w = look_at_to_c2w(
                    position=position.tolist(),
                    look_at=look_at.tolist(),
                    up=job["up"].tolist(),
                )
                camera = camera_from_c2w(
                    view_name=f"{job['name']}_{frame_idx:05d}",
                    c2w=c2w,
                    width=job["width"],
                    height=job["height"],
                    fov_x=job["fov_x"],
                    fov_y=job["fov_y"],
                    device=device,
                )
                image = render_rgb(gaussians, pipe, background, camera)
                if SCENE_OVERLAY.get("enabled", False) and SCENE_OVERLAY.get("apply_to_videos", False):
                    image = apply_scene_overlay(image, camera)
                frame_u8 = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
                writer.add_image(frame_u8)
                if job["save_frames"]:
                    save_img_u8(image, str(frames_root / f"{frame_idx:05d}.png"))


def write_render_plan(
    output_root: Path,
    export_mode: str,
    image_jobs: list[dict],
    video_jobs: list[dict],
    models: list[dict],
) -> None:
    plan = {
        "models": models,
        "export_mode": export_mode,
        "scene_overlay": SCENE_OVERLAY,
        "image_views": [
            {"name": job["name"], **job["metadata"]} for job in image_jobs
        ],
        "video_jobs": [
            {
                "name": job["name"],
                "start_position": job["start_position"].tolist(),
                "end_position": job["end_position"].tolist(),
                "look_at": job["look_at"].tolist() if job["look_at"] is not None else None,
                "start_look_at": job["start_look_at"].tolist() if job["start_look_at"] is not None else None,
                "end_look_at": job["end_look_at"].tolist() if job["end_look_at"] is not None else None,
                "up": job["up"].tolist(),
                "width": job["width"],
                "height": job["height"],
                "fov_x_deg": float(np.rad2deg(job["fov_x"])),
                "fov_y_deg": float(np.rad2deg(job["fov_y"])),
                "num_frames": job["num_frames"],
                "fps": job["fps"],
                "save_frames": job["save_frames"],
            }
            for job in video_jobs
        ],
    }
    out_path = output_root / "render_plan.json"
    out_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Saved render plan: {out_path}")


def main() -> None:
    ensure_models_configured()
    export_mode = normalize_export_mode(EXPORT_MODE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[Warning] CUDA is not available. Rendering will run on CPU and is likely very slow.")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    normalized_models = []
    for model_spec in MODEL_SPECS:
        name = sanitize_name(model_spec["name"])
        model_path = as_path(model_spec["model_path"])
        normalized_models.append({"name": name, "model_path": str(model_path), "iteration": int(model_spec.get("iteration", -1))})

    reference_model_path = as_path(MODEL_SPECS[0]["model_path"])
    reference_cfg = load_cfg_args(reference_model_path)
    reference_cameras = load_reference_cameras(reference_cfg, reference_model_path)
    recommended = build_recommended_views(reference_cameras)
    write_reference_report(OUTPUT_ROOT, reference_cfg, reference_cameras, recommended)

    image_jobs = build_image_render_jobs(reference_cameras, recommended)
    video_jobs = build_video_render_jobs()
    enabled_image_jobs = image_jobs if export_mode in {"images", "both"} else []
    enabled_video_jobs = video_jobs if export_mode in {"videos", "both"} else []
    write_render_plan(OUTPUT_ROOT, export_mode, enabled_image_jobs, enabled_video_jobs, normalized_models)

    if not enabled_image_jobs and not enabled_video_jobs:
        print(
            f"No render jobs are enabled for EXPORT_MODE='{export_mode}'. "
            "Edit IMAGE_VIEW_SPECS / VIDEO_TRAJECTORIES / USE_COLMAP_RECOMMENDED_VIEWS."
        )
        return

    print("Recommended COLMAP views for fixed comparison:")
    for label, camera in recommended.items():
        print(f"  {label:<14} idx={camera.order_index:<4d} image={camera.image_name}")
    print(f"Export mode: {export_mode}")

    pipe = load_pipeline()

    for model_spec in MODEL_SPECS:
        model_name = sanitize_name(model_spec["name"])
        model_path = as_path(model_spec["model_path"])
        cfg = load_cfg_args(model_path)
        iteration = resolve_iteration(model_path, model_spec)
        print(f"Loading model '{model_name}' from {model_path} at iteration {iteration}")

        gaussians = load_gaussians(model_path, cfg, iteration)
        background = get_background_tensor(cfg, device)

        save_image_jobs_for_model(
            model_name=model_name,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            image_jobs=enabled_image_jobs,
            output_root=OUTPUT_ROOT,
        )

        save_video_jobs_for_model(
            model_name=model_name,
            gaussians=gaussians,
            pipe=pipe,
            background=background,
            video_jobs=enabled_video_jobs,
            output_root=OUTPUT_ROOT,
            device=device,
        )

        del gaussians
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Render finished. Outputs saved under: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
