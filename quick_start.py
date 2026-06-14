#!/usr/bin/env python3
"""One-command multi-camera reconstruction pipeline.

This script orchestrates the existing research workflow without rewriting the
individual reconstruction stages:

1. Prepare a scene folder with ``images/<camera_name>/...``.
2. Run multi-camera COLMAP with fixed intrinsics via ``crop3d/reconstruction/run_colmap_demo.py``.
3. Optionally align the sparse model to AprilTag metric coordinates.
4. Optionally generate DA3 depth priors.
5. Convert COLMAP cameras.bin to PINHOLE for the 2DGS loader.
6. Train 2DGS via ``crop3d/reconstruction/twodgs/train.py``.
7. Optionally run render/metrics.

The heavy algorithms remain in the original scripts. This file only provides a
single, parameterized entry point.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
TWODGS_ROOT = PROJECT_ROOT / "crop3d" / "reconstruction" / "twodgs"
DEFAULT_PYTHON = sys.executable
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
DEFAULT_CAMERA_MODEL = "OPENCV"
DEFAULT_CAMERA_PARAMS = "1407.0,1408.3,986.7009,518.1291,-0.0331,0.0209,0,0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full multi-camera CROP-3D reconstruction workflow from an "
            "input image folder."
        )
    )
    parser.add_argument(
        "--input-images",
        required=True,
        help=(
            "Input multi-camera image folder. It can be a scene root containing "
            "'images/', an 'images/' folder containing camera subfolders, or a "
            "folder whose immediate subfolders are cameras."
        ),
    )
    parser.add_argument(
        "--scene",
        default="",
        help=(
            "Scene root to create/use. Default: data/<input-name>_pipeline, or "
            "the input scene root when --input-images already points to a scene."
        ),
    )
    parser.add_argument("--run-name", default="", help="Output run name. Default: pipeline_<timestamp>.")
    parser.add_argument("--python", default=DEFAULT_PYTHON, help="Python executable used for subprocess stages.")
    parser.add_argument(
        "--image-mode",
        choices=("symlink", "copy", "none"),
        default="copy",
        help=(
            "How to place input images under scene/images. Use copy for COLMAP; "
            "'none' expects scene/images to already exist."
        ),
    )
    parser.add_argument(
        "--overwrite-images",
        action="store_true",
        help="Replace an existing prepared scene/images folder when it is safe to do so.",
    )

    # Pipeline controls.
    parser.add_argument("--dry-run", action="store_true", help="Print the planned steps without executing them.")
    parser.add_argument("--skip-colmap", action="store_true", help="Skip COLMAP sparse reconstruction.")
    parser.add_argument("--with-apriltag", action="store_true", help="Run optional AprilTag metric alignment.")
    parser.add_argument("--with-da3", action="store_true", help="Run optional DA3 depth-prior generation.")
    parser.add_argument("--skip-apriltag", action="store_true", help="Skip AprilTag metric alignment.")
    parser.add_argument("--skip-da3", action="store_true", help="Skip DA3 depth-prior generation.")
    parser.add_argument(
        "--skip-pinhole-conversion",
        action="store_true",
        help="Skip converting COLMAP camera intrinsics to PINHOLE before 2DGS training.",
    )
    parser.add_argument("--skip-train", action="store_true", help="Skip 2DGS training.")
    parser.add_argument("--run-render", action="store_true", help="Run render.py after training.")
    parser.add_argument("--run-metrics", action="store_true", help="Run metrics.py after training.")
    parser.add_argument("--render-mesh", action="store_true", help="Allow render.py to export mesh; default skips mesh.")
    parser.add_argument("--render-path", action="store_true", help="Ask render.py to render a camera path.")
    parser.add_argument(
        "--stop-after",
        choices=("prepare-images", "colmap", "apriltag", "da3", "pinhole-conversion", "train", "render"),
        default="",
        help="Stop after a specific stage.",
    )

    # COLMAP stage.
    parser.add_argument(
        "--colmap-bin",
        default="colmap",
        help="COLMAP executable. If 'colmap' is not on PATH, the current Python env's bin/colmap is tried.",
    )
    parser.add_argument("--camera-model", default=DEFAULT_CAMERA_MODEL)
    parser.add_argument(
        "--camera-params",
        default=DEFAULT_CAMERA_PARAMS,
        help="Shared fixed camera params passed to COLMAP and DA3.",
    )
    parser.add_argument(
        "--use-camera-intrinsics",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use fixed camera intrinsics in COLMAP.",
    )
    parser.add_argument("--sequential-overlap", type=int, default=28)
    parser.add_argument("--cross-camera-max-frame-delta", type=int, default=4)
    parser.add_argument("--cross-camera-max-neighbors", type=int, default=4)
    parser.add_argument("--colmap-max-features", default="", help="Override SIFT max features per image.")
    parser.add_argument("--colmap-max-image-size", type=int, default=None, help="Resize long image side before SIFT.")
    parser.add_argument("--use-rig-extrinsics", action="store_true", help="Use scene/rig_config.json if present.")
    parser.add_argument(
        "--rebuild-database",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Rebuild COLMAP database.db.",
    )
    parser.add_argument(
        "--clean-sparse-outputs",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Clean previous sparse reconstruction outputs.",
    )
    parser.add_argument("--skip-feature-extraction", action="store_true")
    parser.add_argument("--skip-sequential-matching", action="store_true")
    parser.add_argument("--skip-cross-camera-matching", action="store_true")
    parser.add_argument("--pinhole-conversion-sparse-subdir", default="0", help="Sparse model subdirectory under scene/sparse/.")
    parser.add_argument(
        "--pinhole-conversion-backup-name",
        default="cameras_before_pinhole_conversion.bin",
        help="Backup filename written beside sparse/0/cameras.bin.",
    )
    parser.add_argument(
        "--pinhole-conversion-force",
        action="store_true",
        help="Overwrite an existing PINHOLE-conversion backup file.",
    )

    # AprilTag alignment stage. Defaults mirror align_colmap_apriltag_demo.py.
    parser.add_argument("--tag-family", default="")
    parser.add_argument("--tag-size-mm", type=float, default=None)
    parser.add_argument("--tag-ids", default="", help="Comma-separated physical left-to-right tag ids.")
    parser.add_argument("--tag-center-spacings-mm", default="", help="Comma-separated tag center spacings in mm.")
    parser.add_argument("--tag-x-positions-mm", default="", help="Comma-separated absolute tag center x positions in mm.")
    parser.add_argument("--first-tag-height-z-mm", type=float, default=None)
    parser.add_argument("--ground-z-mm", type=float, default=None)
    parser.add_argument("--tag-wall-y-mm", type=float, default=None)
    parser.add_argument("--min-tag-count", type=int, default=None)
    parser.add_argument("--search-tag-position-permutations", action="store_true")
    parser.add_argument("--try-reverse-tag-x-order", action="store_true")

    # DA3 stage.
    parser.add_argument("--da3-device", default="cuda")
    parser.add_argument("--da3-local-model-dir", default="./depth-anything/weight/DA3-LARGE-1.1")
    parser.add_argument("--da3-hf-model-id", default="depth-anything/DA3-LARGE-1.1")
    parser.add_argument("--depth-dir", default="depth_da3")
    parser.add_argument("--da3-process-res", type=int, default=504)
    parser.add_argument("--da3-window-size", type=int, default=3)
    parser.add_argument("--da3-window-mode", choices=("shared_points", "temporal", "hybrid"), default="hybrid")
    parser.add_argument("--da3-pose-conditioning", choices=("off", "colmap", "auto"), default="colmap")
    parser.add_argument("--fallback-cpu", action="store_true", help="Allow DA3 to fall back to CPU if CUDA fails.")
    parser.add_argument(
        "--global-depth-smooth",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable DA3 global depth smoothing.",
    )
    parser.add_argument(
        "--da3-save-vis",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save DA3 visualization PNGs.",
    )

    # Training stage.
    parser.add_argument("--model-path", default="", help="2DGS model output path. Default: outputs/<scene>/<run-name>.")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--data-device", default="")
    parser.add_argument("--eval", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb-project", default="")
    parser.add_argument("--lambda-depth", type=float, default=None)
    parser.add_argument("--depth-start-iter", type=int, default=None)
    parser.add_argument("--trajectory-cameras-per-frame", type=int, default=None)
    parser.add_argument("--trajectory-axis", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--incremental-train", action="store_true")
    parser.add_argument(
        "--test-iterations",
        default="",
        help="Comma-separated test iterations. Default is derived from --iterations or train_config.py.",
    )
    parser.add_argument(
        "--save-iterations",
        default="",
        help="Comma-separated save iterations. Empty string keeps train_config.py default.",
    )
    args = parser.parse_args()
    args.skip_apriltag = bool(args.skip_apriltag or not args.with_apriltag)
    args.skip_da3 = bool(args.skip_da3 or not args.with_da3)
    return args


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_images).expanduser().resolve()
    scene = resolve_scene_path(input_path, args.scene)
    run_name = args.run_name.strip() or f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = resolve_model_path(scene, run_name, args.model_path)
    manifest = {
        "command": " ".join(shlex.quote(x) for x in sys.argv),
        "project_root": str(PROJECT_ROOT),
        "input_images": str(input_path),
        "scene": str(scene),
        "run_name": run_name,
        "model_path": str(model_path),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "steps": [],
    }

    image_source, camera_dirs = resolve_image_source(input_path, allow_empty=args.dry_run)
    camera_count = max(1, len(camera_dirs))
    if not args.dry_run and not args.skip_colmap and len(camera_dirs) < 2:
        raise ValueError(
            "The COLMAP stage requires at least two camera folders under images/. "
            "Use --skip-colmap only when a valid scene/sparse/0 model already exists."
        )
    print_header(args, image_source, scene, model_path, camera_dirs)

    if args.dry_run:
        print("\n[DRY-RUN] No files will be changed and no stage will be executed.")
        print_plan(args, scene, model_path, camera_count)
        return 0

    try:
        run_prepare_images(args, image_source, scene, camera_dirs)
        record_step(manifest, "prepare-images", "done")
        write_manifest(scene, manifest)
        if should_stop(args, "prepare-images"):
            return 0

        if not args.skip_colmap:
            validate_colmap_image_tree(scene)
            run_colmap(args, scene)
            record_step(manifest, "colmap", "done")
            write_manifest(scene, manifest)
        else:
            record_step(manifest, "colmap", "skipped")
        if should_stop(args, "colmap"):
            return 0

        if not args.skip_apriltag:
            run_apriltag_alignment(args, scene)
            record_step(manifest, "apriltag", "done")
            write_manifest(scene, manifest)
        else:
            record_step(manifest, "apriltag", "skipped")
        if should_stop(args, "apriltag"):
            return 0

        if not args.skip_da3:
            run_da3_depth(args, scene)
            record_step(manifest, "da3", "done")
            write_manifest(scene, manifest)
        else:
            record_step(manifest, "da3", "skipped")
        if should_stop(args, "da3"):
            return 0

        if not args.skip_pinhole_conversion:
            run_pinhole_conversion(args, scene)
            record_step(manifest, "pinhole-conversion", "done")
            write_manifest(scene, manifest)
        else:
            record_step(manifest, "pinhole-conversion", "skipped")
        if should_stop(args, "pinhole-conversion"):
            return 0

        if not args.skip_train:
            run_training(args, scene, model_path, run_name, camera_count, depth_enabled=not args.skip_da3)
            record_step(manifest, "train", "done")
            write_manifest(scene, manifest)
        else:
            record_step(manifest, "train", "skipped")
        if should_stop(args, "train"):
            return 0

        if args.run_render or args.run_metrics:
            run_render_and_metrics(args, model_path)
            record_step(manifest, "render", "done")
            write_manifest(scene, manifest)
        else:
            record_step(manifest, "render", "skipped")

        manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
        write_manifest(scene, manifest)
        print("\nPipeline complete.")
        print(f"  scene: {scene}")
        print(f"  model: {model_path}")
        print(f"  manifest: {scene / 'pipeline_manifest.json'}")
        return 0
    except Exception as exc:
        record_step(manifest, "error", "failed", {"message": str(exc)})
        write_manifest(scene, manifest)
        raise


def resolve_scene_path(input_path: Path, scene_arg: str) -> Path:
    if scene_arg:
        return Path(scene_arg).expanduser().resolve()
    if (input_path / "images").is_dir():
        return input_path
    if input_path.name == "images":
        parent = input_path.parent
        if parent != PROJECT_ROOT:
            return parent
    stem = input_path.parent.name if input_path.name == "images" else input_path.name
    return (PROJECT_ROOT / "data" / f"{stem}_pipeline").resolve()


def resolve_model_path(scene: Path, run_name: str, model_path_arg: str) -> Path:
    if model_path_arg:
        return Path(model_path_arg).expanduser().resolve()
    try:
        scene_rel = scene.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        scene_rel = Path(scene.name)
    return (PROJECT_ROOT / "outputs" / scene_rel / run_name).resolve()


def resolve_image_source(input_path: Path, allow_empty: bool = False) -> tuple[Path, list[Path]]:
    if not input_path.exists():
        if allow_empty:
            return input_path, []
        raise FileNotFoundError(f"Input image path does not exist: {input_path}")
    image_root = input_path / "images" if (input_path / "images").is_dir() else input_path
    camera_dirs = discover_camera_dirs(image_root)
    if not camera_dirs and has_images(image_root):
        camera_dirs = [image_root]
    if not camera_dirs:
        if allow_empty:
            return image_root, []
        raise FileNotFoundError(
            f"No image files found under {image_root}. Expected images/<camera_name>/*.jpg or similar."
        )
    return image_root, camera_dirs


def discover_camera_dirs(image_root: Path) -> list[Path]:
    return sorted([p for p in image_root.iterdir() if p.is_dir() and has_images(p)], key=lambda p: p.name)


def has_images(path: Path) -> bool:
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
            return True
    return False


def iter_image_files(path: Path) -> Iterable[Path]:
    return sorted(
        [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.relative_to(path).as_posix(),
    )


def validate_colmap_image_tree(scene: Path) -> None:
    image_root = scene / "images"
    symlink_images = [p for p in iter_image_files(image_root) if p.is_symlink()]
    if symlink_images:
        examples = ", ".join(str(p.relative_to(image_root)) for p in symlink_images[:5])
        raise ValueError(
            "COLMAP requires real image files under scene/images for stable "
            f"single_camera_per_folder grouping. Found symlink image files: {examples}. "
            "Re-run with --image-mode copy --overwrite-images, or replace scene/images with copied images."
        )


def run_prepare_images(args: argparse.Namespace, image_root: Path, scene: Path, camera_dirs: Sequence[Path]) -> None:
    scene.mkdir(parents=True, exist_ok=True)
    target_images = scene / "images"
    if args.image_mode == "none":
        if not target_images.is_dir():
            raise FileNotFoundError(f"--image-mode none requires existing scene/images: {target_images}")
        print(f"[prepare-images] using existing {target_images}")
        return

    if same_path(image_root, target_images):
        print(f"[prepare-images] input is already scene/images: {target_images}")
        return

    if target_images.exists():
        if args.overwrite_images and is_safe_prepared_images_dir(target_images, image_root):
            shutil.rmtree(target_images)
        elif any(target_images.iterdir()):
            print(f"[prepare-images] existing non-empty {target_images}; keeping it")
            return

    target_images.mkdir(parents=True, exist_ok=True)
    if len(camera_dirs) == 1 and same_path(camera_dirs[0], image_root):
        camera_name = "cam_00"
        place_tree(camera_dirs[0], target_images / camera_name, args.image_mode)
    else:
        for camera_dir in camera_dirs:
            place_tree(camera_dir, target_images / camera_dir.name, args.image_mode)
    print(f"[prepare-images] prepared {len(camera_dirs)} camera folder(s) under {target_images}")


def place_tree(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    image_files = list(iter_image_files(src))
    if not image_files:
        raise FileNotFoundError(f"No supported image files found under {src}")
    dst.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        for src_file in image_files:
            rel = src_file.relative_to(src)
            dst_file = dst / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            dst_file.symlink_to(src_file)
    elif mode == "copy":
        for src_file in image_files:
            rel = src_file.relative_to(src)
            dst_file = dst / rel
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
    else:
        raise ValueError(f"Unsupported image mode: {mode}")


def is_safe_prepared_images_dir(target_images: Path, image_root: Path) -> bool:
    return target_images.name == "images" and not same_path(target_images, image_root)


def same_path(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except FileNotFoundError:
        return False


def run_colmap(args: argparse.Namespace, scene: Path) -> None:
    print("\n[colmap] running sparse reconstruction")
    module = importlib.import_module("crop3d.reconstruction.run_colmap_demo")
    module.SCENE = str(scene)
    module.COLMAP_BIN = resolve_colmap_bin(args.colmap_bin)
    module.REBUILD_DATABASE = bool(args.rebuild_database)
    module.CLEAN_SPARSE_OUTPUTS = bool(args.clean_sparse_outputs)
    module.SKIP_FEATURE_EXTRACTION = bool(args.skip_feature_extraction)
    module.SKIP_SEQUENTIAL_MATCHING = bool(args.skip_sequential_matching)
    module.SKIP_CROSS_CAMERA_MATCHING = bool(args.skip_cross_camera_matching)
    module.USE_CAMERA_INTRINSICS = bool(args.use_camera_intrinsics)
    module.CAMERA_MODEL = args.camera_model
    module.CAMERA_PARAMS = args.camera_params
    module.SEQUENTIAL_OVERLAP = int(args.sequential_overlap)
    module.CROSS_CAMERA_MAX_FRAME_DELTA = int(args.cross_camera_max_frame_delta)
    module.CROSS_CAMERA_MAX_NEIGHBORS_PER_IMAGE = int(args.cross_camera_max_neighbors)
    if args.colmap_max_features:
        module.MAX_FEATURES = str(args.colmap_max_features)
    if args.colmap_max_image_size is not None:
        module.SIFT_MAX_IMAGE_SIZE = int(args.colmap_max_image_size)
    module.USE_RIG_EXTRINSICS = bool(args.use_rig_extrinsics)
    ret = module.main()
    if ret != 0:
        raise RuntimeError(f"COLMAP stage failed with exit code {ret}")


def resolve_colmap_bin(colmap_bin: str) -> str:
    if colmap_bin != "colmap":
        return colmap_bin
    found = shutil.which("colmap")
    if found:
        return found
    candidate = Path(sys.executable).resolve().parent / "colmap"
    if candidate.is_file():
        return str(candidate)
    return colmap_bin


def run_apriltag_alignment(args: argparse.Namespace, scene: Path) -> None:
    print("\n[apriltag] aligning sparse model to metric coordinates")
    module = importlib.import_module("crop3d.reconstruction.align_colmap_apriltag_demo")
    module.SCENE = str(scene)
    module.IMAGES_SUBDIR = "images"
    if args.tag_family:
        module.TAG_FAMILY = args.tag_family
    set_if_not_none(module, "TAG_SIZE_MM", args.tag_size_mm)
    set_if_not_none(module, "FIRST_TAG_HEIGHT_Z_MM", args.first_tag_height_z_mm)
    set_if_not_none(module, "GROUND_Z_MM", args.ground_z_mm)
    set_if_not_none(module, "TAG_WALL_Y_MM", args.tag_wall_y_mm)
    set_if_not_none(module, "MIN_TAG_COUNT_FOR_ALIGN", args.min_tag_count)
    if args.tag_ids:
        module.TAG_IDS_IN_X_ORDER = parse_int_list(args.tag_ids)
    if args.tag_center_spacings_mm:
        module.TAG_CENTER_SPACINGS_MM = parse_float_list(args.tag_center_spacings_mm)
    if args.tag_x_positions_mm:
        module.TAG_X_POSITIONS_MM = parse_float_list(args.tag_x_positions_mm)
        module.TAG_CENTER_SPACINGS_MM = []
    if args.search_tag_position_permutations:
        module.SEARCH_TAG_POSITION_PERMUTATIONS = True
    if args.try_reverse_tag_x_order:
        module.TRY_REVERSE_TAG_X_ORDER = True
    ret = module.main()
    if ret != 0:
        raise RuntimeError(f"AprilTag alignment failed with exit code {ret}")


def run_da3_depth(args: argparse.Namespace, scene: Path) -> None:
    print("\n[da3] generating depth priors")
    cmd = [
        args.python,
        str(PROJECT_ROOT / "crop3d" / "reconstruction" / "prepare_da3_depth_demo.py"),
        "--source_path",
        str(scene),
        "--images",
        "images",
        "--depth_dir",
        args.depth_dir,
        "--hf_model_id",
        args.da3_hf_model_id,
        "--local_model_dir",
        args.da3_local_model_dir,
        "--device",
        args.da3_device,
        "--camera_params",
        args.camera_params,
        "--da3-process-res",
        str(args.da3_process_res),
        "--window-size",
        str(args.da3_window_size),
        "--window-mode",
        args.da3_window_mode,
        "--pose-conditioning",
        args.da3_pose_conditioning,
    ]
    cmd.append("--fallback-cpu" if args.fallback_cpu else "--no-fallback-cpu")
    cmd.append("--global-depth-smooth" if args.global_depth_smooth else "--no-global-depth-smooth")
    cmd.append("--save_vis" if args.da3_save_vis else "--no-save_vis")
    run_command(cmd, cwd=PROJECT_ROOT)


def run_pinhole_conversion(args: argparse.Namespace, scene: Path) -> None:
    print("\n[pinhole-conversion] converting COLMAP cameras.bin to PINHOLE for 2DGS")
    module = importlib.import_module("crop3d.reconstruction.convert_colmap_cameras_to_pinhole")
    module.convert_to_pinhole(
        scene=scene,
        sparse_subdir=args.pinhole_conversion_sparse_subdir,
        backup_name=args.pinhole_conversion_backup_name,
        force=bool(args.pinhole_conversion_force),
    )


def run_training(
    args: argparse.Namespace,
    scene: Path,
    model_path: Path,
    run_name: str,
    camera_count: int,
    depth_enabled: bool,
) -> None:
    print("\n[train] training 2DGS model")
    if str(TWODGS_ROOT) not in sys.path:
        sys.path.insert(0, str(TWODGS_ROOT))
    train_config = importlib.import_module("train_config")
    train_config.SCENE = str(scene)
    train_config.IMAGES = "images"
    train_config.TIMESTAMP = run_name
    train_config.MODEL_PATH = str(model_path)
    train_config.RUN_RENDER_AND_METRICS = False
    train_config.eval_split = bool(args.eval)
    train_config.use_wandb = bool(args.use_wandb)
    train_config.wandb_name = run_name
    train_config.wandb_tags = ["pipeline"]
    if args.wandb_project:
        train_config.wandb_project = args.wandb_project
    if args.iterations is not None:
        train_config.iterations = int(args.iterations)
        if not args.test_iterations:
            train_config.test_iterations = default_test_iterations(int(args.iterations))
    if args.resolution is not None:
        train_config.resolution = int(args.resolution)
    if args.data_device:
        train_config.data_device = args.data_device
    if args.lambda_depth is not None:
        train_config.lambda_depth = float(args.lambda_depth)
    elif not depth_enabled:
        train_config.lambda_depth = 0.0
    if args.depth_start_iter is not None:
        train_config.depth_start_iter = int(args.depth_start_iter)
    if args.trajectory_cameras_per_frame is not None:
        train_config.trajectory_cameras_per_frame = int(args.trajectory_cameras_per_frame)
    else:
        train_config.trajectory_cameras_per_frame = int(camera_count)
    train_config.trajectory_axis_enable = bool(args.trajectory_axis)
    if args.incremental_train:
        train_config.incremental_train = True
    if args.test_iterations:
        train_config.test_iterations = parse_int_list(args.test_iterations)
    if args.save_iterations:
        train_config.save_iterations = parse_int_list(args.save_iterations)

    train_args = train_config.build_train_args()
    cmd = [args.python, str(TWODGS_ROOT / "train.py"), *train_args]
    model_path.mkdir(parents=True, exist_ok=True)
    with (model_path / "pipeline_train_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "scene": str(scene),
                "model_path": str(model_path),
                "camera_count": camera_count,
                "depth_enabled": depth_enabled,
                "train_args": train_args,
                "full_command": " ".join(shlex.quote(x) for x in cmd),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    run_command(cmd, cwd=TWODGS_ROOT)


def run_render_and_metrics(args: argparse.Namespace, model_path: Path) -> None:
    if args.run_render:
        print("\n[render] exporting renders")
        cmd = [args.python, str(TWODGS_ROOT / "render.py"), "-m", str(model_path)]
        if not args.render_mesh:
            cmd.append("--skip_mesh")
        if args.render_path:
            cmd.append("--render_path")
        run_command(cmd, cwd=TWODGS_ROOT)
    if args.run_metrics:
        print("\n[metrics] evaluating renders")
        run_command([args.python, str(TWODGS_ROOT / "metrics.py"), "-m", str(model_path)], cwd=TWODGS_ROOT)


def print_header(
    args: argparse.Namespace,
    image_source: Path,
    scene: Path,
    model_path: Path,
    camera_dirs: Sequence[Path],
) -> None:
    print("========== CROP-3D Reconstruction Pipeline ==========")
    print(f"image source : {image_source}")
    print(f"scene        : {scene}")
    print(f"model output : {model_path}")
    print(f"camera dirs  : {len(camera_dirs)}")
    for camera_dir in camera_dirs:
        print(f"  - {camera_dir.name}")
    steps = ["prepare-images"]
    if not args.skip_colmap:
        steps.append("colmap")
    if not args.skip_apriltag:
        steps.append("apriltag")
    if not args.skip_da3:
        steps.append("da3")
    if not args.skip_pinhole_conversion:
        steps.append("pinhole-conversion")
    if not args.skip_train:
        steps.append("train")
    if args.run_render or args.run_metrics:
        steps.append("render/metrics")
    print("steps        : " + " -> ".join(steps))
    print(f"dry run      : {args.dry_run}")
    print("=====================================================")


def print_plan(args: argparse.Namespace, scene: Path, model_path: Path, camera_count: int) -> None:
    print(f"prepare images: {args.image_mode} -> {scene / 'images'}")
    if not args.skip_colmap:
        print(f"run_colmap_demo.py: multi-camera, colmap={resolve_colmap_bin(args.colmap_bin)}")
    if not args.skip_apriltag:
        print("align_colmap_apriltag_demo.py: enabled")
    if not args.skip_da3:
        print(f"prepare_da3_depth_demo.py: depth_dir={args.depth_dir}, device={args.da3_device}")
    if not args.skip_pinhole_conversion:
        print(
            "convert_colmap_cameras_to_pinhole.py: "
            f"sparse/{args.pinhole_conversion_sparse_subdir}/cameras.bin -> PINHOLE"
        )
    if not args.skip_train:
        print(f"train.py: model_path={model_path}, cameras_per_frame={camera_count}")
    if args.run_render:
        print("render.py: enabled")
    if args.run_metrics:
        print("metrics.py: enabled")


def run_command(cmd: Sequence[str], cwd: Path) -> None:
    print("+ " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def should_stop(args: argparse.Namespace, stage: str) -> bool:
    return args.stop_after == stage


def set_if_not_none(module, name: str, value) -> None:
    if value is not None:
        setattr(module, name, value)


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def default_test_iterations(iterations: int) -> list[int]:
    if iterations <= 0:
        return []
    if iterations <= 10_000:
        return [iterations]
    vals = list(range(10_000, iterations, 10_000))
    if not vals or vals[-1] != iterations:
        vals.append(iterations)
    return vals


def record_step(manifest: dict, name: str, status: str, extra: dict | None = None) -> None:
    item = {"name": name, "status": status, "time": datetime.now().isoformat(timespec="seconds")}
    if extra:
        item.update(extra)
    manifest.setdefault("steps", []).append(item)


def write_manifest(scene: Path, manifest: dict) -> None:
    scene.mkdir(parents=True, exist_ok=True)
    with (scene / "pipeline_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
