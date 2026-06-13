#!/usr/bin/env python3
"""Configure and launch 2DGS training for a COLMAP scene.

Expected layout:
    scene/
      images/
      sparse/0/

Edit the user settings below, then run:
    python train_config.py
    python train_config.py --dry-run
    python train_config.py --no-render
"""

from __future__ import annotations
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

RUN_RENDER_AND_METRICS = False

# -----------------------------------------------------------------------------
# Scene And Output
# -----------------------------------------------------------------------------
SCENE = "./data/4.13RLU_B2_ba_lite_gpu"
IMAGES = "images"

TIMESTAMP = datetime.now().strftime("%m-%d_%H:%M")
PORT = random.randint(10000, 30000)
MODEL_PATH = f"outputs/{Path(SCENE).as_posix().lstrip('./')}/{TIMESTAMP}"

# -----------------------------------------------------------------------------
# Loading And Data
# -----------------------------------------------------------------------------
resolution = 2
eval_split = True
white_background = False
data_device = "cuda"
sh_degree = 3

# -----------------------------------------------------------------------------
# Trajectory Filtering
# -----------------------------------------------------------------------------
trajectory_axis_enable = True
trajectory_preserve_world_coords = True
trajectory_depth_min = 0.2
trajectory_depth_max = 2.0
trajectory_cameras_per_frame = 3
trajectory_segment_buffer = 0.5
trajectory_prune_interval = 0

# -----------------------------------------------------------------------------
# Training Schedule
# -----------------------------------------------------------------------------
iterations = 60_000

test_iterations =  [10000, 20000,30000,40000,50000]
save_iterations: list[int] = []
checkpoint_iterations: list[int] = []
start_checkpoint = ""

ip = "127.0.0.1"
detect_anomaly = False
quiet = False

# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
depth_ratio = 0.0
debug = False

# -----------------------------------------------------------------------------
# Optimization Learning Rates
# -----------------------------------------------------------------------------
position_lr_init = 0.00016
position_lr_final = 0.0000016
position_lr_delay_mult = 0.01
position_lr_max_steps = 30_000

feature_lr = 0.0025
opacity_lr = 0.05
scaling_lr = 0.004
rotation_lr = 0.001

# -----------------------------------------------------------------------------
# Losses And Regularization
# -----------------------------------------------------------------------------
lambda_dssim = 0.2
lambda_normal = 0.02
lambda_dist = 0.02
lambda_depth = 0.1
depth_start_iter = 500
depth_min = 0.2
depth_max = 2.0
lambda_smooth = 0.005
smooth_start_iter = 1500

# -----------------------------------------------------------------------------
# Photometric Compensation
# -----------------------------------------------------------------------------
use_photometric_compensation = False
photometric_embedding_dim = 32
temporal_smoothness_weight = 0.05
photometric_compensation_lr_init = 0.05
photometric_compensation_lr_final = 0.0005

# -----------------------------------------------------------------------------
# Densification And Pruning
# -----------------------------------------------------------------------------
percent_dense = 0.01
densification_interval = 100
opacity_reset_interval = 3000
densify_from_iter = 5000
densify_until_iter = 25_000
densify_grad_threshold = 0.0003
opacity_cull = 0.05

# -----------------------------------------------------------------------------
# Splat Size Limits
# -----------------------------------------------------------------------------
prune_screen_size = 12
max_scale_extent_ratio = 0.07
prune_screen_start_iter = 3000

# -----------------------------------------------------------------------------
# Incremental Training
# -----------------------------------------------------------------------------
incremental_train = False
init_frame_num = 6
init_iteration = 3_000
local_iter = 200
global_iter = 400
post_iter = 10000

window_size = 11
local_window_views = 0
overlap_threshold = 0.2

# -----------------------------------------------------------------------------
# Weights & Biases
# -----------------------------------------------------------------------------
use_wandb = True

wandb_project = "2d-gaussian-splatting"
wandb_entity = ""
wandb_name = ""
wandb_tags: list[str] = []
wandb_log_interval = 10
wandb_psnr_interval = 1000


def build_train_args() -> list[str]:
    """Build argv for train.py."""
    scene = str(Path(SCENE).expanduser().resolve())
    model = str(Path(MODEL_PATH).expanduser().resolve())

    args: list[str] = [
        "-s",
        scene,
        "-m",
        model,
        "--images",
        IMAGES,
        "-r",
        str(resolution),
        "--data_device",
        data_device,
        "--sh_degree",
        str(sh_degree),
        "--iterations",
        str(iterations),
        "--position_lr_init",
        str(position_lr_init),
        "--position_lr_final",
        str(position_lr_final),
        "--position_lr_delay_mult",
        str(position_lr_delay_mult),
        "--position_lr_max_steps",
        str(position_lr_max_steps),
        "--feature_lr",
        str(feature_lr),
        "--opacity_lr",
        str(opacity_lr),
        "--scaling_lr",
        str(scaling_lr),
        "--rotation_lr",
        str(rotation_lr),
        "--lambda_dssim",
        str(lambda_dssim),
        "--lambda_normal",
        str(lambda_normal),
        "--lambda_dist",
        str(lambda_dist),
        "--lambda_depth",
        str(lambda_depth),
        "--depth_start_iter",
        str(depth_start_iter),
        "--depth_min",
        str(depth_min),
        "--depth_max",
        str(depth_max),
        "--lambda_smooth",
        str(lambda_smooth),
        "--smooth_start_iter",
        str(smooth_start_iter),
        "--percent_dense",
        str(percent_dense),
        "--densification_interval",
        str(densification_interval),
        "--opacity_reset_interval",
        str(opacity_reset_interval),
        "--densify_from_iter",
        str(densify_from_iter),
        "--densify_until_iter",
        str(densify_until_iter),
        "--densify_grad_threshold",
        str(densify_grad_threshold),
        "--opacity_cull",
        str(opacity_cull),
        "--prune_screen_size",
        str(prune_screen_size),
        "--max_scale_extent_ratio",
        str(max_scale_extent_ratio),
        "--prune_screen_start_iter",
        str(prune_screen_start_iter),
        "--depth_ratio",
        str(depth_ratio),
        "--ip",
        ip,
        "--port",
        str(PORT),
        "--wandb_project",
        wandb_project,
        "--wandb_log_interval",
        str(wandb_log_interval),
        "--wandb_psnr_interval",
        str(wandb_psnr_interval),
    ]

    if eval_split:
        args.append("--eval")
    if white_background:
        args.append("--white_background")
    if debug:
        args.append("--debug")
    if detect_anomaly:
        args.append("--detect_anomaly")
    if quiet:
        args.append("--quiet")

    if not use_wandb:
        args.append("--no_wandb")
    else:
        _run = wandb_name.strip() if wandb_name and wandb_name.strip() else TIMESTAMP
        args += ["--wandb_name", _run]

    if wandb_entity:
        args += ["--wandb_entity", wandb_entity]
    if wandb_tags:
        args.append("--wandb_tags")
        args.extend(wandb_tags)

    args.append("--test_iterations")
    args.extend(str(x) for x in test_iterations)
    args.append("--save_iterations")
    args.extend(str(x) for x in save_iterations)

    if checkpoint_iterations:
        args.append("--checkpoint_iterations")
        args.extend(str(x) for x in checkpoint_iterations)

    if start_checkpoint:
        args += ["--start_checkpoint", str(Path(start_checkpoint).expanduser().resolve())]

    if incremental_train:
        args.append("--incremental_train")
        args += [
            "--init_frame_num",
            str(init_frame_num),
            "--init_iteration",
            str(init_iteration),
            "--local_iter",
            str(local_iter),
            "--global_iter",
            str(global_iter),
            "--post_iter",
            str(post_iter),
            "--window_size",
            str(window_size),
            "--local_window_views",
            str(local_window_views),
            "--overlap_threshold",
            str(overlap_threshold),
        ]

    if trajectory_axis_enable:
        args.append("--trajectory_axis_enable")
    if trajectory_preserve_world_coords:
        args.append("--trajectory_preserve_world_coords")
    args += [
        "--trajectory_depth_min",
        str(trajectory_depth_min),
        "--trajectory_depth_max",
        str(trajectory_depth_max),
        "--trajectory_cameras_per_frame",
        str(trajectory_cameras_per_frame),
        "--trajectory_segment_buffer",
        str(trajectory_segment_buffer),
        "--trajectory_prune_interval",
        str(trajectory_prune_interval),
    ]

    if use_photometric_compensation:
        args.append("--use_photometric_compensation")
    args += [
        "--photometric_embedding_dim",
        str(photometric_embedding_dim),
        "--temporal_smoothness_weight",
        str(temporal_smoothness_weight),
        "--photometric_compensation_lr_init",
        str(photometric_compensation_lr_init),
        "--photometric_compensation_lr_final",
        str(photometric_compensation_lr_final),
    ]

    return args


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    no_render = "--no-render" in sys.argv

    train_args = build_train_args()
    cmd = [sys.executable, str(PROJECT_ROOT / "train.py"), *train_args]

    print("========== Train config ==========")
    print("timestamp:", TIMESTAMP)
    print("scene:", SCENE)
    print("model_path:", MODEL_PATH)
    print("port:", PORT)
    print("full command:")
    print(" ", " \\\n  ".join(cmd))
    print("==================================")

    if dry_run:
        return

    Path(MODEL_PATH).mkdir(parents=True, exist_ok=True)
    config_log = {
        "timestamp": TIMESTAMP,
        "scene": SCENE,
        "images": IMAGES,
        "model_path": MODEL_PATH,
        "port": PORT,
        "train_args": train_args,
        "full_command": " ".join(cmd),
    }
    with open(Path(MODEL_PATH) / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config_log, f, indent=2, ensure_ascii=False)

    ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if ret.returncode != 0:
        sys.exit(ret.returncode)

    if not RUN_RENDER_AND_METRICS or no_render:
        return

    model_abs = str(Path(MODEL_PATH).resolve())
    print("Running render.py ...")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "render.py"), "-m", model_abs],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    print("Running metrics.py ...")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "metrics.py"), "-m", model_abs],
        cwd=str(PROJECT_ROOT),
        check=False,
    )


if __name__ == "__main__":
    main()
