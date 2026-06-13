#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.trajectory_axis_utils import prune_gaussians_by_depth
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Cached VGG-LPIPS for validation (same backend as metrics.py; avoid reinstantiating each view).
_lpips_vgg_model = None
_lpips_import_error = None


def _to_bchw_image(t: torch.Tensor) -> torch.Tensor:
    """(C,H,W) or (1,C,H,W) -> (1,C,H,W) for SSIM / LPIPS / PSNR."""
    if t.dim() == 3:
        return t.unsqueeze(0)
    if t.dim() == 4:
        return t
    raise ValueError("expected image tensor with 3 or 4 dimensions, got shape {}".format(tuple(t.shape)))


def _lpips_mean_bchw(pred_bchw: torch.Tensor, gt_bchw: torch.Tensor):
    """Mean LPIPS over batch; returns None if lpipsPyTorch unavailable."""
    global _lpips_vgg_model, _lpips_import_error
    if _lpips_import_error is not None:
        return None
    try:
        from lpipsPyTorch.modules.lpips import LPIPS
    except ImportError as e:
        _lpips_import_error = e
        return None
    device = pred_bchw.device
    if _lpips_vgg_model is None:
        _lpips_vgg_model = LPIPS("vgg", "0.1").to(device).eval()
    else:
        _lpips_vgg_model = _lpips_vgg_model.to(device)
    with torch.no_grad():
        return _lpips_vgg_model(pred_bchw, gt_bchw).mean()


def _wandb_config_dict(args):
    """Serialize argparse Namespace for wandb.config (primitives only)."""
    out = {}
    for k, v in vars(args).items():
        if k.startswith("_"):
            continue
        if v is None:
            continue
        if isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            if all(isinstance(x, (bool, int, float, str)) for x in v):
                out[k] = list(v)
    return out


def _tensor_chw_to_wandb_image(t):
    """Convert float tensor (C,H,W) or (1,C,H,W) in [0,1] to wandb.Image."""
    if t.dim() == 4:
        t = t[0]
    t = t.detach().cpu().float().clamp(0.0, 1.0)
    c, h, w = t.shape
    arr = (t.numpy() * 255.0).astype(np.uint8)
    if c == 1:
        arr = np.repeat(arr, 3, axis=0)
    arr = np.transpose(arr, (1, 2, 0))
    return wandb.Image(arr)


def incremental_total_steps(num_views: int, init_frame_num: int, opt) -> int:
    """total_steps = init_iteration + (N - K) * (local + global) + post_iter; K clamped to [1, N]."""
    K = min(max(1, init_frame_num), num_views)
    return (
        opt.init_iteration
        + (num_views - K) * (opt.local_iter + opt.global_iter)
        + opt.post_iter
    )


@torch.no_grad()
def _dynamic_update_start_view_id(
    gaussians, pipe, background, train_cams, start_view_id: int, end_view_id: int, opt
) -> int:
    """Shrink local window from the front when overlap with the newest view is low."""
    if getattr(opt, "local_window_views", 0) > 0:
        return start_view_id
    if end_view_id <= 1:
        return start_view_id
    ws = int(getattr(opt, "window_size", 5))
    thr = float(getattr(opt, "overlap_threshold", 0.2))
    end_cam = train_cams[end_view_id - 1]
    p_end = render(end_cam, gaussians, pipe, background)
    end_mask = p_end["visibility_filter"]
    cur = start_view_id
    while end_view_id - cur > ws:
        start_cam = train_cams[cur]
        p_s = render(start_cam, gaussians, pipe, background)
        start_mask = p_s["visibility_filter"]
        c_s = int(start_mask.count_nonzero().item())
        c_e = int(end_mask.count_nonzero().item())
        denom = min(c_s, c_e)
        if denom == 0:
            rho = 0.0
        else:
            rho = float(torch.logical_and(start_mask, end_mask).count_nonzero().item()) / float(denom)
        if rho < thr:
            cur += 1
        else:
            break
    return cur


def _edge_aware_smooth_loss(depth: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Edge-aware smoothness on rendered depth."""
    grad_d_x = (depth[:, :, 1:] - depth[:, :, :-1]).abs()
    grad_d_y = (depth[:, 1:, :] - depth[:, :-1, :]).abs()
    grad_i_x = (image[:, :, 1:] - image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    grad_i_y = (image[:, 1:, :] - image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    smooth_x = grad_d_x * torch.exp(-grad_i_x)
    smooth_y = grad_d_y * torch.exp(-grad_i_y)
    return smooth_x.mean() + smooth_y.mean()


def _train_one_step(
    global_step: int,
    viewpoint_cam,
    scene: Scene,
    gaussians,
    opt,
    pipe,
    background,
    dataset,
    iter_start,
    iter_end,
):
    iter_start.record()
    gaussians.update_learning_rate(global_step)
    if global_step % 1000 == 0:
        gaussians.oneupSHdegree()

    render_pkg = render(viewpoint_cam, gaussians, pipe, background)
    image = render_pkg["render"]
    viewspace_point_tensor = render_pkg["viewspace_points"]
    visibility_filter = render_pkg["visibility_filter"]
    radii = render_pkg["radii"]

    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

    lambda_normal = opt.lambda_normal if global_step > 7000 else 0.0
    lambda_dist = opt.lambda_dist if global_step > 3000 else 0.0

    rend_dist = render_pkg["rend_dist"]
    rend_normal = render_pkg["rend_normal"]
    surf_normal = render_pkg["surf_normal"]
    surf_depth = render_pkg["surf_depth"]
    normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
    normal_loss = lambda_normal * (normal_error).mean()
    dist_loss = lambda_dist * (rend_dist).mean()
    total_loss = loss + dist_loss + normal_loss
    depth_loss = torch.tensor(0.0, device=image.device)
    smooth_loss = torch.tensor(0.0, device=image.device)

    lambda_depth = getattr(opt, "lambda_depth", 0.0)
    depth_start = getattr(opt, "depth_start_iter", 1000)
    if (
        lambda_depth > 0
        and global_step >= depth_start
        and getattr(viewpoint_cam, "depth_map", None) is not None
    ):
        gt_depth = viewpoint_cam.depth_map.cuda()
        d_min = getattr(opt, "depth_min", 0.3)
        d_max = getattr(opt, "depth_max", 1.5)
        valid_mask = ((gt_depth > d_min) & (gt_depth < d_max)).float()
        n_valid = valid_mask.sum().clamp(min=1.0)

        pred_v = surf_depth * valid_mask
        gt_v = gt_depth * valid_mask

        mean_d = (gt_v.sum() / n_valid).clamp(min=1e-3)
        l1_d = (pred_v - gt_v).abs().sum() / n_valid / mean_d

        pred_mu = pred_v.sum() / n_valid
        gt_mu = gt_v.sum() / n_valid
        pred_f = (pred_v - pred_mu) * valid_mask
        gt_f = (gt_v - gt_mu) * valid_mask
        pearson = (pred_f * gt_f).sum() / (pred_f.norm() * gt_f.norm() + 1e-8)
        pearson_loss = 1.0 - pearson

        depth_loss = lambda_depth * (0.5 * l1_d + 0.5 * pearson_loss)
        total_loss = total_loss + depth_loss

    lambda_smooth = getattr(opt, "lambda_smooth", 0.0)
    smooth_start = getattr(opt, "smooth_start_iter", 1000)
    if lambda_smooth > 0 and global_step >= smooth_start:
        smooth_loss = lambda_smooth * _edge_aware_smooth_loss(surf_depth, image.detach())
        total_loss = total_loss + smooth_loss

    if (
        getattr(opt, "temporal_smoothness_weight", 0) > 0
        and getattr(gaussians, "use_photometric_compensation", False)
        and gaussians.embedding_photometric is not None
    ):
        uid = int(viewpoint_cam.uid)
        n_emb = gaussians.embedding_photometric.num_embeddings
        device = gaussians.embedding_photometric.weight.device
        L_smooth = torch.tensor(0.0, device=device)
        if uid > 0:
            emb_cur = gaussians.embedding_photometric(
                torch.tensor([uid], dtype=torch.long, device=device)
            )
            emb_prev = gaussians.embedding_photometric(
                torch.tensor([uid - 1], dtype=torch.long, device=device)
            )
            L_smooth = L_smooth + (emb_cur - emb_prev).pow(2).sum()
        if uid < n_emb - 1:
            emb_cur = gaussians.embedding_photometric(
                torch.tensor([uid], dtype=torch.long, device=device)
            )
            emb_next = gaussians.embedding_photometric(
                torch.tensor([uid + 1], dtype=torch.long, device=device)
            )
            L_smooth = L_smooth + (emb_cur - emb_next).pow(2).sum()
        total_loss = total_loss + opt.temporal_smoothness_weight * L_smooth

    total_loss.backward()
    iter_end.record()

    return {
        "Ll1": Ll1,
        "loss": loss,
        "dist_loss": dist_loss,
        "normal_loss": normal_loss,
        "depth_loss": depth_loss,
        "smooth_loss": smooth_loss,
        "visibility_filter": visibility_filter,
        "viewspace_point_tensor": viewspace_point_tensor,
        "radii": radii,
    }


def _post_step_logging_and_optim(
    global_step: int,
    total_steps: int,
    step_out: dict,
    scene: Scene,
    gaussians,
    opt,
    dataset,
    pipe,
    background,
    ema_loss_for_log: float,
    ema_dist_for_log: float,
    ema_normal_for_log: float,
    ema_depth_for_log: float,
    ema_smooth_for_log: float,
    progress_bar,
    tb_writer,
    wandb_run,
    wandb_log_interval: int,
    wandb_psnr_interval: int,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    iter_start,
    iter_end,
    full_args,
):
    loss = step_out["loss"]
    dist_loss = step_out["dist_loss"]
    normal_loss = step_out["normal_loss"]
    depth_loss = step_out["depth_loss"]
    smooth_loss = step_out["smooth_loss"]
    Ll1 = step_out["Ll1"]
    visibility_filter = step_out["visibility_filter"]
    viewspace_point_tensor = step_out["viewspace_point_tensor"]
    radii = step_out["radii"]

    with torch.no_grad():
        ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
        ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
        ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log
        ema_depth_for_log = 0.4 * depth_loss.item() + 0.6 * ema_depth_for_log
        ema_smooth_for_log = 0.4 * smooth_loss.item() + 0.6 * ema_smooth_for_log

        if global_step % 10 == 0:
            loss_dict = {
                "Loss": f"{ema_loss_for_log:.{5}f}",
                "distort": f"{ema_dist_for_log:.{5}f}",
                "normal": f"{ema_normal_for_log:.{5}f}",
                "depth": f"{ema_depth_for_log:.{5}f}",
                "smooth": f"{ema_smooth_for_log:.{5}f}",
                "Points": f"{len(gaussians.get_xyz)}",
            }
            progress_bar.set_postfix(loss_dict)
            progress_bar.update(10)
        if global_step == total_steps:
            progress_bar.close()

        if tb_writer is not None:
            tb_writer.add_scalar("train_loss_patches/dist_loss", ema_dist_for_log, global_step)
            tb_writer.add_scalar("train_loss_patches/normal_loss", ema_normal_for_log, global_step)
            tb_writer.add_scalar("train_loss_patches/depth_loss", ema_depth_for_log, global_step)
            tb_writer.add_scalar("train_loss_patches/smooth_loss", ema_smooth_for_log, global_step)

        if (
            wandb_run is not None
            and WANDB_AVAILABLE
            and wandb_log_interval > 0
            and (global_step % wandb_log_interval == 0)
        ):
            wandb.log(
                {
                    "train_loss_patches/dist_loss": ema_dist_for_log,
                    "train_loss_patches/normal_loss": ema_normal_for_log,
                    "train_loss_patches/depth_loss": ema_depth_for_log,
                    "train_loss_patches/smooth_loss": ema_smooth_for_log,
                },
                step=global_step,
            )

        training_report(
            tb_writer,
            wandb_run,
            wandb_log_interval,
            wandb_psnr_interval,
            global_step,
            Ll1,
            loss,
            l1_loss,
            iter_start.elapsed_time(iter_end),
            testing_iterations,
            scene,
            render,
            (pipe, background),
        )
        if global_step in saving_iterations:
            print("\n[ITER {}] Saving Gaussians".format(global_step))
            scene.save(global_step)

        if global_step < opt.densify_until_iter:
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if global_step > opt.densify_from_iter and global_step % opt.densification_interval == 0:
                ps = int(getattr(opt, "prune_screen_size", 20))
                start_scr = int(getattr(opt, "prune_screen_start_iter", 0))
                size_threshold = None
                if ps > 0:
                    if start_scr > 0:
                        if global_step > start_scr:
                            size_threshold = ps
                    elif global_step > opt.opacity_reset_interval:
                        size_threshold = ps
                ratio = float(getattr(opt, "max_scale_extent_ratio", 0.1))
                gaussians.densify_and_prune(
                    opt.densify_grad_threshold,
                    opt.opacity_cull,
                    scene.cameras_extent,
                    size_threshold,
                    max_scale_extent_ratio=ratio,
                )

            if global_step % opt.opacity_reset_interval == 0 or (
                dataset.white_background and global_step == opt.densify_from_iter
            ):
                gaussians.reset_opacity()

        pi = int(getattr(dataset, "trajectory_prune_interval", 0))
        if (
            getattr(scene, "trajectory_axis_enable", False)
            and float(getattr(scene, "trajectory_depth_max", 0.0)) > 0
            and pi > 0
            and global_step > 0
            and global_step % pi == 0
        ):
            prune_gaussians_by_depth(
                gaussians,
                scene.trajectory_origin,
                scene.trajectory_axis,
                float(scene.trajectory_depth_min),
                float(scene.trajectory_depth_max),
                float(scene.trajectory_t_min),
                float(scene.trajectory_t_max),
            )

        if global_step < total_steps:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)

        if global_step in checkpoint_iterations:
            print("\n[ITER {}] Saving Checkpoint".format(global_step))
            torch.save(
                (gaussians.capture(), global_step),
                scene.model_path + "/chkpnt" + str(global_step) + ".pth",
            )

    with torch.no_grad():
        if network_gui.conn == None:
            network_gui.try_connect(dataset.render_items)
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                if custom_cam != None:
                    render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                    net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                metrics_dict = {
                    "#": gaussians.get_opacity.shape[0],
                    "loss": ema_loss_for_log,
                }
                network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                if do_training and ((global_step < int(total_steps)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

    return ema_loss_for_log, ema_dist_for_log, ema_normal_for_log, ema_depth_for_log, ema_smooth_for_log


def _training_incremental(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    full_args,
    tb_writer,
    wandb_run,
    scene: Scene,
    gaussians,
    background,
):
    wandb_log_interval = getattr(full_args, "wandb_log_interval", 10) if full_args is not None else 10
    wandb_psnr_interval = getattr(full_args, "wandb_psnr_interval", 1000) if full_args is not None else 0

    train_cams = scene.getTrainCameras()
    num_views = len(train_cams)
    if num_views < 1:
        raise ValueError("incremental_train requires at least one training camera.")
    K = min(max(1, opt.init_frame_num), num_views)
    total_steps = incremental_total_steps(num_views, opt.init_frame_num, opt)
    if total_steps not in saving_iterations:
        saving_iterations.append(total_steps)
        saving_iterations.sort()

    print(
        "Incremental training: N={} init_frame_num={} (used K={}), total_steps={}".format(
            num_views, opt.init_frame_num, K, total_steps
        )
    )

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0
    ema_depth_for_log = 0.0
    ema_smooth_for_log = 0.0
    global_step = 0
    progress_bar = tqdm(range(0, total_steps), desc="Training progress")

    def run_phase_steps(num_steps: int, camera_subset: list):
        nonlocal global_step, ema_loss_for_log, ema_dist_for_log, ema_normal_for_log, ema_depth_for_log, ema_smooth_for_log
        if not camera_subset:
            return
        viewpoint_stack = None
        for _ in range(num_steps):
            global_step += 1
            if not viewpoint_stack:
                viewpoint_stack = camera_subset.copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
            step_out = _train_one_step(
                global_step,
                viewpoint_cam,
                scene,
                gaussians,
                opt,
                pipe,
                background,
                dataset,
                iter_start,
                iter_end,
            )
            ema_loss_for_log, ema_dist_for_log, ema_normal_for_log, ema_depth_for_log, ema_smooth_for_log = _post_step_logging_and_optim(
                global_step,
                total_steps,
                step_out,
                scene,
                gaussians,
                opt,
                dataset,
                pipe,
                background,
                ema_loss_for_log,
                ema_dist_for_log,
                ema_normal_for_log,
                ema_depth_for_log,
                ema_smooth_for_log,
                progress_bar,
                tb_writer,
                wandb_run,
                wandb_log_interval,
                wandb_psnr_interval,
                testing_iterations,
                saving_iterations,
                checkpoint_iterations,
                iter_start,
                iter_end,
                full_args,
            )

    # Init: cameras [0, K)
    run_phase_steps(opt.init_iteration, train_cams[0:K])

    end_view_id = K
    start_view_id = 0
    end_view_id += 1

    while end_view_id <= num_views:
        run_phase_steps(opt.local_iter, train_cams[start_view_id:end_view_id])
        run_phase_steps(opt.global_iter, train_cams[0:end_view_id])

        start_view_id = _dynamic_update_start_view_id(
            gaussians, pipe, background, train_cams, start_view_id, end_view_id, opt
        )
        end_view_id += 1
        if getattr(opt, "local_window_views", 0) > 0:
            start_view_id = max(0, end_view_id - int(opt.local_window_views))

    run_phase_steps(opt.post_iter, train_cams)


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, full_args=None):
    first_iter = 0
    wandb_log_interval = getattr(full_args, "wandb_log_interval", 10) if full_args is not None else 10
    wandb_psnr_interval = getattr(full_args, "wandb_psnr_interval", 1000) if full_args is not None else 0
    tb_writer, wandb_run = prepare_output_and_logger(dataset, full_args)
    try:
        gaussians = GaussianModel(dataset.sh_degree)
        shuffle_cams = not getattr(opt, "incremental_train", False)
        scene = Scene(dataset, gaussians, shuffle=shuffle_cams)
        if getattr(opt, "incremental_train", False):
            _n_train = len(scene.getTrainCameras())
            _ts = incremental_total_steps(_n_train, opt.init_frame_num, opt)
            opt.position_lr_max_steps = max(opt.position_lr_max_steps, _ts)
        if getattr(opt, "use_photometric_compensation", False):
            gaussians.use_photometric_compensation = True
            gaussians.photometric_embedding_dim = opt.photometric_embedding_dim
            gaussians.set_photometric_compensation(len(scene.getTrainCameras()))
        gaussians.training_setup(opt)

        if getattr(opt, "incremental_train", False):
            if checkpoint:
                print(
                    "incremental_train: loading Gaussian checkpoint weights; "
                    "iteration schedule still runs full incremental (state not restored)."
                )
                (model_params, _) = torch.load(checkpoint)
                gaussians.restore(model_params, opt)
            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            _training_incremental(
                dataset,
                opt,
                pipe,
                testing_iterations,
                saving_iterations,
                checkpoint_iterations,
                full_args,
                tb_writer,
                wandb_run,
                scene,
                gaussians,
                background,
            )
        else:
            if checkpoint:
                (model_params, first_iter) = torch.load(checkpoint)
                gaussians.restore(model_params, opt)

            bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            iter_start = torch.cuda.Event(enable_timing=True)
            iter_end = torch.cuda.Event(enable_timing=True)

            viewpoint_stack = None
            ema_loss_for_log = 0.0
            ema_dist_for_log = 0.0
            ema_normal_for_log = 0.0
            ema_depth_for_log = 0.0
            ema_smooth_for_log = 0.0

            progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
            first_iter += 1
            for iteration in range(first_iter, opt.iterations + 1):
                if not viewpoint_stack:
                    viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

                step_out = _train_one_step(
                    iteration,
                    viewpoint_cam,
                    scene,
                    gaussians,
                    opt,
                    pipe,
                    background,
                    dataset,
                    iter_start,
                    iter_end,
                )
                ema_loss_for_log, ema_dist_for_log, ema_normal_for_log, ema_depth_for_log, ema_smooth_for_log = _post_step_logging_and_optim(
                    iteration,
                    opt.iterations,
                    step_out,
                    scene,
                    gaussians,
                    opt,
                    dataset,
                    pipe,
                    background,
                    ema_loss_for_log,
                    ema_dist_for_log,
                    ema_normal_for_log,
                    ema_depth_for_log,
                    ema_smooth_for_log,
                    progress_bar,
                    tb_writer,
                    wandb_run,
                    wandb_log_interval,
                    wandb_psnr_interval,
                    testing_iterations,
                    saving_iterations,
                    checkpoint_iterations,
                    iter_start,
                    iter_end,
                    full_args,
                )
    finally:
        if wandb_run is not None and WANDB_AVAILABLE:
            try:
                wandb.finish()
            except Exception:
                pass

def prepare_output_and_logger(dataset, full_args=None):
    if not dataset.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        dataset.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(dataset.model_path))
    os.makedirs(dataset.model_path, exist_ok = True)
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(dataset))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(dataset.model_path)
    else:
        print("Tensorboard not available: not logging progress")

    wandb_run = None
    if (
        full_args is not None
        and WANDB_AVAILABLE
        and not getattr(full_args, "no_wandb", False)
    ):
        try:
            tags = getattr(full_args, "wandb_tags", None)
            if tags:
                tags = list(tags)
            else:
                tags = None
            entity = getattr(full_args, "wandb_entity", None) or None
            name = getattr(full_args, "wandb_name", None) or None
            wandb_run = wandb.init(
                project=getattr(full_args, "wandb_project", "2d-gaussian-splatting"),
                entity=entity,
                name=name,
                tags=tags,
                config=_wandb_config_dict(full_args),
            )
        except Exception as e:
            print("wandb.init failed ({}); continuing without Weights & Biases.".format(e))
            wandb_run = None

    return tb_writer, wandb_run

@torch.no_grad()
def training_report(
    tb_writer,
    wandb_run,
    wandb_log_interval,
    wandb_psnr_interval,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    renderArgs,
):
    wb_scalars = (
        wandb_run is not None
        and WANDB_AVAILABLE
        and wandb_log_interval > 0
        and (iteration % wandb_log_interval == 0)
    )
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    if wb_scalars:
        wandb.log(
            {
                "train_loss_patches/reg_loss": Ll1.item(),
                "train_loss_patches/total_loss": loss.item(),
                "iter_time": elapsed,
                "total_points": scene.gaussians.get_xyz.shape[0],
            },
            step=iteration,
        )

    # Full tensorboard + optional wandb images: only on test_iterations.
    # PSNR/L1 on wandb: every wandb_psnr_interval (when wandb active); 0 = disable periodic PSNR.
    do_full_eval = iteration in testing_iterations
    do_wandb_psnr = (
        wandb_run is not None
        and WANDB_AVAILABLE
        and wandb_psnr_interval > 0
        and (iteration % wandb_psnr_interval == 0)
    )

    if do_full_eval or do_wandb_psnr:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_sum = 0.0
                lpips_sum = 0.0
                lpips_ok = True
                n_cam = len(config["cameras"])
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    img_b = _to_bchw_image(image)
                    gt_b = _to_bchw_image(gt_image)
                    ssim_sum += float(ssim(img_b, gt_b).item())
                    lp = _lpips_mean_bchw(img_b, gt_b)
                    if lp is None:
                        lpips_ok = False
                    else:
                        lpips_sum += float(lp.item())
                    need_viz = do_full_eval and (idx < 5) and (
                        tb_writer or (wandb_run is not None and WANDB_AVAILABLE)
                    )
                    if need_viz:
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        prefix = config['name'] + "_view_{}/".format(viewpoint.image_name)
                        if tb_writer:
                            tb_writer.add_images(prefix + "depth", depth[None], global_step=iteration)
                            tb_writer.add_images(prefix + "render", image[None], global_step=iteration)

                        wb_media = {}
                        if wandb_run is not None and WANDB_AVAILABLE:
                            wb_media[prefix + "depth"] = _tensor_chw_to_wandb_image(depth[None])
                            wb_media[prefix + "render"] = _tensor_chw_to_wandb_image(image[None])

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            if tb_writer:
                                tb_writer.add_images(prefix + "rend_normal", rend_normal[None], global_step=iteration)
                                tb_writer.add_images(prefix + "surf_normal", surf_normal[None], global_step=iteration)
                                tb_writer.add_images(prefix + "rend_alpha", rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            if tb_writer:
                                tb_writer.add_images(prefix + "rend_dist", rend_dist[None], global_step=iteration)

                            if wandb_run is not None and WANDB_AVAILABLE:
                                wb_media[prefix + "rend_normal"] = _tensor_chw_to_wandb_image(rend_normal[None])
                                wb_media[prefix + "surf_normal"] = _tensor_chw_to_wandb_image(surf_normal[None])
                                wb_media[prefix + "rend_alpha"] = _tensor_chw_to_wandb_image(rend_alpha[None])
                                wb_media[prefix + "rend_dist"] = _tensor_chw_to_wandb_image(rend_dist[None])
                        except Exception:
                            pass

                        if iteration == testing_iterations[0]:
                            if tb_writer:
                                tb_writer.add_images(prefix + "ground_truth", gt_image[None], global_step=iteration)
                            if wandb_run is not None and WANDB_AVAILABLE:
                                wb_media[prefix + "ground_truth"] = _tensor_chw_to_wandb_image(gt_image[None])

                        if wandb_run is not None and WANDB_AVAILABLE and wb_media:
                            wandb.log(wb_media, step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(img_b, gt_b).mean().double()

                psnr_test /= n_cam
                l1_test /= n_cam
                ssim_avg = ssim_sum / n_cam
                lpips_avg = (lpips_sum / n_cam) if lpips_ok else None
                if lpips_avg is not None:
                    print(
                        "\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} LPIPS {}".format(
                            iteration, config["name"], l1_test, psnr_test, ssim_avg, lpips_avg
                        )
                    )
                else:
                    print(
                        "\n[ITER {}] Evaluating {}: L1 {} PSNR {} SSIM {} (LPIPS n/a)".format(
                            iteration, config["name"], l1_test, psnr_test, ssim_avg
                        )
                    )
                if do_full_eval and tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_avg, iteration)
                    if lpips_avg is not None:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_avg, iteration)
                if wandb_run is not None and WANDB_AVAILABLE and (do_full_eval or do_wandb_psnr):
                    wb_metrics = {
                        config['name'] + '/loss_viewpoint - l1_loss': l1_test,
                        config['name'] + '/loss_viewpoint - psnr': psnr_test,
                        config['name'] + '/loss_viewpoint - ssim': ssim_avg,
                    }
                    if lpips_avg is not None:
                        wb_metrics[config['name'] + '/loss_viewpoint - lpips'] = lpips_avg
                    wandb.log(wb_metrics, step=iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument(
        "--save_iterations",
        nargs="*",
        type=int,
        default=[7_000, 30_000],
        help="Steps to save point_cloud.ply; empty list (--save_iterations with no numbers) saves only the final step after iterations append (non-incremental).",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--no_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="2d-gaussian-splatting")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_tags", nargs="*", default=None)
    parser.add_argument("--wandb_log_interval", type=int, default=10)
    parser.add_argument(
        "--wandb_psnr_interval",
        type=int,
        default=1000,
        help="If >0 and wandb is on, compute test/train PSNR & L1 every N steps and log to wandb (no extra images). Use 0 to disable.",
    )
    args = parser.parse_args(sys.argv[1:])
    if not getattr(args, "incremental_train", False):
        args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        full_args=args,
    )

    # All done
    print("\nTraining complete.")
