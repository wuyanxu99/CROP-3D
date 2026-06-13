#
# Trajectory-axis depth filtering and world alignment (COLMAP / 3DGS camera convention).
# Camera center: -R @ T with stored R, T as in scene/dataset_readers.py.
#

from __future__ import annotations

import numpy as np
import torch

from utils.graphics_utils import BasicPointCloud


def dist_to_line(points: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Per-point distance to infinite line (origin, axis). axis need not be unit."""
    origin = np.asarray(origin, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    v = np.asarray(points, dtype=np.float64).reshape(-1, 3) - origin
    t = (v @ axis).reshape(-1, 1)
    proj = t * axis
    diff = v - proj
    return np.linalg.norm(diff, axis=1)


def dist_to_segment(
    points: np.ndarray,
    origin: np.ndarray,
    axis: np.ndarray,
    t_min: float,
    t_max: float,
) -> np.ndarray:
    """Per-point distance to segment [origin + t_min*axis, origin + t_max*axis]."""
    origin = np.asarray(origin, dtype=np.float64)
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    v = pts - origin
    t = (v @ axis).flatten()
    t_clip = np.clip(t, t_min, t_max)
    nearest = origin + np.outer(t_clip, axis)
    return np.linalg.norm(pts - nearest, axis=1)


def _camera_centers_from_RT_list(R_list, T_list) -> np.ndarray:
    centers = []
    for R, T in zip(R_list, T_list):
        R = np.asarray(R, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        centers.append(-R @ T)
    return np.array(centers, dtype=np.float64)


def compute_trajectory_segment_from_RT(
    R_list: list,
    T_list: list,
    cameras_per_frame: int = 1,
    buffer: float = 0.0,
):
    """Returns (origin, axis_unit, t_min, t_max)."""
    if len(R_list) == 0:
        return np.zeros(3), np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0, 1.0
    K = max(1, int(cameras_per_frame))
    centers = _camera_centers_from_RT_list(R_list, T_list)
    if K <= 1 or len(centers) < 2 * K:
        origin = centers[0].copy()
        end = centers[-1].copy()
    else:
        n_frames = len(centers) // K
        origin = np.mean(centers[0:K], axis=0)
        end = np.mean(centers[(n_frames - 1) * K : n_frames * K], axis=0)
    seg_vec = end - origin
    length = float(np.linalg.norm(seg_vec))
    buf = max(0.0, float(buffer))
    if length < 1e-8:
        return origin, np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0, 0.0
    axis = seg_vec / length
    t_min = -buf
    t_max = length + buf
    return origin, axis, t_min, t_max


def compute_trajectory_segment(cam_infos, cameras_per_frame: int = 1, buffer: float = 0.0):
    """cam_infos: sequence with .R and .T (numpy)."""
    R_list = [c.R for c in cam_infos]
    T_list = [c.T for c in cam_infos]
    return compute_trajectory_segment_from_RT(R_list, T_list, cameras_per_frame, buffer)


def filter_pcd_by_axis_depth(
    pcd: BasicPointCloud,
    origin: np.ndarray,
    axis: np.ndarray,
    depth_min: float,
    depth_max: float,
    t_min: float | None = None,
    t_max: float | None = None,
) -> BasicPointCloud:
    points = np.asarray(pcd.points)
    if t_min is not None and t_max is not None:
        d = dist_to_segment(points, origin, axis, t_min, t_max)
    else:
        d = dist_to_line(points, origin, axis)
    mask = (d >= depth_min) & (d <= depth_max)
    pts = points[mask]
    colors = np.asarray(pcd.colors)[mask] if pcd.colors is not None else None
    normals = np.asarray(pcd.normals)[mask] if pcd.normals is not None else None
    return BasicPointCloud(points=pts, colors=colors, normals=normals)


def compute_trajectory_axis_and_up(cam_infos):
    """PCA trajectory direction; mean camera Y column for up. Returns (traj_axis, up_avg, origin)."""
    R_list = [np.asarray(c.R, dtype=np.float64) for c in cam_infos]
    T_list = [np.asarray(c.T, dtype=np.float64) for c in cam_infos]
    centers = _camera_centers_from_RT_list(R_list, T_list)
    origin = centers.mean(axis=0)
    centered = centers - origin
    cov = centered.T @ centered
    U, S, _ = np.linalg.svd(cov)
    if S[0] > 1e-10:
        traj_axis = U[:, 0].astype(np.float64)
    else:
        traj_axis = (-R_list[0].T[:, 2]).astype(np.float64)
    traj_axis = traj_axis / (np.linalg.norm(traj_axis) + 1e-8)
    up_avg = np.mean([R.T[:, 1] for R in R_list], axis=0).astype(np.float64)
    up_avg = up_avg / (np.linalg.norm(up_avg) + 1e-8)
    return traj_axis, up_avg, origin


def build_align_to_world_z(traj_axis: np.ndarray, up_avg: np.ndarray, origin: np.ndarray):
    """p_new = p @ R_align + t_align (row vectors). Returns (R_align 3x3, t_align 3,)."""
    z_col = np.asarray(traj_axis, dtype=np.float64)
    up = np.asarray(up_avg, dtype=np.float64)
    y_col = up - (up @ z_col) * z_col
    y_col = y_col / (np.linalg.norm(y_col) + 1e-8)
    x_col = np.cross(y_col, z_col)
    x_col = x_col / (np.linalg.norm(x_col) + 1e-8)
    R_align = np.column_stack([x_col, y_col, z_col])
    t_align = -R_align.T @ np.asarray(origin, dtype=np.float64)
    return R_align.astype(np.float64), t_align.astype(np.float64)


def apply_align_to_pcd(pcd: BasicPointCloud, R_align: np.ndarray, t_align: np.ndarray) -> BasicPointCloud:
    pts = (np.asarray(pcd.points) @ R_align) + t_align
    return BasicPointCloud(points=pts, colors=pcd.colors, normals=pcd.normals)


def apply_align_to_camera_infos(cam_infos, R_align: np.ndarray, origin: np.ndarray):
    """Updates extrinsics so the same world points map correctly after p' = p @ R_align + t_align."""
    Ra_t = np.asarray(R_align.T, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    out = []
    for cam in cam_infos:
        R_old = np.asarray(cam.R, dtype=np.float64)
        T_old = np.asarray(cam.T, dtype=np.float64)
        R_new = Ra_t @ R_old
        T_new = (R_old.T @ origin) + T_old
        out.append(cam._replace(R=R_new, T=T_new))
    return out


def prune_gaussians_by_depth(
    gaussians,
    origin: np.ndarray,
    axis: np.ndarray,
    depth_min: float,
    depth_max: float,
    t_min: float | None = None,
    t_max: float | None = None,
):
    """Remove Gaussians whose distance to trajectory segment/line is outside [depth_min, depth_max]."""
    if depth_max <= 0:
        return
    xyz = gaussians.get_xyz
    o = torch.from_numpy(np.asarray(origin, dtype=np.float32)).to(xyz.device)
    a = torch.from_numpy(np.asarray(axis, dtype=np.float32)).to(xyz.device)
    a = a / (a.norm() + 1e-8)
    v = xyz - o
    t = (v * a).sum(dim=-1)
    if t_min is not None and t_max is not None:
        t_clip = torch.clamp(t, float(t_min), float(t_max))
        nearest = o + t_clip.unsqueeze(-1) * a
        d = (xyz - nearest).norm(dim=-1)
    else:
        proj = t.unsqueeze(-1) * a
        d = (v - proj).norm(dim=-1)
    prune_mask = ~((d >= depth_min) & (d <= depth_max))
    if prune_mask.any():
        gaussians.prune_points(prune_mask)
