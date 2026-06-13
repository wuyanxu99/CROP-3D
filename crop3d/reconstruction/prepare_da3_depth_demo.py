#!/usr/bin/env python3
"""Generate DA3 depth priors for a COLMAP/2DGS scene.

Expected layout:
    scene/
      images/
        camera_1/
          frame_000001.jpg
      sparse/
        0/
          cameras.bin
          images.bin
          points3D.bin

The script writes ``*.npy`` depth maps under ``scene/depth_da3`` by default.
Edit the user settings below, or override them with command-line arguments.
"""

from __future__ import annotations

# =============================================================================
# User settings
# =============================================================================

# Required: scene root containing images/ and sparse/0/.
SOURCE_PATH = "./examples/reconstruction_demo/scene"
IMAGES = "images"
DEPTH_DIR = "depth_da3"

# Required: DA3 model source. LOCAL_MODEL_DIR is used first when it exists.
HF_MODEL_ID = "depth-anything/DA3-LARGE-1.1"
LOCAL_MODEL_DIR = "./checkpoints/DA3-LARGE-1.1"

# Align model depth to COLMAP sparse depth in meters: d_metric = scale * d_model + shift.
COLMAP_AFFINE_ALIGN = True
RANSAC_THRESHOLD = 0.05

# Inference device: cuda or cpu.
DEVICE = "cuda"
FALLBACK_CPU_IF_CUDA_FAILS = False

# Disable cuDNN when the local CUDA/cuDNN stack has convolution engine issues.
TORCH_DISABLE_CUDNN = True

SAVE_VIS = True

MIN_COLMAP_PTS = 15
WARN_ABS_ERR_M = 0.3

# Optional global intrinsics override: "fx,fy,cx,cy,...". Empty uses COLMAP cameras.bin.
CAMERA_PARAMS = "1407.0,1408.3,986.7009,518.1291,-0.0331,0.0209,0,0"

# Multi-view window inference.
WINDOW_INFERENCE = True
WINDOW_SIZE = 3
WINDOW_MODE = "hybrid"
WINDOW_MIN_SHARED_POINTS = 80
WINDOW_MAX_FRAME_DELTA = 8
WINDOW_PREFER_CROSS_CAMERA = True
WINDOW_REQUIRE_CONTEXT = False

# Pose conditioning: off, colmap, or auto.
POSE_CONDITIONING = "colmap"
POSE_ALIGN_TO_INPUT_EXT_SCALE = True
POSE_MIN_SHARED_POINTS = 120
POSE_MIN_WINDOW_VIEWS = 2

# DA3 inference options.
DA3_PROCESS_RES = 504
DA3_PROCESS_RES_METHOD = "upper_bound_resize"
DA3_REF_VIEW_STRATEGY = "saddle_balanced"

# Optional global smoothing.
GLOBAL_DEPTH_SMOOTH = False
GLOBAL_SMOOTH_LAMBDA_RATIO = 1.0
GLOBAL_SMOOTH_ADJACENCY = "none"
GLOBAL_SMOOTH_TEMPORAL_WINDOW = 1
GLOBAL_SMOOTH_MAX_OBS_PER_POINT = 12
GLOBAL_SMOOTH_RATIO_EDGE_MODE = "chain"
GLOBAL_SMOOTH_LOG_EPS = 1e-6
GLOBAL_SMOOTH_ANCHOR_WEIGHT = 10.0
GLOBAL_RAW_TMP_DIR = "_depth_da3_raw_tmp"
SAVE_WORLD_NORMALS = True
NORMAL_WORLD_DIR = "depth_da3_normal_world"
IVD_SAMPLE_COUNT = 1000
IVD_SAMPLE_SEED = 0

# =============================================================================

import argparse
import importlib.util
import os
import re
import shutil
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def _apply_da3_torch_cuda_backends() -> None:
    env = os.environ.get("DA3_TORCH_DISABLE_CUDNN", "").strip().lower()
    # Explicit "0"/"false"/"no" keeps cuDNN enabled.
    if env in ("0", "false", "no"):
        return
    if not TORCH_DISABLE_CUDNN and env not in ("1", "true", "yes"):
        return
    torch.backends.cudnn.enabled = False
    print(
        "[INFO ] cuDNN disabled; CUDA convolutions will use non-cuDNN kernels. "
        "Set TORCH_DISABLE_CUDNN=False or DA3_TORCH_DISABLE_CUDNN=0 to enable cuDNN."
    )


_apply_da3_torch_cuda_backends()

# Load colmap_loader directly to avoid importing 2DGS CUDA extensions.
_REPO_ROOT = Path(__file__).resolve().parent
_colmap_path = _REPO_ROOT / "twodgs" / "scene" / "colmap_loader.py"
_spec = importlib.util.spec_from_file_location("_da3_colmap_loader", _colmap_path)
_colmap_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_colmap_mod)
qvec2rotmat = _colmap_mod.qvec2rotmat
read_extrinsics_binary = _colmap_mod.read_extrinsics_binary
read_extrinsics_text = _colmap_mod.read_extrinsics_text
read_intrinsics_binary = _colmap_mod.read_intrinsics_binary
read_intrinsics_text = _colmap_mod.read_intrinsics_text


def _resolve_device(requested: str, *, allow_cpu_fallback: bool) -> str:
    """Check CUDA before loading large model weights."""
    if requested != "cuda":
        return requested
    if not torch.cuda.is_available():
        msg = '[ERROR] torch.cuda.is_available() is False. Install GPU PyTorch or set DEVICE = "cpu".'
        if allow_cpu_fallback:
            print(f"{msg}\n[WARN] Falling back to CPU.\n")
            return "cpu"
        raise SystemExit(msg)
    try:
        x = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        del x
    except RuntimeError as e:
        hint = (
            "\n[ERROR] CUDA failed to initialize. This often means the NVIDIA driver is older "
            "than the CUDA runtime used by PyTorch.\n"
            "Options:\n"
            "  1) Update the NVIDIA driver: https://www.nvidia.com/Download/index.aspx\n"
            "  2) Reinstall PyTorch with a CUDA build compatible with your driver, for example:\n"
            "       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121\n"
            "     Use cu118 or another compatible wheel when needed.\n"
            '  3) Set DEVICE = "cpu" or pass --device cpu for a slow CPU-only run.\n'
            "  4) Set FALLBACK_CPU_IF_CUDA_FAILS = True or pass --fallback-cpu.\n"
            f"\nOriginal error: {e}\n"
        )
        if allow_cpu_fallback:
            print(hint)
            print("[WARN] Falling back to CPU.\n")
            return "cpu"
        raise SystemExit(hint)
    return "cuda"


# ---------------------------------------------------------------------------
# COLMAP loading
# ---------------------------------------------------------------------------

def _read_points3d_binary(path: str) -> Dict[int, np.ndarray]:
    points = {}
    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_points):
            data = struct.unpack("<QdddBBBd", f.read(43))
            point_id = int(data[0])
            xyz = np.array(data[1:4], dtype=np.float64)
            track_len = struct.unpack("<Q", f.read(8))[0]
            if track_len > 0:
                f.read(8 * track_len)
            points[point_id] = xyz
    return points


def _read_points3d_text(path: str) -> Dict[int, np.ndarray]:
    points = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            items = line.split()
            point_id = int(items[0])
            xyz = np.array([float(items[1]), float(items[2]), float(items[3])], dtype=np.float64)
            points[point_id] = xyz
    return points


def _normalize_rel(name: str) -> str:
    return (name or "").replace("\\", "/").strip().lstrip("/")


def _load_colmap_sparse(sparse_dir: Path):
    """Load COLMAP extrinsics, intrinsics, and 3D points."""
    extrinsics = (
        read_extrinsics_binary(str(sparse_dir / "images.bin"))
        if (sparse_dir / "images.bin").exists()
        else read_extrinsics_text(str(sparse_dir / "images.txt"))
    )
    cam_intrinsics = (
        read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
        if (sparse_dir / "cameras.bin").exists()
        else read_intrinsics_text(str(sparse_dir / "cameras.txt"))
    )
    points3d = (
        _read_points3d_binary(str(sparse_dir / "points3D.bin"))
        if (sparse_dir / "points3D.bin").exists()
        else _read_points3d_text(str(sparse_dir / "points3D.txt"))
    )
    return extrinsics, cam_intrinsics, points3d


def _build_K(cam_intr, camera_params_override: str = "") -> Tuple[np.ndarray, int, int]:
    """Build a 3x3 intrinsic matrix and return (K, width, height)."""
    if camera_params_override and camera_params_override.strip():
        vals = [float(x.strip()) for x in camera_params_override.split(",") if x.strip()]
        if len(vals) < 4:
            raise ValueError(f"camera_params requires at least fx,fy,cx,cy. Got: {camera_params_override}")
        fx, fy, cx, cy = vals[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        return K, cam_intr.width, cam_intr.height

    p = cam_intr.params
    if cam_intr.model == "SIMPLE_PINHOLE":
        K = np.array([[p[0], 0, p[1]], [0, p[0], p[2]], [0, 0, 1]], dtype=np.float32)
    elif cam_intr.model == "PINHOLE":
        K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]], dtype=np.float32)
    else:
        # Fallback for unsupported models.
        f = float(max(cam_intr.width, cam_intr.height))
        K = np.array(
            [[f, 0, cam_intr.width / 2.0], [0, f, cam_intr.height / 2.0], [0, 0, 1]],
            dtype=np.float32,
        )
        print(f"  [WARN] Camera model {cam_intr.model} not fully supported, using estimated focal.")
    return K, cam_intr.width, cam_intr.height


# ---------------------------------------------------------------------------
# COLMAP affine alignment
# ---------------------------------------------------------------------------

def _robust_affine_fit(
    x: np.ndarray, y: np.ndarray, ransac_threshold: float, min_colmap_pts: int
) -> Tuple[float, float, int]:
    """Fit y ~= s*x + t with one residual-based inlier refinement."""
    if x.size < min_colmap_pts:
        return 1.0, 0.0, 0

    A = np.stack([x, np.ones_like(x)], axis=1)
    s, t = np.linalg.lstsq(A, y, rcond=None)[0]
    pred = s * x + t
    resid = np.abs(pred - y)
    inliers = resid < ransac_threshold

    if int(inliers.sum()) >= min_colmap_pts:
        A2 = np.stack([x[inliers], np.ones_like(x[inliers])], axis=1)
        s, t = np.linalg.lstsq(A2, y[inliers], rcond=None)[0]
        return float(s), float(t), int(inliers.sum())
    return float(s), float(t), int(x.size)


def _colmap_affine_to_meters(
    depth_raw: np.ndarray,
    image_obj,
    points3d: Dict[int, np.ndarray],
    ransac_threshold: float,
    min_colmap_pts: int,
) -> Tuple[np.ndarray, float, float, int]:
    """Fit d_colmap = scale * d_model + shift at sparse observations."""
    xys = image_obj.xys
    pids = image_obj.point3D_ids
    valid = pids >= 0
    if not np.any(valid):
        return depth_raw.astype(np.float32), 1.0, 0.0, 0

    xys = xys[valid]
    pids = pids[valid]

    colmap_depth: list = []
    model_sparse: list = []
    R = qvec2rotmat(image_obj.qvec)
    t = image_obj.tvec
    h, w = depth_raw.shape
    for xy, pid in zip(xys, pids):
        xyz = points3d.get(int(pid))
        if xyz is None:
            continue
        z_colmap = float((R @ xyz + t)[2])
        if z_colmap <= 0:
            continue
        u = int(round(float(xy[0])))
        v = int(round(float(xy[1])))
        if u < 0 or u >= w or v < 0 or v >= h:
            continue
        d = float(depth_raw[v, u])
        if not np.isfinite(d):
            continue
        colmap_depth.append(z_colmap)
        model_sparse.append(d)

    if len(colmap_depth) < min_colmap_pts:
        return depth_raw.astype(np.float32), 1.0, 0.0, len(colmap_depth)

    y = np.array(colmap_depth, dtype=np.float64)
    x = np.array(model_sparse, dtype=np.float64)
    scale, shift, inlier_num = _robust_affine_fit(x, y, ransac_threshold, min_colmap_pts)
    aligned = scale * depth_raw + shift
    aligned = np.where(aligned > 0.0, aligned, 0.0).astype(np.float32)
    return aligned, scale, shift, inlier_num


# ---------------------------------------------------------------------------
# COLMAP consistency check
# ---------------------------------------------------------------------------

def _colmap_consistency_check(
    depth: np.ndarray,
    image_obj,
    points3d: Dict[int, np.ndarray],
    min_pts: int,
) -> Tuple[int, float, float]:
    """Compare DA3 depth against COLMAP sparse depths."""
    xys = image_obj.xys
    pids = image_obj.point3D_ids
    valid = pids >= 0
    if not np.any(valid):
        return 0, float("inf"), float("inf")

    xys = xys[valid]
    pids = pids[valid]
    R = qvec2rotmat(image_obj.qvec)
    t = image_obj.tvec
    h, w = depth.shape

    colmap_d, da3_d = [], []
    for xy, pid in zip(xys, pids):
        xyz = points3d.get(int(pid))
        if xyz is None:
            continue
        z = float((R @ xyz + t)[2])
        if z <= 0:
            continue
        u, v = int(round(float(xy[0]))), int(round(float(xy[1])))
        if not (0 <= u < w and 0 <= v < h):
            continue
        d = float(depth[v, u])
        if not np.isfinite(d) or d <= 0:
            continue
        colmap_d.append(z)
        da3_d.append(d)

    n = len(colmap_d)
    if n < min_pts:
        return n, float("inf"), float("inf")

    y = np.array(colmap_d, dtype=np.float64)
    x = np.array(da3_d, dtype=np.float64)
    abs_err = np.abs(x - y)
    rel_err = abs_err / np.maximum(y, 1e-3)
    return n, float(rel_err.mean()), float(abs_err.mean())


# ---------------------------------------------------------------------------
# Global smoothing helpers
# ---------------------------------------------------------------------------


def _parse_frame_index_from_rel(rel: str) -> Optional[int]:
    base = rel.split("/")[-1]
    m = re.search(r"(\d+)", base)
    return int(m.group(1)) if m else None


def _adjacency_ratio_ok(
    name_i: str,
    name_j: str,
    cam_i: int,
    cam_j: int,
    adjacency: str,
    temporal_window: int,
) -> bool:
    if adjacency == "none":
        return True
    if adjacency != "same_frame_and_temporal":
        raise ValueError(f"unknown adjacency: {adjacency}")
    fi = _parse_frame_index_from_rel(name_i)
    fj = _parse_frame_index_from_rel(name_j)
    if fi is None or fj is None:
        return True
    if fi == fj and cam_i != cam_j:
        return True
    if cam_i == cam_j and fi != fj and abs(fi - fj) <= temporal_window:
        return True
    return False


def _sample_d_from_mmap(
    mm: np.ndarray, u: int, v: int
) -> Optional[float]:
    h, w = mm.shape
    if u < 0 or u >= w or v < 0 or v >= h:
        return None
    d = float(mm[v, u])
    if not np.isfinite(d) or d <= 0:
        return None
    return d


def _collect_sparse_observations_memmap(
    ordered_image_ids: List[int],
    depth_paths: Dict[int, Path],
    extrinsics,
    points3d: Dict[int, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Read sparse-observation depths from per-image memmaps."""
    j_list: List[int] = []
    d_list: List[float] = []
    z_list: List[float] = []
    pid_list: List[int] = []
    u_list: List[float] = []
    v_list: List[float] = []
    cam_list: List[int] = []
    frame_list: List[int] = []
    rel_list: List[str] = []

    id_to_j = {img_id: j for j, img_id in enumerate(ordered_image_ids)}
    for img_id in ordered_image_ids:
        j = id_to_j[img_id]
        image_obj = extrinsics[img_id]
        rel = _normalize_rel(image_obj.name)
        cam = int(image_obj.camera_id)
        fi = _parse_frame_index_from_rel(rel)
        fr = int(fi) if fi is not None else -1
        path = depth_paths[img_id]
        mm = np.load(str(path), mmap_mode="r")
        if mm.ndim != 2:
            mm = np.asarray(mm)
        R = qvec2rotmat(image_obj.qvec)
        t = image_obj.tvec
        xys = image_obj.xys
        pids = image_obj.point3D_ids
        valid = pids >= 0
        for xy, pid in zip(xys[valid], pids[valid]):
            xyz = points3d.get(int(pid))
            if xyz is None:
                continue
            z_colmap = float((R @ xyz + t)[2])
            if z_colmap <= 0:
                continue
            u = int(round(float(xy[0])))
            v = int(round(float(xy[1])))
            got = _sample_d_from_mmap(mm, u, v)
            if got is None:
                continue
            j_list.append(j)
            d_list.append(got)
            z_list.append(z_colmap)
            pid_list.append(int(pid))
            u_list.append(float(xy[0]))
            v_list.append(float(xy[1]))
            cam_list.append(cam)
            frame_list.append(fr)
            rel_list.append(rel)
        del mm

    return (
        np.array(j_list, dtype=np.int32),
        np.array(d_list, dtype=np.float64),
        np.array(z_list, dtype=np.float64),
        np.array(pid_list, dtype=np.int64),
        np.array(u_list, dtype=np.float64),
        np.array(v_list, dtype=np.float64),
        np.array(cam_list, dtype=np.int32),
        np.array(frame_list, dtype=np.int32),
        rel_list,
    )


def _warm_start_sparse(
    obs_j: np.ndarray,
    obs_d: np.ndarray,
    obs_z: np.ndarray,
    n: int,
    min_colmap_pts: int,
    ransac_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    warm_s = np.ones(n, dtype=np.float64)
    warm_t = np.zeros(n, dtype=np.float64)
    n_obs = np.bincount(obs_j.astype(np.int64), minlength=n).astype(np.int32)
    for j in range(n):
        mask = obs_j == j
        c = int(mask.sum())
        if c < min_colmap_pts:
            continue
        x = obs_d[mask]
        y = obs_z[mask]
        s, t, _ = _robust_affine_fit(x, y, ransac_threshold, min_colmap_pts)
        warm_s[j] = s
        warm_t[j] = t
    return warm_s, warm_t, n_obs


def _build_ratio_arrays(
    ordered_image_ids: List[int],
    obs_j: np.ndarray,
    obs_d: np.ndarray,
    obs_z: np.ndarray,
    obs_pid: np.ndarray,
    obs_rel: List[str],
    obs_cam: np.ndarray,
    adjacency: str,
    temporal_window: int,
    max_obs_per_point: int,
    log_eps: float,
    edge_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """edge_mode: chain | all_pairs"""
    pid_to_idx: Dict[int, List[int]] = defaultdict(list)
    for k in range(obs_j.shape[0]):
        pid_to_idx[int(obs_pid[k])].append(k)

    ri_list: List[int] = []
    rj_list: List[int] = []
    rdi_list: List[float] = []
    rdj_list: List[float] = []
    rlog_list: List[float] = []
    rng = np.random.default_rng(0)

    def _try_add_edge(ia: int, ib: int) -> None:
        ja, jb = int(obs_j[ia]), int(obs_j[ib])
        if ja == jb:
            return
        name_a = obs_rel[ia]
        name_b = obs_rel[ib]
        if not _adjacency_ratio_ok(
            name_a, name_b, int(obs_cam[ia]), int(obs_cam[ib]), adjacency, temporal_window
        ):
            return
        za, zb = float(obs_z[ia]), float(obs_z[ib])
        if za <= log_eps or zb <= log_eps:
            return
        log_tgt = np.log(za + log_eps) - np.log(zb + log_eps)
        ri_list.append(ja)
        rj_list.append(jb)
        rdi_list.append(float(obs_d[ia]))
        rdj_list.append(float(obs_d[ib]))
        rlog_list.append(float(log_tgt))

    for _pid, idxs in pid_to_idx.items():
        if len(idxs) < 2:
            continue
        rows = list(idxs)
        if len(rows) > max_obs_per_point:
            sel = rng.choice(len(rows), size=max_obs_per_point, replace=False)
            rows = [rows[int(s)] for s in sel]
        if edge_mode == "chain":
            rows.sort(key=lambda kk: ordered_image_ids[int(obs_j[kk])])
            for t in range(len(rows) - 1):
                _try_add_edge(rows[t], rows[t + 1])
        elif edge_mode == "all_pairs":
            m = len(rows)
            for a in range(m):
                for b in range(a + 1, m):
                    _try_add_edge(rows[a], rows[b])
        else:
            raise ValueError(f"unknown edge_mode: {edge_mode}")

    if not ri_list:
        z = np.zeros(0, dtype=np.int32)
        return z, z, np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    return (
        np.array(ri_list, dtype=np.int32),
        np.array(rj_list, dtype=np.int32),
        np.array(rdi_list, dtype=np.float64),
        np.array(rdj_list, dtype=np.float64),
        np.array(rlog_list, dtype=np.float64),
    )


def _pack_st(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    x = np.empty(2 * s.size, dtype=np.float64)
    x[0::2] = s
    x[1::2] = t
    return x


def _unpack_st(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return x[0::2].copy(), x[1::2].copy()


def _solve_global_affine_sparse(
    ordered_image_ids: List[int],
    obs_j: np.ndarray,
    obs_d: np.ndarray,
    obs_z: np.ndarray,
    ri: np.ndarray,
    rj: np.ndarray,
    rdi: np.ndarray,
    rdj: np.ndarray,
    rlog: np.ndarray,
    n: int,
    warm_s: np.ndarray,
    warm_t: np.ndarray,
    n_obs_per: np.ndarray,
    *,
    lambda_ratio: float,
    log_eps: float,
    anchor_weight: float,
    min_colmap_pts: int,
) -> Tuple[Dict[int, Tuple[float, float]], dict]:
    from scipy.optimize import least_squares

    n_abs = int(obs_j.size)
    n_ratio = int(ri.size)
    sqrt_lam = float(np.sqrt(max(lambda_ratio, 0.0)))
    sqrt_anchor = float(np.sqrt(max(anchor_weight, 0.0)))

    def residual_vec(p: np.ndarray) -> np.ndarray:
        s, t = _unpack_st(p)
        blocks: List[np.ndarray] = []
        if n_abs > 0:
            pred = s[obs_j] * obs_d + t[obs_j]
            blocks.append(pred - obs_z)
        if n_ratio > 0 and sqrt_lam > 0.0:
            zi_hat = np.maximum(s[ri] * rdi + t[ri], log_eps)
            zj_hat = np.maximum(s[rj] * rdj + t[rj], log_eps)
            lr = np.log(zi_hat) - np.log(zj_hat) - rlog
            blocks.append(sqrt_lam * lr)
        if sqrt_anchor > 0.0:
            anch_s, anch_t = [], []
            for k in range(n):
                if n_obs_per[k] < min_colmap_pts:
                    anch_s.append(s[k] - warm_s[k])
                    anch_t.append(t[k] - warm_t[k])
            if anch_s:
                blocks.append(sqrt_anchor * np.array(anch_s, dtype=np.float64))
                blocks.append(sqrt_anchor * np.array(anch_t, dtype=np.float64))
        if not blocks:
            return np.zeros(1, dtype=np.float64)
        return np.concatenate(blocks)

    x0 = _pack_st(warm_s, warm_t)
    r0 = residual_vec(x0)
    cost0 = float(np.dot(r0, r0))
    n_anchor_imgs = int((n_obs_per < min_colmap_pts).sum())
    n_res = n_abs + n_ratio + 2 * n_anchor_imgs
    ls_kw: Dict[str, object] = {
        "method": "trf",
        "max_nfev": max(200, min(8000, 50 * n + n_res // 500)),
        "verbose": 0,
    }
    if n_res > 50_000:
        ls_kw["tr_solver"] = "lsmr"
    result = least_squares(residual_vec, x0, **ls_kw)
    s_opt, t_opt = _unpack_st(result.x)
    r1 = residual_vec(result.x)
    cost1 = float(np.dot(r1, r1))
    info = {
        "success": bool(result.success),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "cost_initial": cost0,
        "cost_final": cost1,
        "n_abs": n_abs,
        "n_ratio": n_ratio,
        "n_obs_per": [int(x) for x in n_obs_per.tolist()],
    }
    out_map = {ordered_image_ids[k]: (float(s_opt[k]), float(t_opt[k])) for k in range(n)}
    return out_map, info


def _world_from_cam_z(
    u: float,
    v: float,
    z: float,
    K: np.ndarray,
    R_w2c: np.ndarray,
    t_w2c: np.ndarray,
) -> np.ndarray:
    """COLMAP uses X_c = R X_w + t; pixel depth is camera-space Z."""
    Kinv = np.linalg.inv(K.astype(np.float64))
    uv1 = np.array([u, v, 1.0], dtype=np.float64)
    x_c = z * (Kinv @ uv1)
    R = R_w2c.astype(np.float64)
    t = t_w2c.astype(np.float64)
    x_w = R.T @ (x_c - t)
    return x_w


def _compute_ivd(
    ordered_image_ids: List[int],
    extrinsics,
    cam_intrinsics,
    camera_params: str,
    obs_j: np.ndarray,
    obs_d: np.ndarray,
    obs_pid: np.ndarray,
    obs_u: np.ndarray,
    obs_v: np.ndarray,
    warm_s: np.ndarray,
    warm_t: np.ndarray,
    s_opt: np.ndarray,
    t_opt: np.ndarray,
    sample_n: int,
    seed: int,
) -> Tuple[float, float, int]:
    """Measure multi-view back-projection dispersion for sampled shared points."""
    pid_to_idxs: Dict[int, List[int]] = defaultdict(list)
    for k in range(obs_j.shape[0]):
        pid_to_idxs[int(obs_pid[k])].append(k)
    candidates = [p for p, idxs in pid_to_idxs.items() if len(idxs) >= 2]
    if not candidates:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    take = min(sample_n, len(candidates))
    chosen = rng.choice(np.array(candidates, dtype=np.int64), size=take, replace=False)

    K_cache: Dict[int, Tuple[np.ndarray, int, int]] = {}

    def get_K_cam(j: int) -> Tuple[np.ndarray, int, int]:
        img_id = ordered_image_ids[j]
        cam_id = int(extrinsics[img_id].camera_id)
        if cam_id not in K_cache:
            K_cache[cam_id] = _build_K(cam_intrinsics[cam_id], camera_params)
        return K_cache[cam_id]

    stds_before: List[float] = []
    stds_after: List[float] = []

    for pid in chosen:
        idxs = pid_to_idxs[int(pid)]
        Xb: List[np.ndarray] = []
        Xa: List[np.ndarray] = []
        for k in idxs:
            j = int(obs_j[k])
            img_id = ordered_image_ids[j]
            im = extrinsics[img_id]
            R = qvec2rotmat(im.qvec)
            t = im.tvec
            K, _, _ = get_K_cam(j)
            u = float(obs_u[k])
            v = float(obs_v[k])
            d_raw = float(obs_d[k])
            zb = warm_s[j] * d_raw + warm_t[j]
            za = s_opt[j] * d_raw + t_opt[j]
            if zb <= 0 or za <= 0:
                continue
            Xb.append(_world_from_cam_z(u, v, zb, K, R, t))
            Xa.append(_world_from_cam_z(u, v, za, K, R, t))
        if len(Xb) < 2:
            continue
        Pb = np.stack(Xb, axis=0)
        Pa = np.stack(Xa, axis=0)
        sb = float(np.sqrt(np.mean(np.var(Pb, axis=0) ** 2)))
        sa = float(np.sqrt(np.mean(np.var(Pa, axis=0) ** 2)))
        stds_before.append(sb)
        stds_after.append(sa)

    if not stds_before:
        return float("nan"), float("nan"), 0
    return float(np.mean(stds_before)), float(np.mean(stds_after)), len(stds_before)


def _depth_to_world_normals(
    depth: np.ndarray,
    K: np.ndarray,
    R_w2c: np.ndarray,
    t_w2c: np.ndarray,
) -> np.ndarray:
    """Convert an aligned depth map to world-space unit normals."""
    del t_w2c
    h, w = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    v_coords, u_coords = np.indices((h, w), dtype=np.float64)
    z = depth.astype(np.float64)
    xc = (u_coords - cx) / fx * z
    yc = (v_coords - cy) / fy * z
    zc = z
    Pc = np.stack([xc, yc, zc], axis=-1)
    dPdu = np.gradient(Pc, axis=1)
    dPdv = np.gradient(Pc, axis=0)
    Nc = np.cross(dPdu, dPdv)
    ln = np.linalg.norm(Nc, axis=-1, keepdims=True)
    Nc = np.where(ln > 1e-12, Nc / np.maximum(ln, 1e-12), 0.0)
    R = R_w2c.astype(np.float64)
    Nw = (R.T @ Nc.reshape(-1, 3).T).T.reshape(h, w, 3)
    ln2 = np.linalg.norm(Nw, axis=-1, keepdims=True)
    Nw = np.where(ln2 > 1e-12, Nw / np.maximum(ln2, 1e-12), 0.0)
    valid = depth > 0
    Nw = np.where(valid[..., None], Nw, 0.0)
    return Nw.astype(np.float32)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _save_vis(depth: np.ndarray, vis_path: Path) -> None:
    valid = depth > 0
    if not np.any(valid):
        Image.fromarray(np.zeros_like(depth, dtype=np.uint8)).save(vis_path)
        return
    vmin = float(np.percentile(depth[valid], 2))
    vmax = float(np.percentile(depth[valid], 98))
    vis = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    Image.fromarray((vis * 255).astype(np.uint8)).save(vis_path)


@dataclass
class ImageRecord:
    img_id: int
    rel: str
    image_path: Path
    camera_id: int
    frame_idx: int
    K: np.ndarray
    orig_h: int
    orig_w: int
    ext_w2c: np.ndarray
    valid_pids: frozenset[int]
    exists: bool


def _image_ext_w2c(image_obj) -> np.ndarray:
    ext = np.eye(4, dtype=np.float32)
    ext[:3, :3] = qvec2rotmat(image_obj.qvec).astype(np.float32)
    ext[:3, 3] = np.asarray(image_obj.tvec, dtype=np.float32)
    return ext


def _build_image_records(
    extrinsics,
    cam_intrinsics,
    image_dir: Path,
    camera_params: str,
) -> Tuple[List[ImageRecord], Dict[int, ImageRecord]]:
    records: List[ImageRecord] = []
    rec_map: Dict[int, ImageRecord] = {}
    ordered = sorted(extrinsics.values(), key=lambda im: (_normalize_rel(im.name), int(im.id)))
    for image_obj in ordered:
        rel = _normalize_rel(image_obj.name)
        image_path = image_dir / rel
        K, orig_w, orig_h = _build_K(cam_intrinsics[image_obj.camera_id], camera_params)
        pids = image_obj.point3D_ids
        valid = pids >= 0
        valid_pids = frozenset(int(pid) for pid in pids[valid].tolist())
        frame_idx = _parse_frame_index_from_rel(rel)
        rec = ImageRecord(
            img_id=int(image_obj.id),
            rel=rel,
            image_path=image_path,
            camera_id=int(image_obj.camera_id),
            frame_idx=(int(frame_idx) if frame_idx is not None else -1),
            K=K.astype(np.float32),
            orig_h=int(orig_h),
            orig_w=int(orig_w),
            ext_w2c=_image_ext_w2c(image_obj),
            valid_pids=valid_pids,
            exists=image_path.exists(),
        )
        records.append(rec)
        rec_map[rec.img_id] = rec
    return records, rec_map


def _shared_pid_count(a: ImageRecord, b: ImageRecord) -> int:
    if not a.valid_pids or not b.valid_pids:
        return 0
    if len(a.valid_pids) > len(b.valid_pids):
        a, b = b, a
    return sum(1 for pid in a.valid_pids if pid in b.valid_pids)


def _candidate_window_score(
    target: ImageRecord,
    cand: ImageRecord,
    args,
) -> Tuple[Optional[Tuple[int, int, int, int, int]], int]:
    if cand.img_id == target.img_id or not cand.exists:
        return None, 0

    shared = _shared_pid_count(target, cand)
    known_dt = target.frame_idx >= 0 and cand.frame_idx >= 0
    dt = abs(target.frame_idx - cand.frame_idx) if known_dt else 10**9
    temporal_ok = int(known_dt and dt <= args.window_max_frame_delta)
    shared_ok = int(shared >= args.window_min_shared_points)
    cross_ok = int(target.camera_id != cand.camera_id)
    cross_pref = cross_ok if args.window_prefer_cross_camera else 0

    mode = args.window_mode
    if mode == "temporal":
        if not temporal_ok:
            return None, shared
        score = (temporal_ok, cross_pref, -dt, shared, -cand.img_id)
    elif mode == "shared_points":
        if shared <= 0:
            return None, shared
        score = (shared_ok, shared, cross_pref, temporal_ok, -dt)
    elif mode == "hybrid":
        if shared <= 0 and not temporal_ok:
            return None, shared
        score = (shared_ok, shared, cross_pref, temporal_ok, -dt)
    else:
        raise ValueError(f"unknown window_mode: {mode}")
    return score, shared


def _select_context_views(
    target: ImageRecord,
    records: List[ImageRecord],
    args,
) -> List[Tuple[ImageRecord, int]]:
    need = max(0, int(args.window_size) - 1)
    if need <= 0:
        return []
    ranked: List[Tuple[Tuple[int, int, int, int, int], int, ImageRecord]] = []
    for cand in records:
        score, shared = _candidate_window_score(target, cand, args)
        if score is None:
            continue
        ranked.append((score, shared, cand))
    ranked.sort(key=lambda item: item[0], reverse=True)
    picked = [(cand, shared) for _, shared, cand in ranked[:need]]
    if args.window_require_context and len(picked) < need:
        return []
    return picked


def _should_use_pose_conditioning(
    shared_with_target: List[int],
    window_size: int,
    args,
) -> bool:
    mode = args.pose_conditioning
    if mode == "off":
        return False
    if window_size < max(1, int(args.pose_min_window_views)):
        return False
    if mode == "colmap":
        return True
    if mode != "auto":
        raise ValueError(f"unknown pose_conditioning: {mode}")
    if not shared_with_target:
        return False
    return max(shared_with_target) >= int(args.pose_min_shared_points)


def _resize_depth_to_hw(depth_proc: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
    if depth_proc.shape != (orig_h, orig_w):
        depth_proc = cv2.resize(depth_proc, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(depth_proc, 0.0, None).astype(np.float32)


def _infer_depth_single(model, record: ImageRecord, args) -> np.ndarray:
    prediction = model.inference(
        [str(record.image_path)],
        intrinsics=record.K.reshape(1, 3, 3),
        process_res=int(args.da3_process_res),
        process_res_method=args.da3_process_res_method,
    )
    return _resize_depth_to_hw(prediction.depth[0].astype(np.float32), record.orig_h, record.orig_w)


def _infer_depth_window(
    model,
    window_records: List[ImageRecord],
    use_pose_conditioning: bool,
    args,
) -> np.ndarray:
    image_list = [str(rec.image_path) for rec in window_records]
    intrinsics = np.stack([rec.K for rec in window_records], axis=0).astype(np.float32)
    infer_kwargs = {
        "image": image_list,
        "intrinsics": intrinsics,
        "process_res": int(args.da3_process_res),
        "process_res_method": args.da3_process_res_method,
        "ref_view_strategy": args.da3_ref_view_strategy,
    }
    if use_pose_conditioning:
        infer_kwargs["extrinsics"] = np.stack(
            [rec.ext_w2c for rec in window_records], axis=0
        ).astype(np.float32)
        infer_kwargs["align_to_input_ext_scale"] = bool(args.pose_align_to_input_ext_scale)
    prediction = model.inference(**infer_kwargs)
    # The first window image is always the target image.
    return _resize_depth_to_hw(
        prediction.depth[0].astype(np.float32),
        window_records[0].orig_h,
        window_records[0].orig_w,
    )


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        "Generate DA3 depth priors with optional COLMAP affine alignment in meters.\n"
        "Edit the user settings at the top of the file, or override them here."
    )
    parser.add_argument("--source_path", default=SOURCE_PATH, type=str)
    parser.add_argument("--images", default=IMAGES, type=str)
    parser.add_argument("--depth_dir", default=DEPTH_DIR, type=str)
    parser.add_argument("--hf_model_id", default=HF_MODEL_ID, type=str)
    parser.add_argument("--local_model_dir", default=LOCAL_MODEL_DIR, type=str)
    parser.add_argument("--device", default=DEVICE, type=str)
    parser.add_argument(
        "--fallback-cpu",
        default=FALLBACK_CPU_IF_CUDA_FAILS,
        action=argparse.BooleanOptionalAction,
        help="Fall back to CPU if CUDA initialization fails.",
    )
    parser.add_argument("--min_colmap_pts", default=MIN_COLMAP_PTS, type=int)
    parser.add_argument("--ransac_threshold", default=RANSAC_THRESHOLD, type=float)
    parser.add_argument(
        "--colmap_affine_align",
        default=COLMAP_AFFINE_ALIGN,
        action=argparse.BooleanOptionalAction,
        help="Align model depth to COLMAP scale with d_m = s*d_model+t.",
    )
    parser.add_argument("--warn_abs_err_m", default=WARN_ABS_ERR_M, type=float)
    parser.add_argument(
        "--save_vis",
        default=SAVE_VIS,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--camera_params", default=CAMERA_PARAMS, type=str)
    parser.add_argument(
        "--window-inference",
        default=WINDOW_INFERENCE,
        action=argparse.BooleanOptionalAction,
        help="Run DA3 with target plus context images.",
    )
    parser.add_argument("--window-size", default=WINDOW_SIZE, type=int)
    parser.add_argument(
        "--window-mode",
        default=WINDOW_MODE,
        choices=["shared_points", "temporal", "hybrid"],
    )
    parser.add_argument("--window-min-shared-points", default=WINDOW_MIN_SHARED_POINTS, type=int)
    parser.add_argument("--window-max-frame-delta", default=WINDOW_MAX_FRAME_DELTA, type=int)
    parser.add_argument(
        "--window-prefer-cross-camera",
        default=WINDOW_PREFER_CROSS_CAMERA,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--window-require-context",
        default=WINDOW_REQUIRE_CONTEXT,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--pose-conditioning",
        default=POSE_CONDITIONING,
        choices=["off", "colmap", "auto"],
    )
    parser.add_argument(
        "--pose-align-to-input-ext-scale",
        default=POSE_ALIGN_TO_INPUT_EXT_SCALE,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--pose-min-shared-points", default=POSE_MIN_SHARED_POINTS, type=int)
    parser.add_argument("--pose-min-window-views", default=POSE_MIN_WINDOW_VIEWS, type=int)
    parser.add_argument("--da3-process-res", default=DA3_PROCESS_RES, type=int)
    parser.add_argument("--da3-process-res-method", default=DA3_PROCESS_RES_METHOD, type=str)
    parser.add_argument("--da3-ref-view-strategy", default=DA3_REF_VIEW_STRATEGY, type=str)
    parser.add_argument(
        "--global-depth-smooth",
        default=GLOBAL_DEPTH_SMOOTH,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--global-smooth-lambda-ratio", default=GLOBAL_SMOOTH_LAMBDA_RATIO, type=float)
    parser.add_argument(
        "--global-smooth-adjacency",
        default=GLOBAL_SMOOTH_ADJACENCY,
        choices=["none", "same_frame_and_temporal"],
    )
    parser.add_argument("--global-smooth-temporal-window", default=GLOBAL_SMOOTH_TEMPORAL_WINDOW, type=int)
    parser.add_argument("--global-smooth-max-obs-per-point", default=GLOBAL_SMOOTH_MAX_OBS_PER_POINT, type=int)
    parser.add_argument(
        "--global-smooth-ratio-edge-mode",
        default=GLOBAL_SMOOTH_RATIO_EDGE_MODE,
        choices=["chain", "all_pairs"],
        help="chain links adjacent views per 3D point; all_pairs links every pair.",
    )
    parser.add_argument("--global-smooth-log-eps", default=GLOBAL_SMOOTH_LOG_EPS, type=float)
    parser.add_argument("--global-smooth-anchor-weight", default=GLOBAL_SMOOTH_ANCHOR_WEIGHT, type=float)
    parser.add_argument("--global-raw-tmp-dir", default=GLOBAL_RAW_TMP_DIR, type=str)
    parser.add_argument(
        "--save-world-normals",
        default=SAVE_WORLD_NORMALS,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--normal-world-dir", default=NORMAL_WORLD_DIR, type=str)
    parser.add_argument("--ivd-sample-count", default=IVD_SAMPLE_COUNT, type=int)
    parser.add_argument("--ivd-sample-seed", default=IVD_SAMPLE_SEED, type=int)
    args = parser.parse_args()

    args.device = _resolve_device(
        args.device, allow_cpu_fallback=bool(args.fallback_cpu)
    )

    source_path = Path(args.source_path).expanduser().resolve()
    sparse_dir = source_path / "sparse" / "0"
    image_dir = source_path / args.images
    out_dir = source_path / args.depth_dir
    vis_dir = source_path / f"{args.depth_dir}_vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load COLMAP data.
    print("Loading COLMAP sparse reconstruction ...")
    extrinsics, cam_intrinsics, points3d = _load_colmap_sparse(sparse_dir)
    print(f"  extrinsics: {len(extrinsics)} images, cameras: {len(cam_intrinsics)}, points3D: {len(points3d)}")
    records, _records_by_id = _build_image_records(extrinsics, cam_intrinsics, image_dir, args.camera_params.strip())
    if args.camera_params.strip() and len(cam_intrinsics) > 1:
        print(
            "[WARN] --camera_params overrides all camera IDs. For multi-camera scenes "
            "with different intrinsics, leave --camera_params empty to use COLMAP intrinsics."
        )

    # 2. Load DA3 model.
    from depth_anything_3.api import DepthAnything3

    local_dir = Path(args.local_model_dir) if args.local_model_dir else None
    if local_dir and local_dir.exists():
        print(f"Loading DA3 from local dir: {local_dir}")
        model = DepthAnything3.from_pretrained(str(local_dir))
    else:
        print(f"Loading DA3 from HuggingFace: {args.hf_model_id}")
        model = DepthAnything3.from_pretrained(args.hf_model_id)

    model = model.to(args.device).eval()
    print(f"DA3 model ready on {args.device}\n")

    affine_ok, affine_low, skipped, n_saved = 0, 0, 0, 0
    cp = args.camera_params.strip()
    infer_stats: Dict[str, int] = defaultdict(int)

    def _infer_depth_raw_target(record: ImageRecord) -> Tuple[Optional[np.ndarray], Optional[str]]:
        if not record.exists:
            return None, record.rel

        if (not args.window_inference) or int(args.window_size) <= 1:
            infer_stats["single"] += 1
            return _infer_depth_single(model, record, args), record.rel

        ctx_infos = _select_context_views(record, records, args)
        if not ctx_infos:
            infer_stats["single_fallback_no_context"] += 1
            return _infer_depth_single(model, record, args), record.rel

        window_records = [record] + [ctx for ctx, _ in ctx_infos]
        shared_with_target = [shared for _, shared in ctx_infos]
        use_pose_conditioning = _should_use_pose_conditioning(
            shared_with_target, len(window_records), args
        )
        try:
            depth_raw = _infer_depth_window(
                model, window_records, use_pose_conditioning, args
            )
        except Exception as e:
            print(
                f"[WARN] window inference failed for {record.rel}: {e}. "
                "Fallback to single-view inference."
            )
            infer_stats["single_fallback_window_error"] += 1
            return _infer_depth_single(model, record, args), record.rel

        infer_stats["window"] += 1
        if use_pose_conditioning:
            infer_stats["window_pose"] += 1
        else:
            infer_stats["window_no_pose"] += 1
        return depth_raw, record.rel

    if args.global_depth_smooth:
        print(
            "[GLOBAL SMOOTH] Pass 1 writes raw depths to a temp memmap directory; "
            "Pass 2 optimizes sparse observations; Apply reads images one by one.\n"
        )
        raw_tmp = source_path / args.global_raw_tmp_dir
        if raw_tmp.exists():
            shutil.rmtree(raw_tmp)
        raw_tmp.mkdir(parents=True, exist_ok=True)
        depth_paths: Dict[int, Path] = {}

        for record in tqdm(records, desc="DA3 depth to memmap"):
            depth_raw, rel = _infer_depth_raw_target(record)
            if depth_raw is None:
                print(f"  [SKIP] image not found: {image_dir / rel}")
                skipped += 1
                continue
            img_id = int(record.img_id)
            p = raw_tmp / f"{img_id}.npy"
            np.save(str(p), depth_raw)
            depth_paths[img_id] = p
            del depth_raw

        ordered_ids = sorted(depth_paths.keys())
        if not ordered_ids:
            shutil.rmtree(raw_tmp, ignore_errors=True)
            raise SystemExit("[ERROR] GLOBAL_DEPTH_SMOOTH: no valid cached depth maps.")

        n_img = len(ordered_ids)
        print(
            f"[GLOBAL SMOOTH] cached {n_img} raw npy under {raw_tmp} "
            f"(~{sum(p.stat().st_size for p in depth_paths.values()) / (1024**3):.2f} GiB on disk)"
        )

        print("[GLOBAL SMOOTH] collecting sparse observations (memmap, no full-array RAM stack) ...", flush=True)
        obs_j, obs_d, obs_z, obs_pid, obs_u, obs_v, obs_cam, _obs_frame, obs_rel = (
            _collect_sparse_observations_memmap(ordered_ids, depth_paths, extrinsics, points3d)
        )
        if obs_j.size == 0:
            shutil.rmtree(raw_tmp, ignore_errors=True)
            raise SystemExit("[ERROR] GLOBAL_DEPTH_SMOOTH: no sparse observations.")

        warm_s, warm_t, n_obs_per = _warm_start_sparse(
            obs_j, obs_d, obs_z, n_img, args.min_colmap_pts, args.ransac_threshold
        )
        ri, rj, rdi, rdj, rlog = _build_ratio_arrays(
            ordered_ids,
            obs_j,
            obs_d,
            obs_z,
            obs_pid,
            obs_rel,
            obs_cam,
            args.global_smooth_adjacency,
            args.global_smooth_temporal_window,
            args.global_smooth_max_obs_per_point,
            args.global_smooth_log_eps,
            args.global_smooth_ratio_edge_mode,
        )
        print(
            f"[GLOBAL SMOOTH] sparse rows={obs_j.size} ratio_edges={ri.size} "
            f"(adjacency={args.global_smooth_adjacency}, ratio_edges={args.global_smooth_ratio_edge_mode})",
            flush=True,
        )

        scale_shift_map, ginfo = _solve_global_affine_sparse(
            ordered_ids,
            obs_j,
            obs_d,
            obs_z,
            ri,
            rj,
            rdi,
            rdj,
            rlog,
            n_img,
            warm_s,
            warm_t,
            n_obs_per,
            lambda_ratio=args.global_smooth_lambda_ratio,
            log_eps=args.global_smooth_log_eps,
            anchor_weight=args.global_smooth_anchor_weight,
            min_colmap_pts=args.min_colmap_pts,
        )
        s_opt = np.array([scale_shift_map[i][0] for i in ordered_ids], dtype=np.float64)
        t_opt = np.array([scale_shift_map[i][1] for i in ordered_ids], dtype=np.float64)
        print(
            f"[GLOBAL SMOOTH] least_squares success={ginfo['success']} {ginfo['message']} "
            f"nfev={ginfo['nfev']} cost {ginfo['cost_initial']:.4e} -> {ginfo['cost_final']:.4e}",
            flush=True,
        )

        ivd_b, ivd_a, ivd_n = _compute_ivd(
            ordered_ids,
            extrinsics,
            cam_intrinsics,
            cp,
            obs_j,
            obs_d,
            obs_pid,
            obs_u,
            obs_v,
            warm_s,
            warm_t,
            s_opt,
            t_opt,
            sample_n=args.ivd_sample_count,
            seed=args.ivd_sample_seed,
        )
        print(
            f"[GLOBAL SMOOTH] IVD (mean RMS of per-axis std over multi-view points): "
            f"before={ivd_b:.6f} after={ivd_a:.6f} (n_points={ivd_n}, sample_cap={args.ivd_sample_count})",
            flush=True,
        )

        normal_root = source_path / args.normal_world_dir
        if args.save_world_normals:
            normal_root.mkdir(parents=True, exist_ok=True)

        id_to_j = {img_id: j for j, img_id in enumerate(ordered_ids)}
        for img_id in tqdm(ordered_ids, desc="Apply depth + optional normals"):
            image_obj = extrinsics[img_id]
            rel = _normalize_rel(image_obj.name)
            j = id_to_j[img_id]
            s, t = scale_shift_map[img_id]
            mm = np.load(str(depth_paths[img_id]), mmap_mode="r")
            depth_out = (s * np.asarray(mm, dtype=np.float32) + t).astype(np.float32)
            depth_out = np.where(depth_out > 0, depth_out, 0.0).astype(np.float32)
            del mm

            n_obs = int(n_obs_per[j])
            if n_obs >= args.min_colmap_pts:
                affine_ok += 1
            else:
                affine_low += 1

            pts, mean_rel, mean_abs = _colmap_consistency_check(
                depth_out, image_obj, points3d, args.min_colmap_pts
            )
            n_saved += 1
            out_path = out_dir / Path(rel).with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), depth_out)
            if args.save_vis:
                vp = vis_dir / Path(rel).with_suffix(".png")
                vp.parent.mkdir(parents=True, exist_ok=True)
                _save_vis(depth_out, vp)
            if args.save_world_normals:
                cam_intr = cam_intrinsics[image_obj.camera_id]
                K, _, _ = _build_K(cam_intr, cp)
                R = qvec2rotmat(image_obj.qvec)
                tw = image_obj.tvec
                nw = _depth_to_world_normals(depth_out, K, R, tw)
                np_path = normal_root / Path(rel).with_suffix(".npy")
                np_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(np_path), nw)

            warn_m = args.warn_abs_err_m
            tag = "OK  " if (np.isfinite(mean_abs) and mean_abs < warn_m) else "WARN"
            print(
                f"[{tag}] {rel:<40} inliers={n_obs:>4} scale={s:.6f} shift={t:.6f}  "
                f"chk_pts={pts:>4} abs_err={mean_abs:.4f}m rel={mean_rel:.3f}",
                flush=True,
            )

        shutil.rmtree(raw_tmp, ignore_errors=True)
        print(f"[GLOBAL SMOOTH] removed temp dir {raw_tmp}", flush=True)

    else:
        for record in tqdm(records, desc="DA3 depth"):
            depth_raw, rel = _infer_depth_raw_target(record)
            if depth_raw is None:
                print(f"  [SKIP] image not found: {image_dir / rel}")
                skipped += 1
                continue
            image_obj = extrinsics[record.img_id]

            if args.colmap_affine_align:
                depth_out, scale, shift, inliers = _colmap_affine_to_meters(
                    depth_raw,
                    image_obj,
                    points3d,
                    args.ransac_threshold,
                    args.min_colmap_pts,
                )
                if inliers >= args.min_colmap_pts:
                    affine_ok += 1
                else:
                    affine_low += 1
            else:
                depth_out = depth_raw
                scale, shift, inliers = 1.0, 0.0, -1

            pts, mean_rel, mean_abs = _colmap_consistency_check(
                depth_out, image_obj, points3d, args.min_colmap_pts
            )

            n_saved += 1
            out_path = out_dir / Path(rel).with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), depth_out)
            if args.save_vis:
                vp = vis_dir / Path(rel).with_suffix(".png")
                vp.parent.mkdir(parents=True, exist_ok=True)
                _save_vis(depth_out, vp)

            warn_m = args.warn_abs_err_m
            if args.colmap_affine_align:
                tag = "OK  " if (np.isfinite(mean_abs) and mean_abs < warn_m) else "WARN"
                print(
                    f"[{tag}] {rel:<40} inliers={inliers:>4} "
                    f"scale={scale:.6f} shift={shift:.6f}  "
                    f"chk_pts={pts:>4} abs_err={mean_abs:.4f}m rel={mean_rel:.3f}"
                )
            else:
                tag = "OK  " if (np.isfinite(mean_abs) and mean_abs < warn_m) else "WARN"
                print(
                    f"[{tag}] {rel:<40}  chk_pts={pts:>4} "
                    f"abs_err={mean_abs:.4f}m rel={mean_rel:.3f}"
                )

    msg = f"\nDone. saved={n_saved}, image_missing={skipped}, output={out_dir}"
    if args.colmap_affine_align or args.global_depth_smooth:
        msg += (
            f"\n  Affine stats: inliers>={args.min_colmap_pts} -> {affine_ok}, "
            f"below -> {affine_low}"
        )
    if args.global_depth_smooth:
        msg += "\n  GLOBAL_DEPTH_SMOOTH: joint (s,t), sparse memmap, IVD, and optional world normals."
    msg += (
        "\n  Inference stats: "
        f"window={infer_stats.get('window', 0)} "
        f"(pose={infer_stats.get('window_pose', 0)}, no_pose={infer_stats.get('window_no_pose', 0)}), "
        f"single={infer_stats.get('single', 0)}, "
        f"fallback_no_context={infer_stats.get('single_fallback_no_context', 0)}, "
        f"fallback_window_error={infer_stats.get('single_fallback_window_error', 0)}"
    )
    print(msg)


if __name__ == "__main__":
    main()
