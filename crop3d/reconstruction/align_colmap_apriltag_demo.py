#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optionally align a COLMAP sparse model to an AprilTag world frame.

This step is not required for normal 2DGS training or rendering. Run it only
when downstream measurements need metric/world coordinates.

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

The first run backs up ``sparse/0`` to ``sparse/0_raw`` and writes the aligned
``images.bin`` and ``points3D.bin`` back to ``sparse/0``. Configure ``SCENE``
and the AprilTag layout below before running.
"""

from __future__ import annotations

import itertools
import json
import math
import re
import shutil
import struct
from collections import defaultdict
import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pupil_apriltags import Detector

# =============================================================================
# User settings
# =============================================================================
# Required: scene root containing images/ and sparse/0/.
SCENE = "./examples/reconstruction_demo/scene"
IMAGES_SUBDIR = "images"

# Required: AprilTag family and physical tag size.
TAG_FAMILY = "tag36h11"
TAG_SIZE_MM = 30.0

# Required: world layout in millimeters. The world Z axis points upward.
GROUND_Z_MM = 0.0
USE_TAG_CENTER_HEIGHT = True
FIRST_TAG_HEIGHT_Z_MM = -576.0
TAG_WALL_Y_MM = 0.0

# Required: tag center X positions, left to right. Center spacings override positions.
TAG_X_POSITIONS_MM = []
TAG_CENTER_SPACINGS_MM: list[float] = [1508.0, 1497.0, 1504.0, 1503.0, 1500.0]

# Required: physical tag ID order from left to right.
TAG_IDS_IN_X_ORDER: list[int] = [12,13,14,15,16,17]

USE_CAMERAS_BIN_INTRINSICS = True

# Fallback intrinsics used only when cameras.bin cannot provide supported values.
FX, FY = 1407.0, 1408.3
CX, CY = 986.7009, 518.1291
K1, K2, P1, P2 = -0.0331, 0.0209, 0.0, 0.0

SEARCH_TAG_POSITION_PERMUTATIONS = False
TRY_REVERSE_TAG_X_ORDER = False

SPARSE_WORK_DIR_NAME = "0"
SPARSE_RAW_DIR_NAME = "0_raw"
MIN_OBS_PER_CORNER = 2
MIN_TAG_COUNT_FOR_ALIGN = 4

TRIANGULATION_MAD_K = 3.0

RANSAC_SIM3_ITERATIONS = 2000
RANSAC_SIM3_MIN_SAMPLES = 6
RANSAC_INLIER_THRESH_MM = 15.0
RANSAC_RANDOM_SEED = 0

# True writes aligned sparse coordinates in meters; False writes millimeters.
ALIGN_OUTPUT_METERS = True
MM_PER_M = 1000.0

DIAG_MAX_IMAGES = 24
DIAG_MAX_OBS_PER_IMAGE = 200
DIAG_JSON_NAME = "apriltag_alignment_diagnostics.json"


def resolve_tag_x_positions_mm() -> list[float]:
    if TAG_CENTER_SPACINGS_MM:
        for s in TAG_CENTER_SPACINGS_MM:
            if float(s) <= 0.0:
                raise ValueError("TAG_CENTER_SPACINGS_MM values must be positive millimeters")
        xs = [0.0]
        for s in TAG_CENTER_SPACINGS_MM:
            xs.append(xs[-1] + float(s))
        return xs
    if not TAG_X_POSITIONS_MM:
        raise ValueError("Set TAG_X_POSITIONS_MM or TAG_CENTER_SPACINGS_MM")
    return [float(x) for x in TAG_X_POSITIONS_MM]


def resolve_tag_center_z_mm() -> float:
    h = float(FIRST_TAG_HEIGHT_Z_MM)
    if USE_TAG_CENTER_HEIGHT:
        return float(GROUND_Z_MM) + h
    return float(GROUND_Z_MM) + h + 0.5 * float(TAG_SIZE_MM)


@dataclass
class Point3DRecord:
    point3d_id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2d_idxs: np.ndarray


CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image",
    ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"],
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = {m.model_id: m for m in CAMERA_MODELS}


def fallback_K_D() -> tuple[np.ndarray, np.ndarray]:
    K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([K1, K2, P1, P2], dtype=np.float64)
    return K, D


def colmap_camera_to_K_D(cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(cam.params, dtype=np.float64).reshape(-1)
    name = cam.model
    if name == "PINHOLE" and p.size >= 4:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return K, np.zeros(4, dtype=np.float64)
    if name == "OPENCV" and p.size >= 8:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        k1, k2, p1, p2 = p[4], p[5], p[6], p[7]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return K, np.array([k1, k2, p1, p2], dtype=np.float64)
    if name == "SIMPLE_PINHOLE" and p.size >= 3:
        f, cx, cy = p[0], p[1], p[2]
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return K, np.zeros(4, dtype=np.float64)
    if name == "SIMPLE_RADIAL" and p.size >= 4:
        f, cx, cy, k = p[0], p[1], p[2], p[3]
        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        return K, np.array([k, 0.0, 0.0, 0.0], dtype=np.float64)
    raise ValueError(f"Unsupported or unexpected camera model: {name} (nparams={p.size})")


def build_camera_kd_map(
    cameras: dict[int, Camera],
    fallback_K: np.ndarray,
    fallback_D: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for cid, cam in cameras.items():
        try:
            out[cid] = colmap_camera_to_K_D(cam)
        except (ValueError, IndexError):
            out[cid] = (fallback_K.copy(), fallback_D.copy())
    return out


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array(
        [
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
        ]
    ) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def read_extrinsics_binary(path_to_model_file: str) -> dict[int, Image]:
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            data = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = data[0]
            qvec = np.array(data[1:5], dtype=np.float64)
            tvec = np.array(data[5:8], dtype=np.float64)
            camera_id = data[8]
            name = ""
            ch = read_next_bytes(fid, 1, "c")[0]
            while ch != b"\x00":
                name += ch.decode("utf-8")
                ch = read_next_bytes(fid, 1, "c")[0]
            n2d = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id = read_next_bytes(fid, num_bytes=24 * n2d, format_char_sequence="ddq" * n2d)
            xys = np.column_stack(
                [
                    tuple(map(float, x_y_id[0::3])),
                    tuple(map(float, x_y_id[1::3])),
                ]
            )
            point3d_ids = np.array(tuple(map(int, x_y_id[2::3])))
            images[image_id] = Image(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=name,
                xys=xys,
                point3D_ids=point3d_ids,
            )
    return images


def read_intrinsics_binary(path_to_model_file: str) -> dict[int, Camera]:
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            props = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = props[0]
            model_id = props[1]
            model_name = CAMERA_MODEL_IDS[model_id].model_name
            width = props[2]
            height = props[3]
            n_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8 * n_params, format_char_sequence="d" * n_params)
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
    return cameras


def read_points3d_binary_full(path_to_model_file: Path) -> dict[int, Point3DRecord]:
    points = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            data = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3d_id = data[0]
            xyz = np.array(data[1:4], dtype=np.float64)
            rgb = np.array(data[4:7], dtype=np.uint8)
            error = float(data[7])
            track_len = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track = read_next_bytes(fid, num_bytes=8 * track_len, format_char_sequence="ii" * track_len)
            image_ids = np.array(track[0::2], dtype=np.int32)
            point2d_idxs = np.array(track[1::2], dtype=np.int32)
            points[point3d_id] = Point3DRecord(
                point3d_id=point3d_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2d_idxs=point2d_idxs,
            )
    return points


def write_points3d_binary_full(points: dict[int, Point3DRecord], output_path: Path) -> None:
    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for pid in sorted(points.keys()):
            p = points[pid]
            rgb = [int(v) for v in p.rgb.tolist()]
            f.write(
                struct.pack(
                    "<QdddBBBd",
                    int(p.point3d_id),
                    float(p.xyz[0]),
                    float(p.xyz[1]),
                    float(p.xyz[2]),
                    rgb[0],
                    rgb[1],
                    rgb[2],
                    float(p.error),
                )
            )
            track_len = int(len(p.image_ids))
            f.write(struct.pack("<Q", track_len))
            for iid, p2d in zip(p.image_ids.tolist(), p.point2d_idxs.tolist()):
                f.write(struct.pack("<ii", int(iid), int(p2d)))


def write_images_binary(images: dict, output_path: Path) -> None:
    with open(output_path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for image_id in sorted(images.keys()):
            im = images[image_id]
            q = np.asarray(im.qvec, dtype=np.float64).reshape(4)
            t = np.asarray(im.tvec, dtype=np.float64).reshape(3)
            f.write(
                struct.pack(
                    "<idddddddi",
                    int(im.id),
                    float(q[0]),
                    float(q[1]),
                    float(q[2]),
                    float(q[3]),
                    float(t[0]),
                    float(t[1]),
                    float(t[2]),
                    int(im.camera_id),
                )
            )
            f.write(im.name.encode("utf-8") + b"\x00")
            n = int(im.xys.shape[0])
            f.write(struct.pack("<Q", n))
            for i in range(n):
                x, y = im.xys[i]
                pid = int(im.point3D_ids[i])
                f.write(struct.pack("<ddq", float(x), float(y), pid))


def make_target_corner(center_x_mm: float, corner_idx: int, half_size_mm: float) -> np.ndarray:
    """Return the target corner in a Z-up world frame."""
    z0 = resolve_tag_center_z_mm()
    yw = float(TAG_WALL_Y_MM)
    h = float(half_size_mm)
    if corner_idx == 0:  # TL
        return np.array([center_x_mm - h, yw, z0 + h], dtype=np.float64)
    if corner_idx == 1:  # TR
        return np.array([center_x_mm + h, yw, z0 + h], dtype=np.float64)
    if corner_idx == 2:  # BR
        return np.array([center_x_mm + h, yw, z0 - h], dtype=np.float64)
    if corner_idx == 3:  # BL
        return np.array([center_x_mm - h, yw, z0 - h], dtype=np.float64)
    raise ValueError(f"Invalid corner index: {corner_idx}")


def square_corner_permutations() -> list[tuple[int, int, int, int]]:
    # Keep only rotational solutions to avoid mirrored square mappings.
    return [
        (0, 1, 2, 3),
        (1, 2, 3, 0),
        (2, 3, 0, 1),
        (3, 0, 1, 2),
    ]


def triangulate_dlt_pair(R1, t1, x1, y1, R2, t2, x2, y2) -> np.ndarray:
    P1 = np.hstack([R1, t1.reshape(3, 1)])
    P2 = np.hstack([R2, t2.reshape(3, 1)])
    A = np.zeros((4, 4), dtype=np.float64)
    A[0] = x1 * P1[2] - P1[0]
    A[1] = y1 * P1[2] - P1[1]
    A[2] = x2 * P2[2] - P2[0]
    A[3] = y2 * P2[2] - P2[1]
    _, _, vh = np.linalg.svd(A)
    X = vh[-1]
    return X[:3] / X[3]


def robust_triangulate(obs: list[tuple[np.ndarray, np.ndarray, float, float]]) -> np.ndarray:
    # obs: [(R_wc, t_wc, x_norm, y_norm), ...]
    if len(obs) < MIN_OBS_PER_CORNER:
        raise ValueError("Not enough observations for triangulation")
    if len(obs) == 2:
        return triangulate_dlt_pair(obs[0][0], obs[0][1], obs[0][2], obs[0][3], obs[1][0], obs[1][1], obs[1][2], obs[1][3])

    points: list[np.ndarray] = []
    for a, b in itertools.combinations(range(len(obs)), 2):
        pa = triangulate_dlt_pair(obs[a][0], obs[a][1], obs[a][2], obs[a][3], obs[b][0], obs[b][1], obs[b][2], obs[b][3])
        if np.all(np.isfinite(pa)):
            points.append(pa)
    if not points:
        raise ValueError("All triangulations are invalid")
    pts = np.stack(points, axis=0)
    if TRIANGULATION_MAD_K > 0.0 and pts.shape[0] >= 4:
        med = np.median(pts, axis=0)
        dists = np.linalg.norm(pts - med.reshape(1, 3), axis=1)
        med_d = float(np.median(dists))
        mad_d = float(np.median(np.abs(dists - med_d)))
        sigma = 1.4826 * max(mad_d, 1e-12)
        thr = med_d + float(TRIANGULATION_MAD_K) * sigma
        keep = dists <= thr
        if int(np.count_nonzero(keep)) >= max(2, pts.shape[0] // 4):
            pts = pts[keep]
    return np.median(pts, axis=0)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    # src, dst: Nx3
    if src.shape != dst.shape or src.shape[1] != 3:
        raise ValueError("src/dst must be Nx3 and same shape")
    n = src.shape[0]
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_src = np.mean(np.sum(src_c * src_c, axis=1))
    scale = np.trace(np.diag(D) @ S) / max(var_src, 1e-12)
    t = dst_mean - scale * (R @ src_mean)
    return float(scale), R, t


def similarity_align_points(
    src: np.ndarray,
    dst: np.ndarray,
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    return (scale * (R @ src.T)).T + t.reshape(1, 3)


def ransac_umeyama_sim3(
    src: np.ndarray,
    dst: np.ndarray,
    min_samples: int,
    inlier_thresh_mm: float,
    max_iterations: int,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Run RANSAC + Umeyama and return the final inlier fit."""
    n = src.shape[0]
    if n < min_samples:
        scale, R, t = umeyama(src, dst)
        aligned = similarity_align_points(src, dst, scale, R, t)
        err = np.linalg.norm(aligned - dst, axis=1)
        rmse = float(np.sqrt(np.mean(err**2)))
        return scale, R, t, {
            "inlier_count": n,
            "inlier_ratio": 1.0,
            "rmse_all_mm": rmse,
            "rmse_inlier_mm": rmse,
        }

    min_s = int(min(min_samples, n))
    best_inlier_count = -1
    best_rmse_inliers = math.inf
    best_mask: np.ndarray | None = None

    for _ in range(max_iterations):
        idx = rng.choice(n, size=min_s, replace=False)
        try:
            s_h, r_h, t_h = umeyama(src[idx], dst[idx])
        except (ValueError, np.linalg.LinAlgError):
            continue
        aligned = similarity_align_points(src, dst, s_h, r_h, t_h)
        err = np.linalg.norm(aligned - dst, axis=1)
        mask = err < inlier_thresh_mm
        cnt = int(np.count_nonzero(mask))
        rmse_in = float(np.sqrt(np.mean(err[mask] ** 2))) if cnt > 0 else math.inf
        if cnt > best_inlier_count or (cnt == best_inlier_count and rmse_in < best_rmse_inliers):
            best_inlier_count = cnt
            best_rmse_inliers = rmse_in
            best_mask = mask

    if best_mask is None or best_inlier_count < min_s:
        scale, R, t = umeyama(src, dst)
        aligned = similarity_align_points(src, dst, scale, R, t)
        err = np.linalg.norm(aligned - dst, axis=1)
        rmse = float(np.sqrt(np.mean(err**2)))
        return scale, R, t, {
            "inlier_count": n,
            "inlier_ratio": 1.0,
            "rmse_all_mm": rmse,
            "rmse_inlier_mm": rmse,
            "ransac_fallback_all_points": True,
        }

    assert best_mask is not None
    scale, R, t = umeyama(src[best_mask], dst[best_mask])
    aligned = similarity_align_points(src, dst, scale, R, t)
    err_all = np.linalg.norm(aligned - dst, axis=1)
    inlier_final = err_all < inlier_thresh_mm
    if int(np.count_nonzero(inlier_final)) >= min_s:
        scale, R, t = umeyama(src[inlier_final], dst[inlier_final])
        aligned = similarity_align_points(src, dst, scale, R, t)
        err_all = np.linalg.norm(aligned - dst, axis=1)
        inlier_final = err_all < inlier_thresh_mm

    rmse_all = float(np.sqrt(np.mean(err_all**2)))
    inl = int(np.count_nonzero(inlier_final))
    rmse_in = float(np.sqrt(np.mean(err_all[inlier_final] ** 2))) if inl > 0 else math.nan
    return scale, R, t, {
        "inlier_count": inl,
        "inlier_ratio": float(inl) / float(n),
        "rmse_all_mm": rmse_all,
        "rmse_inlier_mm": rmse_in,
    }


def project_with_distortion(R_wc: np.ndarray, t_wc: np.ndarray, xyz_w: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R_wc)
    proj, _ = cv2.projectPoints(xyz_w.reshape(1, 3), rvec, t_wc.reshape(3, 1), K, D)
    return proj.reshape(2)


def _natural_sort_key(s: str):
    parts = re.split(r"(\d+)", s)
    out = []
    for p in parts:
        if p.isdigit():
            out.append((1, int(p)))
        else:
            out.append((0, p.lower()))
    return tuple(out)


def _camera_center_from_image(im) -> np.ndarray:
    R_wc = qvec2rotmat(im.qvec)
    t_wc = np.asarray(im.tvec, dtype=np.float64).reshape(3)
    return -R_wc.T @ t_wc


def _trajectory_diagnostics(images_dict: dict[int, Image]) -> dict[str, dict[str, float]]:
    groups: dict[str, list] = defaultdict(list)
    for _iid, im in images_dict.items():
        parent = Path(im.name).parent.as_posix() or "."
        groups[parent].append(im)

    out: dict[str, dict[str, float]] = {}
    for group, ims in groups.items():
        ims = sorted(ims, key=lambda x: _natural_sort_key(x.name))
        centers = np.stack([_camera_center_from_image(im) for im in ims], axis=0)
        if centers.shape[0] < 2:
            out[group] = {
                "num_images": float(centers.shape[0]),
                "path_length": 0.0,
                "straightness": 1.0,
                "mean_step": 0.0,
                "median_step": 0.0,
                "backtrack_ratio": 0.0,
            }
            continue
        diffs = np.diff(centers, axis=0)
        steps = np.linalg.norm(diffs, axis=1)
        path_len = float(np.sum(steps))
        chord = float(np.linalg.norm(centers[-1] - centers[0]))
        straightness = chord / max(path_len, 1e-12)
        mean_step = float(np.mean(steps))
        median_step = float(np.median(steps))
        backtracks = 0
        pairs = 0
        if diffs.shape[0] >= 2:
            dirs = diffs / np.maximum(steps.reshape(-1, 1), 1e-12)
            for i in range(dirs.shape[0] - 1):
                pairs += 1
                if float(np.dot(dirs[i], dirs[i + 1])) < 0.0:
                    backtracks += 1
        out[group] = {
            "num_images": float(centers.shape[0]),
            "path_length": path_len,
            "straightness": float(straightness),
            "mean_step": mean_step,
            "median_step": median_step,
            "backtrack_ratio": float(backtracks) / float(max(pairs, 1)),
        }
    return out


def _sampled_reprojection_diagnostics(
    images_dict: dict[int, Image],
    points_dict: dict[int, Point3DRecord],
    kd_map: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    max_images: int,
    max_obs_per_image: int,
) -> dict[str, float]:
    image_ids = sorted(images_dict.keys())
    if not image_ids:
        return {"count": 0.0, "mean": math.nan, "median": math.nan, "p95": math.nan}
    if len(image_ids) > max_images:
        sel_idx = np.linspace(0, len(image_ids) - 1, max_images, dtype=int)
        image_ids = [image_ids[int(i)] for i in sel_idx.tolist()]

    errs: list[float] = []
    for iid in image_ids:
        im = images_dict[iid]
        K_i, D_i = kd_map.get(im.camera_id, fallback_K_D())
        R_wc = qvec2rotmat(im.qvec)
        t_wc = np.asarray(im.tvec, dtype=np.float64).reshape(3)
        valid_idx = [i for i, pid in enumerate(im.point3D_ids) if int(pid) >= 0 and int(pid) in points_dict]
        if not valid_idx:
            continue
        if len(valid_idx) > max_obs_per_image:
            step = max(1, len(valid_idx) // max_obs_per_image)
            valid_idx = valid_idx[::step][:max_obs_per_image]
        for i in valid_idx:
            pid = int(im.point3D_ids[i])
            xyz = np.asarray(points_dict[pid].xyz, dtype=np.float64).reshape(3)
            uv = np.asarray(im.xys[i], dtype=np.float64).reshape(2)
            uv_hat = project_with_distortion(R_wc, t_wc, xyz, K_i, D_i)
            errs.append(float(np.linalg.norm(uv_hat - uv)))
    if not errs:
        return {"count": 0.0, "mean": math.nan, "median": math.nan, "p95": math.nan}
    arr = np.asarray(errs, dtype=np.float64)
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def _paired_projection_drift_diagnostics(
    images_before: dict[int, Image],
    points_before: dict[int, Point3DRecord],
    images_after: dict[int, Image],
    points_after: dict[int, Point3DRecord],
    kd_map: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    max_images: int,
    max_obs_per_image: int,
) -> dict[str, float]:
    image_ids = sorted(set(images_before.keys()) & set(images_after.keys()))
    if not image_ids:
        return {"count": 0.0, "mean": math.nan, "median": math.nan, "p95": math.nan}
    if len(image_ids) > max_images:
        sel_idx = np.linspace(0, len(image_ids) - 1, max_images, dtype=int)
        image_ids = [image_ids[int(i)] for i in sel_idx.tolist()]

    drifts: list[float] = []
    for iid in image_ids:
        im_b = images_before[iid]
        im_a = images_after[iid]
        K_i, D_i = kd_map.get(im_b.camera_id, fallback_K_D())
        Rb = qvec2rotmat(im_b.qvec)
        tb = np.asarray(im_b.tvec, dtype=np.float64).reshape(3)
        Ra = qvec2rotmat(im_a.qvec)
        ta = np.asarray(im_a.tvec, dtype=np.float64).reshape(3)
        valid_idx = [
            i for i, pid in enumerate(im_b.point3D_ids)
            if int(pid) >= 0 and int(pid) in points_before and int(pid) in points_after
        ]
        if not valid_idx:
            continue
        if len(valid_idx) > max_obs_per_image:
            step = max(1, len(valid_idx) // max_obs_per_image)
            valid_idx = valid_idx[::step][:max_obs_per_image]
        for i in valid_idx:
            pid = int(im_b.point3D_ids[i])
            xyz_b = np.asarray(points_before[pid].xyz, dtype=np.float64).reshape(3)
            xyz_a = np.asarray(points_after[pid].xyz, dtype=np.float64).reshape(3)
            uv_b = project_with_distortion(Rb, tb, xyz_b, K_i, D_i)
            uv_a = project_with_distortion(Ra, ta, xyz_a, K_i, D_i)
            drifts.append(float(np.linalg.norm(uv_a - uv_b)))
    if not drifts:
        return {"count": 0.0, "mean": math.nan, "median": math.nan, "p95": math.nan}
    arr = np.asarray(drifts, dtype=np.float64)
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
    }


def _print_trajectory_diagnostics(title: str, stats: dict[str, dict[str, float]]) -> None:
    print(title)
    for group in sorted(stats.keys(), key=_natural_sort_key):
        s = stats[group]
        print(
            "  {:<20} n={} path_len={:.6f} straightness={:.6f} mean_step={:.6f} median_step={:.6f} backtrack={:.3f}".format(
                group,
                int(s["num_images"]),
                s["path_length"],
                s["straightness"],
                s["mean_step"],
                s["median_step"],
                s["backtrack_ratio"],
            )
        )


def main() -> int:
    scene = Path(SCENE).resolve()
    sparse_parent = scene / "sparse"
    sparse_work = sparse_parent / SPARSE_WORK_DIR_NAME
    sparse_raw = sparse_parent / SPARSE_RAW_DIR_NAME
    sparse_in = sparse_raw if sparse_raw.exists() else sparse_work
    sparse_out = sparse_work
    images_dir = scene / IMAGES_SUBDIR
    images_bin_in = sparse_in / "images.bin"
    cameras_bin_in = sparse_in / "cameras.bin"
    points_bin_in = sparse_in / "points3D.bin"

    if not images_bin_in.exists() or not cameras_bin_in.exists() or not points_bin_in.exists():
        raise FileNotFoundError(
            f"Missing COLMAP sparse files under {sparse_in}. "
            f"Please place the COLMAP model in sparse/{SPARSE_WORK_DIR_NAME}, "
            f"or keep a raw backup in sparse/{SPARSE_RAW_DIR_NAME}."
        )
    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    tag_x_mm = resolve_tag_x_positions_mm()
    if len(tag_x_mm) < MIN_TAG_COUNT_FOR_ALIGN:
        raise ValueError(f"Need at least {MIN_TAG_COUNT_FOR_ALIGN} tag X positions, got {len(tag_x_mm)}")
    tag_center_z_mm = resolve_tag_center_z_mm()
    if float(tag_center_z_mm) < float(GROUND_Z_MM):
        print(
            "[WARN] The configured tag center is below the ground plane. "
            "Please verify the Z-up world layout."
        )

    fb_K, fb_D = fallback_K_D()
    half = TAG_SIZE_MM * 0.5

    print(f"[1/7] Loading COLMAP sparse model from: {sparse_in}")
    if TAG_CENTER_SPACINGS_MM:
        print(f"  Tag layout: cumulative X from {len(TAG_CENTER_SPACINGS_MM)} measured center spacings (mm)")
        print(f"    centers -> {tag_x_mm}")
    else:
        print(f"  Tag layout: TAG_X_POSITIONS_MM ({len(tag_x_mm)} centers)")
    print(
        f"  World: ground Z={GROUND_Z_MM:.3f} mm, Z positive upward, wall Y={TAG_WALL_Y_MM:.3f} mm; "
        f"first tag {'center' if USE_TAG_CENTER_HEIGHT else 'bottom'} height along +Z={FIRST_TAG_HEIGHT_Z_MM:.3f} mm, "
        f"resolved tag center Z={tag_center_z_mm:.3f} mm"
    )
    ext = read_extrinsics_binary(str(images_bin_in))
    cameras = read_intrinsics_binary(str(cameras_bin_in))
    points = read_points3d_binary_full(points_bin_in)
    if USE_CAMERAS_BIN_INTRINSICS:
        kd_map = build_camera_kd_map(cameras, fb_K, fb_D)
        print(f"  Intrinsics: from cameras.bin (fallback on unsupported models: FX={FX:.1f})")
    else:
        kd_map = {cid: (fb_K.copy(), fb_D.copy()) for cid in cameras.keys()}
        print("  Intrinsics: CONFIG fallback only (USE_CAMERAS_BIN_INTRINSICS=False)")
    print(f"  Registered images: {len(ext)}, points3D: {len(points)}")
    traj_before = _trajectory_diagnostics(ext)
    reproj_before = _sampled_reprojection_diagnostics(
        ext,
        points,
        kd_map,
        max_images=DIAG_MAX_IMAGES,
        max_obs_per_image=DIAG_MAX_OBS_PER_IMAGE,
    )
    _print_trajectory_diagnostics("[DIAG] Trajectory stats BEFORE alignment", traj_before)
    print(
        "[DIAG] Reprojection BEFORE alignment: count={} mean={:.4f}px median={:.4f}px p95={:.4f}px".format(
            int(reproj_before["count"]),
            reproj_before["mean"],
            reproj_before["median"],
            reproj_before["p95"],
        )
    )

    print(f"[2/7] Detecting AprilTag ({TAG_FAMILY})")
    detector = Detector(
        families=TAG_FAMILY,
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    # tag_id -> corner_idx -> list[(image_id, uv)]
    observations: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
    tag_center_x_pixels: dict[int, list[float]] = defaultdict(list)
    valid_image_count = 0
    total_det = 0

    for image_id in sorted(ext.keys()):
        im = ext[image_id]
        img_path = images_dir / im.name
        if not img_path.is_file():
            continue
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        dets = detector.detect(gray, estimate_tag_pose=False)
        if not dets:
            continue
        valid_image_count += 1
        for det in dets:
            tid = int(det.tag_id)
            corners = np.asarray(det.corners, dtype=np.float64).reshape(4, 2)
            center_x = float(np.mean(corners[:, 0]))
            tag_center_x_pixels[tid].append(center_x)
            total_det += 1
            for c in range(4):
                observations[(tid, c)].append((image_id, corners[c]))

    if not tag_center_x_pixels:
        raise RuntimeError("No AprilTag detected. Please check tag family / image quality.")
    print(f"  Images with detections: {valid_image_count}, total detections: {total_det}, unique tag IDs: {len(tag_center_x_pixels)}")

    print("[3/7] Selecting tag IDs used for alignment")
    desired_num_tags = len(tag_x_mm)
    freq = {tid: len(xs) for tid, xs in tag_center_x_pixels.items()}
    selected_ids = sorted(freq.keys(), key=lambda k: freq[k], reverse=True)[:desired_num_tags]
    if len(selected_ids) < MIN_TAG_COUNT_FOR_ALIGN:
        raise RuntimeError(f"Only {len(selected_ids)} tags selected, need >= {MIN_TAG_COUNT_FOR_ALIGN}.")

    if len(selected_ids) < desired_num_tags:
        print(f"  Warning: selected tag count {len(selected_ids)} < configured {desired_num_tags}. Using first {len(selected_ids)} positions.")
    print("  Selected IDs (by detection frequency):")
    for tid in selected_ids:
        print(f"    {tid} (count={freq[tid]}, mean_u={np.mean(tag_center_x_pixels[tid]):.1f})")

    print("[4/7] Triangulating tag corners")
    triangulated_src = []
    triangulated_dst = []
    triangulated_meta = []  # (tid, corner_idx, xyz_src)
    reproj_errors = []

    for tid in selected_ids:
        for cidx in range(4):
            obs_list = observations.get((tid, cidx), [])
            # Convert image points to normalized camera coordinates.
            tri_obs = []
            for image_id, uv in obs_list:
                if image_id not in ext:
                    continue
                im = ext[image_id]
                K_i, D_i = kd_map.get(im.camera_id, (fb_K, fb_D))
                R_wc = qvec2rotmat(im.qvec)
                t_wc = np.asarray(im.tvec, dtype=np.float64).reshape(3)
                und = cv2.undistortPoints(
                    np.array([[[uv[0], uv[1]]]], dtype=np.float64),
                    K_i,
                    D_i,
                    P=None,
                )
                x_n, y_n = und.reshape(2)
                tri_obs.append((R_wc, t_wc, float(x_n), float(y_n), uv, K_i, D_i))

            if len(tri_obs) < MIN_OBS_PER_CORNER:
                continue
            xyz = robust_triangulate([(o[0], o[1], o[2], o[3]) for o in tri_obs])
            triangulated_src.append(xyz)
            triangulated_meta.append((tid, cidx, xyz))

            # Triangulation reprojection error in pixels.
            for R_wc, t_wc, _, _, uv, K_i, D_i in tri_obs:
                uv_hat = project_with_distortion(R_wc, t_wc, xyz, K_i, D_i)
                reproj_errors.append(float(np.linalg.norm(uv_hat - uv)))

    if len(triangulated_src) < MIN_TAG_COUNT_FOR_ALIGN * 4:
        raise RuntimeError(
            f"Triangulated corners = {len(triangulated_src)}, insufficient for stable alignment. "
            f"Need >= {MIN_TAG_COUNT_FOR_ALIGN*4}."
        )
    src_arr = np.asarray(triangulated_src, dtype=np.float64)
    print(f"  Triangulated corners: {len(src_arr)}")

    print("[5/7] Solving similarity transform (RANSAC + Umeyama)")
    x_positions = tag_x_mm[: len(selected_ids)]
    src_meta = [(tid, cidx) for tid, cidx, _ in triangulated_meta]
    corner_maps = square_corner_permutations()
    rng = np.random.default_rng(RANSAC_RANDOM_SEED)

    if SEARCH_TAG_POSITION_PERMUTATIONS:
        tag_to_x_candidates = [
            {tid: float(x) for tid, x in zip(selected_ids, perm)}
            for perm in itertools.permutations(x_positions, len(selected_ids))
        ]
    else:
        order_modes = [False, True] if TRY_REVERSE_TAG_X_ORDER else [False]
        tag_to_x_candidates = []
        if TAG_IDS_IN_X_ORDER:
            ids_order = [int(x) for x in TAG_IDS_IN_X_ORDER]
            if len(ids_order) != len(x_positions):
                raise ValueError(
                    f"TAG_IDS_IN_X_ORDER length {len(ids_order)} does not match "
                    f"the configured tag count {len(x_positions)}"
                )
            if set(ids_order) != set(selected_ids):
                raise ValueError(
                    f"TAG_IDS_IN_X_ORDER={ids_order} does not match selected_ids={sorted(selected_ids)}"
                )
            for rev in order_modes:
                xs = list(reversed(x_positions)) if rev else list(x_positions)
                tag_to_x_candidates.append({tid: float(x) for tid, x in zip(ids_order, xs)})
        else:
            ids_asc = sorted(selected_ids)
            for rev in order_modes:
                xs = list(reversed(x_positions)) if rev else list(x_positions)
                tag_to_x_candidates.append({tid: float(x) for tid, x in zip(ids_asc, xs)})

    best = None
    min_samples = min(RANSAC_SIM3_MIN_SAMPLES, len(src_arr))

    for tag_to_x in tag_to_x_candidates:
        if any(tid not in tag_to_x for tid, _ in src_meta):
            continue
        for cmap in corner_maps:
            dst_arr = np.asarray(
                [make_target_corner(tag_to_x[tid], cmap[cidx], half) for tid, cidx in src_meta],
                dtype=np.float64,
            )
            s_try, r_try, t_try, rstats = ransac_umeyama_sim3(
                src_arr,
                dst_arr,
                min_samples=min_samples,
                inlier_thresh_mm=RANSAC_INLIER_THRESH_MM,
                max_iterations=RANSAC_SIM3_ITERATIONS,
                rng=rng,
            )
            aligned_try = similarity_align_points(src_arr, dst_arr, s_try, r_try, t_try)
            rmse_try = float(np.sqrt(np.mean(np.sum((aligned_try - dst_arr) ** 2, axis=1))))
            score = (int(rstats["inlier_count"]), -rmse_try)
            best_score = None if best is None else (int(best["inlier_count"]), -best["rmse"])
            if best is None or score > best_score:
                best = {
                    "rmse": rmse_try,
                    "scale": s_try,
                    "R": r_try,
                    "t": t_try,
                    "tag_to_x": tag_to_x,
                    "corner_map": cmap,
                    "ransac": rstats,
                    "inlier_count": rstats["inlier_count"],
                }

    if best is None:
        raise RuntimeError(
            "No valid tag-to-X mapping produced a similarity fit. "
            "Check TAG_X_POSITIONS_MM / TAG_CENTER_SPACINGS_MM, detections, or enable SEARCH_TAG_POSITION_PERMUTATIONS."
        )
    scale_mm = float(best["scale"])
    R_align = best["R"]
    t_align = np.asarray(best["t"], dtype=np.float64).reshape(3)
    align_rmse = best["rmse"]
    if ALIGN_OUTPUT_METERS:
        scale_apply = scale_mm / MM_PER_M
        t_apply = t_align / MM_PER_M
    else:
        scale_apply = scale_mm
        t_apply = t_align

    R_post = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
    R_align = R_post @ R_align
    t_apply = R_post @ t_apply

    dst_arr = np.asarray(
        [make_target_corner(best["tag_to_x"][tid], best["corner_map"][cidx], half) for tid, cidx in src_meta],
        dtype=np.float64,
    )
    print("  Best mapping (detected_id -> center_x_mm):")
    for tid in sorted(best["tag_to_x"].keys()):
        print(f"    {tid} -> {best['tag_to_x'][tid]:.3f}")
    print(f"  Best corner map (src_idx -> dst_idx): {best['corner_map']}")
    print(f"  Similarity scale (COLMAP unit -> mm): {scale_mm:.8f}")
    print(f"  alignment RMSE (fit target, mm) = {align_rmse:.4f}")
    if ALIGN_OUTPUT_METERS:
        print(
            f"  Output: meters - scale_apply (colmap->m) = {scale_apply:.10f}, "
            f"|t| = {float(np.linalg.norm(t_apply)):.6f} m"
        )
    print("  Applied final world-axis correction: rotate 180 deg around +X (flip Y/Z)")
    rs = best.get("ransac", {})
    if rs:
        print(
            f"  RANSAC inliers: {rs['inlier_count']}/{len(src_arr)} "
            f"({100.0 * float(rs['inlier_ratio']):.1f}%), "
            f"RMSE all/inlier (mm) = {rs['rmse_all_mm']:.4f} / {rs['rmse_inlier_mm']:.4f}"
        )

    print("[6/7] Applying transform to cameras and points3D")
    updated_images = {}
    cam_centers_z = []
    for image_id, im in ext.items():
        R_wc_old = qvec2rotmat(im.qvec)
        t_wc_old = np.asarray(im.tvec, dtype=np.float64).reshape(3)
        c_old = -R_wc_old.T @ t_wc_old

        c_new = scale_apply * (R_align @ c_old) + t_apply
        R_wc_new = R_wc_old @ R_align.T
        t_wc_new = -R_wc_new @ c_new
        cam_centers_z.append(float(c_new[2]))

        q_new = rotmat2qvec(R_wc_new)
        updated_images[image_id] = im._replace(qvec=q_new, tvec=t_wc_new)

    # Warn only; do not apply a reflection that would break camera rotations.
    neg_ratio = float(np.mean(np.array(cam_centers_z) < 0.0))
    if neg_ratio > 0.8:
        print("  Warning: most transformed camera centers have Z < 0. Please verify Z-axis direction convention.")

    updated_points = {}
    for pid, p in points.items():
        xyz_new = scale_apply * (R_align @ p.xyz) + t_apply
        updated_points[pid] = Point3DRecord(
            point3d_id=p.point3d_id,
            xyz=xyz_new,
            rgb=p.rgb,
            error=p.error,
            image_ids=p.image_ids,
            point2d_idxs=p.point2d_idxs,
        )

    traj_after_mem = _trajectory_diagnostics(updated_images)
    reproj_after_mem = _sampled_reprojection_diagnostics(
        updated_images,
        updated_points,
        kd_map,
        max_images=DIAG_MAX_IMAGES,
        max_obs_per_image=DIAG_MAX_OBS_PER_IMAGE,
    )
    drift_mem = _paired_projection_drift_diagnostics(
        ext,
        points,
        updated_images,
        updated_points,
        kd_map,
        max_images=DIAG_MAX_IMAGES,
        max_obs_per_image=DIAG_MAX_OBS_PER_IMAGE,
    )
    _print_trajectory_diagnostics("[DIAG] Trajectory stats AFTER alignment (in-memory)", traj_after_mem)
    print(
        "[DIAG] Reprojection AFTER alignment (in-memory): count={} mean={:.4f}px median={:.4f}px p95={:.4f}px".format(
            int(reproj_after_mem["count"]),
            reproj_after_mem["mean"],
            reproj_after_mem["median"],
            reproj_after_mem["p95"],
        )
    )
    print(
        "[DIAG] Projection drift BEFORE vs AFTER (in-memory): count={} mean={:.6f}px median={:.6f}px p95={:.6f}px".format(
            int(drift_mem["count"]),
            drift_mem["mean"],
            drift_mem["median"],
            drift_mem["p95"],
        )
    )

    print("[7/7] Writing updated sparse model")
    if not sparse_raw.exists():
        if not sparse_work.exists():
            raise FileNotFoundError(
                f"Working sparse dir not found: {sparse_work}. "
                "Expected run_colmap output under sparse/0."
            )
        shutil.copytree(sparse_work, sparse_raw)
        print(f"  Created raw backup: {sparse_work} -> {sparse_raw}")
        sparse_in = sparse_raw
    else:
        print(f"  Using existing raw backup: {sparse_raw}")

    if sparse_out.exists():
        shutil.rmtree(sparse_out)
    shutil.copytree(sparse_in, sparse_out)
    print(f"  Refreshed working sparse model: {sparse_in} -> {sparse_out}")

    images_bin_out = sparse_out / "images.bin"
    points_bin_out = sparse_out / "points3D.bin"
    write_images_binary(updated_images, images_bin_out)
    write_points3d_binary_full(updated_points, points_bin_out)

    # Remove stale rig files because this script updates images.bin directly.
    removed_stale: list[Path] = []
    for stale_name in ("frames.bin", "rigs.bin"):
        stale_path = sparse_out / stale_name
        if stale_path.exists():
            stale_path.unlink()
            removed_stale.append(stale_path)
    ply_path = sparse_out / "points3D.ply"
    if ply_path.exists():
        ply_path.unlink()

    ext_written = read_extrinsics_binary(str(images_bin_out))
    points_written = read_points3d_binary_full(points_bin_out)
    traj_after_disk = _trajectory_diagnostics(ext_written)
    reproj_after_disk = _sampled_reprojection_diagnostics(
        ext_written,
        points_written,
        kd_map,
        max_images=DIAG_MAX_IMAGES,
        max_obs_per_image=DIAG_MAX_OBS_PER_IMAGE,
    )
    drift_disk = _paired_projection_drift_diagnostics(
        ext,
        points,
        ext_written,
        points_written,
        kd_map,
        max_images=DIAG_MAX_IMAGES,
        max_obs_per_image=DIAG_MAX_OBS_PER_IMAGE,
    )
    _print_trajectory_diagnostics("[DIAG] Trajectory stats AFTER alignment (re-read from disk)", traj_after_disk)
    print(
        "[DIAG] Reprojection AFTER alignment (re-read): count={} mean={:.4f}px median={:.4f}px p95={:.4f}px".format(
            int(reproj_after_disk["count"]),
            reproj_after_disk["mean"],
            reproj_after_disk["median"],
            reproj_after_disk["p95"],
        )
    )
    print(
        "[DIAG] Projection drift BEFORE vs AFTER (re-read): count={} mean={:.6f}px median={:.6f}px p95={:.6f}px".format(
            int(drift_disk["count"]),
            drift_disk["mean"],
            drift_disk["median"],
            drift_disk["p95"],
        )
    )

    diag_path = sparse_out / DIAG_JSON_NAME
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "reprojection_before": reproj_before,
                "reprojection_after_in_memory": reproj_after_mem,
                "reprojection_after_re_read": reproj_after_disk,
                "projection_drift_in_memory": drift_mem,
                "projection_drift_re_read": drift_disk,
                "trajectory_before": traj_before,
                "trajectory_after_in_memory": traj_after_mem,
                "trajectory_after_re_read": traj_after_disk,
                "scale_mm": scale_mm,
                "scale_apply": scale_apply,
                "align_rmse_mm": align_rmse,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Final summary.
    reproj_mean = float(np.mean(reproj_errors)) if reproj_errors else math.nan
    reproj_median = float(np.median(reproj_errors)) if reproj_errors else math.nan
    reproj_p95 = float(np.percentile(reproj_errors, 95)) if reproj_errors else math.nan
    print("\nAlignment done.")
    print(f"- Used tags: {len(selected_ids)}")
    print(f"- Used corners: {len(src_arr)}")
    print(f"- Sparse coordinates: {'meters' if ALIGN_OUTPUT_METERS else 'millimeters'}")
    print(
        f"- Scale (colmap->{ 'm' if ALIGN_OUTPUT_METERS else 'mm'}): {scale_apply:.10f} "
        f"(raw colmap->mm: {scale_mm:.8f})"
    )
    print(f"- Triangulation reprojection error (px): mean={reproj_mean:.3f}, median={reproj_median:.3f}, p95={reproj_p95:.3f}")
    print(f"- Similarity fit RMSE (target mm): {align_rmse:.4f}")
    print(f"- Input sparse model: {sparse_in}")
    print(f"- Working sparse model: {sparse_out}")
    print(f"- Updated files: {images_bin_out}, {points_bin_out}")
    if removed_stale:
        print(f"- Removed stale rig files: {', '.join(str(p) for p in removed_stale)}")
    print(f"- Deleted stale ply: {ply_path}")
    print(f"- Diagnostics JSON: {diag_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise
