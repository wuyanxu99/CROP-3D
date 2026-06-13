#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert COLMAP cameras.bin to PINHOLE for the 2DGS loader.

The training reader only accepts PINHOLE / SIMPLE_PINHOLE cameras. COLMAP
reconstruction often keeps OPENCV intrinsics, so this script rewrites only
cameras.bin while leaving images.bin and points3D.bin untouched.
"""

from __future__ import annotations

import argparse
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PINHOLE_MODEL_ID = 1
PINHOLE_NUM_PARAMS = 4
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = str(REPO_ROOT / "examples" / "reconstruction_demo" / "scene")

CAMERA_MODEL_NUM_PARAMS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass
class CameraRecord:
    camera_id: int
    model_id: int
    width: int
    height: int
    params: np.ndarray


def _read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise EOFError(f"Unexpected end of file while reading {n} bytes")
    return data


def read_cameras_binary(path: Path) -> list[CameraRecord]:
    cameras: list[CameraRecord] = []
    with path.open("rb") as f:
        num_cameras = struct.unpack("<Q", _read_exact(f, 8))[0]
        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", _read_exact(f, 4))[0]
            model_id = struct.unpack("<i", _read_exact(f, 4))[0]
            width = struct.unpack("<Q", _read_exact(f, 8))[0]
            height = struct.unpack("<Q", _read_exact(f, 8))[0]
            if model_id not in CAMERA_MODEL_NUM_PARAMS:
                raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
            _, num_params = CAMERA_MODEL_NUM_PARAMS[model_id]
            params = np.fromfile(f, dtype=np.float64, count=num_params)
            if params.size != num_params:
                raise EOFError(f"Camera {camera_id} has incomplete parameter data")
            cameras.append(CameraRecord(camera_id, model_id, width, height, params))
    return cameras


def pinhole_params(camera: CameraRecord) -> np.ndarray:
    model_name, _ = CAMERA_MODEL_NUM_PARAMS[camera.model_id]
    p = camera.params
    if model_name == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        return np.array([f, f, cx, cy], dtype=np.float64)
    if p.size < PINHOLE_NUM_PARAMS:
        raise ValueError(f"Camera {camera.camera_id} model {model_name} cannot be converted to PINHOLE")
    return p[:PINHOLE_NUM_PARAMS].astype(np.float64, copy=True)


def write_pinhole_cameras_binary(path: Path, cameras: list[CameraRecord]) -> None:
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            params = pinhole_params(camera)
            f.write(struct.pack("<I", camera.camera_id))
            f.write(struct.pack("<i", PINHOLE_MODEL_ID))
            f.write(struct.pack("<Q", camera.width))
            f.write(struct.pack("<Q", camera.height))
            for value in params:
                f.write(struct.pack("<d", float(value)))


def convert_to_pinhole(scene: Path, sparse_subdir: str, backup_name: str, force: bool) -> Path:
    sparse_dir = scene / "sparse" / sparse_subdir
    cameras_path = sparse_dir / "cameras.bin"
    if not cameras_path.is_file():
        raise FileNotFoundError(f"cameras.bin not found: {cameras_path}")

    cameras = read_cameras_binary(cameras_path)
    if not cameras:
        raise ValueError(f"No cameras found in {cameras_path}")

    if all(camera.model_id == PINHOLE_MODEL_ID for camera in cameras):
        print(f"Already PINHOLE: {cameras_path}")
        return cameras_path

    backup_path = sparse_dir / backup_name
    if backup_path.exists() and not force:
        raise FileExistsError(f"Backup already exists: {backup_path}; pass --force to overwrite")
    shutil.copy2(cameras_path, backup_path)
    write_pinhole_cameras_binary(cameras_path, cameras)

    converted = read_cameras_binary(cameras_path)
    if any(camera.model_id != PINHOLE_MODEL_ID or camera.params.size != PINHOLE_NUM_PARAMS for camera in converted):
        raise RuntimeError(f"Failed to verify converted cameras: {cameras_path}")

    print(f"Converted {len(converted)} camera(s) to PINHOLE: {cameras_path}")
    print(f"Backup: {backup_path}")
    return cameras_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=DEFAULT_SCENE, help="COLMAP scene root containing sparse/0")
    parser.add_argument("--sparse-subdir", default="0", help="Sparse model subdirectory under sparse/")
    parser.add_argument("--backup-name", default="cameras_before_pinhole_conversion.bin", help="Backup filename")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing backup file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    convert_to_pinhole(
        scene=Path(args.scene).expanduser().resolve(),
        sparse_subdir=args.sparse_subdir,
        backup_name=args.backup_name,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
