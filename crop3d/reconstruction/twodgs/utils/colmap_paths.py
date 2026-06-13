"""
Helpers for resolving paths from COLMAP Image.name.

Rule:
- RGB is stored under <scene>/images/<relative_name_from_colmap>
- Depth is stored under <scene>/depth_da3 or depth_da2 with the same relative path and .npy suffix
"""
from __future__ import annotations

import os
from pathlib import Path


def normalize_colmap_image_rel_path(colmap_image_name: str) -> str:
    """Normalize COLMAP image name as a POSIX-like relative path."""
    return (colmap_image_name or "").replace("\\", "/").strip().lstrip("/")


def image_path_under_images_dir(images_folder: str | os.PathLike[str], colmap_image_name: str) -> str:
    """Build absolute image path from images folder and COLMAP relative name."""
    rel = normalize_colmap_image_rel_path(colmap_image_name)
    return os.path.join(os.fspath(images_folder), rel.replace("/", os.sep))


def image_name_from_colmap_rel(colmap_image_name: str) -> str:
    """
    View identifier used by training/logging.
    Example: cam_left/frame_0001.jpg -> cam_left/frame_0001
    """
    rel = normalize_colmap_image_rel_path(colmap_image_name)
    return Path(rel).with_suffix("").as_posix()


def _depth_candidate(scene_root: Path, colmap_image_name: str, depth_subdir: str) -> Path:
    rel = normalize_colmap_image_rel_path(colmap_image_name)
    return scene_root / depth_subdir / Path(rel).with_suffix(".npy")


def resolve_depth_path(scene_root: Path | str, colmap_image_name: str) -> str:
    """Try depth_da3 first, then depth_da2. Return empty string if not found."""
    root = Path(scene_root)
    for depth_subdir in ("depth_da3", "depth_da2"):
        p = _depth_candidate(root, colmap_image_name, depth_subdir)
        if p.is_file():
            return str(p)
    return ""
