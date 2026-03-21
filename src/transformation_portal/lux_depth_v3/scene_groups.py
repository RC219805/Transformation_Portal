"""Scene-group utilities for deterministic multi-view reconstruction grouping."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class SceneGroup:
    """
    Logical grouping of images that belong to the same reconstruction scene.

    Current behavior: deterministic scene IDs and image grouping only.
    Camera resolution is handled by SceneContext/camera loader stages.
    """

    scene_id: str
    images: Tuple[Path, ...]


def _normalize_relative_path(path: Path, dataset_root: Path) -> str:
    """Normalize path relative to dataset root with stable cross-platform formatting."""
    root_resolved = dataset_root.resolve()
    path_resolved = path.resolve()
    try:
        rel = path_resolved.relative_to(root_resolved)
    except ValueError:
        # Fall back to absolute normalized path when input is outside dataset root.
        rel = path_resolved
    return rel.as_posix().lower()


def normalize_relative_path(path: Path, dataset_root: Path) -> str:
    """Public wrapper for stable path normalization used across scene modules."""
    return _normalize_relative_path(path, dataset_root)


def _compute_scene_id(images: Tuple[Path, ...], dataset_root: Path) -> str:
    """Compute deterministic scene id from normalized relative image paths."""
    normalized = [normalize_relative_path(p, dataset_root) for p in images]
    payload = "\n".join(sorted(normalized)).encode("utf-8")
    return hashlib.sha1(payload, usedforsecurity=False).hexdigest()[:12]


def compute_scene_id(images: Tuple[Path, ...], dataset_root: Path) -> str:
    """Public wrapper for deterministic scene identifier derivation."""
    return _compute_scene_id(images, dataset_root)


def _group_key(path: Path, dataset_root: Path) -> str:
    """Group key for parent-directory grouping mode."""
    normalized = normalize_relative_path(path, dataset_root)
    if "/" not in normalized:
        return ""
    return normalized.rsplit("/", 1)[0]


def build_scene_groups(
    images: Sequence[Path],
    dataset_root: Path,
    grouping_mode: str = "single",
) -> List[SceneGroup]:
    """
    Build deterministic scene groups from image paths.

    Modes:
    - single: each image forms its own scene (default, inert behavior)
    - parent_dir: group images by normalized relative parent directory
    """
    mode = grouping_mode.strip().lower()
    image_list = [Path(img) for img in images]

    if mode == "single":
        groups: List[SceneGroup] = []
        for img in image_list:
            group_images = (img,)
            groups.append(
                SceneGroup(
                    scene_id=compute_scene_id(group_images, dataset_root),
                    images=group_images,
                )
            )
        return groups

    if mode == "parent_dir":
        # Stable ordering regardless of caller order.
        sorted_images = sorted(image_list, key=lambda p: normalize_relative_path(p, dataset_root))
        parent_groups: List[SceneGroup] = []
        for _, grouped_iter in groupby(sorted_images, key=lambda p: _group_key(p, dataset_root)):
            grouped_images = tuple(grouped_iter)
            parent_groups.append(
                SceneGroup(
                    scene_id=compute_scene_id(grouped_images, dataset_root),
                    images=grouped_images,
                )
            )
        return parent_groups

    raise ValueError(f"Unknown grouping_mode '{grouping_mode}'. Expected one of: single, parent_dir")
