"""Scene-group scaffolding for future multi-view reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class SceneGroup:
    """
    Logical grouping of images that belong to the same reconstruction scene.

    Current behavior: 1 image per group (no behavior change).
    Future: multi-view grouping + camera parameters.
    """

    scene_id: str
    images: Tuple[Path, ...]


def build_scene_groups(images: List[Path]) -> List[SceneGroup]:
    """
    Phase A scaffold: preserve existing per-image processing.

    Each image forms its own scene group.
    """
    groups = []

    for img in images:
        scene_id = img.with_suffix("").as_posix()
        groups.append(
            SceneGroup(
                scene_id=scene_id,
                images=(img,),
            )
        )

    return groups
