from __future__ import annotations

from typing import Dict, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata
from transformation_portal.spatial_ai.segmentation.tiling.config import MergeConfig, SegmentationTilingConfig
from transformation_portal.spatial_ai.segmentation.tiling.types import GlobalSeedHints, TileManifest, TileSegmentationResult


@runtime_checkable
class TilingPlanner(Protocol):
    def plan(
        self,
        *,
        image_hash: str,
        W: int,
        H: int,
        config: SegmentationTilingConfig,
        global_hints: Optional[GlobalSeedHints],
        prompts: Optional[Dict],
        mode: str,
    ) -> TileManifest: ...


@runtime_checkable
class TileMerger(Protocol):
    def merge(
        self,
        *,
        image_hash: str,
        W: int,
        H: int,
        manifest: TileManifest,
        tile_results: Sequence[TileSegmentationResult],
        global_hints: Optional[GlobalSeedHints],
        merge_config: MergeConfig,
    ) -> Tuple["np.ndarray", "np.ndarray", list[MaskMetadata], dict]: ...


@runtime_checkable
class MergeValidator(Protocol):
    def validate(
        self,
        *,
        manifest: TileManifest,
        merge_stats: dict,
        config: SegmentationTilingConfig,
    ) -> tuple[bool, dict]: ...
