"""Default deterministic tiling planner."""

from __future__ import annotations

from typing import Dict, Optional

from transformation_portal.spatial_ai.segmentation.tiling.config import SegmentationTilingConfig
from transformation_portal.spatial_ai.segmentation.tiling.types import BBox, GlobalSeedHints, TileManifest, TileSpec


def _tile_origins(extent: int, tile_size_px: int, stride: int) -> list[int]:
    """Start offsets along one axis, stopping at the first tile that reaches the edge.

    A further origin would start a tile wholly inside the previous one (down to a 1 px sliver), which adds no
    coverage but costs a segmentation pass and feeds duplicate instances to the merger.
    """
    origins: list[int] = []
    for origin in range(0, extent, stride):
        origins.append(origin)
        if origin + tile_size_px >= extent:
            break
    return origins


class UniformTilingPlanner:
    """Plan deterministic row-major uniform tiles."""

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
    ) -> TileManifest:
        del global_hints, prompts
        stride = max(1, config.tile_size_px - config.overlap_px)

        # Prompt tiling is intentionally not exposed by config yet. If this
        # planner is called directly for prompted modes, keep legacy full-image
        # behavior until ROI tiling has dedicated contract coverage.
        if mode in {"points", "bbox"}:
            return TileManifest(
                image_hash=image_hash,
                W=W,
                H=H,
                tile_size_px=config.tile_size_px,
                overlap_px=config.overlap_px,
                stride_px=stride,
                policy=config.policy,
                seed=config.seed,
                tiles=(
                    TileSpec(
                        tile_id="tile_0_0",
                        bbox=BBox(0, 0, W, H),
                        overlap_px=config.overlap_px,
                        pad_mode=config.pad_mode,
                    ),
                ),
            )

        tiles: list[TileSpec] = []
        tile_idx = 0
        for y0 in _tile_origins(H, config.tile_size_px, stride):
            for x0 in _tile_origins(W, config.tile_size_px, stride):
                x1 = min(x0 + config.tile_size_px, W)
                y1 = min(y0 + config.tile_size_px, H)
                tiles.append(
                    TileSpec(
                        tile_id=f"tile_{tile_idx}",
                        bbox=BBox(int(x0), int(y0), int(x1), int(y1)),
                        overlap_px=config.overlap_px,
                        pad_mode=config.pad_mode,
                    )
                )
                tile_idx += 1

        return TileManifest(
            image_hash=image_hash,
            W=W,
            H=H,
            tile_size_px=config.tile_size_px,
            overlap_px=config.overlap_px,
            stride_px=stride,
            policy=config.policy,
            seed=config.seed,
            tiles=tuple(tiles),
        )
