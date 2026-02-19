from __future__ import annotations

from typing import Dict, Optional, Protocol, runtime_checkable

import numpy as np

from transformation_portal.spatial_ai.segmentation.tiling.types import GlobalSeedHints, TileInstance, TileSpec


@runtime_checkable
class TiledSegmentationBackend(Protocol):
    name: str
    device: str

    def global_seed_pass(
        self,
        *,
        image_linear: np.ndarray,
        image_hash: str,
        longest_side: int,
        rng_seed: int,
    ) -> GlobalSeedHints: ...

    def segment_tile(
        self,
        *,
        tile_linear: np.ndarray,
        image_hash: str,
        tile_spec: TileSpec,
        mode: str,
        prompts: Optional[Dict],
        global_hints: Optional[GlobalSeedHints],
        rng_seed: int,
    ) -> tuple[TileInstance, ...]: ...

    def unload(self) -> None: ...
