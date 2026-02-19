from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return max(0, self.x1 - self.x0)

    @property
    def h(self) -> int:
        return max(0, self.y1 - self.y0)


@dataclass(frozen=True)
class TileSpec:
    tile_id: str
    bbox: BBox
    overlap_px: int
    pad_mode: str


@dataclass(frozen=True)
class TileManifest:
    image_hash: str
    W: int
    H: int
    tile_size_px: int
    overlap_px: int
    stride_px: int
    policy: str
    seed: int
    tiles: Tuple[TileSpec, ...]


@dataclass(frozen=True)
class GlobalSeedHints:
    image_hash: str
    low_res_longest_side: int
    low_res_W: int
    low_res_H: int
    scale_x: float
    scale_y: float
    saliency_map: Optional[np.ndarray] = None
    coarse_mask: Optional[np.ndarray] = None
    border_crossing: Tuple[BBox, ...] = ()
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SoftMaskPatch:
    bbox: BBox
    values: np.ndarray
    space: Literal["prob", "logits"] = "prob"


@dataclass(frozen=True)
class TileInstance:
    local_id: str
    score: float
    stability_score: float
    soft_mask: SoftMaskPatch
    material_label: Optional[str] = None
    material_confidence: Optional[float] = None
    embedding: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TileSegmentationResult:
    image_hash: str
    tile_id: str
    tile_spec: TileSpec
    instances: Tuple[TileInstance, ...]
    runtime_s: float
    backend_meta: Dict[str, Any] = field(default_factory=dict)
