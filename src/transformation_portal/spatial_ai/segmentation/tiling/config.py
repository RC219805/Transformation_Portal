from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional


@dataclass
class GlobalPassConfig:
    enabled: bool = True
    longest_side: int = 1280

    def __post_init__(self) -> None:
        if self.longest_side <= 0:
            raise ValueError(f"global_pass.longest_side must be > 0, got {self.longest_side}")


@dataclass
class InstanceMergeConfig:
    enabled: bool = True
    iou_threshold: float = 0.35
    border_only: bool = True
    embedding_cosine_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(f"instance_merge.iou_threshold must be in [0,1], got {self.iou_threshold}")
        if self.embedding_cosine_threshold is not None and not -1.0 <= self.embedding_cosine_threshold <= 1.0:
            raise ValueError(
                f"instance_merge.embedding_cosine_threshold must be in [-1,1], got {self.embedding_cosine_threshold}"
            )


@dataclass
class MergeConfig:
    mode: Literal["weighted_soft", "binary_union"] = "weighted_soft"
    window: Literal["hann", "cosine", "linear"] = "hann"
    instance_merge: InstanceMergeConfig = field(default_factory=InstanceMergeConfig)


@dataclass
class ValidationConfig:
    enabled: bool = True
    seam_discontinuity_threshold: float = 0.25
    max_retries: int = 1
    auto_increase_overlap: bool = True

    def __post_init__(self) -> None:
        if self.seam_discontinuity_threshold < 0.0:
            raise ValueError("validation.seam_discontinuity_threshold must be >= 0")
        if self.max_retries < 0:
            raise ValueError("validation.max_retries must be >= 0")


@dataclass
class SegmentationTilingConfig:
    enabled: bool = False
    policy: Literal["uniform", "content_adaptive"] = "content_adaptive"
    tile_size_px: int = 1536
    overlap_px: int = 256
    pad_mode: Literal["reflect", "edge", "constant"] = "reflect"
    seed: int = 1337
    apply_to_modes: tuple[str, ...] = ("auto", "points", "bbox")
    global_pass: GlobalPassConfig = field(default_factory=GlobalPassConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        if self.tile_size_px <= 0:
            raise ValueError(f"tile_size_px must be > 0, got {self.tile_size_px}")
        if self.overlap_px < 0:
            raise ValueError(f"overlap_px must be >= 0, got {self.overlap_px}")
        if self.overlap_px >= self.tile_size_px:
            raise ValueError(f"overlap_px ({self.overlap_px}) must be < tile_size_px ({self.tile_size_px})")
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be >= 1")

    @staticmethod
    def from_dict(data: Optional[Mapping[str, Any]]) -> "SegmentationTilingConfig":
        if not data:
            return SegmentationTilingConfig(enabled=False)

        gp = data.get("global_pass", {}) if isinstance(data.get("global_pass", {}), Mapping) else {}
        mg = data.get("merge", {}) if isinstance(data.get("merge", {}), Mapping) else {}
        im = mg.get("instance_merge", {}) if isinstance(mg.get("instance_merge", {}), Mapping) else {}
        val = data.get("validation", {}) if isinstance(data.get("validation", {}), Mapping) else {}
        apply_modes = data.get("apply_to_modes", ("auto", "points", "bbox"))
        if isinstance(apply_modes, list):
            apply_modes = tuple(apply_modes)

        return SegmentationTilingConfig(
            enabled=bool(data.get("enabled", True)),
            policy=data.get("policy", "content_adaptive"),
            tile_size_px=int(data.get("tile_size_px", 1536)),
            overlap_px=int(data.get("overlap_px", 256)),
            pad_mode=data.get("pad_mode", "reflect"),
            seed=int(data.get("seed", 1337)),
            apply_to_modes=tuple(apply_modes),
            global_pass=GlobalPassConfig(
                enabled=bool(gp.get("enabled", True)),
                longest_side=int(gp.get("longest_side", 1280)),
            ),
            merge=MergeConfig(
                mode=mg.get("mode", "weighted_soft"),
                window=mg.get("window", "hann"),
                instance_merge=InstanceMergeConfig(
                    enabled=bool(im.get("enabled", True)),
                    iou_threshold=float(im.get("iou_threshold", 0.35)),
                    border_only=bool(im.get("border_only", True)),
                    embedding_cosine_threshold=(
                        float(im["embedding_cosine_threshold"]) if "embedding_cosine_threshold" in im else None
                    ),
                ),
            ),
            validation=ValidationConfig(
                enabled=bool(val.get("enabled", True)),
                seam_discontinuity_threshold=float(val.get("seam_discontinuity_threshold", 0.25)),
                max_retries=int(val.get("max_retries", 1)),
                auto_increase_overlap=bool(val.get("auto_increase_overlap", True)),
            ),
            max_concurrency=int(data.get("max_concurrency", 1)),
        )
