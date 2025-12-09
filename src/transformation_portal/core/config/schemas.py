"""
Unified configuration schemas for all pipelines.

Uses Pydantic for validation and type safety.
Extracted common patterns from multiple pipeline configs.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional, Any, Dict
from pydantic import BaseModel, Field, field_validator


class DeviceType(str, Enum):
    """Supported compute device types."""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    COREML = "coreml"


class PrecisionType(str, Enum):
    """Computation precision types."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


class DeviceConfig(BaseModel):
    """Unified device configuration."""
    device: DeviceType = Field(
        default=DeviceType.AUTO,
        description="Compute device (auto, cpu, cuda, mps, coreml)"
    )
    precision: PrecisionType = Field(
        default=PrecisionType.FP16,
        description="Computation precision"
    )
    enable_cudnn_benchmark: bool = Field(
        default=True,
        description="Enable cuDNN benchmarking for CUDA"
    )
    memory_fraction: float = Field(
        default=0.85,
        ge=0.1,
        le=0.95,
        description="Fraction of device memory to use"
    )
    prefer_neural_engine: bool = Field(
        default=True,
        description="Prefer Apple Neural Engine when available"
    )

    @field_validator("memory_fraction")
    @classmethod
    def validate_memory_fraction(cls, v: float) -> float:
        """Ensure memory fraction is in valid range."""
        if not 0.1 <= v <= 0.95:
            raise ValueError("memory_fraction must be between 0.1 and 0.95")
        return v


class PathsConfig(BaseModel):
    """Unified paths configuration."""
    input_dir: Optional[Path] = Field(
        default=None,
        description="Input directory for batch processing"
    )
    output_dir: Optional[Path] = Field(
        default=None,
        description="Output directory for results"
    )
    cache_dir: Path = Field(
        default=Path(".cache"),
        description="Cache directory for intermediate files"
    )
    checkpoint_dir: Path = Field(
        default=Path(".checkpoints"),
        description="Checkpoint directory for error recovery"
    )
    model_weights_dir: Optional[Path] = Field(
        default=None,
        description="Directory for model weights"
    )

    @field_validator("input_dir", "output_dir", "cache_dir", "checkpoint_dir", "model_weights_dir")
    @classmethod
    def expand_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Expand user paths."""
        if v is None:
            return None
        return v.expanduser().resolve()


class PerformanceConfig(BaseModel):
    """Unified performance configuration."""
    max_workers: int = Field(
        default=1,
        ge=1,
        description="Maximum concurrent workers"
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Processing batch size"
    )
    tile_size: int = Field(
        default=512,
        ge=64,
        description="Tile size for memory-efficient processing"
    )
    tile_overlap: int = Field(
        default=64,
        ge=0,
        description="Overlap between tiles in pixels"
    )
    enable_tiling: bool = Field(
        default=True,
        description="Enable tiled processing for large images"
    )
    enable_async_io: bool = Field(
        default=True,
        description="Enable asynchronous I/O operations"
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable result caching"
    )
    memory_budget_gb: Optional[float] = Field(
        default=None,
        ge=1.0,
        description="Memory budget in GB (None = auto)"
    )

    @field_validator("tile_overlap")
    @classmethod
    def validate_tile_overlap(cls, v: int, info) -> int:
        """Ensure tile overlap is reasonable."""
        # Access tile_size from info.data if available
        tile_size = info.data.get("tile_size", 512)
        if v >= tile_size:
            raise ValueError(f"tile_overlap ({v}) must be less than tile_size ({tile_size})")
        return v


class OutputConfig(BaseModel):
    """Unified output configuration."""
    save_master: bool = Field(
        default=True,
        description="Save master output (16-bit TIFF)"
    )
    save_preview: bool = Field(
        default=True,
        description="Save preview (JPEG/PNG)"
    )
    preview_scale: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        description="Preview scale factor"
    )
    compression: Optional[str] = Field(
        default="lzw",
        description="TIFF compression (lzw, deflate, none)"
    )
    skip_existing: bool = Field(
        default=True,
        description="Skip processing if output exists"
    )
    overwrite: bool = Field(
        default=False,
        description="Overwrite existing outputs"
    )
    write_outputs: bool = Field(
        default=True,
        description="Master switch for filesystem output"
    )


class ValidationConfig(BaseModel):
    """Unified validation configuration."""
    enable_validation: bool = Field(
        default=True,
        description="Enable input validation"
    )
    validate_ai_output: bool = Field(
        default=True,
        description="Validate AI-generated outputs"
    )
    max_input_size_mb: float = Field(
        default=500.0,
        ge=1.0,
        description="Maximum input file size in MB"
    )
    allowed_extensions: tuple[str, ...] = Field(
        default=(".tif", ".tiff", ".jpg", ".jpeg", ".png"),
        description="Allowed input file extensions"
    )
    strict_mode: bool = Field(
        default=False,
        description="Strict validation mode (fail on warnings)"
    )


class ConfigSchema(BaseModel):
    """
    Unified configuration schema for all pipelines.
    
    This is the top-level configuration that combines all sub-configs.
    Individual pipelines can extend this schema with their specific needs.
    """
    device: DeviceConfig = Field(
        default_factory=DeviceConfig,
        description="Device configuration"
    )
    paths: PathsConfig = Field(
        default_factory=PathsConfig,
        description="Paths configuration"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance configuration"
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration"
    )
    validation: ValidationConfig = Field(
        default_factory=ValidationConfig,
        description="Validation configuration"
    )
    
    # Pipeline-specific extensions
    extras: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline-specific configuration extensions"
    )

    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
        validate_assignment = True
        extra = "allow"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ConfigSchema:
        """Create from dictionary."""
        return cls(**data)
