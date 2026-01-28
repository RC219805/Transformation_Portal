"""
Configuration Schemas.

Defines Pydantic models for type-safe configuration validation.
"""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal

from pydantic import BaseModel, Field, validator


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


class PrecisionType(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"


class DeviceConfig(BaseModel):
    """Hardware acceleration configuration."""
    device: DeviceType = Field(default=DeviceType.CUDA, description="Compute device")
    precision: PrecisionType = Field(default=PrecisionType.FP16, description="Calculation precision")
    gpu_id: int = Field(default=0, ge=0, description="CUDA device index")
    enable_cudnn_benchmark: bool = True
    
    @validator("device")
    def validate_device_availability(cls, v):
        # In a real app, we might check torch.cuda.is_available() here
        # keeping it pure config for now.
        return v


class PathsConfig(BaseModel):
    """FileSystem paths configuration."""
    input_dir: Path = Field(..., description="Source directory for processing")
    output_dir: Path = Field(..., description="Destination directory")
    models_dir: Path = Field(default=Path("models"), description="Model weights cache")
    temp_dir: Path = Field(default=Path("tmp"), description="Temporary processing artifacts")
    
    # Optional specific file overrides
    log_file: Optional[Path] = None


class PerformanceConfig(BaseModel):
    """Runtime performance tuning."""
    batch_size: int = Field(default=1, ge=1, description="Processing batch size")
    num_workers: int = Field(default=4, ge=0, description="DataLoader workers")
    tile_size: int = Field(default=512, ge=256, description="Tiled processing size (0 for full image)")
    tile_overlap: int = Field(default=64, ge=0, description="Overlap between tiles in pixels")
    memory_limit_gb: float = Field(default=8.0, gt=0, description="VRAM limit hint")


class OutputConfig(BaseModel):
    """Output format settings."""
    format: Literal["jpg", "png", "tiff", "exr"] = "jpg"
    quality: int = Field(default=95, ge=1, le=100, description="JPEG/Compression quality")
    preserve_metadata: bool = True
    embed_workflow: bool = Field(default=True, description="Embed generation metadata in image")
    naming_pattern: str = "{original_name}_enhanced"


class ValidationConfig(BaseModel):
    """Quality Gate settings."""
    enabled: bool = True
    min_resolution: int = 1024
    max_resolution: int = 8192
    check_blur: bool = True
    check_exposure: bool = True
    # Integration with VLM validators
    semantic_validation: bool = False
    allowed_materials: List[str] = Field(default_factory=list)


class ConfigSchema(BaseModel):
    """Root configuration object."""
    version: str = "1.0.0"
    mode: Literal["render", "process", "analyze"] = "render"
    
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    paths: PathsConfig
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    
    # Allow arbitrary extra fields for pipeline-specific params (e.g. 'skygan')
    pipeline_params: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "ignore" # Ignore unknown fields to allow forward compatibility
