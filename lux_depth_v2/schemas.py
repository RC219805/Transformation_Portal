"""Output Schemas for Lux Depth V2 Pipeline.

Provides versioned, deterministic output schemas for:
- Processing reports (per-image results)
- Run cards (batch/service execution evidence)
- Error payloads (consistent service errors)

Schema versioning ensures backward compatibility and reliable automation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time
import platform
import sys


# Schema version follows semantic versioning
SCHEMA_VERSION = "2.0.0"
PIPELINE_VERSION = "2.0.0"  # Matches lux_depth_v2 version


class ProcessingStatus(str, Enum):
    """Processing status for images and batches."""
    
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    PARTIAL = "partial"  # Some stages succeeded, some failed


class StageName(str, Enum):
    """Pipeline stage names for tracking."""
    
    INGEST = "ingest"
    DEPTH_INFERENCE = "depth_inference"
    MATERIAL_SEGMENTATION = "material_segmentation"
    POST_PROCESSING = "post_processing"
    UPSCALING = "upscaling"
    EXPORT = "export"


@dataclass
class StageResult:
    """Result of a single pipeline stage."""
    
    stage: str  # StageName.value
    status: str  # ProcessingStatus.value
    elapsed_ms: float
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ImageReport:
    """Processing report for a single image.
    
    This is the canonical output format for lux_depth_v2.
    Schema version and pipeline version are always included.
    """
    
    # Required fields (contract)
    schema_version: str = SCHEMA_VERSION
    pipeline_version: str = PIPELINE_VERSION
    image_path: str = ""
    status: str = ProcessingStatus.SKIPPED.value
    
    # Output artifacts (standardized naming)
    output_master16: Optional[str] = None  # *_master16.tif
    output_upscaled16: Optional[str] = None  # *_upscaled16.tif
    output_marketing: Optional[str] = None  # *_marketing.png
    output_depth: Optional[str] = None  # *_depth.tif
    output_metadata: Optional[str] = None  # *_metadata.json
    
    # Timing and stages
    elapsed_ms: float = 0.0
    stages: List[StageResult] = field(default_factory=list)
    
    # Configuration snapshot
    preset: Optional[str] = None
    device: Optional[str] = None
    upscale_factor: Optional[int] = None
    
    # Errors and warnings
    error: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert StageResult objects to dicts
        data["stages"] = [s.to_dict() if hasattr(s, "to_dict") else s for s in self.stages]
        return data
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, output_path: Path) -> None:
        """Save report to JSON file."""
        output_path.write_text(self.to_json())


@dataclass
class RunCard:
    """Execution evidence for batch or service runs.
    
    Answers: What ran, with what config, on what device, for how long,
    producing what artifacts, with what warnings/errors.
    
    This is the operational observability artifact.
    """
    
    # Required fields
    schema_version: str = SCHEMA_VERSION
    pipeline_version: str = PIPELINE_VERSION
    run_id: str = ""
    timestamp: float = field(default_factory=time.time)
    
    # Execution context
    device: str = "cpu"
    preset: str = "photo_realistic"
    execution_mode: str = "batch"  # 'batch', 'service', 'single'
    
    # Input summary
    input_count: int = 0
    input_dir: Optional[str] = None
    input_files: List[str] = field(default_factory=list)  # Relative paths or basenames
    
    # Output summary
    output_dir: str = ""
    artifacts: List[str] = field(default_factory=list)  # Generated output files
    
    # Timing breakdown
    total_elapsed_ms: float = 0.0
    stage_timings: Dict[str, float] = field(default_factory=dict)  # stage_name -> total_ms
    
    # Results summary
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    
    # Errors and warnings (aggregated)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # System info
    python_version: str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    platform: str = platform.system()
    
    # Optional fields
    git_sha: Optional[str] = None
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save(self, output_path: Path) -> None:
        """Save run card to JSON file."""
        output_path.write_text(self.to_json())


@dataclass
class ServiceError:
    """Consistent error payload for service endpoints.
    
    Provides actionable error responses with:
    - Error code for programmatic handling
    - Human-readable message
    - Optional hint for resolution
    - Request ID for debugging
    """
    
    error_code: str  # e.g., "INVALID_INPUT", "PROCESSING_FAILED"
    message: str
    hint: Optional[str] = None
    request_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# Preset metadata for governance
@dataclass
class PresetMetadata:
    """Metadata for a single preset.
    
    Used for --list-presets and --describe-preset.
    """
    
    name: str
    display_name: str
    description: str
    intended_use: str
    quality_tier: str  # 'standard', 'max', 'apex'
    stability: str  # 'stable', 'canary', 'experimental'
    performance: Dict[str, Any] = field(default_factory=dict)  # throughput, memory, etc.
    parameters: Dict[str, Any] = field(default_factory=dict)  # preset parameters
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# Validation helpers
def validate_schema_version(data: Dict[str, Any], required_major: int = 2) -> bool:
    """Validate that schema version is compatible.
    
    Args:
        data: Parsed JSON data
        required_major: Required major version
        
    Returns:
        True if compatible, False otherwise
    """
    if "schema_version" not in data:
        return False
    
    version = data["schema_version"]
    try:
        major = int(version.split(".")[0])
        return major == required_major
    except (ValueError, IndexError):
        return False


def load_image_report(path: Path) -> ImageReport:
    """Load and validate an ImageReport from JSON.
    
    Args:
        path: Path to report JSON
        
    Returns:
        ImageReport instance
        
    Raises:
        ValueError: If schema version is incompatible
    """
    data = json.loads(path.read_text())
    
    if not validate_schema_version(data):
        raise ValueError(f"Incompatible schema version: {data.get('schema_version')}")
    
    # Reconstruct StageResult objects
    stages = []
    for stage_data in data.get("stages", []):
        if isinstance(stage_data, dict):
            stages.append(StageResult(**stage_data))
        else:
            stages.append(stage_data)
    
    data["stages"] = stages
    return ImageReport(**data)


def load_run_card(path: Path) -> RunCard:
    """Load and validate a RunCard from JSON.
    
    Args:
        path: Path to run card JSON
        
    Returns:
        RunCard instance
        
    Raises:
        ValueError: If schema version is incompatible
    """
    data = json.loads(path.read_text())
    
    if not validate_schema_version(data):
        raise ValueError(f"Incompatible schema version: {data.get('schema_version')}")
    
    return RunCard(**data)
