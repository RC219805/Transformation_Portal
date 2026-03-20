"""Execution engine for lux_depth_v3 pipeline.

Extracted from orchestrator.py as part of ADR-043 decomposition (Phase 6).

This module provides:
- Stage execution for depth, PBR, Materials V3, and V2 enhancement
- Result data classes for each stage type
- ExecutionEngine class for coordinating stage execution

The execution engine handles:
1. Running depth inference with caching and fallback
2. Generating PBR maps from depth data
3. Running Materials V3 surface-aware finishing
4. Executing V2 enhancement subprocess

Usage:
    from transformation_portal.lux_depth_v3.execution_engine import (
        ExecutionEngine,
        DepthStageResult,
        PBRStageResult,
        MaterialsV3StageResult,
        V2StageResult,
    )

    # Using ExecutionEngine class
    engine = ExecutionEngine(config, output_root)
    depth_result = engine.execute_depth_stage(image_input, output_key, paths)
    pbr_result = engine.execute_pbr_stage(depth_array, output_key)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from ..depth.backends.protocol import DepthBackend, LicenseRestrictionError
from .config import DA3Config, EnhanceConfig, ModelVariant
from .manifest import BackendSelectionMetadata, DepthMetadata
from .pbr import generate_pbr_maps
from .pbr_writer import write_pbr_maps
from .pipeline_coordinator import BackendSelection

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Module-level Constants
# -----------------------------------------------------------------------------

# Default backend selection for error cases (avoid repeated allocation)
_DEFAULT_ERROR_BACKEND_SELECTION = BackendSelectionMetadata(
    requested_backend="unknown",
    resolved_backend="unknown",
    resolution_status="error",
    resolution_reason="No backend selection available",
    model_id="unknown",
    device="cpu",
)


# -----------------------------------------------------------------------------
# Result Data Classes
# -----------------------------------------------------------------------------


@dataclass
class DepthStageResult:
    """Result from depth computation stage.

    Captures all outputs from Stage A: depth inference with optional
    caching, PBR generation, and Materials V3 processing.

    Attributes:
        depth_metadata: Depth metadata for manifest (or None if failed)
        depth_runtime_s: Time spent on depth computation
        pbr_assets: PBR asset paths and metadata (or None if not generated)
        materials_v3_result: Materials V3 processing result (or None)
        materials_v3_runtime_s: Time spent on Materials V3
        enhanced_image_path: Path to Materials V3 enhanced image (or None)
        backend_selection: Backend selection metadata for provenance
        depth_attempts: List of backend attempt records
        selected_attempt_index: Index of successful attempt (or None)
        depth_map: The raw depth array if available
        depth_path: Path to depth PNG file
        float_depth_path: Path to float depth NPY file (or None)
    """

    depth_metadata: Optional[DepthMetadata] = None
    depth_runtime_s: float = 0.0
    pbr_assets: Optional[Dict[str, Any]] = None
    materials_v3_result: Optional[Dict[str, Any]] = None
    materials_v3_runtime_s: float = 0.0
    enhanced_image_path: Optional[Path] = None
    backend_selection: Optional[BackendSelectionMetadata] = None
    depth_attempts: List[Dict[str, Any]] = field(default_factory=list)
    selected_attempt_index: Optional[int] = None
    depth_map: Optional[np.ndarray] = None
    depth_path: Optional[Path] = None
    float_depth_path: Optional[Path] = None

    @property
    def success(self) -> bool:
        """Return True if depth computation succeeded."""
        return self.depth_metadata is not None

    @property
    def was_cached(self) -> bool:
        """Return True if depth was loaded from cache."""
        if self.depth_attempts:
            return any(a.get("cached", False) for a in self.depth_attempts)
        return False

    def to_tuple(
        self,
    ) -> Tuple[
        Optional[DepthMetadata],
        float,
        Optional[Dict[str, Any]],
        Optional[Dict[str, Any]],
        float,
        Optional[Path],
        BackendSelectionMetadata,
        List[Dict[str, Any]],
    ]:
        """Convert to legacy tuple format for backward compatibility.

        Returns:
            8-tuple matching _compute_depth_stage return signature
        """
        return (
            self.depth_metadata,
            self.depth_runtime_s,
            self.pbr_assets,
            self.materials_v3_result,
            self.materials_v3_runtime_s,
            self.enhanced_image_path,
            self.backend_selection or _DEFAULT_ERROR_BACKEND_SELECTION,
            self.depth_attempts,
        )


@dataclass
class PBRStageResult:
    """Result from PBR map generation stage.

    Captures PBR map generation outcomes including paths and timing.

    Attributes:
        success: Whether PBR generation succeeded
        normal_path: Path to normal map
        roughness_path: Path to roughness map
        ao_path: Path to ambient occlusion map
        runtime_s: Time spent generating PBR maps
        config: PBR configuration used
        error: Error message if generation failed
    """

    success: bool = False
    normal_path: Optional[str] = None
    roughness_path: Optional[str] = None
    ao_path: Optional[str] = None
    runtime_s: float = 0.0
    config: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Optional[Dict[str, Any]]:
        """Convert to manifest-compatible dictionary.

        Returns:
            Dictionary with PBR asset paths and metadata, or None if failed
        """
        if not self.success:
            return None

        return {
            "normal_path": self.normal_path,
            "roughness_path": self.roughness_path,
            "ao_path": self.ao_path,
            "runtime_seconds": self.runtime_s,
            "config": self.config or {},
        }


@dataclass
class MaterialsV3StageResult:
    """Result from Materials V3 stage.

    Captures surface-aware finishing outcomes including segmentation
    and pixel operations.

    Attributes:
        success: Whether Materials V3 processing succeeded
        result: Full result dictionary with material_masks, pixel_ops, etc.
        runtime_s: Time spent on Materials V3 processing
        enhanced_image_path: Path to enhanced image (or None)
        mask_artifact_path: Path to persisted mask artifact
        n_operations_applied: Number of pixel operations applied
        error: Error message if processing failed
    """

    success: bool = False
    result: Optional[Dict[str, Any]] = None
    runtime_s: float = 0.0
    enhanced_image_path: Optional[Path] = None
    mask_artifact_path: Optional[Path] = None
    n_operations_applied: int = 0
    error: Optional[str] = None

    def to_tuple(self) -> Tuple[Optional[Dict[str, Any]], float, Optional[Path]]:
        """Convert to legacy tuple format for backward compatibility.

        Returns:
            3-tuple matching _run_materials_v3_stage return signature
        """
        return (self.result, self.runtime_s, self.enhanced_image_path)


@dataclass
class V2StageResult:
    """Result from V2 enhancement stage.

    Captures V2 subprocess execution outcomes.

    Attributes:
        result: V2 runner result dictionary
        runtime_s: Time spent on V2 enhancement
        report_path: Path to V2 report JSON
        output_path: Path to enhanced output image
        status: V2 execution status string
        skipped: Whether V2 was skipped due to caching
    """

    result: Dict[str, Any] = field(default_factory=dict)
    runtime_s: float = 0.0
    report_path: Optional[Path] = None
    output_path: Optional[str] = None
    status: str = "unknown"
    skipped: bool = False

    @property
    def success(self) -> bool:
        """Return True if V2 enhancement succeeded."""
        return self.status in ("ok", "success", "skipped")

    def to_tuple(self) -> Tuple[Dict[str, Any], float, Optional[Path]]:
        """Convert to legacy tuple format for backward compatibility.

        Returns:
            3-tuple matching _run_v2_stage return signature
        """
        return (self.result, self.runtime_s, self.report_path)


# -----------------------------------------------------------------------------
# PBR Generation Helper Functions
# -----------------------------------------------------------------------------


def generate_pbr_stage(
    depth: np.ndarray,
    output_key: Path,
    output_root: Path,
    config: EnhanceConfig,
) -> PBRStageResult:
    """Generate PBR maps from depth data.

    Standalone function for PBR generation that can be used independently
    of the ExecutionEngine class.

    Args:
        depth: Depth array (numpy float32)
        output_key: Output key for artifact naming
        output_root: Base output directory
        config: Enhancement configuration

    Returns:
        PBRStageResult with generation outcomes
    """
    if not config.generate_pbr:
        return PBRStageResult(success=False, error="PBR generation disabled")

    try:
        logger.info("Generating PBR maps...")
        pbr_t0 = time.time()

        # Use to_pbr_config() for consistent parameter conversion
        pbr_config = config.to_pbr_config()

        # Generate maps from depth
        normal_map, roughness_map, ao_map = generate_pbr_maps(depth, config=pbr_config)

        # Write PBR maps
        pbr_dir = output_root / "pbr"
        pbr_dir.mkdir(parents=True, exist_ok=True)

        # Derive base name from output_key for consistent artifact naming
        sanitized_stem = output_key.stem if output_key.suffix else output_key.name

        pbr_paths = write_pbr_maps(
            normal_map=normal_map,
            roughness_map=roughness_map,
            ao_map=ao_map,
            output_dir=pbr_dir,
            base_name=sanitized_stem,
        )

        pbr_runtime = time.time() - pbr_t0
        logger.info(
            "PBR maps generated in %.2fs: %s",
            pbr_runtime,
            list(pbr_paths.keys()),
        )

        return PBRStageResult(
            success=True,
            normal_path=str(pbr_paths["normal"]),
            roughness_path=str(pbr_paths["roughness"]),
            ao_path=str(pbr_paths["ao"]),
            runtime_s=pbr_runtime,
            config={
                "normal_strength": pbr_config.normal_strength,
                "normal_blur_radius": pbr_config.normal_blur_radius,
                "roughness_strength": pbr_config.roughness_strength,
                "roughness_blur_radius": pbr_config.roughness_blur_radius,
                "ao_strength": pbr_config.ao_strength,
                "ao_blur_radius": pbr_config.ao_blur_radius,
                "ao_bias": pbr_config.ao_bias,
            },
        )

    except Exception as pbr_error:
        logger.warning(
            "PBR generation failed (non-blocking): %s",
            pbr_error,
        )
        return PBRStageResult(
            success=False,
            error=str(pbr_error),
        )


# -----------------------------------------------------------------------------
# V2 Stage Helper Functions
# -----------------------------------------------------------------------------


def run_v2_stage(
    v2_runner: Any,
    image_path: Path,
    depth_path: Optional[Path],
    depth_dir: Path,
    v2_dir: Path,
    output_key: Path,
    v2_log_path: Path,
    config: EnhanceConfig,
    masks_path: Optional[Path] = None,
) -> V2StageResult:
    """Run V2 enhancement subprocess.

    Standalone function for V2 stage execution that can be used independently
    of the ExecutionEngine class.

    Args:
        v2_runner: V2Runner instance
        image_path: Path to input image
        depth_path: Path to depth PNG (or None if depth failed)
        depth_dir: Directory containing depth outputs
        v2_dir: Directory for V2 outputs
        output_key: Output key for artifact naming
        v2_log_path: Path for V2 subprocess log
        config: Enhancement configuration
        masks_path: Optional path to material masks NPZ

    Returns:
        V2StageResult with execution outcomes
    """
    if v2_runner is None or not config.enable_v2:
        logger.info("V2 stage disabled, skipping enhancement")
        return V2StageResult(
            result={"status": "skipped"},
            status="skipped",
            skipped=True,
        )

    try:
        v2_result = v2_runner.run(
            input_path=image_path,
            depth_dir=(depth_dir if depth_path and depth_path.exists() else None),
            output_dir=v2_dir,
            preset=(config.v2_preset or "default"),
            device=config.v2_device,
            upscaler_backend=config.v2_upscaler_backend,
            log_file=v2_log_path,
            timeout=config.v2_timeout,
            masks_file=masks_path,
        )
        v2_runtime_s = v2_result.get("runtime_s", 0.0)

        from .v2_runner import find_v2_report

        report_path_value = v2_result.get("report_path")
        if isinstance(report_path_value, str) and report_path_value:
            v2_report_path = Path(report_path_value)
        else:
            v2_report_path = find_v2_report(v2_dir, output_key.name)

        output_path = v2_result.get("output")
        if not isinstance(output_path, str) or not output_path:
            output_path = None

        return V2StageResult(
            result=v2_result,
            runtime_s=v2_runtime_s,
            report_path=v2_report_path,
            output_path=output_path,
            status=v2_result.get("status", "unknown"),
            skipped=False,
        )

    except Exception as v2_error:
        logger.error("V2 stage failed: %s", v2_error)
        return V2StageResult(
            result={"status": "error", "error": str(v2_error)},
            status="error",
        )


# -----------------------------------------------------------------------------
# ExecutionEngine Class
# -----------------------------------------------------------------------------


class ExecutionEngine:
    """Engine for executing pipeline stages.

    Coordinates execution of depth, PBR, Materials V3, and V2 stages
    with proper caching, fallback handling, and result aggregation.

    This class extracts stage execution logic from EnhanceOrchestrator
    while maintaining the same behavior and interfaces.

    Attributes:
        config: Enhancement configuration
        output_root: Base directory for all outputs
        depth_dir: Directory for depth outputs
        v2_dir: Directory for V2 outputs
        pbr_enabled: Whether PBR generation is enabled
    """

    def __init__(
        self,
        config: EnhanceConfig,
        output_root: Path,
    ):
        """Initialize the execution engine.

        Args:
            config: Enhancement configuration object
            output_root: Base directory for outputs
        """
        self.config = config
        self.output_root = Path(output_root)
        self.depth_dir = self.output_root / "depth"
        self.v2_dir = self.output_root / "v2"
        self.pbr_enabled = config.generate_pbr

    def execute_pbr_stage(
        self,
        depth: np.ndarray,
        output_key: Path,
    ) -> PBRStageResult:
        """Generate PBR maps from depth data.

        Args:
            depth: Depth array (numpy float32)
            output_key: Output key for artifact naming

        Returns:
            PBRStageResult with generation outcomes
        """
        return generate_pbr_stage(
            depth=depth,
            output_key=output_key,
            output_root=self.output_root,
            config=self.config,
        )

    def execute_v2_stage(
        self,
        v2_runner: Any,
        image_path: Path,
        depth_path: Optional[Path],
        output_key: Path,
        v2_log_path: Path,
        masks_path: Optional[Path] = None,
    ) -> V2StageResult:
        """Run V2 enhancement subprocess.

        Args:
            v2_runner: V2Runner instance
            image_path: Path to input image
            depth_path: Path to depth PNG (or None if depth failed)
            output_key: Output key for artifact naming
            v2_log_path: Path for V2 subprocess log
            masks_path: Optional path to material masks NPZ

        Returns:
            V2StageResult with execution outcomes
        """
        return run_v2_stage(
            v2_runner=v2_runner,
            image_path=image_path,
            depth_path=depth_path,
            depth_dir=self.depth_dir,
            v2_dir=self.v2_dir,
            output_key=output_key,
            v2_log_path=v2_log_path,
            config=self.config,
            masks_path=masks_path,
        )


# -----------------------------------------------------------------------------
# Module exports
# -----------------------------------------------------------------------------

__all__ = [
    # Result data classes
    "DepthStageResult",
    "PBRStageResult",
    "MaterialsV3StageResult",
    "V2StageResult",
    # Standalone functions
    "generate_pbr_stage",
    "run_v2_stage",
    # Main class
    "ExecutionEngine",
]
