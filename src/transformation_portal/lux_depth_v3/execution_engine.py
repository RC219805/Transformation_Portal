"""Execution engine helpers for the lux_depth_v3 pipeline.

Extracted from orchestrator.py as part of ADR-043 decomposition (Phase 6).

This module provides:
- Helpers for generating and writing PBR maps
- Helpers for invoking the V2 enhancement subprocess
- Helpers for persisting depth artifacts (PNG, metadata JSON)
- Helpers for persisting Materials V3 enhanced images
- Result data classes for the supported stages
- ExecutionEngine class for coordinating PBR + V2 stage execution

The depth inference and Materials V3 segmentation logic remain in the
orchestrator due to tight coupling with per-image state management
(backend fallback tracking, APEX gates, cache coordination). This module
handles the stateless artifact persistence and subprocess coordination.

Usage:
    from transformation_portal.lux_depth_v3.execution_engine import (
        ExecutionEngine,
        PBRStageResult,
        V2StageResult,
        persist_depth_artifacts,
        persist_enhanced_image,
    )

    # Using ExecutionEngine class
    engine = ExecutionEngine(config, output_root)
    pbr_result = engine.execute_pbr_stage(depth_array, output_key)
    v2_result = engine.execute_v2_stage(image_input, output_key, paths)

    # Using standalone artifact persistence
    persist_depth_artifacts(depth_map, depth_path, metadata, config)
    persist_enhanced_image(enhanced_array, output_path, config)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import EnhanceConfig
from .depth_writer import atomic_write_depth_u16_png_with_stats
from .io_atomic import atomic_temp_file, atomic_write_pil_png
from .manifest import BackendSelectionMetadata, DepthMetadata
from .pbr import generate_pbr_maps
from .pbr_writer import write_pbr_maps

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
    timing_ms: Optional[Dict[str, float]] = None
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
            "timing_ms": dict(self.timing_ms or {}),
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
        pbr_perf_t0 = time.perf_counter()

        # Use to_pbr_config() for consistent parameter conversion
        pbr_config = config.to_pbr_config()

        # Generate maps from depth
        pbr_generate_t0 = time.perf_counter()
        normal_map, roughness_map, ao_map = generate_pbr_maps(depth, config=pbr_config)
        pbr_generate_ms = round((time.perf_counter() - pbr_generate_t0) * 1000.0, 3)

        # Write PBR maps
        pbr_write_t0 = time.perf_counter()
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
        pbr_write_ms = round((time.perf_counter() - pbr_write_t0) * 1000.0, 3)

        pbr_runtime = time.time() - pbr_t0
        pbr_total_ms = round((time.perf_counter() - pbr_perf_t0) * 1000.0, 3)
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
            timing_ms={
                "generate_maps": pbr_generate_ms,
                "write_maps": pbr_write_ms,
                "total": pbr_total_ms,
            },
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

    # Ensure output directories exist
    v2_dir.mkdir(parents=True, exist_ok=True)
    v2_log_path.parent.mkdir(parents=True, exist_ok=True)

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
            # Pass canonical asset key for depth/report identity alignment
            asset_key=output_key.name,
            output_bit_depth=config.output_bit_depth,
        )
        v2_runtime_s = v2_result.get("runtime_s", 0.0)

        from .v2_runner import find_v2_report

        report_path_value = v2_result.get("report_path")
        v2_report_path: Optional[Path] = None
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
# Depth Artifact Persistence Helper Functions
# -----------------------------------------------------------------------------


@dataclass
class DepthArtifactPaths:
    """Paths for depth artifacts.

    Attributes:
        depth_path: Path to quantized depth PNG (uint16)
        float_depth_path: Path to float depth NPY (optional)
        metadata_path: Path to depth metadata JSON
    """

    depth_path: Path
    float_depth_path: Optional[Path] = None
    metadata_path: Optional[Path] = None


@dataclass
class DepthArtifactResult:
    """Result from depth artifact persistence.

    Attributes:
        success: Whether persistence succeeded
        depth_path: Path to written depth PNG
        float_depth_path: Path to float depth NPY (or None)
        metadata_path: Path to metadata JSON
        scaling_stats: Depth scaling statistics
        error: Error message if persistence failed
    """

    success: bool = False
    depth_path: Optional[Path] = None
    float_depth_path: Optional[Path] = None
    metadata_path: Optional[Path] = None
    scaling_stats: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def persist_depth_artifacts(
    depth_map: np.ndarray,
    depth_path: Path,
    float_depth_path: Optional[Path],
    depth_metadata: DepthMetadata,
    config: EnhanceConfig,
) -> DepthArtifactResult:
    """Persist depth artifacts to disk.

    Writes the depth map as a quantized uint16 PNG, optionally saves
    the float32 NPY, and writes metadata JSON. This function handles
    the I/O portion of depth stage completion.

    Args:
        depth_map: Float32 depth array from inference
        depth_path: Target path for quantized depth PNG
        float_depth_path: Target path for float depth NPY (optional)
        depth_metadata: Depth metadata for JSON sidecar
        config: Enhancement configuration

    Returns:
        DepthArtifactResult with persistence outcomes
    """
    from ..ingest.canonical_json import dump_json

    try:
        # Ensure parent directories exist
        depth_path.parent.mkdir(parents=True, exist_ok=True)

        # Write quantized depth PNG (uint16)
        _, _, depth_stats = atomic_write_depth_u16_png_with_stats(
            depth_path,
            depth_map,
            method=config.depth_quantization,
            debug_verify=config.verify_depth_writes,
            compute_encoded_unique_values=(str(getattr(config, "quality_tier", "")).lower() == "apex"),
        )

        # Save float depth NPY if enabled
        if float_depth_path and getattr(config, "save_float_depth", False):
            float_depth_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(float_depth_path), depth_map)
            logger.debug("Saved float depth: %s", float_depth_path)

        # Write depth metadata JSON sidecar
        # Use actual computed values to ensure sidecar matches persisted artifacts
        metadata_path = depth_path.parent / f"{depth_path.stem}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            dump_json(
                {
                    "model": depth_metadata.model,
                    "depth_path": str(depth_path),
                    "runtime_seconds": depth_metadata.runtime_seconds,
                    "scaling": depth_stats._asdict() if depth_stats else depth_metadata.scaling,
                    "stats": depth_stats._asdict() if depth_stats else depth_metadata.stats,
                },
                f,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
        logger.debug("Wrote depth metadata: %s", metadata_path)

        return DepthArtifactResult(
            success=True,
            depth_path=depth_path,
            float_depth_path=(float_depth_path if float_depth_path and getattr(config, "save_float_depth", False) else None),
            metadata_path=metadata_path,
            # depth_stats is DepthWriteStats (dataclass with _asdict() method)
            scaling_stats=depth_stats._asdict() if depth_stats else None,
        )

    except Exception as e:
        logger.error("Depth artifact persistence failed: %s", e)
        return DepthArtifactResult(
            success=False,
            error=str(e),
        )


# -----------------------------------------------------------------------------
# Enhanced Image Persistence Helper Functions
# -----------------------------------------------------------------------------


@dataclass
class EnhancedImageResult:
    """Result from enhanced image persistence.

    Attributes:
        success: Whether persistence succeeded
        output_path: Path to written enhanced image
        format: Output format ("png" or "tiff")
        bit_depth: Bit depth (8 or 16)
        n_operations_applied: Number of pixel operations applied
        error: Error message if persistence failed
    """

    success: bool = False
    output_path: Optional[Path] = None
    format: str = "png"
    bit_depth: int = 8
    n_operations_applied: int = 0
    error: Optional[str] = None


def persist_enhanced_image(
    enhanced_image: np.ndarray,
    output_path: Path,
    config: EnhanceConfig,
    n_operations_applied: int = 0,
) -> EnhancedImageResult:
    """Persist Materials V3 enhanced image to disk.

    Writes the enhanced image as either 8-bit PNG or 16-bit TIFF
    based on configuration. This function handles the I/O portion
    of Materials V3 stage completion.

    Args:
        enhanced_image: Float32 enhanced image array (normalized 0-1)
        output_path: Target path for output image
        config: Enhancement configuration
        n_operations_applied: Number of pixel operations applied (for logging)

    Returns:
        EnhancedImageResult with persistence outcomes
    """
    from PIL import Image as PILImage

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if config.output_bit_depth == 16:
            # 16-bit TIFF output
            import tifffile

            enhanced_uint16 = (np.clip(enhanced_image, 0, 1) * 65535 + 0.5).astype(np.uint16)

            # Ensure .tif extension for 16-bit
            if output_path.suffix.lower() not in {".tif", ".tiff"}:
                output_path = output_path.with_suffix(".tif")

            with atomic_temp_file(
                output_path,
                suffix=".tif",
                create_file=False,
            ) as temp_path:
                tifffile.imwrite(
                    temp_path,
                    enhanced_uint16,
                    photometric="rgb",
                    compression="lzw",
                    metadata={"software": "Transformation Portal v3"},
                )

            logger.info(
                "Materials V3 enhanced image with %d pixel operations - " "saved to %s (16-bit TIFF) for V2 stage",
                n_operations_applied,
                output_path,
            )

            return EnhancedImageResult(
                success=True,
                output_path=output_path,
                format="tiff",
                bit_depth=16,
                n_operations_applied=n_operations_applied,
            )

        else:
            # 8-bit PNG output
            enhanced_uint8 = (np.clip(enhanced_image, 0, 1) * 255).astype(np.uint8)

            # Ensure .png extension for 8-bit
            if output_path.suffix.lower() != ".png":
                output_path = output_path.with_suffix(".png")

            output_path = atomic_write_pil_png(
                output_path,
                PILImage.fromarray(enhanced_uint8),
                optimize=True,
            )

            logger.info(
                "Materials V3 enhanced image with %d pixel operations - " "saved to %s (8-bit PNG) for V2 stage",
                n_operations_applied,
                output_path,
            )

            return EnhancedImageResult(
                success=True,
                output_path=output_path,
                format="png",
                bit_depth=8,
                n_operations_applied=n_operations_applied,
            )

    except Exception as e:
        logger.error("Enhanced image persistence failed: %s", e)
        return EnhancedImageResult(
            success=False,
            error=str(e),
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

    def persist_depth(
        self,
        depth_map: np.ndarray,
        output_key: Path,
        depth_metadata: DepthMetadata,
    ) -> DepthArtifactResult:
        """Persist depth artifacts to disk.

        Args:
            depth_map: Float32 depth array from inference
            output_key: Output key for artifact naming (preserves parent structure)
            depth_metadata: Depth metadata for JSON sidecar

        Returns:
            DepthArtifactResult with persistence outcomes
        """
        # Preserve output_key.parent structure to match orchestrator path layout
        # Pattern: depth_dir / output_key.parent / {output_key.name}_depth.png
        depth_path = self.depth_dir / output_key.parent / f"{output_key.name}_depth.png"
        float_depth_path = (
            self.depth_dir / output_key.parent / f"{output_key.name}_depth.npy"
            if getattr(self.config, "save_float_depth", False)
            else None
        )

        return persist_depth_artifacts(
            depth_map=depth_map,
            depth_path=depth_path,
            float_depth_path=float_depth_path,
            depth_metadata=depth_metadata,
            config=self.config,
        )

    def persist_enhanced(
        self,
        enhanced_image: np.ndarray,
        output_key: Path,
        n_operations_applied: int = 0,
    ) -> EnhancedImageResult:
        """Persist Materials V3 enhanced image to disk.

        Args:
            enhanced_image: Float32 enhanced image array (normalized 0-1)
            output_key: Output key for artifact naming (preserves parent structure)
            n_operations_applied: Number of pixel operations applied

        Returns:
            EnhancedImageResult with persistence outcomes
        """
        temp_dir = self.output_root / "temp"
        extension = ".tif" if self.config.output_bit_depth == 16 else ".png"
        # Preserve output_key.parent structure to match orchestrator path layout
        output_path = temp_dir / output_key.parent / f"{output_key.name}_materials_v3_enhanced{extension}"

        return persist_enhanced_image(
            enhanced_image=enhanced_image,
            output_path=output_path,
            config=self.config,
            n_operations_applied=n_operations_applied,
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
    # Artifact result data classes
    "DepthArtifactPaths",
    "DepthArtifactResult",
    "EnhancedImageResult",
    # Standalone functions
    "generate_pbr_stage",
    "run_v2_stage",
    "persist_depth_artifacts",
    "persist_enhanced_image",
    # Main class
    "ExecutionEngine",
]
