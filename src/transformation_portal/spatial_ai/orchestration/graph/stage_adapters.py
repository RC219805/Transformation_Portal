"""Stage adapters for bridging legacy pipeline stages with ExecutionGraph (ADR-029).

This module provides concrete Stage implementations that wrap existing pipeline
functionality to enable graph-based execution.

Architecture (ADR-029):
- Pure function semantics: Each stage is a stateless transformation
- Explicit resource declarations: GPU/CPU memory, time estimates
- Content-addressable caching: Deterministic cache keys from inputs
- Zero coupling: Adapters import from spatial_ai modules only

Stage Implementations:
- IngestStage: Wraps LinearDecoder for gamma=1.0 ingest
- SegmentationStage: Wraps SAM2Backend for mask generation
- MaterialsStage: Wraps MaterialBackend for PBR texture generation

Example:
    >>> from transformation_portal.spatial_ai.orchestration.graph import (
    ...     ExecutionGraph,
    ...     Executor,
    ... )
    >>> from transformation_portal.spatial_ai.orchestration.graph.stage_adapters import (
    ...     IngestStage,
    ...     SegmentationStage,
    ...     MaterialsStage,
    ... )
    >>>
    >>> # Build graph with stage adapters
    >>> graph = ExecutionGraph()
    >>> graph.add_stage("ingest", IngestStage(), inputs={})
    >>> graph.add_stage(
    ...     "segment",
    ...     SegmentationStage(),
    ...     inputs={"linear_rgb": "ingest.linear_rgb"},
    ... )
    >>> graph.add_stage(
    ...     "materials",
    ...     MaterialsStage(),
    ...     inputs={
    ...         "linear_rgb": "ingest.linear_rgb",
    ...         "masks": "segment.masks",
    ...     },
    ... )
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .stage import CheckpointPolicy, ResourceRequirements, StageMetadata

logger = logging.getLogger(__name__)

# Resource constants for GPU memory requirements (MB)
# These values are based on empirical measurements with RTX 3090
SAM2_LARGE_GPU_MB = 2048  # SAM2 Large model GPU memory
SAM2_BASE_GPU_MB = 1024  # SAM2 Base model GPU memory
MATERIALS_GPU_MB = 2048  # Neural materials backend GPU memory


@dataclass
class IngestStageConfig:
    """Configuration for IngestStage.

    Attributes:
        strict_ingest: Reject 8-bit inputs (require 16-bit or higher).
        emit_exr: Save linear output as OpenEXR file.
        emit_provenance: Save provenance metadata alongside output.
    """

    strict_ingest: bool = False
    emit_exr: bool = False
    emit_provenance: bool = False


class IngestStage:
    """Stage adapter for LinearDecoder (Phase 1 ingest).

    Transforms raw input images to linear RGB (gamma=1.0) with optional
    EXR output and provenance tracking.

    Inputs:
        input_path: Path to input image file.

    Outputs:
        linear_rgb: (H, W, 3) float32 linear RGB array.
        input_size: (height, width) tuple.
        input_dtype: Original input dtype string.
        gamma: Always 1.0 (enforced).

    Resource Profile:
        GPU: 0 MB (CPU-only)
        CPU: 512 MB
        Time: ~500ms (typical TIFF)
    """

    def __init__(self, config: Optional[IngestStageConfig] = None):
        """Initialize IngestStage.

        Args:
            config: Optional configuration. Defaults to IngestStageConfig().
        """
        self._config = config or IngestStageConfig()

    @property
    def metadata(self) -> StageMetadata:
        """Stage metadata for IngestStage."""
        return StageMetadata(
            name="linear_ingest",
            version="1.0.0",
            description="Linear RGB ingest with gamma=1.0 enforcement (Phase 1)",
            resource_requirements=ResourceRequirements(
                gpu_memory_mb=0,
                cpu_memory_mb=512,
                min_disk_mb=50,
                gpu_required=False,
                estimated_time_ms=500,
                can_parallelize=True,
            ),
            deterministic=True,
            idempotent=True,
            checkpoint_policy=CheckpointPolicy.ALWAYS,
        )

    def execute(self, inputs: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Execute linear ingest.

        Args:
            inputs: Must contain "input_path" key.
            context: ExecutionContext with device, config, output_dir.

        Returns:
            Dict with linear_rgb, input_size, input_dtype, gamma.

        Raises:
            ValueError: If input_path missing or invalid.
            RuntimeError: If decode fails.
        """
        from transformation_portal.spatial_ai.ingest.linear_decoder import (
            LinearDecoder,
        )

        input_path = inputs.get("input_path")
        if input_path is None:
            raise ValueError("IngestStage requires 'input_path' input")

        input_path = Path(input_path)
        if not input_path.exists():
            raise ValueError(f"Input file not found: {input_path}")

        # Get config from context or use defaults
        config = context.config if hasattr(context, "config") else {}
        strict = config.get("strict_ingest", self._config.strict_ingest)
        emit_exr = config.get("emit_exr", self._config.emit_exr)
        emit_prov = config.get("emit_provenance", self._config.emit_provenance)

        # Output directory from context
        output_dir = getattr(context, "output_dir", None)
        if output_dir:
            output_dir = Path(output_dir)

        # Execute decoder
        decoder = LinearDecoder(gamma=1.0, bit_depth=32, strict_ingest=strict)
        result = decoder.decode(
            input_path=input_path,
            output_dir=output_dir,
            emit_exr=emit_exr and output_dir is not None,
            emit_provenance=emit_prov and output_dir is not None,
        )

        return {
            "linear_rgb": result.linear_rgb,
            "input_size": result.input_size,
            "input_dtype": result.input_dtype,
            "gamma": 1.0,
        }

    def compute_cache_key(self, inputs: Dict[str, Any], context: Any) -> str:
        """Compute content-addressed cache key.

        Key components:
        - Stage version
        - Input file content hash (first 4KB + size)
        - Strict ingest flag

        Args:
            inputs: Stage inputs.
            context: Execution context.

        Returns:
            64-character SHA256 hex string.
        """
        input_path = inputs.get("input_path")
        if input_path is None:
            # No input - return deterministic empty key
            return hashlib.sha256(b"ingest:no_input").hexdigest()

        input_path = Path(input_path)

        # Hash full file content (streaming SHA256 for collision resistance)
        # Note: For very large files (>100MB), this can impact pipeline throughput.
        # Future optimization: Add a size threshold for hybrid approach (size + chunks).
        file_hash = "missing"
        if input_path.exists():
            try:
                hasher = hashlib.sha256()
                # Stream file in chunks to avoid memory issues with large files
                chunk_size = 65536  # 64KB chunks
                with open(input_path, "rb") as f:
                    while chunk := f.read(chunk_size):
                        hasher.update(chunk)
                file_hash = hasher.hexdigest()
            except (OSError, IOError) as e:
                # Handle permission errors, I/O errors gracefully
                logger.warning(f"Could not read file for cache key: {e}")
                file_hash = "error"

        # Config components
        config = context.config if hasattr(context, "config") else {}
        strict = config.get("strict_ingest", self._config.strict_ingest)

        # Combine components
        components = [
            f"v{self.metadata.version}",
            f"file:{file_hash}",
            f"strict:{strict}",
        ]
        combined = "|".join(components)
        return hashlib.sha256(combined.encode()).hexdigest()


class SegmentationStage:
    """Stage adapter for SAM2Backend (Phase 2.1 segmentation).

    Generates automatic segmentation masks using SAM2.

    Inputs:
        linear_rgb: (H, W, 3) float32 linear RGB array.

    Outputs:
        masks: (N, H, W) bool array of segmentation masks.
        scores: (N,) float32 confidence scores.
        num_masks: Number of masks generated.

    Resource Profile:
        GPU: 2048 MB (SAM2 Large)
        CPU: 512 MB
        Time: ~3000ms (1024x1024)
    """

    def __init__(
        self,
        model_size: str = "large",
        enable_material_classification: bool = False,
    ):
        """Initialize SegmentationStage.

        Args:
            model_size: SAM2 model size ("base" or "large").
            enable_material_classification: Enable CLIP material classification.
        """
        self._model_size = model_size
        self._enable_material_classification = enable_material_classification

    @property
    def metadata(self) -> StageMetadata:
        """Stage metadata for SegmentationStage."""
        # GPU memory depends on model size
        gpu_mb = SAM2_LARGE_GPU_MB if self._model_size == "large" else SAM2_BASE_GPU_MB

        return StageMetadata(
            name="sam2_segmentation",
            version="2.1.0",
            description="SAM2 automatic mask generation (Phase 2.1)",
            resource_requirements=ResourceRequirements(
                gpu_memory_mb=gpu_mb,
                cpu_memory_mb=512,
                min_disk_mb=10,
                gpu_required=False,  # Can fall back to CPU
                estimated_time_ms=3000,
                can_parallelize=False,
            ),
            deterministic=True,
            idempotent=True,
            checkpoint_policy=CheckpointPolicy.ALWAYS,
        )

    def execute(self, inputs: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Execute SAM2 segmentation.

        Args:
            inputs: Must contain "linear_rgb" key.
            context: ExecutionContext with device, config, output_dir.

        Returns:
            Dict with masks, scores, num_masks.

        Raises:
            ValueError: If linear_rgb missing.
            RuntimeError: If segmentation fails.
        """
        from transformation_portal.spatial_ai.segmentation.contracts import (
            SegmentationInput,
        )
        from transformation_portal.spatial_ai.segmentation.sam2_backend import (
            SAM2Backend,
        )

        linear_rgb = inputs.get("linear_rgb")
        if linear_rgb is None:
            raise ValueError("SegmentationStage requires 'linear_rgb' input")

        # Validate input
        if not isinstance(linear_rgb, np.ndarray):
            raise ValueError(f"linear_rgb must be numpy array, got {type(linear_rgb)}")
        if linear_rgb.ndim != 3 or linear_rgb.shape[2] != 3:
            raise ValueError(f"linear_rgb must be (H, W, 3), got {linear_rgb.shape}")

        # Get config from context
        config = context.config if hasattr(context, "config") else {}
        device = getattr(context, "device", "cpu")
        model_size = config.get("model_size", self._model_size)
        enable_material = config.get(
            "enable_material_classification",
            self._enable_material_classification,
        )

        # Create segmentation input
        seg_input = SegmentationInput(
            image=linear_rgb,
            gamma=1.0,
            mode="auto",
        )

        # Initialize backend and segment
        backend = SAM2Backend(
            model_size=model_size,
            device=device,
            enable_material_classification=enable_material,
        )

        result = backend.segment(seg_input)

        # Extract cache-safe metadata arrays from SegmentationResult
        # These are used by pipeline.py to reconstruct MaskMetadata objects
        areas = np.array([m.area for m in result.metadata], dtype=np.int64)
        bboxes = np.array([m.bbox for m in result.metadata], dtype=np.int64)
        stabilities = np.array([m.stability_score for m in result.metadata], dtype=np.float32)

        return {
            "masks": result.masks,
            "scores": result.scores,
            "num_masks": len(result.masks),
            # Cache-safe metadata arrays (no object types)
            "metadata.area": areas,
            "metadata.bbox": bboxes,
            "metadata.stability_score": stabilities,
        }

    def compute_cache_key(self, inputs: Dict[str, Any], context: Any) -> str:
        """Compute content-addressed cache key.

        Key components:
        - Stage version
        - Input RGB content hash
        - Model size
        - Material classification flag

        Args:
            inputs: Stage inputs.
            context: Execution context.

        Returns:
            64-character SHA256 hex string.
        """
        linear_rgb = inputs.get("linear_rgb")
        if linear_rgb is None:
            return hashlib.sha256(b"segment:no_input").hexdigest()

        # Hash RGB content
        rgb_hash = hashlib.sha256(linear_rgb.tobytes()).hexdigest()[:16]

        # Config components
        config = context.config if hasattr(context, "config") else {}
        model_size = config.get("model_size", self._model_size)
        enable_material = config.get(
            "enable_material_classification",
            self._enable_material_classification,
        )

        # Combine components
        components = [
            f"v{self.metadata.version}",
            f"rgb:{rgb_hash}",
            f"model:{model_size}",
            f"material:{enable_material}",
        ]
        combined = "|".join(components)
        return hashlib.sha256(combined.encode()).hexdigest()


class MaterialsStage:
    """Stage adapter for MaterialBackend (Phase 2.2 materials).

    Generates PBR textures from segmented regions.

    Inputs:
        linear_rgb: (H, W, 3) float32 linear RGB array.
        masks: (N, H, W) bool array of segmentation masks.

    Outputs:
        pbr_textures: Dict mapping segment_id to PBRTextures contract.
        num_segments: Number of processed segments.

    Resource Profile:
        GPU: 2048 MB (depends on backend)
        CPU: 1024 MB
        Time: ~5000ms per segment
    """

    def __init__(
        self,
        backend: str = "heuristic",
        device: str = "cuda",
    ):
        """Initialize MaterialsStage.

        Args:
            backend: Material backend ("heuristic", "pbr_fusion", "nvdiffrec").
            device: Compute device ("cuda", "mps", "cpu").
        """
        self._backend = backend
        self._device = device

    @property
    def metadata(self) -> StageMetadata:
        """Stage metadata for MaterialsStage.

        Note: checkpoint_policy is set to NEVER because this stage outputs
        nested dicts with PBRTextures objects, which are not cache-serializable
        by ArtifactStore (.npz format). To enable caching, the stage would need
        to flatten PBR textures into separate numpy arrays.
        """
        # GPU memory depends on backend
        gpu_mb = 0 if self._backend == "heuristic" else MATERIALS_GPU_MB

        return StageMetadata(
            name="pbr_materials",
            version="2.2.0",
            description="PBR texture generation (Phase 2.2)",
            resource_requirements=ResourceRequirements(
                gpu_memory_mb=gpu_mb,
                cpu_memory_mb=1024,
                min_disk_mb=100,
                gpu_required=False,
                estimated_time_ms=5000,
                can_parallelize=True,  # Can process segments in parallel
            ),
            deterministic=True,
            idempotent=True,
            checkpoint_policy=CheckpointPolicy.NEVER,  # Outputs not cache-serializable
        )

    def execute(self, inputs: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Execute PBR texture generation.

        Args:
            inputs: Must contain "linear_rgb" and "masks" keys.
            context: ExecutionContext with device, config, output_dir.

        Returns:
            Dict with pbr_textures and num_segments.

        Raises:
            ValueError: If required inputs missing.
            RuntimeError: If generation fails.
        """
        from transformation_portal.spatial_ai.materials.contracts import (
            MaterialInput,
        )
        from transformation_portal.spatial_ai.materials.material_backend import (
            MaterialBackend,
        )

        linear_rgb = inputs.get("linear_rgb")
        masks = inputs.get("masks")

        if linear_rgb is None:
            raise ValueError("MaterialsStage requires 'linear_rgb' input")
        if masks is None:
            raise ValueError("MaterialsStage requires 'masks' input")

        # Get config from context
        config = context.config if hasattr(context, "config") else {}
        device = config.get("device", getattr(context, "device", self._device))
        backend_name = config.get("backend", self._backend)

        # Initialize backend
        backend = MaterialBackend(
            backend=backend_name,
            device=device,
        )

        # Process each mask segment
        pbr_textures = {}
        for i, mask in enumerate(masks):
            segment_id = f"segment_{i:03d}"

            # Create material input
            mat_input = MaterialInput(
                image=linear_rgb,
                mask=mask,
                depth=inputs.get("depth"),
                material_hint=inputs.get("material_hint"),
            )

            # Generate PBR textures
            pbr = backend.generate(mat_input)
            pbr_textures[segment_id] = pbr

        return {
            "pbr_textures": pbr_textures,
            "num_segments": len(pbr_textures),
        }

    def compute_cache_key(self, inputs: Dict[str, Any], context: Any) -> str:
        """Compute content-addressed cache key.

        Key components:
        - Stage version
        - Input RGB content hash
        - Masks content hash
        - Backend type

        Args:
            inputs: Stage inputs.
            context: Execution context.

        Returns:
            64-character SHA256 hex string.
        """
        linear_rgb = inputs.get("linear_rgb")
        masks = inputs.get("masks")

        if linear_rgb is None or masks is None:
            return hashlib.sha256(b"materials:no_input").hexdigest()

        # Hash RGB and masks content
        rgb_hash = hashlib.sha256(linear_rgb.tobytes()).hexdigest()[:16]
        masks_hash = hashlib.sha256(masks.tobytes()).hexdigest()[:16]

        # Config components
        config = context.config if hasattr(context, "config") else {}
        backend_name = config.get("backend", self._backend)

        # Combine components
        components = [
            f"v{self.metadata.version}",
            f"rgb:{rgb_hash}",
            f"masks:{masks_hash}",
            f"backend:{backend_name}",
        ]
        combined = "|".join(components)
        return hashlib.sha256(combined.encode()).hexdigest()


def build_spatial_ai_graph(
    stages: Optional[list] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build ExecutionGraph for Spatial AI pipeline.

    Factory function that creates an ExecutionGraph with appropriate
    stage adapters based on configuration.

    Args:
        stages: List of stage names to include (default: ["ingest", "segment"]).
            Valid stages: "ingest", "segment", "materials"
        config: Optional configuration dict with stage-specific settings.

    Returns:
        ExecutionGraph configured for the requested stages.

    Example:
        >>> graph = build_spatial_ai_graph(
        ...     stages=["ingest", "segment", "materials"],
        ...     config={"model_size": "large", "backend": "heuristic"},
        ... )
        >>> plan = graph.plan()
        >>> print(f"Stages: {[s.stage_id for s in plan.stages]}")
    """
    from .execution_graph import ExecutionGraph

    if stages is None:
        stages = ["ingest", "segment"]
    if config is None:
        config = {}

    graph = ExecutionGraph()

    # Add ingest stage
    if "ingest" in stages:
        ingest_config = IngestStageConfig(
            strict_ingest=config.get("strict_ingest", False),
            emit_exr=config.get("emit_exr", False),
            emit_provenance=config.get("emit_provenance", False),
        )
        graph.add_stage(
            "ingest",
            IngestStage(config=ingest_config),
            inputs={},
        )

    # Add segmentation stage
    if "segment" in stages or "segmentation" in stages:
        if "ingest" not in stages:
            raise ValueError("Segmentation requires ingest stage")

        graph.add_stage(
            "segment",
            SegmentationStage(
                model_size=config.get("model_size", "large"),
                enable_material_classification=config.get("enable_material_classification", False),
            ),
            inputs={"linear_rgb": "ingest.linear_rgb"},
        )

    # Add materials stage
    if "materials" in stages:
        if "segment" not in stages and "segmentation" not in stages:
            raise ValueError("Materials requires segmentation stage")

        graph.add_stage(
            "materials",
            MaterialsStage(
                backend=config.get("backend", "heuristic"),
                device=config.get("device", "cuda"),
            ),
            inputs={
                "linear_rgb": "ingest.linear_rgb",
                "masks": "segment.masks",
            },
        )

    return graph
