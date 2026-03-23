"""Main pipeline orchestrator for Spatial AI (Phase 2.4).

Ties together all spatial_ai phases into a cohesive E2E pipeline:
- Phase 1: Linear ingest
- Phase 2.1: SAM2 segmentation
- Phase 2.2: PBR materials
- Phase 2.3: 3D reconstruction

Key features:
- Stage composition (configurable pipeline)
- Resource management (GPU memory, model lifecycle)
- Error recovery (retry, CPU fallback, graceful degradation)
- Progress tracking
- Provenance logging

Architecture (ADR-027, ADR-028):
- Contract validation at phase boundaries
- Tier enforcement (3DGS research license)
- OpenEXR preflight for strict_ingest
- Deterministic outputs (same input → same output)

Example:
    >>> pipeline = SpatialAIPipeline.from_preset("spatial_ai_standard")
    >>> result = pipeline.process(
    ...     input_path="scene.tiff",
    ...     output_dir="output/"
    ... )
    >>> print(f"Stages: {', '.join(result.stages_completed)}")
    >>> print(f"Time: {result.execution_time:.1f}s, Memory: {result.peak_memory_mb:.1f}MB")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

from transformation_portal.spatial_ai.ingest.linear_decoder import LinearDecoder, LinearIngestResult
from transformation_portal.spatial_ai.materials.contracts import MaterialInput, PBRTextures
from transformation_portal.spatial_ai.materials.material_backend import MaterialBackend
from transformation_portal.spatial_ai.reconstruction.contracts import Scene3D
from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput, SegmentationResult
from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend
from transformation_portal.spatial_ai.segmentation.tiling.config import SegmentationTilingConfig

from .error_handler import ErrorHandler, ErrorRecoveryStrategy, PipelineError
from .progress_tracker import ProgressTracker
from .resource_manager import ResourceLimits, ResourceManager

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for Spatial AI pipeline.

    Attributes:
        tier: Tier level (standard, apex_research, experimental).
        stages: Stages to execute (["ingest", "segment", "materials", "reconstruct"]).
        ingest: Ingest configuration.
        segmentation: Segmentation configuration.
        materials: Materials configuration.
        reconstruction: Reconstruction configuration.
        resource_limits: Resource limits for execution.
        error_strategy: Error recovery strategy.
        use_execution_graph: If True, use ADR-029 graph-based execution (default: False).
    """

    tier: str
    stages: List[str] = field(default_factory=lambda: ["ingest", "segment"])
    ingest: Dict[str, Any] = field(default_factory=dict)
    segmentation: Dict[str, Any] = field(default_factory=dict)
    materials: Dict[str, Any] = field(default_factory=dict)
    reconstruction: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Optional[ResourceLimits] = None
    error_strategy: ErrorRecoveryStrategy = ErrorRecoveryStrategy.RETRY
    use_execution_graph: bool = False

    def __post_init__(self):
        """Validate configuration."""
        VALID_STAGES = ["ingest", "segment", "segmentation", "materials", "reconstruction"]
        for stage in self.stages:
            if stage not in VALID_STAGES:
                raise ValueError(f"Invalid stage '{stage}'. Valid: {VALID_STAGES}")

        # Tier validation
        VALID_TIERS = ["standard", "apex_research", "apex_research_ultra", "experimental"]
        if self.tier not in VALID_TIERS:
            raise ValueError(f"Invalid tier '{self.tier}'. Valid: {VALID_TIERS}")

        # Reconstruction requires research tier
        if "reconstruction" in self.stages and self.tier not in [
            "apex_research",
            "apex_research_ultra",
            "experimental",
        ]:
            raise ValueError(f"Reconstruction requires research tier, got '{self.tier}' " "(Inria 3DGS license restriction)")


@dataclass
class PipelineResult:
    """Result from end-to-end pipeline execution.

    Attributes:
        input_path: Input file path.
        output_dir: Output directory.
        stages_completed: List of completed stages.
        linear_image: Linear ingest result (if ingest stage run).
        segmentation: Segmentation result (if segment stage run).
        materials: PBR textures per segment (if materials stage run).
        scene_3d: 3D scene reconstruction (if reconstruct stage run).
        execution_time: Total execution time in seconds.
        peak_memory_mb: Peak GPU memory usage in MB.
        errors: List of error messages.
        warnings: List of warning messages.
        metadata: Additional metadata.
    """

    input_path: Path
    output_dir: Path
    stages_completed: List[str]

    linear_image: Optional[LinearIngestResult] = None
    segmentation: Optional[SegmentationResult] = None
    materials: Optional[Dict[str, PBRTextures]] = None
    scene_3d: Optional[Scene3D] = None

    execution_time: float = 0.0
    peak_memory_mb: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save_summary(self, path: Path) -> None:
        """Save execution summary as JSON.

        Args:
            path: Output path for summary JSON.
        """
        summary = {
            "input": str(self.input_path),
            "output_dir": str(self.output_dir),
            "stages_completed": self.stages_completed,
            "execution_time": self.execution_time,
            "peak_memory_mb": self.peak_memory_mb,
            "errors": self.errors,
            "warnings": self.warnings,
            "results": {
                "linear_image": self.linear_image is not None,
                "segmentation": {
                    "completed": self.segmentation is not None,
                    "num_masks": len(self.segmentation.masks) if self.segmentation else 0,
                },
                "materials": {
                    "completed": self.materials is not None,
                    "num_segments": len(self.materials) if self.materials else 0,
                },
                "scene_3d": {
                    "completed": self.scene_3d is not None,
                    "num_gaussians": self.scene_3d.splats.num_gaussians if self.scene_3d else 0,
                    "rmse": self.scene_3d.rmse if self.scene_3d else None,
                },
            },
            "metadata": self.metadata,
        }

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved pipeline summary: {path}")


@dataclass
class MultiViewReconstructionResult:
    """Result from multi-view reconstruction pipeline.

    Attributes:
        scene: Reconstructed 3D scene (Scene3D).
        ply_path: Path to exported PLY file.
        sidecar_path: Path to provenance JSON sidecar.
        output_dir: Output directory.
        execution_time: Total execution time in seconds.
        peak_memory_mb: Peak GPU memory usage in MB.
        stages_completed: List of completed stages.
        request_metadata: Original request metadata for traceability.
        errors: List of error messages (if any).
        warnings: List of warning messages (if any).
    """

    scene: Scene3D
    ply_path: Path
    sidecar_path: Path
    output_dir: Path
    execution_time: float = 0.0
    peak_memory_mb: float = 0.0
    stages_completed: List[str] = field(default_factory=list)
    request_metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def save_summary(self, path: Path) -> None:
        """Save reconstruction summary as JSON.

        Args:
            path: Output path for summary JSON.
        """
        summary = {
            "output_dir": str(self.output_dir),
            "ply_path": str(self.ply_path),
            "sidecar_path": str(self.sidecar_path),
            "stages_completed": self.stages_completed,
            "execution_time": self.execution_time,
            "peak_memory_mb": self.peak_memory_mb,
            "errors": self.errors,
            "warnings": self.warnings,
            "scene": {
                "num_gaussians": self.scene.splats.num_gaussians,
                "rmse": self.scene.rmse,
                "convergence": self.scene.convergence,
                "quality_score": self.scene.quality_score,
                "iteration": self.scene.iteration,
            },
            "request_metadata": self.request_metadata,
        }

        with open(path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved reconstruction summary: {path}")


class SpatialAIPipeline:
    """End-to-end Spatial AI pipeline orchestrator.

    Coordinates execution across all spatial_ai phases with:
    - Resource management (GPU memory, model lifecycle)
    - Error recovery (retry, CPU fallback)
    - Progress tracking
    - Provenance logging

    Example:
        >>> # From preset
        >>> pipeline = SpatialAIPipeline.from_preset("spatial_ai_standard")
        >>> result = pipeline.process("scene.tiff", "output/")
        >>>
        >>> # Custom config
        >>> config = PipelineConfig(
        ...     tier="apex_research",
        ...     stages=["ingest", "segment", "materials"],
        ... )
        >>> pipeline = SpatialAIPipeline(config)
        >>> result = pipeline.process("scene.tiff", "output/")
    """

    def __init__(self, config: Union[PipelineConfig, Dict, str, Path]):
        """Initialize pipeline.

        Args:
            config: Pipeline configuration (PipelineConfig, dict, preset name, or YAML path).
        """
        if isinstance(config, PipelineConfig):
            self.config = config
        elif isinstance(config, dict):
            self.config = self._dict_to_config(config)
        elif isinstance(config, (str, Path)):
            # Try as preset name first, then as file path
            try:
                self.config = self._load_preset(str(config))
            except FileNotFoundError:
                config_path = Path(config)
                if config_path.exists():
                    self.config = self._load_config_file(config_path)
                else:
                    raise FileNotFoundError(f"Config not found as preset or file: {config}")
        else:
            raise TypeError(f"config must be PipelineConfig, dict, str, or Path, got {type(config)}")

        # Initialize components
        self.resource_manager = ResourceManager(self.config.resource_limits)
        self.error_handler = ErrorHandler(max_retries=3)
        self.progress_tracker = ProgressTracker(total_stages=len(self.config.stages))

        # Track stateful backends for sequence lifecycle reset (ADR-026 §2.3)
        self._stateful_backends: Dict[str, Any] = {}

        logger.info(f"Initialized pipeline: tier={self.config.tier}, stages={self.config.stages}")

    @classmethod
    def from_preset(cls, preset_name: str) -> SpatialAIPipeline:
        """Create pipeline from preset configuration.

        Args:
            preset_name: Preset name (e.g., "spatial_ai_standard", "spatial_ai_research").

        Returns:
            Configured pipeline.

        Raises:
            FileNotFoundError: If preset not found.
        """
        config = cls._load_preset(preset_name)
        return cls(config)

    def register_stateful_backend(self, name: str, backend: Any) -> None:
        """Register a stateful backend for sequence lifecycle management.

        Backends registered here will have ``reset_state(sequence_id)`` called
        at sequence boundaries (ADR-026 §2.3).

        Args:
            name: Human-readable backend name (e.g., "depth_ensemble").
            backend: Backend instance. Must expose a callable ``reset_state()``.
        """
        reset_fn = getattr(backend, "reset_state", None)
        if not callable(reset_fn):
            logger.warning(
                "Backend '%s' has no callable reset_state() method; " "skipping stateful registration.",
                name,
            )
            return
        self._stateful_backends[name] = backend
        logger.debug("Registered stateful backend: %s", name)

    def reset_sequence(self, sequence_id: Optional[str] = None) -> None:
        """Reset all stateful backends for a new sequence.

        Must be called between unrelated video sequences or scene switches
        to prevent temporal bleed (ADR-026 §2.3).

        Args:
            sequence_id: Optional identifier for the new sequence.
        """
        for name, backend in self._stateful_backends.items():
            try:
                backend.reset_state(sequence_id)
                logger.debug(
                    "Reset stateful backend '%s' for sequence '%s'",
                    name,
                    sequence_id,
                )
            except Exception as exc:
                logger.error(
                    "Failed to reset backend '%s': %s",
                    name,
                    exc,
                    exc_info=True,
                )
        logger.info(
            "Sequence reset complete: %d backends reset (sequence_id=%s)",
            len(self._stateful_backends),
            sequence_id,
        )

    def process(
        self,
        input_path: Union[str, Path],
        output_dir: Union[str, Path],
        save_intermediates: bool = True,
        sequence_id: Optional[str] = None,
    ) -> PipelineResult:
        """Execute end-to-end pipeline.

        Args:
            input_path: Input image path.
            output_dir: Output directory for artifacts.
            save_intermediates: Save intermediate outputs (linear EXR, masks, etc.).
            sequence_id: Optional sequence identifier.  When provided,
                ``reset_sequence(sequence_id)`` is called before processing
                to clear any accumulated temporal state (ADR-026 §2.3).

        Returns:
            PipelineResult with all outputs and metadata.

        Raises:
            PipelineError: If pipeline fails and error_strategy is FAIL_FAST.
        """
        input_path = Path(input_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # ADR-026 §2.3: Reset stateful backends at sequence boundaries
        if sequence_id is not None:
            self.reset_sequence(sequence_id)

        logger.info(f"Starting pipeline: {input_path} -> {output_dir}")
        logger.info(f"Stages: {', '.join(self.config.stages)}")

        # Use graph-based execution if enabled (ADR-029)
        if self.config.use_execution_graph:
            return self._process_with_graph(input_path, output_dir, save_intermediates)

        # Initialize result
        result = PipelineResult(
            input_path=input_path,
            output_dir=output_dir,
            stages_completed=[],
        )

        # Start progress tracking
        self.progress_tracker.start_pipeline()

        try:
            with self.resource_manager:
                # Phase 1: Ingest
                if "ingest" in self.config.stages:
                    result.linear_image = self._run_ingest(input_path, output_dir, save_intermediates)
                    result.stages_completed.append("ingest")

                # Phase 2.1: Segmentation
                if "segment" in self.config.stages or "segmentation" in self.config.stages:
                    if result.linear_image is None:
                        raise PipelineError("segmentation", "Ingest stage required before segmentation")

                    result.segmentation = self._run_segmentation(result.linear_image, output_dir, save_intermediates)
                    result.stages_completed.append("segmentation")

                # Phase 2.2: Materials
                if "materials" in self.config.stages:
                    if result.linear_image is None or result.segmentation is None:
                        raise PipelineError("materials", "Ingest and segmentation stages required before materials")

                    result.materials = self._run_materials(
                        result.linear_image, result.segmentation, output_dir, save_intermediates
                    )
                    result.stages_completed.append("materials")

                # Phase 2.3: Reconstruction
                if "reconstruction" in self.config.stages:
                    if result.linear_image is None:
                        raise PipelineError("reconstruction", "Ingest stage required before reconstruction")

                    result.scene_3d = self._run_reconstruction(
                        result.linear_image, result.segmentation, output_dir, save_intermediates
                    )
                    result.stages_completed.append("reconstruction")

            # Complete
            result.execution_time = self.progress_tracker._get_elapsed_time()
            result.peak_memory_mb = self.resource_manager.get_peak_memory_mb()

            self.progress_tracker.complete_pipeline(success=True)
            logger.info(
                f"Pipeline completed: {len(result.stages_completed)} stages "
                f"in {result.execution_time:.1f}s, "
                f"peak memory {result.peak_memory_mb:.1f}MB"
            )

            # Save summary
            if save_intermediates:
                summary_path = output_dir / "pipeline_summary.json"
                result.save_summary(summary_path)

            return result

        except Exception as e:
            result.errors.append(str(e))
            result.execution_time = self.progress_tracker._get_elapsed_time()
            result.peak_memory_mb = self.resource_manager.get_peak_memory_mb()

            self.progress_tracker.complete_pipeline(success=False)
            logger.error(f"Pipeline failed: {e}")

            if self.config.error_strategy == ErrorRecoveryStrategy.RETURN_PARTIAL:
                logger.warning("Returning partial results due to error")
                return result
            else:
                raise

    def process_multiview(
        self,
        request: "MultiViewReconstructionRequest",  # noqa: F821
        output_dir: Union[str, Path],
        save_intermediates: bool = True,
    ) -> "MultiViewReconstructionResult":
        """Execute multi-view reconstruction pipeline.

        This is the dedicated multi-view entrypoint that does NOT overload
        the single-image ``process()`` contract. Use this for multi-view
        3D Gaussian Splatting reconstruction.

        Args:
            request: Multi-view reconstruction request with cameras and images.
                Must pass all contract validations (view count, camera policy, tier).
            output_dir: Output directory for reconstruction artifacts.
            save_intermediates: Save intermediate outputs.

        Returns:
            MultiViewReconstructionResult with Scene3D and export paths.

        Raises:
            PipelineError: If reconstruction fails.
            CameraValidationError: If camera validation fails.
            ValueError: If request contract is violated.

        Note:
            Reconstruction requires research tier (apex_research or higher)
            due to Inria 3DGS license. Camera validation is fail-closed
            by default (synthetic cameras rejected).

        Example:
            >>> from transformation_portal.core.geometry import (
            ...     CoreCameraParams,
            ...     MultiViewReconstructionRequest,
            ... )
            >>> cameras = [
            ...     CoreCameraParams(fx=800, fy=800, cx=512, cy=384,
            ...                      width=1024, height=768, source="explicit"),
            ...     CoreCameraParams(fx=800, fy=800, cx=512, cy=384,
            ...                      width=1024, height=768, source="explicit"),
            ... ]
            >>> request = MultiViewReconstructionRequest(
            ...     cameras=cameras,
            ...     image_paths=[Path("view1.png"), Path("view2.png")],
            ...     tier="apex_research",
            ... )
            >>> result = pipeline.process_multiview(request, output_dir="output/")
            >>> print(f"Exported: {result.ply_path}")
        """
        from transformation_portal.core.geometry import MultiViewReconstructionRequest
        from transformation_portal.spatial_ai.reconstruction import (
            PLYExportConfig,
            PLYExporter,
            SceneBuilder,
        )
        from transformation_portal.spatial_ai.reconstruction.contracts import (
            ReconstructionInput,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Validate request is properly typed
        if not isinstance(request, MultiViewReconstructionRequest):
            raise TypeError(
                f"Expected MultiViewReconstructionRequest, got {type(request).__name__}. "
                "Use process() for single-image pipeline."
            )

        logger.info(f"Starting multi-view reconstruction: {request.num_views} views -> {output_dir}")
        logger.info(f"Camera sources: {request.get_camera_source_summary()}, " f"tier={request.tier}")

        # Re-validate tier (defense-in-depth)
        if request.tier not in self.VALID_RECONSTRUCTION_TIERS:
            raise ValueError(
                f"Reconstruction requires research tier {self.VALID_RECONSTRUCTION_TIERS}, " f"got '{request.tier}'."
            )

        self.progress_tracker.start_pipeline()

        try:
            with self.resource_manager:
                # Convert CoreCameraParams to reconstruction CameraParams
                recon_cameras = self._convert_to_reconstruction_cameras(request)

                # Load images if paths provided
                images = self._load_multiview_images(request)

                # Validate ReconstructionInput (build but not stored)
                ReconstructionInput(
                    images=images,
                    gamma=request.gamma,
                    cameras=recon_cameras,
                    depth_maps=request.depth_maps,
                    masks=request.masks,
                    material_maps=request.material_maps,
                    tier=request.tier,
                )

                # Build scene via SceneBuilder/GaussianBackend
                self.progress_tracker.start_stage("reconstruction", "3D Reconstruction")

                builder = SceneBuilder(
                    tier=request.tier,
                    device=self.resource_manager.select_device(),
                    backend_config={
                        "optimization_seed": request.optimization_seed,
                    },
                )

                # Get iteration count from config or use default
                iterations = self.config.reconstruction.get("iterations", 2000)

                scene = builder.build_from_arrays(
                    images=images,
                    cameras=recon_cameras,
                    depth_maps=request.depth_maps,
                    masks=request.masks,
                    material_maps=request.material_maps,
                    iterations=iterations,
                    gamma=request.gamma,
                )

                self.progress_tracker.complete_stage("reconstruction", success=True)
                logger.info(
                    f"Reconstruction complete: {scene.splats.num_gaussians} Gaussians, "
                    f"RMSE={scene.rmse:.4f}, convergence={scene.convergence}"
                )

                # Export PLY
                self.progress_tracker.start_stage("export", "PLY Export")

                export_config = PLYExportConfig(
                    binary=self.config.reconstruction.get("export_binary", True),
                    include_attributes=self.config.reconstruction.get("export_include_attributes", True),
                )
                exporter = PLYExporter(export_config)

                ply_path = output_dir / "reconstruction.ply"
                exporter.export(
                    scene,
                    ply_path,
                    write_sidecar=True,
                    additional_metadata=request.to_metadata_dict(),
                )

                self.progress_tracker.complete_stage("export", success=True)

            # Build result
            execution_time = self.progress_tracker._get_elapsed_time()
            peak_memory = self.resource_manager.get_peak_memory_mb()

            result = MultiViewReconstructionResult(
                scene=scene,
                ply_path=ply_path,
                sidecar_path=ply_path.with_suffix(".provenance.json"),
                output_dir=output_dir,
                execution_time=execution_time,
                peak_memory_mb=peak_memory,
                stages_completed=["reconstruction", "export"],
                request_metadata=request.to_metadata_dict(),
            )

            self.progress_tracker.complete_pipeline(success=True)
            logger.info(f"Multi-view reconstruction completed in {execution_time:.1f}s, " f"peak memory {peak_memory:.1f}MB")

            # Save summary if requested
            if save_intermediates:
                summary_path = output_dir / "reconstruction_summary.json"
                result.save_summary(summary_path)

            return result

        except Exception as e:
            self.progress_tracker.complete_pipeline(success=False)
            logger.error(f"Multi-view reconstruction failed: {e}")
            raise PipelineError(
                "reconstruction",
                f"Multi-view reconstruction failed: {e}",
                original_error=e,
            ) from e

    def _convert_to_reconstruction_cameras(
        self,
        request: "MultiViewReconstructionRequest",  # noqa: F821
    ) -> list:
        """Convert CoreCameraParams to reconstruction CameraParams.

        Creates full reconstruction camera parameters with identity extrinsics
        from the simpler core camera intrinsics.
        """
        from transformation_portal.spatial_ai.reconstruction.contracts import CameraParams

        cameras = []
        for i, core_cam in enumerate(request.cameras):
            # Build intrinsics matrix
            intrinsics = np.array(
                [
                    [core_cam.fx, 0.0, core_cam.cx],
                    [0.0, core_cam.fy, core_cam.cy],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )

            # Default to identity extrinsics (camera at origin)
            # In production, these would come from camera pose estimation
            extrinsics = np.eye(4, dtype=np.float32)

            camera = CameraParams(
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                width=core_cam.width,
                height=core_cam.height,
                camera_id=f"view_{i:03d}",
            )
            cameras.append(camera)

        return cameras

    def _load_multiview_images(
        self,
        request: "MultiViewReconstructionRequest",  # noqa: F821
    ) -> list:
        """Load images from paths or return provided arrays."""
        if request.images is not None:
            return list(request.images)

        if request.image_paths is None:
            raise ValueError("Either image_paths or images must be provided")

        from PIL import Image

        images = []
        for path in request.image_paths:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Image not found: {path}")

            img = Image.open(path)
            img_array = np.array(img).astype(np.float32) / 255.0

            # Ensure RGB
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]  # Drop alpha

            images.append(img_array)

        return images

    # Valid tiers for reconstruction (research-only due to Inria 3DGS license)
    VALID_RECONSTRUCTION_TIERS = ("apex_research", "apex_research_ultra", "experimental")

    def _run_ingest(self, input_path: Path, output_dir: Path, save_intermediates: bool) -> LinearIngestResult:
        """Run linear ingest stage.

        Args:
            input_path: Input file path.
            output_dir: Output directory.
            save_intermediates: Save intermediate outputs.

        Returns:
            LinearIngestResult.
        """
        self.progress_tracker.start_stage("ingest", "Linear Ingest")

        try:
            # Parse config
            strict_ingest = self.config.ingest.get("strict_ingest", False)
            # For apex_research_ultra, EXR + provenance are contract artifacts,
            # not intermediates — always emit regardless of save_intermediates (C2).
            is_ultra = self.config.tier == "apex_research_ultra"
            emit_exr = self.config.ingest.get("emit_exr", False) and (save_intermediates or is_ultra)
            emit_provenance = self.config.ingest.get("emit_provenance", False) and (save_intermediates or is_ultra)

            # OpenEXR preflight if strict_ingest enabled
            if strict_ingest and emit_exr:
                try:
                    import OpenEXR  # noqa: F401
                except ImportError:
                    raise RuntimeError(
                        "strict_ingest=True with emit_exr=True requires OpenEXR. " "Install with: pip install OpenEXR Imath"
                    )

            # Execute
            decoder = LinearDecoder(gamma=1.0, bit_depth=32, strict_ingest=strict_ingest)

            def _decode():
                return decoder.decode(
                    input_path=input_path,
                    output_dir=output_dir,
                    emit_exr=emit_exr,
                    emit_provenance=emit_provenance,
                )

            # Map RETURN_PARTIAL to FAIL_FAST for stage execution
            # Pipeline level will catch and return partial results
            stage_strategy = (
                ErrorRecoveryStrategy.FAIL_FAST
                if self.config.error_strategy == ErrorRecoveryStrategy.RETURN_PARTIAL
                else self.config.error_strategy
            )

            result = self.error_handler.execute_with_retry(
                func=_decode,
                stage="ingest",
                strategy=stage_strategy,
                device="cpu",  # Ingest is CPU-only
            )

            self.progress_tracker.complete_stage("ingest", success=True)
            logger.info(
                f"Ingest completed: {result.input_size[1]}x{result.input_size[0]}, "
                f"range=[{result.linear_rgb.min():.3f}, {result.linear_rgb.max():.3f}]"
            )

            return result

        except Exception as e:
            self.progress_tracker.complete_stage("ingest", success=False, error_message=str(e))
            raise PipelineError("ingest", f"Linear ingest failed: {e}", original_error=e) from e

    def _run_segmentation(
        self, ingest_result: LinearIngestResult, output_dir: Path, save_intermediates: bool
    ) -> SegmentationResult:
        """Run segmentation stage.

        Args:
            ingest_result: Result from ingest stage.
            output_dir: Output directory.
            save_intermediates: Save intermediate outputs.

        Returns:
            SegmentationResult.
        """
        self.progress_tracker.start_stage("segment", "Segmentation")

        try:
            # Parse config
            backend_cfg = self.config.segmentation.get("backend", "sam2")
            if backend_cfg != "sam2":
                raise ValueError(f"Only sam2 backend supported, got '{backend_cfg}'")

            model_cfg = self.config.segmentation.get("model", {})
            model_size = model_cfg.get("size", "large")
            enable_material = bool(self.config.segmentation.get("material_classification", False))
            material_threshold = float(self.config.segmentation.get("material_confidence_threshold", 0.3))
            tiling_cfg = SegmentationTilingConfig.from_dict(self.config.segmentation.get("tiling"))

            # Select device
            device = self.resource_manager.select_device()

            # Create backend (no repo_id/revision - backend uses direct checkpoint loading)
            backend = SAM2Backend(
                model_size=model_size,
                device=device,
                enable_material_classification=enable_material,
                material_confidence_threshold=material_threshold,
                tiling=tiling_cfg,
            )

            # Register model for tracking
            self.resource_manager.register_model("sam2", backend)

            # Create input contract
            seg_input = SegmentationInput(
                image=ingest_result.linear_rgb,
                gamma=ingest_result.gamma,
                mode="auto",
            )

            # Execute
            def _segment():
                return backend.segment(seg_input)

            # Map RETURN_PARTIAL to FAIL_FAST for stage execution
            # Pipeline level will catch and return partial results
            stage_strategy = (
                ErrorRecoveryStrategy.FAIL_FAST
                if self.config.error_strategy == ErrorRecoveryStrategy.RETURN_PARTIAL
                else self.config.error_strategy
            )

            result = self.error_handler.execute_with_retry(
                func=_segment,
                stage="segment",
                strategy=stage_strategy,
                device=device,
            )

            # Save masks if requested
            if save_intermediates:
                masks_path = output_dir / "segmentation_masks.npz"
                np.savez_compressed(
                    masks_path,
                    masks=result.masks,
                    scores=result.scores,
                )
                logger.debug(f"Saved segmentation masks: {masks_path}")

            self.progress_tracker.complete_stage("segment", success=True)
            logger.info(
                f"Segmentation completed: {len(result.masks)} masks, "
                f"scores=[{result.scores.min():.3f}, {result.scores.max():.3f}]"
            )

            # Unload model to free memory
            self.resource_manager.unload_model("sam2")

            return result

        except Exception as e:
            self.progress_tracker.complete_stage("segment", success=False, error_message=str(e))
            raise PipelineError("segment", f"Segmentation failed: {e}", original_error=e) from e

    def _run_materials(
        self,
        ingest_result: LinearIngestResult,
        seg_result: SegmentationResult,
        output_dir: Path,
        save_intermediates: bool,
    ) -> Dict[str, PBRTextures]:
        """Run materials stage.

        Args:
            ingest_result: Result from ingest stage.
            seg_result: Result from segmentation stage.
            output_dir: Output directory.
            save_intermediates: Save intermediate outputs.

        Returns:
            Dict mapping segment IDs to PBR textures.
        """
        self.progress_tracker.start_stage("materials", "PBR Materials")

        try:
            # Parse config
            backend_cfg = self.config.materials.get("backend", "heuristic")
            material_hints = self.config.materials.get("material_hints", True)

            # Select device
            device = self.resource_manager.select_device()

            # Create backend
            backend = MaterialBackend(backend=backend_cfg, device=device)

            # Register model for resource tracking (C3: match segmentation lifecycle)
            self.resource_manager.register_model("materials", backend)

            # Generate materials for each segment
            materials = {}

            for i, (mask, metadata) in enumerate(zip(seg_result.masks, seg_result.metadata)):
                # Create material input
                material_hint = metadata.material_label if material_hints else None

                mat_input = MaterialInput(
                    image=ingest_result.linear_rgb,
                    gamma=ingest_result.gamma,
                    mask=mask,
                    material_hint=material_hint,
                )

                # Generate PBR textures
                def _generate():
                    return backend.generate(mat_input)

                pbr_textures = self.error_handler.execute_with_retry(
                    func=_generate,
                    stage="materials",
                    strategy=ErrorRecoveryStrategy.SKIP_STAGE,  # Skip failed segments
                    device=device,
                )

                if pbr_textures is not None:
                    materials[f"segment_{i}"] = pbr_textures

                # Update progress
                progress = ((i + 1) / len(seg_result.masks)) * 100.0
                self.progress_tracker.update_stage("materials", progress)

            # Save textures if requested
            if save_intermediates:
                textures_dir = output_dir / "materials"
                textures_dir.mkdir(exist_ok=True)

                for seg_id, pbr in materials.items():
                    seg_dir = textures_dir / seg_id
                    seg_dir.mkdir(exist_ok=True)

                    # Save each texture as numpy array
                    np.save(seg_dir / "albedo.npy", pbr.albedo)
                    np.save(seg_dir / "normal.npy", pbr.normal)
                    np.save(seg_dir / "roughness.npy", pbr.roughness)
                    np.save(seg_dir / "metallic.npy", pbr.metallic)
                    np.save(seg_dir / "ao.npy", pbr.ambient_occlusion)
                    if pbr.height is not None:
                        np.save(seg_dir / "height.npy", pbr.height)

                logger.debug(f"Saved PBR textures: {textures_dir}")

            self.progress_tracker.complete_stage("materials", success=True)
            logger.info(f"Materials completed: {len(materials)} segments")

            # Unload model to free memory (C3: match segmentation lifecycle)
            self.resource_manager.unload_model("materials")

            return materials

        except Exception as e:
            self.progress_tracker.complete_stage("materials", success=False, error_message=str(e))
            raise PipelineError("materials", f"Materials generation failed: {e}", original_error=e) from e

    def _run_reconstruction(
        self,
        ingest_result: LinearIngestResult,
        seg_result: Optional[SegmentationResult],
        output_dir: Path,
        save_intermediates: bool,
    ) -> Scene3D:
        """Run 3D reconstruction stage.

        Args:
            ingest_result: Result from ingest stage.
            seg_result: Optional segmentation result.
            output_dir: Output directory.
            save_intermediates: Save intermediate outputs.

        Returns:
            Scene3D.

        Note:
            This is a placeholder for single-view reconstruction.
            Full multi-view 3DGS requires camera poses and multiple views.
        """
        self.progress_tracker.start_stage("reconstruction", "3D Reconstruction")

        try:
            # For single-view, we can't do full 3DGS
            # This would require multi-view input or camera pose estimation
            raise NotImplementedError(
                "3D reconstruction requires multi-view input. "
                "Single-view reconstruction is not yet implemented. "
                "Use SceneBuilder directly with multiple views for 3DGS."
            )

        except Exception as e:
            self.progress_tracker.complete_stage("reconstruction", success=False, error_message=str(e))
            raise PipelineError("reconstruction", f"Reconstruction failed: {e}", original_error=e) from e

    def _process_with_graph(
        self,
        input_path: Path,
        output_dir: Path,
        save_intermediates: bool,
    ) -> PipelineResult:
        """Execute pipeline using ADR-029 execution graph abstraction.

        This method provides graph-based execution with:
        - Explicit DAG modeling of stage dependencies
        - Content-addressable caching
        - Automatic provenance tracking
        - Fail-fast resource validation

        Args:
            input_path: Input image path.
            output_dir: Output directory.
            save_intermediates: Save intermediate outputs.

        Returns:
            PipelineResult with all outputs and metadata.

        Note:
            This is the ADR-029 compliant execution path. Enable via
            ``use_execution_graph=True`` in PipelineConfig.
        """
        from .graph import ArtifactStore, Executor, build_spatial_ai_graph

        logger.info("Using ADR-029 graph-based execution")

        # Graph mode does not yet support reconstruction.
        # Fail explicitly rather than silently dropping the stage.
        if "reconstruction" in self.config.stages:
            raise PipelineError(
                "graph",
                "Reconstruction is not supported in graph mode (ADR-029). "
                "Either disable graph mode (use_execution_graph=False) or "
                "remove reconstruction from stages.",
            )

        # Build execution graph from config
        graph_stages = list(self.config.stages)

        # Build merged config for graph
        graph_config = {
            "strict_ingest": self.config.ingest.get("strict_ingest", False),
            "emit_exr": self.config.ingest.get("emit_exr", False),
            "emit_provenance": self.config.ingest.get("emit_provenance", False),
            "model_size": self.config.segmentation.get("model", {}).get("size", "large"),
            "enable_material_classification": bool(self.config.segmentation.get("material_classification", False)),
            "backend": self.config.materials.get("backend", "heuristic"),
            "device": self.resource_manager.select_device(),
        }

        graph = build_spatial_ai_graph(stages=graph_stages, config=graph_config)

        # Create artifact store for caching (optional)
        cache_dir = output_dir / ".cache" / "spatial_ai"
        artifact_store = ArtifactStore(cache_dir=cache_dir)

        # Create executor
        executor = Executor(
            artifact_store=artifact_store,
            resource_limits=self.config.resource_limits,
            device=graph_config["device"],
        )

        # Execute graph
        self.progress_tracker.start_pipeline()
        try:
            exec_result = executor.execute(
                graph=graph,
                inputs={"input_path": str(input_path)},
                output_dir=output_dir,
                config=graph_config,
            )

            # Convert ExecutionResult to PipelineResult
            result = PipelineResult(
                input_path=input_path,
                output_dir=output_dir,
                stages_completed=[sr.stage_id for sr in exec_result.stage_results],
                execution_time=exec_result.total_time_ms / 1000.0,
                peak_memory_mb=self.resource_manager.get_peak_memory_mb(),
            )

            # Extract outputs from graph results
            if "ingest.linear_rgb" in exec_result.outputs:
                from transformation_portal.spatial_ai.ingest.linear_decoder import (
                    LinearIngestResult,
                )

                # Reconstruct LinearIngestResult from graph outputs
                # IngestStage emits cache-safe primitives; we reconstruct the full result.
                linear_rgb = exec_result.outputs["ingest.linear_rgb"]
                input_size = exec_result.outputs.get("ingest.input_size", linear_rgb.shape[:2])

                # Required fields with sensible defaults derived from graph outputs
                result.linear_image = LinearIngestResult(
                    linear_rgb=linear_rgb,
                    input_path=input_path,
                    input_size=tuple(input_size) if not isinstance(input_size, tuple) else input_size,
                    gamma=1.0,
                    bit_depth=32,  # Always float32 for linear ingest
                    dtype="float32",
                    input_format=exec_result.outputs.get("ingest.input_format", "TIFF"),
                    color_space=exec_result.outputs.get("ingest.color_space", "linear_sRGB"),
                )

            if "segment.masks" in exec_result.outputs:
                from transformation_portal.spatial_ai.segmentation.contracts import (
                    MaskMetadata,
                    SegmentationResult,
                )

                masks = exec_result.outputs["segment.masks"]
                scores = exec_result.outputs.get("segment.scores", np.ones(len(masks)))

                # Optional cached metadata arrays from the segmentation stage.
                # These arrays are normally present (SegmentationStage emits them).
                # Fallback computation only executes for legacy/external graph outputs.
                areas = exec_result.outputs.get("segment.metadata.area")
                bboxes = exec_result.outputs.get("segment.metadata.bbox")
                stabilities = exec_result.outputs.get("segment.metadata.stability_score")

                def _compute_bbox(mask: np.ndarray) -> tuple:
                    """Compute a tight (x, y, w, h) bbox for a boolean/uint8 mask.

                    Falls back to (0, 0, 0, 0) for empty masks to preserve determinism.
                    Note: This fallback is rarely hit since SegmentationStage emits
                    pre-computed metadata arrays. Consider cv2.findNonZero if profiling
                    shows this path is a bottleneck.
                    """
                    ys, xs = np.where(mask)
                    if ys.size == 0 or xs.size == 0:
                        return (0, 0, 0, 0)
                    x_min, x_max = int(xs.min()), int(xs.max())
                    y_min, y_max = int(ys.min()), int(ys.max())
                    return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)

                metadata: List[MaskMetadata] = []
                num_masks = len(masks)

                for i in range(num_masks):
                    mask = masks[i]

                    if areas is not None and i < len(areas):
                        area = int(areas[i])
                    else:
                        # Fallback: derive area directly from the mask.
                        area = int(np.count_nonzero(mask))

                    if bboxes is not None and i < len(bboxes):
                        bbox = tuple(int(v) for v in bboxes[i])
                    else:
                        bbox = _compute_bbox(mask)

                    if stabilities is not None and i < len(stabilities):
                        stability_score = float(stabilities[i])
                    else:
                        # Conservative default when stability is not available.
                        stability_score = 0.5

                    # Ensure area is at least 1 to satisfy MaskMetadata validation
                    metadata.append(
                        MaskMetadata(
                            area=max(1, area),
                            bbox=bbox,
                            stability_score=stability_score,
                        )
                    )

                result.segmentation = SegmentationResult(
                    masks=masks,
                    scores=scores,
                    metadata=metadata,
                )

            if "materials.pbr_textures" in exec_result.outputs:
                result.materials = exec_result.outputs["materials.pbr_textures"]

            # Save summary
            if save_intermediates:
                summary_path = output_dir / "pipeline_summary.json"
                result.save_summary(summary_path)

                # Also save graph execution metadata
                graph_meta_path = output_dir / "graph_execution.json"
                import json

                graph_meta = {
                    "stages_executed": exec_result.stages_executed,
                    "stages_cached": exec_result.stages_cached,
                    "total_time_ms": exec_result.total_time_ms,
                    "stage_results": [
                        {
                            "stage_id": sr.stage_id,
                            "cache_hit": sr.cache_hit,
                            "execution_time_ms": sr.execution_time_ms,
                            "cache_key": sr.cache_key,
                        }
                        for sr in exec_result.stage_results
                    ],
                }
                with open(graph_meta_path, "w") as f:
                    json.dump(graph_meta, f, indent=2)

            self.progress_tracker.complete_pipeline(success=True)
            logger.info(
                f"Graph execution completed: {exec_result.stages_executed} executed, "
                f"{exec_result.stages_cached} cached, "
                f"{exec_result.total_time_ms:.1f}ms total"
            )

            return result

        except Exception as e:
            self.progress_tracker.complete_pipeline(success=False)
            logger.error(f"Graph execution failed: {e}")

            result = PipelineResult(
                input_path=input_path,
                output_dir=output_dir,
                stages_completed=[],
                errors=[str(e)],
            )

            if self.config.error_strategy == ErrorRecoveryStrategy.RETURN_PARTIAL:
                return result
            # Wrap in PipelineError to preserve API contract (docstring states PipelineError is raised)
            raise PipelineError("graph", f"Graph execution failed: {e}", original_error=e) from e

    @staticmethod
    def _load_preset(preset_name: str) -> PipelineConfig:
        """Load preset configuration.

        Args:
            preset_name: Preset name.

        Returns:
            PipelineConfig.

        Raises:
            FileNotFoundError: If preset not found.
        """
        # Look in config/presets/
        preset_paths = [
            Path(f"config/presets/{preset_name}.yaml"),
            Path(f"config/presets/spatial_ai/{preset_name}.yaml"),
        ]

        for preset_path in preset_paths:
            if preset_path.exists():
                return SpatialAIPipeline._load_config_file(preset_path)

        raise FileNotFoundError(f"Preset not found: {preset_name} (tried: {preset_paths})")

    @staticmethod
    def _load_config_file(path: Path) -> PipelineConfig:
        """Load configuration from YAML file.

        Args:
            path: Path to YAML config file.

        Returns:
            PipelineConfig.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        return SpatialAIPipeline._dict_to_config(data)

    @staticmethod
    def _dict_to_config(data: Dict) -> PipelineConfig:
        """Convert dict to PipelineConfig.

        Args:
            data: Configuration dict.

        Returns:
            PipelineConfig.
        """
        # Extract pipeline section
        pipeline_data = data.get("pipeline", {})

        # Parse resource limits if provided
        resource_limits = None
        if "resource_limits" in data:
            limits_data = data["resource_limits"]
            resource_limits = ResourceLimits(
                max_gpu_memory_gb=limits_data.get("max_gpu_memory_gb", 16.0),
                max_models_loaded=limits_data.get("max_models_loaded", 3),
                batch_size=limits_data.get("batch_size", 1),
                device_preference=limits_data.get("device_preference", ["cuda", "mps", "cpu"]),
            )

        # Parse error strategy if provided
        error_strategy = ErrorRecoveryStrategy.RETRY  # Default
        if "error_strategy" in data:
            strategy_str = data["error_strategy"]
            # Map string to enum
            strategy_map = {
                "retry": ErrorRecoveryStrategy.RETRY,
                "retry_cpu_fallback": ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK,
                "retry_with_cpu_fallback": ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK,
                "skip_stage": ErrorRecoveryStrategy.SKIP_STAGE,
                "fail_fast": ErrorRecoveryStrategy.FAIL_FAST,
                "return_partial": ErrorRecoveryStrategy.RETURN_PARTIAL,
            }
            if strategy_str in strategy_map:
                error_strategy = strategy_map[strategy_str]
            else:
                logger.warning(f"Unknown error strategy '{strategy_str}', using RETRY")

        # Normalize stage aliases: "segment" → "segmentation" (C1 correctness fix)
        segmentation_data = pipeline_data.get("segmentation") or pipeline_data.get("segment", {})
        stages = list(pipeline_data.keys())
        stages = ["segmentation" if s == "segment" else s for s in stages]

        # Parse use_execution_graph flag (ADR-029)
        use_execution_graph = data.get("use_execution_graph", False)

        # Build config
        config = PipelineConfig(
            tier=data.get("tier", "standard"),
            stages=stages,
            ingest=pipeline_data.get("ingest", {}),
            segmentation=segmentation_data,
            materials=pipeline_data.get("materials", {}),
            reconstruction=pipeline_data.get("reconstruction", {}),
            resource_limits=resource_limits,
            error_strategy=error_strategy,
            use_execution_graph=use_execution_graph,
        )

        return config
