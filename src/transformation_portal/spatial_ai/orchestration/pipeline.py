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

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union, cast

import numpy as np

from transformation_portal.compliance import (
    load_and_validate_preset,
    validate_materials_preset,
    validate_non_commercial_preset,
)
from transformation_portal.reporting.contracts import build_stage_report
from transformation_portal.spatial_ai.ingest.linear_decoder import LinearDecoder, LinearIngestResult
from transformation_portal.spatial_ai.materials.contracts import MaterialInput, PBRTextures
from transformation_portal.spatial_ai.materials.material_backend import MaterialBackend
from transformation_portal.spatial_ai.reconstruction.contracts import Scene3D
from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationInput, SegmentationResult
from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend
from transformation_portal.spatial_ai.segmentation.tiling.config import SegmentationTilingConfig

from .artifact_utils import _sanitize_json_value, _sha256_array
from .config import (
    PipelineConfig,
    _extract_materials_governance_overrides,
    _is_reload_safe_pipeline_config,
    _normalise_segmentation_cache_policy,
)
from .error_handler import ErrorHandler, ErrorRecoveryStrategy, PipelineError
from .json_io import write_json_atomic as _write_json_atomic
from .progress_tracker import ProgressTracker
from .resource_manager import ResourceLimits, ResourceManager
from .results import MultiViewReconstructionResult, PipelineResult
from .segmentation_cache import (
    _SEGMENTATION_CACHE_SCHEMA_VERSION,
    _build_segmentation_cache_key,
    _file_identity,
    _metadata_from_cache_dict,
    _metadata_to_cache_dict,
    _read_segmentation_cache,
    _segmentation_cache_paths,
    _segmentation_mask_count,
    _segmentation_result_checksum,
    _sha256_file,
    _sha256_file_cached,
    _write_segmentation_cache,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformation_portal.core.geometry import MultiViewReconstructionRequest


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
        elif _is_reload_safe_pipeline_config(config):
            self.config = cast(PipelineConfig, config)
        elif isinstance(config, dict):
            self._validate_runtime_config_dict(config)
            self.config = self._dict_to_config(config)
        elif isinstance(config, (str, Path)):
            # Try as preset name first, then as file path
            try:
                self.config = self._load_preset(str(config))
            except FileNotFoundError as exc:
                config_path = Path(config)
                if config_path.exists():
                    self.config = self._load_config_file(config_path)
                else:
                    raise FileNotFoundError(f"Config not found as preset or file: {config}") from exc
        else:
            raise TypeError(f"config must be PipelineConfig, dict, str, or Path, got {type(config)}")

        # Initialize components
        self.resource_manager = ResourceManager(self.config.resource_limits)
        self.error_handler = ErrorHandler(max_retries=3)
        self.progress_tracker = ProgressTracker(total_stages=len(self.config.stages))

        # Track stateful backends for sequence lifecycle reset (ADR-026 §2.3)
        self._stateful_backends: Dict[str, Any] = {}
        self._last_segmentation_stage_metadata: Dict[str, Any] = {}
        self._last_materials_stage_metadata: Dict[str, Any] = {}

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
                "Backend '%s' has no callable reset_state() method; skipping stateful registration.",
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

        if "reconstruction" in self.config.stages:
            raise PipelineError(
                "reconstruction",
                "Single-image pipeline does not support reconstruction. "
                "Use process_multiview() with a MultiViewReconstructionRequest.",
            )

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
                    result.stage_reports.append(build_stage_report(stage="ingest", status="completed"))

                # Phase 2.1: Segmentation
                if "segment" in self.config.stages or "segmentation" in self.config.stages:
                    if result.linear_image is None:
                        raise PipelineError("segmentation", "Ingest stage required before segmentation")

                    result.segmentation = self._run_segmentation(result.linear_image, output_dir, save_intermediates)
                    result.stages_completed.append("segmentation")
                    result.stage_reports.append(
                        build_stage_report(
                            stage="segmentation",
                            status="completed",
                            metadata=self._last_segmentation_stage_metadata,
                        )
                    )

                # Phase 2.2: Materials
                if "materials" in self.config.stages:
                    if result.linear_image is None or result.segmentation is None:
                        raise PipelineError("materials", "Ingest and segmentation stages required before materials")

                    result.materials = self._run_materials(
                        result.linear_image, result.segmentation, output_dir, save_intermediates
                    )
                    result.stages_completed.append("materials")
                    result.stage_reports.append(
                        build_stage_report(
                            stage="materials",
                            status="completed",
                            metadata=self._last_materials_stage_metadata,
                        )
                    )

                # Phase 2.3: Reconstruction
                if "reconstruction" in self.config.stages:
                    if result.linear_image is None:
                        raise PipelineError("reconstruction", "Ingest stage required before reconstruction")

                    result.scene_3d = self._run_reconstruction(
                        result.linear_image,
                        result.segmentation,
                        output_dir,
                        save_intermediates,
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
            raise ValueError(f"Reconstruction requires research tier {self.VALID_RECONSTRUCTION_TIERS}, got '{request.tier}'.")

        # Execution-graph mode is not supported for multi-view reconstruction.
        # Fail fast rather than silently running the imperative path when the flag is enabled.
        if getattr(self.config, "use_execution_graph", False):
            raise PipelineError(
                "reconstruction",
                "Multi-view reconstruction does not support execution-graph mode "
                "(use_execution_graph=True). Disable execution graph or use process() "
                "for single-view pipelines.",
            )

        # Multi-view reconstruction always runs 2 stages: reconstruction + export.
        # Update total_stages to ensure progress tracking doesn't exceed 100%.
        self.progress_tracker.total_stages = 2

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
                stage_reports=[
                    build_stage_report(stage="reconstruction", status="completed"),
                    build_stage_report(stage="export", status="completed"),
                ],
            )

            self.progress_tracker.complete_pipeline(success=True)
            logger.info("Multi-view reconstruction completed in %.1fs, peak memory %.1fMB", execution_time, peak_memory)

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

            with Image.open(path) as img:
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
                except ImportError as exc:
                    raise RuntimeError(
                        "strict_ingest=True with emit_exr=True requires OpenEXR. Install with: pip install OpenEXR Imath"
                    ) from exc

            # Execute
            decoder = LinearDecoder(gamma=1.0, bit_depth=32, strict_ingest=strict_ingest)

            def _decode() -> LinearIngestResult:
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
        self._last_segmentation_stage_metadata = {}
        timing_ms: dict[str, float] = {}
        t_stage = time.perf_counter()

        try:
            # Parse config
            backend_cfg = self.config.segmentation.get("backend", "sam2")
            if backend_cfg != "sam2":
                raise ValueError(f"Only sam2 backend supported, got '{backend_cfg}'")

            model_cfg = self.config.segmentation.get("model", {})
            model_size = model_cfg.get("size", "large")
            repo_id = model_cfg.get("repo_id")
            revision = model_cfg.get("revision")
            checkpoint_path = model_cfg.get("checkpoint_path")
            prefer_hf_pipeline = bool(model_cfg.get("prefer_hf_pipeline", False))
            generator_kwargs = dict(self.config.segmentation.get("generator", {}))
            enable_material = bool(self.config.segmentation.get("material_classification", False))
            material_threshold = float(self.config.segmentation.get("material_confidence_threshold", 0.3))
            tiling_cfg = SegmentationTilingConfig.from_dict(self.config.segmentation.get("tiling"))
            cache_policy = _normalise_segmentation_cache_policy(self.config.segmentation.get("cache_policy", "read_write"))

            active_device: Dict[str, Literal["cuda", "mps", "cpu"]] = {
                "value": cast(Literal["cuda", "mps", "cpu"], self.resource_manager.select_device())
            }
            backend_holder: Dict[str, SAM2Backend] = {}

            def _build_backend(exec_device: Literal["cuda", "mps", "cpu"], *, replace_tracking: bool = False) -> SAM2Backend:
                active_device["value"] = exec_device
                if replace_tracking:
                    old_backend = backend_holder.pop("backend", None)
                    if old_backend is not None:
                        teardown = getattr(old_backend, "unload_model", None) or getattr(old_backend, "unload", None)
                        if callable(teardown):
                            try:
                                teardown()
                            except Exception as cleanup_exc:
                                logger.debug("SAM2 backend teardown failed during CPU fallback rebuild: %s", cleanup_exc)
                    try:
                        self.resource_manager.unload_model("sam2")
                    except Exception as cleanup_exc:
                        logger.debug("SAM2 ResourceManager cleanup failed during CPU fallback rebuild: %s", cleanup_exc)

                backend = SAM2Backend(
                    model_size=model_size,
                    device=exec_device,
                    checkpoint_path=checkpoint_path,
                    repo_id=repo_id,
                    revision=revision,
                    prefer_hf_pipeline=prefer_hf_pipeline,
                    generator_kwargs=generator_kwargs,
                    enable_material_classification=enable_material,
                    material_confidence_threshold=material_threshold,
                    tiling=tiling_cfg,
                )
                backend_holder["backend"] = backend
                self.resource_manager.register_model("sam2", backend)
                return backend

            # Create input contract
            seg_input = SegmentationInput(
                image=ingest_result.linear_rgb,
                gamma=ingest_result.gamma,
                mode="auto",
            )

            cache_key: Optional[str] = None
            cache_payload: Dict[str, Any] = {}
            cache_key_device: Optional[Literal["cuda", "mps", "cpu"]] = None
            if cache_policy == "read_write":
                t_cache = time.perf_counter()
                cache_key_device = active_device["value"]
                cache_key, cache_payload = _build_segmentation_cache_key(
                    image=ingest_result.linear_rgb,
                    segmentation_cfg=self.config.segmentation,
                    device=cache_key_device,
                )
                cached_result = _read_segmentation_cache(
                    cache_dir=output_dir / ".cache" / "spatial_ai" / "segmentation",
                    cache_key=cache_key,
                    key_payload=cache_payload,
                )
                timing_ms["cache_lookup"] = round((time.perf_counter() - t_cache) * 1000.0, 3)
                if cached_result is not None:
                    self.progress_tracker.complete_stage("segment", success=True)
                    timing_ms["total"] = round((time.perf_counter() - t_stage) * 1000.0, 3)
                    self._last_segmentation_stage_metadata = {
                        "cache_hit": True,
                        "cache_key": cache_key,
                        "cache_policy": cache_policy,
                        "timing_ms": timing_ms,
                        "mask_count": _segmentation_mask_count(cached_result),
                        "backend": backend_cfg,
                        "device": active_device["value"],
                        "model_size": model_size,
                    }
                    logger.info("Segmentation cache hit: %s", cache_key)
                    return cached_result

            _build_backend(active_device["value"])

            # Execute
            def _segment() -> SegmentationResult:
                return backend_holder["backend"].segment(seg_input)

            def _on_device_change(new_device: str, attempt: int, exc: Exception) -> None:
                rebuilt_device = cast(Literal["cuda", "mps", "cpu"], new_device)
                logger.warning(
                    "Rebuilding SAM2 backend on %s after attempt %d failed: %s",
                    rebuilt_device,
                    attempt,
                    exc,
                )
                _build_backend(rebuilt_device, replace_tracking=True)

            # Map RETURN_PARTIAL to FAIL_FAST for stage execution
            # Pipeline level will catch and return partial results
            stage_strategy = (
                ErrorRecoveryStrategy.FAIL_FAST
                if self.config.error_strategy == ErrorRecoveryStrategy.RETURN_PARTIAL
                else self.config.error_strategy
            )

            t_segment = time.perf_counter()
            result = self.error_handler.execute_with_retry(
                func=_segment,
                stage="segment",
                strategy=stage_strategy,
                device=active_device["value"],
                on_device_change=_on_device_change,
            )
            timing_ms["backend_segment"] = round((time.perf_counter() - t_segment) * 1000.0, 3)
            classifier = getattr(backend_holder["backend"], "_material_classifier", None)
            classifier_timing = getattr(classifier, "_last_timing_ms", None)
            if isinstance(classifier_timing, dict) and classifier_timing:
                clip_timing = {
                    str(key): float(value) for key, value in classifier_timing.items() if isinstance(value, (int, float))
                }
                timing_ms["clip_classification"] = round(
                    sum(value for key, value in clip_timing.items() if key != "batch_size"),
                    3,
                )
            else:
                clip_timing = {}

            if cache_policy == "read_write" and cache_key:
                if cache_key_device != active_device["value"]:
                    t_cache_rekey = time.perf_counter()
                    cache_key, cache_payload = _build_segmentation_cache_key(
                        image=ingest_result.linear_rgb,
                        segmentation_cfg=self.config.segmentation,
                        device=active_device["value"],
                    )
                    cache_key_device = active_device["value"]
                    timing_ms["cache_rekey"] = round((time.perf_counter() - t_cache_rekey) * 1000.0, 3)

                t_cache_write = time.perf_counter()
                try:
                    _write_segmentation_cache(
                        cache_dir=output_dir / ".cache" / "spatial_ai" / "segmentation",
                        cache_key=cache_key,
                        key_payload=cache_payload,
                        result=result,
                    )
                except Exception as exc:
                    logger.debug("Spatial segmentation cache write failed: %s", exc)
                timing_ms["cache_write"] = round((time.perf_counter() - t_cache_write) * 1000.0, 3)

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
            timing_ms["total"] = round((time.perf_counter() - t_stage) * 1000.0, 3)
            self._last_segmentation_stage_metadata = {
                "cache_hit": False,
                "cache_key": cache_key,
                "cache_policy": cache_policy,
                "timing_ms": timing_ms,
                "mask_count": _segmentation_mask_count(result),
                "backend": backend_cfg,
                "device": active_device["value"],
                "model_size": model_size,
            }
            if clip_timing:
                self._last_segmentation_stage_metadata["clip_classification"] = {"timing_ms": clip_timing}
            scores = np.asarray(getattr(result, "scores", []), dtype=np.float32)
            if scores.size:
                score_summary = f"[{scores.min():.3f}, {scores.max():.3f}]"
            else:
                score_summary = "empty"
            logger.info(f"Segmentation completed: {_segmentation_mask_count(result)} masks, " f"scores={score_summary}")

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
        self._last_materials_stage_metadata = {}
        timing_ms: dict[str, float] = {}
        t_stage = time.perf_counter()

        try:
            # Parse config
            materials_cfg = dict(self.config.materials)
            backend_cfg = materials_cfg.get("backend", "heuristic")
            material_hints = materials_cfg.get("material_hints", True)
            model_repo_id = materials_cfg.get("model_repo_id")
            model_revision = materials_cfg.get("model_revision")

            active_device: Dict[str, Literal["cuda", "mps", "cpu"]] = {
                "value": cast(Literal["cuda", "mps", "cpu"], self.resource_manager.select_device())
            }
            backend_holder: Dict[str, MaterialBackend] = {}

            def _build_backend(
                exec_device: Literal["cuda", "mps", "cpu"], *, replace_tracking: bool = False
            ) -> MaterialBackend:
                active_device["value"] = exec_device
                if replace_tracking:
                    old_backend = backend_holder.pop("backend", None)
                    if old_backend is not None:
                        teardown = getattr(old_backend, "unload_model", None) or getattr(old_backend, "unload", None)
                        if callable(teardown):
                            try:
                                teardown()
                            except Exception as cleanup_exc:
                                logger.debug("Material backend teardown failed during CPU fallback rebuild: %s", cleanup_exc)
                    try:
                        self.resource_manager.unload_model("materials")
                    except Exception as cleanup_exc:
                        logger.debug("Material ResourceManager cleanup failed during CPU fallback rebuild: %s", cleanup_exc)

                generation_overrides = {
                    "backend": backend_cfg,
                    "device": exec_device,
                    "resolution": materials_cfg.get("resolution"),
                    "optimize_iterations": materials_cfg.get("optimize_iterations"),
                    "use_depth": materials_cfg.get("use_depth"),
                    "normal_strength": materials_cfg.get("normal_strength"),
                    "ao_intensity": materials_cfg.get("ao_intensity"),
                    "strict_backend": materials_cfg.get("strict_backend"),
                }

                backend = MaterialBackend(
                    backend=backend_cfg,
                    device=exec_device,
                    model_repo_id=model_repo_id,
                    model_revision=model_revision,
                    generation_config_overrides=generation_overrides,
                )
                backend_holder["backend"] = backend
                self.resource_manager.register_model("materials", backend)
                return backend

            _build_backend(active_device["value"])

            depth_map = None
            if materials_cfg.get("use_depth", False):
                depth_required = bool(materials_cfg.get("require_depth", False))
                depth_map = getattr(ingest_result, "depth", None)
                if depth_map is None:
                    depth_message = "Material generation requested use_depth=True, but no depth map is available from ingest"
                    if depth_required:
                        raise RuntimeError(f"{depth_message} (require_depth=True, tier={self.config.tier!r})")
                    logger.warning(
                        "%s; continuing without depth (tier=%s, require_depth=%s)",
                        depth_message,
                        self.config.tier,
                        depth_required,
                    )

            # Generate materials for each segment
            materials = {}
            material_artifact_entries: list[tuple[str, int, PBRTextures]] = []

            if self.config.error_strategy in {
                ErrorRecoveryStrategy.FAIL_FAST,
                ErrorRecoveryStrategy.RETURN_PARTIAL,
            }:
                per_segment_strategy = ErrorRecoveryStrategy.FAIL_FAST
            elif self.config.error_strategy == ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK:
                per_segment_strategy = ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK
            else:
                per_segment_strategy = ErrorRecoveryStrategy.SKIP_STAGE

            for i, (mask, metadata) in enumerate(zip(seg_result.masks, seg_result.metadata)):
                # Create material input
                material_hint = metadata.material_label if material_hints else None

                mat_input = MaterialInput(
                    image=ingest_result.linear_rgb,
                    gamma=ingest_result.gamma,
                    mask=mask,
                    depth=depth_map,
                    material_hint=material_hint,
                )

                # Generate PBR textures
                def _generate(material_input: MaterialInput = mat_input) -> PBRTextures:
                    return backend_holder["backend"].generate(material_input)

                def _on_device_change(new_device: str, attempt: int, exc: Exception) -> None:
                    rebuilt_device = cast(Literal["cuda", "mps", "cpu"], new_device)
                    logger.warning(
                        "Rebuilding material backend on %s after attempt %d failed: %s",
                        rebuilt_device,
                        attempt,
                        exc,
                    )
                    _build_backend(rebuilt_device, replace_tracking=True)

                try:
                    t_generate = time.perf_counter()
                    pbr_textures = self.error_handler.execute_with_retry(
                        func=_generate,
                        stage="materials",
                        strategy=per_segment_strategy,
                        device=active_device["value"],
                        on_device_change=_on_device_change,
                    )
                    timing_ms[f"segment_{i}"] = round((time.perf_counter() - t_generate) * 1000.0, 3)
                except PipelineError:
                    if per_segment_strategy == ErrorRecoveryStrategy.FAIL_FAST:
                        raise
                    logger.warning("Material generation failed for segment_%d; skipping", i)
                    pbr_textures = None

                if pbr_textures is not None:
                    seg_id = f"segment_{i}"
                    materials[seg_id] = pbr_textures
                    material_artifact_entries.append((seg_id, i, pbr_textures))

                # Update progress
                progress = ((i + 1) / len(seg_result.masks)) * 100.0
                self.progress_tracker.update_stage("materials", progress)

            # Save textures if requested
            if save_intermediates:
                textures_dir = output_dir / "materials"
                textures_dir.mkdir(exist_ok=True)

                for seg_id, segment_index, pbr in material_artifact_entries:
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
                    self._save_material_artifacts(
                        seg_dir=seg_dir,
                        seg_id=seg_id,
                        segment_index=segment_index,
                        pbr=pbr,
                        mask=seg_result.masks[segment_index],
                        segment_metadata=seg_result.metadata[segment_index],
                        ingest_result=ingest_result,
                        backend_cfg=backend_cfg,
                    )

                logger.debug(f"Saved PBR textures: {textures_dir}")

            self.progress_tracker.complete_stage("materials", success=True)
            timing_ms["total"] = round((time.perf_counter() - t_stage) * 1000.0, 3)
            self._last_materials_stage_metadata = {
                "timing_ms": timing_ms,
                "segment_count": len(materials),
                "backend": backend_cfg,
                "device": active_device["value"],
            }
            logger.info(f"Materials completed: {len(materials)} segments")

            # Unload model to free memory (C3: match segmentation lifecycle)
            self.resource_manager.unload_model("materials")

            return materials

        except Exception as e:
            self.progress_tracker.complete_stage("materials", success=False, error_message=str(e))
            raise PipelineError("materials", f"Materials generation failed: {e}", original_error=e) from e

    def _save_material_artifacts(
        self,
        *,
        seg_dir: Path,
        seg_id: str,
        segment_index: int,
        pbr: PBRTextures,
        mask: np.ndarray,
        segment_metadata: MaskMetadata,
        ingest_result: LinearIngestResult,
        backend_cfg: str,
    ) -> None:
        """Persist materials diagnostics and provenance sidecars for a segment."""
        metadata_dict = pbr.metadata.to_dict() if pbr.metadata is not None else None
        governance_overrides = {
            "allow_research_materials": bool(self.config.materials.get("allow_research_materials", False)),
            "allow_unattested_materials": bool(self.config.materials.get("allow_unattested_materials", False)),
        }
        diagnostics_payload = {
            "schema_version": "1.0.0",
            "segment_id": seg_id,
            "segment_index": segment_index,
            "requested_backend": backend_cfg,
            "governance_overrides": governance_overrides,
            "generation_metadata": _sanitize_json_value(metadata_dict),
            "material_properties": _sanitize_json_value(pbr.properties),
            "mask_area": int(np.count_nonzero(mask)),
            "texture_shapes": {
                "albedo": list(pbr.albedo.shape),
                "normal": list(pbr.normal.shape),
                "roughness": list(pbr.roughness.shape),
                "metallic": list(pbr.metallic.shape),
                "ao": list(pbr.ambient_occlusion.shape),
                "height": None if pbr.height is None else list(pbr.height.shape),
            },
        }
        provenance_payload = {
            "schema_version": "1.0.0",
            "segment_id": seg_id,
            "segment_index": segment_index,
            "input_path": str(getattr(ingest_result, "input_path", "")) or None,
            "input_content_hash": getattr(ingest_result, "content_hash", None),
            "input_gamma": getattr(ingest_result, "gamma", None),
            "input_size": list(getattr(ingest_result, "input_size", pbr.albedo.shape[:2])),
            "mask_metadata": {
                "area": segment_metadata.area,
                "bbox": list(segment_metadata.bbox),
                "stability_score": segment_metadata.stability_score,
                "material_label": getattr(segment_metadata, "material_label", None),
            },
            "backend_decision": None if metadata_dict is None else metadata_dict.get("backend_decision"),
            "governance_overrides": governance_overrides,
            "artifact_payload_hashes": {
                "hash_algorithm": "sha256",
                "hash_target": "numpy_array_bytes",
                "albedo": _sha256_array(pbr.albedo),
                "normal": _sha256_array(pbr.normal),
                "roughness": _sha256_array(pbr.roughness),
                "metallic": _sha256_array(pbr.metallic),
                "ao": _sha256_array(pbr.ambient_occlusion),
                "height": None if pbr.height is None else _sha256_array(pbr.height),
            },
        }
        for filename, payload in (
            ("diagnostics.json", diagnostics_payload),
            ("provenance.json", provenance_payload),
        ):
            try:
                _write_json_atomic(
                    seg_dir / filename,
                    _sanitize_json_value(payload),
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    trailing_newline=True,
                )
            except Exception as exc:
                logger.warning("Failed to write materials %s for %s: %s", filename, seg_id, exc)

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
                "Single-view reconstruction is not yet implemented. (TODO_INVENTORY.md) "
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
        from . import graph_pipeline

        return graph_pipeline.process_with_graph(self, input_path, output_dir, save_intermediates)

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
        data = load_and_validate_preset(path)
        return SpatialAIPipeline._dict_to_config(data)

    @staticmethod
    def _validate_runtime_config_dict(data: Dict[str, Any]) -> None:
        """Apply runtime governance checks before config normalization."""
        validate_non_commercial_preset(data)
        validate_materials_preset(data)

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

        # Normalize stage aliases while preserving backward compatibility.
        segmentation_data = pipeline_data.get("segmentation") or pipeline_data.get("segment", {})
        reconstruction_data = pipeline_data.get("reconstruction") or pipeline_data.get("reconstruct", {})
        materials_data = dict(pipeline_data.get("materials", {}))
        materials_data.update(_extract_materials_governance_overrides(data))
        stages = list(pipeline_data.keys())
        stage_aliases = {"segment": "segmentation", "reconstruct": "reconstruction"}
        stages = [stage_aliases.get(stage_name, stage_name) for stage_name in stages]

        # Parse use_execution_graph flag (ADR-029)
        use_execution_graph = data.get("use_execution_graph", False)

        # Build config
        config = PipelineConfig(
            tier=data.get("tier", "standard"),
            stages=stages,
            ingest=pipeline_data.get("ingest", {}),
            segmentation=segmentation_data,
            materials=materials_data,
            reconstruction=reconstruction_data,
            resource_limits=resource_limits,
            error_strategy=error_strategy,
            use_execution_graph=use_execution_graph,
        )

        return config
