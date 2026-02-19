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
    """

    tier: str
    stages: List[str] = field(default_factory=lambda: ["ingest", "segment"])
    ingest: Dict[str, Any] = field(default_factory=dict)
    segmentation: Dict[str, Any] = field(default_factory=dict)
    materials: Dict[str, Any] = field(default_factory=dict)
    reconstruction: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Optional[ResourceLimits] = None
    error_strategy: ErrorRecoveryStrategy = ErrorRecoveryStrategy.RETRY

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
        if "reconstruction" in self.stages and self.tier not in ["apex_research", "apex_research_ultra", "experimental"]:
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
            repo_id = model_cfg.get("repo_id", "facebook/sam2-hiera-large")
            revision = model_cfg.get("revision", None)  # Should be pinned in preset

            # Select device
            device = self.resource_manager.select_device()

            # Create backend
            backend = SAM2Backend(
                model_size=model_size,
                device=device,
                repo_id=repo_id,
                revision=revision,
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
        )

        return config
