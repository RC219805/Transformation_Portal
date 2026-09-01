"""Orchestrator for V3 depth + V2 enhancement pipeline.

Two-stage pipeline:
1. Stage A (V3): Generate depth assets using
   DA3 (Inference -> Post-Processing -> Write)
2. Stage B (V2): Consume depth assets -> V2 Subprocess -> Output

Improvements implemented (per requirements):
1. Output key generation with directory structure and SHA-1 hash suffix
2. Improved skip logic using stored config fingerprint
3. Lazy image preprocessing (validation/preprocess only when needed)
4. Configurable hash computation (HashMode)
5. PBR generation with cached depth (prefer float depth)
6. Accurate batch execution timestamps
7. Defensive check for output existence
8. Lazy manifest loading with LRU cache (15-20% I/O reduction)
9. Phase 3: xxHash for output keys (5x faster, opt-in)
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import importlib.metadata
import io
import json
import logging
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from enum import Enum
from functools import lru_cache
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, cast

import numpy as np

from transformation_portal.attestation.model_lock_manifest import load_model_lock_manifest as load_model_lock_manifest_payload
from transformation_portal.core.security.model_lock import is_pinned_revision

from ..core.ml_dependency_health import (
    detect_transformers_torch_version_issue,
)
from ..depth.backends.protocol import DepthBackend, LicenseRestrictionError
from ..depth.backends.registry import DepthBackendRegistry
from ..ingest.canonical_json import canonicalize_json, dump_json, dumps_json
from ..reporting.contracts import (
    build_orchestrator_result_capability_report,
    build_quality_gate_report,
    resolve_result_quality_gate,
)
from ..spatial_ai.reconstruction.contracts import (  # noqa: E501
    LicenseRestrictionError as ReconstructionLicenseRestrictionError,
)
from ..vlm_captioning import (
    FASTVLM_MODEL_ROLES,
    FastVLMRuntimeConfig,
    build_fastvlm_sidecar,
    build_vlm_image_proxy,
    default_fastvlm_runtime_root,
    resolve_fastvlm_model_id,
    resolve_fastvlm_model_path,
    resolve_fastvlm_runtime_path,
    run_fastvlm_caption,
)
from ..vlm_captioning.fastvlm_runtime import dumps_sidecar
from ._backend_contract import normalize_backend_id, normalize_backend_provenance
from .apex_codes import APEX_MATERIALS_SEGMENTATION_DOMINATES_NO_PIXEL_OPS

# ADR-043: Artifact management extracted to artifact_manager.py
# NOTE: xxHash support is now handled in artifact_manager.py (ADR-043)
# The XXHASH_AVAILABLE constant is imported from artifact_manager
from .artifact_manager import (
    XXHASH_AVAILABLE,
    ArtifactManager,
    build_artifact_index,
    coerce_output_paths,
    compute_artifact_merkle_root,
    has_expanded_stage_a_fingerprint,
    infer_artifact_type,
    load_existing_manifest,
    make_output_key,
    normalize_backend_provenance_for_reuse,
    normalize_v2_status,
    preserved_v2_result_from_manifest,
    restore_materials_v3_from_manifest,
    segmentation_mask_artifact_path,
    v2_log_filename,
)
from .artifact_tree import build_artifact_tree
from .batch_stats import compute_batch_runtime_stats, detect_runtime_outliers
from .camera_metadata_loader import load_scene_cameras, load_sidecar_payload
from .config import DA3Config, EnhanceConfig, ModelVariant

# ADR-043: Config resolution extracted to config_resolver.py
from .config_resolver import (
    ConfigResolver,
    PresetInfo,
    ResolvedConfig,
    apply_effective_da3_runtime_config,
    apply_effective_raw_runtime_config,
    build_apex_depth_gate_fingerprint_payload,
    build_depth_cache_fingerprint,
    build_depth_cache_payload,
    build_materials_fingerprint_payload,
    build_orchestrator_run_card_config_fingerprint,
    build_pbr_fingerprint_payload,
    build_run_card_config_fingerprint,
    compute_config_fingerprint,
    discover_presets,
    finalize_run_card_config_fingerprint,
    require_model_variant,
    resolve_preset,
)
from .depth_cache import DepthCache
from .depth_writer import atomic_write_depth_u16_png_with_stats
from .input_discovery import DiscoveryConfig, discover_images
from .input_manager import ImageInput
from .io_atomic import atomic_temp_file, atomic_write_pil_png
from .manifest import (
    BackendSelectionMetadata,
    BatchManifest,
    CombinedManifest,
    ConfigFingerprint,
    DepthMetadata,
    InputMetadata,
    ReproMetadata,
    TimingMetadata,
    V2Metadata,
    capture_environment,
    compute_file_sha256,
    get_git_revision,
)
from .path_aliasing import relative_to_path_alias
from .pbr import generate_pbr_maps
from .pbr_writer import write_pbr_maps
from .postprocessing import Postprocessor
from .provenance import ExiftoolNotFoundError, ProvenanceError, capture_provenance
from .reconstruction_runner import (
    diagnostics_artifact_path,
    manifest_artifact_path,
    run_scene_reconstruction,
    write_scene_debug_bundle,
)
from .run_card_contract import build_runtime_licensing_manifest, render_run_card_output_relative_path
from .scene_context import SceneContext
from .scene_groups import SceneGroup, build_scene_groups
from .scene_integrity import (
    build_dataset_triage_report,
    build_scene_manifest,
    check_camera_geometry_sanity,
    compute_scene_fingerprint,
    normalize_camera_poses,
    verify_scene_integrity,
    write_scene_manifest,
)
from .scene_preflight import validate_scene_preflight, write_scene_preflight_artifact
from .security import HashMode, sanitize_file_stem, sanitize_path_component_nonlossy
from .v2_runner import V2Runner, find_v2_report

# ADR-043: Run card validators extracted to validators/run_card_validator.py
# Backward-compatible re-exports preserve existing import paths
from .validators import validate_run_card_backend_semantics, validate_run_card_payload
from .validators.run_card_validator import _default_schema_path as _run_card_schema_path

# Backward-compatible aliases for existing tests and consumers
_validate_run_card_payload = validate_run_card_payload
_validate_run_card_backend_semantics = validate_run_card_backend_semantics
_infer_artifact_type = infer_artifact_type
_build_artifact_index = build_artifact_index
_compute_artifact_merkle_root = compute_artifact_merkle_root
_v2_log_filename = v2_log_filename

# ADR-043: Config resolution backward-compatible aliases
_build_materials_fingerprint_payload = build_materials_fingerprint_payload
_build_pbr_fingerprint_payload = build_pbr_fingerprint_payload
_build_apex_depth_gate_fingerprint_payload = build_apex_depth_gate_fingerprint_payload
_build_depth_cache_payload = build_depth_cache_payload

# ADR-043: Pipeline coordination extracted to pipeline_coordinator.py
from .pipeline_coordinator import (
    BackendSelection,
    ExecutionPlan,
    PipelineCoordinator,
    build_active_depth_state,
    build_backend_metadata_for_attempts,
    default_model_id_for_backend,
    derive_model_id_from_backend_instance,
    expected_output_depth_units_for_backend,
    extract_model_artifact_from_attempts,
    extract_model_id_from_attempts,
    get_or_create_depth_backend,
    infer_operational_error_code,
    initialize_depth_backend_state,
    normalize_sha256,
    resolve_backend_model_artifact,
    resolve_backend_model_id,
    resolve_requested_backend,
    resolve_runtime_backend_chain,
    seed_depth_attempts_from_selection_fallback,
    select_backend,
    typed_nullary_callable,
)

# ADR-043: Pipeline coordination backward-compatible aliases
_resolve_runtime_backend_chain = resolve_runtime_backend_chain
_expected_output_depth_units_for_backend = expected_output_depth_units_for_backend
_default_model_id_for_backend = default_model_id_for_backend
_derive_model_id_from_backend_instance = derive_model_id_from_backend_instance
_resolve_backend_model_id = resolve_backend_model_id
_resolve_requested_backend = resolve_requested_backend
_select_backend = select_backend

# ADR-043 Phase 6: Execution engine extracted to execution_engine.py
# NOTE: The orchestrator keeps its own _generate_pbr_stage and _run_v2_stage methods
# because they have different signatures and return types than the standalone functions.
# - EnhanceOrchestrator._generate_pbr_stage(self, depth, output_key) -> Optional[dict]
# - EnhanceOrchestrator._run_v2_stage(self, image_input, ...) -> tuple[dict, float, Optional[Path]]
# The extracted functions are the new canonical API for standalone use:
# - generate_pbr_stage(...) -> PBRStageResult
# - run_v2_stage(...) -> V2StageResult
# ADR-043 Phase 6 also adds artifact persistence helpers:
# - persist_depth_artifacts(...) -> DepthArtifactResult
# - persist_enhanced_image(...) -> EnhancedImageResult
from .execution_engine import (
    DepthArtifactPaths,
    DepthArtifactResult,
    DepthStageResult,
    EnhancedImageResult,
    ExecutionEngine,
    MaterialsV3StageResult,
    PBRStageResult,
    V2StageResult,
    generate_pbr_stage,
    persist_depth_artifacts,
    persist_enhanced_image,
    run_v2_stage,
)

logger = logging.getLogger(__name__)


class ApexStrictGateError(RuntimeError):
    """Raised when APEX strict quality gates are violated."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.details = details or {}
        super().__init__(f"[{code}] {message}")


class _MaskSerializationRejected(RuntimeError):
    """Internal signal for non-fatal mask serialization rejection."""


def _shape_2d(arr: np.ndarray) -> tuple[int, int]:
    """Extract 2D shape from array as properly typed tuple[int, int].

    This helper converts numpy shape slices into typed 2-element tuples
    to satisfy mypy's strict tuple type checking. Without this, the
    generator expression `tuple(int(v) for v in arr.shape[:2])` produces
    `tuple[int, ...]` which is incompatible with `tuple[int, int]`.

    Args:
        arr: A numpy array with at least 2 dimensions (depth maps, masks, images).

    Returns:
        A tuple of (height, width) representing the first two dimensions.

    Raises:
        IndexError: If the array has fewer than 2 dimensions.
    """
    if arr.ndim < 2:
        raise IndexError(f"_shape_2d requires an array with at least 2 dimensions, got {arr.ndim}D array")
    height, width = int(arr.shape[0]), int(arr.shape[1])
    return (height, width)


def _log_dependency_status() -> dict:
    """Log startup dependency availability report.

    Reports status of optional dependencies with actionable guidance.
    Makes warnings explicit, not vague.

    Returns:
        Dictionary with dependency status for testing/debugging
    """
    status: Dict[str, Any] = {}

    def _distribution_version(distribution_name: str) -> Optional[str]:
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            return None

    torch_version = _distribution_version("torch")
    status["torch"] = torch_version is not None
    if torch_version is not None:
        status["torch_version"] = torch_version
        logger.debug("torch %s installed", torch_version)
    else:
        logger.info(
            "torch not available - ML features disabled. Install: pip install torch",
        )

    transformers_version = _distribution_version("transformers")
    status["transformers"] = transformers_version is not None
    if transformers_version is not None:
        status["transformers_version"] = transformers_version
        logger.debug("transformers %s installed", transformers_version)
    else:
        logger.info(
            "transformers not available - depth models disabled. Install: pip install transformers",
        )

    runtime_issue = detect_transformers_torch_version_issue(torch_version, transformers_version)
    status["torch_transformers_compatible"] = runtime_issue is None
    if runtime_issue:
        logger.warning(runtime_issue)

    coremltools_version = _distribution_version("coremltools")
    status["coremltools"] = coremltools_version is not None
    if coremltools_version is not None:
        status["coremltools_version"] = coremltools_version
        logger.debug("coremltools %s installed", coremltools_version)
    else:
        logger.debug(
            "coremltools not available (optional). Install: pip install coremltools",
        )

    skimage_version = _distribution_version("scikit-image")
    status["scikit-image"] = skimage_version is not None
    if skimage_version is not None:
        status["scikit-image_version"] = skimage_version
        logger.debug("scikit-image %s installed", skimage_version)
    else:
        logger.debug(
            "scikit-image not available (optional for advanced filtering)",
        )

    numba_version = _distribution_version("numba")
    status["numba"] = numba_version is not None
    if numba_version is not None:
        status["numba_version"] = numba_version
        logger.debug(
            "numba %s available - performance optimizations enabled",
            numba_version,
        )
    else:
        logger.debug(
            "numba not available - using NumPy fallback (30-50%% slower for some operations)",
        )

    # Check HF_TOKEN for model downloads
    hf_token = os.environ.get("HF_TOKEN")
    status["hf_token"] = bool(hf_token)
    if hf_token:
        logger.debug(
            "HF_TOKEN present - authenticated" " model downloads enabled",
        )
    else:
        logger.debug(
            "HF_TOKEN not set - using" " unauthenticated downloads" " (rate limits apply, slower" " warm starts)",
        )
        logger.debug(
            "  Set HF_TOKEN for faster" " downloads: export" " HF_TOKEN=<your_token>",
        )

    return status


@lru_cache(maxsize=128)
def _load_manifest_cached(
    manifest_path: str,
    mtime: float,
) -> CombinedManifest:
    """Cache manifests by path + modification time.

    Reduces I/O by 15-20% by avoiding redundant manifest loads during:
    - should_skip_depth() checks
    - V2 stage validation
    - Final manifest updates

    Args:
        manifest_path: Path to manifest file (as string for hashability)
        mtime: File modification time for cache invalidation

    Returns:
        Loaded CombinedManifest instance
    """
    return CombinedManifest.load(Path(manifest_path))


# NOTE: Run card validation functions have been extracted to
# validators/run_card_validator.py as part of ADR-043 decomposition.
# The following are now imported:
# - _run_card_schema_path (aliased from _default_schema_path)
# - validate_run_card_payload (was _validate_run_card_payload)
# - validate_run_card_backend_semantics (was _validate_run_card_backend_semantics)

# NOTE: Artifact management functions have been extracted to
# artifact_manager.py as part of ADR-043 decomposition.
# The following are now imported:
# - make_output_key
# - infer_artifact_type (was _infer_artifact_type)
# - v2_log_filename (was _v2_log_filename)
# - build_artifact_index (was _build_artifact_index)
# - compute_artifact_merkle_root (was _compute_artifact_merkle_root)


class EnhanceOrchestrator:
    """Orchestrates V3 depth generation + V2 enhancement pipeline.

    Attributes:
        config: Enhancement configuration
        output_root: Base directory for all outputs
        verify_outputs: If True, verify cached
            outputs exist on disk before
            skipping (defensive check)
    """

    def __init__(
        self,
        config: EnhanceConfig,
        output_root: Path,
        verify_outputs: bool = True,
    ):
        """Initialize the orchestrator.

        Args:
            config: Enhancement configuration object
            output_root: Base directory to store outputs and manifests
            verify_outputs: Whether to verify
                cached outputs exist before
                skipping (default: True)
        """
        # Log dependency status on first initialization
        _log_dependency_status()

        self.config = apply_effective_da3_runtime_config(config)
        self.config = apply_effective_raw_runtime_config(self.config)
        self.output_root = Path(output_root)
        self.verify_outputs = verify_outputs

        if config.hash_mode == HashMode.NEVER:
            logger.warning(
                "Hash mode set to 'never'" " - manifests will lack" " integrity verification.",
            )

        # Create output directories
        self.depth_dir = self.output_root / "depth"
        self.v2_dir = self.output_root / "v2"
        self.manifests_dir = self.output_root / "manifests"
        self.logs_dir = self.output_root / "logs"
        self.segmentation_dir = self.output_root / "segmentation"
        self.reconstruction_dir = self.output_root / "reconstruction"
        # zones/ directory reserved for future zone-based processing
        # Only created when zoning features are enabled
        self.zones_dir = self.output_root / "zones"

        for d in [
            self.depth_dir,
            self.v2_dir,
            self.manifests_dir,
            self.logs_dir,
            self.segmentation_dir,
            self.reconstruction_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

        # Note: zones_dir intentionally NOT created here
        # Will be created on-demand when zoning features are implemented

        resolved_config = ConfigResolver().resolve(self.config)
        self._resolved_config = resolved_config
        self._resolved_model_contract = resolved_config.resolved_model_contract
        self.config = resolved_config.enhance_config
        da3_config = resolved_config.da3_config

        # Initialize Depth Backend via Registry (ADR-019)
        self.depth_backend: Optional[DepthBackend] = None
        self._initialize_depth_backend()

        # Initialize Postprocessor
        # (FIX: Ensures refine_edges/bilateral
        # settings from preset are applied)
        self.postprocessor = Postprocessor(da3_config.postprocessing)

        # Initialize Materials V3 Engine (if enabled)
        self.materials_v3_engine: Optional[Any] = None
        if config.enable_materials_v3:
            from .materials_v3 import MaterialsV3Engine

            self.materials_v3_engine = MaterialsV3Engine(config)
            logger.info("Materials V3 surface-aware finishing enabled")
        # Initialize V2 Runner and Environment (with fail-fast validation)
        self.v2_runner: Optional[V2Runner] = None
        if config.enable_v2 and config.v2_preset is not None:
            self.v2_runner = V2Runner()
            # Fail-fast: Validate V2 script exists before processing
            if not self.v2_runner.script_path.exists():
                raise FileNotFoundError(
                    "V2 enhancement script"
                    " not found:"
                    f" {self.v2_runner.script_path}"
                    "\nRequired location:"
                    " scripts/enhance_image.py"
                    " in repository root"
                    "\n\nOptions:\n"
                    "  1. Create the V2"
                    " enhancement script at"
                    " the expected location\n"
                    "  2. Set enable_v2=False"
                    " for PBR-only workflows\n"
                    "  3. Set v2_preset=None"
                    " to skip V2 stage"
                )
            logger.info(
                "V2 enhancement enabled" + " with script: %s",
                self.v2_runner.script_path,
            )
        else:
            logger.info(
                "V2 enhancement disabled" + " (PBR-only mode)",
            )

        # Adjusted path logic for
        # src/transformation_portal/lux_depth_v3
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        git_rev = get_git_revision(repo_root)
        self.v3_git = git_rev
        self.v2_git = git_rev
        self.environment = capture_environment()

        # Phase 2: Parallelization setup with stage-aware concurrency
        # MPS/GPU inference: limit to 1-2 workers to avoid memory contention
        # CPU/I/O operations: use moderate parallelism

        # Check for forward-compatible max_gpu_workers override
        max_gpu_workers_override = getattr(config, "max_gpu_workers", None)
        max_workers_override = getattr(config, "max_workers", None)

        if config.depth_device in ("mps", "cuda"):
            # GPU backends: conservative concurrency to avoid VRAM contention
            # Note: max_gpu_workers is inference-specific, may be applied later
            if max_workers_override is not None:
                self.max_workers = max_workers_override
            else:
                self.max_workers = min(2, cpu_count())
            # Store GPU-specific limit separately for inference stage
            self.max_gpu_workers = max_gpu_workers_override if max_gpu_workers_override is not None else self.max_workers
            logger.debug(
                "GPU/MPS device detected -" " limiting workers to %d" " for VRAM management",
                self.max_workers,
            )
        else:
            # CPU backend: moderate parallelism for I/O-bound operations
            if max_workers_override is not None:
                self.max_workers = max_workers_override
            else:
                self.max_workers = config.max_parallel_workers or max(1, cpu_count() - 1)
            self.max_gpu_workers = self.max_workers

        self._use_parallel = config.enable_parallel_processing
        _par_label = "enabled" if self._use_parallel else "disabled"
        logger.debug(
            "Parallel processing: %s" + " (workers=%d)",
            _par_label,
            self.max_workers,
        )

        # Phase 2: Content-addressable depth cache (opt-in)
        self.depth_cache = (
            DepthCache(
                self.output_root,
                max_size_gb=(config.depth_cache_max_size_gb),
            )
            if config.enable_depth_cache
            else None
        )
        if self.depth_cache:
            logger.info(
                "Depth cache enabled: %s",
                self.depth_cache.cache_dir,
            )

        # Injectable seam for lightweight reconstruction tests.
        self.run_scene_reconstruction_fn: Callable[..., Path] = run_scene_reconstruction

        # Per-batch / per-image state (set during enhance_batch).
        # Preserve backend selection resolved in _initialize_depth_backend().
        self._active_batch_id: Optional[str] = None
        self._active_backend_metadata: Optional[BackendSelectionMetadata] = self._backend_metadata
        self._active_depth_attempts: List[Dict[str, Any]] = []
        self._active_selected_attempt_index: Optional[int] = None
        self._active_run_card_segmentation_metadata: Dict[str, Dict[str, Any]] = {}

    @property
    def _model_variant(self) -> ModelVariant:
        """Return model_variant; guaranteed set after __init__."""
        return require_model_variant(self.config)

    def _initialize_depth_backend(self) -> None:
        """Initialize depth backend using registry (ADR-019).

        Implements backend selection with fallback logic:
        1. Try requested backend (from config.depth_backend, with Apple Silicon opt-ins)
        2. Check availability (checkpoint + dependencies)
        3. Fallback to configured operational chain (default: da3 -> da2)
        4. Optionally fallback to synthetic in explicit test/CI mode
        5. Record selection decision in metadata
        """
        state = initialize_depth_backend_state(
            self.config,
            self._model_variant,
            self._resolve_backend_model_id,
            registry_factory=DepthBackendRegistry,
        )
        self._depth_registry = state.registry
        self.depth_backend = state.depth_backend
        self._backend_init_errors = state.init_errors
        self._depth_backend_cache = state.backend_cache
        self._backend_metadata = state.backend_metadata
        self._active_backend_metadata = self._backend_metadata
        self._active_depth_attempts = []
        self._active_selected_attempt_index = None

    # ADR-043: Pipeline coordination methods now delegate to pipeline_coordinator module

    def _resolve_runtime_backend_chain(
        self,
        primary_backend_id: str,
    ) -> List[str]:
        """Resolve ordered runtime fallback chain.

        Delegates to pipeline_coordinator.resolve_runtime_backend_chain().
        """
        return resolve_runtime_backend_chain(primary_backend_id, self.config)

    @staticmethod
    def _expected_output_depth_units_for_backend(
        backend_id: str,
    ) -> str:
        """Return expected output depth units.

        Delegates to pipeline_coordinator.expected_output_depth_units_for_backend().
        """
        return expected_output_depth_units_for_backend(backend_id)

    def _build_depth_cache_fingerprint(
        self,
        backend_id: str,
    ) -> str:
        """Build backend-scoped cache fingerprint."""
        expected_units = self._expected_output_depth_units_for_backend(
            backend_id,
        )
        return build_depth_cache_fingerprint(
            self.config,
            self._model_variant,
            backend_id,
            expected_units,
            resolved_model_contract=getattr(self, "_resolved_model_contract", None),
        )

    def _default_model_id_for_backend(self, backend_id: str) -> str:
        """Return canonical backend model identifier for provenance.

        Delegates to pipeline_coordinator.default_model_id_for_backend().
        """
        return default_model_id_for_backend(backend_id, self._model_variant, config=self.config)

    def _derive_model_id_from_backend_instance(
        self,
        backend_id: str,
        backend: Optional[Any],
    ) -> Optional[str]:
        """Best-effort model id extraction from backend instance.

        Delegates to pipeline_coordinator.derive_model_id_from_backend_instance().
        """
        return derive_model_id_from_backend_instance(backend_id, backend)

    def _resolve_backend_model_id(
        self,
        backend_id: str,
        *,
        result_metadata: Optional[Dict[str, Any]] = None,
        backend: Optional[Any] = None,
    ) -> str:
        """Resolve stable model id for provenance and run-card semantics.

        Delegates to pipeline_coordinator.resolve_backend_model_id().
        """
        return resolve_backend_model_id(
            backend_id,
            result_metadata=result_metadata,
            backend=backend,
            model_variant=self._model_variant,
            config=self.config,
        )

    @staticmethod
    def _normalize_sha256(value: Any) -> Optional[str]:
        """Normalize SHA-256 digest to lowercase hex."""
        return normalize_sha256(value)

    @staticmethod
    def _typed_nullary_callable(value: Any) -> Optional[Callable[[], Any]]:
        """Return a typed no-arg callable for dynamically loaded attributes."""
        return typed_nullary_callable(value)

    def _resolve_backend_model_artifact(
        self,
        backend_id: str,
        *,
        result_metadata: Optional[Dict[str, Any]] = None,
        backend: Optional[Any] = None,
    ) -> Dict[str, Optional[str]]:
        """Resolve backend model artifact identity fields."""
        return resolve_backend_model_artifact(
            backend_id,
            result_metadata=result_metadata,
            backend=backend,
        )

    @staticmethod
    def _extract_model_id_from_attempts(
        selected_backend: str,
        attempts: List[Dict[str, Any]],
        *,
        selected_attempt_index: Optional[int] = None,
    ) -> Optional[str]:
        """Extract selected backend model id from attempt history."""
        return extract_model_id_from_attempts(
            selected_backend,
            attempts,
            selected_attempt_index=selected_attempt_index,
        )

    @staticmethod
    def _extract_model_artifact_from_attempts(
        selected_backend: str,
        attempts: List[Dict[str, Any]],
        *,
        selected_attempt_index: Optional[int] = None,
    ) -> Dict[str, Optional[str]]:
        """Extract selected backend model artifact identity from attempt history."""
        return extract_model_artifact_from_attempts(
            selected_backend,
            attempts,
            selected_attempt_index=selected_attempt_index,
        )

    def _seed_depth_attempts_from_selection_fallback(self) -> List[Dict[str, Any]]:
        """Materialize backend-selection fallback into per-image attempt history."""
        return seed_depth_attempts_from_selection_fallback(
            getattr(self, "_backend_metadata", None),
            getattr(self, "_backend_init_errors", None),
            self.config,
            self._model_variant,
        )

    def _get_or_create_depth_backend(
        self,
        backend_id: str,
    ) -> Any:
        """Fetch backend from cache or registry."""
        return get_or_create_depth_backend(
            backend_id,
            active_backend=getattr(self, "depth_backend", None),
            backend_cache=self._depth_backend_cache,
            registry=self._depth_registry,
            config=self.config,
        )

    @staticmethod
    def _infer_operational_error_code(
        error: Exception,
    ) -> str:
        """Map backend exceptions to error codes."""
        return infer_operational_error_code(error)

    def _set_active_depth_state(
        self,
        backend_metadata: Optional[BackendSelectionMetadata],
        depth_attempts: List[Dict[str, Any]],
        selected_attempt_index: Optional[int],
    ) -> None:
        """Persist per-image depth state for downstream error/reporting paths."""
        active_state = build_active_depth_state(
            backend_metadata,
            depth_attempts,
            selected_attempt_index,
        )
        self._active_backend_metadata = active_state.backend_metadata
        self._active_depth_attempts = active_state.depth_attempts
        self._active_selected_attempt_index = active_state.selected_attempt_index

    def _build_backend_metadata_for_attempts(
        self,
        selected_backend: str,
        attempts: List[Dict[str, Any]],
        result_metadata: Optional[Dict[str, Any]] = None,
        selected_attempt_index: Optional[int] = None,
    ) -> BackendSelectionMetadata:
        """Build per-image backend selection metadata."""
        return build_backend_metadata_for_attempts(
            selected_backend,
            attempts,
            self._backend_metadata,
            self.config,
            self._resolve_backend_model_id,
            result_metadata=result_metadata,
            selected_attempt_index=selected_attempt_index,
            backend_cache=getattr(self, "_depth_backend_cache", {}),
        )

    # ADR-043: Config resolution methods now delegate to config_resolver module

    def _build_materials_fingerprint_payload(self) -> Dict[str, Any]:
        """Build deterministic Materials V3 fingerprint payload.

        Delegates to config_resolver.build_materials_fingerprint_payload().
        """
        return build_materials_fingerprint_payload(self.config)

    def _build_pbr_fingerprint_payload(self) -> Dict[str, Any]:
        """Build deterministic PBR fingerprint payload.

        Delegates to config_resolver.build_pbr_fingerprint_payload().
        """
        return build_pbr_fingerprint_payload(self.config)

    def _build_apex_depth_gate_fingerprint_payload(self) -> Dict[str, Any]:
        """Build deterministic APEX depth-gate fingerprint payload.

        Delegates to config_resolver.build_apex_depth_gate_fingerprint_payload().
        """
        return build_apex_depth_gate_fingerprint_payload(self.config)

    def _build_depth_cache_payload(self) -> Dict[str, Any]:
        """Build depth-cache fingerprint payload.

        This intentionally stays narrower than manifest Stage A invalidation
        because the cache stores postprocessed float depth, not Materials V3,
        PBR, or V2 deliverables.

        Delegates to config_resolver.build_depth_cache_payload().
        """
        return build_depth_cache_payload(
            self.config,
            self._model_variant,
            resolved_model_contract=getattr(self, "_resolved_model_contract", None),
        )

    def compute_config_fingerprint(self) -> ConfigFingerprint:
        """Compute configuration fingerprint for cache validation.

        Delegates to config_resolver.compute_config_fingerprint().
        """
        return compute_config_fingerprint(
            self.config,
            self._model_variant,
            resolved_model_contract=getattr(self, "_resolved_model_contract", None),
        )

    @staticmethod
    def _finalize_run_card_config_fingerprint(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Attach canonical JSON and SHA-256 over the resolved fingerprint payload."""
        return finalize_run_card_config_fingerprint(payload)

    def _build_run_card_config_fingerprint(
        self,
        *,
        backend_selection: Optional[Dict[str, Any]] = None,
        run_card_version: Optional[str] = None,
        include_proofs: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Build run-card config fingerprint.

        Delegates to config_resolver.build_orchestrator_run_card_config_fingerprint().
        """
        return build_orchestrator_run_card_config_fingerprint(
            self.config,
            self._model_variant,
            self._backend_metadata,
            backend_selection=backend_selection,
            run_card_version=run_card_version,
            include_proofs=include_proofs,
            resolved_model_contract=getattr(self, "_resolved_model_contract", None),
        )

    def _extract_v2_depth_handoff_status(
        self,
        v2_result: Optional[Dict[str, Any]],
        v2_report_path: Optional[Path],
    ) -> Optional[bool]:
        """Return whether V2 consumed depth-map input."""
        if isinstance(v2_result, dict):
            if isinstance(v2_result.get("depth_consumed"), bool):
                return bool(v2_result.get("depth_consumed"))
            stage_metadata = v2_result.get("stage_metadata")
            if isinstance(stage_metadata, dict) and "has_depth" in stage_metadata:
                return bool(
                    stage_metadata.get("has_depth"),
                )
            if isinstance(
                v2_result.get("depth_map"),
                str,
            ):
                return True
            if "depth_map" in v2_result and v2_result.get("depth_map") is None:
                return False

        if v2_report_path and v2_report_path.exists():
            try:
                with open(
                    v2_report_path,
                    "r",
                    encoding="utf-8",
                ) as report_file:
                    report_payload = json.load(report_file)
            except Exception as exc:
                logger.debug(
                    "Failed to parse V2" " report for depth" " handoff check" " (%s): %s",
                    v2_report_path,
                    exc,
                )
            else:
                if isinstance(report_payload.get("depth_consumed"), bool):
                    return bool(report_payload.get("depth_consumed"))
                stage_metadata = report_payload.get("stage_metadata")
                if (
                    isinstance(
                        stage_metadata,
                        dict,
                    )
                    and "has_depth" in stage_metadata
                ):
                    return bool(stage_metadata.get("has_depth"))
                if isinstance(report_payload.get("depth_map"), str):
                    return True
                if "depth_map" in report_payload and report_payload.get("depth_map") is None:
                    return False

        return None

    def _enforce_v2_depth_handoff(
        self,
        *,
        depth_path: Optional[Path],
        v2_result: Optional[Dict[str, Any]],
        v2_report_path: Optional[Path],
    ) -> None:
        """Enforce V2 consumes depth when produced."""
        if not depth_path or not depth_path.exists():
            return

        depth_consumed = self._extract_v2_depth_handoff_status(
            v2_result=v2_result,
            v2_report_path=v2_report_path,
        )
        if depth_consumed is None:
            return
        if depth_consumed:
            return

        message = (
            "V2 depth handoff failed: depth"
            " artifact exists but V2 reported"
            " depth_consumed=false. This"
            " indicates stem-resolution"
            " drift."
        )
        details = {
            "depth_path": str(depth_path),
            "v2_report_path": str(v2_report_path) if v2_report_path else None,
            "depth_consumed": depth_consumed,
        }
        if self._is_apex_tier():
            raise ApexStrictGateError(
                "APEX_V2_DEPTH_HANDOFF_MISSING",
                message,
                details=details,
            )
        logger.warning("%s details=%s", message, details)

    def _capture_backend_metadata(self) -> BackendSelectionMetadata:
        """Capture backend selection decision for manifest (ADR-019).

        Tracks requested vs resolved backend for transparency and debugging.
        Uses metadata from _initialize_depth_backend().

        Returns:
            BackendSelectionMetadata with selection audit trail
        """
        # Return the metadata captured during initialization
        return getattr(
            self,
            "_backend_metadata",
            BackendSelectionMetadata(
                requested_backend=None,
                resolved_backend="da3",
                resolution_status="success",
                resolution_reason=None,
                model_id=self._default_model_id_for_backend("da3"),
                device=self.config.depth_device,
                attempts=[],
            ),
        )

    def _load_existing_manifest(
        self,
        manifest_path: Path,
        *,
        purpose: str,
    ) -> Optional[CombinedManifest]:
        """Best-effort manifest loader for cached-run preservation paths."""
        return load_existing_manifest(manifest_path, purpose=purpose)

    @staticmethod
    def _coerce_output_paths(raw_paths: Any) -> List[str]:
        """Normalize V2 output path payloads to a list of strings."""
        return coerce_output_paths(raw_paths)

    @staticmethod
    def _normalize_v2_status(raw_status: Any) -> str:
        """Map runner and manifest status values to the manifest V2 contract."""
        return normalize_v2_status(raw_status)

    def _restore_materials_v3_from_manifest(
        self,
        manifest: Optional[CombinedManifest],
        output_key: Path,
    ) -> tuple[Optional[dict], float, Optional[Path]]:
        """Restore persisted Materials V3 metadata for cached-depth reruns."""
        return restore_materials_v3_from_manifest(
            manifest,
            self._expected_materials_v3_enhanced_path(output_key),
        )

    def _preserved_v2_result_from_manifest(
        self,
        manifest: Optional[CombinedManifest],
    ) -> tuple[dict, Optional[Path]]:
        """Rehydrate V2 result fields from the prior manifest when reruns skip V2."""
        return preserved_v2_result_from_manifest(manifest)

    @staticmethod
    def _normalize_backend_provenance(value: Any) -> Optional[str]:
        """Normalize backend provenance identifiers for reuse checks."""
        return normalize_backend_provenance_for_reuse(value)

    @staticmethod
    def _has_expanded_stage_a_fingerprint(
        config_fingerprint: Optional[ConfigFingerprint],
    ) -> bool:
        """Return True when manifest fingerprint carries the expanded Stage A contract."""
        return has_expanded_stage_a_fingerprint(config_fingerprint)

    def _compute_or_skip_hash(
        self,
        image_path: Path,
        manifest_exists: bool = False,
        saved_hash: Optional[str] = None,
        *,
        for_manifest_write: bool = False,
    ) -> Optional[str]:
        """Compute file hash respecting HashMode configuration.

        Args:
            image_path: Path to the image file
            manifest_exists: Whether a manifest
                exists (for IF_MANIFEST_EXISTS)
            saved_hash: Previously saved hash
                from manifest
            for_manifest_write: If True, compute
                hash for writing manifest.
                If False, compute for comparison.

        Returns:
            SHA256 hash string, or None if hash not computed

        Raises:
            IOError: If hash computation fails
        """
        if self.config.hash_mode == HashMode.NEVER:
            return None

        # IF_MANIFEST_EXISTS: behavior depends on context
        if self.config.hash_mode == HashMode.IF_MANIFEST_EXISTS:
            if for_manifest_write:
                # Writing manifest: always compute
                # hash to establish/update baseline
                pass  # Fall through to compute hash
            else:
                # Reading for comparison: only compute if we have a baseline
                if not manifest_exists or not saved_hash:
                    # No baseline exists - skip comparison
                    return None

        # ALWAYS or IF_MANIFEST_EXISTS
        # (when baseline exists or writing manifest)
        try:
            return compute_file_sha256(image_path)
        except Exception as e:
            logger.error(f"Hash computation failed for {image_path}: {e}")
            raise IOError(f"Hash computation failed: {e}") from e

    def should_skip_depth(
        self,
        depth_path: Path,
        manifest_path: Path,
        image_input: ImageInput,
    ) -> bool:
        """Determine whether to skip depth computation.

        Uses stored config fingerprint for
        comparison rather than reconstructing
        from partial fields. This invalidates
        Stage A reuse when any captured
        depth/Materials/PBR configuration
        changes.

        Args:
            depth_path: Path to the depth output file
            manifest_path: Path to the manifest file
            image_input: Input image information

        Returns:
            True if depth step can be skipped,
                False otherwise
        """
        if not depth_path.exists() or not manifest_path.exists():
            return False

        try:
            # Use cached manifest loading if enabled
            if self.config.enable_manifest_cache:
                mtime = os.path.getmtime(manifest_path)
                manifest = _load_manifest_cached(str(manifest_path), mtime)
            else:
                manifest = CombinedManifest.load(manifest_path)

            # Input Integrity Check - use stored fingerprint
            saved_hash = manifest.input.image_sha256 if manifest.input else None
            if saved_hash and self.config.hash_mode != HashMode.NEVER:
                current_hash = self._compute_or_skip_hash(
                    image_input.path,
                    manifest_exists=True,
                    saved_hash=saved_hash,
                    for_manifest_write=False,
                )
                if current_hash and current_hash != saved_hash:
                    logger.info(
                        "Input image changed" " - regenerating" " depth: %s",
                        image_input.path,
                    )
                    return False

            # Config Fingerprint Check - use stored fingerprint directly
            if not manifest.config_fingerprint:
                logger.debug(
                    "No config fingerprint" " in manifest -" " regenerating depth",
                )
                return False

            stored_fp = manifest.config_fingerprint
            if not self._has_expanded_stage_a_fingerprint(stored_fp):
                logger.info(
                    "Legacy manifest missing expanded Stage A fingerprint fields - regenerating Stage A",
                )
                return False

            # Compare stored Stage A fingerprint with current config
            current_fp = self.compute_config_fingerprint()

            # Compare Stage A config using stored fingerprint's SHA256
            if current_fp.depth_only().to_sha256() != stored_fp.depth_only().to_sha256():
                logger.info("Stage A config changed - regenerating")
                return False

            stored_backend_selection = getattr(
                manifest,
                "backend_selection",
                None,
            )
            stored_resolved_backend = self._normalize_backend_provenance(
                getattr(
                    stored_backend_selection,
                    "resolved_backend",
                    None,
                ),
            )
            if not stored_resolved_backend:
                logger.info(
                    "Missing backend selection provenance in manifest - regenerating Stage A",
                )
                return False

            current_backend_selection = self._capture_backend_metadata()
            current_resolved_backend = self._normalize_backend_provenance(
                getattr(
                    current_backend_selection,
                    "resolved_backend",
                    None,
                ),
            )
            if current_resolved_backend != stored_resolved_backend:
                logger.info(
                    "Resolved backend provenance changed - regenerating Stage A: stored=%s current=%s",
                    stored_resolved_backend,
                    current_resolved_backend,
                )
                return False

            # Depth Metadata Check
            if not manifest.depth:
                return False

            # Defensive output existence check
            if self.verify_outputs:
                if not depth_path.exists():
                    logger.debug(f"Depth file missing on disk: {depth_path}")
                    return False

                # Quick read check to verify file integrity
                from .depth_writer import read_depth_u16_png

                d = read_depth_u16_png(depth_path)
                if d.ndim != 2:
                    logger.debug(
                        "Depth file has invalid" f" dimensions: {d.ndim}",
                    )
                    return False

            logger.debug(f"Resuming with existing depth: {depth_path}")
            return True
        except Exception as e:
            logger.debug(f"Skip check failed: {e}")
            return False

    def should_skip_v2(
        self,
        v2_report_path: Optional[Path],
        manifest_path: Path,
        image_input: ImageInput,
        depth_was_skipped: bool,
    ) -> bool:
        """Determine whether to skip V2 enhancement stage.

        V2 skip logic is independent of PBR
        generation. V2 enhancement is a separate
        stage from PBR map generation, and should
        be evaluated based on V2 config changes
        and output existence.

        Uses stored config fingerprint for
        comparison and performs defensive output
        existence checks if enabled.

        Args:
            v2_report_path: Path to V2 report file
            manifest_path: Path to the manifest file
            image_input: Input image information
            depth_was_skipped: Whether depth step was skipped

        Returns:
            True if V2 stage can be skipped,
                False otherwise
        """
        if not v2_report_path or not v2_report_path.exists() or not manifest_path.exists():
            return False

        try:
            # Use cached manifest loading if enabled
            if self.config.enable_manifest_cache:
                mtime = os.path.getmtime(manifest_path)
                manifest = _load_manifest_cached(str(manifest_path), mtime)
            else:
                manifest = CombinedManifest.load(manifest_path)

            # Config Fingerprint Check - use stored fingerprint directly
            if not manifest.config_fingerprint:
                logger.debug(
                    "No config fingerprint" " in manifest -" " regenerating V2",
                )
                return False

            # Compare V2-stage config using stored fingerprint's SHA256
            current_fp = self.compute_config_fingerprint()
            stored_fp = manifest.config_fingerprint

            if current_fp.v2_only().to_sha256() != stored_fp.v2_only().to_sha256():
                logger.info("V2 config changed - regenerating")
                return False

            # Consistency Check - if depth was recomputed, V2 must also rerun
            if not depth_was_skipped:
                logger.info("Depth was regenerated - V2 must rerun")
                return False

            # V2 Metadata Check - verify V2 ran successfully
            if not manifest.v2 or self._normalize_v2_status(manifest.v2.status) != "ok":
                return False

            # Defensive output check for V2 report
            if self.verify_outputs and v2_report_path:
                if not v2_report_path.exists():
                    logger.debug(f"V2 report missing: {v2_report_path}")
                    return False

            # Defensive output check for PBR assets
            if self.verify_outputs and manifest.pbr_assets:
                pbr_outputs = manifest.pbr_assets
                for label, filepath in pbr_outputs.items():
                    if isinstance(filepath, str) and filepath and label.endswith("_path"):
                        if not os.path.exists(filepath):
                            logger.debug(f"PBR output missing: {filepath}")
                            return False

            return True
        except Exception as e:
            logger.debug(f"V2 skip check failed: {e}")
            return False

    def _compute_depth_stage(
        self,
        image_input: ImageInput,
        output_key: Path,
        depth_path: Path,
        float_depth_path: Path,
        manifest_path: Path,
        skip_depth: bool,
    ) -> tuple[
        Optional[Any],
        float,
        Optional[dict],
        Optional[dict],
        float,
        Optional[Path],
        BackendSelectionMetadata,
        List[Dict[str, Any]],
    ]:
        """Stage A: Depth computation with caching and PBR generation.

        Args:
            image_input: Input image information
            output_key: Output key for artifact naming
            depth_path: Path for quantized depth PNG
            float_depth_path: Path for float depth NPY
            manifest_path: Path for manifest JSON
            skip_depth: Whether to skip depth computation

        Returns:
            Tuple of (depth_metadata,
                depth_runtime_s, pbr_assets,
                materials_v3_result,
                materials_v3_runtime_s,
                enhanced_image_path,
                backend_selection_metadata,
                depth_attempts)
        """
        depth_runtime_s = 0.0
        depth_metadata = None
        pbr_assets = None
        materials_v3_result = None
        materials_v3_runtime_s = 0.0
        # Will be set if Materials V3 produces enhanced_image
        enhanced_image_path = None
        depth_attempts: List[Dict[str, Any]] = self._seed_depth_attempts_from_selection_fallback()
        active_backend_metadata = self._backend_metadata
        self._active_selected_attempt_index = None

        if not skip_depth:
            # Lazy preprocessing: Only validate
            # and preprocess if running depth
            from .preprocessing import preprocess_image, validate_image_format

            # Check for strict verification flag (forward-compatible)
            verify_strict = getattr(self.config, "verify_images", False)

            validated_path = validate_image_format(image_input.path)

            # Optional: strict PIL.verify() for CI/ingest validation
            if verify_strict:
                from PIL import Image

                try:
                    with Image.open(validated_path) as img_verify:
                        img_verify.verify()
                    logger.debug(
                        "Strict verification" " passed: %s",
                        validated_path.name,
                    )
                except Exception as e:
                    logger.error(
                        "Strict verification" " failed: %s - %s",
                        validated_path.name,
                        e,
                    )
                    raise ValueError(
                        "Image failed strict" " verification:" f" {validated_path}",
                    ) from e

            preprocessed_array, original_shape = preprocess_image(
                validated_path,
                raw_config=self.config,
            )

            logger.info(
                "Stage A: Generating" " depth for %s...",
                output_key,
            )
            t0 = time.time()
            try:
                # Phase 2: Check content-addressable depth cache
                image_sha256 = None
                if self.depth_cache:
                    image_sha256 = self._compute_or_skip_hash(
                        image_input.path,
                        manifest_exists=False,
                        for_manifest_write=True,
                    )
                    if image_sha256 is None:
                        logger.debug(
                            "Stage A depth cache lookup skipped for %s: image hash unavailable (hash_mode=%s)",
                            output_key,
                            self.config.hash_mode.value,
                        )
                else:
                    logger.debug(
                        "Stage A depth cache disabled for %s",
                        output_key,
                    )

                # 1. Inference with per-image
                # backend attempt/fallback.
                from PIL import Image

                preprocessed_uint8 = (
                    np.clip(
                        preprocessed_array,
                        0,
                        1,
                    )
                    * 255
                ).astype(np.uint8)
                pil_image = Image.fromarray(preprocessed_uint8)
                _primary_backend_name = (
                    getattr(
                        self.depth_backend,
                        "name",
                        None,
                    )
                    or self.config.depth_backend
                    or "da3"
                )
                attempt_chain = self._resolve_runtime_backend_chain(
                    _primary_backend_name,
                )

                result = None
                depth_map = None
                depth_validity_metrics = None
                selected_backend_id = _primary_backend_name
                selected_attempt_index: Optional[int] = None
                native_depth_shape: Optional[tuple[int, int]] = None
                last_error: Optional[Exception] = None
                attempt_offset = len(depth_attempts)

                for chain_index, backend_id in enumerate(attempt_chain):
                    attempt_start = time.time()
                    attempt_slot = attempt_offset + chain_index
                    attempt_artifact = self._resolve_backend_model_artifact(
                        backend_id,
                        backend=self._depth_backend_cache.get(backend_id),
                    )
                    attempt_record: Dict[str, Any] = {
                        "attempt": attempt_slot,
                        "backend": backend_id,
                        "model_id": self._resolve_backend_model_id(
                            backend_id,
                            backend=self._depth_backend_cache.get(backend_id),
                        ),
                        "device": self.config.depth_device,
                        "status": "started",
                        "failure_kind": None,
                        "error_code": None,
                        "error_message": None,
                        "apex_gate_passed": None,
                        "cached": False,
                        "model_artifact_filename": attempt_artifact.get(
                            "model_artifact_filename",
                        ),
                        "model_artifact_sha256": attempt_artifact.get(
                            "model_artifact_sha256",
                        ),
                    }

                    try:
                        attempt_cache_fp_hash = None
                        cached_depth = None
                        backend_instance = self._depth_backend_cache.get(
                            backend_id,
                        )
                        if self.depth_cache and image_sha256:
                            attempt_cache_fp_hash = self._build_depth_cache_fingerprint(
                                backend_id,
                            )
                            cache_key_preview = f"{image_sha256[:12]}_{attempt_cache_fp_hash[:12]}"
                            logger.debug(
                                "Stage A depth cache lookup for %s (backend=%s, key=%s)",
                                output_key,
                                backend_id,
                                cache_key_preview,
                            )
                            cached_depth = self.depth_cache.get(
                                image_sha256,
                                attempt_cache_fp_hash,
                            )
                            if cached_depth is not None:
                                logger.info(
                                    "Cache hit: using" " cached depth for" " %s (backend=%s)",
                                    output_key,
                                    backend_id,
                                )
                            else:
                                logger.debug(
                                    "Stage A depth cache miss for %s (backend=%s, key=%s)",
                                    output_key,
                                    backend_id,
                                    cache_key_preview,
                                )
                        elif self.depth_cache:
                            logger.debug(
                                "Stage A depth cache lookup unavailable for %s (backend=%s): missing image hash",
                                output_key,
                                backend_id,
                            )
                        attempt_record["cached"] = bool(
                            cached_depth is not None,
                        )

                        if cached_depth is not None:
                            from ..depth.backends.protocol import DepthResult

                            # No inference runs on cache hit;
                            # mark device provenance.
                            attempt_record["device"] = "cache"
                            cache_metadata: Dict[str, Any] = {
                                "cached": True,
                                "output_normalization": "cache_reuse",
                                "cache_backend_id": backend_id,
                                "device": "cache",
                            }
                            if image_sha256 and attempt_cache_fp_hash:
                                cache_metadata["cache_key"] = f"{image_sha256}" f"_{attempt_cache_fp_hash}"

                            result_candidate = DepthResult(
                                depth_map=cached_depth,
                                original_image=preprocessed_uint8,
                                metadata=cache_metadata,
                                depth_units=("meters" if backend_id == "depth_pro" else "relative"),
                                backend_id=backend_id,
                                device="cache",
                                dtype="float32",
                                input_size=original_shape,
                            )
                        else:
                            backend = self._get_or_create_depth_backend(
                                backend_id,
                            )
                            backend_instance = backend
                            self.depth_backend = backend
                            attempt_record.update(
                                self._resolve_backend_model_artifact(
                                    backend_id,
                                    backend=backend_instance,
                                ),
                            )
                            resolved_backend_device = getattr(backend, "_device", None) or getattr(backend, "device", None)
                            if (
                                isinstance(
                                    resolved_backend_device,
                                    str,
                                )
                                and resolved_backend_device
                            ):
                                attempt_record["device"] = resolved_backend_device
                            result_candidate = cast(
                                Any,
                                self.postprocessor.process(
                                    backend.compute(pil_image),
                                ),
                            )  # type: ignore  # noqa: E501
                            if self.depth_cache and image_sha256 and attempt_cache_fp_hash:
                                self.depth_cache.store(
                                    image_sha256,
                                    attempt_cache_fp_hash,
                                    result_candidate.depth_map,
                                )

                        result_device = getattr(
                            result_candidate,
                            "device",
                            None,
                        )
                        if isinstance(result_device, str) and result_device:
                            attempt_record["device"] = result_device

                        depth_candidate = (
                            result_candidate.depth_map
                            if hasattr(
                                result_candidate,
                                "depth_map",
                            )
                            else result_candidate.depth
                        )
                        native_depth_map = np.asarray(
                            depth_candidate,
                            dtype=np.float32,
                        )
                        current_shape = _shape_2d(native_depth_map)

                        result_metadata = dict(
                            getattr(
                                result_candidate,
                                "metadata",
                                None,
                            )
                            or {},
                        )
                        attempt_record["model_id"] = self._resolve_backend_model_id(
                            backend_id,
                            result_metadata=result_metadata,
                            backend=backend_instance,
                        )
                        attempt_record.update(
                            self._resolve_backend_model_artifact(
                                backend_id,
                                result_metadata=result_metadata,
                                backend=backend_instance,
                            ),
                        )
                        attempt_record["source_depth_units"] = result_metadata.get(
                            "source_depth_units",
                            getattr(
                                result_candidate,
                                "depth_units",
                                None,
                            )
                            or "unknown",
                        )
                        attempt_record["output_depth_units"] = result_metadata.get(
                            "output_depth_units",
                            getattr(
                                result_candidate,
                                "depth_units",
                                None,
                            )
                            or "unknown",
                        )
                        attempt_record["output_normalization"] = result_metadata.get(
                            "output_normalization",
                            "unknown",
                        )

                        # 2b. APEX depth validity gate
                        # (plateau/saturation guardrails)
                        gate_result = self._enforce_apex_depth_validity_gate(
                            native_depth_map,
                            depth_units=getattr(
                                result_candidate,
                                "depth_units",
                                None,
                            ),
                            native_shape=current_shape,
                            artifact_shape=original_shape,
                        )

                        artifact_depth_map = native_depth_map
                        if current_shape != original_shape:
                            from PIL import Image as PILImage

                            logger.debug(
                                "Resizing depth map" " from %s back to" " original %s",
                                current_shape,
                                original_shape,
                            )
                            # Preserve raw numeric depth
                            # semantics during resize.
                            depth_pil = PILImage.fromarray(
                                native_depth_map,
                                mode="F",
                            )
                            depth_pil_resized = depth_pil.resize(
                                (
                                    original_shape[1],
                                    original_shape[0],
                                ),
                                PILImage.Resampling.BILINEAR,
                            )
                            artifact_depth_map = np.array(
                                depth_pil_resized,
                                dtype=np.float32,
                            )
                            if hasattr(result_candidate, "depth_map"):
                                result_candidate.depth_map = artifact_depth_map
                            else:
                                object.__setattr__(
                                    result_candidate,
                                    "depth",
                                    artifact_depth_map,
                                )

                        attempt_record.update(
                            {
                                "status": "success",
                                "apex_gate_passed": bool(
                                    gate_result is None
                                    or gate_result.get(
                                        "passed",
                                        False,
                                    )
                                ),
                            }
                        )
                        attempt_record["duration_s"] = time.time() - attempt_start
                        depth_attempts.append(attempt_record)

                        result = result_candidate
                        depth_map = native_depth_map
                        depth_validity_metrics = gate_result
                        selected_backend_id = backend_id
                        selected_attempt_index = attempt_slot
                        native_depth_shape = current_shape
                        break

                    except LicenseRestrictionError as license_error:
                        attempt_record.update(
                            {
                                "status": "failed",
                                "failure_kind": "license",
                                "error_code": "LICENSE_RESTRICTION",
                                "error_message": str(license_error),
                                "apex_gate_passed": False,
                                "duration_s": time.time() - attempt_start,
                            }
                        )
                        depth_attempts.append(attempt_record)
                        last_error = license_error
                        raise

                    except ApexStrictGateError as semantic_error:
                        attempt_record.update(
                            {
                                "status": "failed",
                                "failure_kind": "semantic",
                                "error_code": semantic_error.code,
                                "error_message": str(semantic_error),
                                "error_details": semantic_error.details,
                                "apex_gate_passed": False,
                                "duration_s": time.time() - attempt_start,
                            }
                        )
                        depth_attempts.append(attempt_record)
                        last_error = semantic_error

                        has_next = chain_index + 1 < len(attempt_chain)
                        if self.config.allow_semantic_fallback and has_next:
                            logger.warning(
                                "Semantic gate" " failed on" " backend=%s" " code=%s;" " attempting" " fallback.",
                                backend_id,
                                semantic_error.code,
                            )
                            continue
                        raise

                    except Exception as operational_error:
                        error_code = self._infer_operational_error_code(
                            operational_error,
                        )
                        attempt_record.update(
                            {
                                "status": "failed",
                                "failure_kind": "operational",
                                "error_code": error_code,
                                "error_message": str(operational_error),
                                "apex_gate_passed": False,
                                "duration_s": time.time() - attempt_start,
                            }
                        )
                        depth_attempts.append(attempt_record)
                        last_error = operational_error

                        has_next = chain_index + 1 < len(attempt_chain)
                        if has_next:
                            logger.warning(
                                "Operational depth" " failure on" " backend=%s" " code=%s;" " attempting" " fallback.",
                                backend_id,
                                error_code,
                            )
                            continue
                        raise

                if result is None or depth_map is None:
                    if last_error is not None:
                        raise last_error
                    raise RuntimeError(
                        "Depth inference failed" " before producing a" " result.",
                    )

                depth_runtime_s = time.time() - t0
                active_backend_metadata = self._build_backend_metadata_for_attempts(
                    selected_backend_id,
                    depth_attempts,
                    result_metadata=(
                        getattr(
                            result,
                            "metadata",
                            None,
                        )
                        or {}
                    ),
                    selected_attempt_index=(selected_attempt_index),
                )
                self._set_active_depth_state(
                    active_backend_metadata,
                    depth_attempts,
                    selected_attempt_index,
                )

                # 2c. Materials V3 Processing (if enabled)
                if self.materials_v3_engine:
                    (
                        materials_v3_result,
                        materials_v3_runtime_s,
                        enhanced_image_path,
                    ) = self._run_materials_v3_stage(
                        preprocessed_array=preprocessed_array,
                        depth_map=depth_map,
                        output_key=output_key,
                        artifact_shape=_shape_2d(result.depth),
                    )

                # 3. Write quantized depth (PNG 16-bit)
                _, _, depth_stats = atomic_write_depth_u16_png_with_stats(
                    depth_path,
                    result.depth,
                    method=self.config.depth_quantization,
                    debug_verify=self.config.verify_depth_writes,
                    compute_encoded_unique_values=self._is_apex_tier(),
                )

                # 3b. Save float depth (.npy) for high-precision PBR if enabled
                if getattr(self.config, "save_float_depth", False):
                    np.save(str(float_depth_path), result.depth)
                    logger.debug(f"Saved float depth: {float_depth_path}")

                # Capture backend metadata dynamically (ADR-019)
                _backend = self.depth_backend
                license_str = (
                    _backend.license_type.value if _backend is not None and hasattr(_backend, "license_type") else "unknown"
                )
                stats = {
                    "backend": (_backend.name if _backend is not None else "unknown"),
                    "license": license_str,
                    "non_commercial_ok": self.config.non_commercial_ok,
                    "dtype": "uint16",
                    "shape": list(result.depth.shape[:2]),
                    "native_shape": (
                        list(native_depth_shape) if native_depth_shape is not None else list(result.depth.shape[:2])
                    ),
                    "artifact_shape": list(result.depth.shape[:2]),
                    "representation": "depth",
                    "convention": "higher_is_farther",
                    "unit": (
                        result.depth_units
                        if hasattr(
                            result,
                            "depth_units",
                        )
                        else "relative"
                    ),
                    "depth_png_path": str(
                        depth_path,
                    ),
                    "depth_float_path": (
                        str(float_depth_path)
                        if getattr(
                            self.config,
                            "save_float_depth",
                            False,
                        )
                        else None
                    ),
                    "depth_float_dtype": (
                        "float32"
                        if getattr(
                            self.config,
                            "save_float_depth",
                            False,
                        )
                        else None
                    ),
                    "depth_float_shape": (
                        list(
                            result.depth.shape[:2],
                        )
                        if getattr(
                            self.config,
                            "save_float_depth",
                            False,
                        )
                        else None
                    ),  # noqa: E501
                    "canonical_depth_path": (
                        str(float_depth_path)
                        if getattr(
                            self.config,
                            "save_float_depth",
                            False,
                        )
                        else str(depth_path)
                    ),
                    "attempts": depth_attempts,
                }
                if depth_validity_metrics:
                    stats["apex_depth_validity"] = depth_validity_metrics
                    shape_context = depth_validity_metrics.get("shape_context")
                    if isinstance(shape_context, dict):
                        gate_shape = shape_context.get("gate_evaluated_shape")
                        if isinstance(gate_shape, list):
                            stats["gate_evaluated_shape"] = gate_shape

                # Merge inference provenance into depth stats
                _md = getattr(result, "metadata", None) or {}
                for _k in (
                    "requested_model_id",
                    "resolved_model_id",
                    "resolved_model_source",
                ):
                    if _k in _md:
                        stats[_k] = _md[_k]
                for _k in (
                    "source_depth_units",
                    "output_depth_units",
                    "output_normalization",
                ):
                    if _k in _md:
                        stats[_k] = _md[_k]

                # CRITICAL FIX: Use resolved backend
                # name, not config default. This ensures
                # depth.model matches what actually ran
                # (backend_selection.resolved_backend).
                # ADR-023: identity must match execution.
                resolved_backend = active_backend_metadata
                model_name = resolved_backend.resolved_backend if resolved_backend else self._model_variant.value.name

                depth_metadata = DepthMetadata(
                    model=model_name,
                    depth_path=str(depth_path),
                    runtime_seconds=depth_runtime_s,
                    scaling=depth_stats._asdict(),
                    stats=stats,
                )

                # 4. Write depth metadata JSON
                depth_metadata_path = depth_path.parent / f"{depth_path.stem}_metadata.json"
                with open(depth_metadata_path, "w", encoding="utf-8") as f:
                    dump_json(
                        {
                            "model": depth_metadata.model,
                            "depth_path": depth_metadata.depth_path,
                            "runtime_seconds": (depth_metadata.runtime_seconds),
                            "scaling": depth_metadata.scaling,
                            "stats": depth_metadata.stats,
                        },
                        f,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                logger.debug(f"Wrote depth metadata: {depth_metadata_path}")

                # 5. PBR map generation (optional)
                pbr_assets = self._generate_pbr_stage(result.depth, output_key)

            except Exception as e:
                logger.error(f"Depth failed: {e}")
                self._set_active_depth_state(
                    active_backend_metadata,
                    depth_attempts,
                    selected_attempt_index,
                )
                if isinstance(e, ApexStrictGateError):
                    logger.error(
                        "APEX strict gate failure:" " code=%s details=%s",
                        e.code,
                        e.details,
                    )
                    raise
                if self.config.depth_fallback == "fail":
                    raise
                elif self.config.depth_fallback == "skip":
                    return (
                        None,
                        0.0,
                        None,
                        None,
                        0.0,
                        None,
                        active_backend_metadata,
                        depth_attempts,
                    )
                elif self.config.depth_fallback == "v2-auto":
                    logger.info(
                        "V2 fallback mode:" " V3 failed, will" " attempt V2 with" " independent depth",
                    )
                    if depth_path.exists():
                        depth_path.unlink()
                    return (
                        None,
                        0.0,
                        None,
                        None,
                        0.0,
                        None,
                        active_backend_metadata,
                        depth_attempts,
                    )
                else:
                    raise ValueError(
                        "Unsupported" " depth_fallback" " mode:" f" {self.config.depth_fallback}",
                    ) from e
        else:
            # Depth was skipped - load from cache
            # Preserve Materials V3 metadata
            # from previous run
            existing_manifest = self._load_existing_manifest(
                manifest_path,
                purpose="cached depth reuse",
            )
            if existing_manifest is not None:
                depth_metadata = existing_manifest.depth
                pbr_assets = getattr(
                    existing_manifest,
                    "pbr_assets",
                    None,
                )
                if (
                    getattr(
                        existing_manifest,
                        "backend_selection",
                        None,
                    )
                    is not None
                ):
                    _bs = existing_manifest.backend_selection
                    assert _bs is not None
                    active_backend_metadata = _bs
                    depth_attempts = list(
                        _bs.attempts or [],
                    )
                    self._active_backend_metadata = active_backend_metadata
                    self._active_depth_attempts = depth_attempts
                    success_attempts = [a for a in depth_attempts if a.get("status") == "success"]
                    if success_attempts:
                        self._active_selected_attempt_index = int(
                            success_attempts[-1].get("attempt", 0),
                        )

                (
                    restored_materials_v3_result,
                    restored_materials_v3_runtime_s,
                    restored_enhanced_image_path,
                ) = self._restore_materials_v3_from_manifest(
                    existing_manifest,
                    output_key,
                )
                if restored_materials_v3_result:
                    logger.info(
                        "Preserving Materials" " V3 metadata from" " previous run" " (depth was cached)",
                    )
                    materials_v3_result = restored_materials_v3_result
                    materials_v3_runtime_s = restored_materials_v3_runtime_s
                    enhanced_image_path = restored_enhanced_image_path

            # PBR generation with cached depth
            # (if enabled but not previously generated)
            if self.config.generate_pbr and (
                pbr_assets is None
                or not self._verify_pbr_outputs(
                    pbr_assets,
                )
            ):
                logger.info("Generating PBR maps from cached depth...")
                try:
                    depth_data_for_pbr = self._load_cached_depth(
                        depth_path,
                        float_depth_path,
                    )
                    if depth_data_for_pbr is not None:
                        pbr_assets = self._generate_pbr_stage(
                            depth_data_for_pbr,
                            output_key,
                        )
                except Exception as pbr_error:
                    logger.warning(
                        "PBR generation from" " cache failed: %s",
                        pbr_error,
                    )

        return (
            depth_metadata,
            depth_runtime_s,
            pbr_assets,
            materials_v3_result,
            materials_v3_runtime_s,
            enhanced_image_path,
            active_backend_metadata,
            depth_attempts,
        )

    def _generate_pbr_stage(
        self,
        depth: Any,
        output_key: Path,
    ) -> Optional[dict]:
        """Generate PBR maps from depth data.

        Args:
            depth: Depth array (numpy)
            output_key: Output key for artifact naming

        Returns:
            Dictionary with PBR asset paths
                and metadata, or None
        """
        if not self.config.generate_pbr:
            return None

        try:
            logger.info("Generating PBR maps...")
            pbr_t0 = time.time()
            pbr_perf_t0 = time.perf_counter()

            # Use to_pbr_config() for consistent parameter conversion
            pbr_config = self.config.to_pbr_config()

            # Generate maps from depth
            pbr_generate_t0 = time.perf_counter()
            normal_map, roughness_map, ao_map = generate_pbr_maps(depth, config=pbr_config)
            pbr_generate_ms = round((time.perf_counter() - pbr_generate_t0) * 1000.0, 3)

            # Write PBR maps
            pbr_write_t0 = time.perf_counter()
            pbr_dir = self.output_root / "pbr"
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
                "PBR maps generated in" " %.2fs: %s",
                pbr_runtime,
                list(pbr_paths.keys()),
            )

            # Store paths for manifest
            pbr_assets = {
                "normal_path": str(pbr_paths["normal"]),
                "roughness_path": str(pbr_paths["roughness"]),
                "ao_path": str(pbr_paths["ao"]),
                "runtime_seconds": pbr_runtime,
                "timing_ms": {
                    "generate_maps": pbr_generate_ms,
                    "write_maps": pbr_write_ms,
                    "total": pbr_total_ms,
                },
                "config": {
                    "normal_strength": pbr_config.normal_strength,
                    "normal_blur_radius": pbr_config.normal_blur_radius,
                    "roughness_strength": pbr_config.roughness_strength,
                    "roughness_blur_radius": (pbr_config.roughness_blur_radius),
                    "ao_strength": pbr_config.ao_strength,
                    "ao_blur_radius": pbr_config.ao_blur_radius,
                    "ao_bias": pbr_config.ao_bias,
                },
            }
            return pbr_assets

        except Exception as pbr_error:
            logger.warning(
                "PBR generation failed" " (non-blocking): %s",
                pbr_error,
            )
            return None

    def _expected_materials_v3_enhanced_path(
        self,
        output_key: Path,
    ) -> Path:
        """Return canonical Materials V3 enhanced path."""
        temp_dir = self.output_root / "temp"
        extension = ".tif" if self.config.output_bit_depth == 16 else ".png"
        return temp_dir / f"{output_key.stem}" f"_materials_v3_enhanced" f"{extension}"

    def _segmentation_mask_artifact_path(
        self,
        output_key: Path,
    ) -> Path:
        """Return canonical segmentation mask path."""
        return segmentation_mask_artifact_path(self.segmentation_dir, output_key)

    @staticmethod
    def _resize_float_array_to_shape(
        array: np.ndarray,
        target_shape: tuple[int, int],
        *,
        resample: Any,
    ) -> np.ndarray:
        """Resize 2D or RGB float arrays deterministically."""
        from PIL import Image as PILImage

        float_array = np.asarray(array, dtype=np.float32)
        if float_array.shape[:2] == target_shape:
            return float_array.astype(np.float32, copy=False)

        if float_array.ndim == 2:
            image = PILImage.fromarray(float_array, mode="F")
            resized = image.resize(
                (target_shape[1], target_shape[0]),
                resample=resample,
            )
            return np.asarray(resized, dtype=np.float32)

        if float_array.ndim == 3 and float_array.shape[2] == 3:
            channels = [
                EnhanceOrchestrator._resize_float_array_to_shape(
                    float_array[..., channel_index],
                    target_shape,
                    resample=resample,
                )
                for channel_index in range(float_array.shape[2])
            ]
            return np.stack(channels, axis=-1).astype(np.float32)

        raise ValueError(f"Expected 2D mask or RGB image for handoff resize, got shape {float_array.shape}")

    def _align_materials_v3_handoff_payload(
        self,
        materials_v3_result: Dict[str, Any],
        *,
        artifact_shape: tuple[int, int],
        processing_shape: tuple[int, int],
    ) -> Dict[str, Any]:
        """Resize Materials V3 handoff artifacts to the depth artifact shape."""
        from PIL import Image as PILImage

        target_shape: tuple[int, int] = artifact_shape
        processing_shape_list = [processing_shape[0], processing_shape[1]]
        handoff_shape_list = [target_shape[0], target_shape[1]]

        enhanced_image = materials_v3_result.get("enhanced_image")
        if enhanced_image is not None:
            materials_v3_result["enhanced_image"] = self._resize_float_array_to_shape(
                np.asarray(enhanced_image, dtype=np.float32),
                target_shape,
                resample=PILImage.Resampling.BILINEAR,
            )

        material_masks = materials_v3_result.get("material_masks")
        if isinstance(material_masks, dict) and material_masks:
            aligned_masks: Dict[str, np.ndarray] = {}
            for material_name, mask in material_masks.items():
                aligned_masks[material_name] = self._resize_float_array_to_shape(
                    np.asarray(mask, dtype=np.float32),
                    target_shape,
                    resample=PILImage.Resampling.NEAREST,
                )
            materials_v3_result["material_masks"] = aligned_masks

        materials_v3_metadata = materials_v3_result.setdefault(
            "materials_v3_metadata",
            {},
        )
        if not isinstance(materials_v3_metadata, dict):
            materials_v3_metadata = {}
            materials_v3_result["materials_v3_metadata"] = materials_v3_metadata

        segmentation_metadata = materials_v3_metadata.get(
            "segmentation_metadata",
        )
        segmentation_metadata = dict(segmentation_metadata) if isinstance(segmentation_metadata, dict) else {}
        segmentation_metadata["processing_shape"] = processing_shape_list
        segmentation_metadata["v2_handoff_shape"] = handoff_shape_list
        materials_v3_metadata["segmentation_metadata"] = segmentation_metadata
        return materials_v3_result

    @staticmethod
    def _artifact_image_shape(image_path: Path) -> tuple[int, int]:
        """Read image artifact dimensions as (height, width)."""
        from PIL import Image as PILImage

        with PILImage.open(image_path) as image:
            return int(image.size[1]), int(image.size[0])

    def _material_mask_shape(
        self,
        materials_v3_result: Optional[dict],
    ) -> Optional[tuple[int, int]]:
        """Resolve a common material mask shape from in-memory or persisted artifacts."""
        if not isinstance(materials_v3_result, dict):
            return None

        material_masks = materials_v3_result.get("material_masks")
        if isinstance(material_masks, dict) and material_masks:
            resolved_shape: Optional[tuple[int, int]] = None
            for material_key, mask in material_masks.items():
                mask_shape = _shape_2d(np.asarray(mask))
                if resolved_shape is None:
                    resolved_shape = mask_shape
                    continue
                if resolved_shape != mask_shape:
                    raise ApexStrictGateError(
                        "APEX_MATERIAL_MASK_SHAPE_MISMATCH",
                        "APEX strict mode requires consistent Materials V3 mask shapes before V2 handoff.",
                        details={
                            "source": "material_masks",
                            "material_key": str(material_key),
                            "expected_mask_shape": list(resolved_shape),
                            "observed_mask_shape": list(mask_shape),
                        },
                    )
            return resolved_shape

        mask_artifact_path = self._persisted_material_mask_artifact_path(
            materials_v3_result,
        )
        if mask_artifact_path is None:
            return None

        with np.load(mask_artifact_path) as data:
            resolved_shape = None
            for mask_name in data.files:
                mask_shape = _shape_2d(np.asarray(data[mask_name]))
                if resolved_shape is None:
                    resolved_shape = mask_shape
                    continue
                if resolved_shape != mask_shape:
                    raise ApexStrictGateError(
                        "APEX_MATERIAL_MASK_SHAPE_MISMATCH",
                        "APEX strict mode requires consistent Materials V3 mask shapes before V2 handoff.",
                        details={
                            "source": "mask_artifact",
                            "mask_artifact_path": str(mask_artifact_path),
                            "material_key": str(mask_name),
                            "expected_mask_shape": list(resolved_shape),
                            "observed_mask_shape": list(mask_shape),
                        },
                    )
        return resolved_shape

    def _persist_material_masks_artifact(
        self,
        masks: Dict[str, np.ndarray],
        output_key: Path,
    ) -> Optional[Path]:
        """Persist material masks as artifacts."""
        if not masks:
            return None
        target_dir = self._segmentation_mask_artifact_path(
            output_key,
        ).parent
        target_dir.mkdir(parents=True, exist_ok=True)
        return self._serialize_material_masks(
            masks,
            output_key,
            target_dir,
        )

    def _run_materials_v3_stage(
        self,
        *,
        preprocessed_array: np.ndarray,
        depth_map: np.ndarray,
        output_key: Path,
        artifact_shape: Optional[tuple[int, int]] = None,
    ) -> tuple[Optional[dict], float, Optional[Path]]:
        """Run Materials V3 stage and persist artifacts."""
        if not self.materials_v3_engine:
            return None, 0.0, None

        logger.info("Running Materials V3 surface-aware finishing...")
        t_materials_start = time.time()
        materials_v3_result: Optional[dict] = None
        enhanced_image_path: Optional[Path] = None

        try:
            # APEX strict gate: enforce segmentation
            # prerequisites before Materials V3.
            self._enforce_apex_materials_gate()

            from .segmentation_backend import get_last_segmentation_runtime_metadata, segment_materials

            # Convert float32 [0,1] to uint8 [0,255]
            # for segmentation backend.
            preprocessed_uint8_for_seg = (
                np.clip(
                    preprocessed_array,
                    0,
                    1,
                )
                * 255
            ).astype(np.uint8)
            segmentation_result = {
                "materials": segment_materials(
                    preprocessed_uint8_for_seg,
                    self.config,
                    cache_dir=self.output_root / ".cache" / "material_segmentation",
                ),
            }
            segmentation_runtime = get_last_segmentation_runtime_metadata()
            if segmentation_runtime:
                segmentation_result["segmentation_metadata"] = segmentation_runtime
            self._enforce_apex_materials_gate(
                segmentation_result,
            )

            if segmentation_result.get("materials"):
                logger.info(
                    "Material segmentation:" " %d materials detected" " using %s backend",
                    len(
                        segmentation_result["materials"],
                    ),
                    self.config.material_segmentation_backend,
                )

            materials_v3_result = self.materials_v3_engine.process(
                image=preprocessed_array,
                segmentation_result=segmentation_result,
                depth_map=depth_map,
            )
            runtime_s = time.time() - t_materials_start

            if materials_v3_result:
                if artifact_shape is not None:
                    materials_v3_result = self._align_materials_v3_handoff_payload(
                        materials_v3_result,
                        artifact_shape=artifact_shape,
                        processing_shape=_shape_2d(preprocessed_array),
                    )
                self._enforce_apex_materials_pixel_ops_gate(materials_v3_result)

                material_masks = materials_v3_result.get("material_masks")
                if isinstance(material_masks, dict) and material_masks:
                    t_mask_serialize = time.perf_counter()
                    mask_artifact_path = self._persist_material_masks_artifact(
                        material_masks,
                        output_key,
                    )
                    mask_serialization_ms = round((time.perf_counter() - t_mask_serialize) * 1000.0, 3)
                    if mask_artifact_path:
                        materials_v3_metadata = materials_v3_result.setdefault(
                            "materials_v3_metadata",
                            {},
                        )
                        segmentation_metadata = materials_v3_metadata.get(
                            "segmentation_metadata",
                        )
                        segmentation_metadata = (
                            dict(
                                segmentation_metadata,
                            )
                            if isinstance(
                                segmentation_metadata,
                                dict,
                            )
                            else {}
                        )
                        segmentation_metadata["mask_artifact_path"] = str(
                            mask_artifact_path,
                        )
                        segmentation_metadata["mask_artifact_format"] = "npz"
                        segmentation_metadata["mask_artifact_shape"] = list(
                            np.asarray(next(iter(material_masks.values()))).shape[:2]
                        )
                        timing_metadata = segmentation_metadata.get("timing_ms")
                        timing_metadata = dict(timing_metadata) if isinstance(timing_metadata, dict) else {}
                        timing_metadata["mask_serialization"] = mask_serialization_ms
                        segmentation_metadata["timing_ms"] = timing_metadata
                        materials_v3_metadata["segmentation_metadata"] = segmentation_metadata

                enhanced_image = materials_v3_result.get("enhanced_image")
                if enhanced_image is not None:
                    from PIL import Image as PILImage

                    temp_dir = self.output_root / "temp"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    enhanced_image_path = self._expected_materials_v3_enhanced_path(
                        output_key,
                    )

                    if self.config.output_bit_depth == 16:
                        import tifffile

                        enhanced_uint16 = (
                            np.clip(
                                enhanced_image,
                                0,
                                1,
                            )
                            * 65535
                            + 0.5
                        ).astype(np.uint16)
                        with atomic_temp_file(
                            enhanced_image_path,
                            suffix=".tif",
                            create_file=False,
                        ) as temp_path:
                            tifffile.imwrite(
                                temp_path,
                                enhanced_uint16,
                                photometric="rgb",
                                compression="lzw",
                                metadata={
                                    "software": ("Transformation" " Portal v3"),
                                },
                            )
                        n_ops = len(
                            materials_v3_result.get(
                                "materials_v3_pixel_ops",
                                {},
                            ).get("applied", []),
                        )
                        logger.info(
                            "Materials V3 enhanced"
                            " image with %d pixel"
                            " operations - saved"
                            " to %s (16-bit TIFF)"
                            " for V2 stage",
                            n_ops,
                            enhanced_image_path,
                        )
                    else:
                        enhanced_uint8 = (
                            np.clip(
                                enhanced_image,
                                0,
                                1,
                            )
                            * 255
                        ).astype(np.uint8)
                        enhanced_image_path = atomic_write_pil_png(
                            enhanced_image_path,
                            PILImage.fromarray(
                                enhanced_uint8,
                            ),
                            optimize=True,
                        )
                        n_ops_8 = len(
                            materials_v3_result.get(
                                "materials_v3_pixel_ops",
                                {},
                            ).get("applied", []),
                        )
                        logger.info(
                            "Materials V3 enhanced"
                            " image with %d pixel"
                            " operations - saved"
                            " to %s (8-bit PNG)"
                            " for V2 stage",
                            n_ops_8,
                            enhanced_image_path,
                        )
                else:
                    logger.debug(
                        "Materials V3 did not" " return enhanced_image" ", using original" " image",
                    )

                n_applied = len(
                    materials_v3_result.get(
                        "materials_v3_pixel_ops",
                        {},
                    ).get("applied", []),
                )
                logger.info(
                    "Materials V3 completed" " in %.3fs: %d" " operations applied",
                    runtime_s,
                    n_applied,
                )

            return materials_v3_result, runtime_s, enhanced_image_path

        except ApexStrictGateError:
            # Hard-fail in apex strict mode.
            raise
        except Exception as e:
            if self._is_apex_materials_gate_enabled():
                raise ApexStrictGateError(
                    "APEX_MATERIALS_STAGE_FAILED",
                    "APEX strict mode requires successful" f" Materials V3 execution: {e}",
                    details={
                        "exception_type": type(e).__name__,
                        "exception_message": str(e),
                    },
                ) from e
            logger.warning(
                "Materials V3 processing" " failed: %s",
                e,
                exc_info=True,
            )
            return None, time.time() - t_materials_start, None

    def _ensure_apex_canonical_materials_execution(
        self,
        *,
        image_input: ImageInput,
        output_key: Path,
        depth_path: Path,
        float_depth_path: Path,
        materials_v3_result: Optional[dict],
        materials_v3_runtime_s: float,
        enhanced_image_path: Optional[Path],
    ) -> tuple[Optional[dict], float, Optional[Path]]:
        """Ensure APEX strict mode has artifacts."""
        if not self._is_apex_materials_gate_enabled():
            return (
                materials_v3_result,
                materials_v3_runtime_s,
                enhanced_image_path,
            )

        if not depth_path.exists():
            return (
                materials_v3_result,
                materials_v3_runtime_s,
                enhanced_image_path,
            )

        expected_path = self._expected_materials_v3_enhanced_path(
            output_key,
        )
        expected_path_resolved = expected_path.resolve()
        enhanced_resolved = enhanced_image_path.resolve() if enhanced_image_path else None
        has_canonical_enhanced = bool(
            enhanced_image_path and enhanced_image_path.exists() and enhanced_resolved == expected_path_resolved
        )
        has_masks = bool(
            materials_v3_result
            and materials_v3_result.get(
                "material_masks",
            )
        )

        if has_canonical_enhanced and has_masks:
            return (
                materials_v3_result,
                materials_v3_runtime_s,
                enhanced_image_path,
            )

        logger.info(
            "APEX strict mode: depth"
            " was reused but canonical"
            " Materials V3 handoff was"
            " incomplete; recomputing"
            " Materials V3 stage from"
            " cached depth.",
        )

        if self.materials_v3_engine is None:
            raise ApexStrictGateError(
                "APEX_MATERIALS_ENGINE_MISSING",
                "APEX strict mode requires" " Materials V3 engine for" " canonical cached-depth" " handoff.",
            )

        from .preprocessing import preprocess_image, validate_image_format

        validated_path = validate_image_format(image_input.path)
        preprocessed_array, _ = preprocess_image(
            validated_path,
            raw_config=self.config,
        )
        depth_for_materials = self._load_cached_depth(
            depth_path,
            float_depth_path,
        )
        if depth_for_materials is None:
            raise ApexStrictGateError(
                "APEX_MATERIALS_CACHED_DEPTH_MISSING",
                "APEX strict mode could not" " reload cached depth for" " Materials V3 recomputation.",
                details={
                    "depth_path": str(depth_path),
                    "float_depth_path": str(float_depth_path),
                },
            )

        (
            recomputed_result,
            recomputed_runtime,
            recomputed_enhanced_path,
        ) = self._run_materials_v3_stage(
            preprocessed_array=preprocessed_array,
            depth_map=np.asarray(
                depth_for_materials,
                dtype=np.float32,
            ),
            output_key=output_key,
            artifact_shape=_shape_2d(np.asarray(depth_for_materials)),
        )

        has_recomputed_masks = bool(
            recomputed_result
            and recomputed_result.get(
                "material_masks",
            )
        )
        has_recomputed_enhanced = bool(recomputed_enhanced_path and recomputed_enhanced_path.exists())
        resolved_recomputed = recomputed_enhanced_path.resolve() if recomputed_enhanced_path else None

        if not has_recomputed_masks or not has_recomputed_enhanced or resolved_recomputed != expected_path_resolved:
            raise ApexStrictGateError(
                "APEX_V2_CANONICAL_STEM_DIVERGENCE",
                "APEX strict mode could not" " establish canonical" " Materials V3 handoff" " for V2.",
                details={
                    "expected_v2_input": str(
                        expected_path,
                    ),
                    "recomputed_v2_input": (
                        str(
                            recomputed_enhanced_path,
                        )
                        if recomputed_enhanced_path
                        else None
                    ),
                    "has_material_masks": (has_recomputed_masks),
                    "has_enhanced_image": (has_recomputed_enhanced),
                },
            )

        return (
            recomputed_result,
            recomputed_runtime,
            recomputed_enhanced_path,
        )

    def _enforce_apex_v2_canonical_input_preflight(
        self,
        *,
        depth_path: Optional[Path],
        output_key: Path,
        v2_input_path: Path,
        enhanced_image_path: Optional[Path],
        materials_v3_result: Optional[dict],
    ) -> None:
        """Fail early when APEX strict violates handoff."""
        if not self.config.enable_v2 or self.v2_runner is None:
            return
        if not self._is_apex_materials_gate_enabled():
            return
        if not depth_path or not depth_path.exists():
            return

        expected_path = self._expected_materials_v3_enhanced_path(
            output_key,
        )
        expected_path_resolved = expected_path.resolve()
        actual_path_resolved = Path(v2_input_path).resolve()
        enhanced_path_resolved = enhanced_image_path.resolve() if enhanced_image_path else None
        has_masks = bool(
            materials_v3_result
            and (
                materials_v3_result.get(
                    "material_masks",
                )
                or self._persisted_material_mask_artifact_path(materials_v3_result) is not None
            )
        )

        if actual_path_resolved == expected_path_resolved and expected_path.exists() and has_masks:
            depth_shape = self._artifact_image_shape(depth_path)
            v2_input_shape = self._artifact_image_shape(Path(v2_input_path))
            mask_shape = self._material_mask_shape(materials_v3_result)
            if v2_input_shape != depth_shape or mask_shape != depth_shape:
                raise ApexStrictGateError(
                    "APEX_V2_HANDOFF_DIMENSION_DRIFT",
                    "APEX strict mode forbids V2 handoff dimension drift.",
                    details={
                        "depth_path": str(depth_path),
                        "v2_input_path": str(v2_input_path),
                        "depth_shape": list(depth_shape),
                        "v2_input_shape": list(v2_input_shape),
                        "mask_shape": list(mask_shape) if mask_shape is not None else None,
                    },
                )
            return

        raise ApexStrictGateError(
            "APEX_V2_CANONICAL_STEM_DIVERGENCE",
            "APEX strict mode forbids" " fast-path stem divergence" " before V2 handoff.",
            details={
                "expected_v2_input": str(
                    expected_path,
                ),
                "actual_v2_input": str(
                    v2_input_path,
                ),
                "enhanced_image_path": (str(enhanced_image_path) if enhanced_image_path else None),
                "enhanced_image_matches_expected": bool(enhanced_path_resolved == expected_path_resolved),
                "expected_input_exists": (expected_path.exists()),
                "has_material_masks": has_masks,
            },
        )

    def _serialize_material_masks(
        self,
        masks: Dict[str, np.ndarray],
        output_key: Path,
        output_dir: Path,
    ) -> Optional[Path]:
        """Serialize material masks to compressed NPZ file.

        File format: {output_key.stem}_materials_v3_masks.npz

        Args:
            masks: Dictionary mapping material names to binary masks
            output_key: Output key for artifact naming
            output_dir: Directory where serialized mask files are written

        Returns:
            Path to serialized .npz file, or None on failure

        Raises:
            No exceptions raised - failures are logged and return None
        """
        if not masks:
            logger.debug("No material masks to serialize")
            return None

        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Build mask file path
            mask_filename = f"{output_key.stem}_materials_v3_masks.npz"
            mask_path = output_dir / mask_filename

            # Validate mask data before serialization
            ordered_masks: Dict[str, np.ndarray] = {}
            for mat_name in sorted(masks):
                mask = masks[mat_name]
                if not isinstance(mask, np.ndarray):
                    logger.warning(
                        "Invalid mask type" " for %s: %s," " skipping",
                        mat_name,
                        type(mask),
                    )
                    return None
                if mask.dtype not in (np.float32, np.float64):
                    logger.warning(
                        "Invalid mask dtype" " for %s: %s" " (expected float32" "/float64)," " skipping",
                        mat_name,
                        mask.dtype,
                    )
                    return None
                if mask.ndim != 2:
                    logger.warning(
                        "Invalid mask shape" " for %s: %s" " (expected 2D)," " skipping",
                        mat_name,
                        mask.shape,
                    )
                    return None
                ordered_masks[mat_name] = mask

            fixed_zip_datetime = (1980, 1, 1, 0, 0, 0)
            with atomic_temp_file(
                mask_path,
                suffix=".npz",
                create_file=False,
            ) as temp_path:
                with open(temp_path, "wb") as f:
                    with zipfile.ZipFile(
                        f,
                        mode="w",
                        compression=zipfile.ZIP_DEFLATED,
                        compresslevel=9,
                        strict_timestamps=True,
                    ) as archive:
                        for mat_name, mask in ordered_masks.items():
                            payload = io.BytesIO()
                            np.lib.format.write_array(
                                payload,
                                mask,
                                allow_pickle=False,
                            )

                            # Fixed entry metadata for
                            # stable NPZ bytes.
                            zip_info = zipfile.ZipInfo(
                                filename=f"{mat_name}.npy",
                                date_time=fixed_zip_datetime,
                            )
                            zip_info.compress_type = zipfile.ZIP_DEFLATED
                            zip_info.create_system = 0
                            zip_info.external_attr = 0
                            archive.writestr(
                                zip_info,
                                payload.getvalue(),
                                compress_type=zipfile.ZIP_DEFLATED,
                                compresslevel=9,
                            )
                    f.flush()
                    os.fsync(f.fileno())

                # Check size before atomic rename
                file_size_mb = temp_path.stat().st_size / (1024 * 1024)
                if file_size_mb > 100:
                    logger.warning(
                        "Mask file large:" " %.1fMB." " Rejecting (limit" " 100MB)",
                        file_size_mb,
                    )
                    raise _MaskSerializationRejected(
                        "mask_file_too_large",
                    )

            # Verify final file exists
            if not mask_path.exists():
                logger.warning(
                    "Mask serialization" " failed: file not" " created at %s",
                    mask_path,
                )
                return None

            logger.info(
                "Serialized %d material" + " masks to %s (%.2fMB)",
                len(ordered_masks),
                mask_path.name,
                file_size_mb,
            )
            return mask_path

        except _MaskSerializationRejected:
            return None
        except Exception as e:
            logger.warning(
                "Failed to serialize" + " material masks: %s",
                e,
                exc_info=True,
            )
            return None

    def _persisted_material_mask_artifact_path(
        self,
        materials_v3_result: Optional[dict],
    ) -> Optional[Path]:
        """Return persisted mask artifact path."""
        if not isinstance(materials_v3_result, dict):
            return None

        materials_v3_metadata = materials_v3_result.get(
            "materials_v3_metadata",
        )
        if not isinstance(materials_v3_metadata, dict):
            return None

        segmentation_metadata = materials_v3_metadata.get(
            "segmentation_metadata",
        )
        if not isinstance(segmentation_metadata, dict):
            return None

        mask_artifact_path = segmentation_metadata.get("mask_artifact_path")
        if not isinstance(mask_artifact_path, str) or not mask_artifact_path:
            return None

        artifact_path = Path(mask_artifact_path)
        if artifact_path.exists():
            return artifact_path

        logger.warning(
            "Persisted mask artifact" " path missing on disk," " will fall back to temp" " serialization: %s",
            artifact_path,
        )
        return None

    def _run_v2_stage(
        self,
        image_input: ImageInput,
        depth_path: Optional[Path],
        output_key: Path,
        v2_log_path: Path,
        manifest_path: Path,
        skip_depth: bool,
        materials_v3_result: Optional[dict] = None,
    ) -> tuple[dict, float, Optional[Path]]:
        """Stage B: V2 enhancement subprocess.

        Args:
            image_input: Input image information
            depth_path: Path to depth PNG (or None if depth failed)
            output_key: Output key for artifact naming
            v2_log_path: Path for V2 subprocess log
            manifest_path: Path for manifest JSON
            skip_depth: Whether depth was skipped
            materials_v3_result: Materials V3
                result with material_masks.
                If provided, masks are
                serialized and passed to V2.

        Returns:
            Tuple of (v2_result, v2_runtime_s, v2_report_path)
        """
        # Skip V2 stage if disabled or runner not initialized
        if self.v2_runner is None or not self.config.enable_v2:
            logger.info("V2 stage disabled, skipping enhancement")
            return {"status": "skipped"}, 0.0, None

        v2_report_path = find_v2_report(self.v2_dir, output_key.name)
        skip_v2 = not self.config.force_v2 and self.should_skip_v2(
            v2_report_path,
            manifest_path,
            image_input,
            skip_depth,
        )

        if skip_v2:
            logger.info("V2 outputs valid, skipping.")
            self._enforce_v2_depth_handoff(
                depth_path=depth_path,
                v2_result=None,
                v2_report_path=v2_report_path,
            )
            previous_manifest = self._load_existing_manifest(
                manifest_path,
                purpose="V2 skip preservation",
            )
            preserved_v2_result, preserved_report_path = self._preserved_v2_result_from_manifest(
                previous_manifest,
            )
            if v2_report_path is None:
                v2_report_path = preserved_report_path
            return preserved_v2_result, 0.0, v2_report_path

        # Use persisted segmentation artifact
        # when available; otherwise serialize
        # temp masks for V2 subprocess.
        masks_path: Optional[Path] = self._persisted_material_mask_artifact_path(
            materials_v3_result,
        )
        cleanup_temp_masks = False
        if masks_path:
            logger.info(
                "Reusing persisted material" + " masks for V2: %s",
                masks_path.name,
            )
        elif materials_v3_result and materials_v3_result.get("material_masks"):
            temp_dir = self.output_root / "temp"
            masks_path = self._serialize_material_masks(
                materials_v3_result["material_masks"],
                output_key,
                temp_dir,
            )
            if masks_path:
                cleanup_temp_masks = True
                logger.info(
                    "Material masks serialized" + " for V2: %s",
                    masks_path.name,
                )
            else:
                logger.warning(
                    "Failed to serialize" " material masks, V2" " will run without" " them",
                )

        # V2 runner: Execute subprocess with optional masks
        # depth_dir=None triggers independent depth generation in V2
        try:
            v2_result = self.v2_runner.run(
                input_path=image_input.path,
                depth_dir=(self.depth_dir if (depth_path and depth_path.exists()) else None),
                output_dir=self.v2_dir,
                preset=(self.config.v2_preset or "default"),
                device=self.config.v2_device,
                upscaler_backend=self.config.v2_upscaler_backend,
                log_file=v2_log_path,
                timeout=self.config.v2_timeout,
                # Pass explicit NPZ file path
                masks_file=masks_path,
                # Pass canonical asset key for depth/report identity alignment
                asset_key=output_key.name,
                output_bit_depth=self.config.output_bit_depth,
            )
            v2_runtime_s = v2_result.get("runtime_s", 0.0)
            report_path_value = v2_result.get("report_path")
            if isinstance(report_path_value, str) and report_path_value:
                v2_report_path = Path(report_path_value)
            else:
                v2_report_path = find_v2_report(self.v2_dir, output_key.name)

            self._enforce_v2_depth_handoff(
                depth_path=depth_path,
                v2_result=v2_result,
                v2_report_path=v2_report_path,
            )

            return v2_result, v2_runtime_s, v2_report_path

        finally:
            # Clean up temporary mask file
            if cleanup_temp_masks and masks_path and masks_path.exists():
                try:
                    masks_path.unlink()
                    logger.debug(
                        "Cleaned up temporary" + " masks: %s",
                        masks_path.name,
                    )
                except Exception as cleanup_error:
                    logger.warning(
                        "Failed to clean up" + " temporary masks" + " %s: %s",
                        masks_path,
                        cleanup_error,
                    )

    def _write_manifest(
        self,
        manifest_path: Path,
        image_input: ImageInput,
        depth_metadata: Optional[Any],
        v2_result: dict,
        v2_report_path: Optional[Path],
        pbr_assets: Optional[dict],
        depth_runtime_s: float,
        v2_runtime_s: float,
        pipeline_start_time: float,
        pipeline_end_time: float,
        materials_v3_result: Optional[dict] = None,
        materials_v3_runtime_s: float = 0.0,
        backend_selection_metadata: Optional[BackendSelectionMetadata] = None,
    ) -> Optional[str]:
        """Write combined manifest with metadata.

        Args:
            manifest_path: Path for manifest
            image_input: Input image info
            depth_metadata: Depth stage metadata
            v2_result: V2 stage result dict
            v2_report_path: Path to V2 report
            pbr_assets: PBR asset metadata
            depth_runtime_s: Depth runtime
            v2_runtime_s: V2 stage runtime
            pipeline_start_time: Start time
            pipeline_end_time: End time
            materials_v3_result: V3 result
            materials_v3_runtime_s: V3 runtime
            backend_selection_metadata:
                Per-image backend provenance

        Returns:
            The normalized input SHA-256 recorded for the manifest input when
            hashing is available, or ``None`` when hashing is intentionally
            skipped or unavailable under the current hash-mode contract.
        """
        # --- PROVENANCE CAPTURE (audit-grade) ---
        # Capture provenance sidecar for RAW/TIFF inputs at ingestion point
        # This runs BEFORE manifest write to ensure we have complete metadata
        provenance_sidecar_path = manifest_path.parent / f"{manifest_path.stem}_provenance.json"

        # Determine if this is an audit-grade input (RAW or TIFF)
        # Only RAW/TIFF require exiftool for audit trail
        from .raw_loader import is_raw_file

        is_audit_input = is_raw_file(image_input.path) or image_input.path.suffix.lower() in {".tif", ".tiff"}

        try:
            # Get config fingerprint for provenance
            config_fp = self.compute_config_fingerprint()
            config_fp_str = f"sha256:{config_fp.to_sha256()}"
            from .ingest_adapter import raw_ingest_summary

            raw_summary = raw_ingest_summary(self.config)

            # Capture CLI args from environment
            # if available (set by CLI runner)
            import shlex

            cli_args = (
                shlex.split(
                    os.environ.get(
                        "TP_CLI_ARGS",
                        "",
                    ),
                )
                if "TP_CLI_ARGS" in os.environ
                else None
            )

            # Capture provenance metadata
            # For RAW/TIFF: require exiftool (audit-grade)
            # For JPG/PNG: best-effort (no exiftool requirement)
            provenance = capture_provenance(
                image_path=image_input.path,
                config_fingerprint=config_fp_str,
                cli_args=cli_args,
                # Repository root for git SHA
                repo_root=Path.cwd(),
                require_exiftool=is_audit_input,
                ingest_profile=str(
                    raw_summary.get("profile", ""),
                ),
                ingest_settings_hash=str(
                    raw_summary.get(
                        "settings_hash",
                        "",
                    ),
                ),
            )

            # Write provenance sidecar
            provenance.write_sidecar(provenance_sidecar_path)

        except ExiftoolNotFoundError as e:
            # Hard fail if exiftool missing
            # for RAW/TIFF (audit requirement)
            logger.error(
                "Provenance capture failed:" " exiftool not available" " for RAW/TIFF input",
            )
            raise RuntimeError(
                "Audit-grade provenance for"
                " RAW/TIFF requires exiftool."
                " Install with: apt-get"
                " install"
                " libimage-exiftool-perl"
                " (Ubuntu/Debian) or brew"
                " install exiftool (macOS)"
            ) from e
        except ProvenanceError as e:
            # Hard fail on provenance error
            logger.error(
                "Provenance capture" + " failed: %s",
                e,
            )
            raise RuntimeError(
                "Provenance capture" f" failed: {e}",
            ) from e
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(
                "Unexpected error during" + " provenance capture: %s",
                e,
            )
            raise RuntimeError(
                "Provenance capture failed" f" unexpectedly: {e}",
            ) from e

        previous_manifest = self._load_existing_manifest(
            manifest_path,
            purpose="manifest rewrite preservation",
        )
        previous_v2_metadata = previous_manifest.v2 if previous_manifest else None
        previous_materials_v3_metadata = previous_manifest.materials_v3 if previous_manifest else None

        # V2 metadata
        # Determine V2 I/O bit depth based on
        # emit flags and Materials V3 enhancement
        materials_v3_enhanced_image = (
            materials_v3_result.get(
                "enhanced_image",
            )
            if materials_v3_result
            else None
        )
        _emit_16 = self.config.output_bit_depth == 16
        v2_input_bit_depth = 16 if _emit_16 and materials_v3_enhanced_image is not None else 8
        v2_output_bit_depth = 16 if _emit_16 else 8
        v2_report_path_value = (
            str(v2_report_path)
            if v2_report_path
            else (
                v2_result.get(
                    "report_path",
                    "",
                )
                or (previous_v2_metadata.report_path if previous_v2_metadata and previous_v2_metadata.report_path else "")
            )
        )
        v2_output_paths = self._coerce_output_paths(
            v2_result.get("output_paths"),
        )
        v2_output_value = v2_result.get("output")
        if isinstance(v2_output_value, str) and v2_output_value and v2_output_value not in v2_output_paths:
            v2_output_paths.append(v2_output_value)
        if not v2_output_paths and previous_v2_metadata:
            v2_output_paths = self._coerce_output_paths(
                previous_v2_metadata.output_paths,
            )
        depth_handoff_state = self._extract_v2_depth_handoff_status(
            v2_result=v2_result,
            v2_report_path=v2_report_path,
        )
        v2_runtime_value = (
            v2_runtime_s
            if v2_runtime_s > 0.0
            else (
                float(previous_v2_metadata.runtime_seconds)
                if previous_v2_metadata and previous_v2_metadata.runtime_seconds is not None
                else v2_runtime_s
            )
        )
        v2_error_message = v2_result.get(
            "error",
        )
        if v2_error_message is None and previous_v2_metadata is not None:
            v2_error_message = previous_v2_metadata.error_message
        v2_status = self._normalize_v2_status(
            v2_result.get("status") or (previous_v2_metadata.status if previous_v2_metadata else "skipped"),
        )

        _strict_depth = (
            bool(depth_handoff_state)
            if depth_handoff_state is not None
            else bool(
                depth_metadata is not None
                and Path(
                    depth_metadata.depth_path,
                ).exists()
            )
        )
        v2_metadata = V2Metadata(
            preset=(self.config.v2_preset or "default"),
            strict_depth=_strict_depth,
            output_dir="v2/",
            report_path=v2_report_path_value,
            status=str(v2_status),
            runtime_seconds=v2_runtime_value,
            output_paths=(v2_output_paths or None),
            error_message=v2_error_message,
            input_bit_depth=(v2_input_bit_depth),
            output_bit_depth=(v2_output_bit_depth),
        )

        # Materials V3 metadata
        materials_v3_metadata = None
        if materials_v3_result:
            from .manifest import MaterialsV3Metadata

            raw_materials_v3_metadata = materials_v3_result.get(
                "materials_v3_metadata",
                {},
            )
            current_materials_v3_metadata = raw_materials_v3_metadata if isinstance(raw_materials_v3_metadata, dict) else {}
            response_plan = materials_v3_result.get(
                "materials_v3_response_plan",
            )
            if response_plan is None and previous_materials_v3_metadata is not None:
                response_plan = copy.deepcopy(
                    previous_materials_v3_metadata.response_plan,
                )

            pixel_ops = materials_v3_result.get(
                "materials_v3_pixel_ops",
            )
            if pixel_ops is None and previous_materials_v3_metadata is not None:
                pixel_ops = copy.deepcopy(
                    previous_materials_v3_metadata.pixel_ops,
                )

            segmentation_metadata = current_materials_v3_metadata.get(
                "segmentation_metadata",
            )
            if segmentation_metadata is None and previous_materials_v3_metadata is not None:
                segmentation_metadata = copy.deepcopy(
                    previous_materials_v3_metadata.segmentation_metadata,
                )

            materials_v3_runtime_value = (
                materials_v3_runtime_s
                if materials_v3_runtime_s > 0.0
                else (
                    float(previous_materials_v3_metadata.runtime_seconds)
                    if previous_materials_v3_metadata and previous_materials_v3_metadata.runtime_seconds is not None
                    else materials_v3_runtime_s
                )
            )

            # Determine bit depth based on
            # emit flags and enhanced image
            materials_v3_bit_depth = None
            if materials_v3_enhanced_image is not None:
                materials_v3_bit_depth = self.config.output_bit_depth

            materials_v3_metadata = MaterialsV3Metadata(
                enabled=True,
                version=current_materials_v3_metadata.get("version")
                or (previous_materials_v3_metadata.version if previous_materials_v3_metadata else "3.1"),
                response_plan=response_plan,
                pixel_ops=pixel_ops,
                segmentation_metadata=segmentation_metadata,
                runtime_seconds=materials_v3_runtime_value,
                output_bit_depth=materials_v3_bit_depth,
            )

        # Compute input hash respecting HashMode
        manifest_exists = manifest_path.exists()
        saved_hash = None
        if previous_manifest is not None and previous_manifest.input is not None:
            saved_hash = previous_manifest.input.image_sha256
        elif manifest_exists:
            try:
                m = CombinedManifest.load(manifest_path)
                if m.input:
                    saved_hash = m.input.image_sha256
            except Exception as e:
                logger.debug(
                    "Failed to load previous" " hash from manifest: %s",
                    e,
                )

        input_sha = self._compute_or_skip_hash(
            image_input.path,
            manifest_exists=manifest_exists,
            saved_hash=saved_hash,
            for_manifest_write=True,
        )

        manifest_backend_selection = backend_selection_metadata or self._active_backend_metadata or self._backend_metadata
        manifest_model_contract = self._build_run_card_model_contract(
            backend_selection=manifest_backend_selection.to_dict() if manifest_backend_selection is not None else None,
        )

        manifest = CombinedManifest(
            input=InputMetadata(
                image_path=str(image_input.path),
                image_sha256=input_sha,
                image_size_bytes=None,
                image_dimensions=None,
            ),
            depth=depth_metadata,
            v2=v2_metadata,
            materials_v3=materials_v3_metadata,
            timing=TimingMetadata(
                depth_seconds=depth_runtime_s,
                v2_seconds=v2_runtime_s,
                total_seconds=pipeline_end_time - pipeline_start_time,
                timestamp_utc=(
                    datetime.datetime.now(
                        datetime.timezone.utc,
                    ).isoformat()
                ),
            ),
            pbr_assets=pbr_assets,
            repro=ReproMetadata(
                v3_git_revision=self.v3_git,
                v2_git_revision=self.v2_git,
                environment=self.environment,
            ),
            config_fingerprint=self.compute_config_fingerprint(),
            environment=self.environment,
            # Accurate batch execution timestamps (ISO 8601 format)
            start_time=time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(pipeline_start_time),
            ),
            end_time=time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(pipeline_end_time),
            ),
            # ADR-023 Phase 3: Backend selection
            backend_selection=manifest_backend_selection,
            licensing=build_runtime_licensing_manifest(
                model_contract=manifest_model_contract,
                config=self.config,
            ),
        )
        manifest.write(manifest_path)
        return input_sha

    def enhance_image(
        self,
        image_input: ImageInput,
        input_root: Optional[Path] = None,
        _precomputed_paths: Optional[Dict[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Run full enhancement pipeline on a single image.

        Orchestrates depth computation,
        PBR generation, V2 enhancement,
        and manifest writing stages.
        Implements lazy preprocessing -
        validation and preprocessing only
        run if depth is needed (not cached).

        Args:
            image_input: Input image information
            input_root: Base directory for relative path calculation
            _precomputed_paths: Internal -
                pre-computed paths from
                parallel preprocessing

        Returns:
            Dictionary with processing status and output paths
        """
        # Capture start time for accurate timestamps
        pipeline_start_time = time.time()
        # Reset per-image active state up front so early exceptions cannot leak
        # stale attempt/backend data from a previous image.
        if hasattr(self, "_backend_metadata"):
            self._active_backend_metadata = self._backend_metadata
        else:
            self._active_backend_metadata = self._capture_backend_metadata()
        self._active_depth_attempts = []
        self._active_selected_attempt_index = None

        # PERFORMANCE FIX (#4): Use pre-computed
        # paths from parallel batch if available
        if _precomputed_paths:
            output_key = _precomputed_paths["output_key"]
            depth_path = _precomputed_paths["depth_path"]
            manifest_path = _precomputed_paths["manifest_path"]
            skip_depth = bool(
                _precomputed_paths.get(
                    "should_skip",
                    False,
                ),
            )
            logger.info(
                "Processing %s" + " (using precomputed" + " paths)...",
                output_key,
            )
        else:
            # Generate output key for consistent artifact naming
            use_xxhash = getattr(self.config, "use_xxhash", False)
            output_key = (
                make_output_key(
                    image_input.path,
                    input_root,
                    use_xxhash=use_xxhash,
                )
                if input_root
                else Path(
                    sanitize_file_stem(
                        image_input.path.stem,
                    ),
                )
            )
            logger.info(f"Processing {output_key}...")

            # Define output paths
            depth_path = self.depth_dir / output_key.parent / f"{output_key.name}_depth.png"
            manifest_path = self.manifests_dir / output_key.parent / f"{output_key.name}_combined.json"

            # Determine skip logic
            skip_depth = not self.config.force_depth and self.should_skip_depth(
                depth_path,
                manifest_path,
                image_input,
            )

        # Always compute these paths (not part of skip logic)
        float_depth_path = self.depth_dir / output_key.parent / f"{output_key.name}_depth.npy"
        active_batch_id = getattr(self, "_active_batch_id", None)
        v2_log_path = (
            self.logs_dir
            / output_key.parent
            / _v2_log_filename(
                output_key.name,
                active_batch_id,
            )
        )

        # Ensure output directories exist
        for p in [depth_path, manifest_path, v2_log_path]:
            p.parent.mkdir(parents=True, exist_ok=True)

        # Determine skip logic BEFORE preprocessing (lazy evaluation)
        # (skip_depth already computed above for both paths)

        # --- STAGE A: DEPTH COMPUTATION ---
        (
            depth_metadata,
            depth_runtime_s,
            pbr_assets,
            materials_v3_result,
            materials_v3_runtime_s,
            enhanced_image_path,
            backend_selection_metadata,
            depth_attempts,
        ) = self._compute_depth_stage(
            image_input=image_input,
            output_key=output_key,
            depth_path=depth_path,
            float_depth_path=float_depth_path,
            manifest_path=manifest_path,
            skip_depth=skip_depth,
        )

        # Handle depth stage failures that return early
        if depth_metadata is None and depth_runtime_s == 0.0 and pbr_assets is None:
            if self.config.depth_fallback == "skip":
                return {
                    "status": "skipped",
                    "reason": "Depth computation failed",
                    "image": str(image_input.path),
                    "backend": (backend_selection_metadata.resolved_backend if backend_selection_metadata else None),
                    "fallback_used": bool(
                        backend_selection_metadata and (backend_selection_metadata.resolution_status != "success")
                    ),
                    "attempts": depth_attempts,
                    "selected_attempt_index": None,
                    "quality_gate": None,
                }

        # Runtime invariant checks for attempt-selection consistency.
        selected_attempt_index = getattr(
            self,
            "_active_selected_attempt_index",
            None,
        )
        if depth_metadata is not None and depth_attempts:
            if selected_attempt_index is None or selected_attempt_index < 0 or selected_attempt_index >= len(depth_attempts):
                raise RuntimeError(
                    "Depth attempt invariant" " violated:" " selected_attempt_index" " is out of range for" " attempt history."
                )
            _sai = int(selected_attempt_index)
            selected_attempt_backend = depth_attempts[_sai].get("backend")
            resolved_backend = backend_selection_metadata.resolved_backend if backend_selection_metadata else None
            if selected_attempt_backend != resolved_backend:
                raise RuntimeError(
                    "Depth attempt invariant" " violated: selected" " attempt backend does" " not match resolved" " backend."
                )

        (
            materials_v3_result,
            materials_v3_runtime_s,
            enhanced_image_path,
        ) = self._ensure_apex_canonical_materials_execution(
            image_input=image_input,
            output_key=output_key,
            depth_path=depth_path,
            float_depth_path=float_depth_path,
            materials_v3_result=materials_v3_result,
            materials_v3_runtime_s=materials_v3_runtime_s,
            enhanced_image_path=enhanced_image_path,
        )

        # --- STAGE B: V2 ENHANCEMENT ---
        # Use enhanced image from Materials V3
        # if available, otherwise use original
        v2_input_path = enhanced_image_path if enhanced_image_path else image_input.path
        if enhanced_image_path:
            logger.info(
                "V2 stage using Materials" + " V3 enhanced image: %s",
                enhanced_image_path,
            )

        self._enforce_apex_v2_canonical_input_preflight(
            depth_path=depth_path if depth_metadata else None,
            output_key=output_key,
            v2_input_path=v2_input_path,
            enhanced_image_path=enhanced_image_path,
            materials_v3_result=materials_v3_result,
        )

        v2_result, v2_runtime_s, v2_report_path = self._run_v2_stage(
            image_input=(ImageInput(path=v2_input_path) if enhanced_image_path else image_input),
            depth_path=(depth_path if depth_metadata else None),
            output_key=output_key,
            v2_log_path=v2_log_path,
            manifest_path=manifest_path,
            skip_depth=skip_depth,
            materials_v3_result=materials_v3_result,
        )
        v2_output_path = v2_result.get("output") if isinstance(v2_result, dict) else None
        if not isinstance(v2_output_path, str) or not v2_output_path:
            v2_output_path = None
        if not v2_report_path:
            report_path_value = v2_result.get("report_path") if isinstance(v2_result, dict) else None
            if isinstance(report_path_value, str) and report_path_value:
                v2_report_path = Path(report_path_value)

        vlm_captioning_result = self._run_vlm_captioning(
            image_input=image_input,
            output_key=output_key,
        )

        # Capture end time for accurate timestamps
        pipeline_end_time = time.time()

        # Clean up temporary enhanced image file if it was created, unless
        # the operator has asked to keep intermediates for bisection.
        if enhanced_image_path and enhanced_image_path.exists() and not getattr(self.config, "keep_intermediates", False):
            try:
                enhanced_image_path.unlink()
                logger.debug(
                    "Cleaned up temporary" + " enhanced image: %s",
                    enhanced_image_path,
                )
            except Exception as e:
                logger.warning(
                    "Failed to clean up" + " temporary enhanced" + " image: %s",
                    e,
                )
        elif enhanced_image_path and enhanced_image_path.exists():
            # DEBUG (not INFO) and basename-only to avoid leaking the absolute
            # filesystem layout into batch logs. The full path is already
            # recoverable from <output_root>/temp/ + asset stem.
            logger.debug(
                "keep_intermediates=True; preserving Materials V3 intermediate: %s",
                enhanced_image_path.name,
            )

        # --- MANIFEST WRITING ---
        input_sha = self._write_manifest(
            manifest_path=manifest_path,
            image_input=image_input,
            depth_metadata=depth_metadata,
            v2_result=v2_result,
            v2_report_path=v2_report_path,
            pbr_assets=pbr_assets,
            depth_runtime_s=depth_runtime_s,
            v2_runtime_s=v2_runtime_s,
            pipeline_start_time=pipeline_start_time,
            pipeline_end_time=pipeline_end_time,
            materials_v3_result=materials_v3_result,
            materials_v3_runtime_s=materials_v3_runtime_s,
            backend_selection_metadata=backend_selection_metadata,
        )

        segmentation_metadata = self._extract_run_card_segmentation_metadata(
            materials_v3_result,
        )
        if segmentation_metadata is not None:
            self._active_run_card_segmentation_metadata[str(manifest_path)] = copy.deepcopy(
                segmentation_metadata,
            )

        segmentation_mask_path: Optional[str] = None
        if isinstance(segmentation_metadata, dict):
            mask_artifact_path = segmentation_metadata.get(
                "mask_artifact_path",
            )
            if (
                isinstance(
                    mask_artifact_path,
                    str,
                )
                and mask_artifact_path
            ):
                segmentation_mask_path = mask_artifact_path

        return {
            "status": "ok",
            "image": str(image_input.path),
            "input_sha256": input_sha,
            "backend": (backend_selection_metadata.resolved_backend if backend_selection_metadata else None),
            "fallback_used": bool(backend_selection_metadata and (backend_selection_metadata.resolution_status != "success")),
            "model_id": (backend_selection_metadata.model_id if backend_selection_metadata else None),
            "device": (backend_selection_metadata.device if backend_selection_metadata else None),
            "attempts": depth_attempts,
            "selected_attempt_index": selected_attempt_index,
            "depth_path": (str(depth_path) if depth_metadata else None),
            "depth_float_path": (
                str(float_depth_path)
                if getattr(
                    self.config,
                    "save_float_depth",
                    False,
                )
                and float_depth_path.exists()
                else None
            ),
            "manifest": str(manifest_path),
            "v2_log_path": (str(v2_log_path) if v2_log_path.exists() else None),
            "v2_report_path": (str(v2_report_path) if v2_report_path else None),
            "v2_output_path": v2_output_path,
            "segmentation_mask_path": segmentation_mask_path,
            "vlm_captioning_status": (
                vlm_captioning_result.get("captioning_status") if isinstance(vlm_captioning_result, dict) else None
            ),
            "vlm_caption_proxy_path": (
                vlm_captioning_result.get("proxy_path") if isinstance(vlm_captioning_result, dict) else None
            ),
            "vlm_caption_sidecar_path": (
                vlm_captioning_result.get("sidecar_path") if isinstance(vlm_captioning_result, dict) else None
            ),
            "vlm_caption_raw_path": (
                vlm_captioning_result.get("raw_path") if isinstance(vlm_captioning_result, dict) else None
            ),
            "runtime_s": (pipeline_end_time - pipeline_start_time),
            "quality_gate": build_quality_gate_report(
                getattr(depth_metadata, "stats", {}).get("apex_depth_validity") if depth_metadata is not None else None
            ),
        }

    def _resolve_vlm_captioning_model_path(self, selector: str) -> tuple[Path, Optional[str], str]:
        normalized = str(selector or "default").strip()
        role = normalized.lower()
        if role == "review":
            env_path = os.getenv("TP_FASTVLM_REVIEW_MODEL")
        elif role in {"default", "smoke"}:
            env_path = os.getenv("TP_FASTVLM_MODEL") if role == "default" else None
        else:
            env_path = None
        model_path = resolve_fastvlm_runtime_path(env_path) if env_path else resolve_fastvlm_model_path(normalized)
        model_role = role if role in FASTVLM_MODEL_ROLES else None
        return model_path, model_role, resolve_fastvlm_model_id(model_path, model_role)

    def _run_vlm_captioning(
        self,
        *,
        image_input: ImageInput,
        output_key: Path,
    ) -> Optional[Dict[str, Any]]:
        """Generate optional advisory FastVLM caption sidecar artifacts."""
        if not bool(getattr(self.config, "vlm_captioning_enabled", False)):
            return None

        backend = str(getattr(self.config, "vlm_captioning_backend", "fastvlm") or "fastvlm").strip().lower()
        if backend != "fastvlm":
            logger.warning("Skipping VLM captioning because backend=%r is unsupported.", backend)
            return {
                "captioning_status": {
                    "enabled": True,
                    "backend": backend,
                    "role": "advisory",
                    "status": "unsupported_backend",
                    "sidecar_count": 0,
                    "failed_count": 1,
                    "used_for_quality_gate": False,
                }
            }

        proxy_format = str(getattr(self.config, "vlm_captioning_proxy_format", "png") or "png").strip().lower()
        proxy_suffix = "jpg" if proxy_format == "jpeg" else "png"
        caption_dir = self.output_root / "captioning" / output_key.parent
        caption_dir.mkdir(parents=True, exist_ok=True)
        raw_path = caption_dir / f"{output_key.name}.vlm_captioning.raw.txt"
        sidecar_path = caption_dir / f"{output_key.name}.vlm_captioning.sidecar.json"
        proxy_path: Optional[Path] = None
        selector = str(getattr(self.config, "vlm_captioning_model", "default") or "default").strip()
        model_path, model_role, model_id = self._resolve_vlm_captioning_model_path(selector)

        base_status: Dict[str, Any] = {
            "enabled": True,
            "backend": "fastvlm",
            "model_role": model_role or "custom",
            "model_id": model_id,
            "role": "advisory",
            "sidecar_count": 0,
            "failed_count": 1,
            "used_for_quality_gate": False,
        }

        try:
            proxy = build_vlm_image_proxy(
                image_input.path,
                caption_dir,
                max_side_px=int(getattr(self.config, "vlm_captioning_max_side_px", 1600) or 1600),
                format=cast(Any, proxy_format),
                output_name=f"{output_key.name}_proxy.{proxy_suffix}",
            )
            proxy_path = proxy.proxy_path
        except Exception as exc:
            logger.warning("VLM captioning proxy generation failed for %s: %s", image_input.path, exc)
            raw_path.write_text(f"VLM captioning proxy generation failed: {exc}\n", encoding="utf-8")
            status = {
                **base_status,
                "status": "proxy_error",
                "error": str(exc),
            }
            return {
                "captioning_status": status,
                "raw_path": str(raw_path),
            }

        runtime_root = default_fastvlm_runtime_root()
        python_config = getattr(self.config, "fastvlm_python_executable", None) or os.getenv("TP_FASTVLM_PYTHON")
        mlx_vlm_config = getattr(self.config, "fastvlm_mlx_vlm_dir", None) or os.getenv("TP_FASTVLM_MLX_VLM_DIR")
        python_path = Path(str(python_config).strip()) if python_config else runtime_root / ".venv-fastvlm/bin/python"
        mlx_vlm_dir = resolve_fastvlm_runtime_path(str(mlx_vlm_config)) if mlx_vlm_config else runtime_root / "mlx-vlm"
        max_tokens = int(os.getenv("TP_FASTVLM_MAX_TOKENS", "120"))
        temperature = float(os.getenv("TP_FASTVLM_TEMPERATURE", "0.0"))
        timeout_seconds = int(getattr(self.config, "fastvlm_timeout_seconds", 180) or 180)
        runtime_config = FastVLMRuntimeConfig(
            enabled=True,
            python_path=python_path,
            mlx_vlm_dir=mlx_vlm_dir,
            model_path=model_path,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
        )
        runtime_result = run_fastvlm_caption(runtime_config, proxy.proxy_path, model_role=model_role)
        raw_text = runtime_result.raw_stdout or runtime_result.raw_stderr or runtime_result.error or ""
        raw_path.write_text(raw_text, encoding="utf-8")
        sidecar_payload = build_fastvlm_sidecar(
            enabled=True,
            model_path=model_path,
            image_proxy=proxy,
            runtime_result=runtime_result,
            model_role=model_role,
            model_id=model_id,
        )
        sidecar_path.write_text(dumps_sidecar(sidecar_payload), encoding="utf-8")
        failed = not runtime_result.success
        status = {
            **base_status,
            "status": runtime_result.status,
            "sidecar_count": 1,
            "failed_count": 1 if failed else 0,
            "validated": bool(runtime_result.caption_parse.validated),
        }
        if runtime_result.error:
            status["error"] = runtime_result.error
        return {
            "captioning_status": status,
            "proxy_path": str(proxy_path),
            "sidecar_path": str(sidecar_path),
            "raw_path": str(raw_path),
        }

    def _verify_pbr_outputs(
        self,
        pbr_assets: Optional[Dict[str, Any]],
    ) -> bool:
        """Verify that all PBR output files exist on disk.

        Args:
            pbr_assets: Dictionary containing PBR output paths

        Returns:
            True if all outputs exist, False otherwise
        """
        if not pbr_assets:
            return False

        for key, value in pbr_assets.items():
            if isinstance(value, str) and key.endswith("_path"):
                if not os.path.exists(value):
                    logger.debug(f"PBR output missing: {value}")
                    return False
        return True

    def _is_apex_materials_gate_enabled(self) -> bool:
        """Check if apex + Materials V3 gate is on."""
        return str(
            getattr(
                self.config,
                "quality_tier",
                "",
            ),
        ).lower() == "apex" and bool(
            getattr(
                self.config,
                "enable_materials_v3",
                False,
            ),
        )

    def _is_apex_tier(self) -> bool:
        """Return True when APEX quality tier is active."""
        return (
            str(
                getattr(
                    self.config,
                    "quality_tier",
                    "",
                ),
            ).lower()
            == "apex"
        )

    def _normalize_depth_for_gate(
        self,
        depth_map: np.ndarray,
        depth_units: Optional[str] = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        """Normalize depth map to [0,1] for APEX gate metrics only."""
        raw = np.asarray(depth_map, dtype=np.float32)
        finite_mask = np.isfinite(raw)
        finite_count = int(finite_mask.sum())
        total_count = int(raw.size)
        finite_pct = float(finite_mask.mean()) if total_count else 0.0
        unit_hint = str(depth_units or "").strip().lower()
        is_relative_unit = unit_hint in {"relative", "relative_0_1"}

        if finite_count == 0:
            return np.zeros_like(raw, dtype=np.float32), {
                "finite_count": 0,
                "total_count": total_count,
                "finite_pct": finite_pct,
                "p1": None,
                "p99": None,
                "scaled": False,
                "raw_min": None,
                "raw_max": None,
                "mode": "empty",
            }

        vals = raw[finite_mask]
        p1 = float(np.percentile(vals, 1.0))
        p99 = float(np.percentile(vals, 99.0))
        raw_min = float(np.min(vals))
        raw_max = float(np.max(vals))

        # Preserve existing DA3/DA2 semantics for explicitly-relative depths.
        # If units are unknown, treat [0,1] data as already gate-normalized.
        relative_like_range = raw_min >= -1e-3 and raw_max <= 1.0 + 1e-3
        if is_relative_unit or (not unit_hint and relative_like_range):
            gate = np.clip(raw, 0.0, 1.0).astype(np.float32, copy=False)
            gate[~finite_mask] = 0.0
            return gate, {
                "finite_count": finite_count,
                "total_count": total_count,
                "finite_pct": finite_pct,
                "p1": p1,
                "p99": p99,
                "scaled": False,
                "raw_min": raw_min,
                "raw_max": raw_max,
                "mode": "identity_relative",
            }

        if not np.isfinite(p1) or not np.isfinite(p99) or p99 <= p1:
            return np.zeros_like(raw, dtype=np.float32), {
                "finite_count": finite_count,
                "total_count": total_count,
                "finite_pct": finite_pct,
                "p1": p1,
                "p99": p99,
                "scaled": False,
                "raw_min": raw_min,
                "raw_max": raw_max,
                "mode": "invalid_percentiles",
            }

        gate = np.clip(
            (raw - p1) / (p99 - p1),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        gate[~finite_mask] = 0.0
        return gate, {
            "finite_count": finite_count,
            "total_count": total_count,
            "finite_pct": finite_pct,
            "p1": p1,
            "p99": p99,
            "scaled": True,
            "raw_min": raw_min,
            "raw_max": raw_max,
            "mode": "percentile_1_99",
        }

    def _compute_depth_validity_metrics(
        self,
        depth_map: np.ndarray,
        depth_units: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute depth validity metrics used by APEX quality gate.

        Metrics are computed on a normalized `gate_depth` representation, while
        preserving raw-unit diagnostics in the payload.
        """
        raw_depth = np.asarray(depth_map, dtype=np.float32)
        gate_depth, normalization = self._normalize_depth_for_gate(
            raw_depth,
            depth_units=depth_units,
        )

        depth = gate_depth
        raw_finite_mask = np.isfinite(raw_depth)
        finite_pct = float(raw_finite_mask.mean()) if raw_finite_mask.size else 0.0

        if not raw_finite_mask.any():
            return {
                "finite_pct": finite_pct,
                "source_unit": depth_units or "unknown",
                "gate_unit": "relative_0_1",
                "p75": None,
                "p95": None,
                "upper_iqr": None,
                "saturation_high_fraction": None,
                "saturation_low_fraction": None,
                "gradient_energy": None,
                "unique_hist_bins": None,
                "raw_p75": None,
                "raw_p95": None,
                "raw_upper_iqr": None,
                "gate_normalization": normalization,
            }

        vals = depth[raw_finite_mask]
        p75 = float(np.percentile(vals, 75.0))
        p95 = float(np.percentile(vals, 95.0))
        upper_iqr = p95 - p75

        raw_vals = raw_depth[raw_finite_mask]
        raw_p75 = float(np.percentile(raw_vals, 75.0)) if raw_vals.size else None
        raw_p95 = float(np.percentile(raw_vals, 95.0)) if raw_vals.size else None
        raw_upper_iqr = (raw_p95 - raw_p75) if (raw_p75 is not None and raw_p95 is not None) else None

        saturation_high_value = float(
            getattr(
                self.config,
                "apex_depth_saturation_high_value",
                0.999,
            ),
        )
        saturation_low_value = float(
            getattr(
                self.config,
                "apex_depth_saturation_low_value",
                0.001,
            ),
        )
        saturation_high_fraction = float(
            (vals >= saturation_high_value).mean(),
        )
        saturation_low_fraction = float(
            (vals <= saturation_low_value).mean(),
        )

        grad_y, grad_x = np.gradient(depth)
        grad_mag = np.hypot(grad_x, grad_y)
        grad_mag = grad_mag[np.isfinite(grad_mag)]
        gradient_energy = float(np.mean(np.abs(grad_mag))) if grad_mag.size else 0.0

        hist_bins = int(getattr(self.config, "apex_depth_hist_bins", 64))
        hist, _ = np.histogram(
            np.clip(vals, 0.0, 1.0),
            bins=hist_bins,
            range=(0.0, 1.0),
        )
        unique_hist_bins = int(np.count_nonzero(hist))

        return {
            "finite_pct": finite_pct,
            "source_unit": depth_units or "unknown",
            "gate_unit": "relative_0_1",
            "p75": p75,
            "p95": p95,
            "upper_iqr": upper_iqr,
            "saturation_high_fraction": saturation_high_fraction,
            "saturation_low_fraction": saturation_low_fraction,
            "gradient_energy": gradient_energy,
            "unique_hist_bins": unique_hist_bins,
            "raw_p75": raw_p75,
            "raw_p95": raw_p95,
            "raw_upper_iqr": raw_upper_iqr,
            "gate_normalization": normalization,
        }

    @staticmethod
    def _shape_list(shape: Any) -> Optional[List[int]]:
        """Normalize an arbitrary HxW shape payload to a JSON-safe list."""
        if isinstance(shape, (tuple, list)) and len(shape) >= 2:
            try:
                return [int(shape[0]), int(shape[1])]
            except (TypeError, ValueError):
                return None
        return None

    def _build_apex_depth_shape_context(
        self,
        *,
        gate_evaluated_shape: Any,
        native_shape: Any,
        artifact_shape: Any,
    ) -> Dict[str, Optional[List[int]]]:
        """Build explicit shape context for APEX gate telemetry."""
        gate_shape = self._shape_list(gate_evaluated_shape)
        native_shape_list = self._shape_list(native_shape) or gate_shape
        artifact_shape_list = self._shape_list(artifact_shape) or native_shape_list
        return {
            "gate_evaluated_shape": gate_shape,
            "native_shape": native_shape_list,
            "artifact_shape": artifact_shape_list,
        }

    def _enforce_apex_depth_validity_gate(
        self,
        depth_map: np.ndarray,
        depth_units: Optional[str] = None,
        *,
        native_shape: Optional[Any] = None,
        artifact_shape: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """APEX-only depth quality gate."""
        if not self._is_apex_tier():
            return None

        metrics = self._compute_depth_validity_metrics(
            depth_map,
            depth_units=depth_units,
        )
        shape_context = self._build_apex_depth_shape_context(
            gate_evaluated_shape=np.asarray(depth_map).shape[:2],
            native_shape=native_shape or np.asarray(depth_map).shape[:2],
            artifact_shape=artifact_shape or native_shape or np.asarray(depth_map).shape[:2],
        )

        _cfg = self.config
        thresholds = {
            "finite_pct_min": float(
                getattr(
                    _cfg,
                    "apex_depth_min_finite_pct",
                    0.999,
                ),
            ),
            "upper_iqr_min": float(
                getattr(
                    _cfg,
                    "apex_depth_min_upper_iqr",
                    1e-4,
                ),
            ),
            "saturation_high_fraction_max": float(
                getattr(
                    _cfg,
                    "apex_depth_max_high_" + "saturation_fraction",
                    0.02,
                ),
            ),
            "saturation_low_fraction_max": float(
                getattr(
                    _cfg,
                    "apex_depth_max_low_" + "saturation_fraction",
                    0.02,
                ),
            ),
            "gradient_energy_warning_min": float(
                getattr(
                    _cfg,
                    "apex_depth_min_" + "gradient_energy",
                    5e-4,
                ),
            ),
            "saturation_high_value": float(
                getattr(
                    _cfg,
                    "apex_depth_saturation" + "_high_value",
                    0.999,
                ),
            ),
            "saturation_low_value": float(
                getattr(
                    _cfg,
                    "apex_depth_saturation" + "_low_value",
                    0.001,
                ),
            ),
            "hist_bins": int(
                getattr(
                    _cfg,
                    "apex_depth_hist_bins",
                    64,
                ),
            ),
        }
        comparison_epsilon = max(
            float(
                getattr(
                    _cfg,
                    "apex_depth_threshold_epsilon",
                    1e-6,
                ),
            ),
            0.0,
        )
        scaled_saturation_margin = max(
            float(
                getattr(
                    _cfg,
                    "apex_depth_scaled_saturation_margin",
                    0.0,
                ),
            ),
            0.0,
        )
        low_saturation_warning_band = max(
            float(
                getattr(
                    _cfg,
                    "apex_depth_low_saturation_warning_band",
                    0.0,
                ),
            ),
            0.0,
        )
        gate_normalization = metrics.get("gate_normalization")
        normalization_scaled = bool(
            isinstance(gate_normalization, dict) and gate_normalization.get("scaled"),
        )
        percentile_scaled_gate = bool(
            isinstance(gate_normalization, dict)
            and gate_normalization.get("scaled") is True
            and gate_normalization.get("mode") == "percentile_1_99"
        )
        effective_high_saturation_max = thresholds["saturation_high_fraction_max"] + (
            scaled_saturation_margin if normalization_scaled else 0.0
        )
        effective_low_saturation_max = thresholds["saturation_low_fraction_max"] + (
            scaled_saturation_margin if normalization_scaled else 0.0
        )
        effective_low_saturation_warning_max = effective_low_saturation_max + (
            low_saturation_warning_band if percentile_scaled_gate else 0.0
        )
        thresholds["comparison_epsilon"] = comparison_epsilon
        thresholds["scaled_saturation_margin"] = scaled_saturation_margin if normalization_scaled else 0.0
        thresholds["saturation_high_fraction_max_effective"] = effective_high_saturation_max
        thresholds["saturation_low_fraction_max_effective"] = effective_low_saturation_max
        thresholds["saturation_low_fraction_warning_band"] = low_saturation_warning_band if percentile_scaled_gate else 0.0
        thresholds["saturation_low_fraction_warning_max_effective"] = (
            effective_low_saturation_warning_max if percentile_scaled_gate else effective_low_saturation_max
        )

        finite_fail = float(metrics.get("finite_pct") or 0.0) < (thresholds["finite_pct_min"] - comparison_epsilon)
        plateau_fail = (metrics.get("upper_iqr") is None) or (float(metrics["upper_iqr"]) <= thresholds["upper_iqr_min"])
        high_saturation_fail = (
            metrics.get(
                "saturation_high_fraction",
            )
            is None
        ) or (float(metrics["saturation_high_fraction"]) > (effective_high_saturation_max + comparison_epsilon))
        low_saturation_fail = (
            metrics.get(
                "saturation_low_fraction",
            )
            is None
        ) or (float(metrics["saturation_low_fraction"]) > (effective_low_saturation_max + comparison_epsilon))
        low_gradient = (metrics.get("gradient_energy") is None) or (
            float(metrics["gradient_energy"]) < (thresholds["gradient_energy_warning_min"] - comparison_epsilon)
        )

        failure_codes: List[str] = []
        if finite_fail:
            failure_codes.append("APEX_DEPTH_NONFINITE")
        if plateau_fail:
            failure_codes.append("APEX_DEPTH_PLATEAU")
        if high_saturation_fail:
            failure_codes.append("APEX_DEPTH_SATURATION_HIGH")
        if low_saturation_fail:
            failure_codes.append("APEX_DEPTH_SATURATION_LOW")

        # Stable ordering for deterministic payloads across runs/processes.
        failure_codes = sorted(failure_codes)

        warnings: List[str] = []
        demoted_failure_codes: List[str] = []
        if low_gradient:
            warnings.append("APEX_DEPTH_GRADIENT_LOW")

        low_saturation_metric = metrics.get("saturation_low_fraction")
        low_saturation_value = float(low_saturation_metric) if isinstance(low_saturation_metric, (int, float)) else None
        borderline_low_saturation = (
            percentile_scaled_gate
            and low_saturation_fail
            and low_saturation_value is not None
            and low_saturation_value <= (effective_low_saturation_warning_max + comparison_epsilon)
        )
        can_demote_low_saturation = (
            failure_codes == ["APEX_DEPTH_SATURATION_LOW"] and borderline_low_saturation and not low_gradient
        )

        if can_demote_low_saturation:
            demoted_failure_codes = ["APEX_DEPTH_SATURATION_LOW"]
            warnings.append("APEX_DEPTH_SATURATION_LOW_BORDERLINE")
            logger.warning(
                "APEX depth validity warning: borderline low saturation demoted "
                "to warning (value=%.10f, limit=%.10f, metrics=%s, thresholds=%s)",
                low_saturation_value,
                effective_low_saturation_warning_max,
                metrics,
                thresholds,
            )
            failure_codes = []

        warnings = sorted(warnings)
        demoted_failure_codes = sorted(demoted_failure_codes)

        if failure_codes:
            details = {
                "passed": False,
                "failure_codes": failure_codes,
                "warnings": warnings,
                "demoted_failure_codes": demoted_failure_codes,
                "metrics": metrics,
                "thresholds": thresholds,
                "shape_context": shape_context,
            }
            raise ApexStrictGateError(
                failure_codes[0] if len(failure_codes) == 1 else "APEX_DEPTH_INVALID",
                "APEX depth validity gate" " failed: " + ", ".join(failure_codes),
                details=details,
            )

        if low_gradient:
            logger.warning(
                "APEX depth validity" " warning: low gradient" " energy (metrics=%s," " thresholds=%s)",
                metrics,
                thresholds,
            )

        return {
            "passed": True,
            "failure_codes": [],
            "warnings": warnings,
            "demoted_failure_codes": demoted_failure_codes,
            "metrics": metrics,
            "thresholds": thresholds,
            "shape_context": shape_context,
        }

    def _enforce_apex_materials_gate(
        self,
        segmentation_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Enforce APEX strict Materials V3 gate.

        Gate policy (apex + materials_v3 only):
        - Segmentation must be explicitly enabled
        - Backend must not be stub
        - Strict backend mode must be on (no silent fallback)
        - Segmentation output must contain at least one material mask
        """
        if not self._is_apex_materials_gate_enabled():
            return

        if not bool(
            getattr(
                self.config,
                "enable_material_segmentation",
                False,
            ),
        ):
            raise ApexStrictGateError(
                "APEX_MATERIALS_SEGMENTATION_DISABLED",
                "APEX strict gate violated:"
                " Materials V3 in apex tier"
                " requires segmentation"
                " enabled (set"
                " --enable-segmentation"
                " on).",
            )

        backend_name = str(
            getattr(
                self.config,
                "material_segmentation_backend",
                "stub",
            ),
        ).lower()
        if backend_name == "stub":
            raise ApexStrictGateError(
                "APEX_MATERIALS_STUB_BACKEND",
                "APEX strict gate violated:"
                " Materials V3 in apex tier"
                " cannot use stub"
                " segmentation backend"
                " (set"
                " --segmentation-backend"
                " efficientsam or sam2).",
            )

        if not bool(getattr(self.config, "strict_backend", False)):
            raise ApexStrictGateError(
                "APEX_MATERIALS_STRICT_SEGMENTATION_REQUIRED",
                "APEX strict gate violated:"
                " Materials V3 in apex tier"
                " requires strict"
                " segmentation backend mode"
                " (set"
                " --strict-segmentation).",
            )

        if segmentation_result is None:
            return

        materials = segmentation_result.get("materials", {}) if isinstance(segmentation_result, dict) else {}
        if not materials:
            raise ApexStrictGateError(
                "APEX_MATERIALS_EMPTY_SEGMENTATION",
                "APEX strict gate violated:"
                " segmentation produced no"
                " material masks; failing"
                " instead of continuing"
                " with 0 Materials V3"
                " operations.",
            )

    @staticmethod
    def _pixel_ops_blocked_reasons(pixel_ops: Mapping[str, Any]) -> Dict[str, int]:
        histogram: Dict[str, int] = {}
        blocked = pixel_ops.get("blocked")
        if not isinstance(blocked, list):
            return histogram
        for entry in blocked:
            if not isinstance(entry, dict):
                continue
            reasons = entry.get("blocked_by")
            if isinstance(reasons, list) and reasons:
                for reason in reasons:
                    reason_key = str(reason)
                    histogram[reason_key] = histogram.get(reason_key, 0) + 1
            else:
                reason_key = str(entry.get("reason") or "unknown")
                histogram[reason_key] = histogram.get(reason_key, 0) + 1
        return histogram

    def _enforce_apex_materials_pixel_ops_gate(self, materials_v3_result: Optional[Dict[str, Any]]) -> None:
        """Fail closed when APEX Materials V3 silently produces no pixel ops."""
        if not self._is_apex_materials_gate_enabled():
            return
        if not bool(getattr(self.config, "apply_pixel_ops", False)):
            return
        if not isinstance(materials_v3_result, dict):
            return

        material_masks = materials_v3_result.get("material_masks")
        if not isinstance(material_masks, dict) or not material_masks:
            return

        from .pixel_ops_registry import OP_REGISTRY

        implemented_materials = [
            material
            for material in material_masks
            if any(op.implemented for op in OP_REGISTRY.get(str(material), {}).values())
        ]
        if not implemented_materials:
            return

        pixel_ops = materials_v3_result.get("materials_v3_pixel_ops")
        if not isinstance(pixel_ops, dict):
            return
        applied_ops = pixel_ops.get("applied")
        if isinstance(applied_ops, list) and applied_ops:
            return

        blocked_reasons = self._pixel_ops_blocked_reasons(pixel_ops)
        details = {
            "material_count": len(material_masks),
            "implemented_materials": [str(material) for material in implemented_materials],
            "applied_ops_count": 0,
            "blocked_reasons": blocked_reasons,
        }

        # Soft-apex passthrough: if every implemented op is blocked solely
        # because per-material classifier confidence sat below its APEX threshold,
        # emit the output without pixel ops and surface a non-fatal warning rather
        # than failing the strict gate. All other blocker mixes still fail closed.
        if blocked_reasons and set(blocked_reasons.keys()) == {"below_confidence_threshold"}:
            self._record_apex_materials_passthrough(materials_v3_result, details)
            logger.warning(
                "APEX Materials V3 passthrough:"
                " every implemented op below confidence threshold"
                " (materials=%s, blocked=%s)",
                details["implemented_materials"],
                blocked_reasons,
            )
            return

        raise ApexStrictGateError(
            "APEX_MATERIALS_PIXEL_OPS_EMPTY",
            "Material masks were detected, but every implemented Materials V3 pixel operation was blocked.",
            details=details,
        )

    @staticmethod
    def _record_apex_materials_passthrough(
        materials_v3_result: Dict[str, Any],
        details: Dict[str, Any],
    ) -> None:
        """Attach a non-fatal passthrough warning to the Materials V3 result.

        Surfaces under both ``materials_v3_pixel_ops.passthrough_status`` (consumed
        by the orchestrator's per-image manifest) and
        ``materials_v3_metadata.segmentation_metadata.warnings`` (consumed by the
        run-card summary cache). Idempotent: repeat calls (e.g. on retry / re-entry)
        do not duplicate the warning code in the warnings list.
        """
        from .apex_codes import APEX_MATERIALS_PASSTHROUGH_LOW_CONFIDENCE

        warning_payload = {
            "code": APEX_MATERIALS_PASSTHROUGH_LOW_CONFIDENCE,
            "message": (
                "Materials V3 masks present but every implemented op was below"
                " its confidence threshold; emitting output without pixel ops."
            ),
            "details": dict(details),
        }

        pixel_ops = materials_v3_result.get("materials_v3_pixel_ops")
        if not isinstance(pixel_ops, dict):
            pixel_ops = {}
            materials_v3_result["materials_v3_pixel_ops"] = pixel_ops
        pixel_ops["passthrough_status"] = warning_payload

        materials_v3_metadata = materials_v3_result.setdefault("materials_v3_metadata", {})
        if not isinstance(materials_v3_metadata, dict):
            materials_v3_metadata = {}
            materials_v3_result["materials_v3_metadata"] = materials_v3_metadata
        segmentation_metadata = materials_v3_metadata.get("segmentation_metadata")
        if not isinstance(segmentation_metadata, dict):
            segmentation_metadata = {}
        else:
            segmentation_metadata = dict(segmentation_metadata)
        warnings_list = list(segmentation_metadata.get("warnings") or [])
        if APEX_MATERIALS_PASSTHROUGH_LOW_CONFIDENCE not in warnings_list:
            warnings_list.append(APEX_MATERIALS_PASSTHROUGH_LOW_CONFIDENCE)
        segmentation_metadata["warnings"] = warnings_list
        segmentation_metadata["pixel_ops_passthrough"] = warning_payload
        materials_v3_metadata["segmentation_metadata"] = segmentation_metadata

    def _load_cached_depth(
        self,
        depth_path: Path,
        float_depth_path: Path,
    ) -> Optional[np.ndarray]:
        """Load cached depth data, preferring float precision.

        Args:
            depth_path: Path to quantized depth PNG
            float_depth_path: Path to float depth .npy file

        Returns:
            Depth array (numpy), or None if loading fails
        """

        # Prefer float depth for better PBR
        # quality (avoid quantization artifacts)
        if float_depth_path.exists():
            try:
                depth_data = np.load(str(float_depth_path))
                logger.debug(
                    "Loaded float depth" " from: %s",
                    float_depth_path,
                )
                return depth_data
            except Exception as e:
                logger.warning(
                    "Failed to load float" " depth: %s",
                    e,
                )

        # Fall back to quantized depth image
        if depth_path.exists():
            try:
                from .depth_writer import read_depth_u16_png

                depth_data = read_depth_u16_png(depth_path)

                # Robust normalization
                depth_data = np.asarray(depth_data)
                if depth_data.dtype == np.uint16:
                    # Reader returned uint16 - normalize once
                    depth_data = depth_data.astype(np.float32) / 65535.0
                else:
                    # Reader returned float - ensure correct range
                    depth_data = depth_data.astype(np.float32, copy=False)
                    # If reader returned unnormalized values, normalize
                    maxv = float(np.nanmax(depth_data)) if depth_data.size else 0.0
                    if maxv > 1.5:
                        depth_data /= 65535.0

                logger.debug(
                    "Loaded quantized depth" " from: %s",
                    depth_path,
                )
                return depth_data
            except Exception as e:
                logger.warning(
                    "Failed to load depth" " image: %s",
                    e,
                )

        return None

    def _parallel_preprocess_batch(
        self,
        image_inputs: List[ImageInput],
        input_root: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Parallel preprocessing.

        Phase 2: I/O-bound operations parallelized with ThreadPoolExecutor.

        Args:
            image_inputs: List of images to preprocess
            input_root: Base directory for relative path calculation

        Returns:
            List of preprocessing results with skip flags and paths
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._preprocess_single,
                    img,
                    input_root,
                ): (index, img)
                for index, img in enumerate(image_inputs)
            }

            # Preserve the caller's input ordering even when futures complete out
            # of order, so downstream manifests and result joins stay deterministic.
            results: List[Optional[Dict[str, Any]]] = [None] * len(image_inputs)
            for future in as_completed(futures):
                index, img = futures[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.error(
                        "Preprocessing failed" + " for %s: %s",
                        img.path,
                        e,
                    )
                    results[index] = {
                        "status": "error",
                        "image_input": img,
                        "error": str(e),
                    }

            if any(result is None for result in results):
                raise RuntimeError(
                    "Parallel preprocessing" " returned incomplete" " result set.",
                )
            return cast(List[Dict[str, Any]], results)

    def _preprocess_single(
        self,
        image_input: ImageInput,
        input_root: Optional[Path],
    ) -> Dict[str, Any]:
        """Preprocess single image: generate paths and check skip logic.

        Args:
            image_input: Input image information
            input_root: Base directory for relative path calculation

        Returns:
            Dictionary with preprocessing metadata
        """
        use_xxhash = getattr(self.config, "use_xxhash", False)
        output_key = (
            make_output_key(
                image_input.path,
                input_root,
                use_xxhash=use_xxhash,
            )
            if input_root
            else Path(
                sanitize_file_stem(
                    image_input.path.stem,
                ),
            )
        )

        depth_path = self.depth_dir / output_key.parent / f"{output_key.name}_depth.png"
        manifest_path = self.manifests_dir / output_key.parent / f"{output_key.name}_combined.json"

        # Check skip logic (uses cached manifest loading from Phase 1)
        should_skip = not self.config.force_depth and self.should_skip_depth(
            depth_path,
            manifest_path,
            image_input,
        )

        return {
            "status": "ok",
            "image_input": image_input,
            "output_key": output_key,
            "should_skip": should_skip,
            "depth_path": depth_path,
            "manifest_path": manifest_path,
        }

    def enhance_batch_parallel(
        self,
        image_inputs: List[ImageInput],
        input_root: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Process batch of images with parallel I/O operations.

        Phase 2 Architecture:
        - ThreadPoolExecutor for I/O-bound: validation, skip logic checks
        - Sequential GPU inference (avoid VRAM contention)
        - Parallel postprocessing (PBR generation, file writes)

        Args:
            image_inputs: List of images to process
            input_root: Base directory for relative paths

        Returns:
            List of processing results
        """
        if not self._use_parallel or len(image_inputs) < 4:
            # Fall back to sequential for small batches
            logger.debug(
                "Using sequential" + " processing" + " (batch size: %d)",
                len(image_inputs),
            )
            return [self.enhance_image(img, input_root) for img in image_inputs]

        logger.info(
            "Parallel batch processing:" + " %d images with %d workers",
            len(image_inputs),
            self.max_workers,
        )

        # Phase 1: Parallel preprocessing (I/O-bound)
        preprocessed = self._parallel_preprocess_batch(
            image_inputs,
            input_root,
        )

        # Phase 2: Sequential depth inference (GPU-bound, avoid contention)
        # PERFORMANCE FIX (#4): Pass precomputed paths to avoid redundant I/O
        results = []
        for item in preprocessed:
            if item["status"] == "error":
                results.append(item)
                continue

            try:
                # Extract precomputed paths and pass to enhance_image
                precomputed = {
                    "output_key": item["output_key"],
                    "depth_path": item["depth_path"],
                    "manifest_path": item["manifest_path"],
                    "should_skip": item["should_skip"],
                }
                result = self.enhance_image(
                    item["image_input"],
                    input_root,
                    _precomputed_paths=precomputed,
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    "Enhancement failed" + " for %s: %s",
                    item["image_input"].path,
                    e,
                )
                error_payload: Dict[str, Any] = {
                    "status": "error",
                    "image": str(
                        item["image_input"].path,
                    ),
                    "error": str(e),
                }
                _abm = (
                    getattr(
                        self,
                        "_active_backend_metadata",
                        None,
                    )
                    or self._backend_metadata
                )
                error_payload["backend"] = getattr(
                    _abm,
                    "resolved_backend",
                    None,
                )
                error_payload["attempts"] = list(
                    getattr(
                        self,
                        "_active_depth_attempts",
                        [],
                    )
                    or [],
                )
                error_payload["selected_attempt_index"] = getattr(
                    self,
                    "_active_selected_attempt_index",
                    None,
                )
                error_payload["quality_gate"] = None
                if isinstance(e, ApexStrictGateError):
                    error_payload["error_code"] = e.code
                    error_payload["error_details"] = e.details
                    error_payload["quality_gate"] = build_quality_gate_report(e.details)
                results.append(error_payload)

        return results

    def enhance_batch(
        self,
        input_dir: Path,
        image_extensions: Optional[List[str]] = None,
        input_files: Optional[List[Path]] = None,
    ) -> List[Dict[str, Any]]:
        """Process a batch of images with accurate execution timestamps.

        Args:
            input_dir: Directory containing input images
            image_extensions: List of file extensions to process
            input_files: Optional frozen input selection (P0-1, issue #2065).
                When provided — e.g. from a ResolvedInvocation — these exact
                files are processed and no rediscovery scan runs, so the run
                consumes the plan's input selection instead of recomputing
                one that may diverge.

        Returns:
            List of processing results for each image
        """
        if image_extensions is None:
            image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

        # Capture accurate batch start time
        batch_start_time = time.time()
        batch_start_utc = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(batch_start_time),
        )

        batch_id = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._active_batch_id = batch_id
        self._active_run_card_segmentation_metadata = {}
        logger.info(
            "Batch %s: Scanning %s",
            batch_id,
            input_dir,
        )

        # ADR-023 Phase 3: Capture backend
        # selection metadata and log truth line
        backend_metadata = self._capture_backend_metadata()
        logger.info(
            "Backend selection:" " requested=%s" " resolved=%s" " status=%s device=%s" " model=%s",
            backend_metadata.requested_backend or "auto",
            backend_metadata.resolved_backend,
            backend_metadata.resolution_status,
            backend_metadata.device,
            backend_metadata.model_id,
        )

        # Store backend metadata for use in _write_manifest
        # pylint: disable=attribute-defined-outside-init
        self._backend_metadata = backend_metadata  # type: ignore[attr-defined]

        # Use input discovery to exclude depth artifacts and derived outputs
        # ROBUSTNESS FIX (#6): Pass output_root
        # to explicitly exclude output directory
        if input_files is not None:
            # P0-1: the plan's frozen selection is authoritative — no rescan.
            images = [Path(f) for f in input_files]
        else:
            discovery_config = DiscoveryConfig(
                strict_mode=self.config.strict_inputs,
            )
            images = discover_images(
                input_dir,
                discovery_config,
                image_extensions,
                output_dir=self.output_root,
            )

        # Inert scene-group bridge: preserve
        # existing per-image behavior and order.
        sorted_images = sorted(images)
        scene_groups = build_scene_groups(
            sorted_images,
            dataset_root=input_dir,
            grouping_mode=getattr(self.config, "grouping_mode", "single"),
        )
        image_inputs = [ImageInput(img) for scene in scene_groups for img in scene.images]

        if self._use_parallel and len(image_inputs) >= 4:
            logger.info(
                "Using parallel batch" + " processing for %d" + " images",
                len(image_inputs),
            )
            results = self.enhance_batch_parallel(
                image_inputs,
                input_root=input_dir,
            )
        else:
            # Sequential processing (original behavior)
            results = []
            for img_input in image_inputs:
                try:
                    results.append(
                        self.enhance_image(
                            img_input,
                            input_root=input_dir,
                        ),
                    )
                except Exception as e:
                    logger.error(
                        "Failed %s: %s",
                        img_input.path,
                        e,
                    )
                    error_payload: Dict[str, Any] = {
                        "status": "error",
                        "image": str(
                            img_input.path,
                        ),
                        "error": str(e),
                    }
                    _abm = (
                        getattr(
                            self,
                            "_active_backend_metadata",
                            None,
                        )
                        or self._backend_metadata
                    )
                    error_payload["backend"] = getattr(
                        _abm,
                        "resolved_backend",
                        None,
                    )
                    error_payload["attempts"] = list(
                        getattr(
                            self,
                            "_active_depth_attempts",
                            [],
                        )
                        or [],
                    )
                    error_payload["selected_attempt_index"] = getattr(
                        self,
                        "_active_selected_attempt_index",
                        None,
                    )
                    error_payload["quality_gate"] = None
                    if isinstance(e, ApexStrictGateError):
                        error_payload["error_code"] = e.code
                        error_payload["error_details"] = e.details
                        error_payload["quality_gate"] = build_quality_gate_report(e.details)
                    results.append(error_payload)

        if getattr(self.config, "enable_reconstruction", False):
            self._run_scene_reconstruction_stage(
                scene_groups=scene_groups,
                results=results,
                dataset_root=input_dir,
            )

        # Capture accurate batch end time
        batch_end_time = time.time()
        batch_end_utc = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(batch_end_time),
        )

        # Write batch summary with accurate timestamps
        # Extract runtime_s from successful results for statistics computation
        runtimes = [r.get("runtime_s", 0.0) for r in results if r.get("status") == "ok"]
        runtime_stats = compute_batch_runtime_stats(runtimes)

        # Detect runtime outliers (images taking >5× median time)
        # PERFORMANCE FIX (#3): Compute median
        # once, pass to all outlier checks
        median_runtime = runtime_stats.get("median", 0.0)
        outliers = []
        for r in results:
            if r.get("status") == "ok":
                runtime_s = r.get("runtime_s", 0.0)
                image_name = r.get("image", "unknown")
                outlier_result = detect_runtime_outliers(
                    image_name,
                    runtime_s,
                    runtimes,
                    median=median_runtime,
                )
                if outlier_result:
                    warning_msg, outlier_meta = outlier_result
                    outliers.append(
                        {
                            "image": image_name,
                            "metadata": outlier_meta,
                        }
                    )

        self._enforce_apex_depth_png_uniqueness(results)
        batch_backend_summary = self._compute_backend_summary(results)
        batch_requested_backend_defect = self._requested_backend_fulfillment_defect(
            results,
            batch_backend_summary,
        )
        batch_backend_selection = self._build_backend_selection_payload(
            results,
            batch_backend_summary,
        )
        batch_config_fingerprint = self._build_run_card_config_fingerprint(
            backend_selection=batch_backend_selection,
            run_card_version=getattr(self.config, "run_card_version", "v2"),
            include_proofs=bool(getattr(self.config, "run_card_include_proofs", False)),
        )

        bm = BatchManifest(
            batch_id=batch_id,
            start_time=batch_start_utc,
            end_time=batch_end_utc,
            config=self._build_batch_manifest_config(
                backend_selection=batch_backend_selection,
                config_fingerprint=batch_config_fingerprint,
            ),
            results=self._portable_batch_manifest_results(results, input_root=input_dir),
            stats={
                **runtime_stats,
                "total_images": len(results),
                "batch_runtime_seconds": batch_end_time - batch_start_time,
                "outliers": outliers if outliers else [],
            },
        )
        batch_manifest_path = self.manifests_dir / f"batch_{batch_id}.json"
        bm.write(batch_manifest_path)

        # Emit run card if enabled
        if self.config.emit_run_card:
            self._emit_run_card(
                batch_id,
                batch_start_utc,
                batch_end_utc,
                results,
                runtime_stats,
                outliers,
                batch_manifest_path=batch_manifest_path,
                requested_backend_defect=batch_requested_backend_defect,
            )

        if batch_requested_backend_defect is not None:
            raise RuntimeError(batch_requested_backend_defect)

        return results

    @staticmethod
    def _result_image_key(
        path_value: Any,
    ) -> Optional[str]:
        """Normalize result image path for lookup joins."""
        if not isinstance(path_value, str) or not path_value:
            return None
        try:
            return str(Path(path_value).resolve())
        except (OSError, ValueError, RuntimeError):
            # Invalid path or symlink loop - return None
            return None

    def _run_scene_reconstruction_stage(
        self,
        *,
        scene_groups: List[SceneGroup],
        results: List[Dict[str, Any]],
        dataset_root: Path,
    ) -> None:
        """Run gated scene-level reconstruction for eligible grouped scenes."""
        if not bool(
            getattr(
                self.config,
                "non_commercial_ok",
                False,
            ),
        ):
            raise ReconstructionLicenseRestrictionError(
                "Scene reconstruction requires"
                " non_commercial_ok=True due"
                " to Inria 3D Gaussian"
                " Splatting non-commercial"
                " license terms."
            )
        if not bool(
            getattr(
                self.config,
                "accept_research_tools_license",
                False,
            ),
        ):
            raise ReconstructionLicenseRestrictionError(
                "Scene reconstruction"
                " requires"
                " accept_research_tools"
                "_license=True to"
                " acknowledge research-only"
                " tool licensing"
                " constraints."
            )

        sidecar_value = getattr(self.config, "cameras_sidecar_path", None)
        sidecar_path = Path(sidecar_value) if isinstance(sidecar_value, str) and sidecar_value else None
        sidecar_payload = load_sidecar_payload(sidecar_path)
        reconstruction_tier = str(
            getattr(
                self.config,
                "reconstruction_tier",
                "apex_research",
            ),
        )

        result_by_path: Dict[str, Dict[str, Any]] = {}
        for result in results:
            path_key = self._result_image_key(
                result.get("image"),
            )
            if path_key:
                result_by_path[path_key] = result

        for scene in scene_groups:
            if len(scene.images) < 2:
                continue

            scene_results: List[Dict[str, Any]] = []
            for image_path in scene.images:
                img_result = result_by_path.get(
                    str(Path(image_path).resolve()),
                )
                if not isinstance(img_result, dict) or img_result.get("status") != "ok":
                    scene_results = []
                    break
                scene_results.append(img_result)
            if not scene_results:
                continue

            cameras = load_scene_cameras(
                scene=scene,
                dataset_root=dataset_root,
                sidecar_path=sidecar_path,
                sidecar_payload=sidecar_payload,
            )
            if not cameras:
                logger.info(
                    "Skipping reconstruction" " for scene %s:" " cameras unavailable",
                    scene.scene_id,
                )
                continue
            camera_sources = {camera.provenance.source for camera in cameras}
            if len(camera_sources) > 1:
                logger.warning(
                    "Skipping reconstruction" " for scene %s: mixed" " camera sources %s",
                    scene.scene_id,
                    sorted(camera_sources),
                )
                continue
            if any(camera.provenance.confidence == "low" for camera in cameras):
                logger.warning(
                    "Skipping reconstruction" " for scene %s:" " low-confidence camera" " provenance detected",
                    scene.scene_id,
                )
                continue
            try:
                dataset_health = check_camera_geometry_sanity(cameras)
                threshold_raw: Optional[str] = os.getenv(
                    "TP_RECONSTRUCTION" + "_RISK_THRESHOLD",
                    "0.65",
                )
                risk_threshold: float = 0.65
                try:
                    risk_threshold = float(
                        threshold_raw or "0.65",
                    )
                except ValueError:
                    logger.warning(
                        "Invalid" " TP_RECONSTRUCTION" "_RISK_THRESHOLD=%r;" " using default 0.65",
                        threshold_raw,
                    )
                    risk_threshold = 0.65
                risk_score = float(dataset_health["risk_score"])
                if risk_score > risk_threshold:
                    message = (
                        f"Dataset risk score"
                        f" {risk_score:.3f}"
                        f" exceeds threshold"
                        f" {risk_threshold:.3f}"
                        f" for scene"
                        f" {scene.scene_id}"
                    )
                    triage_report = build_dataset_triage_report(
                        scene.scene_id,
                        dataset_health,
                    )
                    scene_results[0]["reconstruction_risk" "_gate_message"] = message
                    scene_results[0]["reconstruction_risk" "_gate_triage"] = triage_report
                    health_with_triage: Dict[str, Any] = dict(dataset_health)
                    health_with_triage["triage"] = triage_report
                    scene_results[0]["reconstruction" "_dataset_health"] = health_with_triage
                    logger.warning(
                        "RECONSTRUCTION_DATASET" "_RISK_GATE: %s\n%s",
                        message,
                        triage_report,
                    )
                    continue
                cameras, camera_normalization = normalize_camera_poses(cameras)
            except ValueError as exc:
                logger.warning(
                    "Skipping reconstruction" " for scene %s: camera" " geometry validation" " failed (%s)",
                    scene.scene_id,
                    exc,
                )
                continue
            preflight_result = validate_scene_preflight(
                scene=scene,
                cameras=cameras,
            )
            preflight_path = write_scene_preflight_artifact(
                scene_id=scene.scene_id,
                result=preflight_result,
                output_dir=self.reconstruction_dir,
            )
            scene_results[0]["reconstruction_preflight_path"] = str(preflight_path)
            if not preflight_result.valid:
                logger.warning(
                    "Skipping reconstruction" " for scene %s: preflight" " failed (%s)",
                    scene.scene_id,
                    preflight_result.reason,
                )
                continue

            context = SceneContext.build(
                scene=scene,
                dataset_root=dataset_root,
                cameras=cameras,
                metadata={
                    "grouping_mode": str(
                        getattr(
                            self.config,
                            "grouping_mode",
                            "single",
                        ),
                    ),
                    "camera_normalization": camera_normalization,
                    "dataset_health": dataset_health,
                },
            )

            segmentation_artifact_paths = tuple(
                Path(path_value)
                for path_value in (
                    result.get(
                        "segmentation_mask_path",
                    )
                    for result in scene_results
                )
                if isinstance(
                    path_value,
                    str,
                )
                and path_value
            )
            scene_manifest = build_scene_manifest(
                context=context,
                output_root=self.output_root,
                segmentation_artifact_paths=segmentation_artifact_paths,
                camera_sidecar_path=sidecar_path,
            )
            input_hashes = scene_manifest.get("input_hashes", {})
            artifact_index_entries: List[Dict[str, Any]] = []
            if isinstance(input_hashes, dict):
                for relative_path in sorted(
                    input_hashes,
                ):
                    digest = input_hashes.get(
                        relative_path,
                    )
                    if (
                        isinstance(
                            relative_path,
                            str,
                        )
                        and relative_path
                        and isinstance(
                            digest,
                            str,
                        )
                        and len(digest) == 64
                    ):
                        artifact_index_entries.append(
                            {
                                "relative_path": (relative_path),
                                "sha256": digest,
                            },
                        )
            artifact_index_by_relative_path = {
                entry["relative_path"]: entry for entry in artifact_index_entries if isinstance(entry, dict)
            }
            run_card_merkle_root = _compute_artifact_merkle_root(
                artifact_index_entries,
            )
            reconstruction_config = {
                "iterations": int(
                    getattr(
                        self.config,
                        "reconstruction_iterations",
                        1000,
                    ),
                ),
                "tier": reconstruction_tier,
                "grouping_mode": str(
                    getattr(
                        self.config,
                        "grouping_mode",
                        "single",
                    ),
                ),
            }

            try:
                verify_scene_integrity(
                    scene_manifest=scene_manifest,
                    artifact_index=artifact_index_by_relative_path,
                )
                scene_fingerprint = compute_scene_fingerprint(
                    scene_manifest=scene_manifest,
                    artifact_index=artifact_index_by_relative_path,
                    reconstruction_config=reconstruction_config,
                )
                reconstruction_output_dir = self.reconstruction_dir / scene_fingerprint
                scene_manifest_path = write_scene_manifest(
                    scene_manifest=scene_manifest,
                    output_dir=reconstruction_output_dir,
                )
                scene_results[0]["reconstruction_scene" "_manifest_path"] = str(scene_manifest_path)
                scene_results[0]["reconstruction_scene" "_fingerprint"] = scene_fingerprint
                if bool(
                    getattr(
                        self.config,
                        "emit_scene_debug_bundle",
                        False,
                    ),
                ):
                    debug_paths = write_scene_debug_bundle(
                        context=context,
                        segmentation_artifact_paths=(segmentation_artifact_paths),
                        scene_manifest=scene_manifest,
                        output_dir=(reconstruction_output_dir),
                    )
                    debug_scene_manifest = debug_paths.get(
                        "scene_manifest_path",
                    )
                    if (
                        isinstance(
                            debug_scene_manifest,
                            Path,
                        )
                        and debug_scene_manifest.exists()
                    ):
                        scene_results[0]["reconstruction_debug" "_manifest_path"] = str(
                            debug_scene_manifest,
                        )
                    debug_cameras = debug_paths.get(
                        "cameras_path",
                    )
                    if (
                        isinstance(
                            debug_cameras,
                            Path,
                        )
                        and debug_cameras.exists()
                    ):
                        scene_results[0]["reconstruction_debug" "_cameras_path"] = str(
                            debug_cameras,
                        )
                    debug_preview = debug_paths.get(
                        "reprojection_preview" "_path",
                    )
                    if (
                        isinstance(
                            debug_preview,
                            Path,
                        )
                        and debug_preview.exists()
                    ):
                        scene_results[0]["reconstruction_debug" "_preview_path"] = str(
                            debug_preview,
                        )

                report_path = self._maybe_use_reconstruction_cache(
                    scene_id=scene.scene_id,
                    scene_fingerprint=scene_fingerprint,
                    run_card_merkle_root=run_card_merkle_root,
                )
                if report_path is None:
                    report_path = self.run_scene_reconstruction_fn(
                        context=context,
                        output_dir=reconstruction_output_dir,
                        iterations=reconstruction_config["iterations"],
                        tier=reconstruction_tier,
                        scene_fingerprint=scene_fingerprint,
                        run_card_merkle_root=run_card_merkle_root,
                    )
            except ReconstructionLicenseRestrictionError:
                raise
            except Exception as exc:
                logger.warning(
                    "Scene reconstruction" " failed for %s: %s",
                    scene.scene_id,
                    exc,
                )
                continue

            if not isinstance(report_path, Path):
                report_path = Path(str(report_path))
            if not report_path.exists():
                logger.warning(
                    "Scene reconstruction" " returned missing" " report path for" " %s: %s",
                    scene.scene_id,
                    report_path,
                )
                continue

            scene_results[0]["reconstruction_report_path"] = str(report_path)
            scene_results[0]["reconstruction_scene_id"] = scene.scene_id
            manifest_path = manifest_artifact_path(
                scene_id=scene.scene_id,
                output_dir=reconstruction_output_dir,
            )
            if manifest_path.exists():
                scene_results[0]["reconstruction_manifest_path"] = str(manifest_path)
            else:
                logger.warning(
                    "Reconstruction manifest" + " missing for %s: %s",
                    scene.scene_id,
                    manifest_path,
                )
            diagnostics_path = diagnostics_artifact_path(
                scene_id=scene.scene_id,
                output_dir=(reconstruction_output_dir),
            )
            if diagnostics_path.exists():
                scene_results[0]["reconstruction" "_diagnostics_path"] = str(diagnostics_path)
            else:
                logger.warning(
                    "Scene diagnostics" " missing for %s: %s",
                    scene.scene_id,
                    diagnostics_path,
                )
            logger.info(
                "Scene reconstruction" " completed:" " scene_id=%s" " report=%s" " diagnostics=%s",
                scene.scene_id,
                report_path,
                diagnostics_path,
            )

    def _maybe_use_reconstruction_cache(
        self,
        *,
        scene_id: str,
        scene_fingerprint: str,
        run_card_merkle_root: str,
    ) -> Optional[Path]:
        """Use cached reconstruction report."""
        cache_dir = self.reconstruction_dir / scene_fingerprint
        _sid = sanitize_path_component_nonlossy(
            scene_id,
        )
        report_path = cache_dir / f"{_sid}_reconstruction_report.json"
        if not report_path.exists():
            return None
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            # Cache file unreadable, malformed, or encoding error - cache miss
            return None
        if payload.get("scene_fingerprint") != scene_fingerprint:
            return None
        if payload.get("run_card_merkle_root") != run_card_merkle_root:
            return None
        logger.info(
            "Reconstruction cache hit" " for scene_id=%s at %s",
            scene_id,
            report_path,
        )
        return report_path

    def _collect_run_card_artifact_paths(
        self,
        results: List[Dict[str, Any]],
        batch_manifest_path: Optional[Path] = None,
    ) -> List[Path]:
        """Collect artifact paths for the batch."""
        artifact_paths: List[Path] = []
        batch_id: Optional[str] = None

        if batch_manifest_path and batch_manifest_path.exists():
            artifact_paths.append(batch_manifest_path)
            batch_name = batch_manifest_path.stem
            if batch_name.startswith("batch_"):
                prefix_len = len("batch_")
                batch_id = batch_name[prefix_len:]

        for result in results:
            for direct_path_key in (
                "v2_log_path",
                "v2_report_path",
                "v2_output_path",
                "segmentation_mask_path",
                "reconstruction_preflight_path",
                "reconstruction_scene_manifest_path",
                "reconstruction_debug_manifest_path",
                "reconstruction_debug_cameras_path",
                "reconstruction_debug_preview_path",
                "reconstruction_manifest_path",
                "reconstruction_report_path",
                "reconstruction_diagnostics_path",
                "vlm_caption_proxy_path",
                "vlm_caption_sidecar_path",
                "vlm_caption_raw_path",
            ):
                direct_path_value = result.get(direct_path_key)
                if isinstance(direct_path_value, str) and direct_path_value:
                    candidate = Path(direct_path_value)
                    if candidate.exists():
                        artifact_paths.append(candidate)

            manifest_value = result.get("manifest")
            if manifest_value:
                manifest_path = Path(manifest_value)
                if manifest_path.exists():
                    artifact_paths.append(manifest_path)

                    provenance_sidecar_path = manifest_path.with_name(
                        f"{manifest_path.stem}" "_provenance.json",
                    )
                    if provenance_sidecar_path.exists():
                        artifact_paths.append(provenance_sidecar_path)

                    # Include the per-image V2 stage log when available.
                    manifest_name = manifest_path.stem
                    output_key_name = manifest_name.removesuffix("_combined")
                    try:
                        manifest_relative_parent = (
                            manifest_path.resolve()
                            .relative_to(
                                self.manifests_dir.resolve(),
                            )
                            .parent
                        )
                    except ValueError:
                        manifest_relative_parent = Path(".")
                    v2_log_path = (
                        self.logs_dir
                        / manifest_relative_parent
                        / _v2_log_filename(
                            output_key_name,
                            batch_id,
                        )
                    )

                    if v2_log_path.exists():
                        artifact_paths.append(v2_log_path)

                    try:
                        combined_manifest = CombinedManifest.load(
                            manifest_path,
                        )
                    except Exception as exc:
                        logger.debug(
                            "Skipping artifact" " extraction for" " unreadable manifest" " %s: %s",
                            manifest_path,
                            exc,
                        )
                    else:
                        if combined_manifest.depth and combined_manifest.depth.depth_path:
                            artifact_paths.append(
                                Path(
                                    combined_manifest.depth.depth_path,
                                ),
                            )

                        if combined_manifest.v2 and combined_manifest.v2.report_path:
                            report_path = Path(
                                combined_manifest.v2.report_path,
                            )
                            artifact_paths.append(report_path)
                            if report_path.exists():
                                try:
                                    with open(
                                        report_path,
                                        "r",
                                        encoding="utf-8",
                                    ) as report_file:
                                        report_payload = json.load(
                                            report_file,
                                        )
                                except Exception as exc:
                                    logger.debug(
                                        "Failed to parse" " V2 report for" " artifact indexing" " (%s): %s",
                                        report_path,
                                        exc,
                                    )
                                else:
                                    for field in ("output", "depth_map"):
                                        value = report_payload.get(field)
                                        if isinstance(value, str) and value:
                                            artifact_paths.append(Path(value))

                        if combined_manifest.pbr_assets:
                            for key, value in combined_manifest.pbr_assets.items():
                                if (
                                    key.endswith("_path")
                                    and isinstance(
                                        value,
                                        str,
                                    )
                                    and value
                                ):
                                    artifact_paths.append(Path(value))

                        if combined_manifest.materials_v3 and isinstance(
                            combined_manifest.materials_v3.segmentation_metadata,
                            dict,
                        ):
                            seg_md = combined_manifest.materials_v3.segmentation_metadata
                            mask_artifact_path = seg_md.get(
                                "mask_artifact_path",
                            )
                            if (
                                isinstance(
                                    mask_artifact_path,
                                    str,
                                )
                                and mask_artifact_path
                            ):
                                artifact_paths.append(Path(mask_artifact_path))

            depth_value = result.get("depth_path")
            if depth_value:
                depth_path = Path(depth_value)
                artifact_paths.append(depth_path)
                depth_metadata_path = depth_path.with_name(
                    f"{depth_path.stem}" "_metadata.json",
                )
                if depth_metadata_path.exists():
                    artifact_paths.append(depth_metadata_path)
                float_depth_path = depth_path.with_suffix(".npy")
                if float_depth_path.exists():
                    artifact_paths.append(float_depth_path)

        return artifact_paths

    def _compute_backend_summary(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Backend fallback summary for run-card."""
        requested_backend = self._backend_metadata.requested_backend or "auto"
        observed_ok_backends = {
            str(result.get("backend")) for result in results if result.get("status") == "ok" and result.get("backend")
        }
        preferred_backend = self._backend_metadata.resolved_backend
        final_backends_used: List[str] = []
        if preferred_backend in observed_ok_backends:
            final_backends_used.append(preferred_backend)
            final_backends_used.extend(
                sorted(
                    observed_ok_backends - {preferred_backend},
                ),
            )
        else:
            final_backends_used.extend(sorted(observed_ok_backends))
        primary_backend = final_backends_used[0] if final_backends_used else None

        fallback_images = 0
        semantic_fallback_images = 0
        operational_fallback_images = 0
        for result in results:
            if result.get("status") != "ok":
                continue
            attempts = result.get("attempts")
            if not isinstance(attempts, list) or not attempts:
                continue

            used_fallback = bool(result.get("fallback_used")) or len(attempts) > 1
            if not used_fallback:
                continue

            fallback_images += 1
            failure_kinds = {attempt.get("failure_kind") for attempt in attempts if attempt.get("status") == "failed"}
            if "semantic" in failure_kinds:
                semantic_fallback_images += 1
            if "operational" in failure_kinds:
                operational_fallback_images += 1

        return {
            "requested_backend": requested_backend,
            "primary_backend": primary_backend,
            "final_backends_used": final_backends_used,
            "fallback_images": fallback_images,
            "semantic_fallback_images": semantic_fallback_images,
            "operational_fallback_images": operational_fallback_images,
        }

    def _requested_backend_fulfillment_defect(
        self,
        results: List[Dict[str, Any]],
        backend_summary: Dict[str, Any],
    ) -> Optional[str]:
        """Return a defect summary when requested Depth Pro fully falls back."""
        requested_backend = normalize_backend_id(backend_summary.get("requested_backend"))
        primary_backend = normalize_backend_id(backend_summary.get("primary_backend"))
        success_count = sum(1 for result in results if result.get("status") == "ok")
        fallback_images = backend_summary.get("fallback_images")

        if (
            requested_backend != "depth_pro"
            or success_count <= 0
            or primary_backend == requested_backend
            or not isinstance(fallback_images, int)
            or fallback_images != success_count
        ):
            return None

        detail: Optional[str] = None
        for result in results:
            if result.get("status") != "ok":
                continue
            attempts = result.get("attempts")
            if not isinstance(attempts, list):
                continue
            for attempt in attempts:
                if attempt.get("status") == "failed" and normalize_backend_id(attempt.get("backend")) == requested_backend:
                    detail = attempt.get("error_message") or attempt.get("error_code")
                    break
            if detail:
                break

        if detail is None:
            startup_reason = getattr(self._backend_metadata, "resolution_reason", None)
            if isinstance(startup_reason, str) and startup_reason.strip():
                detail = startup_reason.strip()

        message = (
            f"Requested backend '{requested_backend}' was not honored: "
            f"all successful images ({success_count}/{success_count}) used "
            f"fallback backend '{primary_backend}'."
        )
        if isinstance(detail, str) and detail.strip():
            message = f"{message} First fallback reason: {detail.strip()}"
        return message

    def _build_backend_selection_payload(
        self,
        results: List[Dict[str, Any]],
        backend_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the resolved backend-selection payload used by manifests."""
        requested_backend = getattr(self._backend_metadata, "requested_backend", None) or "auto"
        backend_selection_resolved = (
            backend_summary["final_backends_used"][0]
            if backend_summary.get("final_backends_used")
            else getattr(self._backend_metadata, "resolved_backend", None)
        )
        backend_model_artifact = self._resolve_run_card_backend_model_artifact(
            results,
            backend_selection_resolved,
        )
        backend_selection: Dict[str, Any] = {
            "requested": requested_backend,
            "resolved": backend_selection_resolved,
            "device": getattr(self._backend_metadata, "device", None),
            "model_id": self._resolve_run_card_backend_model_id(
                results,
                backend_selection_resolved,
            ),
            "model_artifact_filename": backend_model_artifact.get(
                "model_artifact_filename",
            ),
            "model_artifact_sha256": backend_model_artifact.get(
                "model_artifact_sha256",
            ),
        }
        if (
            getattr(self._backend_metadata, "resolved_backend", None) != backend_selection_resolved
            and backend_summary.get("fallback_images") == 0
        ):
            backend_selection["logical_backend"] = getattr(self._backend_metadata, "resolved_backend", None)
            backend_selection["resolved_engine"] = backend_selection_resolved
        return backend_selection

    def _build_batch_manifest_config(
        self,
        *,
        backend_selection: Dict[str, Any],
        config_fingerprint: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build replayable batch-level config from the resolved execution contract."""
        return {
            "model": config_fingerprint.get("model_variant") or self._model_variant.value.name,
            "depth_backend": backend_selection.get("resolved"),
            "model_id": backend_selection.get("model_id"),
            "model_artifact_filename": backend_selection.get("model_artifact_filename"),
            "model_artifact_sha256": backend_selection.get("model_artifact_sha256"),
            "device": backend_selection.get("device"),
            "license_contract_id": (
                "apple-machine-learning-research-license"
                if normalize_backend_id(backend_selection.get("resolved")) == "depth_pro"
                else None
            ),
            "quality_tier": getattr(self.config, "quality_tier", None),
            "depth_quantization": getattr(self.config, "depth_quantization", None),
            "depth_png_encoding": config_fingerprint.get("depth_png_encoding"),
            "output_depth_units": config_fingerprint.get("output_depth_units"),
            "output_bit_depth": config_fingerprint.get("output_bit_depth"),
            "enable_materials_v3": bool(getattr(self.config, "enable_materials_v3", False)),
            "generate_pbr": bool(getattr(self.config, "generate_pbr", False)),
            "emit_run_card": bool(getattr(self.config, "emit_run_card", False)),
            "vlm_captioning_enabled": bool(getattr(self.config, "vlm_captioning_enabled", False)),
            "vlm_captioning_backend": getattr(self.config, "vlm_captioning_backend", "fastvlm"),
            "vlm_captioning_model": getattr(self.config, "vlm_captioning_model", "default"),
            "vlm_captioning_proxy_format": getattr(self.config, "vlm_captioning_proxy_format", "png"),
            "vlm_captioning_max_side_px": int(getattr(self.config, "vlm_captioning_max_side_px", 1600) or 1600),
            "run_card_version": config_fingerprint.get("run_card_version"),
            "run_card_include_proofs": bool(config_fingerprint.get("run_card_include_proofs", False)),
            "config_fingerprint_sha256": config_fingerprint.get("sha256"),
        }

    def _portable_batch_manifest_results(
        self,
        results: List[Dict[str, Any]],
        *,
        input_root: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Return batch results with output paths made portable under output_root."""
        portable_results: List[Dict[str, Any]] = []
        path_like_keys = {
            "depth_path",
            "depth_float_path",
            "depth_metadata_path",
            "manifest",
            "v2_output",
            "v2_output_path",
            "v2_report",
            "v2_report_path",
            "v2_log",
            "v2_log_path",
            "provenance_sidecar",
            "pbr_manifest_path",
            "reconstruction_manifest_path",
            "reconstruction_report_path",
            "reconstruction_preflight_path",
            "reconstruction_diagnostics_path",
            "segmentation_mask_path",
            "vlm_caption_proxy_path",
            "vlm_caption_sidecar_path",
            "vlm_caption_raw_path",
        }
        for result in results:
            row = copy.deepcopy(result)
            image_value = row.get("image")
            if isinstance(image_value, str) and image_value.strip():
                row["image_basename"] = Path(image_value).name
                row["image"] = self._input_root_relative_path(image_value, input_root=input_root)
            for key in path_like_keys:
                relative_path = self._run_card_output_relative_path(row.get(key))
                if relative_path is not None:
                    row[key] = relative_path
            portable_results.append(row)
        return portable_results

    @staticmethod
    def _input_root_relative_path(path_value: str, *, input_root: Optional[Path]) -> str:
        """Render input image paths as input-root-relative when possible."""
        if input_root is not None:
            try:
                return relative_to_path_alias(path_value, input_root).as_posix()
            except ValueError:
                pass
        path = Path(path_value)
        if path.is_absolute():
            return path.name
        return path.as_posix()

    @staticmethod
    def _selected_successful_attempt(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a successful selected attempt from a result row."""
        attempts = result.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            return None
        selected_attempt_index = result.get("selected_attempt_index")
        if isinstance(selected_attempt_index, int) and 0 <= selected_attempt_index < len(attempts):
            attempt = attempts[selected_attempt_index]
            if isinstance(attempt, dict) and attempt.get("status") == "success":
                return attempt
        for attempt in attempts:
            if isinstance(attempt, dict) and attempt.get("status") == "success":
                return attempt
        return None

    def _enforce_apex_depth_png_uniqueness(self, results: List[Dict[str, Any]]) -> None:
        """Fail suspicious duplicate depth PNGs before provenance is emitted."""
        if not self._is_apex_tier():
            return

        grouped: Dict[
            str,
            Dict[str, Any],
        ] = {}
        for result in results:
            if result.get("status") != "ok":
                continue
            depth_path_value = result.get("depth_path")
            depth_float_path_value = result.get("depth_float_path")
            if not isinstance(depth_path_value, str) or not isinstance(depth_float_path_value, str):
                continue
            depth_path = Path(depth_path_value)
            depth_float_path = Path(depth_float_path_value)
            if not depth_path.exists() or not depth_float_path.exists():
                continue
            png_sha = compute_file_sha256(depth_path)
            float_sha = compute_file_sha256(depth_float_path)
            input_sha = self._normalize_sha256(result.get("input_sha256"))
            if input_sha is None:
                image_value = result.get("image")
                if isinstance(image_value, str) and Path(image_value).is_file():
                    input_sha = compute_file_sha256(Path(image_value))
            row = {
                "image": str(result.get("image")),
                "depth_path": str(depth_path),
                "depth_float_path": str(depth_float_path),
                "input_sha256": input_sha,
                "depth_float_sha256": float_sha,
            }
            if not input_sha:
                continue
            group = grouped.setdefault(
                png_sha,
                {
                    "rows": [],
                    "input_counts": {},
                    "float_counts": {},
                    "pair_counts": {},
                },
            )
            rows = group["rows"]
            input_counts = group["input_counts"]
            float_counts = group["float_counts"]
            pair_counts = group["pair_counts"]
            comparable_count = len(rows)
            pair_key = (input_sha, float_sha)
            covered_by_same_input_or_float = (
                int(input_counts.get(input_sha, 0)) + int(float_counts.get(float_sha, 0)) - int(pair_counts.get(pair_key, 0))
            )
            if comparable_count and covered_by_same_input_or_float < comparable_count:
                conflicting_row = next(
                    existing
                    for existing in rows
                    if existing.get("input_sha256") != input_sha and existing.get("depth_float_sha256") != float_sha
                )
                raise ApexStrictGateError(
                    "APEX_DEPTH_PNG_DUPLICATE_HASH",
                    "Different inputs and float-depth artifacts produced an identical depth PNG hash.",
                    {
                        "passed": False,
                        "failure_codes": ["APEX_DEPTH_PNG_DUPLICATE_HASH"],
                        "warnings": [],
                        "duplicate_depth_png_sha256": png_sha,
                        "conflicting_rows": [conflicting_row, row],
                    },
                )
            rows.append(row)
            input_counts[input_sha] = int(input_counts.get(input_sha, 0)) + 1
            float_counts[float_sha] = int(float_counts.get(float_sha, 0)) + 1
            pair_counts[pair_key] = int(pair_counts.get(pair_key, 0)) + 1

    def _resolve_run_card_backend_model_id(
        self,
        results: List[Dict[str, Any]],
        backend_selection_resolved: Optional[str],
    ) -> str:
        """Resolve run-card backend model_id from selected backend attempts."""
        resolved_backend = str(
            backend_selection_resolved or self._backend_metadata.resolved_backend or "",
        ).strip()
        if resolved_backend:
            for result in results:
                if result.get("status") != "ok":
                    continue
                if result.get("backend") != resolved_backend:
                    continue
                attempts = result.get("attempts")
                if isinstance(attempts, list) and attempts:
                    selected_attempt_index = result.get(
                        "selected_attempt_index",
                    )
                    normalized_selected_index = (
                        int(selected_attempt_index)
                        if isinstance(
                            selected_attempt_index,
                            int,
                        )
                        else None
                    )
                    attempt_model_id = self._extract_model_id_from_attempts(
                        resolved_backend,
                        attempts,
                        selected_attempt_index=normalized_selected_index,
                    )
                    if attempt_model_id:
                        return attempt_model_id
                model_id = result.get("model_id")
                if isinstance(model_id, str) and model_id.strip():
                    return model_id.strip()
            backend_cache = getattr(self, "_depth_backend_cache", {})
            return self._resolve_backend_model_id(
                resolved_backend,
                backend=backend_cache.get(resolved_backend),
            )
        return self._backend_metadata.model_id

    def _resolve_run_card_backend_model_artifact(
        self,
        results: List[Dict[str, Any]],
        backend_selection_resolved: Optional[str],
    ) -> Dict[str, Optional[str]]:
        """Resolve run-card backend model artifact identity from selected attempts."""
        resolved_backend = str(
            backend_selection_resolved or self._backend_metadata.resolved_backend or "",
        ).strip()
        if not resolved_backend:
            return {
                "model_artifact_filename": None,
                "model_artifact_sha256": None,
            }

        for result in results:
            if result.get("status") != "ok":
                continue
            if result.get("backend") != resolved_backend:
                continue
            attempts = result.get("attempts")
            if not isinstance(attempts, list) or not attempts:
                continue
            selected_attempt_index = result.get("selected_attempt_index")
            normalized_selected_index = int(selected_attempt_index) if isinstance(selected_attempt_index, int) else None
            attempt_artifact = self._extract_model_artifact_from_attempts(
                resolved_backend,
                attempts,
                selected_attempt_index=normalized_selected_index,
            )
            if attempt_artifact["model_artifact_filename"] or attempt_artifact["model_artifact_sha256"]:
                return attempt_artifact

        backend_cache = getattr(self, "_depth_backend_cache", {})
        return self._resolve_backend_model_artifact(
            resolved_backend,
            backend=backend_cache.get(resolved_backend),
        )

    def _build_run_card_inputs(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build ordered input records with deterministic source hashes."""
        if self.config.hash_mode == HashMode.NEVER:
            return []

        input_entries: List[tuple[Path, Dict[str, Any]]] = []
        for result in results:
            image_path = result.get("image")
            if not isinstance(image_path, str) or not image_path.strip():
                continue
            candidate = Path(image_path)
            if not candidate.exists() or not candidate.is_file():
                continue
            input_entries.append((candidate, result))

        if not input_entries:
            return []

        input_paths = [candidate for candidate, _ in input_entries]

        try:
            common_root = Path(os.path.commonpath([str(path.parent) for path in input_paths]))
        except ValueError:
            common_root = input_paths[0].parent

        records: List[Dict[str, Any]] = []
        seen_paths: set[str] = set()
        for input_path, result in input_entries:
            try:
                relative_path = str(input_path.relative_to(common_root))
            except ValueError:
                relative_path = input_path.name
            relative_path = relative_path.replace(os.sep, "/")
            if relative_path in seen_paths:
                continue
            seen_paths.add(relative_path)
            source_hash = self._normalize_sha256(result.get("input_sha256"))
            if source_hash is None:
                source_hash = self._compute_or_skip_hash(
                    input_path,
                    manifest_exists=False,
                    for_manifest_write=True,
                )
            if source_hash is None:
                continue
            try:
                size_bytes = input_path.stat().st_size
            except OSError:
                size_bytes = None
            record: Dict[str, Any] = {
                "path": relative_path,
                "sha256": source_hash,
            }
            if isinstance(size_bytes, int):
                record["size_bytes"] = size_bytes
            records.append(record)
        return records

    def _build_run_card_effective_config(
        self,
        *,
        run_card_version: str,
        include_proofs: bool,
    ) -> Dict[str, Any]:
        """Build the replay-oriented effective config surface for the run card."""
        fingerprint = asdict(self.compute_config_fingerprint())
        fingerprint["run_card_version"] = run_card_version
        fingerprint["run_card_include_proofs"] = bool(include_proofs)
        fingerprint["emit_run_card"] = bool(getattr(self.config, "emit_run_card", False))
        fingerprint["vlm_captioning_enabled"] = bool(getattr(self.config, "vlm_captioning_enabled", False))
        fingerprint["vlm_captioning_backend"] = getattr(self.config, "vlm_captioning_backend", "fastvlm")
        fingerprint["vlm_captioning_model"] = getattr(self.config, "vlm_captioning_model", "default")
        fingerprint["vlm_captioning_proxy_format"] = getattr(self.config, "vlm_captioning_proxy_format", "png")
        fingerprint["vlm_captioning_max_side_px"] = int(getattr(self.config, "vlm_captioning_max_side_px", 1600) or 1600)
        fingerprint["fastvlm_timeout_seconds"] = int(getattr(self.config, "fastvlm_timeout_seconds", 180) or 180)
        fingerprint["enable_reconstruction"] = bool(getattr(self.config, "enable_reconstruction", False))
        fingerprint["grouping_mode"] = str(getattr(self.config, "grouping_mode", "single"))
        return fingerprint

    def _build_run_card_model_contract(
        self,
        *,
        results: Optional[List[Dict[str, Any]]] = None,
        backend_selection: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build additive registry-backed model provenance for the run card."""
        if isinstance(backend_selection, dict) and normalize_backend_id(backend_selection.get("resolved")) == "depth_pro":
            artifact_filename = backend_selection.get("model_artifact_filename")
            artifact_sha256 = backend_selection.get("model_artifact_sha256")
            if (not artifact_filename or not artifact_sha256) and results:
                for result in results:
                    if result.get("status") != "ok" or normalize_backend_id(result.get("backend")) != "depth_pro":
                        continue
                    attempt = self._selected_successful_attempt(result)
                    if isinstance(attempt, dict):
                        artifact_filename = artifact_filename or attempt.get("model_artifact_filename")
                        artifact_sha256 = artifact_sha256 or attempt.get("model_artifact_sha256")
                    if artifact_filename or artifact_sha256:
                        break
            return {
                "contract_kind": "local_checkpoint",
                "requested_model_selector": "depth_pro",
                "canonical_model_key": "depth_pro",
                "resolved_repo_id": "apple/ml-depth-pro",
                "resolved_revision": None,
                "model_artifact_filename": artifact_filename,
                "model_artifact_sha256": artifact_sha256,
                "license_id": "apple-machine-learning-research-license",
                "usage_class": "non_commercial_only",
                "requires_non_commercial_ok": True,
                "non_commercial_ok": bool(getattr(self.config, "non_commercial_ok", False)),
                "accept_apple_depth_pro_research_license": bool(
                    getattr(
                        self.config,
                        "accept_apple_depth_pro_research_license",
                        False,
                    )
                ),
                "backend_kind": "depth_pro",
                "accelerator_kind": str(backend_selection.get("device") or getattr(self.config, "depth_device", "cpu")),
                "fallback_chain": [],
                "manifest_schema_version": 1,
            }

        resolved_contract = getattr(self, "_resolved_model_contract", None)
        if resolved_contract is None:
            return None
        if not is_pinned_revision(resolved_contract.revision):
            logger.warning(
                "Skipping run-card model_contract for %s because the resolved revision is not pinned: %r",
                resolved_contract.spec.repo_id,
                resolved_contract.revision,
            )
            return None
        try:
            manifest_payload = load_model_lock_manifest_payload()
            manifest_schema_version = int(
                manifest_payload.get(
                    "manifest_schema_version",
                    manifest_payload.get("version", manifest_payload.get("schema_version", 1)),
                )
            )
        except Exception:
            manifest_schema_version = 1
        return {
            "contract_kind": "hf_revision",
            "requested_model_selector": resolved_contract.requested_selector,
            "resolution_reason": resolved_contract.resolution_reason,
            "canonical_model_key": resolved_contract.canonical_key,
            "resolved_repo_id": resolved_contract.spec.repo_id,
            "resolved_revision": resolved_contract.revision,
            "license_id": resolved_contract.spec.license_id,
            "usage_class": resolved_contract.spec.usage_class.value,
            "requires_non_commercial_ok": resolved_contract.spec.requires_non_commercial_ok,
            "non_commercial_ok": bool(getattr(self.config, "non_commercial_ok", False)),
            "backend_kind": resolved_contract.spec.backend_kind.value,
            "accelerator_kind": resolved_contract.accelerator_kind.value,
            "fallback_chain": list(resolved_contract.fallback_chain),
            "manifest_schema_version": manifest_schema_version,
        }

    def _run_card_output_relative_path(self, path_value: Any) -> Optional[str]:
        """Render an output-root-relative path suitable for run-card summaries."""
        return render_run_card_output_relative_path(path_value, self.output_root)

    @classmethod
    def _extract_run_card_segmentation_metadata(cls, materials_v3_result: Any) -> Optional[Dict[str, Any]]:
        """Extract a replay-safe copy of Materials V3 segmentation metadata."""
        if not isinstance(materials_v3_result, dict):
            return None
        materials_v3_metadata = materials_v3_result.get("materials_v3_metadata")
        if not isinstance(materials_v3_metadata, dict):
            return None
        segmentation_metadata = materials_v3_metadata.get("segmentation_metadata")
        if not isinstance(segmentation_metadata, dict):
            return None
        normalized = copy.deepcopy(segmentation_metadata)

        material_masks = materials_v3_result.get("material_masks")
        if isinstance(material_masks, dict) and normalized.get("mask_count") is None:
            normalized["mask_count"] = len(material_masks)

        pixel_ops = materials_v3_result.get("materials_v3_pixel_ops")
        if isinstance(pixel_ops, dict):
            applied = pixel_ops.get("applied")
            blocked = pixel_ops.get("blocked")
            if isinstance(applied, list) and normalized.get("pixel_ops_applied_count") is None:
                normalized["pixel_ops_applied_count"] = len(applied)
            if isinstance(blocked, list) and normalized.get("pixel_ops_blocked_count") is None:
                normalized["pixel_ops_blocked_count"] = len(blocked)

            passthrough = pixel_ops.get("passthrough_status")
            if isinstance(passthrough, dict) and not isinstance(
                normalized.get("pixel_ops_passthrough"),
                dict,
            ):
                normalized["pixel_ops_passthrough"] = copy.deepcopy(passthrough)

            if normalized.get("pixel_ops_blocked_count") is None:
                blocked_reasons = cls._pixel_ops_blocked_reasons(pixel_ops)
                if blocked_reasons:
                    normalized["pixel_ops_blocked_count"] = sum(blocked_reasons.values())
        return normalized

    @staticmethod
    def _coerce_nonnegative_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return None
        if coerced < 0:
            return None
        return coerced

    @classmethod
    def _build_run_card_materials_summary(
        cls,
        segmentation_metadata: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Summarize Materials V3/SAM2 semantics without conflating V2 handoff."""
        mask_count = cls._coerce_nonnegative_int(segmentation_metadata.get("mask_count")) or 0

        passthrough = segmentation_metadata.get("pixel_ops_passthrough")
        passthrough_details = passthrough.get("details") if isinstance(passthrough, Mapping) else None
        passthrough_details = passthrough_details if isinstance(passthrough_details, Mapping) else {}

        pixel_ops_applied_count = cls._coerce_nonnegative_int(
            segmentation_metadata.get("pixel_ops_applied_count"),
        )
        if pixel_ops_applied_count is None:
            pixel_ops_applied_count = cls._coerce_nonnegative_int(
                passthrough_details.get("applied_ops_count"),
            )
        if pixel_ops_applied_count is None:
            pixel_ops_applied_count = 0

        blocked_count = cls._coerce_nonnegative_int(
            segmentation_metadata.get("pixel_ops_blocked_count"),
        )
        if blocked_count is None:
            blocked_reasons = passthrough_details.get("blocked_reasons")
            if isinstance(blocked_reasons, Mapping):
                blocked_count = sum(
                    count
                    for count in (cls._coerce_nonnegative_int(value) for value in blocked_reasons.values())
                    if count is not None
                )
        if blocked_count is None:
            blocked_count = 0

        passthrough_code = None
        if isinstance(passthrough, Mapping):
            raw_code = passthrough.get("code")
            if isinstance(raw_code, str) and raw_code.strip():
                passthrough_code = raw_code.strip()

        return {
            "masks_generated": mask_count > 0,
            "mask_count": mask_count,
            "pixel_ops_applied": pixel_ops_applied_count > 0,
            "pixel_ops_applied_count": pixel_ops_applied_count,
            "blocked_count": blocked_count,
            "passthrough_code": passthrough_code,
        }

    @classmethod
    def _build_run_card_segmentation_performance_warnings(
        cls,
        segmentation_metadata: Mapping[str, Any],
        *,
        runtime_s: Optional[Any],
        materials_summary: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        backend_name = str(segmentation_metadata.get("backend") or "").strip().lower()
        if backend_name != "sam2":
            return []

        timing_ms = segmentation_metadata.get("timing_ms")
        if not isinstance(timing_ms, Mapping):
            return []

        raw_sam2_runtime_ms = timing_ms.get("backend_segment")
        if raw_sam2_runtime_ms is None or runtime_s is None:
            return []

        try:
            sam2_runtime_ms = float(raw_sam2_runtime_ms)
            total_runtime_ms = float(runtime_s) * 1000.0
        except (TypeError, ValueError):
            return []

        if (
            not np.isfinite(sam2_runtime_ms)
            or not np.isfinite(total_runtime_ms)
            or sam2_runtime_ms < 0
            or total_runtime_ms <= 0
        ):
            return []

        mask_count = cls._coerce_nonnegative_int(materials_summary.get("mask_count")) or 0
        pixel_ops_applied_count = cls._coerce_nonnegative_int(materials_summary.get("pixel_ops_applied_count")) or 0
        sam2_runtime_share = sam2_runtime_ms / total_runtime_ms
        if sam2_runtime_share < 0.90 or mask_count <= 0 or pixel_ops_applied_count != 0:
            return []

        return [
            {
                "code": APEX_MATERIALS_SEGMENTATION_DOMINATES_NO_PIXEL_OPS,
                "severity": "advisory",
                "message": "SAM2 dominated runtime but no material pixel operations were applied.",
                "details": {
                    "sam2_runtime_ms": sam2_runtime_ms,
                    "total_runtime_ms": total_runtime_ms,
                    "sam2_runtime_share": round(sam2_runtime_share, 4),
                    "pixel_ops_applied_count": pixel_ops_applied_count,
                    "mask_count": mask_count,
                },
            }
        ]

    def _build_run_card_segmentation_status(
        self,
        segmentation_metadata: Optional[Dict[str, Any]],
        result_failure_info: Optional[Mapping[str, Any]] = None,
        runtime_s: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build explicit segmentation execution status for run-card summaries.

        When the per-image manifest is absent the segmentation cache is empty,
        so ``segmentation_metadata`` is None even though the failure was a
        well-defined gate violation already recorded on the result row. In that
        case prefer the structured ``error_code`` / ``error_details`` from the
        result over the historical ``missing_evidence`` placeholder.
        """
        if not hasattr(self, "config"):
            return None
        backend = getattr(self.config, "material_segmentation_backend", None)
        strict_backend = bool(getattr(self.config, "strict_backend", False))
        if not bool(getattr(self.config, "enable_materials_v3", False)):
            return {
                "status": "not_requested",
                "enabled": False,
                "reason": "materials_v3_disabled",
                "backend": backend,
                "strict_backend": strict_backend,
            }
        if not bool(getattr(self.config, "enable_material_segmentation", False)):
            return {
                "status": "not_requested",
                "enabled": False,
                "reason": "material_segmentation_disabled",
                "backend": backend,
                "strict_backend": strict_backend,
            }
        if isinstance(segmentation_metadata, dict):
            materials_summary = self._build_run_card_materials_summary(segmentation_metadata)
            return {
                "status": str(segmentation_metadata.get("status") or "ok"),
                "enabled": True,
                "backend": segmentation_metadata.get("backend") or backend,
                "strict_backend": strict_backend,
                "mask_artifact_path": self._run_card_output_relative_path(
                    segmentation_metadata.get("mask_artifact_path"),
                ),
                "mask_count": segmentation_metadata.get("mask_count"),
                "confidence_summary": segmentation_metadata.get("confidence_summary"),
                "warnings": list(segmentation_metadata.get("warnings") or []),
                "errors": list(segmentation_metadata.get("errors") or []),
                "pixel_ops_passthrough": segmentation_metadata.get("pixel_ops_passthrough"),
                "materials_summary": materials_summary,
                "performance_warnings": self._build_run_card_segmentation_performance_warnings(
                    segmentation_metadata,
                    runtime_s=runtime_s,
                    materials_summary=materials_summary,
                ),
            }
        failure_code: Optional[str] = None
        failure_details: Optional[Mapping[str, Any]] = None
        if isinstance(result_failure_info, Mapping):
            raw_code = result_failure_info.get("error_code")
            if isinstance(raw_code, str) and raw_code.strip():
                failure_code = raw_code.strip()
            raw_details = result_failure_info.get("error_details")
            if isinstance(raw_details, Mapping):
                failure_details = dict(raw_details)
        if failure_code is not None:
            return {
                "status": "failed",
                "enabled": True,
                "backend": backend,
                "strict_backend": strict_backend,
                "failure_code": failure_code,
                "failure_details": failure_details,
                "warnings": [],
                "errors": [failure_code],
            }
        return {
            "status": "missing_evidence" if strict_backend else "not_recorded",
            "enabled": True,
            "backend": backend,
            "strict_backend": strict_backend,
            "warnings": ["SEGMENTATION_EVIDENCE_MISSING"],
            "errors": ["SEGMENTATION_EVIDENCE_MISSING"] if strict_backend else [],
        }

    def _build_run_card_result_summary(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build a compact per-image execution summary for replay triage."""
        cached_segmentation_metadata = getattr(
            self,
            "_active_run_card_segmentation_metadata",
            {},
        )
        summary_rows: List[Dict[str, Any]] = []
        for result in results:
            image_path = result.get("image")
            if not isinstance(image_path, str) or not image_path.strip():
                continue
            manifest_path = result.get("manifest")
            segmentation_metadata = None
            if isinstance(cached_segmentation_metadata, dict) and isinstance(manifest_path, str) and manifest_path.strip():
                cached_metadata = cached_segmentation_metadata.get(manifest_path)
                if isinstance(cached_metadata, dict):
                    segmentation_metadata = copy.deepcopy(cached_metadata)
            row = {
                "image": Path(image_path).name,
                "status": result.get("status"),
                "backend": result.get("backend"),
                "runtime_s": result.get("runtime_s"),
                "manifest_path": self._run_card_output_relative_path(manifest_path),
                "error_code": result.get("error_code"),
                "error_message": result.get("error"),
                "error_details": result.get("error_details"),
                "segmentation_metadata": segmentation_metadata,
                "quality_gate": resolve_result_quality_gate(result),
                "capability": build_orchestrator_result_capability_report(
                    result,
                    requested_backend=getattr(
                        getattr(self, "_backend_metadata", None),
                        "requested_backend",
                        None,
                    ),
                    resolution_reason=getattr(
                        getattr(self, "_backend_metadata", None),
                        "resolution_reason",
                        None,
                    ),
                ),
            }
            segmentation_status = self._build_run_card_segmentation_status(
                segmentation_metadata,
                result_failure_info=result,
                runtime_s=result.get("runtime_s"),
            )
            if segmentation_status is not None:
                row["segmentation_status"] = segmentation_status
            captioning_status = result.get("vlm_captioning_status")
            if isinstance(captioning_status, Mapping):
                row["captioning_status"] = copy.deepcopy(dict(captioning_status))
            summary_rows.append(row)
        return summary_rows

    def _build_run_card_captioning_status(self, results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Build aggregate advisory captioning status for the run card."""
        if not bool(getattr(self.config, "vlm_captioning_enabled", False)):
            return None
        selector = str(getattr(self.config, "vlm_captioning_model", "default") or "default").strip()
        model_path, model_role, model_id = self._resolve_vlm_captioning_model_path(selector)
        sidecar_count = sum(1 for result in results if result.get("vlm_caption_sidecar_path"))
        failed_count = 0
        for result in results:
            status = result.get("vlm_captioning_status")
            if not isinstance(status, Mapping):
                failed_count += 1
                continue
            failed_count += int(status.get("failed_count") or 0)
        return {
            "enabled": True,
            "backend": str(getattr(self.config, "vlm_captioning_backend", "fastvlm") or "fastvlm").strip().lower(),
            "model_role": model_role or "custom",
            "model_id": model_id,
            "model_path": str(model_path),
            "role": "advisory",
            "sidecar_count": sidecar_count,
            "failed_count": failed_count,
            "used_for_quality_gate": False,
        }

    def _emit_run_card(
        self,
        batch_id: str,
        start_time: str,
        end_time: str,
        results: List[Dict[str, Any]],
        runtime_stats: Dict[str, Any],
        outliers: List[Dict[str, Any]],
        batch_manifest_path: Optional[Path] = None,
        requested_backend_defect: Optional[str] = None,
    ) -> None:
        """Emit run card for batch reproducibility.

        Hardened JSON serialization:
        - Explicit ConfigFingerprint handling
        - Dataclass-safe conversion
        - Enum normalization
        - Path normalization
        - Deterministic fallback
        """

        run_card_path = self.output_root / f"run_card_{batch_id}.json"

        artifact_paths = self._collect_run_card_artifact_paths(
            results,
            batch_manifest_path=batch_manifest_path,
        )
        artifact_index = _build_artifact_index(
            self.output_root,
            artifact_paths,
        )
        run_card_version = str(getattr(self.config, "run_card_version", "v1") or "v1").strip().lower()
        if run_card_version not in {"v1", "v2"}:
            logger.warning(
                "Unsupported run_card_version=%r; falling back to v1",
                run_card_version,
            )
            run_card_version = "v1"
        artifact_tree = None
        if run_card_version == "v2":
            include_proofs_config = getattr(self.config, "run_card_include_proofs", False)
            include_proofs = include_proofs_config
            if isinstance(include_proofs_config, str):
                include_proofs = include_proofs_config.strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            artifact_tree = build_artifact_tree(artifact_index, include_proofs=bool(include_proofs))
            artifact_merkle_root = None
        else:
            artifact_merkle_root = _compute_artifact_merkle_root(artifact_index)
        backend_summary = self._compute_backend_summary(results)
        if requested_backend_defect is None:
            requested_backend_defect = self._requested_backend_fulfillment_defect(
                results,
                backend_summary,
            )
        if requested_backend_defect is not None:
            logger.error(requested_backend_defect)
            backend_summary = {
                **backend_summary,
                "requested_backend_status": "not_honored",
                "requested_backend_defect": requested_backend_defect,
            }
        backend_selection = self._build_backend_selection_payload(
            results,
            backend_summary,
        )
        captioning_status = self._build_run_card_captioning_status(results)

        artifact_summary_payload: dict[str, Any] = (
            {"artifact_tree": artifact_tree} if run_card_version == "v2" else {"artifact_merkle_root": artifact_merkle_root}
        )
        include_tree_proofs = bool(artifact_tree.get("proofs")) if isinstance(artifact_tree, dict) else False

        run_card = {
            "run_card_version": run_card_version,
            "batch_id": batch_id,
            "start_time": start_time,
            "end_time": end_time,
            "config_fingerprint": self._build_run_card_config_fingerprint(
                backend_selection=backend_selection,
                run_card_version=run_card_version,
                include_proofs=include_tree_proofs,
            ),
            "inputs": self._build_run_card_inputs(results),
            "effective_config": self._build_run_card_effective_config(
                run_card_version=run_card_version,
                include_proofs=include_tree_proofs,
            ),
            "result_summary": self._build_run_card_result_summary(results),
            "backend_selection": backend_selection,
            "backend_summary": backend_summary,
            "environment": self.environment,
            "git_revision": {
                "v3": self.v3_git,
                "v2": self.v2_git,
            },
            "runtime_stats": runtime_stats,
            "outliers": outliers,
            "total_images": len(results),
            "success_count": sum(1 for r in results if r.get("status") == "ok"),
            "error_count": sum(1 for r in results if r.get("status") == "error"),
            "artifact_index": artifact_index,
            **artifact_summary_payload,
        }
        if captioning_status is not None:
            run_card["captioning_status"] = captioning_status
        model_contract = self._build_run_card_model_contract(
            results=results,
            backend_selection=backend_selection,
        )
        if model_contract is not None:
            run_card["model_contract"] = model_contract
        run_card["licensing"] = build_runtime_licensing_manifest(
            model_contract=model_contract,
            config=self.config,
        )

        def _json_default(obj: Any) -> Any:
            # --- ConfigFingerprint ---
            if isinstance(obj, ConfigFingerprint):
                fingerprint_payload = {
                    "model_variant": obj.model_variant,
                    "depth_quantization": obj.depth_quantization,
                    "depth_device": obj.depth_device,
                    "preset": obj.preset,
                    "v2_preset": obj.v2_preset,
                    "v2_device": obj.v2_device,
                    "v2_upscaler_backend": obj.v2_upscaler_backend,
                }
                canonical_json = json.dumps(
                    fingerprint_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                return {
                    **fingerprint_payload,
                    "hash_algorithm": "sha256",
                    "canonical_json": canonical_json,
                    "sha256": hashlib.sha256(
                        canonical_json.encode(
                            "utf-8",
                        ),
                    ).hexdigest(),
                }

            # --- Enum handling ---
            if isinstance(obj, Enum):
                return obj.value

            # --- Path handling ---
            if isinstance(obj, Path):
                return str(obj)

            # --- Dataclass-safe conversion ---
            if hasattr(obj, "__dataclass_fields__"):
                return {k: getattr(obj, k) for k in obj.__dataclass_fields__}

            # --- Explicit to_dict support ---
            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                return obj.to_dict()

            # --- Controlled __dict__ fallback (avoid deep recursion) ---
            if hasattr(
                obj,
                "__dict__",
            ) and not isinstance(
                obj,
                (np.ndarray,),
            ):
                return {k: v for k, v in vars(obj).items() if not k.startswith("_")}

            # --- Final deterministic fallback ---
            return str(obj)

        run_card_write_attempted = False
        sidecar_write_attempted = False
        run_card_self_attestation_path = run_card_path.with_suffix(".self.json")
        try:
            run_card_integrity_payload = {
                "path": self._run_card_output_relative_path(str(run_card_path)),
                "self_indexing": "excluded_self_hash_cycle",
            }
            integrity_canonical_payload = {
                **run_card,
                "run_card_integrity": run_card_integrity_payload,
            }
            integrity_payload_bytes = canonicalize_json(integrity_canonical_payload)
            run_card_integrity_payload["canonical_payload_sha256"] = hashlib.sha256(integrity_payload_bytes).hexdigest()
            run_card["run_card_integrity"] = run_card_integrity_payload
            serialized_run_card = json.loads(
                dumps_json(
                    run_card,
                    default=_json_default,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            _validate_run_card_payload(
                serialized_run_card,
                schema_version=run_card_version,
            )
            _validate_run_card_backend_semantics(serialized_run_card)

            run_card_text = dumps_json(
                run_card,
                indent=2,
                default=_json_default,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            run_card_self_attestation = {
                "run_card_path": self._run_card_output_relative_path(str(run_card_path)),
                "self_indexing": "excluded_self_hash_cycle",
                "final_run_card_sha256": hashlib.sha256(run_card_text.encode("utf-8")).hexdigest(),
                "hash_algorithm": "sha256",
            }
            sidecar_text = dumps_json(
                run_card_self_attestation,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            run_card_write_attempted = True
            with open(run_card_path, "w", encoding="utf-8") as f:
                f.write(run_card_text)
            sidecar_write_attempted = True
            with open(run_card_self_attestation_path, "w", encoding="utf-8") as f:
                f.write(sidecar_text)
        except (OSError, TypeError, ValueError, RuntimeError):
            if run_card_write_attempted or sidecar_write_attempted:
                for partial_path in (run_card_path, run_card_self_attestation_path):
                    try:
                        partial_path.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError:
                        logger.warning(
                            "Failed to remove partial run-card artifact after emission failure: %s",
                            partial_path,
                            exc_info=True,
                        )
            logger.exception(
                "Run card emission failed" " for batch_id=%s" " (output: %s)." " Continuing without" " run card.",
                batch_id,
                run_card_path,
            )
            return

        logger.info(
            "Run card emitted: %s",
            run_card_path,
        )
