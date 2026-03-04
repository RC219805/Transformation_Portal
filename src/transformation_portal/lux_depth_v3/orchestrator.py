"""Orchestrator for V3 depth + V2 enhancement pipeline.

Two-stage pipeline:
1. Stage A (V3): Generate depth assets using DA3 (Inference -> Post-Processing -> Write)
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

import datetime
import hashlib
import io
import json
import logging
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from functools import lru_cache
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# Phase 3: xxHash support (optional dependency)
try:
    import xxhash

    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    xxhash = None  # type: ignore

from ..depth.backends.protocol import LicenseRestrictionError

# Backend registry for depth estimation
from ..depth.backends.registry import DepthBackendRegistry
from ..ingest.canonical_json import dump_json, dumps_json
from ..spatial_ai.reconstruction.contracts import LicenseRestrictionError as ReconstructionLicenseRestrictionError
from .batch_stats import compute_batch_runtime_stats, detect_runtime_outliers
from .camera_metadata_loader import load_scene_cameras

# Note: Imports adjusted to relative for package context compatibility
from .config import DA3Config, EnhanceConfig, ModelVariant
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
    MaterialsV3Metadata,
    ReproMetadata,
    TimingMetadata,
    V2Metadata,
    capture_environment,
    compute_file_sha256,
    get_git_revision,
)
from .pbr import generate_pbr_maps
from .pbr_writer import write_pbr_maps
from .postprocessing import Postprocessor
from .provenance import ExiftoolNotFoundError, ProvenanceError, capture_provenance
from .reconstruction_runner import run_scene_reconstruction
from .scene_context import SceneContext
from .scene_groups import build_scene_groups
from .security import HashMode, sanitize_file_stem, sanitize_path_component_nonlossy
from .v2_runner import V2Runner, find_v2_report

logger = logging.getLogger(__name__)


class ApexStrictGateError(RuntimeError):
    """Raised when APEX strict quality gates are violated."""

    def __init__(self, code: str, message: str, details: Optional[Dict[str, Any]] = None):
        self.code = code
        self.details = details or {}
        super().__init__(f"[{code}] {message}")


class _MaskSerializationRejected(RuntimeError):
    """Internal signal for non-fatal mask serialization rejection."""


def _log_dependency_status() -> dict:
    """Log startup dependency availability report.

    Reports status of optional dependencies with actionable guidance.
    Makes warnings explicit, not vague.

    Returns:
        Dictionary with dependency status for testing/debugging
    """
    status = {}

    # Check torch
    try:
        import torch

        status["torch"] = True
        version = getattr(torch, "__version__", "unknown")
        logger.debug(f"torch {version} available")
    except ImportError:
        status["torch"] = False
        logger.info("torch not available - ML features disabled. Install: pip install torch")

    # Check transformers
    try:
        import transformers

        status["transformers"] = True
        version = getattr(transformers, "__version__", "unknown")
        logger.debug(f"transformers {version} available")
    except ImportError:
        status["transformers"] = False
        logger.info("transformers not available - depth models disabled. Install: pip install transformers")

    # Check coremltools (optional)
    try:
        import coremltools

        status["coremltools"] = True
        version = getattr(coremltools, "__version__", "unknown")
        logger.debug(f"coremltools {version} available")
    except ImportError:
        status["coremltools"] = False
        logger.debug("coremltools not available (optional). Install: pip install coremltools")

    # Check scikit-image (optional for some features)
    try:
        import skimage

        status["scikit-image"] = True
        version = getattr(skimage, "__version__", "unknown")
        logger.debug(f"scikit-image {version} available")
    except ImportError:
        status["scikit-image"] = False
        logger.debug("scikit-image not available (optional for advanced filtering)")

    # Check numba (optional performance enhancement)
    try:
        import numba

        status["numba"] = True
        logger.debug(f"numba {numba.__version__} available - performance optimizations enabled")
    except ImportError:
        status["numba"] = False
        logger.debug("numba not available - using NumPy fallback (30-50% slower for some operations)")

    # Check HF_TOKEN for model downloads
    hf_token = os.environ.get("HF_TOKEN")
    status["hf_token"] = bool(hf_token)
    if hf_token:
        logger.debug("HF_TOKEN present - authenticated model downloads enabled")
    else:
        logger.debug("HF_TOKEN not set - using unauthenticated downloads (rate limits apply, slower warm starts)")
        logger.debug("  Set HF_TOKEN for faster downloads: export HF_TOKEN=<your_token>")

    return status


@lru_cache(maxsize=128)
def _load_manifest_cached(manifest_path: str, mtime: float) -> CombinedManifest:
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


def make_output_key(input_path: Path, input_root: Path, use_xxhash: bool = XXHASH_AVAILABLE) -> Path:
    """Compute a stable, sanitized output key for a given input image.

    The final key preserves the relative directory shape under ``input_root``
    (when possible) and emits ``<stem>_<ext|noext>_<hash8>`` as the terminal
    component. The 8-character suffix is derived from the POSIX-style relative
    path, using xxh64 when enabled/available or SHA-1 otherwise.
    """
    input_resolved = input_path.resolve()
    root_resolved = input_root.resolve()

    try:
        relpath = input_resolved.relative_to(root_resolved)
    except ValueError:
        logger.warning(
            "%s is not relative to %s, using flat naming",
            input_path,
            input_root,
        )
        relpath = Path(input_resolved.name)

    rel_dir = relpath.parent
    name = relpath.stem
    ext = relpath.suffix

    sanitized_parts = [sanitize_path_component_nonlossy(p) for p in rel_dir.parts]
    ext_label = sanitize_path_component_nonlossy(ext.lstrip(".").lower() if ext else "noext")

    hash_input = relpath.as_posix().encode("utf-8")

    if use_xxhash and XXHASH_AVAILABLE:
        hash_suffix = xxhash.xxh64(hash_input).hexdigest()[:8]
    else:
        hash_suffix = hashlib.sha1(hash_input, usedforsecurity=False).hexdigest()[:8]

    stem_sanitized = sanitize_path_component_nonlossy(name)
    key_name = f"{stem_sanitized}_{ext_label}_{hash_suffix}"

    return Path(*sanitized_parts, key_name) if sanitized_parts else Path(key_name)


def _run_card_schema_path() -> Path:
    """Return repository-local run card schema path."""
    return Path(__file__).resolve().parents[3] / "docs" / "schemas" / "run_card" / "run_card.v1.schema.json"


@lru_cache(maxsize=1)
def _load_run_card_schema(schema_path_str: str) -> Dict[str, Any]:
    """Load run card JSON schema once per process."""
    schema_path = Path(schema_path_str)
    with open(schema_path, "r", encoding="utf-8") as schema_file:
        return json.load(schema_file)


@lru_cache(maxsize=1)
def _load_run_card_validator(schema_path_str: str) -> Any:
    """Build cached Draft202012 validator for run card schema."""
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - dependency is pinned, guard for runtime safety
        raise RuntimeError("jsonschema dependency is required for run card schema validation") from exc

    schema = _load_run_card_schema(schema_path_str)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.exceptions.SchemaError as exc:
        raise RuntimeError(f"Run card schema is invalid: {exc.message}") from exc
    return jsonschema.Draft202012Validator(schema)


def _validate_run_card_payload(payload: Dict[str, Any], schema_path: Path) -> None:
    """Validate run card payload against run_card.v1 schema."""
    validator = _load_run_card_validator(str(schema_path))
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if not errors:
        return

    formatted = []
    for error in errors:
        path = ".".join(str(p) for p in error.path) or "<root>"
        formatted.append(f"{path}: {error.message}")
    raise RuntimeError("Run card schema validation failed: " + "; ".join(formatted))


def _validate_run_card_backend_semantics(payload: Dict[str, Any]) -> None:
    """Validate backend resolution semantics for run-card transparency."""
    backend_selection = payload.get("backend_selection")
    backend_summary = payload.get("backend_summary")
    if not isinstance(backend_selection, dict) or not isinstance(backend_summary, dict):
        return

    final_backends_used = backend_summary.get("final_backends_used")
    if not isinstance(final_backends_used, list):
        return

    success_count = payload.get("success_count")
    if not isinstance(success_count, int):
        success_count = 0

    if not final_backends_used:
        if success_count > 0:
            raise RuntimeError(
                "Run card backend semantics validation failed: "
                "backend_summary.final_backends_used must be non-empty when success_count > 0."
            )
        return

    primary_backend = final_backends_used[0]
    if not isinstance(primary_backend, str) or not primary_backend:
        raise RuntimeError(
            "Run card backend semantics validation failed: "
            "backend_summary.final_backends_used[0] must be a non-empty string."
        )

    summary_primary = backend_summary.get("primary_backend")
    if summary_primary != primary_backend:
        raise RuntimeError(
            "Run card backend semantics validation failed: "
            "backend_summary.primary_backend must equal backend_summary.final_backends_used[0]."
        )

    resolved = backend_selection.get("resolved")
    if not isinstance(resolved, str) or not resolved:
        raise RuntimeError(
            "Run card backend semantics validation failed: " "backend_selection.resolved must be a non-empty string."
        )

    if resolved != primary_backend:
        raise RuntimeError(
            "Run card backend semantics validation failed: "
            "backend_selection.resolved must match backend_summary.final_backends_used[0]."
        )

    logical_backend = backend_selection.get("logical_backend")
    resolved_engine = backend_selection.get("resolved_engine")
    wrapper_declared = logical_backend is not None or resolved_engine is not None
    if not wrapper_declared:
        return

    if not isinstance(logical_backend, str) or not logical_backend:
        raise RuntimeError(
            "Run card backend semantics validation failed: "
            "backend_selection.logical_backend must be a non-empty string when wrapper semantics are declared."
        )
    if not isinstance(resolved_engine, str) or not resolved_engine:
        raise RuntimeError(
            "Run card backend semantics validation failed: "
            "backend_selection.resolved_engine must be a non-empty string when wrapper semantics are declared."
        )
    if logical_backend == resolved_engine:
        raise RuntimeError(
            "Run card backend semantics validation failed: "
            "backend_selection.logical_backend and backend_selection.resolved_engine must differ."
        )
    if resolved_engine != primary_backend:
        raise RuntimeError(
            "Run card backend semantics validation failed: "
            "backend_selection.resolved_engine must match backend_summary.final_backends_used[0]."
        )

    fallback_images = backend_summary.get("fallback_images")
    if isinstance(fallback_images, int) and fallback_images != 0:
        raise RuntimeError(
            "Run card backend semantics validation failed: "
            "wrapper semantics are only valid when backend_summary.fallback_images == 0."
        )


def _infer_artifact_type(relative_path: str) -> str:
    """Infer canonical artifact type from output-root relative path."""
    rel = relative_path.lower()
    name = Path(rel).name

    if rel.startswith("segmentation/"):
        if name.endswith(".npz"):
            return "segmentation_mask_npz"
        return "segmentation_aux"

    if rel.startswith("depth/"):
        if name.endswith("_metadata.json"):
            return "depth_metadata"
        if name.endswith(".png"):
            return "depth_u16_png"
        if name.endswith(".npy"):
            return "depth_float_npy"
        return "depth_aux"

    if rel.startswith("v2/"):
        if name.endswith("_report.json"):
            return "v2_report"
        return "v2_output"

    if rel.startswith("manifests/"):
        if name.startswith("batch_") and name.endswith(".json"):
            return "batch_manifest"
        if name.endswith("_provenance.json"):
            return "provenance_sidecar"
        if name.endswith("_combined.json"):
            return "combined_manifest"
        return "manifest_aux"

    if rel.startswith("logs/"):
        return "v2_log"

    if rel.startswith("pbr/"):
        if "normal" in name:
            return "pbr_normal"
        if "roughness" in name:
            return "pbr_roughness"
        if name.startswith("ao_") or "_ao" in name:
            return "pbr_ao"
        return "pbr_aux"

    if rel.startswith("reconstruction/"):
        if name.endswith("_reconstruction_report.json"):
            return "reconstruction_report"
        return "reconstruction_aux"

    return "artifact"


def _v2_log_filename(output_key_name: str, batch_id: Optional[str] = None) -> str:
    """Build deterministic, batch-scoped V2 log filename."""
    filename = f"v2_{output_key_name}"
    if batch_id:
        filename += f"__{sanitize_path_component_nonlossy(str(batch_id))}"
    return f"{filename}.log"


def _build_artifact_index(output_root: Path, artifact_paths: List[Path]) -> List[Dict[str, Any]]:
    """Build deterministic artifact index with size and SHA256."""
    root_resolved = output_root.resolve()
    index_by_relative_path: Dict[str, Dict[str, Any]] = {}

    for candidate in artifact_paths:
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.debug(f"Skipping artifact path due to resolution error ({candidate}): {exc}")
            continue

        if not resolved.is_file():
            continue

        try:
            relative_path = resolved.relative_to(root_resolved).as_posix()
        except ValueError:
            logger.debug(f"Skipping artifact outside output root: {resolved}")
            continue

        if relative_path in index_by_relative_path:
            continue

        stat = resolved.stat()
        index_by_relative_path[relative_path] = {
            "artifact_type": _infer_artifact_type(relative_path),
            "path": relative_path,
            "relative_path": relative_path,
            "size_bytes": stat.st_size,
            "sha256": compute_file_sha256(resolved),
        }

    return [index_by_relative_path[path] for path in sorted(index_by_relative_path)]


def _compute_artifact_merkle_root(artifact_index: List[Dict[str, Any]]) -> str:
    """Compute deterministic run-card Merkle root over artifact SHA256 digests."""
    sorted_artifacts = sorted(artifact_index, key=lambda item: item["relative_path"])
    leaves = []
    for artifact in sorted_artifacts:
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"Invalid artifact sha256 in run card index: {digest!r}")
        leaves.append(bytes.fromhex(digest))

    return hashlib.sha256(b"".join(leaves)).hexdigest()


class EnhanceOrchestrator:
    """Orchestrates V3 depth generation + V2 enhancement pipeline.

    Attributes:
        config: Enhancement configuration
        output_root: Base directory for all outputs
        verify_outputs: If True, verify cached outputs exist on disk before skipping (defensive check)
    """

    def __init__(self, config: EnhanceConfig, output_root: Path, verify_outputs: bool = True):
        """Initialize the orchestrator.

        Args:
            config: Enhancement configuration object
            output_root: Base directory to store outputs and manifests
            verify_outputs: Whether to verify cached outputs exist before skipping (default: True)
        """
        # Log dependency status on first initialization
        _log_dependency_status()

        self.config = config
        self.output_root = Path(output_root)
        self.verify_outputs = verify_outputs

        if config.hash_mode == HashMode.NEVER:
            logger.warning("Hash mode set to 'never' - manifests will lack integrity verification.")

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

        # Initialize V3 Configuration (Priority: User Override > Preset > Default)
        if config.preset is not None:
            da3_config = DA3Config.from_preset(config.preset)
            if config.model_variant is not None:
                logger.info(f"Overriding preset model with user choice: {config.model_variant.value.display_name}")
                da3_config.model_variant = config.model_variant
            else:
                config.model_variant = da3_config.model_variant
        else:
            model = config.model_variant if config.model_variant is not None else ModelVariant.METRIC_LARGE
            da3_config = DA3Config(model_variant=model)
            config.model_variant = model

        da3_config.device.device = config.depth_device

        # Initialize Depth Backend via Registry (ADR-019)
        self._initialize_depth_backend()

        # Initialize Postprocessor (FIX: Ensures refine_edges/bilateral settings from preset are applied)
        self.postprocessor = Postprocessor(da3_config.postprocessing)

        # Initialize Materials V3 Engine (if enabled)
        if config.enable_materials_v3:
            from .materials_v3 import MaterialsV3Engine

            self.materials_v3_engine = MaterialsV3Engine(config)
            logger.info("Materials V3 surface-aware finishing enabled")
        else:
            self.materials_v3_engine = None

        # Initialize V2 Runner and Environment (with fail-fast validation)
        if config.enable_v2 and config.v2_preset is not None:
            self.v2_runner = V2Runner()
            # Fail-fast: Validate V2 script exists before processing
            if not self.v2_runner.script_path.exists():
                raise FileNotFoundError(
                    f"V2 enhancement script not found: {self.v2_runner.script_path}\n"
                    f"Required location: scripts/enhance_image.py in repository root\n"
                    f"\nOptions:\n"
                    f"  1. Create the V2 enhancement script at the expected location\n"
                    f"  2. Set enable_v2=False for PBR-only workflows\n"
                    f"  3. Set v2_preset=None to skip V2 stage"
                )
            logger.info(f"V2 enhancement enabled with script: {self.v2_runner.script_path}")
        else:
            self.v2_runner = None
            logger.info("V2 enhancement disabled (PBR-only mode)")

        # Adjusted path logic for src/transformation_portal/lux_depth_v3 location
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
            logger.debug(f"GPU/MPS device detected - limiting workers to {self.max_workers} for VRAM management")
        else:
            # CPU backend: moderate parallelism for I/O-bound operations
            if max_workers_override is not None:
                self.max_workers = max_workers_override
            else:
                self.max_workers = config.max_parallel_workers or max(1, cpu_count() - 1)
            self.max_gpu_workers = self.max_workers

        self._use_parallel = config.enable_parallel_processing
        logger.debug(f"Parallel processing: {'enabled' if self._use_parallel else 'disabled'} (workers={self.max_workers})")

        # Phase 2: Content-addressable depth cache (opt-in)
        self.depth_cache = (
            DepthCache(self.output_root, max_size_gb=config.depth_cache_max_size_gb) if config.enable_depth_cache else None
        )
        if self.depth_cache:
            logger.info(f"Depth cache enabled: {self.depth_cache.cache_dir}")

        # Injectable seam for lightweight reconstruction tests.
        self.run_scene_reconstruction_fn: Callable[..., Path] = run_scene_reconstruction

    def _initialize_depth_backend(self) -> None:
        """Initialize depth backend using registry (ADR-019).

        Implements backend selection with fallback logic:
        1. Try requested backend (from config.depth_backend, default "da3")
        2. Check availability (checkpoint + dependencies)
        3. Fallback to configured operational chain (default: da3 -> da2)
        4. Optionally fallback to synthetic in explicit test/CI mode
        5. Record selection decision in metadata
        """
        requested = self.config.depth_backend or "da3"
        self._depth_registry = DepthBackendRegistry()
        self._depth_backend_cache: Dict[str, Any] = {}

        allow_synthetic = bool(self.config.allow_synthetic_fallback) or os.getenv("TP_ALLOW_SYNTHETIC_FALLBACK") == "1"
        candidate_chain: List[str] = [requested]
        for fallback_backend in ("da3", "da2"):
            if fallback_backend not in candidate_chain:
                candidate_chain.append(fallback_backend)
        if allow_synthetic and "synthetic" not in candidate_chain:
            candidate_chain.append("synthetic")

        try:
            backend = None
            resolved = None
            status = "error"
            reason = None
            init_errors: Dict[str, str] = {}

            for index, backend_id in enumerate(candidate_chain):
                try:
                    candidate_backend = self._depth_registry.get_backend(backend_id, self.config)
                    candidate_backend.ensure_available()
                    backend = candidate_backend
                    resolved = backend_id
                    if index == 0:
                        status = "success"
                        reason = f"{candidate_backend.name} backend ready"
                    elif backend_id == "synthetic":
                        status = "synthetic_fallback"
                        reason = f"Test environment synthetic fallback after: {init_errors}"
                    else:
                        status = "fallback"
                        requested_error = init_errors.get(requested, "unknown error")
                        reason = f"Requested '{requested}' unavailable: {requested_error}. Selected '{backend_id}'"
                    break
                except LicenseRestrictionError:
                    # Never bypass explicit license restrictions on requested backend.
                    if index == 0:
                        raise
                    init_errors[backend_id] = "license_restriction"
                except ValueError:
                    # Unknown requested backend should remain a hard error.
                    if index == 0:
                        raise
                    init_errors[backend_id] = "unknown_backend"
                except (ImportError, FileNotFoundError, RuntimeError) as backend_error:
                    init_errors[backend_id] = str(backend_error)
                except Exception as backend_error:  # pragma: no cover - defensive hardening
                    init_errors[backend_id] = str(backend_error)

            if backend is None or resolved is None:
                if not allow_synthetic:
                    raise RuntimeError(
                        f"No depth backend available from candidates {candidate_chain}. "
                        f"Errors: {init_errors}. Install ML dependencies (torch, transformers) or "
                        "explicitly enable synthetic fallback for testing "
                        "(config.allow_synthetic_fallback=True or TP_ALLOW_SYNTHETIC_FALLBACK=1)."
                    )
                raise RuntimeError(f"No depth backend available from candidates {candidate_chain}. Errors: {init_errors}")

            self.depth_backend = backend
            self._depth_backend_cache[resolved] = backend
            self._backend_metadata = BackendSelectionMetadata(
                requested_backend=requested,
                resolved_backend=resolved,
                resolution_status=status,
                resolution_reason=reason,
                model_id=self.config.model_variant.value.huggingface_id,
                device=self.config.depth_device,
                attempts=[],
            )
            self._active_backend_metadata = self._backend_metadata
            self._active_depth_attempts: List[Dict[str, Any]] = []
            self._active_selected_attempt_index: Optional[int] = None

            logger.info(f"Depth backend: requested={requested} resolved={resolved} device={self.config.depth_device}")

        except LicenseRestrictionError as e:
            logger.error(f"License restriction: {e}")
            raise
        except Exception as e:
            logger.error(f"Backend initialization failed: {e}")
            raise

    def _resolve_runtime_backend_chain(self, primary_backend_id: str) -> List[str]:
        """Resolve ordered runtime fallback chain for per-image depth attempts."""
        chain: List[str] = [primary_backend_id]
        configured_chain = getattr(self.config, "depth_operational_fallback_chain", ("da3", "da2"))
        for backend_id in configured_chain:
            if backend_id and backend_id not in chain:
                chain.append(backend_id)

        allow_synthetic = bool(self.config.allow_synthetic_fallback) or os.getenv("TP_ALLOW_SYNTHETIC_FALLBACK") == "1"
        if allow_synthetic and "synthetic" not in chain:
            chain.append("synthetic")
        return chain

    @staticmethod
    def _expected_output_depth_units_for_backend(backend_id: str) -> str:
        """Return expected output depth units for cache-key partitioning."""
        return "meters" if backend_id == "depth_pro" else "relative"

    def _build_depth_cache_fingerprint(self, backend_id: str) -> str:
        """Build backend-scoped cache fingerprint for depth reuse safety."""
        base_fp = self.compute_config_fingerprint().depth_only().to_sha256()
        expected_units = self._expected_output_depth_units_for_backend(backend_id)
        payload = f"{base_fp}|backend={backend_id}|units={expected_units}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_or_create_depth_backend(self, backend_id: str):
        """Fetch backend instance from cache or registry and ensure availability."""
        # Respect an already-initialized active backend when it matches the
        # requested backend id. This keeps injected test doubles stable and
        # avoids unnecessary registry lookups.
        active_backend = getattr(self, "depth_backend", None)
        if active_backend is not None and getattr(active_backend, "name", None) == backend_id:
            self._depth_backend_cache[backend_id] = active_backend
            return active_backend

        cached = self._depth_backend_cache.get(backend_id)
        if cached is not None:
            return cached

        backend = self._depth_registry.get_backend(backend_id, self.config)
        backend.ensure_available()
        self._depth_backend_cache[backend_id] = backend
        return backend

    @staticmethod
    def _infer_operational_error_code(error: Exception) -> str:
        """Map backend runtime exceptions to stable operational error codes."""
        if isinstance(error, ImportError):
            return "BACKEND_IMPORT_ERROR"
        if isinstance(error, FileNotFoundError):
            return "BACKEND_RESOURCE_MISSING"
        message = str(error).lower()
        if "torch not compiled with cuda enabled" in message:
            return "CUDA_HARDCODED_IN_BACKEND"
        if "cuda" in message and "not available" in message:
            return "CUDA_UNAVAILABLE"
        if "mps" in message and "not available" in message:
            return "MPS_UNAVAILABLE"
        return "BACKEND_RUNTIME_ERROR"

    def _build_backend_metadata_for_attempts(
        self,
        selected_backend: str,
        attempts: List[Dict[str, Any]],
        result_metadata: Optional[Dict[str, Any]] = None,
    ) -> BackendSelectionMetadata:
        """Build per-image backend selection metadata including attempt provenance."""
        requested = self._backend_metadata.requested_backend or self._backend_metadata.resolved_backend
        resolution_status = "success" if selected_backend == requested else "fallback"
        resolution_reason: Optional[str] = None
        if resolution_status == "fallback":
            failed = [attempt for attempt in attempts if attempt.get("status") == "failed"]
            if failed:
                first_failure = failed[0]
                failure_kind = first_failure.get("failure_kind", "operational")
                failure_code = first_failure.get("error_code", "UNKNOWN")
                resolution_reason = (
                    f"Fallback from '{requested}' to '{selected_backend}' " f"after {failure_kind} failure ({failure_code})"
                )
            else:
                resolution_reason = f"Fallback from '{requested}' to '{selected_backend}'"

        metadata = result_metadata or {}
        model_id = (
            metadata.get("resolved_model_id")
            or metadata.get("requested_model_id")
            or self.config.model_variant.value.huggingface_id
        )

        return BackendSelectionMetadata(
            requested_backend=requested,
            resolved_backend=selected_backend,
            resolution_status=resolution_status,
            resolution_reason=resolution_reason,
            model_id=str(model_id),
            device=self.config.depth_device,
            attempts=attempts,
        )

    def compute_config_fingerprint(self) -> ConfigFingerprint:
        return ConfigFingerprint(
            model_variant=self.config.model_variant.value.name,
            depth_quantization=self.config.depth_quantization,
            depth_device=self.config.depth_device,
            preset=self.config.preset.value if self.config.preset else None,
            v2_preset=self.config.v2_preset,
            v2_device=self.config.v2_device,
            v2_upscaler_backend=self.config.v2_upscaler_backend,
        )

    def _build_run_card_config_fingerprint(self) -> Dict[str, Any]:
        """Build run-card config fingerprint from effective user intent + resolution."""
        base = self.compute_config_fingerprint()

        preset_requested = getattr(self.config, "preset_requested", None) or (
            self.config.preset.value if self.config.preset else None
        )
        preset_resolved = self.config.preset.value if self.config.preset else f"quality_tier:{self.config.quality_tier}"

        requested_backend = self._backend_metadata.requested_backend or "auto"
        resolved_backend = self._backend_metadata.resolved_backend
        requested_device = self.config.depth_device
        resolved_device = self._backend_metadata.device

        payload = {
            "model_variant": base.model_variant,
            "depth_quantization": base.depth_quantization,
            "depth_device": base.depth_device,
            "preset": base.preset,
            "v2_preset": base.v2_preset,
            "v2_device": base.v2_device,
            "v2_upscaler_backend": base.v2_upscaler_backend,
            "preset_requested": preset_requested,
            "preset_resolved": preset_resolved,
            "backend_requested": requested_backend,
            "backend_resolved": resolved_backend,
            "device_requested": requested_device,
            "device_resolved": resolved_device,
            "quality_tier": self.config.quality_tier,
            "strict_inputs": bool(self.config.strict_inputs),
            "strict_segmentation": bool(self.config.strict_backend),
            "apex_strict_mode": self._is_apex_tier(),
        }
        canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return {
            **payload,
            "hash_algorithm": "sha256",
            "canonical_json": canonical_json,
            "sha256": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
        }

    def _extract_v2_depth_handoff_status(
        self,
        v2_result: Optional[Dict[str, Any]],
        v2_report_path: Optional[Path],
    ) -> Optional[bool]:
        """Return whether V2 consumed depth-map input (True/False) when determinable."""
        if isinstance(v2_result, dict):
            if isinstance(v2_result.get("depth_consumed"), bool):
                return bool(v2_result.get("depth_consumed"))
            stage_metadata = v2_result.get("stage_metadata")
            if isinstance(stage_metadata, dict) and "has_depth" in stage_metadata:
                return bool(stage_metadata.get("has_depth"))
            if isinstance(v2_result.get("depth_map"), str):
                return True
            if "depth_map" in v2_result and v2_result.get("depth_map") is None:
                return False

        if v2_report_path and v2_report_path.exists():
            try:
                with open(v2_report_path, "r", encoding="utf-8") as report_file:
                    report_payload = json.load(report_file)
            except Exception as exc:
                logger.debug(f"Failed to parse V2 report for depth handoff check ({v2_report_path}): {exc}")
            else:
                if isinstance(report_payload.get("depth_consumed"), bool):
                    return bool(report_payload.get("depth_consumed"))
                stage_metadata = report_payload.get("stage_metadata")
                if isinstance(stage_metadata, dict) and "has_depth" in stage_metadata:
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
        """Enforce that V2 consumes depth when depth artifacts were produced."""
        if not depth_path or not depth_path.exists():
            return

        depth_consumed = self._extract_v2_depth_handoff_status(v2_result=v2_result, v2_report_path=v2_report_path)
        if depth_consumed is None:
            return
        if depth_consumed:
            return

        message = (
            "V2 depth handoff failed: depth artifact exists but V2 reported depth_consumed=false. "
            "This indicates stem-resolution drift."
        )
        details = {
            "depth_path": str(depth_path),
            "v2_report_path": str(v2_report_path) if v2_report_path else None,
            "depth_consumed": depth_consumed,
        }
        if self._is_apex_tier():
            raise ApexStrictGateError("APEX_V2_DEPTH_HANDOFF_MISSING", message, details=details)
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
                model_id=self.config.model_variant.value.huggingface_id,
                device=self.config.depth_device,
                attempts=[],
            ),
        )

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
            manifest_exists: Whether a manifest exists (for IF_MANIFEST_EXISTS mode)
            saved_hash: Previously saved hash from manifest (for IF_MANIFEST_EXISTS mode)
            for_manifest_write: If True, compute hash for writing manifest (establishes baseline).
                              If False, compute hash for comparison only.

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
                # Writing manifest: always compute hash to establish/update baseline
                pass  # Fall through to compute hash
            else:
                # Reading for comparison: only compute if we have a baseline
                if not manifest_exists or not saved_hash:
                    # No baseline exists - skip comparison
                    return None

        # ALWAYS or IF_MANIFEST_EXISTS (when baseline exists or writing manifest)
        try:
            return compute_file_sha256(image_path)
        except Exception as e:
            logger.error(f"Hash computation failed for {image_path}: {e}")
            raise IOError(f"Hash computation failed: {e}") from e

    def should_skip_depth(self, depth_path: Path, manifest_path: Path, image_input: ImageInput) -> bool:
        """Determine whether to skip depth computation.

        Uses stored config fingerprint for comparison rather than reconstructing
        from partial fields. This ensures any config change invalidates the cache.

        Args:
            depth_path: Path to the depth output file
            manifest_path: Path to the manifest file
            image_input: Input image information

        Returns:
            True if depth step can be skipped (cached result is valid), False otherwise
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
                    image_input.path, manifest_exists=True, saved_hash=saved_hash, for_manifest_write=False
                )
                if current_hash and current_hash != saved_hash:
                    logger.info(f"Input image changed - regenerating depth: {image_input.path}")
                    return False

            # Config Fingerprint Check - use stored fingerprint directly
            if not manifest.config_fingerprint:
                logger.debug("No config fingerprint in manifest - regenerating depth")
                return False

            # Compare stored depth config fingerprint with current config
            current_fp = self.compute_config_fingerprint()
            stored_fp = manifest.config_fingerprint

            # Compare depth-related config using stored fingerprint's SHA256
            if current_fp.depth_only().to_sha256() != stored_fp.depth_only().to_sha256():
                logger.info("Depth config changed - regenerating")
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
                    logger.debug(f"Depth file has invalid dimensions: {d.ndim}")
                    return False

            logger.debug(f"Resuming with existing depth: {depth_path}")
            return True
        except Exception as e:
            logger.debug(f"Skip check failed: {e}")
            return False

    def should_skip_v2(
        self, v2_report_path: Optional[Path], manifest_path: Path, image_input: ImageInput, depth_was_skipped: bool
    ) -> bool:
        """Determine whether to skip V2 enhancement stage.

        V2 skip logic is independent of PBR generation. V2 enhancement is a separate
        stage from PBR map generation, and should be evaluated based on V2 config
        changes and output existence.

        Uses stored config fingerprint for comparison and performs defensive
        output existence checks if enabled.

        Args:
            v2_report_path: Path to V2 report file
            manifest_path: Path to the manifest file
            image_input: Input image information
            depth_was_skipped: Whether depth step was skipped

        Returns:
            True if V2 stage can be skipped (cached result is valid), False otherwise
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
                logger.debug("No config fingerprint in manifest - regenerating V2")
                return False

            # Compare V2/PBR config using stored fingerprint's SHA256
            current_fp = self.compute_config_fingerprint()
            stored_fp = manifest.config_fingerprint

            if current_fp.v2_only().to_sha256() != stored_fp.v2_only().to_sha256():
                logger.info("V2/PBR config changed - regenerating")
                return False

            # Consistency Check - if depth was recomputed, V2 must also rerun
            if not depth_was_skipped:
                logger.info("Depth was regenerated - V2 must rerun")
                return False

            # V2 Metadata Check - verify V2 ran successfully
            if not manifest.v2 or manifest.v2.status != "ok":
                return False

            # Defensive output existence check for V2 report
            if self.verify_outputs and v2_report_path:
                if not v2_report_path.exists():
                    logger.debug(f"V2 report missing: {v2_report_path}")
                    return False

            # Defensive output existence check for PBR assets (only if they exist in manifest)
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
            Tuple of (depth_metadata, depth_runtime_s, pbr_assets, materials_v3_result,
                     materials_v3_runtime_s, enhanced_image_path,
                     backend_selection_metadata, depth_attempts)
        """
        depth_runtime_s = 0.0
        depth_metadata = None
        pbr_assets = None
        materials_v3_result = None
        materials_v3_runtime_s = 0.0
        enhanced_image_path = None  # Will be set if Materials V3 produces enhanced_image
        depth_attempts: List[Dict[str, Any]] = []
        active_backend_metadata = self._backend_metadata
        self._active_selected_attempt_index = None

        if not skip_depth:
            # Lazy preprocessing: Only validate and preprocess if we're running depth
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
                    logger.debug(f"Strict verification passed: {validated_path.name}")
                except Exception as e:
                    logger.error(f"Strict verification failed: {validated_path.name} - {e}")
                    raise ValueError(f"Image failed strict verification: {validated_path}") from e

            preprocessed_array, original_shape = preprocess_image(validated_path)

            logger.info(f"Stage A: Generating depth for {output_key}...")
            t0 = time.time()
            try:
                # Phase 2: Check content-addressable depth cache
                image_sha256 = None
                if self.depth_cache:
                    image_sha256 = self._compute_or_skip_hash(image_input.path, manifest_exists=False, for_manifest_write=True)

                # 1. Inference with per-image backend attempt/fallback state machine.
                from PIL import Image

                preprocessed_uint8 = (np.clip(preprocessed_array, 0, 1) * 255).astype(np.uint8)
                pil_image = Image.fromarray(preprocessed_uint8)
                attempt_chain = self._resolve_runtime_backend_chain(self.depth_backend.name)

                result = None
                depth_map = None
                depth_validity_metrics = None
                selected_backend_id = self.depth_backend.name
                selected_attempt_index: Optional[int] = None
                last_error: Optional[Exception] = None

                for attempt_index, backend_id in enumerate(attempt_chain):
                    attempt_start = time.time()
                    attempt_record: Dict[str, Any] = {
                        "attempt": attempt_index,
                        "backend": backend_id,
                        "device": self.config.depth_device,
                        "status": "started",
                        "failure_kind": None,
                        "error_code": None,
                        "error_message": None,
                        "apex_gate_passed": None,
                        "cached": False,
                    }

                    try:
                        attempt_cache_fp_hash = None
                        cached_depth = None
                        if self.depth_cache and image_sha256:
                            attempt_cache_fp_hash = self._build_depth_cache_fingerprint(backend_id)
                            cached_depth = self.depth_cache.get(image_sha256, attempt_cache_fp_hash)
                            if cached_depth is not None:
                                logger.info("Cache hit: using cached depth for %s (backend=%s)", output_key, backend_id)
                        attempt_record["cached"] = bool(cached_depth is not None)

                        if cached_depth is not None:
                            from ..depth.backends.protocol import DepthResult

                            # No inference runs on cache hit; mark device provenance explicitly.
                            attempt_record["device"] = "cache"
                            cache_metadata: Dict[str, Any] = {
                                "cached": True,
                                "output_normalization": "cache_reuse",
                                "cache_backend_id": backend_id,
                                "device": "cache",
                            }
                            if image_sha256 and attempt_cache_fp_hash:
                                cache_metadata["cache_key"] = f"{image_sha256}_{attempt_cache_fp_hash}"

                            result_candidate = DepthResult(
                                depth_map=cached_depth,
                                original_image=preprocessed_uint8,
                                metadata=cache_metadata,
                                depth_units="meters" if backend_id == "depth_pro" else "relative",
                                backend_id=backend_id,
                                device="cache",
                                dtype="float32",
                                input_size=original_shape,
                            )
                        else:
                            backend = self._get_or_create_depth_backend(backend_id)
                            self.depth_backend = backend
                            resolved_backend_device = getattr(backend, "_device", None) or getattr(backend, "device", None)
                            if isinstance(resolved_backend_device, str) and resolved_backend_device:
                                attempt_record["device"] = resolved_backend_device
                            result_candidate = backend.compute(pil_image)
                            result_candidate = self.postprocessor.process(result_candidate)
                            if self.depth_cache and image_sha256 and attempt_cache_fp_hash:
                                self.depth_cache.store(image_sha256, attempt_cache_fp_hash, result_candidate.depth_map)

                        result_device = getattr(result_candidate, "device", None)
                        if isinstance(result_device, str) and result_device:
                            attempt_record["device"] = result_device

                        # CRITICAL FIX (#2): Resize depth map back to original dimensions
                        depth_candidate = (
                            result_candidate.depth_map if hasattr(result_candidate, "depth_map") else result_candidate.depth
                        )
                        current_shape = depth_candidate.shape[:2]
                        if current_shape != original_shape:
                            from PIL import Image as PILImage

                            logger.debug(f"Resizing depth map from {current_shape} back to original {original_shape}")
                            # Preserve raw numeric depth semantics (relative or metric) during resize.
                            depth_pil = PILImage.fromarray(np.asarray(depth_candidate, dtype=np.float32), mode="F")
                            depth_pil_resized = depth_pil.resize(
                                (original_shape[1], original_shape[0]),
                                PILImage.Resampling.BILINEAR,
                            )
                            depth_candidate = np.array(depth_pil_resized, dtype=np.float32)
                            if hasattr(result_candidate, "depth_map"):
                                result_candidate.depth_map = depth_candidate
                            else:
                                result_candidate.depth = depth_candidate

                        result_metadata = dict(getattr(result_candidate, "metadata", None) or {})
                        attempt_record["model_id"] = (
                            result_metadata.get("resolved_model_id")
                            or result_metadata.get("requested_model_id")
                            or self.config.model_variant.value.huggingface_id
                        )
                        attempt_record["source_depth_units"] = result_metadata.get(
                            "source_depth_units",
                            getattr(result_candidate, "depth_units", None) or "unknown",
                        )
                        attempt_record["output_depth_units"] = result_metadata.get(
                            "output_depth_units",
                            getattr(result_candidate, "depth_units", None) or "unknown",
                        )
                        attempt_record["output_normalization"] = result_metadata.get("output_normalization", "unknown")

                        # 2b. APEX depth validity gate (plateau/saturation guardrails)
                        gate_result = self._enforce_apex_depth_validity_gate(
                            depth_candidate,
                            depth_units=getattr(result_candidate, "depth_units", None),
                        )

                        attempt_record.update(
                            {
                                "status": "success",
                                "apex_gate_passed": bool(gate_result is None or gate_result.get("passed", False)),
                            }
                        )
                        attempt_record["duration_s"] = time.time() - attempt_start
                        depth_attempts.append(attempt_record)

                        result = result_candidate
                        depth_map = depth_candidate
                        depth_validity_metrics = gate_result
                        selected_backend_id = backend_id
                        selected_attempt_index = attempt_index
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

                        has_next = attempt_index + 1 < len(attempt_chain)
                        if self.config.allow_semantic_fallback and has_next:
                            logger.warning(
                                "Semantic gate failed on backend=%s code=%s; attempting fallback backend.",
                                backend_id,
                                semantic_error.code,
                            )
                            continue
                        raise

                    except Exception as operational_error:
                        error_code = self._infer_operational_error_code(operational_error)
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

                        has_next = attempt_index + 1 < len(attempt_chain)
                        if has_next:
                            logger.warning(
                                "Operational depth failure on backend=%s code=%s; attempting fallback backend.",
                                backend_id,
                                error_code,
                            )
                            continue
                        raise

                if result is None or depth_map is None:
                    if last_error is not None:
                        raise last_error
                    raise RuntimeError("Depth inference failed before producing a result.")

                depth_runtime_s = time.time() - t0
                active_backend_metadata = self._build_backend_metadata_for_attempts(
                    selected_backend_id,
                    depth_attempts,
                    result_metadata=getattr(result, "metadata", None) or {},
                )
                self._active_backend_metadata = active_backend_metadata
                self._active_depth_attempts = depth_attempts
                self._active_selected_attempt_index = selected_attempt_index

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
                    )

                # 3. Write quantized depth (PNG 16-bit)
                _, _, depth_stats = atomic_write_depth_u16_png_with_stats(
                    depth_path,
                    result.depth,
                    method=self.config.depth_quantization,
                    debug_verify=self.config.verify_depth_writes,
                )

                # 3b. Save float depth (.npy) for high-precision PBR if enabled
                if getattr(self.config, "save_float_depth", False):
                    np.save(str(float_depth_path), result.depth)
                    logger.debug(f"Saved float depth: {float_depth_path}")

                # Capture backend metadata dynamically (ADR-019)
                license_str = (
                    self.depth_backend.license_type.value if hasattr(self.depth_backend, "license_type") else "unknown"
                )
                stats = {
                    "backend": self.depth_backend.name,
                    "license": license_str,
                    "non_commercial_ok": self.config.non_commercial_ok,
                    "dtype": "uint16",
                    "shape": list(result.depth.shape[:2]),
                    "representation": "depth",
                    "convention": "higher_is_farther",
                    "unit": result.depth_units if hasattr(result, "depth_units") else "relative",
                    "depth_png_path": str(depth_path),
                    "depth_float_path": str(float_depth_path) if getattr(self.config, "save_float_depth", False) else None,
                    "depth_float_dtype": "float32" if getattr(self.config, "save_float_depth", False) else None,
                    "depth_float_shape": (
                        list(result.depth.shape[:2]) if getattr(self.config, "save_float_depth", False) else None
                    ),
                    "canonical_depth_path": (
                        str(float_depth_path) if getattr(self.config, "save_float_depth", False) else str(depth_path)
                    ),
                    "attempts": depth_attempts,
                }
                if depth_validity_metrics:
                    stats["apex_depth_validity"] = depth_validity_metrics

                # Merge inference provenance into depth stats
                _md = getattr(result, "metadata", None) or {}
                for _k in ("requested_model_id", "resolved_model_id", "resolved_model_source"):
                    if _k in _md:
                        stats[_k] = _md[_k]
                for _k in ("source_depth_units", "output_depth_units", "output_normalization"):
                    if _k in _md:
                        stats[_k] = _md[_k]

                # CRITICAL FIX: Use resolved backend name, not config default
                # This ensures depth.model matches what actually ran (backend_selection.resolved_backend)
                # ADR-023 compliance: identity must match execution reality
                resolved_backend = active_backend_metadata
                model_name = resolved_backend.resolved_backend if resolved_backend else self.config.model_variant.value.name

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
                            "runtime_seconds": depth_metadata.runtime_seconds,
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
                if isinstance(e, ApexStrictGateError):
                    logger.error("APEX strict gate failure: code=%s details=%s", e.code, e.details)
                    raise
                if self.config.depth_fallback == "fail":
                    raise
                elif self.config.depth_fallback == "skip":
                    self._active_backend_metadata = active_backend_metadata
                    self._active_depth_attempts = depth_attempts
                    return None, 0.0, None, None, 0.0, None, active_backend_metadata, depth_attempts
                elif self.config.depth_fallback == "v2-auto":
                    logger.info("V2 fallback mode: V3 failed, will attempt V2 with independent depth")
                    if depth_path.exists():
                        depth_path.unlink()
                    self._active_backend_metadata = active_backend_metadata
                    self._active_depth_attempts = depth_attempts
                    return None, 0.0, None, None, 0.0, None, active_backend_metadata, depth_attempts
                else:
                    raise ValueError(f"Unsupported depth_fallback mode: {self.config.depth_fallback}") from e
        else:
            # Depth was skipped - load from cache
            # CRITICAL FIX: Preserve Materials V3 metadata from previous run
            if manifest_path.exists():
                try:
                    m = CombinedManifest.load(manifest_path)
                    depth_metadata = m.depth
                    pbr_assets = getattr(m, "pbr_assets", None)
                    if getattr(m, "backend_selection", None) is not None:
                        active_backend_metadata = m.backend_selection
                        depth_attempts = list(m.backend_selection.attempts or [])
                        self._active_backend_metadata = active_backend_metadata
                        self._active_depth_attempts = depth_attempts
                        success_attempts = [attempt for attempt in depth_attempts if attempt.get("status") == "success"]
                        if success_attempts:
                            self._active_selected_attempt_index = int(success_attempts[-1].get("attempt", 0))

                    # Preserve Materials V3 result from previous run
                    if hasattr(m, "materials_v3") and m.materials_v3:
                        logger.info("Preserving Materials V3 metadata from previous run (depth was cached)")
                        materials_v3_result = {
                            "materials_v3_response_plan": m.materials_v3.response_plan,
                            "materials_v3_pixel_ops": m.materials_v3.pixel_ops,
                            "materials_v3_metadata": {"version": m.materials_v3.version},
                            # Note: enhanced_image and material_masks are not persisted to manifest
                            # This is intentional - only the response plan and telemetry are preserved
                        }
                        materials_v3_runtime_s = (
                            m.materials_v3.runtime_seconds if hasattr(m.materials_v3, "runtime_seconds") else 0.0
                        )
                except Exception as e:
                    logger.debug(f"Failed to load previous manifest metadata: {e}")

            # PBR generation with cached depth (if enabled but not previously generated)
            if self.config.generate_pbr and (pbr_assets is None or not self._verify_pbr_outputs(pbr_assets)):
                logger.info("Generating PBR maps from cached depth...")
                try:
                    depth_data_for_pbr = self._load_cached_depth(depth_path, float_depth_path)
                    if depth_data_for_pbr is not None:
                        pbr_assets = self._generate_pbr_stage(depth_data_for_pbr, output_key)
                except Exception as pbr_error:
                    logger.warning(f"PBR generation from cache failed: {pbr_error}")

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

    def _generate_pbr_stage(self, depth: Any, output_key: Path) -> Optional[dict]:
        """Generate PBR maps from depth data.

        Args:
            depth: Depth array (numpy)
            output_key: Output key for artifact naming

        Returns:
            Dictionary with PBR asset paths and metadata, or None if disabled/failed
        """
        if not self.config.generate_pbr:
            return None

        try:
            logger.info("Generating PBR maps...")
            pbr_t0 = time.time()

            # Use to_pbr_config() for consistent parameter conversion
            pbr_config = self.config.to_pbr_config()

            # Generate maps from depth
            normal_map, roughness_map, ao_map = generate_pbr_maps(depth, config=pbr_config)

            # Write PBR maps
            pbr_dir = self.output_root / "pbr"
            pbr_dir.mkdir(parents=True, exist_ok=True)

            # Derive base name from output_key for consistent artifact naming
            sanitized_stem = output_key.stem if output_key.suffix else output_key.name

            pbr_paths = write_pbr_maps(
                normal_map=normal_map, roughness_map=roughness_map, ao_map=ao_map, output_dir=pbr_dir, base_name=sanitized_stem
            )

            pbr_runtime = time.time() - pbr_t0
            logger.info(f"PBR maps generated in {pbr_runtime:.2f}s: {list(pbr_paths.keys())}")

            # Store paths for manifest
            pbr_assets = {
                "normal_path": str(pbr_paths["normal"]),
                "roughness_path": str(pbr_paths["roughness"]),
                "ao_path": str(pbr_paths["ao"]),
                "runtime_seconds": pbr_runtime,
                "config": {
                    "normal_strength": pbr_config.normal_strength,
                    "normal_blur_radius": pbr_config.normal_blur_radius,
                    "roughness_strength": pbr_config.roughness_strength,
                    "roughness_blur_radius": pbr_config.roughness_blur_radius,
                    "ao_strength": pbr_config.ao_strength,
                    "ao_blur_radius": pbr_config.ao_blur_radius,
                    "ao_bias": pbr_config.ao_bias,
                },
            }
            return pbr_assets

        except Exception as pbr_error:
            logger.warning(f"PBR generation failed (non-blocking): {pbr_error}")
            return None

    def _expected_materials_v3_enhanced_path(self, output_key: Path) -> Path:
        """Return canonical Materials V3 enhanced-image handoff path for V2."""
        temp_dir = self.output_root / "temp"
        extension = ".tif" if (self.config.emit_master16 or self.config.emit_upscaled16) else ".png"
        return temp_dir / f"{output_key.stem}_materials_v3_enhanced{extension}"

    def _segmentation_mask_artifact_path(self, output_key: Path) -> Path:
        """Return canonical persistent segmentation mask artifact path."""
        return self.segmentation_dir / output_key.parent / f"{output_key.stem}_materials_v3_masks.npz"

    def _persist_material_masks_artifact(self, masks: Dict[str, np.ndarray], output_key: Path) -> Optional[Path]:
        """Persist material masks under segmentation/ as deterministic artifacts."""
        if not masks:
            return None
        target_dir = self._segmentation_mask_artifact_path(output_key).parent
        target_dir.mkdir(parents=True, exist_ok=True)
        return self._serialize_material_masks(masks, output_key, target_dir)

    def _run_materials_v3_stage(
        self,
        *,
        preprocessed_array: np.ndarray,
        depth_map: np.ndarray,
        output_key: Path,
    ) -> tuple[Optional[dict], float, Optional[Path]]:
        """Run Materials V3 stage and persist canonical enhanced-image handoff artifact."""
        if not self.materials_v3_engine:
            return None, 0.0, None

        logger.info("Running Materials V3 surface-aware finishing...")
        t_materials_start = time.time()
        materials_v3_result: Optional[dict] = None
        enhanced_image_path: Optional[Path] = None

        try:
            # APEX strict gate: enforce segmentation prerequisites before running Materials V3.
            self._enforce_apex_materials_gate()

            from .segmentation_backend import get_last_segmentation_runtime_metadata, segment_materials

            # Convert preprocessed float32 [0,1] to uint8 [0,255] for segmentation backend.
            preprocessed_uint8_for_seg = (np.clip(preprocessed_array, 0, 1) * 255).astype(np.uint8)
            segmentation_result = {"materials": segment_materials(preprocessed_uint8_for_seg, self.config)}
            segmentation_runtime = get_last_segmentation_runtime_metadata()
            if segmentation_runtime:
                segmentation_result["segmentation_metadata"] = segmentation_runtime
            self._enforce_apex_materials_gate(segmentation_result)

            if segmentation_result.get("materials"):
                logger.info(
                    f"Material segmentation: {len(segmentation_result['materials'])} "
                    f"materials detected using {self.config.material_segmentation_backend} backend"
                )

            materials_v3_result = self.materials_v3_engine.process(
                image=preprocessed_array,
                segmentation_result=segmentation_result,
                depth_map=depth_map,
            )
            runtime_s = time.time() - t_materials_start

            if materials_v3_result:
                material_masks = materials_v3_result.get("material_masks")
                if isinstance(material_masks, dict) and material_masks:
                    mask_artifact_path = self._persist_material_masks_artifact(material_masks, output_key)
                    if mask_artifact_path:
                        materials_v3_metadata = materials_v3_result.setdefault("materials_v3_metadata", {})
                        segmentation_metadata = materials_v3_metadata.get("segmentation_metadata")
                        segmentation_metadata = dict(segmentation_metadata) if isinstance(segmentation_metadata, dict) else {}
                        segmentation_metadata["mask_artifact_path"] = str(mask_artifact_path)
                        segmentation_metadata["mask_artifact_format"] = "npz"
                        materials_v3_metadata["segmentation_metadata"] = segmentation_metadata

                enhanced_image = materials_v3_result.get("enhanced_image")
                if enhanced_image is not None:
                    from PIL import Image as PILImage

                    temp_dir = self.output_root / "temp"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    enhanced_image_path = self._expected_materials_v3_enhanced_path(output_key)

                    if self.config.emit_master16 or self.config.emit_upscaled16:
                        import tifffile

                        enhanced_uint16 = (np.clip(enhanced_image, 0, 1) * 65535 + 0.5).astype(np.uint16)
                        with atomic_temp_file(enhanced_image_path, suffix=".tif", create_file=False) as temp_path:
                            tifffile.imwrite(
                                temp_path,
                                enhanced_uint16,
                                photometric="rgb",
                                compression="lzw",
                                metadata={"software": "Transformation Portal v3"},
                            )
                        logger.info(
                            f"Materials V3 enhanced image with "
                            f"{len(materials_v3_result.get('materials_v3_pixel_ops', {}).get('applied', []))} "
                            f"pixel operations - saved to {enhanced_image_path} (16-bit TIFF) for V2 stage"
                        )
                    else:
                        enhanced_uint8 = (np.clip(enhanced_image, 0, 1) * 255).astype(np.uint8)
                        enhanced_image_path = atomic_write_pil_png(
                            enhanced_image_path, PILImage.fromarray(enhanced_uint8), optimize=True
                        )
                        logger.info(
                            f"Materials V3 enhanced image with "
                            f"{len(materials_v3_result.get('materials_v3_pixel_ops', {}).get('applied', []))} "
                            f"pixel operations - saved to {enhanced_image_path} (8-bit PNG) for V2 stage"
                        )
                else:
                    logger.debug("Materials V3 did not return enhanced_image, using original image")

                logger.info(
                    f"Materials V3 completed in {runtime_s:.3f}s: "
                    f"{len(materials_v3_result.get('materials_v3_pixel_ops', {}).get('applied', []))} "
                    f"operations applied"
                )

            return materials_v3_result, runtime_s, enhanced_image_path

        except ApexStrictGateError:
            # Hard-fail in apex strict mode: do not silently continue with no-op Materials V3.
            raise
        except Exception as e:
            logger.warning(f"Materials V3 processing failed: {e}", exc_info=True)
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
        """Ensure APEX strict mode has canonical Materials->V2 handoff artifacts."""
        if not self._is_apex_materials_gate_enabled():
            return materials_v3_result, materials_v3_runtime_s, enhanced_image_path

        if not depth_path.exists():
            return materials_v3_result, materials_v3_runtime_s, enhanced_image_path

        expected_path = self._expected_materials_v3_enhanced_path(output_key)
        expected_path_resolved = expected_path.resolve()
        enhanced_resolved = enhanced_image_path.resolve() if enhanced_image_path else None
        has_canonical_enhanced = bool(
            enhanced_image_path and enhanced_image_path.exists() and enhanced_resolved == expected_path_resolved
        )
        has_masks = bool(materials_v3_result and materials_v3_result.get("material_masks"))

        if has_canonical_enhanced and has_masks:
            return materials_v3_result, materials_v3_runtime_s, enhanced_image_path

        logger.info(
            "APEX strict mode: depth was reused but canonical Materials V3 handoff was incomplete; "
            "recomputing Materials V3 stage from cached depth."
        )

        if self.materials_v3_engine is None:
            raise ApexStrictGateError(
                "APEX_MATERIALS_ENGINE_MISSING",
                "APEX strict mode requires Materials V3 engine for canonical cached-depth handoff.",
            )

        from .preprocessing import preprocess_image, validate_image_format

        validated_path = validate_image_format(image_input.path)
        preprocessed_array, _ = preprocess_image(validated_path)
        depth_for_materials = self._load_cached_depth(depth_path, float_depth_path)
        if depth_for_materials is None:
            raise ApexStrictGateError(
                "APEX_MATERIALS_CACHED_DEPTH_MISSING",
                "APEX strict mode could not reload cached depth required for Materials V3 recomputation.",
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
            depth_map=np.asarray(depth_for_materials, dtype=np.float32),
            output_key=output_key,
        )

        has_recomputed_masks = bool(recomputed_result and recomputed_result.get("material_masks"))
        has_recomputed_enhanced = bool(recomputed_enhanced_path and recomputed_enhanced_path.exists())
        resolved_recomputed = recomputed_enhanced_path.resolve() if recomputed_enhanced_path else None

        if not has_recomputed_masks or not has_recomputed_enhanced or resolved_recomputed != expected_path_resolved:
            raise ApexStrictGateError(
                "APEX_V2_CANONICAL_STEM_DIVERGENCE",
                "APEX strict mode could not establish canonical Materials V3 handoff for V2.",
                details={
                    "expected_v2_input": str(expected_path),
                    "recomputed_v2_input": str(recomputed_enhanced_path) if recomputed_enhanced_path else None,
                    "has_material_masks": has_recomputed_masks,
                    "has_enhanced_image": has_recomputed_enhanced,
                },
            )

        return recomputed_result, recomputed_runtime, recomputed_enhanced_path

    def _enforce_apex_v2_canonical_input_preflight(
        self,
        *,
        depth_path: Optional[Path],
        output_key: Path,
        v2_input_path: Path,
        enhanced_image_path: Optional[Path],
        materials_v3_result: Optional[dict],
    ) -> None:
        """Fail early when APEX strict cached fast-path would violate canonical handoff."""
        if not self._is_apex_materials_gate_enabled():
            return
        if not depth_path or not depth_path.exists():
            return

        expected_path = self._expected_materials_v3_enhanced_path(output_key)
        expected_path_resolved = expected_path.resolve()
        actual_path_resolved = Path(v2_input_path).resolve()
        enhanced_path_resolved = enhanced_image_path.resolve() if enhanced_image_path else None
        has_masks = bool(materials_v3_result and materials_v3_result.get("material_masks"))

        if actual_path_resolved == expected_path_resolved and expected_path.exists() and has_masks:
            return

        raise ApexStrictGateError(
            "APEX_V2_CANONICAL_STEM_DIVERGENCE",
            "APEX strict mode forbids fast-path stem divergence before V2 handoff.",
            details={
                "expected_v2_input": str(expected_path),
                "actual_v2_input": str(v2_input_path),
                "enhanced_image_path": str(enhanced_image_path) if enhanced_image_path else None,
                "enhanced_image_matches_expected": bool(enhanced_path_resolved == expected_path_resolved),
                "expected_input_exists": expected_path.exists(),
                "has_material_masks": has_masks,
            },
        )

    def _serialize_material_masks(self, masks: Dict[str, np.ndarray], output_key: Path, output_dir: Path) -> Optional[Path]:
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
                    logger.warning(f"Invalid mask type for {mat_name}: {type(mask)}, skipping serialization")
                    return None
                if mask.dtype not in (np.float32, np.float64):
                    logger.warning(f"Invalid mask dtype for {mat_name}: {mask.dtype} (expected float32/float64), skipping")
                    return None
                if mask.ndim != 2:
                    logger.warning(f"Invalid mask shape for {mat_name}: {mask.shape} (expected 2D), skipping")
                    return None
                ordered_masks[mat_name] = mask

            fixed_zip_datetime = (1980, 1, 1, 0, 0, 0)
            with atomic_temp_file(mask_path, suffix=".npz", create_file=False) as temp_path:
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
                            np.lib.format.write_array(payload, mask, allow_pickle=False)

                            # Use fixed entry metadata so NPZ bytes remain stable across runs.
                            zip_info = zipfile.ZipInfo(filename=f"{mat_name}.npy", date_time=fixed_zip_datetime)
                            zip_info.compress_type = zipfile.ZIP_DEFLATED
                            zip_info.create_system = 0
                            zip_info.external_attr = 0
                            archive.writestr(zip_info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
                    f.flush()
                    os.fsync(f.fileno())

                # Check size before atomic rename
                file_size_mb = temp_path.stat().st_size / (1024 * 1024)
                if file_size_mb > 100:
                    logger.warning(
                        f"Mask file unexpectedly large: {file_size_mb:.1f}MB. " f"Rejecting for safety (size limit: 100MB)"
                    )
                    raise _MaskSerializationRejected("mask_file_too_large")

            # Verify final file exists
            if not mask_path.exists():
                logger.warning(f"Mask serialization failed: file not created at {mask_path}")
                return None

            logger.info(f"Serialized {len(ordered_masks)} material masks to {mask_path.name} ({file_size_mb:.2f}MB)")
            return mask_path

        except _MaskSerializationRejected:
            return None
        except Exception as e:
            logger.warning(f"Failed to serialize material masks: {e}", exc_info=True)
            return None

    def _persisted_material_mask_artifact_path(self, materials_v3_result: Optional[dict]) -> Optional[Path]:
        """Return persisted mask artifact path from Materials V3 metadata when available."""
        if not isinstance(materials_v3_result, dict):
            return None

        materials_v3_metadata = materials_v3_result.get("materials_v3_metadata")
        if not isinstance(materials_v3_metadata, dict):
            return None

        segmentation_metadata = materials_v3_metadata.get("segmentation_metadata")
        if not isinstance(segmentation_metadata, dict):
            return None

        mask_artifact_path = segmentation_metadata.get("mask_artifact_path")
        if not isinstance(mask_artifact_path, str) or not mask_artifact_path:
            return None

        artifact_path = Path(mask_artifact_path)
        if artifact_path.exists():
            return artifact_path

        logger.warning(f"Persisted mask artifact path missing on disk, will fall back to temp serialization: {artifact_path}")
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
            materials_v3_result: Materials V3 result with material_masks (optional)
                If provided, masks will be serialized to disk and passed to V2 subprocess.

        Returns:
            Tuple of (v2_result, v2_runtime_s, v2_report_path)
        """
        # Skip V2 stage if disabled or runner not initialized
        if self.v2_runner is None or not self.config.enable_v2:
            logger.info("V2 stage disabled, skipping enhancement")
            return {"status": "skipped"}, 0.0, None

        v2_report_path = find_v2_report(self.v2_dir, output_key.name)
        skip_v2 = not self.config.force_v2 and self.should_skip_v2(v2_report_path, manifest_path, image_input, skip_depth)

        if skip_v2:
            logger.info("V2 outputs valid, skipping.")
            self._enforce_v2_depth_handoff(depth_path=depth_path, v2_result=None, v2_report_path=v2_report_path)
            return {"status": "ok"}, 0.0, v2_report_path

        # Use persisted segmentation artifact when available; otherwise serialize temp masks for V2 subprocess.
        masks_path: Optional[Path] = self._persisted_material_mask_artifact_path(materials_v3_result)
        cleanup_temp_masks = False
        if masks_path:
            logger.info(f"Reusing persisted material masks for V2 subprocess: {masks_path.name}")
        elif materials_v3_result and materials_v3_result.get("material_masks"):
            temp_dir = self.output_root / "temp"
            masks_path = self._serialize_material_masks(
                materials_v3_result["material_masks"],
                output_key,
                temp_dir,
            )
            if masks_path:
                cleanup_temp_masks = True
                logger.info(f"Material masks serialized for V2 subprocess: {masks_path.name}")
            else:
                logger.warning("Failed to serialize material masks, V2 will run without them")

        # V2 runner: Execute subprocess with optional masks
        # depth_dir=None triggers independent depth generation in V2
        try:
            v2_result = self.v2_runner.run(
                input_path=image_input.path,
                depth_dir=self.depth_dir if (depth_path and depth_path.exists()) else None,
                output_dir=self.v2_dir,
                preset=self.config.v2_preset,
                device=self.config.v2_device,
                upscaler_backend=self.config.v2_upscaler_backend,
                log_file=v2_log_path,
                timeout=self.config.v2_timeout,
                masks_file=masks_path,  # Pass explicit NPZ file path (Option B: eliminates naming coupling)
            )
            v2_runtime_s = v2_result.get("runtime_s", 0.0)
            report_path_value = v2_result.get("report_path")
            if isinstance(report_path_value, str) and report_path_value:
                v2_report_path = Path(report_path_value)
            else:
                v2_report_path = find_v2_report(self.v2_dir, output_key.name)

            self._enforce_v2_depth_handoff(depth_path=depth_path, v2_result=v2_result, v2_report_path=v2_report_path)

            return v2_result, v2_runtime_s, v2_report_path

        finally:
            # Clean up temporary mask file (guaranteed cleanup even if V2 fails)
            if cleanup_temp_masks and masks_path and masks_path.exists():
                try:
                    masks_path.unlink()
                    logger.debug(f"Cleaned up temporary masks: {masks_path.name}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temporary masks {masks_path}: {cleanup_error}")

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
    ) -> None:
        """Write combined manifest with all pipeline metadata.

        Args:
            manifest_path: Path for manifest JSON
            image_input: Input image information
            depth_metadata: Depth stage metadata
            v2_result: V2 stage result dictionary
            v2_report_path: Path to V2 report
            pbr_assets: PBR asset metadata
            depth_runtime_s: Depth stage runtime
            v2_runtime_s: V2 stage runtime
            pipeline_start_time: Pipeline start timestamp
            pipeline_end_time: Pipeline end timestamp
            materials_v3_result: Materials V3 result (optional)
            materials_v3_runtime_s: Materials V3 runtime (optional)
            backend_selection_metadata: Per-image backend selection provenance
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

            # Capture CLI args from environment if available (set by CLI runner)
            # Use shlex for proper shell-quoting aware parsing
            import shlex

            cli_args = shlex.split(os.environ.get("TP_CLI_ARGS", "")) if "TP_CLI_ARGS" in os.environ else None

            # Capture provenance metadata
            # For RAW/TIFF: require exiftool (audit-grade)
            # For JPG/PNG: best-effort (no exiftool requirement)
            provenance = capture_provenance(
                image_path=image_input.path,
                config_fingerprint=config_fp_str,
                cli_args=cli_args,
                repo_root=Path.cwd(),  # Repository root for git SHA
                require_exiftool=is_audit_input,
            )

            # Write provenance sidecar
            provenance.write_sidecar(provenance_sidecar_path)

        except ExiftoolNotFoundError as e:
            # Hard fail if exiftool is not available for RAW/TIFF (audit requirement)
            logger.error(f"Provenance capture failed: exiftool not available for RAW/TIFF input")
            raise RuntimeError(
                f"Audit-grade provenance for RAW/TIFF requires exiftool. "
                f"Install with: apt-get install libimage-exiftool-perl (Ubuntu/Debian) "
                f"or brew install exiftool (macOS)"
            ) from e
        except ProvenanceError as e:
            # Hard fail on any provenance error (no silent drops)
            logger.error(f"Provenance capture failed: {e}")
            raise RuntimeError(f"Provenance capture failed: {e}") from e
        except Exception as e:
            # Catch-all for unexpected errors
            logger.error(f"Unexpected error during provenance capture: {e}")
            raise RuntimeError(f"Provenance capture failed unexpectedly: {e}") from e

        # V2 metadata
        # Determine V2 input/output bit depth based on emit flags and actual Materials V3 enhancement usage
        materials_v3_enhanced_image = materials_v3_result.get("enhanced_image") if materials_v3_result else None
        v2_input_bit_depth = (
            16 if (self.config.emit_master16 or self.config.emit_upscaled16) and materials_v3_enhanced_image is not None else 8
        )
        v2_output_bit_depth = 16 if (self.config.emit_master16 or self.config.emit_upscaled16) else 8
        v2_report_path_value = str(v2_report_path) if v2_report_path else v2_result.get("report_path", "")
        v2_output_paths = []
        v2_output_value = v2_result.get("output")
        if isinstance(v2_output_value, str) and v2_output_value:
            v2_output_paths.append(v2_output_value)
        depth_handoff_state = self._extract_v2_depth_handoff_status(v2_result=v2_result, v2_report_path=v2_report_path)

        v2_metadata = V2Metadata(
            preset=self.config.v2_preset,
            strict_depth=(
                bool(depth_handoff_state)
                if depth_handoff_state is not None
                else bool(depth_metadata is not None and Path(depth_metadata.depth_path).exists())
            ),
            output_dir="v2/",
            report_path=v2_report_path_value,
            status=v2_result["status"],
            runtime_seconds=v2_runtime_s,
            output_paths=v2_output_paths or None,
            error_message=v2_result.get("error"),
            input_bit_depth=v2_input_bit_depth,
            output_bit_depth=v2_output_bit_depth,
        )

        # Materials V3 metadata
        materials_v3_metadata = None
        if materials_v3_result:
            from .manifest import MaterialsV3Metadata

            # Determine bit depth based on emit flags and actual enhanced image generation
            materials_v3_bit_depth = None
            if materials_v3_enhanced_image is not None:
                materials_v3_bit_depth = 16 if (self.config.emit_master16 or self.config.emit_upscaled16) else 8

            materials_v3_metadata = MaterialsV3Metadata(
                enabled=True,
                version=materials_v3_result.get("materials_v3_metadata", {}).get("version", "3.1"),
                response_plan=materials_v3_result.get("materials_v3_response_plan"),
                pixel_ops=materials_v3_result.get("materials_v3_pixel_ops"),
                segmentation_metadata=materials_v3_result.get("materials_v3_metadata", {}).get("segmentation_metadata"),
                runtime_seconds=materials_v3_runtime_s,
                output_bit_depth=materials_v3_bit_depth,
            )

        # Compute input hash respecting HashMode
        manifest_exists = manifest_path.exists()
        saved_hash = None
        if manifest_exists:
            try:
                m = CombinedManifest.load(manifest_path)
                if m.input:
                    saved_hash = m.input.image_sha256
            except Exception as e:
                logger.debug(f"Failed to load previous hash from manifest: {e}")

        input_sha = self._compute_or_skip_hash(
            image_input.path, manifest_exists=manifest_exists, saved_hash=saved_hash, for_manifest_write=True
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
                timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
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
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(pipeline_start_time)),
            end_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(pipeline_end_time)),
            # ADR-023 Phase 3: Backend selection metadata
            backend_selection=backend_selection_metadata or self._active_backend_metadata or self._backend_metadata,
        )
        manifest.write(manifest_path)

    def enhance_image(
        self, image_input: ImageInput, input_root: Optional[Path] = None, _precomputed_paths: Optional[Dict[str, Path]] = None
    ) -> Dict[str, Any]:
        """Run full enhancement pipeline on a single image.

        Orchestrates the depth computation, PBR generation, V2 enhancement,
        and manifest writing stages. Implements lazy preprocessing - validation
        and preprocessing only run if depth computation is needed (not cached).

        Args:
            image_input: Input image information
            input_root: Base directory for relative path calculation
            _precomputed_paths: Internal - pre-computed paths from parallel preprocessing

        Returns:
            Dictionary with processing status and output paths
        """
        # Capture start time for accurate timestamps
        pipeline_start_time = time.time()
        # Reset per-image active state up front so early exceptions cannot leak
        # stale attempt/backend data from a previous image.
        self._active_backend_metadata = getattr(self, "_backend_metadata", self._capture_backend_metadata())
        self._active_depth_attempts = []
        self._active_selected_attempt_index = None

        # PERFORMANCE FIX (#4): Use pre-computed paths from parallel batch if available
        if _precomputed_paths:
            output_key = _precomputed_paths["output_key"]
            depth_path = _precomputed_paths["depth_path"]
            manifest_path = _precomputed_paths["manifest_path"]
            skip_depth = _precomputed_paths.get("should_skip", False)
            logger.info(f"Processing {output_key} (using precomputed paths)...")
        else:
            # Generate output key for consistent artifact naming
            use_xxhash = getattr(self.config, "use_xxhash", False)
            output_key = (
                make_output_key(image_input.path, input_root, use_xxhash=use_xxhash)
                if input_root
                else Path(sanitize_file_stem(image_input.path.stem))
            )
            logger.info(f"Processing {output_key}...")

            # Define output paths
            depth_path = self.depth_dir / output_key.parent / f"{output_key.name}_depth.png"
            manifest_path = self.manifests_dir / output_key.parent / f"{output_key.name}_combined.json"

            # Determine skip logic
            skip_depth = not self.config.force_depth and self.should_skip_depth(depth_path, manifest_path, image_input)

        # Always compute these paths (not part of skip logic)
        float_depth_path = self.depth_dir / output_key.parent / f"{output_key.name}_depth.npy"
        active_batch_id = getattr(self, "_active_batch_id", None)
        v2_log_path = self.logs_dir / output_key.parent / _v2_log_filename(output_key.name, active_batch_id)

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
                    "backend": backend_selection_metadata.resolved_backend if backend_selection_metadata else None,
                    "fallback_used": bool(
                        backend_selection_metadata and backend_selection_metadata.resolution_status != "success"
                    ),
                    "attempts": depth_attempts,
                    "selected_attempt_index": None,
                }

        # Runtime invariant checks for attempt-selection consistency.
        selected_attempt_index = getattr(self, "_active_selected_attempt_index", None)
        if depth_metadata is not None and depth_attempts:
            if selected_attempt_index is None or selected_attempt_index < 0 or selected_attempt_index >= len(depth_attempts):
                raise RuntimeError(
                    "Depth attempt invariant violated: selected_attempt_index is out of range for attempt history."
                )
            selected_attempt_backend = depth_attempts[selected_attempt_index].get("backend")
            resolved_backend = backend_selection_metadata.resolved_backend if backend_selection_metadata else None
            if selected_attempt_backend != resolved_backend:
                raise RuntimeError(
                    "Depth attempt invariant violated: selected attempt backend does not match resolved backend."
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
        # Use enhanced image from Materials V3 if available, otherwise use original
        v2_input_path = enhanced_image_path if enhanced_image_path else image_input.path
        if enhanced_image_path:
            logger.info(f"V2 stage using Materials V3 enhanced image: {enhanced_image_path}")

        self._enforce_apex_v2_canonical_input_preflight(
            depth_path=depth_path if depth_metadata else None,
            output_key=output_key,
            v2_input_path=v2_input_path,
            enhanced_image_path=enhanced_image_path,
            materials_v3_result=materials_v3_result,
        )

        v2_result, v2_runtime_s, v2_report_path = self._run_v2_stage(
            image_input=ImageInput(path=v2_input_path) if enhanced_image_path else image_input,
            depth_path=depth_path if depth_metadata else None,
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

        # Capture end time for accurate timestamps
        pipeline_end_time = time.time()

        # Clean up temporary enhanced image file if it was created
        if enhanced_image_path and enhanced_image_path.exists():
            try:
                enhanced_image_path.unlink()
                logger.debug(f"Cleaned up temporary enhanced image: {enhanced_image_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary enhanced image: {e}")

        # --- MANIFEST WRITING ---
        self._write_manifest(
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

        segmentation_mask_path: Optional[str] = None
        if materials_v3_result:
            materials_v3_metadata = materials_v3_result.get("materials_v3_metadata")
            if isinstance(materials_v3_metadata, dict):
                segmentation_metadata = materials_v3_metadata.get("segmentation_metadata")
                if isinstance(segmentation_metadata, dict):
                    mask_artifact_path = segmentation_metadata.get("mask_artifact_path")
                    if isinstance(mask_artifact_path, str) and mask_artifact_path:
                        segmentation_mask_path = mask_artifact_path

        return {
            "status": "ok",
            "image": str(image_input.path),
            "backend": backend_selection_metadata.resolved_backend if backend_selection_metadata else None,
            "fallback_used": bool(backend_selection_metadata and backend_selection_metadata.resolution_status != "success"),
            "model_id": backend_selection_metadata.model_id if backend_selection_metadata else None,
            "device": backend_selection_metadata.device if backend_selection_metadata else None,
            "attempts": depth_attempts,
            "selected_attempt_index": selected_attempt_index,
            "depth_path": str(depth_path) if depth_metadata else None,
            "depth_float_path": (
                str(float_depth_path)
                if getattr(self.config, "save_float_depth", False) and float_depth_path.exists()
                else None
            ),
            "manifest": str(manifest_path),
            "v2_log_path": str(v2_log_path) if v2_log_path.exists() else None,
            "v2_report_path": str(v2_report_path) if v2_report_path else None,
            "v2_output_path": v2_output_path,
            "segmentation_mask_path": segmentation_mask_path,
            "runtime_s": pipeline_end_time - pipeline_start_time,
        }

    def _verify_pbr_outputs(self, pbr_assets: Optional[Dict[str, Any]]) -> bool:
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
        """Return True when apex + Materials V3 strict gate should be enforced."""
        return str(getattr(self.config, "quality_tier", "")).lower() == "apex" and bool(
            getattr(self.config, "enable_materials_v3", False)
        )

    def _is_apex_tier(self) -> bool:
        """Return True when APEX quality tier is active."""
        return str(getattr(self.config, "quality_tier", "")).lower() == "apex"

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

        gate = np.clip((raw - p1) / (p99 - p1), 0.0, 1.0).astype(np.float32, copy=False)
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
        gate_depth, normalization = self._normalize_depth_for_gate(raw_depth, depth_units=depth_units)

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

        saturation_high_value = float(getattr(self.config, "apex_depth_saturation_high_value", 0.999))
        saturation_low_value = float(getattr(self.config, "apex_depth_saturation_low_value", 0.001))
        saturation_high_fraction = float((vals >= saturation_high_value).mean())
        saturation_low_fraction = float((vals <= saturation_low_value).mean())

        grad_y, grad_x = np.gradient(depth)
        grad_mag = np.hypot(grad_x, grad_y)
        grad_mag = grad_mag[np.isfinite(grad_mag)]
        gradient_energy = float(np.mean(np.abs(grad_mag))) if grad_mag.size else 0.0

        hist_bins = int(getattr(self.config, "apex_depth_hist_bins", 64))
        hist, _ = np.histogram(np.clip(vals, 0.0, 1.0), bins=hist_bins, range=(0.0, 1.0))
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

    def _enforce_apex_depth_validity_gate(
        self,
        depth_map: np.ndarray,
        depth_units: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """APEX-only depth quality gate to prevent plateau/saturation degradation."""
        if not self._is_apex_tier():
            return None

        metrics = self._compute_depth_validity_metrics(depth_map, depth_units=depth_units)

        thresholds = {
            "finite_pct_min": float(getattr(self.config, "apex_depth_min_finite_pct", 0.999)),
            "upper_iqr_min": float(getattr(self.config, "apex_depth_min_upper_iqr", 1e-4)),
            "saturation_high_fraction_max": float(getattr(self.config, "apex_depth_max_high_saturation_fraction", 0.02)),
            "saturation_low_fraction_max": float(getattr(self.config, "apex_depth_max_low_saturation_fraction", 0.02)),
            "gradient_energy_min": float(getattr(self.config, "apex_depth_min_gradient_energy", 5e-4)),
            "saturation_high_value": float(getattr(self.config, "apex_depth_saturation_high_value", 0.999)),
            "saturation_low_value": float(getattr(self.config, "apex_depth_saturation_low_value", 0.001)),
            "hist_bins": int(getattr(self.config, "apex_depth_hist_bins", 64)),
        }

        finite_fail = float(metrics.get("finite_pct") or 0.0) < thresholds["finite_pct_min"]
        plateau_fail = (metrics.get("upper_iqr") is None) or (float(metrics["upper_iqr"]) <= thresholds["upper_iqr_min"])
        high_saturation_fail = (metrics.get("saturation_high_fraction") is None) or (
            float(metrics["saturation_high_fraction"]) >= thresholds["saturation_high_fraction_max"]
        )
        low_saturation_fail = (metrics.get("saturation_low_fraction") is None) or (
            float(metrics["saturation_low_fraction"]) >= thresholds["saturation_low_fraction_max"]
        )
        low_gradient = (metrics.get("gradient_energy") is None) or (
            float(metrics["gradient_energy"]) <= thresholds["gradient_energy_min"]
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

        if failure_codes:
            details = {
                "passed": False,
                "failure_codes": failure_codes,
                "metrics": metrics,
                "thresholds": thresholds,
            }
            raise ApexStrictGateError(
                failure_codes[0] if len(failure_codes) == 1 else "APEX_DEPTH_INVALID",
                "APEX depth validity gate failed: " + ", ".join(failure_codes),
                details=details,
            )

        warnings: List[str] = []
        if low_gradient:
            warnings.append("APEX_DEPTH_GRADIENT_LOW")
            logger.warning("APEX depth validity warning: low gradient energy (metrics=%s, thresholds=%s)", metrics, thresholds)
        warnings = sorted(warnings)

        return {
            "passed": True,
            "failure_codes": [],
            "warnings": warnings,
            "metrics": metrics,
            "thresholds": thresholds,
        }

    def _enforce_apex_materials_gate(self, segmentation_result: Optional[Dict[str, Any]] = None) -> None:
        """Enforce APEX strict Materials V3 gate.

        Gate policy (apex + materials_v3 only):
        - Segmentation must be explicitly enabled
        - Backend must not be stub
        - Strict backend mode must be on (no silent fallback)
        - Segmentation output must contain at least one material mask
        """
        if not self._is_apex_materials_gate_enabled():
            return

        if not bool(getattr(self.config, "enable_material_segmentation", False)):
            raise ApexStrictGateError(
                "APEX_MATERIALS_SEGMENTATION_DISABLED",
                "APEX strict gate violated: Materials V3 in apex tier requires segmentation enabled "
                "(set --enable-segmentation on).",
            )

        backend_name = str(getattr(self.config, "material_segmentation_backend", "stub")).lower()
        if backend_name == "stub":
            raise ApexStrictGateError(
                "APEX_MATERIALS_STUB_BACKEND",
                "APEX strict gate violated: Materials V3 in apex tier cannot use stub segmentation backend "
                "(set --segmentation-backend efficientsam or sam2).",
            )

        if not bool(getattr(self.config, "strict_backend", False)):
            raise ApexStrictGateError(
                "APEX_MATERIALS_STRICT_SEGMENTATION_REQUIRED",
                "APEX strict gate violated: Materials V3 in apex tier requires strict segmentation backend mode "
                "(set --strict-segmentation).",
            )

        if segmentation_result is None:
            return

        materials = segmentation_result.get("materials", {}) if isinstance(segmentation_result, dict) else {}
        if not materials:
            raise ApexStrictGateError(
                "APEX_MATERIALS_EMPTY_SEGMENTATION",
                "APEX strict gate violated: segmentation produced no material masks; "
                "failing instead of continuing with 0 Materials V3 operations.",
            )

    def _load_cached_depth(self, depth_path: Path, float_depth_path: Path):
        """Load cached depth data, preferring float precision.

        Args:
            depth_path: Path to quantized depth PNG
            float_depth_path: Path to float depth .npy file

        Returns:
            Depth array (numpy), or None if loading fails
        """

        # Prefer float depth for better PBR quality (avoid quantization artifacts)
        if float_depth_path.exists():
            try:
                depth_data = np.load(str(float_depth_path))
                logger.debug(f"Loaded float depth from: {float_depth_path}")
                return depth_data
            except Exception as e:
                logger.warning(f"Failed to load float depth: {e}")

        # Fall back to quantized depth image
        if depth_path.exists():
            try:
                from .depth_writer import read_depth_u16_png

                depth_data = read_depth_u16_png(depth_path)

                # Robust normalization - handle both uint16 and pre-normalized float
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

                logger.debug(f"Loaded quantized depth from: {depth_path}")
                return depth_data
            except Exception as e:
                logger.warning(f"Failed to load depth image: {e}")

        return None

    def _parallel_preprocess_batch(
        self, image_inputs: List[ImageInput], input_root: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """Parallel preprocessing: validation, output key generation, skip logic.

        Phase 2: I/O-bound operations parallelized with ThreadPoolExecutor.

        Args:
            image_inputs: List of images to preprocess
            input_root: Base directory for relative path calculation

        Returns:
            List of preprocessing results with skip flags and paths
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._preprocess_single, img, input_root): img for img in image_inputs}

            results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    img = futures[future]
                    logger.error(f"Preprocessing failed for {img.path}: {e}")
                    results.append({"status": "error", "image_input": img, "error": str(e)})

            return results

    def _preprocess_single(self, image_input: ImageInput, input_root: Optional[Path]) -> Dict[str, Any]:
        """Preprocess single image: generate paths and check skip logic.

        Args:
            image_input: Input image information
            input_root: Base directory for relative path calculation

        Returns:
            Dictionary with preprocessing metadata
        """
        use_xxhash = getattr(self.config, "use_xxhash", False)
        output_key = (
            make_output_key(image_input.path, input_root, use_xxhash=use_xxhash)
            if input_root
            else Path(sanitize_file_stem(image_input.path.stem))
        )

        depth_path = self.depth_dir / output_key.parent / f"{output_key.name}_depth.png"
        manifest_path = self.manifests_dir / output_key.parent / f"{output_key.name}_combined.json"

        # Check skip logic (uses cached manifest loading from Phase 1)
        should_skip = not self.config.force_depth and self.should_skip_depth(depth_path, manifest_path, image_input)

        return {
            "status": "ok",
            "image_input": image_input,
            "output_key": output_key,
            "should_skip": should_skip,
            "depth_path": depth_path,
            "manifest_path": manifest_path,
        }

    def enhance_batch_parallel(
        self, image_inputs: List[ImageInput], input_root: Optional[Path] = None
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
            logger.debug(f"Using sequential processing (batch size: {len(image_inputs)})")
            return [self.enhance_image(img, input_root) for img in image_inputs]

        logger.info(f"Parallel batch processing: {len(image_inputs)} images with {self.max_workers} workers")

        # Phase 1: Parallel preprocessing (I/O-bound)
        preprocessed = self._parallel_preprocess_batch(image_inputs, input_root)

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
                result = self.enhance_image(item["image_input"], input_root, _precomputed_paths=precomputed)
                results.append(result)
            except Exception as e:
                logger.error(f"Enhancement failed for {item['image_input'].path}: {e}")
                error_payload: Dict[str, Any] = {"status": "error", "image": str(item["image_input"].path), "error": str(e)}
                error_payload["backend"] = getattr(self, "_active_backend_metadata", self._backend_metadata).resolved_backend
                error_payload["attempts"] = list(getattr(self, "_active_depth_attempts", []) or [])
                error_payload["selected_attempt_index"] = getattr(self, "_active_selected_attempt_index", None)
                if isinstance(e, ApexStrictGateError):
                    error_payload["error_code"] = e.code
                    error_payload["error_details"] = e.details
                results.append(error_payload)

        return results

    def enhance_batch(self, input_dir: Path, image_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Process a batch of images with accurate execution timestamps.

        Args:
            input_dir: Directory containing input images
            image_extensions: List of file extensions to process

        Returns:
            List of processing results for each image
        """
        if image_extensions is None:
            image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

        # Capture accurate batch start time
        batch_start_time = time.time()
        batch_start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(batch_start_time))

        batch_id = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._active_batch_id = batch_id
        logger.info(f"Batch {batch_id}: Scanning {input_dir}")

        # ADR-023 Phase 3: Capture backend selection metadata and log truth line
        backend_metadata = self._capture_backend_metadata()
        logger.info(
            "Backend selection: requested=%s resolved=%s status=%s device=%s model=%s",
            backend_metadata.requested_backend or "auto",
            backend_metadata.resolved_backend,
            backend_metadata.resolution_status,
            backend_metadata.device,
            backend_metadata.model_id,
        )

        # Store backend metadata for use in _write_manifest
        self._backend_metadata = backend_metadata

        # Use input discovery to exclude depth artifacts and derived outputs
        # ROBUSTNESS FIX (#6): Pass output_root to explicitly exclude output directory
        discovery_config = DiscoveryConfig(strict_mode=self.config.strict_inputs)
        images = discover_images(input_dir, discovery_config, image_extensions, output_dir=self.output_root)

        # Inert scene-group bridge: preserve existing per-image behavior and order.
        sorted_images = sorted(images)
        scene_groups = build_scene_groups(
            sorted_images,
            dataset_root=input_dir,
            grouping_mode=getattr(self.config, "grouping_mode", "single"),
        )
        image_inputs = [ImageInput(img) for scene in scene_groups for img in scene.images]

        if self._use_parallel and len(image_inputs) >= 4:
            logger.info(f"Using parallel batch processing for {len(image_inputs)} images")
            results = self.enhance_batch_parallel(image_inputs, input_root=input_dir)
        else:
            # Sequential processing (original behavior)
            results = []
            for img_input in image_inputs:
                try:
                    results.append(self.enhance_image(img_input, input_root=input_dir))
                except Exception as e:
                    logger.error(f"Failed {img_input.path}: {e}")
                    error_payload: Dict[str, Any] = {"status": "error", "image": str(img_input.path), "error": str(e)}
                    error_payload["backend"] = getattr(
                        self, "_active_backend_metadata", self._backend_metadata
                    ).resolved_backend
                    error_payload["attempts"] = list(getattr(self, "_active_depth_attempts", []) or [])
                    error_payload["selected_attempt_index"] = getattr(self, "_active_selected_attempt_index", None)
                    if isinstance(e, ApexStrictGateError):
                        error_payload["error_code"] = e.code
                        error_payload["error_details"] = e.details
                    results.append(error_payload)

        if getattr(self.config, "enable_reconstruction", False):
            self._run_scene_reconstruction_stage(scene_groups=scene_groups, results=results, dataset_root=input_dir)

        # Capture accurate batch end time
        batch_end_time = time.time()
        batch_end_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(batch_end_time))

        # Write batch summary with accurate timestamps
        # Extract runtime_s from successful results for statistics computation
        runtimes = [r.get("runtime_s", 0.0) for r in results if r.get("status") == "ok"]
        runtime_stats = compute_batch_runtime_stats(runtimes)

        # Detect runtime outliers (images taking >5× median time)
        # PERFORMANCE FIX (#3): Compute median once, pass to all outlier checks (O(n) instead of O(n²))
        median_runtime = runtime_stats.get("median", 0.0)
        outliers = []
        for r in results:
            if r.get("status") == "ok":
                runtime_s = r.get("runtime_s", 0.0)
                image_name = r.get("image", "unknown")
                outlier_result = detect_runtime_outliers(image_name, runtime_s, runtimes, median=median_runtime)
                if outlier_result:
                    warning_msg, outlier_meta = outlier_result
                    outliers.append(
                        {
                            "image": image_name,
                            "metadata": outlier_meta,
                        }
                    )

        bm = BatchManifest(
            batch_id=batch_id,
            start_time=batch_start_utc,
            end_time=batch_end_utc,
            config={"model": self.config.model_variant.value.name},
            results=results,
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
            )

        return results

    @staticmethod
    def _result_image_key(path_value: str) -> Optional[str]:
        """Normalize result image path for lookup joins."""
        if not isinstance(path_value, str) or not path_value:
            return None
        try:
            return str(Path(path_value).resolve())
        except Exception:
            return None

    def _run_scene_reconstruction_stage(
        self,
        *,
        scene_groups: List[Any],
        results: List[Dict[str, Any]],
        dataset_root: Path,
    ) -> None:
        """Run gated scene-level reconstruction for eligible grouped scenes."""
        if not bool(getattr(self.config, "non_commercial_ok", False)):
            raise LicenseRestrictionError(
                "Scene reconstruction requires non_commercial_ok=True due to "
                "Inria 3D Gaussian Splatting non-commercial license terms."
            )
        if not bool(getattr(self.config, "accept_research_tools_license", False)):
            raise LicenseRestrictionError(
                "Scene reconstruction requires accept_research_tools_license=True "
                "to acknowledge research-only tool licensing constraints."
            )

        sidecar_value = getattr(self.config, "cameras_sidecar_path", None)
        sidecar_path = Path(sidecar_value) if isinstance(sidecar_value, str) and sidecar_value else None
        reconstruction_tier = str(getattr(self.config, "reconstruction_tier", "apex_research"))

        result_by_path: Dict[str, Dict[str, Any]] = {}
        for result in results:
            path_key = self._result_image_key(result.get("image"))
            if path_key:
                result_by_path[path_key] = result

        for scene in scene_groups:
            if len(scene.images) < 2:
                continue

            scene_results: List[Dict[str, Any]] = []
            for image_path in scene.images:
                result = result_by_path.get(str(Path(image_path).resolve()))
                if not isinstance(result, dict) or result.get("status") != "ok":
                    scene_results = []
                    break
                scene_results.append(result)
            if not scene_results:
                continue

            cameras = load_scene_cameras(scene=scene, dataset_root=dataset_root, sidecar_path=sidecar_path)
            if not cameras:
                logger.info("Skipping reconstruction for scene %s: cameras unavailable", scene.scene_id)
                continue
            camera_sources = {camera.provenance.source for camera in cameras}
            if len(camera_sources) > 1:
                logger.warning(
                    "Skipping reconstruction for scene %s: mixed camera sources %s",
                    scene.scene_id,
                    sorted(camera_sources),
                )
                continue
            if any(camera.provenance.confidence == "low" for camera in cameras):
                logger.warning(
                    "Skipping reconstruction for scene %s: low-confidence camera provenance detected",
                    scene.scene_id,
                )
                continue

            context = SceneContext.build(
                scene=scene,
                dataset_root=dataset_root,
                cameras=cameras,
            )

            try:
                report_path = self.run_scene_reconstruction_fn(
                    context=context,
                    output_dir=self.reconstruction_dir,
                    iterations=int(getattr(self.config, "reconstruction_iterations", 1000)),
                    tier=reconstruction_tier,
                )
            except (LicenseRestrictionError, ReconstructionLicenseRestrictionError):
                raise
            except Exception as exc:
                logger.warning("Scene reconstruction failed for %s: %s", scene.scene_id, exc)
                continue

            if not isinstance(report_path, Path):
                report_path = Path(str(report_path))
            if not report_path.exists():
                logger.warning("Scene reconstruction returned missing report path for %s: %s", scene.scene_id, report_path)
                continue

            scene_results[0]["reconstruction_report_path"] = str(report_path)
            scene_results[0]["reconstruction_scene_id"] = scene.scene_id
            logger.info("Scene reconstruction completed: scene_id=%s report=%s", scene.scene_id, report_path)

    def _collect_run_card_artifact_paths(
        self, results: List[Dict[str, Any]], batch_manifest_path: Optional[Path] = None
    ) -> List[Path]:
        """Collect deterministic artifact paths associated with the current batch."""
        artifact_paths: List[Path] = []
        batch_id: Optional[str] = None

        if batch_manifest_path and batch_manifest_path.exists():
            artifact_paths.append(batch_manifest_path)
            batch_name = batch_manifest_path.stem
            if batch_name.startswith("batch_"):
                batch_id = batch_name[len("batch_") :]

        for result in results:
            for direct_path_key in (
                "v2_log_path",
                "v2_report_path",
                "v2_output_path",
                "segmentation_mask_path",
                "reconstruction_report_path",
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

                    provenance_sidecar_path = manifest_path.with_name(f"{manifest_path.stem}_provenance.json")
                    if provenance_sidecar_path.exists():
                        artifact_paths.append(provenance_sidecar_path)

                    # Include the per-image V2 stage log when available.
                    manifest_name = manifest_path.stem
                    output_key_name = manifest_name.removesuffix("_combined")
                    try:
                        manifest_relative_parent = manifest_path.resolve().relative_to(self.manifests_dir.resolve()).parent
                    except ValueError:
                        manifest_relative_parent = Path(".")
                    v2_log_path = self.logs_dir / manifest_relative_parent / _v2_log_filename(output_key_name, batch_id)

                    if v2_log_path.exists():
                        artifact_paths.append(v2_log_path)

                    try:
                        combined_manifest = CombinedManifest.load(manifest_path)
                    except Exception as exc:
                        logger.debug(f"Skipping artifact extraction for unreadable manifest {manifest_path}: {exc}")
                    else:
                        if combined_manifest.depth and combined_manifest.depth.depth_path:
                            artifact_paths.append(Path(combined_manifest.depth.depth_path))

                        if combined_manifest.v2 and combined_manifest.v2.report_path:
                            report_path = Path(combined_manifest.v2.report_path)
                            artifact_paths.append(report_path)
                            if report_path.exists():
                                try:
                                    with open(report_path, "r", encoding="utf-8") as report_file:
                                        report_payload = json.load(report_file)
                                except Exception as exc:
                                    logger.debug(f"Failed to parse V2 report for artifact indexing ({report_path}): {exc}")
                                else:
                                    for field in ("output", "depth_map"):
                                        value = report_payload.get(field)
                                        if isinstance(value, str) and value:
                                            artifact_paths.append(Path(value))

                        if combined_manifest.pbr_assets:
                            for key, value in combined_manifest.pbr_assets.items():
                                if key.endswith("_path") and isinstance(value, str) and value:
                                    artifact_paths.append(Path(value))

                        if combined_manifest.materials_v3 and isinstance(
                            combined_manifest.materials_v3.segmentation_metadata, dict
                        ):
                            mask_artifact_path = combined_manifest.materials_v3.segmentation_metadata.get("mask_artifact_path")
                            if isinstance(mask_artifact_path, str) and mask_artifact_path:
                                artifact_paths.append(Path(mask_artifact_path))

            depth_value = result.get("depth_path")
            if depth_value:
                depth_path = Path(depth_value)
                artifact_paths.append(depth_path)
                depth_metadata_path = depth_path.with_name(f"{depth_path.stem}_metadata.json")
                if depth_metadata_path.exists():
                    artifact_paths.append(depth_metadata_path)
                float_depth_path = depth_path.with_suffix(".npy")
                if float_depth_path.exists():
                    artifact_paths.append(float_depth_path)

        return artifact_paths

    def _compute_backend_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute concise backend fallback summary for run-card telemetry."""
        requested_backend = self._backend_metadata.requested_backend or "auto"
        observed_ok_backends = {
            str(result.get("backend")) for result in results if result.get("status") == "ok" and result.get("backend")
        }
        preferred_backend = self._backend_metadata.resolved_backend
        final_backends_used: List[str] = []
        if preferred_backend in observed_ok_backends:
            final_backends_used.append(preferred_backend)
            final_backends_used.extend(sorted(observed_ok_backends - {preferred_backend}))
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

    def _emit_run_card(
        self,
        batch_id: str,
        start_time: str,
        end_time: str,
        results: List[Dict[str, Any]],
        runtime_stats: Dict[str, Any],
        outliers: List[Dict[str, Any]],
        batch_manifest_path: Optional[Path] = None,
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

        artifact_paths = self._collect_run_card_artifact_paths(results, batch_manifest_path=batch_manifest_path)
        artifact_index = _build_artifact_index(self.output_root, artifact_paths)
        artifact_merkle_root = _compute_artifact_merkle_root(artifact_index)
        backend_summary = self._compute_backend_summary(results)
        requested_backend = self._backend_metadata.requested_backend or "auto"
        backend_selection_resolved = (
            backend_summary["final_backends_used"][0]
            if backend_summary["final_backends_used"]
            else (self._backend_metadata.resolved_backend)
        )

        backend_selection: Dict[str, Any] = {
            "requested": requested_backend,
            "resolved": backend_selection_resolved,
            "device": self._backend_metadata.device,
            "model_id": self._backend_metadata.model_id,
        }
        # Explicit wrapper semantics when logical backend delegates to a different runtime engine.
        if self._backend_metadata.resolved_backend != backend_selection_resolved and backend_summary["fallback_images"] == 0:
            backend_selection["logical_backend"] = self._backend_metadata.resolved_backend
            backend_selection["resolved_engine"] = backend_selection_resolved

        run_card = {
            "batch_id": batch_id,
            "start_time": start_time,
            "end_time": end_time,
            "config_fingerprint": self._build_run_card_config_fingerprint(),
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
            "artifact_merkle_root": artifact_merkle_root,
        }

        def _json_default(obj: Any):
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
                canonical_json = json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":"))
                return {
                    **fingerprint_payload,
                    "hash_algorithm": "sha256",
                    "canonical_json": canonical_json,
                    "sha256": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
                }

            # --- Enum handling ---
            if isinstance(obj, Enum):
                return obj.value

            # --- Path handling ---
            if isinstance(obj, Path):
                return str(obj)

            # --- Dataclass-safe conversion ---
            if hasattr(obj, "__dataclass_fields__"):
                return {k: getattr(obj, k) for k in obj.__dataclass_fields__.keys()}

            # --- Explicit to_dict support ---
            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                return obj.to_dict()

            # --- Controlled __dict__ fallback (avoid deep recursion) ---
            if hasattr(obj, "__dict__") and not isinstance(obj, (np.ndarray,)):
                return {k: v for k, v in vars(obj).items() if not k.startswith("_")}

            # --- Final deterministic fallback ---
            return str(obj)

        schema_path = _run_card_schema_path()
        if not schema_path.exists():
            logger.warning(
                "Run card schema not found at %s; skipping run card emission for batch_id=%s",
                schema_path,
                batch_id,
            )
            return

        try:
            serialized_run_card = json.loads(
                dumps_json(
                    run_card,
                    default=_json_default,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            _validate_run_card_payload(serialized_run_card, schema_path)
            _validate_run_card_backend_semantics(serialized_run_card)

            with open(run_card_path, "w", encoding="utf-8") as f:
                dump_json(
                    run_card,
                    f,
                    indent=2,
                    default=_json_default,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
        except Exception:
            logger.exception(
                "Run card emission failed for batch_id=%s (schema: %s, output: %s). Continuing without run card.",
                batch_id,
                schema_path,
                run_card_path,
            )
            return

        logger.info(f"✅ Run card emitted: {run_card_path}")
