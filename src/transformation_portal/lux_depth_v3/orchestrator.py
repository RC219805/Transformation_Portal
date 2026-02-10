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
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from .batch_stats import compute_batch_runtime_stats, detect_runtime_outliers

# Note: Imports adjusted to relative for package context compatibility
from .config import DA3Config, EnhanceConfig, ModelVariant
from .depth_cache import DepthCache
from .depth_writer import atomic_write_depth_u16_png_with_stats
from .input_discovery import DiscoveryConfig, discover_images
from .input_manager import ImageInput
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
from .security import HashMode, sanitize_file_stem, sanitize_path_component_nonlossy
from .v2_runner import V2Runner, find_v2_report

logger = logging.getLogger(__name__)


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
    import os

    hf_token = os.environ.get("HF_TOKEN")
    status["hf_token"] = bool(hf_token)
    if hf_token:
        logger.debug("HF_TOKEN present - authenticated model downloads enabled")
    else:
        logger.debug("HF_TOKEN not set - using unauthenticated downloads (rate limits apply, slower warm starts)")
        logger.debug("  Set HF_TOKEN for faster downloads: export HF_TOKEN=<your_token>")

    return status


logger = logging.getLogger(__name__)


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
    """Generate collision-free output key preserving directory structure.

    Creates a unique output key that:
    1. Preserves the input's directory structure relative to input_root
    2. Includes the sanitized original extension (without dot)
    3. Appends an 8-character hash of the full relative path

    Phase 3: xxHash is now default when available (5x faster than SHA-1).

    This ensures unique output names even for files with the same name
    in different directories or with different extensions.

    Args:
        input_path: Full path to input file
        input_root: Base directory for relative path calculation
        use_xxhash: Use xxHash instead of SHA-1 (default: True if available, else False)

    Returns:
        Path object representing the output key (without final extension)

    Example:
        Input: photos/scene1/image.JPG with root=photos/
        Output: scene1/image_jpg_1a2b3c4d
    """
    try:
        relpath = input_path.relative_to(input_root)
    except ValueError:
        logger.warning(f"{input_path} is not relative to {input_root}, using flat naming")
        relpath = Path(input_path.name)

    # Extract directory structure and file components
    rel_dir = relpath.parent
    name = relpath.stem
    ext = relpath.suffix  # e.g., ".jpg"

    # Sanitize directory parts
    sanitized_parts = [sanitize_path_component_nonlossy(p) for p in rel_dir.parts]

    # Sanitize extension (remove dot, lowercase, default to "noext")
    ext_label = ext.lstrip(".").lower() if ext else "noext"
    ext_label = sanitize_path_component_nonlossy(ext_label)

    # Compute 8-char hash of full relative path for uniqueness
    # Phase 3: Use xxHash if available and enabled (5x faster)
    hash_input = relpath.as_posix().encode("utf-8")
    if use_xxhash and XXHASH_AVAILABLE:
        hash_suffix = xxhash.xxh64(hash_input).hexdigest()[:8]
    else:
        # Use SHA1 for file naming (not cryptographic security)
        hash_suffix = hashlib.sha1(hash_input, usedforsecurity=False).hexdigest()[:8]

    # Sanitize stem
    stem_sanitized = sanitize_path_component_nonlossy(name)

    # Construct output key: {stem}_{ext}_{hash}
    key_name = f"{stem_sanitized}_{ext_label}_{hash_suffix}"

    if sanitized_parts:
        return Path(*sanitized_parts) / key_name
    else:
        return Path(key_name)


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
        # zones/ directory reserved for future zone-based processing
        # Only created when zoning features are enabled
        self.zones_dir = self.output_root / "zones"

        for d in [self.depth_dir, self.v2_dir, self.manifests_dir, self.logs_dir]:
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

    def _initialize_depth_backend(self) -> None:
        """Initialize depth backend using registry (ADR-019).

        Implements backend selection with fallback logic:
        1. Try requested backend (from config.depth_backend)
        2. Check availability (checkpoint + dependencies)
        3. Fallback to DA3 if unavailable
        4. Record selection decision in metadata
        """
        requested = self.config.depth_backend or "da3"
        registry = DepthBackendRegistry()

        try:
            # Get backend from registry
            backend = registry.get_backend(requested, self.config)

            # Check availability
            try:
                backend.ensure_available()
                available = True
                reason = f"{backend.name} backend ready"
            except (ImportError, FileNotFoundError) as e:
                available = False
                reason = str(e)

            if not available:
                # Fallback to DA3
                logger.warning(f"Backend '{requested}' unavailable: {reason}. Falling back to DA3.")
                backend = registry.get_backend("da3", self.config)
                try:
                    backend.ensure_available()
                    resolved = "da3"
                    status = "fallback"
                except (ImportError, FileNotFoundError) as fallback_error:
                    # DA3 also unavailable (likely test environment without ML dependencies)
                    # Check if synthetic fallback is explicitly allowed
                    if not self.config.allow_synthetic_fallback:
                        # Check environment variable override (for CI)
                        import os

                        if not os.getenv("TP_ALLOW_SYNTHETIC_FALLBACK") == "1":
                            raise RuntimeError(
                                f"No depth backend available: {fallback_error}. "
                                "Install ML dependencies (torch, transformers) or explicitly enable "
                                "synthetic fallback for testing (config.allow_synthetic_fallback=True "
                                "or TP_ALLOW_SYNTHETIC_FALLBACK=1)."
                            ) from fallback_error

                    # Synthetic fallback explicitly allowed
                    logger.warning(
                        f"DA3 fallback also unavailable: {fallback_error}. "
                        f"Using synthetic backend for testing (no ML dependencies)."
                    )
                    backend = registry.get_backend("synthetic", self.config)
                    resolved = "synthetic"
                    status = "synthetic_fallback"
                    reason = f"Test environment - using synthetic depth: {fallback_error}"
            else:
                resolved = requested
                status = "success"

            self.depth_backend = backend
            self._backend_metadata = BackendSelectionMetadata(
                requested_backend=requested,
                resolved_backend=resolved,
                resolution_status=status,
                resolution_reason=reason,
                model_id=self.config.model_variant.value.huggingface_id,
                device=self.config.depth_device,
            )

            logger.info(f"Depth backend: requested={requested} resolved={resolved} device={self.config.depth_device}")

        except LicenseRestrictionError as e:
            logger.error(f"License restriction: {e}")
            raise
        except Exception as e:
            logger.error(f"Backend initialization failed: {e}")
            raise

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
    ) -> tuple[Optional[Any], float, Optional[dict], Optional[dict], float, Optional[Path]]:
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
                     materials_v3_runtime_s, enhanced_image_path)
        """
        depth_runtime_s = 0.0
        depth_metadata = None
        pbr_assets = None
        materials_v3_result = None
        materials_v3_runtime_s = 0.0
        enhanced_image_path = None  # Will be set if Materials V3 produces enhanced_image

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

            # Initialize working_image - this will be updated by Materials V3 if enabled
            working_image = preprocessed_array

            logger.info(f"Stage A: Generating depth for {output_key}...")
            t0 = time.time()
            try:
                # Phase 2: Check content-addressable depth cache
                cached_depth = None
                image_sha256 = None
                if self.depth_cache:
                    image_sha256 = self._compute_or_skip_hash(image_input.path, manifest_exists=False, for_manifest_write=True)
                    if image_sha256:
                        config_fp_hash = self.compute_config_fingerprint().depth_only().to_sha256()
                        cached_depth = self.depth_cache.get(image_sha256, config_fp_hash)
                        if cached_depth is not None:
                            logger.info(f"Cache hit: using cached depth for {output_key}")

                # 1. Inference (using preprocessed numpy array or cached depth)
                if cached_depth is not None:
                    # Use cached depth - wrap in result-like object
                    from ..depth.backends.protocol import DepthResult

                    result = DepthResult(depth_map=cached_depth, original_image=preprocessed_array, metadata={"cached": True})
                else:
                    # Run inference via backend
                    # CRITICAL FIX: preprocessing returns float32 [0,1], must scale to uint8 [0,255] for PIL
                    from PIL import Image

                    preprocessed_uint8 = (np.clip(preprocessed_array, 0, 1) * 255).astype(np.uint8)
                    pil_image = Image.fromarray(preprocessed_uint8)
                    result = self.depth_backend.compute(pil_image)

                    # Store in cache if enabled
                    if self.depth_cache and image_sha256:
                        self.depth_cache.store(image_sha256, config_fp_hash, result.depth_map)

                # 2. Post-Processing (Refinement) - skip for cached depths
                if cached_depth is None:
                    result = self.postprocessor.process(result)

                # CRITICAL FIX (#2): Resize depth map back to original dimensions
                # preprocessing.py pads/crops to multiple of 14, but depth must match original source
                depth_map = result.depth_map if hasattr(result, "depth_map") else result.depth
                current_shape = depth_map.shape[:2]  # (H, W)

                if current_shape != original_shape:
                    from PIL import Image as PILImage

                    logger.debug(f"Resizing depth map from {current_shape} back to original {original_shape}")
                    # Convert to PIL for high-quality resize
                    depth_pil = PILImage.fromarray((depth_map * 65535).astype(np.uint16), mode="I;16")
                    # Resize to original dimensions (H, W) -> PIL expects (W, H)
                    depth_pil_resized = depth_pil.resize((original_shape[1], original_shape[0]), PILImage.Resampling.LANCZOS)
                    depth_map = np.array(depth_pil_resized, dtype=np.float32) / 65535.0
                    # Update result object
                    if hasattr(result, "depth_map"):
                        result.depth_map = depth_map
                    else:
                        result.depth = depth_map

                depth_runtime_s = time.time() - t0

                # 2b. Materials V3 Processing (if enabled)
                materials_v3_result = None
                materials_v3_runtime_s = 0.0
                if self.materials_v3_engine:
                    logger.info("Running Materials V3 surface-aware finishing...")
                    t_materials_start = time.time()
                    try:
                        # Material segmentation (configurable backend)
                        # Current: stub backend (returns empty masks)
                        # Future: EfficientSAM integration for real material detection
                        from .segmentation_backend import segment_materials

                        segmentation_result = {"materials": segment_materials(preprocessed_array, self.config)}

                        # Log segmentation result if materials were detected
                        if segmentation_result.get("materials"):
                            logger.info(
                                f"Material segmentation: {len(segmentation_result['materials'])} "
                                f"materials detected using {self.config.material_segmentation_backend} backend"
                            )

                        materials_v3_result = self.materials_v3_engine.process(
                            image=preprocessed_array, segmentation_result=segmentation_result, depth_map=depth_map
                        )
                        materials_v3_runtime_s = time.time() - t_materials_start
                        if materials_v3_result:
                            # CRITICAL FIX: Use enhanced_image from Materials V3 for downstream processing
                            enhanced_image = materials_v3_result.get("enhanced_image")
                            if enhanced_image is not None:
                                working_image = enhanced_image

                                # Write enhanced image to temporary file for V2 stage
                                # V2 subprocess requires a file path, not an array
                                #
                                # NOTE: 8-bit PNG conversion is intentional:
                                # - V2 subprocess expects standard image formats (PNG/JPG, not 16-bit TIFF)
                                # - working_image is already tone-mapped/perceptually encoded (not linear)
                                # - V2 operates in perceptual color space, not linear
                                # - This matches what V2 receives when Materials V3 is disabled (original 8-bit input)
                                # - Temporary file is cleaned up after V2 completes (see line ~1352)
                                import tempfile

                                from PIL import Image as PILImage

                                # Create temp file in output directory to avoid cross-filesystem issues
                                temp_dir = self.output_root / "temp"
                                temp_dir.mkdir(parents=True, exist_ok=True)

                                # Convert working_image (float32 [0,1]) to uint8 for saving
                                enhanced_uint8 = (np.clip(working_image, 0, 1) * 255).astype(np.uint8)
                                enhanced_pil = PILImage.fromarray(enhanced_uint8)

                                # Save with output_key stem to make it identifiable
                                enhanced_image_path = temp_dir / f"{output_key.stem}_materials_v3_enhanced.png"
                                enhanced_pil.save(enhanced_image_path)

                                logger.info(
                                    f"Materials V3 enhanced image with "
                                    f"{len(materials_v3_result.get('materials_v3_pixel_ops', {}).get('applied', []))} "
                                    f"pixel operations - saved to {enhanced_image_path} for V2 stage"
                                )
                            else:
                                logger.debug("Materials V3 did not return enhanced_image, using original image")

                            logger.info(
                                f"Materials V3 completed in {materials_v3_runtime_s:.3f}s: "
                                f"{len(materials_v3_result.get('materials_v3_pixel_ops', {}).get('applied', []))} "
                                f"operations applied"
                            )
                    except Exception as e:
                        logger.warning(f"Materials V3 processing failed: {e}", exc_info=True)
                        materials_v3_result = None
                        materials_v3_runtime_s = time.time() - t_materials_start

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
                }

                # Merge inference provenance into depth stats
                _md = getattr(result, "metadata", None) or {}
                for _k in ("requested_model_id", "resolved_model_id", "resolved_model_source"):
                    if _k in _md:
                        stats[_k] = _md[_k]

                depth_metadata = DepthMetadata(
                    model=self.config.model_variant.value.name,
                    depth_path=str(depth_path),
                    runtime_seconds=depth_runtime_s,
                    scaling=depth_stats._asdict(),
                    stats=stats,
                )

                # 4. Write depth metadata JSON
                depth_metadata_path = depth_path.parent / f"{depth_path.stem}_metadata.json"
                with open(depth_metadata_path, "w") as f:
                    json.dump(
                        {
                            "model": depth_metadata.model,
                            "depth_path": depth_metadata.depth_path,
                            "runtime_seconds": depth_metadata.runtime_seconds,
                            "scaling": depth_metadata.scaling,
                            "stats": depth_metadata.stats,
                        },
                        f,
                        indent=2,
                    )
                logger.debug(f"Wrote depth metadata: {depth_metadata_path}")

                # 5. PBR map generation (optional)
                pbr_assets = self._generate_pbr_stage(result.depth, output_key)

            except Exception as e:
                logger.error(f"Depth failed: {e}")
                if self.config.depth_fallback == "fail":
                    raise
                elif self.config.depth_fallback == "skip":
                    return None, 0.0, None, None, 0.0, None
                elif self.config.depth_fallback == "v2-auto":
                    logger.info("V2 fallback mode: V3 failed, will attempt V2 with independent depth")
                    if depth_path.exists():
                        depth_path.unlink()
                    return None, 0.0, None, None, 0.0, None
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

        return depth_metadata, depth_runtime_s, pbr_assets, materials_v3_result, materials_v3_runtime_s, enhanced_image_path

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
                Note: Material masks are available but not currently passed to subprocess.
                Future: Implement mask serialization for V2 subprocess integration.

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
            return {"status": "ok"}, 0.0, v2_report_path

        # Note: material_masks from materials_v3_result are available but not passed to subprocess
        # V2Runner is a subprocess wrapper - material masks would require serialization to disk
        # Future enhancement: Serialize masks to temporary directory and pass path to V2
        # For now, infrastructure is ready (MaterialsV3Engine.process returns material_masks)
        if materials_v3_result and materials_v3_result.get("material_masks"):
            logger.debug(
                f"Materials V3 masks available ({len(materials_v3_result['material_masks'])} materials) "
                f"but not passed to V2 subprocess (requires serialization - future work)"
            )

        # V2 runner: depth_dir=None triggers independent depth generation in V2
        v2_result = self.v2_runner.run(
            input_path=image_input.path,
            depth_dir=self.depth_dir if (depth_path and depth_path.exists()) else None,
            output_dir=self.v2_dir,
            preset=self.config.v2_preset,
            device=self.config.v2_device,
            upscaler_backend=self.config.v2_upscaler_backend,
            log_file=v2_log_path,
            timeout=self.config.v2_timeout,
        )
        v2_runtime_s = v2_result.get("runtime_s", 0.0)
        v2_report_path = find_v2_report(self.v2_dir, output_key.name)

        return v2_result, v2_runtime_s, v2_report_path

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
            logger.info(f"Wrote provenance sidecar: {provenance_sidecar_path}")

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
        v2_metadata = V2Metadata(
            preset=self.config.v2_preset,
            strict_depth=depth_metadata is not None,
            output_dir="v2/",
            report_path=str(v2_report_path) if v2_report_path else "",
            status=v2_result["status"],
            error_message=v2_result.get("error"),
        )

        # Materials V3 metadata
        materials_v3_metadata = None
        if materials_v3_result:
            from .manifest import MaterialsV3Metadata

            materials_v3_metadata = MaterialsV3Metadata(
                enabled=True,
                version=materials_v3_result.get("materials_v3_metadata", {}).get("version", "3.1"),
                response_plan=materials_v3_result.get("materials_v3_response_plan"),
                pixel_ops=materials_v3_result.get("materials_v3_pixel_ops"),
                runtime_seconds=materials_v3_runtime_s,
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
            backend_selection=self._backend_metadata,
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
        v2_log_path = self.logs_dir / output_key.parent / f"v2_{output_key.name}.log"

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
                return {"status": "skipped", "reason": "Depth computation failed", "image": str(image_input.path)}

        # --- STAGE B: V2 ENHANCEMENT ---
        # Use enhanced image from Materials V3 if available, otherwise use original
        v2_input_path = enhanced_image_path if enhanced_image_path else image_input.path
        if enhanced_image_path:
            logger.info(f"V2 stage using Materials V3 enhanced image: {enhanced_image_path}")

        v2_result, v2_runtime_s, v2_report_path = self._run_v2_stage(
            image_input=ImageInput(path=v2_input_path) if enhanced_image_path else image_input,
            depth_path=depth_path if depth_metadata else None,
            output_key=output_key,
            v2_log_path=v2_log_path,
            manifest_path=manifest_path,
            skip_depth=skip_depth,
            materials_v3_result=materials_v3_result,
        )

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
        )

        return {
            "status": "ok",
            "image": str(image_input.path),
            "depth_path": str(depth_path) if depth_metadata else None,
            "manifest": str(manifest_path),
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

    def _load_cached_depth(self, depth_path: Path, float_depth_path: Path):
        """Load cached depth data, preferring float precision.

        Args:
            depth_path: Path to quantized depth PNG
            float_depth_path: Path to float depth .npy file

        Returns:
            Depth array (numpy), or None if loading fails
        """
        import numpy as np

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
                results.append({"status": "error", "image": str(item["image_input"].path), "error": str(e)})

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

        # Phase 2: Use parallel batch processing if enabled
        image_inputs = [ImageInput(img) for img in sorted(images)]

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
                    results.append({"status": "error", "image": str(img_input.path), "error": str(e)})

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
        bm.write(self.manifests_dir / f"batch_{batch_id}.json")

        # Emit run card if enabled
        if self.config.emit_run_card:
            self._emit_run_card(batch_id, batch_start_utc, batch_end_utc, results, runtime_stats, outliers)

        return results

    def _emit_run_card(
        self,
        batch_id: str,
        start_time: str,
        end_time: str,
        results: List[Dict[str, Any]],
        runtime_stats: Dict[str, Any],
        outliers: List[Dict[str, Any]],
    ) -> None:
        """Emit run card for batch reproducibility.

        Creates a human-readable and machine-parseable run card with:
        - Configuration fingerprint
        - Runtime statistics
        - Outlier warnings
        - Environment metadata

        Args:
            batch_id: Unique batch identifier
            start_time: Batch start time (ISO 8601 UTC)
            end_time: Batch end time (ISO 8601 UTC)
            results: List of per-image processing results
            runtime_stats: Aggregated runtime statistics
            outliers: List of detected runtime outliers
        """
        run_card_path = self.output_root / f"run_card_{batch_id}.json"

        run_card = {
            "batch_id": batch_id,
            "start_time": start_time,
            "end_time": end_time,
            "config_fingerprint": self.compute_config_fingerprint(),
            "backend_selection": {
                "requested": self._backend_metadata.requested_backend or "auto",
                "resolved": self._backend_metadata.resolved_backend,
                "device": self._backend_metadata.device,
                "model_id": self._backend_metadata.model_id,
            },
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
        }

        try:
            with open(run_card_path, "w") as f:
                json.dump(run_card, f, indent=2)
            logger.info(f"✅ Run card emitted: {run_card_path}")
        except Exception as e:
            logger.warning(f"Failed to emit run card: {e}")
