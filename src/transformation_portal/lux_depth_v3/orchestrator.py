"""
Orchestrator for depth + enhancement pipeline (V3+).

Two-stage pipeline:
1) Stage A: Depth (selected backend via registry) -> optional postprocessing -> write artifacts (+ optional PBR maps)
2) Stage B: Optional V2 enhancement subprocess (consumes depth artifacts when available)

This version is a forward-looking, correctness-first refinement that focuses on:
- Deterministic, collision-resistant output keying (structure-preserving + stronger hash suffix)
- Cache correctness (postprocessed depth is what gets cached + cache keys include backend + postprocess fingerprint)
- True “lazy preprocessing” (cache lookup can happen before image decode/preprocess)
- Robust skip logic (config fingerprint + pipeline/version + backend + postprocess fingerprints + defensive output checks)
- PBR regeneration safety (file existence + config fingerprinting, not just “files exist”)
- Better batch reporting ergonomics (root-level batch_report.json alias for compatibility)
- Cleaner parallel preprocessing (no duplicated work; plans are reused)

NOTE:
- This file assumes existing project modules/interfaces, and only adds metadata fields inside
  existing dict containers (e.g., depth.stats, pbr_assets) to avoid schema breakage.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional dependency for faster non-cryptographic hashing
try:
    import xxhash

    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    xxhash = None  # type: ignore

from .batch_stats import compute_batch_runtime_stats
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
from .security import HashMode, sanitize_file_stem, sanitize_path_component_nonlossy
from .v2_runner import V2Runner, find_v2_report

from ..depth.backends.registry import DepthBackendRegistry
from ..depth.backends.protocol import LicenseRestrictionError

logger = logging.getLogger(__name__)

# -----------------------------
# Versioning / forward-compat
# -----------------------------
ORCHESTRATOR_VERSION = "v3.1.0"

# Bump this when depth semantics change in a way that should invalidate reuse
# (e.g., cache stores *postprocessed* depth, normalization changes, etc.)
DEPTH_PIPELINE_VERSION = "depth-v3.1"

# Bump this when PBR generation semantics change materially
PBR_PIPELINE_VERSION = "pbr-v1.1"

# Output key hash defaults
DEFAULT_OUTPUT_KEY_HASH_LEN = 12  # 48-bit suffix (dramatically lower collision risk than 32-bit)
DEFAULT_OUTPUT_KEY_HASH_ALGO = "xxh3_64"  # if supported by xxhash build; otherwise xxh64


# -----------------------------
# Helpers: stable serialization & fingerprints
# -----------------------------
def _to_jsonable(obj: Any) -> Any:
    """Convert arbitrary objects into a JSON-serializable structure.

    Used ONLY for fingerprinting / metadata snapshots (not security).
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]

    if isinstance(obj, dict):
        # Sort keys for stability
        return {str(k): _to_jsonable(obj[k]) for k in sorted(obj.keys(), key=lambda x: str(x))}

    # Dataclasses
    if dataclasses.is_dataclass(obj):
        return _to_jsonable(dataclasses.asdict(obj))

    # Pydantic v2
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
        try:
            return _to_jsonable(obj.model_dump())
        except Exception:
            pass

    # Pydantic v1 / other "dict()"
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
        try:
            return _to_jsonable(obj.dict())
        except Exception:
            pass

    # Fallback to __dict__
    if hasattr(obj, "__dict__"):
        try:
            return _to_jsonable(vars(obj))
        except Exception:
            pass

    return str(obj)


def _stable_json_dumps(obj: Any) -> str:
    return json.dumps(_to_jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fingerprint_sha256(obj: Any) -> str:
    """Stable SHA-256 fingerprint of an object (for cache keys / skip logic)."""
    return _sha256_hex(_stable_json_dumps(obj).encode("utf-8"))


def _safe_sha1_hex(data: bytes) -> str:
    """SHA1 for naming only (not security), with compatibility fallback."""
    try:
        return hashlib.sha1(data, usedforsecurity=False).hexdigest()
    except TypeError:
        # Python < 3.9 compatibility
        return hashlib.sha1(data).hexdigest()


def _find_repo_root(start: Path) -> Path:
    """Find repository root by walking upward for .git / pyproject.toml / setup.cfg."""
    for p in [start, *start.parents]:
        if (p / ".git").exists():
            return p
        if (p / "pyproject.toml").exists():
            return p
        if (p / "setup.cfg").exists():
            return p
    # Fallback: original behavior-ish
    return start.parents[3] if len(start.parents) >= 4 else start


# -----------------------------
# Manifests: cached loading
# -----------------------------
@lru_cache(maxsize=128)
def _load_manifest_cached(manifest_path: str, mtime_ns: int, size: int) -> CombinedManifest:
    """Cache manifests by (path, mtime_ns, size) for safe invalidation."""
    return CombinedManifest.load(Path(manifest_path))


# -----------------------------
# Output keying
# -----------------------------
def make_output_key(
    input_path: Path,
    input_root: Path,
    use_xxhash: bool = XXHASH_AVAILABLE,
    *,
    hash_len: int = DEFAULT_OUTPUT_KEY_HASH_LEN,
    hash_algo: str = DEFAULT_OUTPUT_KEY_HASH_ALGO,
) -> Path:
    """Generate a structure-preserving output key.

    Properties:
    - Preserves directory structure relative to input_root
    - Incorporates sanitized extension label (no dot)
    - Appends hash suffix of the relative path (to prevent same-name collisions)

    Hash suffix:
    - Defaults to xxHash if available (fast), falls back to SHA-1 otherwise.
    - Uses a longer default suffix (12 hex chars) to reduce collision probability at scale.

    Example:
        Input: photos/scene1/image.JPG with root=photos/
        Output: scene1/image_jpg_1a2b3c4d5e6f
    """
    try:
        relpath = input_path.relative_to(input_root)
        hash_basis = relpath.as_posix()
    except ValueError:
        logger.warning("%s is not relative to %s; using non-structural fallback keying", input_path, input_root)
        relpath = Path(input_path.name)
        # Still hash something unique-ish: absolute path string (best effort)
        hash_basis = str(input_path.resolve())

    rel_dir = relpath.parent
    name = relpath.stem
    ext = relpath.suffix

    sanitized_parts = [sanitize_path_component_nonlossy(p) for p in rel_dir.parts]

    ext_label = ext.lstrip(".").lower() if ext else "noext"
    ext_label = sanitize_path_component_nonlossy(ext_label)

    stem_sanitized = sanitize_path_component_nonlossy(name)

    hash_input = hash_basis.encode("utf-8", errors="replace")

    # Prefer xxhash if enabled + available
    if use_xxhash and XXHASH_AVAILABLE:
        # Prefer XXH3 if present, else XXH64
        if hash_algo.lower() in ("xxh3_64", "xxh3") and hasattr(xxhash, "xxh3_64"):
            digest = xxhash.xxh3_64(hash_input).hexdigest()
        else:
            digest = xxhash.xxh64(hash_input).hexdigest()
        hash_suffix = digest[: max(8, int(hash_len))]
    else:
        digest = _safe_sha1_hex(hash_input)
        hash_suffix = digest[: max(8, int(hash_len))]

    key_name = f"{stem_sanitized}_{ext_label}_{hash_suffix}"

    if sanitized_parts:
        return Path(*sanitized_parts) / key_name
    return Path(key_name)


# -----------------------------
# Internal planning (avoid duplicate work)
# -----------------------------
@dataclasses.dataclass(frozen=True)
class ProcessingPlan:
    image_input: ImageInput
    input_root: Optional[Path]
    output_key: Path
    depth_path: Path
    float_depth_path: Path
    manifest_path: Path
    v2_log_path: Path


class EnhanceOrchestrator:
    """Orchestrates depth generation + optional V2 enhancement + optional PBR generation."""

    def __init__(self, config: EnhanceConfig, output_root: Path, verify_outputs: bool = True):
        self.config = config
        self.output_root = Path(output_root)
        self.verify_outputs = verify_outputs

        if config.hash_mode == HashMode.NEVER:
            logger.warning("Hash mode set to 'never' - manifests will lack integrity verification.")

        # Output dirs
        self.depth_dir = self.output_root / "depth"
        self.v2_dir = self.output_root / "v2"
        self.pbr_dir = self.output_root / "pbr"
        self.manifests_dir = self.output_root / "manifests"
        self.logs_dir = self.output_root / "logs"
        self.zones_dir = self.output_root / "zones"

        for d in [self.depth_dir, self.v2_dir, self.pbr_dir, self.manifests_dir, self.logs_dir, self.zones_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Output key behavior (configurable without breaking old configs)
        self._use_xxhash_keys = getattr(self.config, "use_xxhash", XXHASH_AVAILABLE)
        self._output_key_hash_len = int(getattr(self.config, "output_key_hash_len", DEFAULT_OUTPUT_KEY_HASH_LEN))
        self._output_key_hash_algo = str(getattr(self.config, "output_key_hash_algo", DEFAULT_OUTPUT_KEY_HASH_ALGO))

        # V3 depth config (user override > preset > default)
        if config.preset is not None:
            da3_config = DA3Config.from_preset(config.preset)
            if config.model_variant is not None:
                logger.info("Overriding preset model with user choice: %s", config.model_variant.value.display_name)
                da3_config.model_variant = config.model_variant
            else:
                config.model_variant = da3_config.model_variant
        else:
            model = config.model_variant if config.model_variant is not None else ModelVariant.METRIC_LARGE
            da3_config = DA3Config(model_variant=model)
            config.model_variant = model

        da3_config.device.device = config.depth_device

        # Backend selection (ADR-019)
        self._initialize_depth_backend()

        # Postprocessor (ensure preset params apply)
        self.postprocessor = Postprocessor(da3_config.postprocessing)

        # Fingerprints that should invalidate reuse even if ConfigFingerprint misses fields
        self._postprocess_fp = _fingerprint_sha256(da3_config.postprocessing)

        # Depth cache (opt-in)
        self.depth_cache: Optional[DepthCache] = (
            DepthCache(self.output_root, max_size_gb=config.depth_cache_max_size_gb) if config.enable_depth_cache else None
        )
        if self.depth_cache:
            logger.info("Depth cache enabled: %s", self.depth_cache.cache_dir)

        # Precompute “depth cache config key” that includes backend + postprocessing + pipeline version
        self._depth_cache_config_key = self._compute_depth_cache_config_key()

        # V2 runner setup (fail-fast)
        if config.enable_v2 and config.v2_preset is not None:
            self.v2_runner = V2Runner()
            if not self.v2_runner.script_path.exists():
                raise FileNotFoundError(
                    f"V2 enhancement script not found: {self.v2_runner.script_path}\n"
                    f"Required location: scripts/enhance_image.py in repository root\n\n"
                    f"Options:\n"
                    f"  1) Create the V2 enhancement script at the expected location\n"
                    f"  2) Set enable_v2=False for depth+PBR-only workflows\n"
                    f"  3) Set v2_preset=None to skip V2 stage\n"
                )
            logger.info("V2 enhancement enabled with script: %s", self.v2_runner.script_path)
        else:
            self.v2_runner = None
            logger.info("V2 enhancement disabled (PBR-only mode)")

        # Environment + revisions
        repo_root = _find_repo_root(Path(__file__).resolve())
        git_rev = get_git_revision(repo_root)
        self.v3_git = git_rev
        self.v2_git = git_rev
        self.environment = capture_environment()

        # Parallelization knobs
        self.max_workers = int(config.max_parallel_workers or max(1, cpu_count() - 1))
        self._use_parallel = bool(config.enable_parallel_processing)
        logger.debug(
            "Parallel processing: %s (workers=%d)", "enabled" if self._use_parallel else "disabled", self.max_workers
        )

    # -----------------------------
    # Backend initialization
    # -----------------------------
    def _initialize_depth_backend(self) -> None:
        requested = self.config.depth_backend or "da3"
        registry = DepthBackendRegistry()

        try:
            backend = registry.get_backend(requested, self.config)

            try:
                backend.ensure_available()
                available = True
                reason = f"{backend.name} backend ready"
            except (ImportError, FileNotFoundError) as e:
                available = False
                reason = str(e)

            if not available:
                logger.warning("Backend '%s' unavailable: %s. Falling back to DA3.", requested, reason)
                backend = registry.get_backend("da3", self.config)
                try:
                    backend.ensure_available()
                    resolved = "da3"
                    status = "fallback"
                except (ImportError, FileNotFoundError) as fallback_error:
                    # Test environment: no ML deps
                    logger.warning("DA3 fallback also unavailable: %s. Using mock backend for testing.", fallback_error)
                    from unittest.mock import Mock

                    backend = Mock()
                    backend.name = "mock"
                    backend.compute = Mock(side_effect=ImportError(
                        "Mock backend used - ML dependencies not installed. "
                        "This orchestrator cannot process images."
                    ))
                    resolved = "mock"
                    status = "test_mode"
                    reason = f"Test environment (no ML dependencies): {fallback_error}"
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

            logger.info("Depth backend: requested=%s resolved=%s device=%s", requested, resolved, self.config.depth_device)

        except LicenseRestrictionError as e:
            logger.error("License restriction: %s", e)
            raise
        except Exception as e:
            logger.error("Backend initialization failed: %s", e)
            raise

    def _capture_backend_metadata(self) -> BackendSelectionMetadata:
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

    # -----------------------------
    # Fingerprints / cache keys
    # -----------------------------
    def compute_config_fingerprint(self) -> ConfigFingerprint:
        # Keep existing schema stable; encode extra invalidators separately (depth.stats/pbr_assets).
        return ConfigFingerprint(
            model_variant=self.config.model_variant.value.name,
            depth_quantization=self.config.depth_quantization,
            depth_device=self.config.depth_device,
            preset=self.config.preset.value if self.config.preset else None,
            v2_preset=self.config.v2_preset,
            v2_device=self.config.v2_device,
            v2_upscaler_backend=self.config.v2_upscaler_backend,
        )

    def _compute_depth_cache_config_key(self) -> str:
        """Key that prevents incorrect cache reuse across backend/postprocessing changes."""
        fp = self.compute_config_fingerprint().depth_only().to_sha256()
        backend = self._backend_metadata.resolved_backend
        # Include pipeline version so semantic changes invalidate cache
        payload = {
            "depth_only_fp": fp,
            "resolved_backend": backend,
            "postprocess_fp": self._postprocess_fp,
            "depth_pipeline_version": DEPTH_PIPELINE_VERSION,
        }
        return _fingerprint_sha256(payload)

    def _compute_pbr_fingerprint(self) -> str:
        """Fingerprint PBR config so we can safely decide when to regenerate PBR outputs."""
        pbr_config = self.config.to_pbr_config()
        payload = {
            "pbr_config": _to_jsonable(pbr_config),
            "pbr_pipeline_version": PBR_PIPELINE_VERSION,
        }
        return _fingerprint_sha256(payload)

    # -----------------------------
    # Hash computation (manifest integrity)
    # -----------------------------
    def _compute_or_skip_hash(
        self,
        image_path: Path,
        manifest_exists: bool = False,
        saved_hash: Optional[str] = None,
        *,
        for_manifest_write: bool = False,
    ) -> Optional[str]:
        if self.config.hash_mode == HashMode.NEVER:
            return None

        if self.config.hash_mode == HashMode.IF_MANIFEST_EXISTS:
            if not for_manifest_write:
                if not manifest_exists or not saved_hash:
                    return None

        try:
            return compute_file_sha256(image_path)
        except Exception as e:
            logger.error("Hash computation failed for %s: %s", image_path, e)
            raise IOError(f"Hash computation failed: {e}") from e

    # -----------------------------
    # Skip logic: depth
    # -----------------------------
    def should_skip_depth(self, depth_path: Path, manifest_path: Path, image_input: ImageInput) -> bool:
        if not depth_path.exists() or not manifest_path.exists():
            return False

        try:
            # Load manifest (cached)
            if self.config.enable_manifest_cache:
                st = os.stat(manifest_path)
                manifest = _load_manifest_cached(str(manifest_path), st.st_mtime_ns, st.st_size)
            else:
                manifest = CombinedManifest.load(manifest_path)

            # Sanity: manifest's recorded depth path should match expected depth_path
            if manifest.depth and getattr(manifest.depth, "depth_path", None):
                if str(depth_path) != str(manifest.depth.depth_path):
                    logger.debug("Manifest depth_path mismatch; regenerating. expected=%s manifest=%s",
                                 depth_path, manifest.depth.depth_path)
                    return False

            # Input integrity check (if baseline exists)
            saved_hash = manifest.input.image_sha256 if manifest.input else None
            if saved_hash and self.config.hash_mode != HashMode.NEVER:
                current_hash = self._compute_or_skip_hash(
                    image_input.path, manifest_exists=True, saved_hash=saved_hash, for_manifest_write=False
                )
                if current_hash and current_hash != saved_hash:
                    logger.info("Input image changed - regenerating depth: %s", image_input.path)
                    return False

            # Config fingerprint check (existing behavior)
            if not manifest.config_fingerprint:
                logger.debug("No config fingerprint in manifest - regenerating depth")
                return False

            current_fp = self.compute_config_fingerprint()
            stored_fp = manifest.config_fingerprint

            if current_fp.depth_only().to_sha256() != stored_fp.depth_only().to_sha256():
                logger.info("Depth config changed - regenerating")
                return False

            # Extra invalidators (forward-looking correctness)
            if not manifest.depth:
                return False

            depth_stats = getattr(manifest.depth, "stats", None) or {}
            if depth_stats.get("depth_pipeline_version") != DEPTH_PIPELINE_VERSION:
                logger.info("Depth pipeline version changed - regenerating")
                return False

            if depth_stats.get("resolved_backend") and depth_stats.get("resolved_backend") != self._backend_metadata.resolved_backend:
                logger.info("Depth backend changed - regenerating (stored=%s current=%s)",
                            depth_stats.get("resolved_backend"), self._backend_metadata.resolved_backend)
                return False

            if depth_stats.get("postprocess_fp") and depth_stats.get("postprocess_fp") != self._postprocess_fp:
                logger.info("Postprocessing config changed - regenerating")
                return False

            # Defensive output existence / quick integrity
            if self.verify_outputs:
                if not depth_path.exists():
                    logger.debug("Depth file missing on disk: %s", depth_path)
                    return False

                from .depth_writer import read_depth_u16_png

                d = read_depth_u16_png(depth_path)
                if getattr(d, "ndim", None) != 2:
                    logger.debug("Depth file has invalid dimensions: %s", getattr(d, "ndim", None))
                    return False

            logger.debug("Resuming with existing depth: %s", depth_path)
            return True

        except Exception as e:
            logger.debug("Skip check failed: %s", e)
            return False

    # -----------------------------
    # Skip logic: V2
    # -----------------------------
    def should_skip_v2(
        self,
        v2_report_path: Optional[Path],
        manifest_path: Path,
        image_input: ImageInput,
        depth_was_skipped: bool,
    ) -> bool:
        if not v2_report_path or not v2_report_path.exists() or not manifest_path.exists():
            return False

        try:
            if self.config.enable_manifest_cache:
                st = os.stat(manifest_path)
                manifest = _load_manifest_cached(str(manifest_path), st.st_mtime_ns, st.st_size)
            else:
                manifest = CombinedManifest.load(manifest_path)

            if not manifest.config_fingerprint:
                logger.debug("No config fingerprint in manifest - regenerating V2")
                return False

            current_fp = self.compute_config_fingerprint()
            stored_fp = manifest.config_fingerprint

            if current_fp.v2_only().to_sha256() != stored_fp.v2_only().to_sha256():
                logger.info("V2 config changed - regenerating")
                return False

            # If depth was recomputed this run, V2 must rerun for consistency
            if not depth_was_skipped:
                logger.info("Depth was regenerated - V2 must rerun")
                return False

            if not manifest.v2 or manifest.v2.status != "ok":
                return False

            if self.verify_outputs:
                if not v2_report_path.exists():
                    logger.debug("V2 report missing: %s", v2_report_path)
                    return False

                # If manifest references PBR paths, ensure they still exist
                if manifest.pbr_assets:
                    for label, filepath in manifest.pbr_assets.items():
                        if isinstance(filepath, str) and filepath and label.endswith("_path"):
                            if not os.path.exists(filepath):
                                logger.debug("PBR output missing: %s", filepath)
                                return False

            return True

        except Exception as e:
            logger.debug("V2 skip check failed: %s", e)
            return False

    # -----------------------------
    # Plan building (paths + skip flags)
    # -----------------------------
    def _build_plan(self, image_input: ImageInput, input_root: Optional[Path]) -> ProcessingPlan:
        if input_root:
            output_key = make_output_key(
                image_input.path,
                input_root,
                use_xxhash=self._use_xxhash_keys,
                hash_len=self._output_key_hash_len,
                hash_algo=self._output_key_hash_algo,
            )
        else:
            output_key = Path(sanitize_file_stem(image_input.path.stem))

        depth_path = self.depth_dir / output_key.parent / f"{output_key.name}_depth.png"
        float_depth_path = self.depth_dir / output_key.parent / f"{output_key.name}_depth.npy"
        manifest_path = self.manifests_dir / output_key.parent / f"{output_key.name}_combined.json"
        v2_log_path = self.logs_dir / output_key.parent / f"v2_{output_key.name}.log"

        # Ensure dirs exist
        for p in [depth_path, manifest_path, v2_log_path]:
            p.parent.mkdir(parents=True, exist_ok=True)

        return ProcessingPlan(
            image_input=image_input,
            input_root=input_root,
            output_key=output_key,
            depth_path=depth_path,
            float_depth_path=float_depth_path,
            manifest_path=manifest_path,
            v2_log_path=v2_log_path,
        )

    # -----------------------------
    # Stage A: depth + optional PBR
    # -----------------------------
    def _compute_depth_stage(
        self,
        plan: ProcessingPlan,
        *,
        skip_depth: bool,
    ) -> Tuple[Optional[Any], float, Optional[dict]]:
        depth_runtime_s = 0.0
        depth_metadata = None
        pbr_assets = None

        output_key = plan.output_key
        depth_path = plan.depth_path
        float_depth_path = plan.float_depth_path
        manifest_path = plan.manifest_path
        image_input = plan.image_input

        # If skipping, load previous manifest metadata and optionally repair PBR
        if skip_depth:
            if manifest_path.exists():
                try:
                    m = CombinedManifest.load(manifest_path)
                    depth_metadata = m.depth
                    pbr_assets = getattr(m, "pbr_assets", None)
                except Exception as e:
                    logger.debug("Failed to load previous manifest metadata: %s", e)

            # If float depth missing but depth png exists, we can reconstruct a float approximation
            if getattr(self.config, "save_float_depth", False) and (not float_depth_path.exists()) and depth_path.exists():
                try:
                    depth_recon = self._load_cached_depth(depth_path, float_depth_path)
                    if depth_recon is not None:
                        np.save(str(float_depth_path), np.asarray(depth_recon, dtype=np.float32))
                        logger.debug("Reconstructed float depth from PNG: %s", float_depth_path)
                except Exception as e:
                    logger.debug("Float depth reconstruction failed (non-blocking): %s", e)

            # PBR: regenerate if missing or invalid or config changed
            if self.config.generate_pbr:
                current_pbr_fp = self._compute_pbr_fingerprint()
                if self._pbr_needs_regen(pbr_assets, current_pbr_fp):
                    logger.info("Generating PBR maps from cached depth (skip_depth=True)...")
                    depth_for_pbr = self._load_cached_depth(depth_path, float_depth_path)
                    if depth_for_pbr is not None:
                        pbr_assets = self._generate_pbr_stage(depth_for_pbr, output_key, current_pbr_fp)
            return depth_metadata, depth_runtime_s, pbr_assets

        # Not skipping: compute depth (and optionally use content-addressable cache)
        logger.info("Stage A: Depth for %s...", output_key)
        t0 = time.time()

        depth_float: Optional[np.ndarray] = None
        cached_depth_hit = False
        image_sha_for_cache: Optional[str] = None

        try:
            # Depth cache lookup BEFORE image decode/preprocess (true laziness)
            if self.depth_cache is not None:
                try:
                    image_sha_for_cache = compute_file_sha256(image_input.path)
                    cached = self.depth_cache.get(image_sha_for_cache, self._depth_cache_config_key)
                    if cached is not None:
                        depth_float = np.asarray(cached, dtype=np.float32)
                        cached_depth_hit = True
                        logger.info("Depth cache hit for %s (postprocessed cache).", output_key)
                except Exception as cache_e:
                    logger.debug("Depth cache lookup failed (non-fatal): %s", cache_e)

            if depth_float is None:
                # Lazy preprocessing only when cache miss
                from .preprocessing import preprocess_image, validate_image_format
                from PIL import Image

                validated_path = validate_image_format(image_input.path)
                preprocessed_array, _original_shape = preprocess_image(validated_path)

                pil_image = Image.fromarray(preprocessed_array.astype(np.uint8))
                result = self.depth_backend.compute(pil_image)

                # Postprocessing is part of the depth product; apply it on cache miss
                result = self.postprocessor.process(result)

                # Ensure float32 depth map
                depth_float = np.asarray(getattr(result, "depth", getattr(result, "depth_map", None)), dtype=np.float32)
                if depth_float.ndim != 2:
                    raise ValueError(f"Backend returned non-2D depth map: ndim={depth_float.ndim}")

                # Store POSTPROCESSED depth in cache (correctness!)
                if self.depth_cache is not None:
                    if image_sha_for_cache is None:
                        image_sha_for_cache = compute_file_sha256(image_input.path)
                    try:
                        self.depth_cache.store(image_sha_for_cache, self._depth_cache_config_key, depth_float)
                    except Exception as store_e:
                        logger.debug("Depth cache store failed (non-fatal): %s", store_e)

                # Capture inference provenance if present
                result_metadata = getattr(result, "metadata", None) or {}
            else:
                result_metadata = {"cached": True}

            depth_runtime_s = time.time() - t0

            # Write quantized depth PNG
            _, _, depth_stats = atomic_write_depth_u16_png_with_stats(
                depth_path,
                depth_float,
                method=self.config.depth_quantization,
                debug_verify=self.config.verify_depth_writes,
            )

            # Save float depth (optional)
            if getattr(self.config, "save_float_depth", False):
                try:
                    np.save(str(float_depth_path), np.asarray(depth_float, dtype=np.float32))
                    logger.debug("Saved float depth: %s", float_depth_path)
                except Exception as e:
                    logger.warning("Failed to save float depth (non-blocking): %s", e)

            # Build stats metadata
            backend_meta = self._backend_metadata
            stats: Dict[str, Any] = {
                "orchestrator_version": ORCHESTRATOR_VERSION,
                "depth_pipeline_version": DEPTH_PIPELINE_VERSION,
                "resolved_backend": backend_meta.resolved_backend,
                "requested_backend": backend_meta.requested_backend,
                "backend_resolution_status": backend_meta.resolution_status,
                "backend_resolution_reason": backend_meta.resolution_reason,
                "model_id": backend_meta.model_id,
                "device": backend_meta.device,
                "postprocess_fp": self._postprocess_fp,
                "cached": cached_depth_hit,
                "dtype": "uint16",
                "shape": list(depth_float.shape[:2]),
                "representation": "depth",
                "convention": "higher_is_farther",
                "unit": "relative",
            }

            # Merge backend-provided metadata (if any)
            for k in ("license", "requested_model_id", "resolved_model_id", "resolved_model_source"):
                if k in result_metadata:
                    stats[k] = result_metadata[k]

            if getattr(self.config, "save_float_depth", False):
                stats["float_depth_path"] = str(float_depth_path)
                stats["float_dtype"] = "float32"

            depth_metadata = DepthMetadata(
                model=self.config.model_variant.value.name,
                depth_path=str(depth_path),
                runtime_seconds=depth_runtime_s,
                scaling=depth_stats._asdict(),
                stats=stats,
            )

            # Write depth metadata JSON
            depth_metadata_path = depth_path.parent / f"{depth_path.stem}_metadata.json"
            with open(depth_metadata_path, "w", encoding="utf-8") as f:
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
            logger.debug("Wrote depth metadata: %s", depth_metadata_path)

            # PBR generation (optional; correctness gated by fingerprint)
            if self.config.generate_pbr:
                current_pbr_fp = self._compute_pbr_fingerprint()
                pbr_assets = self._generate_pbr_stage(depth_float, output_key, current_pbr_fp)

            return depth_metadata, depth_runtime_s, pbr_assets

        except Exception as e:
            logger.error("Depth failed for %s: %s", output_key, e)
            if self.config.depth_fallback == "fail":
                raise
            if self.config.depth_fallback == "skip":
                return None, 0.0, None
            if self.config.depth_fallback == "v2-auto":
                logger.info("V2 fallback mode: V3 failed, will attempt V2 with independent depth")
                if depth_path.exists():
                    try:
                        depth_path.unlink()
                    except Exception:
                        pass
                return None, 0.0, None
            raise ValueError(f"Unsupported depth_fallback mode: {self.config.depth_fallback}") from e

    # -----------------------------
    # PBR helpers
    # -----------------------------
    def _pbr_needs_regen(self, pbr_assets: Optional[Dict[str, Any]], current_pbr_fp: str) -> bool:
        if not self.config.generate_pbr:
            return False
        if not pbr_assets:
            return True
        if not self._verify_pbr_outputs(pbr_assets):
            return True
        if pbr_assets.get("fingerprint_sha256") != current_pbr_fp:
            return True
        if pbr_assets.get("pbr_pipeline_version") != PBR_PIPELINE_VERSION:
            return True
        return False

    def _generate_pbr_stage(self, depth: Any, output_key: Path, pbr_fp: str) -> Optional[dict]:
        try:
            logger.info("Generating PBR maps...")
            pbr_t0 = time.time()

            pbr_config = self.config.to_pbr_config()

            # Generate maps from depth (current implementation)
            normal_map, roughness_map, ao_map = generate_pbr_maps(depth, config=pbr_config)

            self.pbr_dir.mkdir(parents=True, exist_ok=True)

            base_name = output_key.name  # output_key has no suffix; name is stable

            pbr_paths = write_pbr_maps(
                normal_map=normal_map,
                roughness_map=roughness_map,
                ao_map=ao_map,
                output_dir=self.pbr_dir,
                base_name=base_name,
            )

            pbr_runtime = time.time() - pbr_t0
            logger.info("PBR maps generated in %.2fs: %s", pbr_runtime, list(pbr_paths.keys()))

            return {
                "normal_path": str(pbr_paths["normal"]),
                "roughness_path": str(pbr_paths["roughness"]),
                "ao_path": str(pbr_paths["ao"]),
                "runtime_seconds": pbr_runtime,
                "pbr_pipeline_version": PBR_PIPELINE_VERSION,
                "fingerprint_sha256": pbr_fp,
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

        except Exception as pbr_error:
            logger.warning("PBR generation failed (non-blocking): %s", pbr_error)
            return None

    def _verify_pbr_outputs(self, pbr_assets: Optional[Dict[str, Any]]) -> bool:
        if not pbr_assets:
            return False
        for key, value in pbr_assets.items():
            if isinstance(value, str) and key.endswith("_path"):
                if not os.path.exists(value):
                    logger.debug("PBR output missing: %s", value)
                    return False
        return True

    def _load_cached_depth(self, depth_path: Path, float_depth_path: Path) -> Optional[np.ndarray]:
        # Prefer float depth for PBR quality
        if float_depth_path.exists():
            try:
                depth_data = np.load(str(float_depth_path))
                return np.asarray(depth_data, dtype=np.float32)
            except Exception as e:
                logger.warning("Failed to load float depth: %s", e)

        # Fall back to quantized depth image
        if depth_path.exists():
            try:
                from .depth_writer import read_depth_u16_png

                depth_data = read_depth_u16_png(depth_path)
                depth_data = np.asarray(depth_data)

                if depth_data.dtype == np.uint16:
                    depth_data = depth_data.astype(np.float32) / 65535.0
                else:
                    depth_data = depth_data.astype(np.float32, copy=False)
                    maxv = float(np.nanmax(depth_data)) if depth_data.size else 0.0
                    if maxv > 1.5:
                        depth_data /= 65535.0

                return depth_data
            except Exception as e:
                logger.warning("Failed to load depth image: %s", e)

        return None

    # -----------------------------
    # Stage B: V2
    # -----------------------------
    def _run_v2_stage(
        self,
        plan: ProcessingPlan,
        *,
        depth_path_for_v2: Optional[Path],
        skip_depth: bool,
    ) -> Tuple[dict, float, Optional[Path]]:
        if self.v2_runner is None or not self.config.enable_v2:
            logger.info("V2 stage disabled, skipping enhancement")
            return {"status": "skipped"}, 0.0, None

        output_key = plan.output_key
        v2_report_path = find_v2_report(self.v2_dir, output_key.name)

        skip_v2 = (
            (not self.config.force_v2)
            and self.should_skip_v2(v2_report_path, plan.manifest_path, plan.image_input, depth_was_skipped=skip_depth)
        )

        if skip_v2:
            logger.info("V2 outputs valid, skipping.")
            return {"status": "ok"}, 0.0, v2_report_path

        v2_result = self.v2_runner.run(
            input_path=plan.image_input.path,
            depth_dir=self.depth_dir if (depth_path_for_v2 and depth_path_for_v2.exists()) else None,
            output_dir=self.v2_dir,
            preset=self.config.v2_preset,
            device=self.config.v2_device,
            upscaler_backend=self.config.v2_upscaler_backend,
            log_file=plan.v2_log_path,
            timeout=self.config.v2_timeout,
        )

        v2_runtime_s = float(v2_result.get("runtime_s", 0.0))
        v2_report_path = find_v2_report(self.v2_dir, output_key.name)
        return v2_result, v2_runtime_s, v2_report_path

    # -----------------------------
    # Manifest writing
    # -----------------------------
    def _write_manifest(
        self,
        plan: ProcessingPlan,
        *,
        depth_metadata: Optional[Any],
        v2_result: dict,
        v2_report_path: Optional[Path],
        pbr_assets: Optional[dict],
        depth_runtime_s: float,
        v2_runtime_s: float,
        pipeline_start_time: float,
        pipeline_end_time: float,
    ) -> None:
        manifest_path = plan.manifest_path

        v2_metadata = V2Metadata(
            preset=self.config.v2_preset,
            strict_depth=depth_metadata is not None,
            output_dir="v2/",
            report_path=str(v2_report_path) if v2_report_path else "",
            status=v2_result.get("status", "unknown"),
            error_message=v2_result.get("error"),
        )

        # Load existing saved hash baseline if present
        manifest_exists = manifest_path.exists()
        saved_hash = None
        if manifest_exists:
            try:
                m_prev = CombinedManifest.load(manifest_path)
                if m_prev.input:
                    saved_hash = m_prev.input.image_sha256
            except Exception as e:
                logger.debug("Failed to load previous hash from manifest: %s", e)

        input_sha = self._compute_or_skip_hash(
            plan.image_input.path,
            manifest_exists=manifest_exists,
            saved_hash=saved_hash,
            for_manifest_write=True,
        )

        manifest = CombinedManifest(
            input=InputMetadata(
                image_path=str(plan.image_input.path),
                image_sha256=input_sha,
                image_size_bytes=None,
                image_dimensions=None,
            ),
            depth=depth_metadata,
            v2=v2_metadata,
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
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(pipeline_start_time)),
            end_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(pipeline_end_time)),
            backend_selection=self._backend_metadata,
        )
        manifest.write(manifest_path)

    # -----------------------------
    # Public: single image
    # -----------------------------
    def enhance_image(self, image_input: ImageInput, input_root: Optional[Path] = None) -> Dict[str, Any]:
        plan = self._build_plan(image_input, input_root)
        return self._enhance_with_plan(plan)

    def _enhance_with_plan(self, plan: ProcessingPlan) -> Dict[str, Any]:
        pipeline_start_time = time.time()

        logger.info("Processing %s...", plan.output_key)

        # Determine skip BEFORE any decode/preprocess
        skip_depth = (not self.config.force_depth) and self.should_skip_depth(plan.depth_path, plan.manifest_path, plan.image_input)

        # Stage A
        depth_metadata, depth_runtime_s, pbr_assets = self._compute_depth_stage(plan, skip_depth=skip_depth)

        # Early-exit behavior for depth_fallback=skip
        if depth_metadata is None and depth_runtime_s == 0.0 and pbr_assets is None:
            if self.config.depth_fallback == "skip":
                return {"status": "skipped", "reason": "Depth computation failed", "image": str(plan.image_input.path)}

        # Stage B
        v2_result, v2_runtime_s, v2_report_path = self._run_v2_stage(
            plan,
            depth_path_for_v2=(plan.depth_path if depth_metadata else None),
            skip_depth=skip_depth,
        )

        pipeline_end_time = time.time()

        # Write manifest
        self._write_manifest(
            plan,
            depth_metadata=depth_metadata,
            v2_result=v2_result,
            v2_report_path=v2_report_path,
            pbr_assets=pbr_assets,
            depth_runtime_s=depth_runtime_s,
            v2_runtime_s=v2_runtime_s,
            pipeline_start_time=pipeline_start_time,
            pipeline_end_time=pipeline_end_time,
        )

        return {
            "status": "ok",
            "image": str(plan.image_input.path),
            "output_key": str(plan.output_key),
            "depth_path": str(plan.depth_path) if depth_metadata else None,
            "float_depth_path": str(plan.float_depth_path) if getattr(self.config, "save_float_depth", False) else None,
            "pbr": pbr_assets,
            "v2_status": v2_result.get("status"),
            "manifest": str(plan.manifest_path),
            "runtime_s": pipeline_end_time - pipeline_start_time,
        }

    # -----------------------------
    # Parallel batch: plan build in parallel, execution sequential (GPU safety)
    # -----------------------------
    def _parallel_build_plans(self, image_inputs: List[ImageInput], input_root: Optional[Path]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = [None] * len(image_inputs)  # type: ignore

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map = {
                executor.submit(self._build_plan, img, input_root): idx for idx, img in enumerate(image_inputs)
            }
            for fut in as_completed(future_map):
                idx = future_map[fut]
                img = image_inputs[idx]
                try:
                    plan = fut.result()
                    # Precompute skip flag once here (reused later)
                    skip_depth = (not self.config.force_depth) and self.should_skip_depth(
                        plan.depth_path, plan.manifest_path, plan.image_input
                    )
                    results[idx] = {"status": "ok", "plan": plan, "skip_depth": skip_depth}
                except Exception as e:
                    logger.error("Planning failed for %s: %s", img.path, e)
                    results[idx] = {"status": "error", "image": str(img.path), "error": str(e)}

        return results

    def enhance_batch_parallel(self, image_inputs: List[ImageInput], input_root: Optional[Path] = None) -> List[Dict[str, Any]]:
        # For small batches, sequential is usually faster and simpler
        if (not self._use_parallel) or len(image_inputs) < 4:
            logger.debug("Using sequential processing (batch size: %d)", len(image_inputs))
            return [self.enhance_image(img, input_root) for img in image_inputs]

        logger.info("Parallel batch planning: %d images with %d workers", len(image_inputs), self.max_workers)

        planned = self._parallel_build_plans(image_inputs, input_root)

        # Execution remains sequential to avoid GPU contention
        results: List[Dict[str, Any]] = []
        for item in planned:
            if item["status"] != "ok":
                results.append(item)
                continue
            plan: ProcessingPlan = item["plan"]
            skip_depth: bool = item["skip_depth"]
            try:
                # Use the plan; do not rebuild paths/keys/dirs
                pipeline_start_time = time.time()
                depth_metadata, depth_runtime_s, pbr_assets = self._compute_depth_stage(plan, skip_depth=skip_depth)

                if depth_metadata is None and depth_runtime_s == 0.0 and pbr_assets is None and self.config.depth_fallback == "skip":
                    results.append({"status": "skipped", "reason": "Depth computation failed", "image": str(plan.image_input.path)})
                    continue

                v2_result, v2_runtime_s, v2_report_path = self._run_v2_stage(
                    plan,
                    depth_path_for_v2=(plan.depth_path if depth_metadata else None),
                    skip_depth=skip_depth,
                )
                pipeline_end_time = time.time()

                self._write_manifest(
                    plan,
                    depth_metadata=depth_metadata,
                    v2_result=v2_result,
                    v2_report_path=v2_report_path,
                    pbr_assets=pbr_assets,
                    depth_runtime_s=depth_runtime_s,
                    v2_runtime_s=v2_runtime_s,
                    pipeline_start_time=pipeline_start_time,
                    pipeline_end_time=pipeline_end_time,
                )

                results.append(
                    {
                        "status": "ok",
                        "image": str(plan.image_input.path),
                        "output_key": str(plan.output_key),
                        "depth_path": str(plan.depth_path) if depth_metadata else None,
                        "manifest": str(plan.manifest_path),
                        "runtime_s": pipeline_end_time - pipeline_start_time,
                    }
                )
            except Exception as e:
                logger.error("Enhancement failed for %s: %s", plan.image_input.path, e)
                results.append({"status": "error", "image": str(plan.image_input.path), "error": str(e)})

        return results

    # -----------------------------
    # Public: batch
    # -----------------------------
    def enhance_batch(self, input_dir: Path, image_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if image_extensions is None:
            image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

        batch_start_time = time.time()
        batch_start_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(batch_start_time))

        batch_id = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        logger.info("Batch %s: Scanning %s", batch_id, input_dir)

        backend_metadata = self._capture_backend_metadata()
        logger.info(
            "Backend selection: requested=%s resolved=%s status=%s device=%s model=%s",
            backend_metadata.requested_backend or "auto",
            backend_metadata.resolved_backend,
            backend_metadata.resolution_status,
            backend_metadata.device,
            backend_metadata.model_id,
        )
        self._backend_metadata = backend_metadata

        discovery_config = DiscoveryConfig(strict_mode=self.config.strict_inputs)
        images = discover_images(input_dir, discovery_config, image_extensions)

        image_inputs = [ImageInput(img) for img in sorted(images)]

        if self._use_parallel and len(image_inputs) >= 4:
            logger.info("Using parallel batch mode for %d images", len(image_inputs))
            results = self.enhance_batch_parallel(image_inputs, input_root=input_dir)
        else:
            results = []
            for img_input in image_inputs:
                try:
                    results.append(self.enhance_image(img_input, input_root=input_dir))
                except Exception as e:
                    logger.error("Failed %s: %s", img_input.path, e)
                    results.append({"status": "error", "image": str(img_input.path), "error": str(e)})

        batch_end_time = time.time()
        batch_end_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(batch_end_time))

        runtimes = [float(r.get("runtime_s", 0.0)) for r in results if r.get("status") == "ok"]
        runtime_stats = compute_batch_runtime_stats(runtimes)

        bm = BatchManifest(
            batch_id=batch_id,
            start_time=batch_start_utc,
            end_time=batch_end_utc,
            config={
                "model": self.config.model_variant.value.name,
                "orchestrator_version": ORCHESTRATOR_VERSION,
                "depth_pipeline_version": DEPTH_PIPELINE_VERSION,
                "pbr_pipeline_version": PBR_PIPELINE_VERSION,
                "depth_cache_enabled": bool(self.depth_cache),
                "resolved_backend": backend_metadata.resolved_backend,
            },
            results=results,
            stats={
                **runtime_stats,
                "total_images": len(results),
                "batch_runtime_seconds": batch_end_time - batch_start_time,
            },
        )

        # Canonical batch manifest under manifests/
        batch_path = self.manifests_dir / f"batch_{batch_id}.json"
        bm.write(batch_path)

        # Compatibility alias at output root (so scripts looking for batch_report.json keep working)
        # Atomic replace pattern
        alias_path = self.output_root / "batch_report.json"
        tmp_path = alias_path.with_suffix(".json.tmp")
        try:
            shutil.copy2(batch_path, tmp_path)
            os.replace(tmp_path, alias_path)
        except Exception as e:
            logger.debug("Failed to write batch_report.json alias (non-blocking): %s", e)
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass

        return results
