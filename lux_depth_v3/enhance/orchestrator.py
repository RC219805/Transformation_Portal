"""Orchestrator for V3 depth + V2 enhancement pipeline.

Two-stage pipeline:
1. Stage A (V3): Generate depth assets using DA3
2. Stage B (V2): Consume depth assets → weights → grade → upscale → export → report
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
import time
import logging

from lux_depth_v3.config import DA3Config, ModelVariant, Preset
from lux_depth_v3.inference import DA3InferenceEngine
from lux_depth_v3.input_manager import ImageInput
from .depth_writer import atomic_write_depth_u16_png_with_stats
from .v2_runner import V2Runner, find_v2_report
from .security import (
    HashMode,
    sanitize_file_stem,
    sanitize_path_component_nonlossy,
    validate_device_spec,
    validate_quantization_method,
    validate_depth_fallback,
)
from .manifest import (
    CombinedManifest,
    ConfigFingerprint,
    InputMetadata,
    DepthMetadata,
    V2Metadata,
    TimingMetadata,
    ReproMetadata,
    BatchManifest,
    compute_file_sha256,
    get_git_revision,
    capture_environment,
)
from .batch_stats import compute_batch_runtime_stats

logger = logging.getLogger(__name__)


def make_output_key(
    input_path: Path,
    input_root: Path,
) -> Path:
    """Generate collision-free output key preserving directory structure.

    Uses non-lossy sanitization: encodes invalid characters rather than
    dropping them, ensuring distinct inputs never collapse to same key.

    Args:
        input_path: Full path to input image
        input_root: Root directory of inputs

    Returns:
        Relative path suitable for output (without extension)

    Examples:
        renders/kitchen/view.jpg → kitchen/view
        renders/exterior/view.jpg → exterior/view
        renders/kitchen:1/view.jpg → kitchen%3A1/view  (encoded colon)

    Security:
        - Prevents path traversal (strips .., leading dots)
        - Limits component length (200 chars)
        - Uses deterministic hashing for reproducibility
    """
    try:
        relpath = input_path.relative_to(input_root)
    except ValueError:
        # If input_path is not relative to input_root, use flat naming
        logger.warning(f"{input_path} is not relative to {input_root}, using flat naming")
        relpath = Path(input_path.name)

    # Sanitize each path component independently
    sanitized_parts = []
    for part in relpath.parent.parts:
        sanitized = sanitize_path_component_nonlossy(part)
        sanitized_parts.append(sanitized)

    # Sanitize stem
    stem_sanitized = sanitize_path_component_nonlossy(relpath.stem)

    # Build output key
    if sanitized_parts:
        return Path(*sanitized_parts) / stem_sanitized
    else:
        return Path(stem_sanitized)


@dataclass
class EnhanceConfig:
    """Configuration for enhance orchestrator.

    Note on model_variant vs preset interaction:
    - If preset is specified and model_variant is None: use preset's model choice
    - If preset is specified and model_variant is set: override preset's model
    - If preset is None and model_variant is None: use METRIC_LARGE as default
    - If preset is None and model_variant is set: use specified model
    """

    # Model config
    model_variant: Optional[ModelVariant] = None  # None means "use preset's choice or default"
    preset: Optional[Preset] = None

    # V2 config
    v2_preset: str = "production_ultra"
    v2_device: str = "auto"
    v2_upscaler_backend: str = "torch"

    # Depth config
    depth_device: str = "auto"
    depth_quantization: str = "p1p99"
    depth_zones: str = "off"  # off|preview (diagnostics only; does not affect V2)

    # Execution control
    execution_mode: str = "sequential"  # "sequential" or "pipelined"
    depth_fallback: str = "fail"  # "fail", "skip", "v2-auto"
    force_depth: bool = False
    force_v2: bool = False

    # Verification
    verify_depth_writes: bool = False  # Set True for paranoid mode, False for production speed
    hash_mode: HashMode = HashMode.IF_MANIFEST_EXISTS  # Control hash computation timing

    # License
    non_commercial_ok: bool = False

    # Timeout
    v2_timeout: Optional[float] = 600.0  # 10 minutes default

    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate device specifications
        validate_device_spec(self.depth_device)
        validate_device_spec(self.v2_device)

        # Validate depth quantization method
        validate_quantization_method(self.depth_quantization)

        # Validate depth fallback policy
        validate_depth_fallback(self.depth_fallback)

        # Validate depth zones mode
        if self.depth_zones not in ("off", "preview"):
            raise ValueError(f"Invalid depth_zones: {self.depth_zones} (expected off|preview)")


class EnhanceOrchestrator:
    """Orchestrates V3 depth generation + V2 enhancement pipeline."""

    def __init__(self, config: EnhanceConfig, output_root: Path):
        """Initialize orchestrator.

        Args:
            config: Enhance configuration
            output_root: Root output directory
        """
        self.config = config
        self.output_root = Path(output_root)

        # FIX 4: Warn about security implications of "never" hash mode
        if config.hash_mode == HashMode.NEVER:
            logger.warning(
                "Hash mode set to 'never' - manifests will not include input file hashes. "
                "This provides no protection against input tampering and prevents cache validation. "
                "Use only in trusted environments where performance is critical."
            )

        # Create output directories
        self.depth_dir = self.output_root / "depth"
        self.v2_dir = self.output_root / "v2"
        self.manifests_dir = self.output_root / "manifests"
        self.logs_dir = self.output_root / "logs"
        self.zones_dir = self.output_root / "zones"

        for dir_path in [
            self.depth_dir,
            self.v2_dir,
            self.manifests_dir,
            self.logs_dir,
            self.zones_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize V3 inference engine
        # Handle preset vs explicit model_variant override logic
        if config.preset is not None:
            # Start from preset configuration
            da3_config = DA3Config.from_preset(config.preset)

            # Override model_variant ONLY if user explicitly provided one (not None)
            if config.model_variant is not None:
                preset_model = da3_config.model_variant
                logger.info(
                    f"Overriding preset '{config.preset.value}' model "
                    f"({preset_model.value.display_name}) with user choice "
                    f"({config.model_variant.value.display_name})"
                )
                da3_config.model_variant = config.model_variant
            else:
                # Use preset's model choice
                logger.info(f"Using preset '{config.preset.value}' model: {da3_config.model_variant.value.display_name}")
                # Update config to reflect actual model being used
                config.model_variant = da3_config.model_variant
        else:
            # No preset: use explicit model_variant or default to METRIC_LARGE
            model = config.model_variant if config.model_variant is not None else ModelVariant.METRIC_LARGE
            da3_config = DA3Config(
                model_variant=model,
            )
            # Update config to reflect actual model being used
            config.model_variant = model

        # Apply device override (always respect CLI device choice)
        da3_config.device.device = config.depth_device

        self.inference_engine = DA3InferenceEngine(
            config=da3_config,
            commercial_use=not config.non_commercial_ok,
            validate_license_strict=True,
        )

        # Initialize V2 runner
        self.v2_runner = V2Runner()

        # Track git revisions for reproducibility
        # Both V2 and V3 are in the same monorepo, so find the repo root
        # The .git directory is at Transformation_Portal-main/
        repo_root = Path(__file__).parent.parent.parent  # Transformation_Portal-main

        # Get git revision from repo root (works for monorepo structure)
        git_revision = get_git_revision(repo_root)

        # Use same revision for both V2 and V3 (they're in the same repo)
        self.v3_git = git_revision
        self.v2_git = git_revision

        # Capture environment once at initialization (cached)
        self.environment = capture_environment()
        logger.info(
            f"Environment: Python {self.environment.python}, "
            f"Torch {self.environment.torch or 'N/A'}, "
            f"OS {self.environment.os_platform or 'N/A'}"
        )

    def compute_config_fingerprint(self) -> ConfigFingerprint:
        """Compute fingerprint of current configuration.

        Returns:
            ConfigFingerprint object with all output-determining parameters
        """
        return ConfigFingerprint(
            model_variant=self.config.model_variant.value.name,  # Use name string, not ModelInfo
            depth_quantization=self.config.depth_quantization,
            depth_device=self.config.depth_device,
            preset=self.config.preset.value if self.config.preset else None,
            v2_preset=self.config.v2_preset,
            v2_device=self.config.v2_device,
            v2_upscaler_backend=self.config.v2_upscaler_backend,
        )

    def _compute_or_skip_hash(self, image_path: Path, manifest_path: Optional[Path] = None) -> Optional[str]:
        """Compute file hash according to hash_mode policy (FIX 5).

        Args:
            image_path: Path to input image
            manifest_path: Optional path to existing manifest (for IF_MANIFEST_EXISTS mode)

        Returns:
            SHA256 hash string if computed, None if skipped

        Raises:
            IOError: If hash computation is required but fails (FIX 2)

        Security notes:
            - NEVER mode skips hashing entirely (no integrity verification)
            - IF_MANIFEST_EXISTS computes hash on first run AND when manifest exists
            - ALWAYS mode computes unconditionally (maximum security)
            - Failure to compute hash when required raises exception (fail-fast)
        """
        # Determine if hash computation is needed
        should_compute = False

        if self.config.hash_mode == HashMode.ALWAYS:
            should_compute = True
        elif self.config.hash_mode == HashMode.IF_MANIFEST_EXISTS:
            # FIX Issue #5: Compute hash on FIRST run (manifest doesn't exist yet)
            # and on subsequent runs (manifest exists). This ensures manifests
            # always have hashes for integrity verification.
            should_compute = True
        elif self.config.hash_mode == HashMode.NEVER:
            should_compute = False
        else:
            # Should never reach here due to enum validation, but be defensive
            logger.error(f"Unknown hash_mode: {self.config.hash_mode}")
            should_compute = True  # Default to safe behavior

        if not should_compute:
            logger.debug(f"Skipping hash computation (mode={self.config.hash_mode.value})")
            return None

        # FIX 2: Fail-fast if hash is required but computation fails
        try:
            hash_value = compute_file_sha256(image_path)
            logger.debug(f"Computed hash for {image_path.name}: {hash_value[:16]}...")
            return hash_value
        except Exception as e:
            # Hash was required but failed - this is a critical error
            error_msg = (
                f"Hash computation failed for {image_path} (mode={self.config.hash_mode.value}). "
                f"Cannot create verifiable manifest. Error: {e}"
            )
            logger.error(error_msg)
            raise IOError(error_msg) from e

    def should_skip_depth(
        self,
        depth_path: Path,
        manifest_path: Path,
        image_input: ImageInput,
    ) -> bool:
        """Determine if depth generation can be safely skipped.

        Returns True only if:
        - Depth file exists and is valid uint16 PNG
        - Combined manifest exists
        - Input image hash matches manifest
        - Depth-related config matches (model, quantization, device, preset)
        - Previous depth run succeeded

        Args:
            depth_path: Path to depth PNG
            manifest_path: Path to combined manifest
            image_input: Input image metadata

        Returns:
            True if depth can be skipped, False if must regenerate
        """
        if not depth_path.exists():
            logger.debug("Depth file missing - will generate")
            return False

        if not manifest_path.exists():
            logger.warning(f"Depth exists but no manifest - regenerating for safety: {depth_path}")
            return False

        try:
            manifest = CombinedManifest.load(manifest_path)

            # Check input hash if available in manifest
            # Note: If manifest has hash but current hash_mode=NEVER, we skip validation
            # This allows graceful degradation when switching modes
            if manifest.input and manifest.input.image_sha256:
                # Manifest has hash - validate if hash_mode allows
                if self.config.hash_mode != HashMode.NEVER:
                    current_hash = self._compute_or_skip_hash(image_input.path, manifest_path=manifest_path)
                    if current_hash and current_hash != manifest.input.image_sha256:
                        logger.info(f"Input image changed - regenerating depth: {image_input.path}")
                        return False
                else:
                    logger.debug("Skipping hash validation (hash_mode=NEVER)")
            else:
                # Old manifest lacks hash - cannot validate input integrity
                if self.config.hash_mode == HashMode.ALWAYS:
                    logger.info("Old manifest lacks hash and hash_mode=ALWAYS - regenerating for security")
                    return False

            # Check config fingerprint (depth portion)
            if not manifest.config_fingerprint:
                logger.info("Old manifest lacks config fingerprint - regenerating")
                return False

            current_config_fp = self.compute_config_fingerprint()
            manifest_fp = ConfigFingerprint(
                model_variant=manifest.depth.model if manifest.depth else "",
                depth_quantization=(manifest.depth.scaling.get("method", "") if manifest.depth else ""),
                depth_device=self.config.depth_device,
                preset=self.config.preset.value if self.config.preset else None,
                v2_preset=manifest.v2.preset if manifest.v2 else "",
                v2_device=self.config.v2_device,
                v2_upscaler_backend=self.config.v2_upscaler_backend,
            )

            if current_config_fp.depth_only() != manifest_fp.depth_only():
                logger.info("Depth config changed - regenerating")
                return False

            # Check depth status
            if not manifest.depth:
                logger.warning("Previous depth run incomplete - regenerating")
                return False

            # Quick validation: verify depth file is readable uint16
            try:
                from .depth_writer import read_depth_u16_png

                depth_verify = read_depth_u16_png(depth_path)
                if depth_verify.ndim != 2:
                    logger.warning(f"Depth file has wrong dimensions - regenerating: {depth_path}")
                    return False
            except Exception as e:
                logger.warning(f"Depth file unreadable: {e} - regenerating")
                return False

            logger.debug(f"Resuming with existing depth: {depth_path}")
            return True

        except Exception as e:
            logger.warning(f"Manifest read failed: {e} - regenerating for safety")
            return False

    def should_skip_v2(
        self,
        v2_report_path: Optional[Path],
        manifest_path: Path,
        image_input: ImageInput,
        depth_was_skipped: bool,
    ) -> bool:
        """Determine if V2 enhancement can be safely skipped.

        Returns True only if:
        - V2 report exists
        - Combined manifest exists
        - Input image hash matches
        - V2-related config matches (preset, device, upscaler)
        - Depth status is consistent (if depth changed, V2 must rerun)
        - Previous V2 run succeeded

        Args:
            v2_report_path: Path to V2 report (if exists)
            manifest_path: Path to combined manifest
            image_input: Input image metadata
            depth_was_skipped: True if depth was skipped (reused)

        Returns:
            True if V2 can be skipped, False if must rerun
        """
        if not v2_report_path or not v2_report_path.exists():
            logger.debug("V2 report missing - will run V2")
            return False

        if not manifest_path.exists():
            logger.warning("V2 report exists but no manifest - rerunning for safety")
            return False

        try:
            manifest = CombinedManifest.load(manifest_path)

            # Check input hash if available in manifest
            # Same logic as should_skip_depth for consistency
            if manifest.input and manifest.input.image_sha256:
                # Manifest has hash - validate if hash_mode allows
                if self.config.hash_mode != HashMode.NEVER:
                    current_hash = self._compute_or_skip_hash(image_input.path, manifest_path=manifest_path)
                    if current_hash and current_hash != manifest.input.image_sha256:
                        logger.info("Input changed - rerunning V2")
                        return False
                else:
                    logger.debug("Skipping hash validation (hash_mode=NEVER)")
            else:
                # Old manifest lacks hash - cannot validate input integrity
                if self.config.hash_mode == HashMode.ALWAYS:
                    logger.info("Old manifest lacks hash and hash_mode=ALWAYS - rerunning for security")
                    return False

            # Check config fingerprint (V2 portion)
            if not manifest.config_fingerprint:
                logger.info("Old manifest lacks config fingerprint - rerunning V2")
                return False

            current_config_fp = self.compute_config_fingerprint()
            manifest_fp = ConfigFingerprint(
                model_variant=manifest.depth.model if manifest.depth else "",
                depth_quantization=(manifest.depth.scaling.get("method", "") if manifest.depth else ""),
                depth_device=self.config.depth_device,
                preset=self.config.preset.value if self.config.preset else None,
                v2_preset=manifest.v2.preset if manifest.v2 else "",
                v2_device=self.config.v2_device,
                v2_upscaler_backend=self.config.v2_upscaler_backend,
            )

            if current_config_fp.v2_only() != manifest_fp.v2_only():
                logger.info("V2 config changed - rerunning")
                return False

            # Check depth consistency: if depth was regenerated, V2 must rerun
            if not depth_was_skipped:
                logger.info("Depth was regenerated - V2 must rerun to use new depth")
                return False

            # Check V2 status
            if not manifest.v2 or manifest.v2.status != "ok":
                logger.warning("Previous V2 run incomplete - rerunning")
                return False

            logger.debug(f"Resuming with existing V2 outputs: {v2_report_path}")
            return True

        except Exception as e:
            logger.warning(f"Manifest check failed: {e} - rerunning V2 for safety")
            return False

    def enhance_image(
        self,
        image_input: ImageInput,
        input_root: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Process single image through V3 + V2 pipeline with EXIF pre-normalization.

        Args:
            image_input: Input image metadata
            input_root: Optional root directory for collision-free nested paths

        Returns:
            Dictionary with processing results and paths
        """
        # Generate collision-free output key
        if input_root:
            output_key = make_output_key(image_input.path, input_root)
        else:
            # Flat naming: just sanitize stem (backward compatible)
            raw_stem = image_input.path.stem
            stem = sanitize_file_stem(raw_stem)
            if stem != raw_stem:
                logger.warning(f"File stem sanitized: '{raw_stem}' -> '{stem}'")
            output_key = Path(stem)

        logger.info(f"Processing {output_key}...")

        # Build paths with nested structure
        depth_path = self.depth_dir / output_key.parent / f"{output_key.name}_depth.png"
        combined_manifest_path = self.manifests_dir / output_key.parent / f"{output_key.name}_combined.json"
        v2_log_path = self.logs_dir / output_key.parent / f"v2_{output_key.name}.log"

        # Ensure parent directories exist BEFORE any writes
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        combined_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        v2_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Pre-normalize EXIF orientation for PIL/OpenCV alignment
        from .preprocessing import (
            normalize_exif_orientation,
            validate_depth_image_alignment,
        )

        tmp_inputs_dir = self.output_root / "tmp_inputs"
        tmp_inputs_dir.mkdir(parents=True, exist_ok=True)
        normalized_path = tmp_inputs_dir / f"{output_key.name}_normalized.png"

        # Always normalize EXIF orientation for PIL/OpenCV alignment
        # (return value is always True now, so we don't store it)
        normalize_exif_orientation(image_input.path, normalized_path)

        # Use normalized file for both DA3 and V2
        normalized_input = ImageInput(path=normalized_path)

        # Check depth resume with config fingerprint validation
        skip_depth = not self.config.force_depth and self.should_skip_depth(depth_path, combined_manifest_path, image_input)

        # Stage A: Generate depth (using normalized input)
        depth_result = None
        depth_runtime_s = 0.0
        depth_metadata = None

        if not skip_depth:
            logger.info(f"Stage A: Generating depth for {output_key}...")
            start_time = time.time()

            try:
                # Use normalized input for depth estimation
                depth_result = self.inference_engine.predict(normalized_input)
                depth_runtime_s = time.time() - start_time

                # Write depth atomically with detailed statistics
                p1, p99, depth_stats = atomic_write_depth_u16_png_with_stats(
                    depth_path,
                    depth_result.depth,
                    method=self.config.depth_quantization,
                    debug_verify=self.config.verify_depth_writes,
                )

                # Create enhanced depth metadata with detailed scaling stats
                depth_metadata = DepthMetadata(
                    backend="da3",
                    model=self.config.model_variant.value.name,  # Use name string, not ModelInfo
                    license="CC-BY-NC",
                    non_commercial_ok=self.config.non_commercial_ok,
                    depth_path=f"depth/{output_key.parent / output_key.name}_depth.png".replace("\\", "/"),
                    dtype="uint16",
                    shape=list(depth_result.depth.shape[:2]),
                    scaling={
                        "method": depth_stats.method,
                        "p_low_percentile": depth_stats.p_low_percentile,
                        "p_high_percentile": depth_stats.p_high_percentile,
                        "v_low_value": depth_stats.v_low_value,
                        "v_high_value": depth_stats.v_high_value,
                        "clipped_low_frac": depth_stats.clipped_low_frac,
                        "clipped_high_frac": depth_stats.clipped_high_frac,
                        "invalid_frac": depth_stats.invalid_frac,
                    },
                    runtime_ms=depth_runtime_s * 1000,
                    representation="depth",  # DA3 outputs depth, not inverse depth
                    convention="higher_is_farther",  # DA3 convention
                    unit="relative",  # DA3 outputs relative depth
                )

                logger.info(f"Depth generated in {depth_runtime_s:.2f}s")

            except Exception as e:
                logger.error(f"Depth generation failed: {e}")

                # Handle failure according to policy
                if self.config.depth_fallback == "fail":
                    raise
                elif self.config.depth_fallback == "skip":
                    logger.warning(f"Skipping {output_key} due to depth failure")
                    return {
                        "status": "skipped",
                        "reason": f"depth_failed: {str(e)}",
                        "image": str(image_input.path),
                    }
                elif self.config.depth_fallback == "v2-auto":
                    logger.warning("Depth failed, V2 will auto-generate depth")
                    # Clear depth_dir for this image so V2 uses its own depth
                    # Also clean up any partially written depth file
                    if depth_path and depth_path.exists():
                        try:
                            depth_path.unlink()
                            logger.debug(f"Removed failed depth file: {depth_path}")
                        except Exception as unlink_exc:
                            logger.warning(f"Could not remove failed depth file: {unlink_exc}")
                    depth_path = None
                else:
                    raise ValueError(f"Unknown depth_fallback: {self.config.depth_fallback}")

        # Load existing depth metadata if skipped
        else:
            if combined_manifest_path.exists():
                # Load from previous run
                try:
                    manifest = CombinedManifest.load(combined_manifest_path)
                    depth_metadata = manifest.depth
                except Exception:
                    logger.warning("Could not load existing depth metadata")

        # Stage A.5: Depth zones (diagnostics-only preview)
        if self.config.depth_zones == "preview" and depth_path and depth_path.exists():
            try:
                from PIL import Image
                import numpy as np

                from .depth_writer import read_depth_u16_png
                from .depth_zones import DepthZoneConfig, DepthZoneGenerator

                # Read uint16 depth and convert to float32 [0,1]
                depth_u16 = read_depth_u16_png(depth_path)
                depth01 = (depth_u16.astype(np.float32) / 65535.0).clip(0.0, 1.0)

                # Optional: load normalized RGB for future sky heuristics (but keep it off by default)
                rgb01 = None
                try:
                    with Image.open(normalized_path) as im:
                        im = im.convert("RGB")
                        rgb01 = np.asarray(im).astype(np.float32) / 255.0
                except Exception:
                    rgb01 = None  # purely optional

                zones_cfg = DepthZoneConfig(
                    mode="preview",
                    apply_sky_heuristic=False,  # keep conservative
                )
                gen = DepthZoneGenerator(zones_cfg)

                zones, stats = gen.generate_zones(depth=depth01, image=rgb01 if rgb01 is not None else None)

                # Write artifacts to output/zones/(subdir)/{stem}_*
                zones_out_dir = self.zones_dir / output_key.parent
                zones_out_dir.mkdir(parents=True, exist_ok=True)

                stem = output_key.name
                gen.save_zone_masks(zones, zones_out_dir, stem)
                gen.save_preview(zones, zones_out_dir / f"{stem}_zones_preview.png")
                gen.save_stats_json(stats, zones_out_dir / f"{stem}_zone_stats.json")

                logger.info(f"Stage A.5: Wrote depth zones diagnostics for {output_key}")

            except Exception as e:
                logger.warning(f"Stage A.5 depth zones preview failed for {output_key}: {e}")

        # Stage B: Run V2 enhancement
        logger.info(f"Stage B: Running V2 enhancement for {output_key}...")
        # v2_start_time tracked for potential future timing metrics

        # Preflight validation: Check depth/image alignment BEFORE invoking V2
        # This catches EXIF orientation mismatches early with clear error messages
        if depth_path and depth_path.exists():
            try:
                validate_depth_image_alignment(normalized_path, depth_path)
            except ValueError as e:
                logger.error(f"Preflight validation failed: {e}")
                raise

        # Check V2 resume with config fingerprint validation
        v2_report_path_existing = find_v2_report(self.v2_dir, output_key.name)
        skip_v2 = not self.config.force_v2 and self.should_skip_v2(
            v2_report_path_existing,
            combined_manifest_path,
            image_input,
            depth_was_skipped=skip_depth,
        )

        if skip_v2:
            logger.info(f"V2 outputs exist and valid, skipping: {v2_report_path_existing}")
            v2_runtime_s = 0.0
            v2_result = {"status": "ok"}
            v2_report_path = v2_report_path_existing
        else:
            # Use normalized input for V2 processing
            v2_result = self.v2_runner.run(
                input_path=normalized_path,  # Use normalized file
                depth_dir=self.depth_dir if depth_path else None,
                output_dir=self.v2_dir,
                preset=self.config.v2_preset,
                device=self.config.v2_device,
                upscaler_backend=self.config.v2_upscaler_backend,
                log_file=v2_log_path,
                timeout=self.config.v2_timeout,
            )
            v2_runtime_s = v2_result.get("runtime_s", 0.0)
            # Find V2 report after running
            v2_report_path = find_v2_report(self.v2_dir, output_key.name)

        # Create V2 metadata
        v2_metadata = V2Metadata(
            preset=self.config.v2_preset,
            strict_depth=depth_path is not None,
            output_dir="v2/",
            report_path=f"v2/{v2_report_path.name}" if v2_report_path else "",
            status=v2_result["status"],
            error_message=v2_result.get("error"),
        )

        # Compute input hash using policy-aware helper (FIX 2 & FIX 5)
        # This may return None if hash_mode=NEVER, or raise if hash required but fails
        input_sha256 = self._compute_or_skip_hash(image_input.path, manifest_path=combined_manifest_path)

        # Compute config fingerprint
        config_fp = self.compute_config_fingerprint()

        # Build combined manifest
        manifest = CombinedManifest(
            input=InputMetadata(
                image_path=str(image_input.path),
                image_sha256=input_sha256,
                exif_normalized=True,  # Always true - we always create normalized file
                normalized_path=str(normalized_path),  # Always set - normalized file always created
            ),
            depth=depth_metadata,
            v2=v2_metadata,
            timing=TimingMetadata(
                depth_s=depth_runtime_s,
                v2_s=v2_runtime_s,
                total_s=depth_runtime_s + v2_runtime_s,
            ),
            repro=ReproMetadata(
                v3_git=self.v3_git,
                v2_git=self.v2_git,
                device=self.config.depth_device,
            ),
            config_fingerprint=config_fp.to_sha256(),  # Config fingerprint for cache validation
            environment=self.environment,  # NEW: Toolchain environment for reproducibility
        )

        # Write manifest
        manifest.write(combined_manifest_path)

        logger.info(f"Completed {output_key} in {depth_runtime_s + v2_runtime_s:.2f}s")

        return {
            "status": "ok",
            "image": str(image_input.path),
            "depth_path": str(depth_path) if depth_path else None,
            "v2_report": str(v2_report_path) if v2_report_path else None,
            "manifest": str(combined_manifest_path),
            "runtime_s": depth_runtime_s + v2_runtime_s,
        }

    def enhance_batch(
        self,
        input_dir: Path,
        image_extensions: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Process batch of images through V3 + V2 pipeline.

        Args:
            input_dir: Input directory
            image_extensions: Image extensions to process (default: common formats)

        Returns:
            List of results for each image
        """
        import datetime

        if image_extensions is None:
            image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

        # Generate batch ID
        batch_id = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        start_time = datetime.datetime.now().isoformat()

        # Collect images (including nested directories)
        images = []
        for ext in image_extensions:
            images.extend(input_dir.rglob(f"*{ext}"))
            images.extend(input_dir.rglob(f"*{ext.upper()}"))

        logger.info(f"Found {len(images)} images in {input_dir} (including subdirectories)")
        logger.info(f"Batch ID: {batch_id}")

        # Process images with explicit input_root (stateless)
        results = []
        for img_path in sorted(images):
            image_input = ImageInput(path=img_path)
            try:
                # Pass input_dir as input_root for collision-free paths
                result = self.enhance_image(image_input, input_root=input_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {img_path}: {e}")
                results.append(
                    {
                        "status": "error",
                        "image": str(img_path),
                        "error": str(e),
                    }
                )

        # Compute end time and summary
        end_time = datetime.datetime.now().isoformat()

        succeeded = sum(1 for r in results if r.get("status") == "ok")
        failed = sum(1 for r in results if r.get("status") == "error")
        skipped = sum(1 for r in results if r.get("status") == "skipped")

        # Compute runtime stats using shared utility
        runtime_stats = compute_batch_runtime_stats(results)

        logger.info(f"Batch complete: {succeeded} succeeded, {failed} failed, {skipped} skipped")

        # Build batch manifest
        batch_manifest = BatchManifest(
            batch_id=batch_id,
            start_time=start_time,
            end_time=end_time,
            config={
                "model_variant": self.config.model_variant.value.to_dict(),  # Convert ModelInfo to dict
                "preset": self.config.preset.value if self.config.preset else None,
                "depth_quantization": self.config.depth_quantization,
                "v2_preset": self.config.v2_preset,
                "v2_upscaler_backend": self.config.v2_upscaler_backend,
                "execution_mode": self.config.execution_mode,
                "depth_fallback": self.config.depth_fallback,
            },
            images=[
                {
                    "stem": Path(r["image"]).stem,
                    "status": r.get("status", "unknown"),
                    "manifest": (str(r.get("manifest", "")) if r.get("status") == "ok" else None),
                    "runtime_s": (r.get("runtime_s", 0.0) if r.get("status") == "ok" else None),
                    "error": r.get("error") if r.get("status") == "error" else None,
                }
                for r in results
            ],
            summary={
                "total": len(results),
                "ok": succeeded,
                "error": failed,
                "skipped": skipped,
                **runtime_stats,
            },
        )

        # Write batch manifest
        batch_manifest_path = self.manifests_dir / f"batch_{batch_id}.json"
        batch_manifest.write(batch_manifest_path)
        logger.info(f"Batch summary written to {batch_manifest_path}")

        return results
