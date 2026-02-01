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
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any
import time
import logging
import datetime
import json
import hashlib
import os

# Note: Imports adjusted to relative for package context compatibility
from .config import DA3Config, ModelVariant, EnhanceConfig
from .inference import DA3InferenceEngine
from .postprocessing import Postprocessor
from .input_manager import ImageInput
from .depth_writer import atomic_write_depth_u16_png_with_stats
from .pbr import generate_pbr_maps
from .pbr_writer import write_pbr_maps
from .v2_runner import V2Runner, find_v2_report
from .security import (
    HashMode,
    sanitize_file_stem,
    sanitize_path_component_nonlossy,
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


def make_output_key(input_path: Path, input_root: Path) -> Path:
    """Generate collision-free output key preserving directory structure.

    Creates a unique output key that:
    1. Preserves the input's directory structure relative to input_root
    2. Includes the sanitized original extension (without dot)
    3. Appends an 8-character SHA-1 hash of the full relative path

    This ensures unique output names even for files with the same name
    in different directories or with different extensions.

    Args:
        input_path: Full path to input file
        input_root: Base directory for relative path calculation

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
    ext_label = ext.lstrip('.').lower() if ext else "noext"
    ext_label = sanitize_path_component_nonlossy(ext_label)

    # Compute 8-char SHA-1 hash of full relative path for uniqueness
    hash_input = relpath.as_posix().encode('utf-8')
    hash_suffix = hashlib.sha1(hash_input).hexdigest()[:8]

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
        self.zones_dir = self.output_root / "zones"

        for d in [self.depth_dir, self.v2_dir, self.manifests_dir, self.logs_dir, self.zones_dir]:
            d.mkdir(parents=True, exist_ok=True)

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

        # Initialize Inference Engine
        self.inference_engine = DA3InferenceEngine(
            config=da3_config,
            commercial_use=not config.non_commercial_ok,
            validate_license_strict=True,
        )

        # Initialize Postprocessor (FIX: Ensures refine_edges/bilateral settings from preset are applied)
        self.postprocessor = Postprocessor(da3_config.postprocessing)

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

    def _compute_or_skip_hash(
        self,
        image_path: Path,
        manifest_exists: bool = False,
        saved_hash: Optional[str] = None,
        *,
        for_manifest_write: bool = False
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
            manifest = CombinedManifest.load(manifest_path)

            # Input Integrity Check - use stored fingerprint
            saved_hash = manifest.input.image_sha256 if manifest.input else None
            if saved_hash and self.config.hash_mode != HashMode.NEVER:
                current_hash = self._compute_or_skip_hash(
                    image_input.path,
                    manifest_exists=True,
                    saved_hash=saved_hash,
                    for_manifest_write=False
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
        self, v2_report_path: Optional[Path], manifest_path: Path,
        image_input: ImageInput, depth_was_skipped: bool
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
                    if isinstance(filepath, str) and filepath and label.endswith('_path'):
                        if not os.path.exists(filepath):
                            logger.debug(f"PBR output missing: {filepath}")
                            return False

            return True
        except Exception as e:
            logger.debug(f"V2 skip check failed: {e}")
            return False

    def enhance_image(
        self, image_input: ImageInput, input_root: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run full enhancement pipeline on a single image.

        Implements lazy preprocessing - validation and preprocessing only run
        if depth computation is needed (not cached).

        Args:
            image_input: Input image information
            input_root: Base directory for relative path calculation

        Returns:
            Dictionary with processing status and output paths
        """
        # Capture start time for accurate timestamps
        pipeline_start_time = time.time()

        output_key = (
            make_output_key(image_input.path, input_root)
            if input_root else Path(sanitize_file_stem(image_input.path.stem))
        )
        logger.info(f"Processing {output_key}...")

        # Paths
        depth_path = self.depth_dir / output_key.parent / f"{output_key.name}_depth.png"
        float_depth_path = self.depth_dir / output_key.parent / f"{output_key.name}_depth.npy"
        manifest_path = self.manifests_dir / output_key.parent / f"{output_key.name}_combined.json"
        v2_log_path = self.logs_dir / output_key.parent / f"v2_{output_key.name}.log"

        # Ensure dirs
        for p in [depth_path, manifest_path, v2_log_path]:
            p.parent.mkdir(parents=True, exist_ok=True)

        # --- STAGE A: DEPTH ---
        # Determine skip logic BEFORE preprocessing (lazy evaluation)
        skip_depth = not self.config.force_depth and self.should_skip_depth(depth_path, manifest_path, image_input)
        depth_runtime_s = 0.0
        depth_metadata = None

        # Initialize pbr_assets for all code paths (prevents UnboundLocalError)
        pbr_assets = None

        if not skip_depth:
            # Lazy preprocessing: Only validate and preprocess if we're running depth
            from .preprocessing import validate_image_format, preprocess_image
            validated_path = validate_image_format(image_input.path)
            preprocessed_array, original_shape = preprocess_image(validated_path)

            logger.info(f"Stage A: Generating depth for {output_key}...")
            t0 = time.time()
            try:
                # 1. Inference (using preprocessed numpy array)
                result = self.inference_engine.predict(preprocessed_array)

                # 2. Post-Processing (Refinement)
                result = self.postprocessor.process(result)

                depth_runtime_s = time.time() - t0

                # 3. Write quantized depth (PNG 16-bit)
                _, _, depth_stats = atomic_write_depth_u16_png_with_stats(
                    depth_path,
                    result.depth,
                    method=self.config.depth_quantization,
                    debug_verify=self.config.verify_depth_writes
                )

                # 3b. Save float depth (.npy) for high-precision PBR if enabled
                if getattr(self.config, 'save_float_depth', False):
                    import numpy as np
                    np.save(str(float_depth_path), result.depth)
                    logger.debug(f"Saved float depth: {float_depth_path}")

                stats = {
                    "backend": "da3",
                    "license": "CC-BY-NC",
                    "non_commercial_ok": self.config.non_commercial_ok,
                    "dtype": "uint16",
                    "shape": list(result.depth.shape[:2]),
                    "representation": "depth",
                    "convention": "higher_is_farther",
                    "unit": "relative",
                }

                # Merge inference provenance into depth stats (requested vs resolved model ids)
                _md = getattr(result, 'metadata', None) or {}
                for _k in ('requested_model_id','resolved_model_id','resolved_model_source'):
                    if _k in _md:
                        stats[_k] = _md[_k]

                depth_metadata = DepthMetadata(
                    model=self.config.model_variant.value.name,
                    depth_path=str(depth_path),
                    runtime_seconds=depth_runtime_s,
                    scaling=depth_stats._asdict(),
                    stats=stats,
                )

                # 4. Write depth metadata JSON (quick access to depth stats)
                depth_metadata_path = depth_path.parent / f"{depth_path.stem}_metadata.json"
                with open(depth_metadata_path, 'w') as f:
                    json.dump({
                        "model": depth_metadata.model,
                        "depth_path": depth_metadata.depth_path,
                        "runtime_seconds": depth_metadata.runtime_seconds,
                        "scaling": depth_metadata.scaling,
                        "stats": depth_metadata.stats,
                    }, f, indent=2)
                logger.debug(f"Wrote depth metadata: {depth_metadata_path}")

                # 5. PBR map generation (optional)
                if self.config.generate_pbr:
                    try:
                        logger.info("Generating PBR maps...")
                        pbr_t0 = time.time()

                        # Use to_pbr_config() for consistent parameter conversion
                        pbr_config = self.config.to_pbr_config()

                        # Generate maps from depth
                        normal_map, roughness_map, ao_map = generate_pbr_maps(
                            result.depth,
                            config=pbr_config
                        )

                        # Write PBR maps
                        pbr_dir = self.output_root / "pbr"
                        pbr_dir.mkdir(parents=True, exist_ok=True)

                        # Derive base name from output_key for consistent artifact naming
                        sanitized_stem = output_key.stem if output_key.suffix else output_key.name

                        pbr_paths = write_pbr_maps(
                            normal_map=normal_map,
                            roughness_map=roughness_map,
                            ao_map=ao_map,
                            output_dir=pbr_dir,
                            base_name=sanitized_stem
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
                            }
                        }

                    except Exception as pbr_error:
                        logger.warning(f"PBR generation failed (non-blocking): {pbr_error}")
                        pbr_assets = None

            except Exception as e:
                logger.error(f"Depth failed: {e}")
                if self.config.depth_fallback == "fail":
                    raise
                elif self.config.depth_fallback == "skip":
                    return {"status": "skipped", "reason": str(e), "image": str(image_input.path)}
                elif self.config.depth_fallback == "v2-auto":
                    logger.info("V2 fallback mode: V3 failed, will attempt V2 with independent depth generation")
                    if depth_path.exists():
                        depth_path.unlink()
                    depth_path = None
                    depth_metadata = None
                else:
                    raise ValueError(f"Unsupported depth_fallback mode: {self.config.depth_fallback}") from e
        else:
            # Depth was skipped - load from cache
            if manifest_path.exists():
                try:
                    m = CombinedManifest.load(manifest_path)
                    depth_metadata = m.depth
                    # Preserve previous PBR paths when resuming from cached depth
                    pbr_assets = getattr(m, "pbr_assets", None)
                except Exception as e:
                    logger.debug(f"Failed to load previous manifest metadata: {e}")

            # PBR generation with cached depth (if enabled but not previously generated)
            if self.config.generate_pbr and (pbr_assets is None or not self._verify_pbr_outputs(pbr_assets)):
                logger.info("Generating PBR maps from cached depth...")
                try:
                    # Load depth data - prefer float depth for quality
                    depth_data_for_pbr = self._load_cached_depth(depth_path, float_depth_path)

                    if depth_data_for_pbr is not None:
                        pbr_t0 = time.time()
                        pbr_config = self.config.to_pbr_config()

                        # Generate maps from cached depth
                        normal_map, roughness_map, ao_map = generate_pbr_maps(
                            depth_data_for_pbr,
                            config=pbr_config
                        )

                        # Write PBR maps
                        pbr_dir = self.output_root / "pbr"
                        pbr_dir.mkdir(parents=True, exist_ok=True)

                        sanitized_stem = output_key.stem if output_key.suffix else output_key.name

                        pbr_paths = write_pbr_maps(
                            normal_map=normal_map,
                            roughness_map=roughness_map,
                            ao_map=ao_map,
                            output_dir=pbr_dir,
                            base_name=sanitized_stem
                        )

                        pbr_runtime = time.time() - pbr_t0
                        logger.info(f"PBR maps generated from cache in {pbr_runtime:.2f}s")

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
                            }
                        }
                except Exception as pbr_error:
                    logger.warning(f"PBR generation from cache failed: {pbr_error}")

        # --- STAGE B: V2 ENHANCE (Optional) ---
        # Skip V2 stage if disabled or runner not initialized
        if self.v2_runner is None or not self.config.enable_v2:
            logger.info("V2 stage disabled, skipping enhancement")
            v2_runtime_s = 0.0
            v2_result = {"status": "skipped"}
            v2_report_path = None
        else:
            v2_report_path = find_v2_report(self.v2_dir, output_key.name)
            skip_v2 = not self.config.force_v2 and self.should_skip_v2(v2_report_path, manifest_path, image_input, skip_depth)

            if skip_v2:
                logger.info("V2 outputs valid, skipping.")
                v2_runtime_s = 0.0
                v2_result = {"status": "ok"}
            else:
                # V2 runner: depth_dir=None triggers independent depth generation in V2
                v2_result = self.v2_runner.run(
                    input_path=image_input.path,
                    depth_dir=self.depth_dir if (depth_path and depth_path.exists()) else None,
                    output_dir=self.v2_dir,
                    preset=self.config.v2_preset,
                    device=self.config.v2_device,
                    upscaler_backend=self.config.v2_upscaler_backend,
                    log_file=v2_log_path,
                    timeout=self.config.v2_timeout
                )
                v2_runtime_s = v2_result.get("runtime_s", 0.0)
                v2_report_path = find_v2_report(self.v2_dir, output_key.name)

        # Manifest
        v2_metadata = V2Metadata(
            preset=self.config.v2_preset,
            strict_depth=depth_path is not None,
            output_dir="v2/",
            report_path=str(v2_report_path) if v2_report_path else "",
            status=v2_result["status"],
            error_message=v2_result.get("error"),
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
            image_input.path,
            manifest_exists=manifest_exists,
            saved_hash=saved_hash,
            for_manifest_write=True
        )

        # Capture end time for accurate timestamps
        pipeline_end_time = time.time()

        manifest = CombinedManifest(
            input=InputMetadata(
                image_path=str(image_input.path),
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
            # Accurate batch execution timestamps (ISO 8601 format)
            start_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(pipeline_start_time)),
            end_time=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(pipeline_end_time)),
        )
        manifest.write(manifest_path)

        return {
            "status": "ok",
            "image": str(image_input.path),
            "depth_path": str(depth_path) if depth_path else None,
            "manifest": str(manifest_path),
            "runtime_s": pipeline_end_time - pipeline_start_time
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
            if isinstance(value, str) and key.endswith('_path'):
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

        images = []
        for ext in image_extensions:
            images.extend(input_dir.rglob(f"*{ext}"))
            images.extend(input_dir.rglob(f"*{ext.upper()}"))

        results = []
        for img in sorted(images):
            try:
                results.append(self.enhance_image(ImageInput(img), input_root=input_dir))
            except Exception as e:
                logger.error(f"Failed {img}: {e}")
                results.append({"status": "error", "image": str(img), "error": str(e)})

        # Capture accurate batch end time
        batch_end_time = time.time()
        batch_end_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(batch_end_time))

        # Write batch summary with accurate timestamps
        # Extract runtime_s from successful results for statistics computation
        runtimes = [r.get("runtime_s", 0.0) for r in results if r.get("status") == "ok"]
        runtime_stats = compute_batch_runtime_stats(runtimes)
        bm = BatchManifest(
            batch_id=batch_id,
            start_time=batch_start_utc,
            end_time=batch_end_utc,
            config={"model": self.config.model_variant.value.name},
            results=results,
            stats={**runtime_stats, "total_images": len(results), "batch_runtime_seconds": batch_end_time - batch_start_time}
        )
        bm.write(self.manifests_dir / f"batch_{batch_id}.json")
        return results
