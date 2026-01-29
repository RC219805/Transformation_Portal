"""Orchestrator for V3 depth + V2 enhancement pipeline.

Two-stage pipeline:
1. Stage A (V3): Generate depth assets using DA3 (Inference -> Post-Processing -> Write)
2. Stage B (V2): Consume depth assets -> V2 Subprocess -> Output
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
import time
import logging
import datetime
import json

# Note: Imports adjusted to relative for package context compatibility
from .config import DA3Config, ModelVariant, Preset, EnhanceConfig
from .inference import DA3InferenceEngine
from .postprocessing import Postprocessor
from .input_manager import ImageInput
from .depth_writer import atomic_write_depth_u16_png_with_stats
from .pbr import PBRConfig, generate_pbr_maps
from .pbr_writer import write_pbr_maps
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

def make_output_key(input_path: Path, input_root: Path) -> Path:
    """Generate collision-free output key preserving directory structure."""
    try:
        relpath = input_path.relative_to(input_root)
    except ValueError:
        logger.warning(f"{input_path} is not relative to {input_root}, using flat naming")
        relpath = Path(input_path.name)

    sanitized_parts = [sanitize_path_component_nonlossy(p) for p in relpath.parent.parts]
    stem_sanitized = sanitize_path_component_nonlossy(relpath.stem)

    if sanitized_parts:
        return Path(*sanitized_parts) / stem_sanitized
    else:
        return Path(stem_sanitized)

class EnhanceOrchestrator:
    """Orchestrates V3 depth generation + V2 enhancement pipeline."""

    def __init__(self, config: EnhanceConfig, output_root: Path):
        self.config = config
        self.output_root = Path(output_root)

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

        # Initialize V2 Runner and Environment
        self.v2_runner = V2Runner()
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

    def _compute_or_skip_hash(self, image_path: Path, manifest_path: Optional[Path] = None) -> Optional[str]:
        if self.config.hash_mode == HashMode.NEVER:
            return None

        # logic for IF_MANIFEST_EXISTS is handled by caller or defaults to compute
        try:
            return compute_file_sha256(image_path)
        except Exception as e:
            logger.error(f"Hash computation failed for {image_path}: {e}")
            raise IOError(f"Hash computation failed: {e}") from e

    def should_skip_depth(self, depth_path: Path, manifest_path: Path, image_input: ImageInput) -> bool:
        if not depth_path.exists() or not manifest_path.exists():
            return False

        try:
            manifest = CombinedManifest.load(manifest_path)

            # Input Integrity Check
            if manifest.input and manifest.input.image_sha256 and self.config.hash_mode != HashMode.NEVER:
                current_hash = self._compute_or_skip_hash(image_input.path)
                if current_hash and current_hash != manifest.input.image_sha256:
                    logger.info(f"Input image changed - regenerating depth: {image_input.path}")
                    return False

            # Config Integrity Check
            if not manifest.config_fingerprint:
                return False

            current_fp = self.compute_config_fingerprint()
            manifest_fp = ConfigFingerprint(
                model_variant=manifest.depth.model if manifest.depth else "",
                depth_quantization=(manifest.depth.scaling.get("method", "") if manifest.depth else ""),
                depth_device=self.config.depth_device,
                preset=self.config.preset.value if self.config.preset else None,
                v2_preset=manifest.v2.preset if manifest.v2 else "",
                v2_device=self.config.v2_device,
                v2_upscaler_backend=self.config.v2_upscaler_backend,
            )

            if current_fp.depth_only() != manifest_fp.depth_only():
                logger.info("Depth config changed - regenerating")
                return False

            # File Integrity Check
            if not manifest.depth:
                return False

            # Quick read check
            from .depth_writer import read_depth_u16_png
            d = read_depth_u16_png(depth_path)
            if d.ndim != 2: return False

            logger.debug(f"Resuming with existing depth: {depth_path}")
            return True
        except Exception:
            return False

    def should_skip_v2(self, v2_report_path: Optional[Path], manifest_path: Path, image_input: ImageInput, depth_was_skipped: bool) -> bool:
        if not v2_report_path or not v2_report_path.exists() or not manifest_path.exists():
            return False

        try:
            manifest = CombinedManifest.load(manifest_path)

            # Config Check
            if not manifest.config_fingerprint: return False

            current_fp = self.compute_config_fingerprint()
            manifest_fp = ConfigFingerprint(
                model_variant=manifest.depth.model if manifest.depth else "",
                depth_quantization=(manifest.depth.scaling.get("method", "") if manifest.depth else ""),
                depth_device=self.config.depth_device,
                preset=self.config.preset.value if self.config.preset else None,
                v2_preset=manifest.v2.preset if manifest.v2 else "",
                v2_device=self.config.v2_device,
                v2_upscaler_backend=self.config.v2_upscaler_backend,
            )

            if current_fp.v2_only() != manifest_fp.v2_only():
                return False

            # Consistency Check
            if not depth_was_skipped:
                logger.info("Depth was regenerated - V2 must rerun")
                return False

            if not manifest.v2 or manifest.v2.status != "ok":
                return False

            return True
        except Exception:
            return False

    def enhance_image(self, image_input: ImageInput, input_root: Optional[Path] = None) -> Dict[str, Any]:
        output_key = make_output_key(image_input.path, input_root) if input_root else Path(sanitize_file_stem(image_input.path.stem))
        logger.info(f"Processing {output_key}...")

        # Paths
        depth_path = self.depth_dir / output_key.parent / f"{output_key.name}_depth.png"
        manifest_path = self.manifests_dir / output_key.parent / f"{output_key.name}_combined.json"
        v2_log_path = self.logs_dir / output_key.parent / f"v2_{output_key.name}.log"

        # Ensure dirs
        for p in [depth_path, manifest_path, v2_log_path]:
            p.parent.mkdir(parents=True, exist_ok=True)

        # Preprocess Input (Validation + Normalization)
        from .preprocessing import validate_image_format, preprocess_image
        validated_path = validate_image_format(image_input.path)
        # Keep normalized_path alias for backward-compatible manifest metadata
        normalized_path = validated_path
        preprocessed_array, original_shape = preprocess_image(validated_path)

        # --- STAGE A: DEPTH ---
        skip_depth = not self.config.force_depth and self.should_skip_depth(depth_path, manifest_path, image_input)
        depth_runtime_s = 0.0
        depth_metadata = None

        # Initialize pbr_assets for all code paths (prevents UnboundLocalError)
        pbr_assets = None

        if not skip_depth:
            logger.info(f"Stage A: Generating depth for {output_key}...")
            t0 = time.time()
            try:
                # 1. Inference (using preprocessed numpy array)
                result = self.inference_engine.predict(preprocessed_array)

                # 2. Post-Processing (Refinement)
                result = self.postprocessor.process(result)

                depth_runtime_s = time.time() - t0

                # 3. Write
                _, _, depth_stats = atomic_write_depth_u16_png_with_stats(
                    depth_path,
                    result.depth,
                    method=self.config.depth_quantization,
                    debug_verify=self.config.verify_depth_writes
                )

                depth_metadata = DepthMetadata(
                    model=self.config.model_variant.value.name,
                    depth_path=str(depth_path),
                    runtime_seconds=depth_runtime_s,
                    scaling=depth_stats._asdict(),
                    stats={
                        "backend": "da3",
                        "license": "CC-BY-NC",
                        "non_commercial_ok": self.config.non_commercial_ok,
                        "dtype": "uint16",
                        "shape": list(result.depth.shape[:2]),
                        "representation": "depth",
                        "convention": "higher_is_farther",
                        "unit": "relative",
                    },
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
                pbr_assets = None
                if self.config.enable_pbr:
                    try:
                        logger.info("Generating PBR maps...")
                        pbr_t0 = time.time()

                        # Build PBR configuration from EnhanceConfig
                        pbr_config = PBRConfig(
                            normal_strength=self.config.pbr_normal_strength,
                            normal_blur_radius=self.config.pbr_normal_blur_radius,
                            roughness_strength=self.config.pbr_roughness_strength,
                            roughness_blur_radius=self.config.pbr_roughness_blur_radius,
                            ao_strength=self.config.pbr_ao_strength,
                            ao_blur_radius=self.config.pbr_ao_blur_radius,
                        )

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
            if manifest_path.exists():
                try:
                    m = CombinedManifest.load(manifest_path)
                    depth_metadata = m.depth
                    # Preserve previous PBR paths when resuming from cached depth
                    pbr_assets = getattr(m, "pbr_assets", None)
                except Exception:  # Do not swallow KeyboardInterrupt/SystemExit
                    pass

        # --- STAGE B: V2 ENHANCE ---
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

        input_sha = self._compute_or_skip_hash(image_input.path)

        manifest = CombinedManifest(
            input=InputMetadata(str(image_input.path), input_sha, True, str(normalized_path)),
            depth=depth_metadata,
            v2=v2_metadata,
            timing=TimingMetadata(
                total_seconds=depth_runtime_s + v2_runtime_s,
                depth_seconds=depth_runtime_s,
                v2_seconds=v2_runtime_s,
                timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            ),
            pbr_assets=pbr_assets,
            repro=ReproMetadata(
                v3_git_revision=self.v3_git,
                v2_git_revision=self.v2_git,
                environment=self.environment,
            ),
            config_fingerprint=self.compute_config_fingerprint(),
            environment=self.environment
        )
        manifest.write(manifest_path)

        return {
            "status": "ok",
            "image": str(image_input.path),
            "depth_path": str(depth_path) if depth_path else None,
            "manifest": str(manifest_path),
            "runtime_s": depth_runtime_s + v2_runtime_s
        }

    def enhance_batch(self, input_dir: Path, image_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        if image_extensions is None:
            image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

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

        # Write batch summary
        runtime_stats = compute_batch_runtime_stats(results)
        bm = BatchManifest(
            batch_id, datetime.datetime.now().isoformat(), datetime.datetime.now().isoformat(),
            {"model": self.config.model_variant.value.name},
            results,
            {"total": len(results), **runtime_stats}
        )
        bm.write(self.manifests_dir / f"batch_{batch_id}.json")
        return results
